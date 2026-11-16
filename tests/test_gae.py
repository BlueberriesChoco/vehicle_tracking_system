"""图自编码器（GAE）与图异常指数 G 的回归测试。

覆盖两层契约：
1. 结构契约：形状、可复现、保存/加载、训练收敛；
2. 迁移契约：训练图与评分图节点数不同仍可打分，且结构扰动能被识别——
   这一条是「基线图训练 + 目标图评分」成立的前提（指标 11）。
"""

import numpy as np

from src.graph.gae import (
    NODE_FEATURE_NAMES,
    GraphAutoencoder,
    GraphAutoencoderDetector,
    node_features,
    normalize_adjacency,
)


def _two_cliques_with_outlier():
    n = 17
    A = np.zeros((n, n), dtype=np.float64)
    for clique in (range(0, 8), range(8, 16)):
        for i in clique:
            for j in clique:
                if i != j:
                    A[i, j] = 1.0
    # 节点 16 只与节点 0 相连，连接模式明显异常
    A[16, 0] = A[0, 16] = 1.0
    return A


def _ring(n=24):
    """每个节点度都为 2 的环：度分布极窄，适合检验"超范围 hub"能否被识别。"""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1.0
    return A


def _preferential_attachment(n=40, seed=3):
    """优先连接图：度分布 2~18，结构异质，模型才有可学的模式。"""
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n), dtype=np.float64)
    A[0, 1] = A[1, 0] = 1.0
    for node in range(2, n):
        degree = A.sum(axis=1)[:node]
        weights = degree / degree.sum() if degree.sum() > 0 else None
        for target in rng.choice(node, 2, replace=False, p=weights):
            A[node, target] = A[target, node] = 1.0
    return A


def _three_cliques():
    """节点数与基线图不同的目标图（3 个 7 节点团，共 21 节点）。"""
    n = 21
    A = np.zeros((n, n), dtype=np.float64)
    for clique in (range(0, 7), range(7, 14), range(14, 21)):
        for i in clique:
            for j in clique:
                if i != j:
                    A[i, j] = 1.0
    return A


def test_normalize_adjacency_is_symmetric():
    A = _two_cliques_with_outlier()
    normalized = normalize_adjacency(A)

    np.testing.assert_allclose(normalized, normalized.T, atol=1e-9)
    assert normalized.shape == A.shape


def test_node_features_shape_and_range():
    A = _two_cliques_with_outlier()
    features = node_features(A)

    assert features.shape == (A.shape[0], len(NODE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features.min() >= 0.0
    assert features.max() <= 1.0 + 1e-6


def test_gae_forward_shapes():
    import torch

    model = GraphAutoencoder(in_dim=len(NODE_FEATURE_NAMES), hidden_dim=8, latent_dim=4)
    features = torch.zeros(10, len(NODE_FEATURE_NAMES))
    adjacency_norm = torch.zeros(10, 10)

    reconstruction, embeddings = model(features, adjacency_norm)

    assert tuple(reconstruction.shape) == (10, 10)
    assert tuple(embeddings.shape) == (10, 4)


def test_gae_training_reduces_loss():
    A = _two_cliques_with_outlier()
    detector = GraphAutoencoderDetector(
        hidden_dim=16, latent_dim=8, epochs=60, learning_rate=0.02, random_state=1
    )

    detector.fit(A)

    history = detector.training_history
    assert history[-1]["loss"] < history[0]["loss"]


def test_gae_flags_out_of_support_hub():
    """基线图每个节点度都是 2，评分图中一个节点连到 12 个节点，应被排到最前。"""
    baseline = _ring()
    hub = 5
    target = baseline.copy()
    for node in range(6, 18):
        target[hub, node] = target[node, hub] = 1.0

    detector = GraphAutoencoderDetector(
        hidden_dim=32, latent_dim=8, epochs=300, learning_rate=0.05, random_state=23
    )
    detector.fit(baseline)
    error = detector.score(target)["reconstruction_error"]

    assert error[hub] >= np.percentile(error, 95), (hub, error[hub], np.percentile(error, 95))


def test_gae_learns_to_separate_edges_from_non_edges():
    """负采样训练的直接效果：真实边的重构值应高于非边，否则模型退化为全 1。

    用结构异质的图：完全对称的图（如全连接团）里所有节点特征相同，
    任何特征驱动模型都不可能有区分度，那不是训练问题。
    """
    A = _preferential_attachment()
    detector = GraphAutoencoderDetector(
        hidden_dim=32, latent_dim=8, epochs=300, learning_rate=0.02, random_state=11
    )
    detector.fit(A)

    reconstruction = detector._reconstruction(A)
    binary = (A > 0).astype(float)
    np.fill_diagonal(binary, 0.0)
    off_diagonal = ~np.eye(A.shape[0], dtype=bool)

    edge_mean = float(reconstruction[(binary > 0)].mean())
    non_edge_mean = float(reconstruction[(binary == 0) & off_diagonal].mean())

    assert edge_mean - non_edge_mean > 0.05, (edge_mean, non_edge_mean)


def test_gae_scores_graph_with_different_node_count():
    """迁移契约：基线图训练、目标图评分，两图节点数不同。"""
    baseline = _two_cliques_with_outlier()
    target = _three_cliques()

    detector = GraphAutoencoderDetector(
        hidden_dim=32, latent_dim=8, epochs=120, random_state=17
    )
    detector.fit(baseline)
    scores = detector.score(target)

    assert scores["reconstruction_error"].shape == (target.shape[0],)
    assert scores["graph_anomaly_index"].shape == (target.shape[0],)
    assert np.isfinite(scores["reconstruction_error"]).all()


def test_gae_detects_hub_node_after_transfer():
    """基线图训练、目标图评分时，连接数远超正常范围的节点应排在前列。

    注入方式是"给一个节点额外连 8 条边"（模拟与大量无关车辆建立关系的异常车）。
    只删边的扰动在本数据上落在正常度分布内（度 3~8），本来就不可检出，
    见 GOAL_STATUS 指标 11 的实测说明。
    """
    baseline = _two_cliques_with_outlier()
    target = _three_cliques()
    victim = 4

    perturbed = target.copy()
    extra = [node for node in range(target.shape[0]) if node != victim and target[victim, node] == 0][:8]
    for node in extra:
        perturbed[victim, node] = perturbed[node, victim] = 1.0

    detector = GraphAutoencoderDetector(
        hidden_dim=32, latent_dim=8, epochs=200, learning_rate=0.05, random_state=23
    )
    detector.fit(baseline)
    error = detector.score(perturbed)["reconstruction_error"]

    assert error[victim] >= np.percentile(error, 99), (victim, error[victim], np.percentile(error, 99))


def test_gae_save_and_load(tmp_path):
    A = _two_cliques_with_outlier()
    detector = GraphAutoencoderDetector(
        hidden_dim=16, latent_dim=8, epochs=20, random_state=5
    )
    detector.fit(A)

    prefix = str(tmp_path / "gae")
    detector.save(prefix)
    loaded = GraphAutoencoderDetector.load(prefix)

    np.testing.assert_allclose(
        loaded.reconstruction_error(A), detector.reconstruction_error(A), rtol=1e-5
    )
    # 加载后仍可对节点数不同的图打分
    assert loaded.reconstruction_error(_three_cliques()).shape == (21,)


def test_gae_is_deterministic_with_same_seed():
    """同一种子下两次训练应给出相同的重构误差，否则结果不可复现。"""
    A = _two_cliques_with_outlier()

    first = GraphAutoencoderDetector(hidden_dim=8, latent_dim=4, epochs=20, random_state=7)
    second = GraphAutoencoderDetector(hidden_dim=8, latent_dim=4, epochs=20, random_state=7)
    first.fit(A)
    second.fit(A)

    np.testing.assert_allclose(
        first.reconstruction_error(A), second.reconstruction_error(A), rtol=1e-5
    )
