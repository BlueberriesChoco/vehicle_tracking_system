"""图自编码器（GAE）与图异常指数 G 的回归测试。"""

import numpy as np

from src.graph.gae import (
    GraphAutoencoder,
    GraphAutoencoderDetector,
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


def test_normalize_adjacency_is_symmetric():
    A = _two_cliques_with_outlier()
    normalized = normalize_adjacency(A)

    np.testing.assert_allclose(normalized, normalized.T, atol=1e-9)
    assert normalized.shape == A.shape


def test_gae_forward_shapes():
    import torch

    model = GraphAutoencoder(num_nodes=10, hidden_dim=8, latent_dim=4)
    adjacency_norm = torch.zeros(10, 10)

    reconstruction, embeddings = model(adjacency_norm)

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


def test_gae_flags_structurally_anomalous_node():
    A = _two_cliques_with_outlier()
    detector = GraphAutoencoderDetector(
        hidden_dim=32, latent_dim=8, epochs=200, learning_rate=0.02, random_state=3
    )
    detector.fit(A)

    scores = detector.score(A)
    error = scores["reconstruction_error"]

    # 节点 16 的连接模式最异常，其重构误差应处于最高档
    assert error[16] >= np.percentile(error, 90)


def test_gae_save_and_load(tmp_path):
    A = _two_cliques_with_outlier()
    detector = GraphAutoencoderDetector(
        hidden_dim=16, latent_dim=8, epochs=20, random_state=5
    )
    detector.fit(A)

    prefix = str(tmp_path / "gae")
    detector.save(prefix)
    loaded = GraphAutoencoderDetector.load(prefix, num_nodes=A.shape[0])

    np.testing.assert_allclose(
        loaded.reconstruction_error(A), detector.reconstruction_error(A), rtol=1e-5
    )


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
