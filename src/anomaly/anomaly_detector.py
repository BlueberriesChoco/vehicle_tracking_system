from typing import List, Optional

from .baseline import BaselineBuilder, BaselineStore
from .rule_engine import RuleEngine


class AnomalyDetector:
    """Phase 3 anomaly detector.

    组合四类信号：
      1. 规则引擎（超速、长滞留、路径偏离、夜间停留）；
      2. Z-score 统计基线（可选的 JSON 基线）；
      3. Isolation Forest 无监督异常指数 A（可选，研究要点二·路径二）；
      4. Transformer 轨迹预测残差（可选，研究要点二·路径一）。

    任一信号命中即 ``is_anomaly = 1``，命中原因写入 ``alert_reason``。
    """

    def __init__(
        self,
        thresholds: Optional[dict] = None,
        baseline: Optional[dict] = None,
        baseline_path: Optional[str] = None,
        baseline_threshold: float = 3.0,
        target_groups: Optional[list] = None,
        iforest=None,
        trajectory_predictor=None,
    ):
        self.rule_engine = RuleEngine(thresholds=thresholds)
        self.baseline_builder = BaselineBuilder(target_groups=target_groups)
        self.baseline = baseline or (BaselineStore.load(baseline_path) if baseline_path else None)
        self.baseline_threshold = baseline_threshold
        self.iforest = iforest
        self.trajectory_predictor = trajectory_predictor

    def detect(self, vectors: List[dict]) -> List[dict]:
        """Annotate vectors or passage events with anomaly fields."""
        # 需要整批打分的模型先统一算好，再按行合并
        iforest_results = self.iforest.score(vectors) if self.iforest is not None else None
        traj_results = (
            self.trajectory_predictor.score(vectors)
            if self.trajectory_predictor is not None
            else None
        )

        results = []
        for index, vector in enumerate(vectors):
            record = dict(vector)
            # 非行政通道研判对象只保留输出，不参与规则告警和统计基线评分。
            is_target = self.baseline_builder.is_analysis_target(record)
            record["analysis_target"] = int(is_target)
            reasons = self.rule_engine.check(record) if is_target else []
            baseline_score = 0.0
            baseline_feature_count = 0

            # 统计基线是 Phase3 的离线筛查能力；没有基线时只运行规则引擎。
            if is_target and self.baseline is not None:
                baseline_result = self.baseline_builder.score(
                    record,
                    self.baseline,
                    threshold=self.baseline_threshold,
                )
                baseline_score = baseline_result["baseline_score"]
                baseline_feature_count = baseline_result["baseline_feature_count"]
                if baseline_result["baseline_anomaly"]:
                    reasons.extend(baseline_result["baseline_reasons"])

            # 无监督异常指数 A：仅对分析对象生效
            if iforest_results is not None:
                iforest_record = iforest_results[index]
                record["iforest_score"] = iforest_record.get("iforest_score", 0.0)
                record["iforest_anomaly"] = iforest_record.get("iforest_anomaly", 0)
                if is_target and record["iforest_anomaly"]:
                    reasons.append(f"iforest_score={record['iforest_score']:.2f}")
            else:
                record.setdefault("iforest_score", 0.0)
                record.setdefault("iforest_anomaly", 0)

            # 轨迹预测残差：仅对分析对象生效
            if traj_results is not None:
                traj_record = traj_results[index]
                record["traj_residual"] = traj_record.get("traj_residual", 0.0)
                record["traj_anomaly"] = traj_record.get("traj_anomaly", 0)
                if is_target and record["traj_anomaly"]:
                    reasons.append(f"traj_residual={record['traj_residual']:.1f}px")
            else:
                record.setdefault("traj_residual", 0.0)
                record.setdefault("traj_anomaly", 0)

            record["baseline_feature_count"] = baseline_feature_count
            record["anomaly_score"] = round(float(baseline_score), 3)
            record["is_anomaly"] = 1 if reasons else 0
            record["alert_reason"] = ";".join(reasons)
            results.append(record)

        return results
