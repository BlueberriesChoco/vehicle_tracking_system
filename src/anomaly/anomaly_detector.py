from typing import List, Optional

from .baseline import BaselineBuilder, BaselineStore
from .rule_engine import RuleEngine


class AnomalyDetector:
    """Phase 3 anomaly detector combining rules and normal-behavior baselines."""

    def __init__(
        self,
        thresholds: Optional[dict] = None,
        baseline: Optional[dict] = None,
        baseline_path: Optional[str] = None,
        baseline_threshold: float = 3.0,
        target_groups: Optional[list] = None,
    ):
        self.rule_engine = RuleEngine(thresholds=thresholds)
        self.baseline_builder = BaselineBuilder(target_groups=target_groups)
        self.baseline = baseline or (BaselineStore.load(baseline_path) if baseline_path else None)
        self.baseline_threshold = baseline_threshold

    def detect(self, vectors: List[dict]) -> List[dict]:
        """Annotate vectors or passage events with anomaly fields."""
        results = []
        for vector in vectors:
            record = dict(vector)
            is_target = self.baseline_builder.is_analysis_target(record)
            record["analysis_target"] = int(is_target)
            reasons = self.rule_engine.check(record) if is_target else []
            baseline_score = 0.0
            baseline_feature_count = 0

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

            record["baseline_feature_count"] = baseline_feature_count
            record["anomaly_score"] = round(float(baseline_score), 3)
            record["is_anomaly"] = 1 if reasons else 0
            record["alert_reason"] = ";".join(reasons)
            results.append(record)

        return results
