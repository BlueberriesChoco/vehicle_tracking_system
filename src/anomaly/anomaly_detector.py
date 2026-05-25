from typing import List
import numpy as np

from .rule_engine import RuleEngine


class AnomalyDetector:
    """异常检测器（Phase 3 启用）。

    组合规则引擎和统计模型（隔离森林/LOF），
    对行为向量做多级异常判定。
    """

    def __init__(self):
        self.rule_engine = RuleEngine()
        self.statistical_model = None  # Phase 3: IsolationForest / LOF

    def detect(self, vectors: List[dict]) -> List[dict]:
        """对一批行为向量执行异常检测。

        Returns:
            每个向量增加 anomaly_score, is_anomaly, alert_reason 字段
        """
        for vec in vectors:
            # 规则引擎
            reasons = self.rule_engine.check(vec)

            # 统计模型（Phase 3 启用）
            anomaly_score = 0.0
            if self.statistical_model is not None:
                numeric = np.array([[
                    vec.get("avg_speed_ms", 0),
                    vec.get("max_speed_ms", 0),
                    vec.get("speed_variance", 0),
                    vec.get("max_dwell_sec", 0),
                    vec.get("stop_count", 0),
                    vec.get("dwell_ratio", 0),
                    vec.get("path_deviation", 0),
                    vec.get("path_smoothness", 0),
                    vec.get("night_ratio", 0),
                    vec.get("freq_index", 0),
                    vec.get("aggregation_index", 0),
                ]])
                # 孤立森林预测
                # pred = self.statistical_model.predict(numeric)
                # anomaly_score = float(pred[0])
                pass

            vec["anomaly_score"] = round(anomaly_score, 3)
            vec["is_anomaly"] = 1 if (reasons or anomaly_score < -0.5) else 0
            vec["alert_reason"] = ";".join(reasons) if reasons else ""

        return vectors
