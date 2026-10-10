from enum import Enum


class ActionRisk(str, Enum):
    READ_ONLY = "read_only"
    CONTROLLED = "controlled"
    PROHIBITED = "prohibited"


class ActionPolicy:
    def __init__(self) -> None:
        self._risk_by_action = {
            "get_inference_config": ActionRisk.READ_ONLY,
            "get_evaluation_summary": ActionRisk.READ_ONLY,
            "get_evaluation_insights": ActionRisk.READ_ONLY,
            "get_digit_metrics": ActionRisk.READ_ONLY,
            "get_model_summary": ActionRisk.READ_ONLY,
            "predict_digit": ActionRisk.READ_ONLY,
            "set_confidence_threshold": ActionRisk.CONTROLLED,
            "delete_evaluation_history": ActionRisk.PROHIBITED,
        }

    def classify(self, action_name: str) -> ActionRisk:
        try:
            return self._risk_by_action[action_name]
        except KeyError as exc:
            raise ValueError(f"Unknown action: {action_name}") from exc

    def requires_approval(self, action_name: str) -> bool:
        return self.classify(action_name) is ActionRisk.CONTROLLED

    def is_prohibited(self, action_name: str) -> bool:
        return self.classify(action_name) is ActionRisk.PROHIBITED