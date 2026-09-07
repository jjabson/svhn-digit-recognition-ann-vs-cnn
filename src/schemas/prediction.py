from pydantic import BaseModel, ConfigDict
from dataclasses import dataclass
from src.schemas.orchestration import InferenceDecision


class OrchestratedPredictionResponse(BaseModel):
    predicted_digit: int | None
    confidence: float | None
    confidence_percent: str | None

    selected_model: str | None
    status: str
    decision_reason: str | None

    fallback_used: bool
    fallback_reason: str | None

    review_required: bool

class PredictionResponse(BaseModel):
    """
    API response returned after classifying an SVHN digit image.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_digit": 7,
                "confidence": 0.9842,
                "confidence_percent": "98.42%",
                "probabilities": {
                    "0": 0.0011,
                    "1": 0.0004,
                    "2": 0.0020,
                    "3": 0.0007,
                    "4": 0.0015,
                    "5": 0.0003,
                    "6": 0.0010,
                    "7": 0.9842,
                    "8": 0.0041,
                    "9": 0.0047,
                },
            }
        }
    )

    predicted_digit: int
    confidence: float
    confidence_percent: str
    probabilities: dict[str, float]

class InferenceConfigResponse(BaseModel):
    model_name: str
    confidence_threshold: float

@dataclass(frozen=True)
class PredictionResult:
    """
    Represents the raw structured result produced by
    a model prediction before orchestration decisions
    are applied.
    """

    predicted_digit: int
    confidence: float
    probabilities: dict[str, float]

    @property
    def confidence_percent(self) -> str:
        return f"{self.confidence * 100:.2f}%"

def inference_decision_to_response(
    decision: InferenceDecision,
) -> OrchestratedPredictionResponse:
    confidence_percent = (
        f"{decision.confidence * 100:.2f}%"
        if decision.confidence is not None
        else None
    )

    return OrchestratedPredictionResponse(
        predicted_digit=decision.predicted_digit,
        confidence=decision.confidence,
        confidence_percent=confidence_percent,
        selected_model=decision.selected_model,
        status=decision.status.value,
        decision_reason=decision.decision_reason,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        review_required=decision.review_required,
    )