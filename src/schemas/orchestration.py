from dataclasses import dataclass
from enum import Enum


class DecisionStatus(str, Enum):
    """
    Final semantic outcome of an inference orchestration decision.
    """

    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    FAILED = "failed"

@dataclass(frozen=True)
class InferenceAttempt:
    """
    Represents the raw result of a single model inference attempt.

    This object records what one model returned before the
    orchestration layer applies decision policies.
    """

    model_name: str
    predicted_digit: int | None
    confidence: float | None
    succeeded: bool
    error_message: str | None

@dataclass(frozen=True)
class InferenceDecision:
    """
    Represents the final decision produced by the inference
    orchestration layer.
    """

    selected_model: str | None
    predicted_digit: int | None
    confidence: float | None

    status: DecisionStatus

    decision_reason: str | None

    fallback_used: bool
    fallback_reason: str | None

    review_required: bool

@dataclass(frozen=True)
class InferencePolicy:
    """
        Defines the decision thresholds used by the
        inference orchestration layer.
        """
    confidence_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )
