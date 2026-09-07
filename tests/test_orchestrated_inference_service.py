import pytest

from src.orchestration.errors import InvalidInferenceInputError
from src.preprocessing import InvalidImageError
from src.orchestration.inference_service import OrchestratedInferenceService
from src.schemas.orchestration import (
    DecisionStatus,
    InferencePolicy,
)

from src.schemas.prediction import PredictionResult


class FakePredictor:
    def __init__(
        self,
        predicted_digit: int,
        confidence: float,
    ) -> None:
        self.predicted_digit = predicted_digit
        self.confidence = confidence

    def predict(
        self,
        image_bytes: bytes,
    ) -> PredictionResult:
        return PredictionResult(
            predicted_digit=self.predicted_digit,
            confidence=self.confidence,
            probabilities={},
        )


def test_service_accepts_high_confidence_prediction():
    service = OrchestratedInferenceService(
        predictor=FakePredictor(
            predicted_digit=7,
            confidence=0.98,
        ),
        policy=InferencePolicy(
            confidence_threshold=0.90,
        ),
    )

    decision = service.predict(
        b"fake-image-data",
    )

    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 7
    assert decision.confidence == 0.98
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.decision_reason is None
    assert decision.fallback_used is False
    assert decision.review_required is False


def test_service_marks_low_confidence_prediction_uncertain():
    service = OrchestratedInferenceService(
        predictor=FakePredictor(
            predicted_digit=8,
            confidence=0.81,
        ),
        policy=InferencePolicy(
            confidence_threshold=0.90,
        ),
    )

    decision = service.predict(
        b"fake-image-data",
    )

    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 8
    assert decision.confidence == 0.81
    assert decision.status == DecisionStatus.UNCERTAIN
    assert decision.decision_reason == "low_confidence"
    assert decision.fallback_used is False
    assert decision.review_required is True

class InvalidImagePredictor:
    def predict(self, image_bytes: bytes):
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        )


def test_service_propagates_invalid_inference_input():
    service = OrchestratedInferenceService(
        predictor=InvalidImagePredictor(),
        policy=InferencePolicy(
            confidence_threshold=0.90,
        ),
        model_name="cnn",
    )

    with pytest.raises(
        InvalidInferenceInputError,
        match="The uploaded file is not a valid image.",
    ):
        service.predict(
            b"invalid-image-data"
        )