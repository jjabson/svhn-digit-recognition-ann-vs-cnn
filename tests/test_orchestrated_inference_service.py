import pytest

from src.orchestration.errors import InvalidInferenceInputError
from src.preprocessing import InvalidImageError
from src.orchestration.inference_service import OrchestratedInferenceService
from src.schemas.orchestration import (
    DecisionStatus,
    InferencePolicy,
)

from src.schemas.prediction import PredictionResult
from config.inference_config import InferenceConfig
from config.inference_config_service import InferenceConfigService

def create_config_service(
    *,
    model_name: str = "cnn",
    confidence_threshold: float = 0.90,
) -> InferenceConfigService:
    return InferenceConfigService(
        InferenceConfig(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
    )


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

    def create_config_service(
            *,
            model_name: str = "cnn",
            confidence_threshold: float = 0.90,
    ) -> InferenceConfigService:
        return InferenceConfigService(
            InferenceConfig(
                model_name=model_name,
                confidence_threshold=confidence_threshold,
            )
        )


def test_service_accepts_high_confidence_prediction():
    service = OrchestratedInferenceService(
        predictor=FakePredictor(
            predicted_digit=7,
            confidence=0.98,
        ),
        config_service=create_config_service(
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
    assert decision.review_required is False


def test_service_accepts_high_confidence_prediction():
    service = OrchestratedInferenceService(
        predictor=FakePredictor(
            predicted_digit=7,
            confidence=0.98,
        ),
        config_service=create_config_service(
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
    assert decision.review_required is False

class InvalidImagePredictor:
    def predict(self, image_bytes: bytes):
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        )


def test_service_propagates_invalid_inference_input():
    service = OrchestratedInferenceService(
        predictor=InvalidImagePredictor(),
        config_service=create_config_service(
            confidence_threshold=0.90,
        ),
    )

    with pytest.raises(
        InvalidInferenceInputError,
        match="The uploaded file is not a valid image.",
    ):
        service.predict(
            b"fake-image-data",
        )