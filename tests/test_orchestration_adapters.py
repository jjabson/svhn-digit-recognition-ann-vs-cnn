import pytest
from src.orchestration.errors import InvalidInferenceInputError
from src.preprocessing import InvalidImageError
from src.orchestration.adapters import create_svhn_predict_fn
from src.orchestration.execution import execute_model_attempt
from src.schemas.prediction import PredictionResult


class FakeSVHNPredictor:
    def predict(self, image_bytes: bytes) -> PredictionResult:
        return PredictionResult(
            predicted_digit=7,
            confidence=0.982,
            probabilities={},
        )


def test_svhn_adapter_returns_generic_prediction_tuple():
    predictor = FakeSVHNPredictor()

    predict_fn = create_svhn_predict_fn(
        predictor=predictor,
        image_bytes=b"fake-image-data",
    )

    predicted_digit, confidence = predict_fn()

    assert predicted_digit == 7
    assert confidence == 0.982


def test_svhn_adapter_integrates_with_execution_layer():
    predictor = FakeSVHNPredictor()

    predict_fn = create_svhn_predict_fn(
        predictor=predictor,
        image_bytes=b"fake-image-data",
    )

    attempt = execute_model_attempt(
        model_name="cnn",
        predict_fn=predict_fn,
    )

    assert attempt.model_name == "cnn"
    assert attempt.predicted_digit == 7
    assert attempt.confidence == 0.982
    assert attempt.succeeded is True
    assert attempt.error_message is None

class InvalidImagePredictor:
    def predict(self, image_bytes: bytes):
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        )


def test_svhn_adapter_translates_invalid_image_error():
    predictor = InvalidImagePredictor()

    predict_fn = create_svhn_predict_fn(
        predictor=predictor,
        image_bytes=b"invalid-image-data",
    )

    with pytest.raises(
        InvalidInferenceInputError,
        match="The uploaded file is not a valid image.",
    ):
        predict_fn()