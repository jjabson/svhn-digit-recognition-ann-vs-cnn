import importlib

from fastapi.testclient import TestClient
from src.orchestration.errors import InvalidInferenceInputError

from src.schemas.orchestration import (
    DecisionStatus,
    InferenceDecision,
)


api_module = importlib.import_module("api.app")


class FakeOrchestratedInferenceService:
    def predict(
        self,
        image_bytes: bytes,
    ) -> InferenceDecision:
        return InferenceDecision(
            selected_model="cnn",
            predicted_digit=8,
            confidence=0.818678081035614,
            status=DecisionStatus.UNCERTAIN,
            decision_reason="low_confidence",
            fallback_used=False,
            fallback_reason=None,
            review_required=True,
        )


def test_orchestrated_prediction_endpoint_returns_decision(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: FakeOrchestratedInferenceService(),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict/orchestrated",
            files={
                "file": (
                    "digit.png",
                    b"fake-image-data",
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_digit"] == 8
    assert body["confidence"] == 0.818678081035614
    assert body["confidence_percent"] == "81.87%"
    assert body["selected_model"] == "cnn"
    assert body["status"] == "uncertain"
    assert body["decision_reason"] == "low_confidence"
    assert body["fallback_used"] is False
    assert body["fallback_reason"] is None
    assert body["review_required"] is True

def test_orchestrated_prediction_rejects_unsupported_content_type(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: FakeOrchestratedInferenceService(),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict/orchestrated",
            files={
                "file": (
                    "digit.txt",
                    b"not-an-image",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Only PNG and JPEG images are supported."
    }

def test_orchestrated_prediction_returns_failed_model_decision(
    monkeypatch,
):
    class FailedInferenceService:
        def predict(
            self,
            image_bytes: bytes,
        ) -> InferenceDecision:
            return InferenceDecision(
                selected_model=None,
                predicted_digit=None,
                confidence=None,
                status=DecisionStatus.FAILED,
                decision_reason="primary_model_failed",
                fallback_used=False,
                fallback_reason=None,
                review_required=True,
            )

    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: FailedInferenceService(),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict/orchestrated",
            files={
                "file": (
                    "digit.png",
                    b"valid-input-placeholder",
                    "image/png",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "failed"
    assert body["decision_reason"] == "primary_model_failed"
    assert body["selected_model"] is None
    assert body["predicted_digit"] is None
    assert body["review_required"] is True

def test_orchestrated_prediction_rejects_invalid_image_input(
    monkeypatch,
):
    class InvalidInputService:
        def predict(self, image_bytes: bytes):
            raise InvalidInferenceInputError(
                "The uploaded file is not a valid image."
            )

    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: InvalidInputService(),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict/orchestrated",
            files={
                "file": (
                    "digit.png",
                    b"invalid-image-data",
                    "image/png",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The uploaded file is not a valid image."
    }