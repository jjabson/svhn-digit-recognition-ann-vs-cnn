from fastapi.testclient import TestClient

import api.app as api_module
from src.orchestration.inference_service import (
    OrchestratedInferenceService,
)
from src.schemas.orchestration import InferencePolicy


def test_inference_config_endpoint_returns_serving_metadata(
    monkeypatch,
):
    class FakePredictor:
        pass

    service = OrchestratedInferenceService(
        predictor=FakePredictor(),
        policy=InferencePolicy(
            confidence_threshold=0.87,
        ),
        model_name="test-cnn",
    )

    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: service,
    )

    with TestClient(api_module.app) as client:
        response = client.get(
            "/inference/config",
        )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-cnn",
        "confidence_threshold": 0.87,
    }

def test_inference_config_endpoint_returns_503_when_service_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda: None,
    )

    with TestClient(api_module.app) as client:
        response = client.get(
            "/inference/config",
        )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Inference service is not available."
    }