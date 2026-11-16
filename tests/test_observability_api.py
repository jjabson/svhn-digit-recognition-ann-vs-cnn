from fastapi.testclient import TestClient

import api.app as api_module
from src.schemas.orchestration import (
    DecisionStatus,
    InferenceDecision,
)

def test_api_lifespan_wires_process_local_inference_metrics():
    assert api_module.inference_service is None
    assert api_module.inference_metrics is None

    with TestClient(api_module.app):
        assert api_module.inference_service is not None
        assert api_module.inference_metrics is not None

        assert (
            api_module.inference_service.metrics
            is api_module.inference_metrics
        )

        snapshot = api_module.inference_metrics.snapshot()

        assert snapshot.total_attempts == 0
        assert snapshot.accepted == 0
        assert snapshot.uncertain == 0
        assert snapshot.failed == 0
        assert snapshot.total_latency_ms == 0.0
        assert snapshot.average_latency_ms == 0.0
        assert snapshot.max_latency_ms == 0.0

    assert api_module.inference_service is None
    assert api_module.inference_metrics is None


def test_inference_metrics_endpoint_returns_process_local_snapshot():
    with TestClient(api_module.app) as client:
        assert api_module.inference_metrics is not None

        api_module.inference_metrics.record(
            status="ACCEPTED",
            duration_ms=10.0,
        )
        api_module.inference_metrics.record(
            status="UNCERTAIN",
            duration_ms=30.0,
        )

        response = client.get(
            "/observability/inference"
        )

        assert response.status_code == 200

        assert response.json() == {
            "total_attempts": 2,
            "accepted": 1,
            "uncertain": 1,
            "failed": 0,
            "total_latency_ms": 40.0,
            "average_latency_ms": 20.0,
            "max_latency_ms": 30.0,
        }


def test_inference_metrics_endpoint_returns_503_when_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda *, metrics=None: None,
    )

    with TestClient(api_module.app) as client:
        api_module.inference_metrics = None

        response = client.get(
            "/observability/inference"
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Inference metrics are unavailable."
        }



def test_inference_metrics_endpoint_reflects_orchestrated_prediction(
    monkeypatch,
):
    class MetricsRecordingInferenceService:
        def __init__(self, metrics):
            self.metrics = metrics

        def predict(
            self,
            image_bytes: bytes,
        ) -> InferenceDecision:
            self.metrics.record(
                status="UNCERTAIN",
                duration_ms=12.5,
            )

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

    monkeypatch.setattr(
        api_module,
        "create_orchestrated_inference_service",
        lambda *, metrics=None: (
            MetricsRecordingInferenceService(metrics)
        ),
    )

    with TestClient(api_module.app) as client:
        prediction_response = client.post(
            "/predict/orchestrated",
            files={
                "file": (
                    "digit.png",
                    b"fake-image-data",
                    "image/png",
                )
            },
        )

        metrics_response = client.get(
            "/observability/inference"
        )

    assert prediction_response.status_code == 200
    assert metrics_response.status_code == 200

    body = metrics_response.json()

    assert body["total_attempts"] == 1
    assert body["accepted"] == 0
    assert body["uncertain"] == 1
    assert body["failed"] == 0
    assert body["total_latency_ms"] == 12.5
    assert body["average_latency_ms"] == 12.5
    assert body["max_latency_ms"] == 12.5