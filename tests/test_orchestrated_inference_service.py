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
from src.observability.metrics import InferenceMetrics
from types import SimpleNamespace

import src.orchestration.inference_service as inference_service_module

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

def test_service_records_accepted_metrics():
    predictor = FakePredictor(
        predicted_digit=7,
        confidence=0.99,
    )

    metrics = InferenceMetrics()

    service = OrchestratedInferenceService(
        predictor=predictor,
        config_service=InferenceConfigService(
            InferenceConfig(
                model_name="cnn",
                confidence_threshold=0.90,
            )
        ),
        metrics=metrics,
    )

    decision = service.predict(
        b"valid-image"
    )

    snapshot = metrics.snapshot()

    assert (
        decision.status.value
        == "accepted"
    )

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 1
    assert snapshot.uncertain == 0
    assert snapshot.failed == 0

    assert snapshot.total_latency_ms >= 0.0
    assert snapshot.max_latency_ms >= 0.0

def test_service_records_uncertain_metrics():
    predictor = FakePredictor(
        predicted_digit=7,
        confidence=0.50,
    )

    metrics = InferenceMetrics()

    service = OrchestratedInferenceService(
        predictor=predictor,
        config_service=InferenceConfigService(
            InferenceConfig(
                model_name="cnn",
                confidence_threshold=0.90,
            )
        ),
        metrics=metrics,
    )

    decision = service.predict(
        b"valid-image"
    )

    snapshot = metrics.snapshot()

    assert (
        decision.status.value
        == "uncertain"
    )

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 1
    assert snapshot.failed == 0


def test_invalid_input_is_not_recorded_in_metrics():
    metrics = InferenceMetrics()

    service = OrchestratedInferenceService(
        predictor=InvalidImagePredictor(),
        config_service=InferenceConfigService(
            InferenceConfig(
                model_name="cnn",
                confidence_threshold=0.90,
            )
        ),
        metrics=metrics,
    )

    with pytest.raises(
        InvalidInferenceInputError
    ):
        service.predict(
            b"invalid-image"
        )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 0
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 0
    assert snapshot.failed == 0

def test_service_records_failed_metrics(
    monkeypatch,
):
    metrics = InferenceMetrics()

    failed_decision = SimpleNamespace(
        status=SimpleNamespace(
            value="failed"
        ),
        fallback_used=False,
    )

    def fake_run_orchestrated_inference(
        **kwargs,
    ):
        return failed_decision

    monkeypatch.setattr(
        inference_service_module,
        "run_orchestrated_inference",
        fake_run_orchestrated_inference,
    )

    service = OrchestratedInferenceService(
        predictor=object(),
        config_service=InferenceConfigService(
            InferenceConfig(
                model_name="cnn",
                confidence_threshold=0.90,
            )
        ),
        metrics=metrics,
    )

    decision = service.predict(
        b"valid-image"
    )

    snapshot = metrics.snapshot()

    assert decision.status.value == "failed"

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 0
    assert snapshot.failed == 1

    assert snapshot.total_latency_ms >= 0.0
    assert snapshot.max_latency_ms >= 0.0


def test_service_records_fallback_usage(
    monkeypatch,
):
    metrics = InferenceMetrics()

    fallback_decision = SimpleNamespace(
        status=SimpleNamespace(
            value="accepted"
        ),
        fallback_used=True,
    )

    def fake_run_orchestrated_inference(
        **kwargs,
    ):
        return fallback_decision

    monkeypatch.setattr(
        inference_service_module,
        "run_orchestrated_inference",
        fake_run_orchestrated_inference,
    )

    service = OrchestratedInferenceService(
        predictor=object(),
        config_service=InferenceConfigService(
            InferenceConfig(
                model_name="cnn",
                confidence_threshold=0.90,
            )
        ),
        metrics=metrics,
    )

    decision = service.predict(
        b"valid-image"
    )

    snapshot = metrics.snapshot()

    assert decision.fallback_used is True

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 1
    assert snapshot.fallback_used_count == 1
    assert snapshot.fallback_rate == 1.0