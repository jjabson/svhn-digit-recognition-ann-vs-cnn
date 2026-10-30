from config.inference_config_service import InferenceConfigService

from src.orchestration.inference_service import OrchestratedInferenceService
from src.orchestration.service_factory import (
    create_orchestrated_inference_service,
)

from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)

from src.observability.metrics import InferenceMetrics

def test_service_factory_creates_canonical_inference_service(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    service = create_orchestrated_inference_service()

    assert isinstance(
        service,
        OrchestratedInferenceService,
    )

    assert isinstance(
        service.predictor,
        FakePredictor,
    )

    assert (
        service.policy.confidence_threshold
        == DEFAULT_INFERENCE_CONFIG.confidence_threshold
    )

    assert (
        service.model_name
        == DEFAULT_INFERENCE_CONFIG.model_name
    )

def test_service_factory_accepts_custom_inference_config(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    custom_config = InferenceConfig(
        confidence_threshold=0.75,
        model_name="test-model",
    )

    service = create_orchestrated_inference_service(
        config=custom_config,
    )

    assert (
        service.policy.confidence_threshold
        == 0.75
    )

    assert service.model_name == "test-model"


def test_service_factory_injects_metrics(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    metrics = InferenceMetrics()

    service = create_orchestrated_inference_service(
        config_service=config_service,
        metrics=metrics,
    )

    assert service.metrics is metrics
    assert service.config_service is config_service

def test_service_factory_allows_metrics_to_be_omitted(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    service = create_orchestrated_inference_service(
        config_service=config_service,
    )

    assert service.metrics is None