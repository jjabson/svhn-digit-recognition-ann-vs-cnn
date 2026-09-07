from config.inference_config import DEFAULT_INFERENCE_CONFIG

from src.orchestration.inference_service import OrchestratedInferenceService
from src.orchestration.service_factory import (
    create_orchestrated_inference_service,
)

from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)

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