import pytest
from config.inference_config_service import InferenceConfigService

from src.orchestration.inference_service import OrchestratedInferenceService
from src.orchestration.service_factory import (
    create_orchestrated_inference_service,
)

from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)
from config.fallback_inference_config import (
    FallbackInferenceConfig,
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


def test_service_factory_injects_fallback_predictor(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    fallback_predictor = FakePredictor()

    service = create_orchestrated_inference_service(
        fallback_predictor=fallback_predictor,
        fallback_model_name="ann",
    )

    assert service.fallback_predictor is fallback_predictor
    assert service.fallback_model_name == "ann"


def test_service_factory_defaults_to_no_fallback(
    monkeypatch,
):
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    service = create_orchestrated_inference_service()

    assert service.fallback_predictor is None
    assert service.fallback_model_name is None


def test_service_factory_loads_enabled_fallback_config(
    monkeypatch,
    tmp_path,
):
    created_paths = []

    class FakePredictor:
        def __init__(
            self,
            model_path=None,
        ):
            created_paths.append(model_path)

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    fallback_path = (
        tmp_path / "fallback.keras"
    )

    fallback_path.write_bytes(
        b"fake-model"
    )

    fallback_config = FallbackInferenceConfig(
        enabled=True,
        model_name="ann",
        model_path=fallback_path,
    )

    service = create_orchestrated_inference_service(
        fallback_config=fallback_config,
    )

    assert len(created_paths) == 2

    assert created_paths[0] is None
    assert created_paths[1] == fallback_path

    assert service.fallback_model_name == "ann"

    assert isinstance(
        service.fallback_predictor,
        FakePredictor,
    )


def test_service_factory_does_not_load_disabled_fallback_config(
    monkeypatch,
):
    created_paths = []

    class FakePredictor:
        def __init__(
            self,
            model_path=None,
        ):
            created_paths.append(model_path)

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    fallback_config = FallbackInferenceConfig(
        enabled=False,
    )

    service = create_orchestrated_inference_service(
        fallback_config=fallback_config,
    )

    assert created_paths == [None]

    assert service.fallback_predictor is None
    assert service.fallback_model_name is None


def test_service_factory_rejects_explicit_and_automatic_fallback(
    monkeypatch,
    tmp_path,
):
    class FakePredictor:
        def __init__(
            self,
            model_path=None,
        ):
            pass

    monkeypatch.setattr(
        "src.orchestration.service_factory.SVHNPredictor",
        FakePredictor,
    )

    fallback_path = (
        tmp_path / "fallback.keras"
    )

    fallback_path.write_bytes(
        b"fake-model"
    )

    fallback_config = FallbackInferenceConfig(
        enabled=True,
        model_name="ann",
        model_path=fallback_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            "fallback_predictor cannot be supplied"
        ),
    ):
        create_orchestrated_inference_service(
            fallback_predictor=FakePredictor(),
            fallback_model_name="ann",
            fallback_config=fallback_config,
        )