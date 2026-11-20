from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)
from config.inference_config_service import (
    InferenceConfigService,
)

from src.inference import SVHNPredictor
from src.observability.metrics import InferenceMetrics
from src.orchestration.inference_service import (
    OrchestratedInferenceService,
)

from config.fallback_inference_config import (
    FallbackInferenceConfig,
)


def create_orchestrated_inference_service(
    config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
    *,
    config_service: InferenceConfigService | None = None,
    metrics: InferenceMetrics | None = None,
    fallback_predictor: SVHNPredictor | None = None,
    fallback_model_name: str | None = None,
    fallback_config: FallbackInferenceConfig | None = None,
) -> OrchestratedInferenceService:
    """
Create the production inference service.

Callers may either supply an InferenceConfig or inject an
existing InferenceConfigService.

Metrics are optional and may be shared across the lifetime
of the serving process.

Fallback inference may either be injected explicitly or
loaded from an enabled FallbackInferenceConfig.
"""
    predictor = SVHNPredictor()

    if fallback_config is not None and fallback_config.enabled:
        if fallback_predictor is not None:
            raise ValueError(
                "fallback_predictor cannot be supplied when "
                "fallback_config enables automatic fallback loading."
            )

        assert fallback_config.model_path is not None
        assert fallback_config.model_name is not None

        fallback_predictor = SVHNPredictor(
            model_path=fallback_config.model_path,
        )

        fallback_model_name = (
            fallback_config.model_name
        )

    if config_service is None:
        config_service = InferenceConfigService(
            config
        )

    return OrchestratedInferenceService(
        predictor=predictor,
        config_service=config_service,
        metrics=metrics,
        fallback_predictor=fallback_predictor,
        fallback_model_name=fallback_model_name,
    )