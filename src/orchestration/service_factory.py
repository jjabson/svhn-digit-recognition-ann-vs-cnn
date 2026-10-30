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


def create_orchestrated_inference_service(
    config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
    *,
    config_service: InferenceConfigService | None = None,
    metrics: InferenceMetrics | None = None,
) -> OrchestratedInferenceService:
    """
    Create the production inference service.

    Callers may either supply an InferenceConfig or inject an
    existing InferenceConfigService.

    Metrics are optional and may be shared across the lifetime
    of the serving process.
    """
    predictor = SVHNPredictor()

    if config_service is None:
        config_service = InferenceConfigService(
            config
        )

    return OrchestratedInferenceService(
        predictor=predictor,
        config_service=config_service,
        metrics=metrics,
    )