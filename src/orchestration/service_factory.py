from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)
from config.inference_config_service import InferenceConfigService

from src.inference import SVHNPredictor
from src.orchestration.inference_service import (
    OrchestratedInferenceService,
)


def create_orchestrated_inference_service(
    config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
    config_service: InferenceConfigService | None = None,
) -> OrchestratedInferenceService:
    """
    Create an inference service using the supplied
    orchestration configuration.

    A shared InferenceConfigService may be injected so runtime
    configuration updates are observed by the inference service.
    """
    predictor = SVHNPredictor()

    if config_service is None:
        config_service = InferenceConfigService(config)

    return OrchestratedInferenceService(
        predictor=predictor,
        config_service=config_service,
    )