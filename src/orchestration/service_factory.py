from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)

from src.inference import SVHNPredictor
from src.orchestration.inference_service import OrchestratedInferenceService
from src.schemas.orchestration import InferencePolicy


def create_orchestrated_inference_service(
    config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
) -> OrchestratedInferenceService:
    """
    Create an inference service using the supplied
    orchestration configuration.
    """
    predictor = SVHNPredictor()

    policy = InferencePolicy(
        confidence_threshold=config.confidence_threshold,
    )

    return OrchestratedInferenceService(
        predictor=predictor,
        policy=policy,
        model_name=config.model_name,
    )