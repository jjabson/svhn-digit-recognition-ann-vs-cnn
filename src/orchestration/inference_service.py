from dataclasses import dataclass

from src.inference import SVHNPredictor
from src.orchestration.adapters import create_svhn_predict_fn
from src.orchestration.service import run_orchestrated_inference
from src.schemas.orchestration import (
    InferenceDecision,
    InferencePolicy,
)
from tools.model.inspect_cnn import ArchitectureInfo
from tools.model.model_service import get_model_architecture


@dataclass
class OrchestratedInferenceService:
    """
    Production-facing inference service that coordinates
    model execution and orchestration policy.
    """

    predictor: SVHNPredictor
    policy: InferencePolicy
    model_name: str = "cnn"

    def get_config(self) -> dict[str, object]:
        """
        Return the active configuration used by this inference service.
        """
        return {
            "model_name": self.model_name,
            "confidence_threshold": self.policy.confidence_threshold,
        }

    def get_model_architecture(self) -> ArchitectureInfo:
        """
        Return the architecture of the model currently used for inference.
        """
        return get_model_architecture(self.predictor.model)

    def predict(
        self,
        image_bytes: bytes,
    ) -> InferenceDecision:
        predict_fn = create_svhn_predict_fn(
            predictor=self.predictor,
            image_bytes=image_bytes,
        )

        return run_orchestrated_inference(
            primary_model_name=self.model_name,
            primary_predict_fn=predict_fn,
            policy=self.policy,
        )