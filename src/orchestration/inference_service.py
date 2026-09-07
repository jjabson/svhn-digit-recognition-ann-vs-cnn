from dataclasses import dataclass

from src.inference import SVHNPredictor
from src.orchestration.adapters import create_svhn_predict_fn
from src.orchestration.service import run_orchestrated_inference
from src.schemas.orchestration import (
    InferenceDecision,
    InferencePolicy,
)


@dataclass
class OrchestratedInferenceService:
    """
    Production-facing inference service that coordinates
    model execution and orchestration policy.
    """

    predictor: SVHNPredictor
    policy: InferencePolicy
    model_name: str = "cnn"

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