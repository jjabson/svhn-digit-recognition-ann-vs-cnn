from dataclasses import dataclass
from time import perf_counter

from config.inference_config_service import InferenceConfigService
from src.inference import SVHNPredictor
from src.observability.metrics import InferenceMetrics
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

    Runtime inference configuration is read from the shared
    InferenceConfigService so configuration changes take effect
    without recreating the inference service.
    """

    predictor: SVHNPredictor
    config_service: InferenceConfigService
    metrics: InferenceMetrics | None = None

    @property
    def model_name(self) -> str:
        """
        Return the currently configured model name.
        """
        return self.config_service.get_config().model_name

    @property
    def policy(self) -> InferencePolicy:
        """
        Build the active inference policy from the current
        runtime configuration.
        """
        config = self.config_service.get_config()

        return InferencePolicy(
            confidence_threshold=config.confidence_threshold,
        )

    def get_config(self) -> dict[str, object]:
        """
        Return the active configuration used by this inference service.
        """
        config = self.config_service.get_config()

        return {
            "model_name": config.model_name,
            "confidence_threshold": config.confidence_threshold,
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
        started_at = perf_counter()

        config = self.config_service.get_config()

        policy = InferencePolicy(
            confidence_threshold=config.confidence_threshold,
        )

        predict_fn = create_svhn_predict_fn(
            predictor=self.predictor,
            image_bytes=image_bytes,
        )

        decision = run_orchestrated_inference(
            primary_model_name=config.model_name,
            primary_predict_fn=predict_fn,
            policy=policy,
        )

        if self.metrics is not None:
            duration_ms = (
                                  perf_counter() - started_at
                          ) * 1000.0

            status = decision.status

            if hasattr(status, "value"):
                status = status.value

            self.metrics.record(
                status=str(status),
                duration_ms=duration_ms,
            )

        return decision