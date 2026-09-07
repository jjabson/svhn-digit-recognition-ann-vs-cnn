"""
Configuration values that control inference and orchestration behavior.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    confidence_threshold: float = 0.90
    model_name: str = "cnn"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name must not be empty."
            )


DEFAULT_INFERENCE_CONFIG = InferenceConfig()