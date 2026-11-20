from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


FALLBACK_ENABLED_ENV = "SVHN_FALLBACK_ENABLED"
FALLBACK_MODEL_NAME_ENV = "SVHN_FALLBACK_MODEL_NAME"
FALLBACK_MODEL_PATH_ENV = "SVHN_FALLBACK_MODEL_PATH"


class FallbackInferenceConfigurationError(ValueError):
    """Raised when fallback inference is configured incorrectly."""


@dataclass(frozen=True)
class FallbackInferenceConfig:
    enabled: bool
    model_name: str | None = None
    model_path: Path | None = None

    @classmethod
    def from_environment(cls) -> "FallbackInferenceConfig":
        enabled = (
            os.getenv(FALLBACK_ENABLED_ENV, "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )

        model_name = os.getenv(
            FALLBACK_MODEL_NAME_ENV
        )

        model_path_value = os.getenv(
            FALLBACK_MODEL_PATH_ENV
        )

        model_path = (
            Path(model_path_value)
            if model_path_value
            else None
        )

        config = cls(
            enabled=enabled,
            model_name=model_name,
            model_path=model_path,
        )

        config.validate()
        return config

    def validate(self) -> None:
        if not self.enabled:
            return

        if (
            self.model_name is None
            or not self.model_name.strip()
        ):
            raise FallbackInferenceConfigurationError(
                f"{FALLBACK_MODEL_NAME_ENV} is required when "
                f"{FALLBACK_ENABLED_ENV}=true."
            )

        if self.model_path is None:
            raise FallbackInferenceConfigurationError(
                f"{FALLBACK_MODEL_PATH_ENV} is required when "
                f"{FALLBACK_ENABLED_ENV}=true."
            )

        if not self.model_path.exists():
            raise FallbackInferenceConfigurationError(
                "Fallback model file was not found: "
                f"{self.model_path}"
            )

        if not self.model_path.is_file():
            raise FallbackInferenceConfigurationError(
                "Fallback model path must reference a file: "
                f"{self.model_path}"
            )