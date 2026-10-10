from __future__ import annotations

from config.inference_config import InferenceConfig
from config.inference_config_store import InferenceConfigStore


class InferenceConfigService:
    def __init__(
        self,
        config: InferenceConfig,
        store: InferenceConfigStore | None = None,
    ) -> None:
        self._store = store

        if self._store is None:
            self._config = config
            return

        persisted_config = self._store.load()

        if persisted_config is None:
            self._config = config
        else:
            self._config = persisted_config

    def get_config(self) -> InferenceConfig:
        return self._config

    def set_confidence_threshold(
        self,
        confidence_threshold: float,
    ) -> InferenceConfig:
        current = self._config

        new_config = InferenceConfig(
            model_name=current.model_name,
            confidence_threshold=confidence_threshold,
        )

        if self._store is not None:
            self._store.save(new_config)

        self._config = new_config

        return new_config