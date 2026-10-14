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
        """
        Return the current active configuration.

        When persistent storage is configured, refresh from SQLite
        before returning so long-running processes observe updates
        written by other processes.
        """
        if self._store is not None:
            persisted_config = self._store.load()

            if persisted_config is not None:
                self._config = persisted_config

        return self._config

    def set_confidence_threshold(
        self,
        confidence_threshold: float,
    ) -> InferenceConfig:
        # Refresh first so a mutation is based on the most recent
        # persisted configuration rather than stale process-local state.
        current = self.get_config()

        new_config = InferenceConfig(
            model_name=current.model_name,
            confidence_threshold=confidence_threshold,
        )

        if self._store is not None:
            self._store.save(new_config)

        self._config = new_config

        return new_config