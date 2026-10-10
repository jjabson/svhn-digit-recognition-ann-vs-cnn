from __future__ import annotations

import sqlite3
from pathlib import Path

from config.inference_config import InferenceConfig
from config.project_paths import RUNTIME_STATE_DATABASE_FILE


class InferenceConfigStore:
    def __init__(
        self,
        database_file: Path = RUNTIME_STATE_DATABASE_FILE,
    ) -> None:
        self._database_file = database_file

    def initialize(self) -> None:
        self._database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self._database_file) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    model_name TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load(self) -> InferenceConfig | None:
        self.initialize()

        with sqlite3.connect(self._database_file) as connection:
            row = connection.execute(
                """
                SELECT
                    model_name,
                    confidence_threshold
                FROM inference_config
                WHERE id = 1
                """
            ).fetchone()

        if row is None:
            return None

        model_name, confidence_threshold = row

        return InferenceConfig(
            model_name=str(model_name),
            confidence_threshold=float(confidence_threshold),
        )

    def save(
        self,
        config: InferenceConfig,
    ) -> None:
        self.initialize()

        with sqlite3.connect(self._database_file) as connection:
            connection.execute(
                """
                INSERT INTO inference_config (
                    id,
                    model_name,
                    confidence_threshold
                )
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    model_name = excluded.model_name,
                    confidence_threshold = excluded.confidence_threshold,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    config.model_name,
                    config.confidence_threshold,
                ),
            )