from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from config.project_paths import APPROVAL_ARTIFACT_DATABASE_FILE
from src.orchestration.approval_artifact import ApprovalArtifact


class ApprovalArtifactStoreError(RuntimeError):
    """Base persistence error for approval artifacts."""


class ApprovalArtifactAlreadyExistsError(
    ApprovalArtifactStoreError
):
    """Raised when immutable approval evidence already exists."""


class ApprovalArtifactNotFoundError(
    ApprovalArtifactStoreError
):
    """Raised when an approval artifact cannot be found."""


class ApprovalArtifactAlreadyConsumedError(
    ApprovalArtifactStoreError
):
    """Raised when replay attempts target a consumed artifact."""


class ApprovalArtifactStore:
    def __init__(
        self,
        database_file: Path = APPROVAL_ARTIFACT_DATABASE_FILE,
    ) -> None:
        self._database_file = database_file

    @property
    def database_file(self) -> Path:
        return self._database_file

    def initialize(self) -> None:
        self._database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with sqlite3.connect(self._database_file) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS approval_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL UNIQUE,
                    action_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    approved INTEGER NOT NULL
                        CHECK (approved IN (0, 1)),
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    issued_by TEXT NOT NULL,
                    nonce TEXT NOT NULL UNIQUE,
                    consumed_at TEXT
                )
                """
            )

    def save(
        self,
        artifact: ApprovalArtifact,
    ) -> None:
        self.initialize()

        try:
            with sqlite3.connect(
                self._database_file
            ) as connection:
                connection.execute(
                    """
                    INSERT INTO approval_artifacts (
                        artifact_id,
                        request_id,
                        action_name,
                        arguments_json,
                        approved,
                        issued_at,
                        expires_at,
                        issued_by,
                        nonce,
                        consumed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.artifact_id,
                        artifact.request_id,
                        artifact.action_name,
                        artifact.arguments_json,
                        int(artifact.approved),
                        artifact.issued_at.isoformat(),
                        artifact.expires_at.isoformat(),
                        artifact.issued_by,
                        artifact.nonce,
                        (
                            artifact.consumed_at.isoformat()
                            if artifact.consumed_at is not None
                            else None
                        ),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ApprovalArtifactAlreadyExistsError(
                "Approval artifact, request, or nonce already exists."
            ) from exc

    def get_by_artifact_id(
        self,
        artifact_id: str,
    ) -> ApprovalArtifact | None:
        self.initialize()

        with sqlite3.connect(self._database_file) as connection:
            row = connection.execute(
                """
                SELECT
                    artifact_id,
                    request_id,
                    action_name,
                    arguments_json,
                    approved,
                    issued_at,
                    expires_at,
                    issued_by,
                    nonce,
                    consumed_at
                FROM approval_artifacts
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()

        return self._row_to_artifact(row)

    def get_by_request_id(
        self,
        request_id: str,
    ) -> ApprovalArtifact | None:
        self.initialize()

        with sqlite3.connect(self._database_file) as connection:
            row = connection.execute(
                """
                SELECT
                    artifact_id,
                    request_id,
                    action_name,
                    arguments_json,
                    approved,
                    issued_at,
                    expires_at,
                    issued_by,
                    nonce,
                    consumed_at
                FROM approval_artifacts
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()

        return self._row_to_artifact(row)

    def consume(
        self,
        artifact_id: str,
        *,
        consumed_at: datetime,
    ) -> ApprovalArtifact:
        if consumed_at.tzinfo is None:
            raise ValueError(
                "consumed_at must be timezone-aware."
            )

        self.initialize()

        with sqlite3.connect(self._database_file) as connection:
            cursor = connection.execute(
                """
                UPDATE approval_artifacts
                SET consumed_at = ?
                WHERE artifact_id = ?
                  AND consumed_at IS NULL
                """,
                (
                    consumed_at.isoformat(),
                    artifact_id,
                ),
            )

            if cursor.rowcount == 0:
                existing = connection.execute(
                    """
                    SELECT consumed_at
                    FROM approval_artifacts
                    WHERE artifact_id = ?
                    """,
                    (artifact_id,),
                ).fetchone()

                if existing is None:
                    raise ApprovalArtifactNotFoundError(
                        f"Unknown approval artifact: {artifact_id}"
                    )

                raise ApprovalArtifactAlreadyConsumedError(
                    f"Approval artifact already consumed: {artifact_id}"
                )

        artifact = self.get_by_artifact_id(artifact_id)

        if artifact is None:
            raise ApprovalArtifactNotFoundError(
                f"Unknown approval artifact: {artifact_id}"
            )

        return artifact

    @staticmethod
    def _row_to_artifact(
        row: tuple | None,
    ) -> ApprovalArtifact | None:
        if row is None:
            return None

        (
            artifact_id,
            request_id,
            action_name,
            arguments_json,
            approved,
            issued_at,
            expires_at,
            issued_by,
            nonce,
            consumed_at,
        ) = row

        return ApprovalArtifact(
            artifact_id=str(artifact_id),
            request_id=str(request_id),
            action_name=str(action_name),
            arguments_json=str(arguments_json),
            approved=bool(approved),
            issued_at=datetime.fromisoformat(str(issued_at)),
            expires_at=datetime.fromisoformat(str(expires_at)),
            issued_by=str(issued_by),
            nonce=str(nonce),
            consumed_at=(
                datetime.fromisoformat(str(consumed_at))
                if consumed_at is not None
                else None
            ),
        )