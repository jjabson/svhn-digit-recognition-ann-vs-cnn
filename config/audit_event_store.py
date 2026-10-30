from __future__ import annotations

import sqlite3
from pathlib import Path

from config.project_paths import (
    AUDIT_EVENT_DATABASE_FILE,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEvent,
    AuditEventType,
)


class AuditEventStoreError(RuntimeError):
    """Base error for audit-event persistence."""


class DuplicateAuditEventError(AuditEventStoreError):
    """Raised when an event_id already exists."""


class AuditEventStore:
    """
    Append-only SQLite persistence for AuditEvent records.

    Audit events represent immutable historical facts.

    Existing events are never updated or deleted through this store.
    """

    def __init__(
        self,
        database_file: Path = AUDIT_EVENT_DATABASE_FILE,
    ) -> None:
        self._database_file = database_file

        self._database_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    @property
    def database_file(self) -> Path:
        return self._database_file

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_file
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_event (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,

                    actor_type TEXT NOT NULL,
                    actor_id TEXT,
                    source_component TEXT NOT NULL,

                    request_id TEXT,
                    thread_id TEXT,

                    payload_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_audit_event_occurred_at
                ON audit_event (
                    occurred_at,
                    event_id
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_audit_event_request_id
                ON audit_event (
                    request_id
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_audit_event_thread_id
                ON audit_event (
                    thread_id
                )
                """
            )

    def append(
        self,
        event: AuditEvent,
    ) -> None:
        """
        Append one immutable audit event.

        Existing event IDs are rejected rather than replaced.
        """

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO audit_event (
                        event_id,
                        event_type,
                        occurred_at,
                        actor_type,
                        actor_id,
                        source_component,
                        request_id,
                        thread_id,
                        payload_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.event_type.value,
                        event.occurred_at.isoformat(),
                        event.actor_type.value,
                        event.actor_id,
                        event.source_component,
                        event.request_id,
                        event.thread_id,
                        event.payload_json,
                    ),
                )

        except sqlite3.IntegrityError as exc:
            raise DuplicateAuditEventError(
                f"Audit event already exists: "
                f"{event.event_id}"
            ) from exc

    def get(
        self,
        event_id: str,
    ) -> AuditEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    event_id,
                    event_type,
                    occurred_at,
                    actor_type,
                    actor_id,
                    source_component,
                    request_id,
                    thread_id,
                    payload_json
                FROM audit_event
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_event(row)

    def list_all(
        self,
    ) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    event_id,
                    event_type,
                    occurred_at,
                    actor_type,
                    actor_id,
                    source_component,
                    request_id,
                    thread_id,
                    payload_json
                FROM audit_event
                ORDER BY
                    occurred_at ASC,
                    event_id ASC
                """
            ).fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def list_by_request_id(
        self,
        request_id: str,
    ) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    event_id,
                    event_type,
                    occurred_at,
                    actor_type,
                    actor_id,
                    source_component,
                    request_id,
                    thread_id,
                    payload_json
                FROM audit_event
                WHERE request_id = ?
                ORDER BY
                    occurred_at ASC,
                    event_id ASC
                """,
                (request_id,),
            ).fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def list_by_thread_id(
        self,
        thread_id: str,
    ) -> list[AuditEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    event_id,
                    event_type,
                    occurred_at,
                    actor_type,
                    actor_id,
                    source_component,
                    request_id,
                    thread_id,
                    payload_json
                FROM audit_event
                WHERE thread_id = ?
                ORDER BY
                    occurred_at ASC,
                    event_id ASC
                """,
                (thread_id,),
            ).fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    @staticmethod
    def _row_to_event(
        row: sqlite3.Row,
    ) -> AuditEvent:
        from datetime import datetime

        return AuditEvent(
            event_id=row["event_id"],
            event_type=AuditEventType(
                row["event_type"]
            ),
            occurred_at=datetime.fromisoformat(
                row["occurred_at"]
            ),
            actor_type=AuditActorType(
                row["actor_type"]
            ),
            actor_id=row["actor_id"],
            source_component=row[
                "source_component"
            ],
            request_id=row["request_id"],
            thread_id=row["thread_id"],
            payload_json=row["payload_json"],
        )