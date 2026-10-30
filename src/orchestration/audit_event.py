from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class AuditEventError(ValueError):
    """Base error for invalid audit events."""


class InvalidAuditEventError(AuditEventError):
    """Raised when an audit event violates domain invariants."""


class AuditEventType(str, Enum):
    """
    Immutable operational facts emitted during a controlled-action lifecycle.
    """

    CONTROLLED_ACTION_REQUESTED = "controlled_action_requested"
    APPROVAL_REQUIRED = "approval_required"

    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_CANCELLED = "approval_cancelled"

    CONTROLLED_ACTION_EXECUTED = "controlled_action_executed"
    CONTROLLED_ACTION_FAILED = "controlled_action_failed"


class AuditActorType(str, Enum):
    """
    Trust class of the actor associated with an audit event.

    This describes who caused or authorized an event. It does not itself
    grant authorization.
    """

    MCP_CLIENT = "mcp_client"
    TRUSTED_HUMAN = "trusted_human"
    SYSTEM = "system"


def canonicalize_audit_payload(
    payload: Mapping[str, Any],
) -> str:
    """
    Convert an audit payload into deterministic JSON.

    Audit payloads are facts associated with an event. Canonical JSON makes
    their stored representation deterministic while rejecting non-portable
    values such as NaN and Infinity.
    """

    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidAuditEventError(
            "Audit payload must contain valid JSON-compatible values."
        ) from exc


@dataclass(frozen=True)
class AuditEvent:
    """
    Immutable operational fact describing one point in a controlled-action
    lifecycle.

    Audit events are facts, not mutable workflow state.

    Persistence will be added separately. Once persisted, events are intended
    to be append-only.
    """

    event_id: str
    event_type: AuditEventType
    occurred_at: datetime

    actor_type: AuditActorType
    source_component: str

    payload_json: str

    request_id: str | None = None
    thread_id: str | None = None
    actor_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_id.strip():
            raise InvalidAuditEventError(
                "event_id must not be empty."
            )

        if not isinstance(self.event_type, AuditEventType):
            raise InvalidAuditEventError(
                "event_type must be an AuditEventType."
            )

        if self.occurred_at.tzinfo is None:
            raise InvalidAuditEventError(
                "occurred_at must be timezone-aware."
            )

        if not isinstance(self.actor_type, AuditActorType):
            raise InvalidAuditEventError(
                "actor_type must be an AuditActorType."
            )

        if (
            not self.source_component
            or not self.source_component.strip()
        ):
            raise InvalidAuditEventError(
                "source_component must not be empty."
            )

        if self.request_id is not None:
            if not self.request_id.strip():
                raise InvalidAuditEventError(
                    "request_id must not be empty when provided."
                )

        if self.thread_id is not None:
            if not self.thread_id.strip():
                raise InvalidAuditEventError(
                    "thread_id must not be empty when provided."
                )

        if self.actor_id is not None:
            if not self.actor_id.strip():
                raise InvalidAuditEventError(
                    "actor_id must not be empty when provided."
                )

        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise InvalidAuditEventError(
                "payload_json must contain valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidAuditEventError(
                "payload_json must represent a JSON object."
            )

        object.__setattr__(
            self,
            "payload_json",
            canonicalize_audit_payload(payload),
        )

    @classmethod
    def create(
        cls,
        *,
        event_type: AuditEventType,
        actor_type: AuditActorType,
        source_component: str,
        payload: Mapping[str, Any],
        request_id: str | None = None,
        thread_id: str | None = None,
        actor_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditEvent:
        occurred_at = (
            occurred_at
            if occurred_at is not None
            else datetime.now(timezone.utc)
        )

        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            occurred_at=occurred_at,
            actor_type=actor_type,
            source_component=source_component,
            payload_json=canonicalize_audit_payload(
                payload
            ),
            request_id=request_id,
            thread_id=thread_id,
            actor_id=actor_id,
        )

    @property
    def payload(self) -> dict[str, Any]:
        """
        Return a new dictionary so callers cannot mutate event state.
        """

        return json.loads(self.payload_json)