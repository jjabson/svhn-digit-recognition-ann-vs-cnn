from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from src.orchestration.audit_event import (
    AuditActorType,
    AuditEvent,
    AuditEventType,
    InvalidAuditEventError,
    canonicalize_audit_payload,
)


def test_create_audit_event():
    occurred_at = datetime(
        2026,
        10,
        23,
        12,
        0,
        tzinfo=timezone.utc,
    )

    event = AuditEvent.create(
        event_type=(
            AuditEventType.CONTROLLED_ACTION_REQUESTED
        ),
        actor_type=AuditActorType.MCP_CLIENT,
        source_component="mcp_server",
        request_id="request-123",
        thread_id="thread-456",
        payload={
            "action_name": "set_confidence_threshold",
            "arguments": {
                "confidence_threshold": 0.90,
            },
        },
        occurred_at=occurred_at,
    )

    assert event.event_id
    assert (
        event.event_type
        == AuditEventType.CONTROLLED_ACTION_REQUESTED
    )
    assert event.actor_type == AuditActorType.MCP_CLIENT
    assert event.source_component == "mcp_server"
    assert event.request_id == "request-123"
    assert event.thread_id == "thread-456"
    assert event.occurred_at == occurred_at

    assert event.payload == {
        "action_name": "set_confidence_threshold",
        "arguments": {
            "confidence_threshold": 0.90,
        },
    }


def test_audit_event_is_immutable():
    event = AuditEvent.create(
        event_type=AuditEventType.APPROVAL_REQUIRED,
        actor_type=AuditActorType.SYSTEM,
        source_component="approval_graph",
        request_id="request-123",
        payload={},
    )

    with pytest.raises(FrozenInstanceError):
        event.request_id = "different-request"


def test_payload_property_returns_defensive_copy():
    event = AuditEvent.create(
        event_type=AuditEventType.CONTROLLED_ACTION_REQUESTED,
        actor_type=AuditActorType.MCP_CLIENT,
        source_component="mcp_server",
        payload={
            "arguments": {
                "confidence_threshold": 0.90,
            }
        },
    )

    payload = event.payload

    payload["arguments"]["confidence_threshold"] = 0.50

    assert event.payload == {
        "arguments": {
            "confidence_threshold": 0.90,
        }
    }


def test_payload_is_canonicalized():
    first = canonicalize_audit_payload(
        {
            "z": 1,
            "a": 2,
        }
    )

    second = canonicalize_audit_payload(
        {
            "a": 2,
            "z": 1,
        }
    )

    assert first == second
    assert first == '{"a":2,"z":1}'


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": object()},
    ],
)
def test_audit_payload_rejects_non_json_values(
    payload,
):
    with pytest.raises(
        InvalidAuditEventError,
        match="JSON-compatible",
    ):
        canonicalize_audit_payload(payload)


def test_audit_event_requires_timezone_aware_timestamp():
    with pytest.raises(
        InvalidAuditEventError,
        match="timezone-aware",
    ):
        AuditEvent(
            event_id="event-123",
            event_type=AuditEventType.APPROVAL_REQUIRED,
            occurred_at=datetime(2026, 10, 23, 12, 0),
            actor_type=AuditActorType.SYSTEM,
            source_component="approval_graph",
            payload_json="{}",
        )


@pytest.mark.parametrize(
    "source_component",
    [
        "",
        "   ",
    ],
)
def test_audit_event_requires_source_component(
    source_component,
):
    with pytest.raises(
        InvalidAuditEventError,
        match="source_component",
    ):
        AuditEvent.create(
            event_type=AuditEventType.APPROVAL_REQUIRED,
            actor_type=AuditActorType.SYSTEM,
            source_component=source_component,
            payload={},
        )


def test_audit_event_rejects_non_object_payload_json():
    with pytest.raises(
        InvalidAuditEventError,
        match="JSON object",
    ):
        AuditEvent(
            event_id="event-123",
            event_type=AuditEventType.APPROVAL_REQUIRED,
            occurred_at=datetime.now(timezone.utc),
            actor_type=AuditActorType.SYSTEM,
            source_component="approval_graph",
            payload_json='["not", "an", "object"]',
        )


def test_optional_identifiers_cannot_be_blank():
    with pytest.raises(
        InvalidAuditEventError,
        match="request_id",
    ):
        AuditEvent.create(
            event_type=AuditEventType.APPROVAL_REQUIRED,
            actor_type=AuditActorType.SYSTEM,
            source_component="approval_graph",
            request_id="   ",
            payload={},
        )