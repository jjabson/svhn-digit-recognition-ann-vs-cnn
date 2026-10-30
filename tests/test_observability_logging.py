from __future__ import annotations

from datetime import datetime, timezone
import json
import logging

import pytest

from src.observability.logging import (
    LogStatus,
    OperationalLogError,
    OperationalLogEvent,
    SensitiveLogFieldError,
    log_operational_event,
)


def test_log_event_requires_event_name():
    with pytest.raises(
        OperationalLogError,
        match="event must not be empty",
    ):
        OperationalLogEvent(
            event="",
            component="mcp_server",
        )


def test_log_event_requires_component_name():
    with pytest.raises(
        OperationalLogError,
        match="component must not be empty",
    ):
        OperationalLogEvent(
            event="server.started",
            component="",
        )


def test_log_event_rejects_negative_duration():
    with pytest.raises(
        OperationalLogError,
        match="duration_ms must not be negative",
    ):
        OperationalLogEvent(
            event="inference.completed",
            component="inference",
            duration_ms=-1.0,
        )


def test_log_event_requires_timezone_aware_timestamp():
    with pytest.raises(
        OperationalLogError,
        match="timezone-aware",
    ):
        OperationalLogEvent(
            event="server.started",
            component="mcp_server",
            occurred_at=datetime(
                2026,
                10,
                28,
                12,
                0,
            ),
        )


def test_log_event_rejects_sensitive_top_level_field():
    with pytest.raises(
        SensitiveLogFieldError,
        match="bearer_token",
    ):
        OperationalLogEvent(
            event="request.received",
            component="mcp_server",
            fields={
                "bearer_token": "do-not-log",
            },
        )


def test_log_event_rejects_sensitive_nested_field():
    with pytest.raises(
        SensitiveLogFieldError,
        match="secret",
    ):
        OperationalLogEvent(
            event="request.received",
            component="mcp_server",
            fields={
                "metadata": {
                    "secret": "do-not-log",
                },
            },
        )


def test_log_event_rejects_non_json_serializable_field():
    with pytest.raises(
        OperationalLogError,
        match="JSON-serializable",
    ):
        OperationalLogEvent(
            event="request.received",
            component="mcp_server",
            fields={
                "invalid": object(),
            },
        )


def test_log_event_payload_contains_correlation_fields():
    checked_at = datetime(
        2026,
        10,
        28,
        12,
        0,
        tzinfo=timezone.utc,
    )

    event = OperationalLogEvent(
        event="controlled_action.completed",
        component="approval_graph",
        status=LogStatus.SUCCEEDED,
        request_id="request-123",
        thread_id="thread-456",
        runtime_instance_id="runtime-789",
        duration_ms=12.5,
        occurred_at=checked_at,
        fields={
            "action_name": (
                "set_confidence_threshold"
            ),
        },
    )

    payload = event.to_dict()

    assert payload == {
        "event": "controlled_action.completed",
        "component": "approval_graph",
        "occurred_at": checked_at.isoformat(),
        "status": "succeeded",
        "request_id": "request-123",
        "thread_id": "thread-456",
        "runtime_instance_id": "runtime-789",
        "duration_ms": 12.5,
        "action_name": (
            "set_confidence_threshold"
        ),
    }


def test_log_event_defensively_copies_fields():
    fields = {
        "metadata": {
            "model_name": "cnn",
        }
    }

    event = OperationalLogEvent(
        event="inference.completed",
        component="inference",
        fields=fields,
    )

    fields["metadata"][
        "model_name"
    ] = "changed"

    assert event.fields == {
        "metadata": {
            "model_name": "cnn",
        }
    }


def test_log_operational_event_emits_json(
    caplog,
):
    logger = logging.getLogger(
        "test.observability"
    )

    event = OperationalLogEvent(
        event="runtime.started",
        component="mcp_server",
        status=LogStatus.STARTED,
        fields={
            "transport": "stdio",
        },
    )

    with caplog.at_level(
        logging.INFO,
        logger="test.observability",
    ):
        log_operational_event(
            logger,
            event,
        )

    assert len(caplog.records) == 1

    payload = json.loads(
        caplog.records[0].message
    )

    assert payload[
        "event"
    ] == "runtime.started"

    assert payload[
        "component"
    ] == "mcp_server"

    assert payload[
        "status"
    ] == "started"

    assert payload[
        "transport"
    ] == "stdio"


def test_default_timestamp_is_timezone_aware():
    event = OperationalLogEvent(
        event="runtime.started",
        component="mcp_server",
    )

    assert event.occurred_at.tzinfo is not None