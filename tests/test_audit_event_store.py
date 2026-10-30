from datetime import (
    datetime,
    timedelta,
    timezone,
)

import pytest

from config.audit_event_store import (
    AuditEventStore,
    DuplicateAuditEventError,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEvent,
    AuditEventType,
)


def create_event(
    *,
    event_type: AuditEventType = (
        AuditEventType.CONTROLLED_ACTION_REQUESTED
    ),
    occurred_at: datetime | None = None,
    request_id: str | None = "request-123",
    thread_id: str | None = "thread-123",
) -> AuditEvent:
    return AuditEvent.create(
        event_type=event_type,
        actor_type=AuditActorType.SYSTEM,
        source_component="test_component",
        request_id=request_id,
        thread_id=thread_id,
        payload={
            "action_name": (
                "set_confidence_threshold"
            ),
            "confidence_threshold": 0.90,
        },
        occurred_at=(
            occurred_at
            or datetime.now(timezone.utc)
        ),
    )


def test_store_creates_database(
    tmp_path,
):
    database_file = (
        tmp_path / "audit_events.db"
    )

    AuditEventStore(database_file)

    assert database_file.exists()


def test_append_and_get_event(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    event = create_event()

    store.append(event)

    loaded = store.get(event.event_id)

    assert loaded == event


def test_get_unknown_event_returns_none(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    assert store.get("missing-event") is None


def test_duplicate_event_id_is_rejected(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    event = create_event()

    store.append(event)

    with pytest.raises(
        DuplicateAuditEventError,
        match="already exists",
    ):
        store.append(event)


def test_list_all_returns_events_in_time_order(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    base_time = datetime(
        2026,
        10,
        26,
        12,
        0,
        tzinfo=timezone.utc,
    )

    later = create_event(
        occurred_at=(
            base_time
            + timedelta(minutes=5)
        ),
    )

    earlier = create_event(
        occurred_at=base_time,
    )

    store.append(later)
    store.append(earlier)

    events = store.list_all()

    assert events == [
        earlier,
        later,
    ]


def test_list_by_request_id(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    first = create_event(
        request_id="request-a",
    )

    second = create_event(
        event_type=(
            AuditEventType.APPROVAL_REQUIRED
        ),
        request_id="request-a",
    )

    unrelated = create_event(
        request_id="request-b",
    )

    store.append(first)
    store.append(second)
    store.append(unrelated)

    events = store.list_by_request_id(
        "request-a"
    )

    assert events == [
        first,
        second,
    ]


def test_list_by_thread_id(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    first = create_event(
        thread_id="thread-a",
    )

    second = create_event(
        event_type=(
            AuditEventType.APPROVAL_REQUIRED
        ),
        thread_id="thread-a",
    )

    unrelated = create_event(
        thread_id="thread-b",
    )

    store.append(first)
    store.append(second)
    store.append(unrelated)

    events = store.list_by_thread_id(
        "thread-a"
    )

    assert events == [
        first,
        second,
    ]


def test_event_survives_store_reconstruction(
    tmp_path,
):
    database_file = (
        tmp_path / "audit_events.db"
    )

    event = create_event()

    first_store = AuditEventStore(
        database_file
    )

    first_store.append(event)

    second_store = AuditEventStore(
        database_file
    )

    loaded = second_store.get(
        event.event_id
    )

    assert loaded == event


def test_payload_round_trip_is_preserved(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    event = AuditEvent.create(
        event_type=(
            AuditEventType.APPROVAL_GRANTED
        ),
        actor_type=(
            AuditActorType.TRUSTED_HUMAN
        ),
        actor_id=(
            "local_trusted_human_channel"
        ),
        source_component=(
            "trusted_approval_service"
        ),
        request_id="request-123",
        thread_id="thread-123",
        payload={
            "approved": True,
            "action_name": (
                "set_confidence_threshold"
            ),
            "arguments": {
                "confidence_threshold": 0.93,
            },
        },
    )

    store.append(event)

    loaded = store.get(event.event_id)

    assert loaded is not None

    assert loaded.payload == {
        "approved": True,
        "action_name": (
            "set_confidence_threshold"
        ),
        "arguments": {
            "confidence_threshold": 0.93,
        },
    }


def test_store_does_not_expose_update_or_delete(
    tmp_path,
):
    store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    assert not hasattr(
        store,
        "update",
    )

    assert not hasattr(
        store,
        "delete",
    )

    assert not hasattr(
        store,
        "replace",
    )