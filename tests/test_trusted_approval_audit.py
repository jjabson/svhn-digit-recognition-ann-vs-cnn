import pytest

from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.audit_event_store import (
    AuditEventStore,
)
from src.orchestration.approval import (
    ApprovalRequest,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEventType,
)
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
    TrustedApprovalService,
)


def _build_service(tmp_path):
    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    audit_store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    service = TrustedApprovalService(
        artifact_store,
        audit_store=audit_store,
    )

    return (
        service,
        artifact_store,
        audit_store,
    )


def _request(
    threshold: float = 0.95,
) -> ApprovalRequest:
    return ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": threshold,
        },
        reason="Trusted approval audit test",
    )


def test_approved_decision_records_granted_event(
    tmp_path,
):
    service, _, audit_store = (
        _build_service(tmp_path)
    )

    request = _request(0.95)

    artifact = service.record_decision(
        request,
        approved=True,
        thread_id="thread-approved",
    )

    events = audit_store.list_all()

    assert len(events) == 1

    event = events[0]

    assert (
        event.event_type
        == AuditEventType.APPROVAL_GRANTED
    )

    assert (
        event.actor_type
        == AuditActorType.TRUSTED_HUMAN
    )

    assert (
        event.actor_id
        == DEFAULT_TRUSTED_APPROVAL_ISSUER
    )

    assert (
        event.source_component
        == "trusted_approval_service"
    )

    assert (
        event.request_id
        == request.request_id
    )

    assert (
        event.thread_id
        == "thread-approved"
    )

    assert (
        event.payload["artifact_id"]
        == artifact.artifact_id
    )

    assert event.payload["approved"] is True


def test_rejected_decision_records_rejected_event(
    tmp_path,
):
    service, _, audit_store = (
        _build_service(tmp_path)
    )

    request = _request(0.92)

    artifact = service.record_decision(
        request,
        approved=False,
        thread_id="thread-rejected",
    )

    events = audit_store.list_all()

    assert len(events) == 1

    event = events[0]

    assert (
        event.event_type
        == AuditEventType.APPROVAL_REJECTED
    )

    assert (
        event.actor_type
        == AuditActorType.TRUSTED_HUMAN
    )

    assert (
        event.request_id
        == request.request_id
    )

    assert (
        event.thread_id
        == "thread-rejected"
    )

    assert (
        event.payload["artifact_id"]
        == artifact.artifact_id
    )

    assert event.payload["approved"] is False


def test_trusted_decision_payload_records_action_facts(
    tmp_path,
):
    service, _, audit_store = (
        _build_service(tmp_path)
    )

    request = _request(0.93)

    service.record_decision(
        request,
        approved=True,
        thread_id="thread-payload",
    )

    event = audit_store.list_all()[0]

    assert event.payload[
        "action_name"
    ] == "set_confidence_threshold"

    assert event.payload[
        "arguments"
    ] == {
        "confidence_threshold": 0.93,
    }

    assert (
        event.payload["issued_by"]
        == DEFAULT_TRUSTED_APPROVAL_ISSUER
    )


def test_failed_artifact_persistence_does_not_append_second_audit_event(
    tmp_path,
):
    service, _, audit_store = (
        _build_service(tmp_path)
    )

    request = _request(0.95)

    service.record_decision(
        request,
        approved=True,
        thread_id="thread-ordering",
    )

    assert len(
        audit_store.list_all()
    ) == 1

    with pytest.raises(Exception):
        service.record_decision(
            request,
            approved=True,
            thread_id="thread-ordering",
        )

    assert len(
        audit_store.list_all()
    ) == 1