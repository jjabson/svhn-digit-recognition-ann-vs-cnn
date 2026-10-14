import pytest

from config.approval_artifact_store import (
    ApprovalArtifactAlreadyExistsError,
    ApprovalArtifactStore,
)
from src.orchestration.approval import ApprovalRequest
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
    TrustedApprovalService,
)


def create_request() -> ApprovalRequest:
    return ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        reason="Increase threshold",
    )


def test_trusted_service_records_approved_artifact(
    tmp_path,
):
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    service = TrustedApprovalService(store)

    request = create_request()

    artifact = service.record_decision(
        request,
        approved=True,
    )

    assert artifact.request_id == request.request_id
    assert artifact.action_name == request.action_name
    assert artifact.arguments == request.arguments
    assert artifact.approved is True
    assert (
        artifact.issued_by
        == DEFAULT_TRUSTED_APPROVAL_ISSUER
    )

    persisted = store.get_by_request_id(
        request.request_id
    )

    assert persisted == artifact


def test_trusted_service_records_rejection(
    tmp_path,
):
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    service = TrustedApprovalService(store)

    request = create_request()

    artifact = service.record_decision(
        request,
        approved=False,
    )

    assert artifact.approved is False


def test_trusted_decision_cannot_be_overwritten(
    tmp_path,
):
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    service = TrustedApprovalService(store)

    request = create_request()

    service.record_decision(
        request,
        approved=True,
    )

    with pytest.raises(
        ApprovalArtifactAlreadyExistsError
    ):
        service.record_decision(
            request,
            approved=False,
        )