from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pytest

from config.approval_artifact_store import ApprovalArtifactStore
from src.orchestration.approval_artifact import ApprovalArtifact
from src.orchestration.trusted_approval import (
    ApprovalActionMismatchError,
    ApprovalAlreadyConsumedError,
    ApprovalArgumentsMismatchError,
    ApprovalArtifactMissingError,
    ApprovalExpiredError,
    ApprovalRejectedError,
    TrustedApprovalVerifier,
    UntrustedApprovalIssuerError,
    VerifiedApprovalDecision,
)


TRUSTED_ISSUER = "local_trusted_human_channel"


@dataclass(frozen=True)
class ApprovalRequestStub:
    request_id: str
    action_name: str
    arguments: Mapping[str, Any]


def create_request(
    *,
    request_id: str = "request-123",
    threshold: float = 0.95,
) -> ApprovalRequestStub:
    return ApprovalRequestStub(
        request_id=request_id,
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": threshold,
        },
    )


def create_artifact(
    request: ApprovalRequestStub,
    *,
    approved: bool = True,
    issued_by: str = TRUSTED_ISSUER,
    issued_at: datetime | None = None,
    ttl_seconds: int = 900,
) -> ApprovalArtifact:
    return ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name=request.action_name,
        arguments=request.arguments,
        approved=approved,
        issued_by=issued_by,
        issued_at=issued_at,
        ttl_seconds=ttl_seconds,
    )


def create_verifier(
    store: ApprovalArtifactStore,
) -> TrustedApprovalVerifier:
    return TrustedApprovalVerifier(
        store,
        trusted_issuers={
            TRUSTED_ISSUER,
        },
    )


def test_verifier_requires_at_least_one_trusted_issuer(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    with pytest.raises(
        ValueError,
        match="trusted approval issuer",
    ):
        TrustedApprovalVerifier(
            store,
            trusted_issuers=set(),
        )


def test_valid_trusted_artifact_creates_verified_decision(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()
    artifact = create_artifact(request)

    store.save(artifact)

    verified_at = datetime.now(timezone.utc)

    decision = verifier.verify(
        request,
        now=verified_at,
    )

    assert isinstance(
        decision,
        VerifiedApprovalDecision,
    )
    assert decision.artifact_id == artifact.artifact_id
    assert decision.request_id == request.request_id
    assert decision.action_name == request.action_name
    assert decision.arguments == request.arguments
    assert decision.issued_by == TRUSTED_ISSUER
    assert decision.nonce == artifact.nonce
    assert decision.verified_at == verified_at


def test_missing_artifact_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()

    with pytest.raises(
        ApprovalArtifactMissingError,
        match="No trusted approval artifact",
    ):
        verifier.verify(request)


def test_action_mismatch_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()

    artifact = ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name="different_action",
        arguments=request.arguments,
        approved=True,
        issued_by=TRUSTED_ISSUER,
    )

    store.save(artifact)

    with pytest.raises(
        ApprovalActionMismatchError,
        match="action does not match",
    ):
        verifier.verify(request)


def test_argument_mismatch_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request(
        threshold=0.95,
    )

    artifact = ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name=request.action_name,
        arguments={
            "confidence_threshold": 0.99,
        },
        approved=True,
        issued_by=TRUSTED_ISSUER,
    )

    store.save(artifact)

    with pytest.raises(
        ApprovalArgumentsMismatchError,
        match="arguments do not match",
    ):
        verifier.verify(request)


def test_argument_order_does_not_cause_false_mismatch(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = ApprovalRequestStub(
        request_id="request-123",
        action_name="example_action",
        arguments={
            "b": 2,
            "a": 1,
        },
    )

    artifact = ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name=request.action_name,
        arguments={
            "a": 1,
            "b": 2,
        },
        approved=True,
        issued_by=TRUSTED_ISSUER,
    )

    store.save(artifact)

    decision = verifier.verify(request)

    assert decision.arguments == {
        "a": 1,
        "b": 2,
    }


def test_untrusted_issuer_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()

    artifact = create_artifact(
        request,
        issued_by="mcp_agent",
    )

    store.save(artifact)

    with pytest.raises(
        UntrustedApprovalIssuerError,
        match="not trusted",
    ):
        verifier.verify(request)


def test_rejected_artifact_cannot_authorize_execution(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()

    artifact = create_artifact(
        request,
        approved=False,
    )

    store.save(artifact)

    with pytest.raises(
        ApprovalRejectedError,
        match="rejected",
    ):
        verifier.verify(request)


def test_expired_artifact_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    issued_at = datetime(
        2026,
        10,
        14,
        12,
        0,
        tzinfo=timezone.utc,
    )

    request = create_request()

    artifact = create_artifact(
        request,
        issued_at=issued_at,
        ttl_seconds=60,
    )

    store.save(artifact)

    with pytest.raises(
        ApprovalExpiredError,
        match="expired",
    ):
        verifier.verify(
            request,
            now=issued_at + timedelta(seconds=60),
        )


def test_artifact_valid_immediately_before_expiration(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    issued_at = datetime(
        2026,
        10,
        14,
        12,
        0,
        tzinfo=timezone.utc,
    )

    request = create_request()

    artifact = create_artifact(
        request,
        issued_at=issued_at,
        ttl_seconds=60,
    )

    store.save(artifact)

    decision = verifier.verify(
        request,
        now=issued_at + timedelta(seconds=59),
    )

    assert decision.artifact_id == artifact.artifact_id


def test_consumed_artifact_is_rejected_as_replay(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()
    artifact = create_artifact(request)

    store.save(artifact)

    store.consume(
        artifact.artifact_id,
        consumed_at=datetime.now(timezone.utc),
    )

    with pytest.raises(
        ApprovalAlreadyConsumedError,
        match="already been consumed",
    ):
        verifier.verify(request)


def test_verification_requires_timezone_aware_time(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()
    artifact = create_artifact(request)

    store.save(artifact)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        verifier.verify(
            request,
            now=datetime.now(),
        )


def test_verified_decision_arguments_are_independent(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )
    verifier = create_verifier(store)

    request = create_request()
    artifact = create_artifact(request)

    store.save(artifact)

    decision = verifier.verify(request)

    arguments = decision.arguments
    arguments["confidence_threshold"] = 0.50

    assert decision.arguments == {
        "confidence_threshold": 0.95,
    }