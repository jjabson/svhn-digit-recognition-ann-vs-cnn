from datetime import datetime, timedelta, timezone

import pytest

from src.orchestration.approval_artifact import (
    ApprovalArtifact,
    InvalidApprovalArtifactError,
    canonicalize_arguments,
)


def test_canonicalize_arguments_sorts_keys() -> None:
    first = canonicalize_arguments(
        {
            "b": 2,
            "a": 1,
        }
    )

    second = canonicalize_arguments(
        {
            "a": 1,
            "b": 2,
        }
    )

    assert first == second
    assert first == '{"a":1,"b":2}'


def test_canonicalize_arguments_rejects_non_json_values() -> None:
    with pytest.raises(
        InvalidApprovalArtifactError,
        match="JSON-compatible",
    ):
        canonicalize_arguments(
            {
                "invalid": object(),
            }
        )


def test_issue_creates_immutable_approval_artifact() -> None:
    issued_at = datetime(
        2026,
        10,
        13,
        12,
        0,
        tzinfo=timezone.utc,
    )

    artifact = ApprovalArtifact.issue(
        request_id="request-123",
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        approved=True,
        issued_by="local_trusted_human_channel",
        ttl_seconds=900,
        issued_at=issued_at,
    )

    assert artifact.artifact_id
    assert artifact.request_id == "request-123"
    assert artifact.action_name == "set_confidence_threshold"
    assert artifact.arguments == {
        "confidence_threshold": 0.95,
    }
    assert artifact.approved is True
    assert artifact.issued_at == issued_at
    assert artifact.expires_at == (
        issued_at + timedelta(seconds=900)
    )
    assert artifact.issued_by == "local_trusted_human_channel"
    assert artifact.nonce
    assert artifact.consumed_at is None
    assert artifact.is_consumed is False


def test_arguments_property_returns_independent_dictionary() -> None:
    artifact = ApprovalArtifact.issue(
        request_id="request-123",
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        approved=True,
        issued_by="local_trusted_human_channel",
    )

    arguments = artifact.arguments
    arguments["confidence_threshold"] = 0.50

    assert artifact.arguments == {
        "confidence_threshold": 0.95,
    }


def test_argument_order_does_not_change_binding() -> None:
    first = ApprovalArtifact.issue(
        request_id="request-1",
        action_name="example_action",
        arguments={
            "b": 2,
            "a": 1,
        },
        approved=True,
        issued_by="local_trusted_human_channel",
    )

    second = ApprovalArtifact.issue(
        request_id="request-2",
        action_name="example_action",
        arguments={
            "a": 1,
            "b": 2,
        },
        approved=True,
        issued_by="local_trusted_human_channel",
    )

    assert first.arguments_json == second.arguments_json


def test_rejected_artifact_is_supported() -> None:
    artifact = ApprovalArtifact.issue(
        request_id="request-123",
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        approved=False,
        issued_by="local_trusted_human_channel",
    )

    assert artifact.approved is False


def test_artifact_reports_expired() -> None:
    issued_at = datetime(
        2026,
        10,
        13,
        12,
        0,
        tzinfo=timezone.utc,
    )

    artifact = ApprovalArtifact.issue(
        request_id="request-123",
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        approved=True,
        issued_by="local_trusted_human_channel",
        ttl_seconds=60,
        issued_at=issued_at,
    )

    assert artifact.is_expired(
        now=issued_at + timedelta(seconds=59),
    ) is False

    assert artifact.is_expired(
        now=issued_at + timedelta(seconds=60),
    ) is True


def test_issue_rejects_non_positive_ttl() -> None:
    with pytest.raises(
        InvalidApprovalArtifactError,
        match="ttl_seconds",
    ):
        ApprovalArtifact.issue(
            request_id="request-123",
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            approved=True,
            issued_by="local_trusted_human_channel",
            ttl_seconds=0,
        )


def test_naive_issued_at_is_rejected() -> None:
    naive_time = datetime(
        2026,
        10,
        13,
        12,
        0,
    )

    with pytest.raises(
        InvalidApprovalArtifactError,
        match="timezone-aware",
    ):
        ApprovalArtifact.issue(
            request_id="request-123",
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            approved=True,
            issued_by="local_trusted_human_channel",
            issued_at=naive_time,
        )


def test_expiration_must_follow_issue_time() -> None:
    issued_at = datetime.now(timezone.utc)

    with pytest.raises(
        InvalidApprovalArtifactError,
        match="expires_at",
    ):
        ApprovalArtifact(
            artifact_id="artifact-123",
            request_id="request-123",
            action_name="set_confidence_threshold",
            arguments_json='{"confidence_threshold":0.95}',
            approved=True,
            issued_at=issued_at,
            expires_at=issued_at,
            issued_by="local_trusted_human_channel",
            nonce="nonce-123",
        )