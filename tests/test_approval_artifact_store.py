from datetime import datetime, timezone

import pytest

from config.approval_artifact_store import (
    ApprovalArtifactAlreadyConsumedError,
    ApprovalArtifactAlreadyExistsError,
    ApprovalArtifactNotFoundError,
    ApprovalArtifactStore,
)
from src.orchestration.approval_artifact import ApprovalArtifact


def create_artifact(
    *,
    request_id: str = "request-123",
    approved: bool = True,
) -> ApprovalArtifact:
    return ApprovalArtifact.issue(
        request_id=request_id,
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        approved=approved,
        issued_by="local_trusted_human_channel",
    )


def test_empty_store_returns_none(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    assert store.get_by_request_id("missing") is None
    assert store.get_by_artifact_id("missing") is None


def test_save_and_load_by_artifact_id(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()

    store.save(artifact)

    loaded = store.get_by_artifact_id(
        artifact.artifact_id
    )

    assert loaded == artifact


def test_save_and_load_by_request_id(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()

    store.save(artifact)

    loaded = store.get_by_request_id(
        artifact.request_id
    )

    assert loaded == artifact


def test_rejected_artifact_is_persisted(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact(
        approved=False,
    )

    store.save(artifact)

    loaded = store.get_by_request_id(
        artifact.request_id
    )

    assert loaded is not None
    assert loaded.approved is False


def test_duplicate_artifact_is_rejected(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()

    store.save(artifact)

    with pytest.raises(
        ApprovalArtifactAlreadyExistsError
    ):
        store.save(artifact)


def test_only_one_artifact_allowed_per_request(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    first = create_artifact(
        request_id="request-123",
    )

    second = create_artifact(
        request_id="request-123",
    )

    store.save(first)

    with pytest.raises(
        ApprovalArtifactAlreadyExistsError
    ):
        store.save(second)


def test_consume_marks_artifact_consumed(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()
    store.save(artifact)

    consumed_at = datetime.now(timezone.utc)

    consumed = store.consume(
        artifact.artifact_id,
        consumed_at=consumed_at,
    )

    assert consumed.is_consumed is True
    assert consumed.consumed_at == consumed_at


def test_consumed_artifact_survives_store_reconstruction(
    tmp_path,
) -> None:
    database_file = tmp_path / "approval_artifacts.db"

    first_store = ApprovalArtifactStore(database_file)

    artifact = create_artifact()
    first_store.save(artifact)

    consumed_at = datetime.now(timezone.utc)

    first_store.consume(
        artifact.artifact_id,
        consumed_at=consumed_at,
    )

    second_store = ApprovalArtifactStore(database_file)

    recovered = second_store.get_by_artifact_id(
        artifact.artifact_id
    )

    assert recovered is not None
    assert recovered.is_consumed is True
    assert recovered.consumed_at == consumed_at


def test_second_consume_is_rejected_as_replay(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()
    store.save(artifact)

    store.consume(
        artifact.artifact_id,
        consumed_at=datetime.now(timezone.utc),
    )

    with pytest.raises(
        ApprovalArtifactAlreadyConsumedError
    ):
        store.consume(
            artifact.artifact_id,
            consumed_at=datetime.now(timezone.utc),
        )


def test_consume_unknown_artifact_fails(tmp_path) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    with pytest.raises(
        ApprovalArtifactNotFoundError
    ):
        store.consume(
            "unknown-artifact",
            consumed_at=datetime.now(timezone.utc),
        )


def test_consume_requires_timezone_aware_timestamp(
    tmp_path,
) -> None:
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    artifact = create_artifact()
    store.save(artifact)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        store.consume(
            artifact.artifact_id,
            consumed_at=datetime.now(),
        )