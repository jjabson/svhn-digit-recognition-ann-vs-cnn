import pytest

from src.orchestration.action_policy import ActionPolicy
from dataclasses import replace
from datetime import datetime, timezone

from config.approval_artifact_store import ApprovalArtifactStore
from src.orchestration.approval_artifact import ApprovalArtifact
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
)
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)
from src.orchestration.controlled_executor import (
    ActionProhibitedError,
    ApprovalDeniedError,
    ApprovalMismatchError,
    ApprovalReplayError,
    ApprovalRequiredError,
    ControlledActionExecutor,
    TrustedApprovalMismatchError,
    TrustedApprovalRequiredError,
)
from src.orchestration.trusted_approval import (
    ApprovalAlreadyConsumedError,
    TrustedApprovalVerifier,
)


def test_read_only_action_executes_without_approval():
    executor = ControlledActionExecutor(ActionPolicy())

    def get_digit_metrics(digit: int):
        return {"digit": digit}

    result = executor.execute(
        action_name="get_digit_metrics",
        arguments={"digit": 7},
        action=get_digit_metrics,
    )

    assert result == {"digit": 7}


def test_controlled_action_requires_approval_request():
    executor = ControlledActionExecutor(ActionPolicy())

    def set_confidence_threshold(confidence_threshold: float):
        return confidence_threshold

    with pytest.raises(
        ApprovalRequiredError,
        match="Approval required",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=set_confidence_threshold,
        )


def test_controlled_action_requires_approval_decision():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    def set_confidence_threshold(confidence_threshold: float):
        return confidence_threshold

    with pytest.raises(
        ApprovalRequiredError,
        match="Approval decision required",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=set_confidence_threshold,
            approval_request=request,
        )


def test_controlled_action_executes_when_approved():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    def set_confidence_threshold(confidence_threshold: float):
        return confidence_threshold

    result = executor.execute(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        action=set_confidence_threshold,
        approval_request=request,
        approval_decision=decision,
    )

    assert result == 0.95


def test_rejected_action_does_not_execute():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.REJECTED,
        decided_by="human",
    )

    called = False

    def set_confidence_threshold(confidence_threshold: float):
        nonlocal called
        called = True
        return confidence_threshold

    with pytest.raises(
        ApprovalDeniedError,
        match="rejected",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=set_confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )

    assert called is False


def test_expired_action_does_not_execute():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.EXPIRED,
        decided_by="system",
    )

    with pytest.raises(
        ApprovalDeniedError,
        match="expired",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )


def test_cancelled_action_does_not_execute():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.CANCELLED,
        decided_by="human",
    )

    with pytest.raises(
        ApprovalDeniedError,
        match="cancelled",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )


def test_mismatched_arguments_do_not_execute():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    with pytest.raises(
        ApprovalMismatchError,
        match="does not match",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.50},
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )


def test_mismatched_request_id_does_not_execute():
    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase confidence threshold",
    )

    decision = ApprovalDecision(
        request_id="different-request-id",
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    with pytest.raises(
        ApprovalMismatchError,
        match="does not belong",
    ):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )


def test_prohibited_action_never_executes():
    executor = ControlledActionExecutor(ActionPolicy())

    called = False

    def delete_evaluation_history():
        nonlocal called
        called = True

    with pytest.raises(
        ActionProhibitedError,
        match="prohibited",
    ):
        executor.execute(
            action_name="delete_evaluation_history",
            arguments={},
            action=delete_evaluation_history,
        )

    assert called is False


def test_unknown_action_fails_closed():
    executor = ControlledActionExecutor(ActionPolicy())

    with pytest.raises(
        ValueError,
        match="Unknown action",
    ):
        executor.execute(
            action_name="unknown_action",
            arguments={},
            action=lambda: None,
        )


TRUSTED_ISSUER = "local_trusted_human_channel"


def create_trusted_executor(tmp_path):
    store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    verifier = TrustedApprovalVerifier(
        store,
        trusted_issuers={
            TRUSTED_ISSUER,
        },
    )

    executor = ControlledActionExecutor(
        ActionPolicy(),
        trusted_approval_verifier=verifier,
        approval_artifact_store=store,
    )

    return executor, store, verifier


def create_trusted_request_and_artifact(
    store,
    *,
    threshold: float = 0.95,
):
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": threshold,
        },
        reason="Increase confidence threshold",
    )

    artifact = ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name=request.action_name,
        arguments=request.arguments,
        approved=True,
        issued_by=TRUSTED_ISSUER,
    )

    store.save(artifact)

    return request, artifact


def test_trusted_controlled_action_executes_once(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, artifact = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    result = executor.execute_trusted(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        action=lambda confidence_threshold: confidence_threshold,
        approval_request=request,
        verified_approval=verified,
    )

    assert result == 0.95

    persisted = store.get_by_artifact_id(
        artifact.artifact_id
    )

    assert persisted is not None
    assert persisted.is_consumed is True


def test_trusted_controlled_action_requires_verified_decision(
    tmp_path,
):
    executor, store, _ = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    with pytest.raises(
        TrustedApprovalRequiredError,
        match="Verified trusted approval required",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
        )


def test_trusted_executor_rejects_forged_artifact_id(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    forged = replace(
        verified,
        artifact_id="forged-artifact-id",
    )

    with pytest.raises(
        TrustedApprovalMismatchError,
        match="artifact_id",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=forged,
        )


def test_trusted_executor_rejects_forged_nonce(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    forged = replace(
        verified,
        nonce="forged-nonce",
    )

    with pytest.raises(
        TrustedApprovalMismatchError,
        match="nonce",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=forged,
        )


def test_trusted_executor_rejects_forged_arguments(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    forged = replace(
        verified,
        arguments_json='{"confidence_threshold":0.50}',
    )

    with pytest.raises(
        TrustedApprovalMismatchError,
        match="arguments",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=forged,
        )


def test_trusted_approval_cannot_be_replayed(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    executor.execute_trusted(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        action=lambda confidence_threshold: confidence_threshold,
        approval_request=request,
        verified_approval=verified,
    )

    with pytest.raises(
            ApprovalAlreadyConsumedError,
            match="already been consumed",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=verified,
        )

def test_failed_action_still_consumes_approval(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, artifact = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    def failing_action(confidence_threshold: float):
        raise RuntimeError("simulated action failure")

    with pytest.raises(
        RuntimeError,
        match="simulated action failure",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=failing_action,
            approval_request=request,
            verified_approval=verified,
        )

    persisted = store.get_by_artifact_id(
        artifact.artifact_id
    )

    assert persisted is not None
    assert persisted.is_consumed is True

    with pytest.raises(
        ApprovalAlreadyConsumedError,
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=verified,
        )


def test_trusted_read_only_action_needs_no_artifact(
    tmp_path,
):
    executor, _, _ = create_trusted_executor(
        tmp_path
    )

    result = executor.execute_trusted(
        action_name="get_digit_metrics",
        arguments={
            "digit": 7,
        },
        action=lambda digit: {
            "digit": digit,
        },
    )

    assert result == {
        "digit": 7,
    }


def test_trusted_prohibited_action_never_executes(
    tmp_path,
):
    executor, _, _ = create_trusted_executor(
        tmp_path
    )

    called = False

    def delete_evaluation_history():
        nonlocal called
        called = True

    with pytest.raises(
        ActionProhibitedError,
        match="prohibited",
    ):
        executor.execute_trusted(
            action_name="delete_evaluation_history",
            arguments={},
            action=delete_evaluation_history,
        )

    assert called is False


def test_trusted_execution_requires_timezone_aware_time(
    tmp_path,
):
    executor, store, verifier = create_trusted_executor(
        tmp_path
    )

    request, _ = create_trusted_request_and_artifact(
        store
    )

    verified = verifier.verify(request)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=lambda confidence_threshold: confidence_threshold,
            approval_request=request,
            verified_approval=verified,
            now=datetime.now(),
        )