import pytest

from src.orchestration.action_policy import ActionPolicy
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)
from src.orchestration.controlled_executor import (
    ActionProhibitedError,
    ApprovalDeniedError,
    ApprovalMismatchError,
    ApprovalRequiredError,
    ControlledActionExecutor,
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