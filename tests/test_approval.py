from datetime import datetime, timezone

import pytest

from src.orchestration.action_policy import ActionRisk
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    resolve_approval_status,
)


def test_new_approval_request_is_pending():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Use a stricter confidence threshold",
    )

    assert request.status is ApprovalStatus.PENDING


def test_request_accepts_only_controlled_actions():
    with pytest.raises(
        ValueError,
        match="CONTROLLED actions",
    ):
        ApprovalRequest(
            action_name="predict_digit",
            arguments={},
            reason="Should not require approval",
            risk=ActionRisk.READ_ONLY,
        )


def test_request_rejects_prohibited_action_risk():
    with pytest.raises(
        ValueError,
        match="CONTROLLED actions",
    ):
        ApprovalRequest(
            action_name="delete_evaluation_history",
            arguments={},
            reason="Should never be approved",
            risk=ActionRisk.PROHIBITED,
        )


def test_request_arguments_are_copied():
    arguments = {"confidence_threshold": 0.95}

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments=arguments,
        reason="Change threshold",
    )

    arguments["confidence_threshold"] = 0.10

    assert request.arguments["confidence_threshold"] == 0.95


def test_request_arguments_are_read_only():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    with pytest.raises(TypeError):
        request.arguments["confidence_threshold"] = 0.10


def test_request_matches_exact_action_and_arguments():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    assert request.matches_action(
        "set_confidence_threshold",
        {"confidence_threshold": 0.95},
    )


def test_request_does_not_match_different_arguments():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    assert not request.matches_action(
        "set_confidence_threshold",
        {"confidence_threshold": 0.50},
    )


def test_request_does_not_match_different_action():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    assert not request.matches_action(
        "change_active_model",
        {"confidence_threshold": 0.95},
    )


def test_approved_decision_permits_execution():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    assert decision.permits_execution is True


def test_rejected_decision_does_not_permit_execution():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.REJECTED,
        decided_by="human",
    )

    assert decision.permits_execution is False


@pytest.mark.parametrize(
    "status",
    [
        ApprovalStatus.REJECTED,
        ApprovalStatus.EXPIRED,
        ApprovalStatus.CANCELLED,
    ],
)
def test_non_approved_terminal_states_do_not_permit_execution(status):
    decision = ApprovalDecision(
        request_id="request-123",
        status=status,
        decided_by="human",
    )

    assert decision.permits_execution is False


def test_pending_is_not_valid_as_a_decision():
    with pytest.raises(
        ValueError,
        match="terminal approval status",
    ):
        ApprovalDecision(
            request_id="request-123",
            status=ApprovalStatus.PENDING,
            decided_by="human",
        )


def test_no_decision_resolves_to_pending():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    assert (
        resolve_approval_status(request, None)
        is ApprovalStatus.PENDING
    )


def test_approved_decision_resolves_to_approved():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    assert (
        resolve_approval_status(request, decision)
        is ApprovalStatus.APPROVED
    )


def test_decision_for_different_request_is_rejected():
    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Change threshold",
    )

    decision = ApprovalDecision(
        request_id="different-request",
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    with pytest.raises(
        ValueError,
        match="does not belong",
    ):
        resolve_approval_status(request, decision)


def test_request_requires_timezone_aware_timestamp():
    naive_time = datetime(2026, 10, 5, 12, 0, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        ApprovalRequest(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            reason="Change threshold",
            created_at=naive_time,
        )


def test_decision_requires_timezone_aware_timestamp():
    naive_time = datetime(2026, 10, 5, 12, 0, 0)

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        ApprovalDecision(
            request_id="request-123",
            status=ApprovalStatus.APPROVED,
            decided_by="human",
            decided_at=naive_time,
        )