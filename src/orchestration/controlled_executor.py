from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.orchestration.action_policy import ActionPolicy, ActionRisk
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    resolve_approval_status,
)


class ApprovalRequiredError(RuntimeError):
    pass


class ActionProhibitedError(RuntimeError):
    pass


class ApprovalMismatchError(RuntimeError):
    pass


class ApprovalDeniedError(RuntimeError):
    pass


class ControlledActionExecutor:
    def __init__(self, policy: ActionPolicy) -> None:
        self._policy = policy

    def execute(
        self,
        *,
        action_name: str,
        arguments: Mapping[str, Any],
        action: Callable[..., Any],
        approval_request: ApprovalRequest | None = None,
        approval_decision: ApprovalDecision | None = None,
    ) -> Any:
        risk = self._policy.classify(action_name)

        if risk is ActionRisk.PROHIBITED:
            raise ActionProhibitedError(
                f"Action is prohibited: {action_name}"
            )

        if risk is ActionRisk.READ_ONLY:
            return action(**dict(arguments))

        if approval_request is None:
            raise ApprovalRequiredError(
                f"Approval required for action: {action_name}"
            )

        if not approval_request.matches_action(
            action_name,
            arguments,
        ):
            raise ApprovalMismatchError(
                "ApprovalRequest does not match the requested "
                "action and arguments"
            )

        if approval_decision is None:
            raise ApprovalRequiredError(
                f"Approval decision required for action: {action_name}"
            )

        try:
            status = resolve_approval_status(
                approval_request,
                approval_decision,
            )
        except ValueError as exc:
            raise ApprovalMismatchError(str(exc)) from exc

        if status is not ApprovalStatus.APPROVED:
            raise ApprovalDeniedError(
                f"Action was not approved: {status.value}"
            )

        return action(**dict(arguments))