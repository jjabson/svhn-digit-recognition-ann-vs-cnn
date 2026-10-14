from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from config.approval_artifact_store import (
    ApprovalArtifactAlreadyConsumedError,
    ApprovalArtifactStore,
)
from src.orchestration.action_policy import ActionPolicy, ActionRisk
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    resolve_approval_status,
)
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
    VerifiedApprovalDecision,
)


class ApprovalRequiredError(RuntimeError):
    pass


class ActionProhibitedError(RuntimeError):
    pass


class ApprovalMismatchError(RuntimeError):
    pass


class ApprovalDeniedError(RuntimeError):
    pass


class TrustedApprovalRequiredError(RuntimeError):
    pass


class TrustedApprovalMismatchError(RuntimeError):
    pass


class ApprovalReplayError(RuntimeError):
    pass


class ControlledActionExecutor:
    def __init__(
        self,
        policy: ActionPolicy,
        *,
        trusted_approval_verifier: TrustedApprovalVerifier | None = None,
        approval_artifact_store: ApprovalArtifactStore | None = None,
    ) -> None:
        self._policy = policy
        self._trusted_approval_verifier = trusted_approval_verifier
        self._approval_artifact_store = approval_artifact_store

    def execute(
        self,
        *,
        action_name: str,
        arguments: Mapping[str, Any],
        action: Callable[..., Any],
        approval_request: ApprovalRequest | None = None,
        approval_decision: ApprovalDecision | None = None,
    ) -> Any:
        """
        Existing approval execution path.

        This remains temporarily available so existing LangGraph/MCP callers
        continue to work while the trusted approval path is integrated.

        The legacy path will be removed after those callers are migrated.
        """
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

    def execute_trusted(
        self,
        *,
        action_name: str,
        arguments: Mapping[str, Any],
        action: Callable[..., Any],
        approval_request: ApprovalRequest | None = None,
        verified_approval: VerifiedApprovalDecision | None = None,
        now: datetime | None = None,
    ) -> Any:
        """
        Execute an action through the trusted human-approval path.

        For CONTROLLED actions:

        1. the ApprovalRequest must match the exact action and arguments;
        2. a VerifiedApprovalDecision must be supplied;
        3. the server-owned artifact is independently reverified;
        4. the supplied verified decision must match authoritative evidence;
        5. the artifact is atomically consumed before action execution.

        Consumption before execution intentionally fails closed. If the action
        itself later raises an exception, the approval remains consumed and
        cannot be replayed.
        """
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

        if verified_approval is None:
            raise TrustedApprovalRequiredError(
                f"Verified trusted approval required for action: "
                f"{action_name}"
            )

        if self._trusted_approval_verifier is None:
            raise TrustedApprovalRequiredError(
                "TrustedApprovalVerifier is not configured."
            )

        if self._approval_artifact_store is None:
            raise TrustedApprovalRequiredError(
                "ApprovalArtifactStore is not configured."
            )

        verification_time = now or datetime.now(timezone.utc)

        if verification_time.tzinfo is None:
            raise ValueError(
                "Execution time must be timezone-aware."
            )

        authoritative_decision = (
            self._trusted_approval_verifier.verify(
                approval_request,
                now=verification_time,
            )
        )

        self._validate_verified_decision(
            supplied=verified_approval,
            authoritative=authoritative_decision,
        )

        try:
            self._approval_artifact_store.consume(
                authoritative_decision.artifact_id,
                consumed_at=verification_time,
            )
        except ApprovalArtifactAlreadyConsumedError as exc:
            raise ApprovalReplayError(
                "Trusted approval artifact has already been consumed."
            ) from exc

        return action(**dict(arguments))

    @staticmethod
    def _validate_verified_decision(
        *,
        supplied: VerifiedApprovalDecision,
        authoritative: VerifiedApprovalDecision,
    ) -> None:
        """
        Prevent a caller from constructing a forged VerifiedApprovalDecision.

        The type alone is not treated as authority. Security-relevant fields
        must exactly match the decision independently reconstructed from the
        server-owned approval artifact store.
        """
        if supplied.artifact_id != authoritative.artifact_id:
            raise TrustedApprovalMismatchError(
                "Verified approval artifact_id does not match "
                "authoritative approval evidence."
            )

        if supplied.request_id != authoritative.request_id:
            raise TrustedApprovalMismatchError(
                "Verified approval request_id does not match "
                "authoritative approval evidence."
            )

        if supplied.action_name != authoritative.action_name:
            raise TrustedApprovalMismatchError(
                "Verified approval action does not match "
                "authoritative approval evidence."
            )

        if supplied.arguments_json != authoritative.arguments_json:
            raise TrustedApprovalMismatchError(
                "Verified approval arguments do not match "
                "authoritative approval evidence."
            )

        if supplied.issued_by != authoritative.issued_by:
            raise TrustedApprovalMismatchError(
                "Verified approval issuer does not match "
                "authoritative approval evidence."
            )

        if supplied.nonce != authoritative.nonce:
            raise TrustedApprovalMismatchError(
                "Verified approval nonce does not match "
                "authoritative approval evidence."
            )