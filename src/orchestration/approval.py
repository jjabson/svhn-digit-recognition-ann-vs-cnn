from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from src.orchestration.action_policy import ActionRisk


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


TERMINAL_APPROVAL_STATUSES = {
    ApprovalStatus.APPROVED,
    ApprovalStatus.REJECTED,
    ApprovalStatus.EXPIRED,
    ApprovalStatus.CANCELLED,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _freeze_arguments(
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Copy the top-level action arguments and expose them as a read-only mapping.

    This prevents callers from changing the request's argument dictionary
    after approval has been requested.
    """
    return MappingProxyType(dict(arguments))


@dataclass(frozen=True)
class ApprovalRequest:
    action_name: str
    arguments: Mapping[str, Any]
    reason: str
    risk: ActionRisk = ActionRisk.CONTROLLED
    request_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

        if not self.action_name.strip():
            raise ValueError("action_name must not be empty")

        if self.risk is not ActionRisk.CONTROLLED:
            raise ValueError(
                "ApprovalRequest may only be created for CONTROLLED actions"
            )

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        object.__setattr__(
            self,
            "arguments",
            _freeze_arguments(self.arguments),
        )

    @property
    def status(self) -> ApprovalStatus:
        return ApprovalStatus.PENDING

    def matches_action(
        self,
        action_name: str,
        arguments: Mapping[str, Any],
    ) -> bool:
        """
        Return True only when an attempted action exactly matches the action
        and arguments represented by this approval request.
        """
        return (
            self.action_name == action_name
            and dict(self.arguments) == dict(arguments)
        )


@dataclass(frozen=True)
class ApprovalDecision:
    request_id: str
    status: ApprovalStatus
    decided_by: str
    reason: str = ""
    decided_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must not be empty")

        if self.status not in TERMINAL_APPROVAL_STATUSES:
            raise ValueError(
                "ApprovalDecision must contain a terminal approval status"
            )

        if not self.decided_by.strip():
            raise ValueError("decided_by must not be empty")

        if self.decided_at.tzinfo is None:
            raise ValueError("decided_at must be timezone-aware")

    @property
    def permits_execution(self) -> bool:
        return self.status is ApprovalStatus.APPROVED


def resolve_approval_status(
    request: ApprovalRequest,
    decision: ApprovalDecision | None,
) -> ApprovalStatus:
    """
    Resolve the current approval state.

    No decision means the request remains pending. A supplied decision must
    refer to the same request.
    """
    if decision is None:
        return ApprovalStatus.PENDING

    if decision.request_id != request.request_id:
        raise ValueError(
            "ApprovalDecision does not belong to this ApprovalRequest"
        )

    return decision.status