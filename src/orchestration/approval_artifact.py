from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import uuid4


DEFAULT_APPROVAL_TTL_SECONDS = 900


class ApprovalArtifactError(ValueError):
    """Base error for invalid approval artifacts."""


class InvalidApprovalArtifactError(ApprovalArtifactError):
    """Raised when an approval artifact violates domain invariants."""


def canonicalize_arguments(arguments: Mapping[str, Any]) -> str:
    """
    Convert action arguments into a deterministic JSON representation.

    Sorting keys ensures argument ordering does not affect approval binding.
    NaN and Infinity are rejected because they are not valid portable JSON
    values and could undermine deterministic comparison.
    """
    try:
        return json.dumps(
            dict(arguments),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidApprovalArtifactError(
            "Approval arguments must be valid JSON-compatible values."
        ) from exc


@dataclass(frozen=True)
class ApprovalArtifact:
    """
    Server-owned evidence recording a trusted human approval decision.

    The artifact is immutable. Consumption is represented by creating a new
    instance with consumed_at populated and persisting that replacement.
    """

    artifact_id: str
    request_id: str
    action_name: str
    arguments_json: str
    approved: bool
    issued_at: datetime
    expires_at: datetime
    issued_by: str
    nonce: str
    consumed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise InvalidApprovalArtifactError(
                "artifact_id must not be empty."
            )

        if not self.request_id:
            raise InvalidApprovalArtifactError(
                "request_id must not be empty."
            )

        if not self.action_name:
            raise InvalidApprovalArtifactError(
                "action_name must not be empty."
            )

        if not self.issued_by:
            raise InvalidApprovalArtifactError(
                "issued_by must not be empty."
            )

        if not self.nonce:
            raise InvalidApprovalArtifactError(
                "nonce must not be empty."
            )

        if self.issued_at.tzinfo is None:
            raise InvalidApprovalArtifactError(
                "issued_at must be timezone-aware."
            )

        if self.expires_at.tzinfo is None:
            raise InvalidApprovalArtifactError(
                "expires_at must be timezone-aware."
            )

        if self.expires_at <= self.issued_at:
            raise InvalidApprovalArtifactError(
                "expires_at must be later than issued_at."
            )

        if self.consumed_at is not None:
            if self.consumed_at.tzinfo is None:
                raise InvalidApprovalArtifactError(
                    "consumed_at must be timezone-aware."
                )

            if self.consumed_at < self.issued_at:
                raise InvalidApprovalArtifactError(
                    "consumed_at cannot be earlier than issued_at."
                )

        try:
            arguments = json.loads(self.arguments_json)
        except json.JSONDecodeError as exc:
            raise InvalidApprovalArtifactError(
                "arguments_json must contain valid JSON."
            ) from exc

        if not isinstance(arguments, dict):
            raise InvalidApprovalArtifactError(
                "arguments_json must represent a JSON object."
            )

        canonical_arguments = canonicalize_arguments(arguments)

        object.__setattr__(
            self,
            "arguments_json",
            canonical_arguments,
        )

    @classmethod
    def issue(
        cls,
        *,
        request_id: str,
        action_name: str,
        arguments: Mapping[str, Any],
        approved: bool,
        issued_by: str,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
        issued_at: datetime | None = None,
    ) -> ApprovalArtifact:
        if ttl_seconds <= 0:
            raise InvalidApprovalArtifactError(
                "ttl_seconds must be greater than zero."
            )

        issued_at = issued_at or datetime.now(timezone.utc)

        if issued_at.tzinfo is None:
            raise InvalidApprovalArtifactError(
                "issued_at must be timezone-aware."
            )

        return cls(
            artifact_id=str(uuid4()),
            request_id=request_id,
            action_name=action_name,
            arguments_json=canonicalize_arguments(arguments),
            approved=approved,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(seconds=ttl_seconds),
            issued_by=issued_by,
            nonce=secrets.token_urlsafe(32),
            consumed_at=None,
        )

    @property
    def arguments(self) -> dict[str, Any]:
        """
        Return a new dictionary containing the bound action arguments.

        A new object is returned so callers cannot mutate artifact state.
        """
        return json.loads(self.arguments_json)

    def is_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> bool:
        now = now or datetime.now(timezone.utc)

        if now.tzinfo is None:
            raise InvalidApprovalArtifactError(
                "now must be timezone-aware."
            )

        return now >= self.expires_at

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at is not None