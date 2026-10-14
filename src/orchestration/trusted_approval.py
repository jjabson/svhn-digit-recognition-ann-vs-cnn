from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Collection, Mapping, Protocol

from config.approval_artifact_store import ApprovalArtifactStore
from src.orchestration.approval_artifact import canonicalize_arguments


class ApprovalRequestLike(Protocol):
    """
    Structural contract required by the trusted approval verifier.

    The existing ApprovalRequest satisfies this contract without coupling
    verification logic to workflow-specific construction details.
    """

    request_id: str
    action_name: str
    arguments: Mapping[str, Any]


class TrustedApprovalVerificationError(RuntimeError):
    """Base error for trusted approval verification failures."""


class ApprovalArtifactMissingError(
    TrustedApprovalVerificationError
):
    """Raised when no trusted artifact exists for a request."""


class ApprovalRequestMismatchError(
    TrustedApprovalVerificationError
):
    """Raised when an artifact belongs to another request."""


class ApprovalActionMismatchError(
    TrustedApprovalVerificationError
):
    """Raised when the approved action does not match the request."""


class ApprovalArgumentsMismatchError(
    TrustedApprovalVerificationError
):
    """Raised when approved arguments do not match the request."""


class UntrustedApprovalIssuerError(
    TrustedApprovalVerificationError
):
    """Raised when approval evidence came from an untrusted issuer."""


class ApprovalRejectedError(
    TrustedApprovalVerificationError
):
    """Raised when the trusted human rejected the request."""


class ApprovalExpiredError(
    TrustedApprovalVerificationError
):
    """Raised when trusted approval evidence has expired."""


class ApprovalAlreadyConsumedError(
    TrustedApprovalVerificationError
):
    """Raised when trusted approval evidence has already been used."""


@dataclass(frozen=True)
class VerifiedApprovalDecision:
    """
    Application-owned proof that trusted approval verification succeeded.

    This type may only be produced by TrustedApprovalVerifier after all
    verification checks pass.
    """

    artifact_id: str
    request_id: str
    action_name: str
    arguments_json: str
    issued_by: str
    nonce: str
    verified_at: datetime

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)


class TrustedApprovalVerifier:
    """
    Validate server-owned trusted approval evidence against an action request.

    Verification does not consume the artifact. Consumption will be enforced
    at the controlled-execution boundary in the next integration phase.
    """

    def __init__(
        self,
        artifact_store: ApprovalArtifactStore,
        *,
        trusted_issuers: Collection[str],
    ) -> None:
        normalized_issuers = frozenset(
            issuer
            for issuer in trusted_issuers
            if issuer
        )

        if not normalized_issuers:
            raise ValueError(
                "At least one trusted approval issuer is required."
            )

        self._artifact_store = artifact_store
        self._trusted_issuers = normalized_issuers

    def verify(
        self,
        request: ApprovalRequestLike,
        *,
        now: datetime | None = None,
    ) -> VerifiedApprovalDecision:
        verification_time = now or datetime.now(timezone.utc)

        if verification_time.tzinfo is None:
            raise ValueError(
                "Verification time must be timezone-aware."
            )

        artifact = self._artifact_store.get_by_request_id(
            request.request_id
        )

        if artifact is None:
            raise ApprovalArtifactMissingError(
                f"No trusted approval artifact exists for "
                f"request {request.request_id}."
            )

        if artifact.request_id != request.request_id:
            raise ApprovalRequestMismatchError(
                "Approval artifact request_id does not match "
                "the controlled action request."
            )

        if artifact.action_name != request.action_name:
            raise ApprovalActionMismatchError(
                "Approval artifact action does not match "
                "the controlled action request."
            )

        requested_arguments_json = canonicalize_arguments(
            request.arguments
        )

        if artifact.arguments_json != requested_arguments_json:
            raise ApprovalArgumentsMismatchError(
                "Approval artifact arguments do not match "
                "the controlled action request."
            )

        if artifact.issued_by not in self._trusted_issuers:
            raise UntrustedApprovalIssuerError(
                f"Approval issuer is not trusted: "
                f"{artifact.issued_by}"
            )

        if artifact.approved is not True:
            raise ApprovalRejectedError(
                "The trusted human rejected this request."
            )

        if artifact.is_expired(now=verification_time):
            raise ApprovalExpiredError(
                "The trusted approval artifact has expired."
            )

        if artifact.is_consumed:
            raise ApprovalAlreadyConsumedError(
                "The trusted approval artifact has already "
                "been consumed."
            )

        return VerifiedApprovalDecision(
            artifact_id=artifact.artifact_id,
            request_id=artifact.request_id,
            action_name=artifact.action_name,
            arguments_json=artifact.arguments_json,
            issued_by=artifact.issued_by,
            nonce=artifact.nonce,
            verified_at=verification_time,
        )