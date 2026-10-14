from __future__ import annotations

from config.approval_artifact_store import ApprovalArtifactStore
from src.orchestration.approval import ApprovalRequest
from src.orchestration.approval_artifact import (
    DEFAULT_APPROVAL_TTL_SECONDS,
    ApprovalArtifact,
)


DEFAULT_TRUSTED_APPROVAL_ISSUER = (
    "local_trusted_human_channel"
)


class TrustedApprovalService:
    """
    Application-owned service for recording trusted human decisions.

    This service is intentionally not exposed through MCP.
    """

    def __init__(
        self,
        artifact_store: ApprovalArtifactStore,
        *,
        issued_by: str = DEFAULT_TRUSTED_APPROVAL_ISSUER,
        ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    ) -> None:
        if not issued_by:
            raise ValueError(
                "Trusted approval issuer must not be empty."
            )

        if ttl_seconds <= 0:
            raise ValueError(
                "Approval TTL must be greater than zero."
            )

        self._artifact_store = artifact_store
        self._issued_by = issued_by
        self._ttl_seconds = ttl_seconds

    def record_decision(
        self,
        request: ApprovalRequest,
        *,
        approved: bool,
    ) -> ApprovalArtifact:
        artifact = ApprovalArtifact.issue(
            request_id=request.request_id,
            action_name=request.action_name,
            arguments=request.arguments,
            approved=approved,
            issued_by=self._issued_by,
            ttl_seconds=self._ttl_seconds,
        )

        self._artifact_store.save(artifact)

        return artifact