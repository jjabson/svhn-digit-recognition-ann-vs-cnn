from __future__ import annotations

from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.audit_event_store import (
    AuditEventStore,
)
from src.orchestration.approval import (
    ApprovalRequest,
)
from src.orchestration.approval_artifact import (
    DEFAULT_APPROVAL_TTL_SECONDS,
    ApprovalArtifact,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEvent,
    AuditEventType,
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
        audit_store: AuditEventStore | None = None,
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
        self._audit_store = audit_store

    def record_decision(
        self,
        request: ApprovalRequest,
        *,
        approved: bool,
        thread_id: str | None = None,
    ) -> ApprovalArtifact:
        artifact = ApprovalArtifact.issue(
            request_id=request.request_id,
            action_name=request.action_name,
            arguments=request.arguments,
            approved=approved,
            issued_by=self._issued_by,
            ttl_seconds=self._ttl_seconds,
        )

        # Persist trusted authorization evidence first.
        #
        # We must not write an APPROVAL_GRANTED or
        # APPROVAL_REJECTED audit event unless the trusted
        # approval artifact was successfully persisted.
        self._artifact_store.save(artifact)

        if self._audit_store is not None:
            event_type = (
                AuditEventType.APPROVAL_GRANTED
                if approved
                else AuditEventType.APPROVAL_REJECTED
            )

            self._audit_store.append(
                AuditEvent.create(
                    event_type=event_type,
                    actor_type=(
                        AuditActorType.TRUSTED_HUMAN
                    ),
                    actor_id=self._issued_by,
                    source_component=(
                        "trusted_approval_service"
                    ),
                    request_id=request.request_id,
                    thread_id=thread_id,
                    payload={
                        "action_name": (
                            request.action_name
                        ),
                        "arguments": dict(
                            request.arguments
                        ),
                        "approved": approved,
                        "artifact_id": (
                            artifact.artifact_id
                        ),
                        "issued_by": (
                            artifact.issued_by
                        ),
                    },
                )
            )

        return artifact