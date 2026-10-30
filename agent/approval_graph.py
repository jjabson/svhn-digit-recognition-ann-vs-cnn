from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.inference_config_service import (
    InferenceConfigService,
)
from src.orchestration.action_policy import ActionPolicy
from src.orchestration.approval import ApprovalRequest
from src.orchestration.controlled_executor import (
    ControlledActionExecutor,
)
from src.orchestration.trusted_approval import (
    ApprovalRejectedError,
    TrustedApprovalVerificationError,
    TrustedApprovalVerifier,
)
from config.audit_event_store import AuditEventStore
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEvent,
    AuditEventType,
)
from time import perf_counter

from src.observability.controlled_action_metrics import (
    ControlledActionMetricEvent,
    ControlledActionMetrics,
)


class ApprovalGraphState(TypedDict, total=False):
    requested_threshold: float
    reason: str

    thread_id: str
    request_actor_type: str
    request_source_component: str
    request_actor_id: str | None

    approval_request: dict[str, Any]
    approval_response: dict[str, Any]

    action_status: str
    action_result: dict[str, Any]


def _request_to_dict(
    request: ApprovalRequest,
) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "action_name": request.action_name,
        "arguments": dict(request.arguments),
        "reason": request.reason,
    }


def _request_from_dict(
    data: dict[str, Any],
) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=data["request_id"],
        action_name=data["action_name"],
        arguments=data["arguments"],
        reason=data["reason"],
    )


def build_approval_graph(
    config_service: InferenceConfigService,
    artifact_store: ApprovalArtifactStore,
    trusted_approval_verifier: TrustedApprovalVerifier,
    checkpointer: BaseCheckpointSaver | None = None,
    audit_store: AuditEventStore | None = None,
    metrics: ControlledActionMetrics | None = None,
):
    """
    Build the durable controlled-action workflow.

    LangGraph owns pause/resume workflow mechanics only.

    Approval authority comes exclusively from persisted trusted
    approval artifacts verified by application-owned code.
    """

    executor = ControlledActionExecutor(
        ActionPolicy(),
        trusted_approval_verifier=trusted_approval_verifier,
        approval_artifact_store=artifact_store,
    )

    audit_enabled = audit_store is not None

    def get_request_actor_type(
            state: ApprovalGraphState,
    ) -> AuditActorType:
        raw_actor_type = state.get(
            "request_actor_type",
            AuditActorType.SYSTEM.value,
        )

        return AuditActorType(raw_actor_type)

    def prepare_request(
            state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request = ApprovalRequest(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": state[
                    "requested_threshold"
                ],
            },
            reason=state.get(
                "reason",
                "Change inference confidence threshold",
            ),
        )

        if metrics is not None:
            metrics.record(
                ControlledActionMetricEvent.REQUESTED
            )

        if audit_enabled:
            audit_store.append(
                AuditEvent.create(
                    event_type=(
                        AuditEventType
                        .CONTROLLED_ACTION_REQUESTED
                    ),
                    actor_type=get_request_actor_type(
                        state
                    ),
                    actor_id=state.get(
                        "request_actor_id"
                    ),
                    source_component=state.get(
                        "request_source_component",
                        "approval_graph",
                    ),
                    request_id=request.request_id,
                    thread_id=state.get("thread_id"),
                    payload={
                        "action_name": (
                            request.action_name
                        ),
                        "arguments": dict(
                            request.arguments
                        ),
                        "reason": request.reason,
                    },
                )
            )

        return {
            "approval_request": _request_to_dict(
                request
            ),
        }

    def record_approval_required(
            state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request_data = state["approval_request"]

        if metrics is not None:
            metrics.record(
                ControlledActionMetricEvent.APPROVAL_REQUIRED
            )

        if audit_enabled:
            current_config = (
                config_service.get_config()
            )

            audit_store.append(
                AuditEvent.create(
                    event_type=(
                        AuditEventType.APPROVAL_REQUIRED
                    ),
                    actor_type=AuditActorType.SYSTEM,
                    source_component="approval_graph",
                    request_id=request_data[
                        "request_id"
                    ],
                    thread_id=state.get(
                        "thread_id"
                    ),
                    payload={
                        "action_name": request_data[
                            "action_name"
                        ],
                        "arguments": request_data[
                            "arguments"
                        ],
                        "current_confidence_threshold": (
                            current_config
                            .confidence_threshold
                        ),
                    },
                )
            )

        return {}

    def approval_gate(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request_data = state["approval_request"]

        current_config = config_service.get_config()

        interrupt(
            {
                "type": "approval_required",
                "request_id": (
                    request_data["request_id"]
                ),
                "action_name": (
                    request_data["action_name"]
                ),
                "arguments": (
                    request_data["arguments"]
                ),
                "reason": request_data["reason"],
                "current_confidence_threshold": (
                    current_config
                    .confidence_threshold
                ),
                "message": (
                    "Trusted human approval is required "
                    "before changing the inference "
                    "confidence threshold from "
                    f"{current_config.confidence_threshold:.2f} "
                    "to "
                    f"{request_data['arguments']['confidence_threshold']:.2f}."
                ),
            }
        )

        # Resume data itself is deliberately ignored.
        # Authorization is obtained from the trusted artifact store.
        return {}

    def execute_controlled_action(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request = _request_from_dict(
            state["approval_request"]
        )

        started_at = perf_counter()

        try:
            verified_approval = (
                trusted_approval_verifier.verify(
                    request
                )
            )

            if metrics is not None:
                metrics.record(
                    ControlledActionMetricEvent.APPROVED
                )

            updated_config = (
                executor.execute_trusted(
                    action_name=(
                        request.action_name
                    ),
                    arguments=request.arguments,
                    action=(
                        config_service
                        .set_confidence_threshold
                    ),
                    approval_request=request,
                    verified_approval=(
                        verified_approval
                    ),
                )
            )

        except ApprovalRejectedError:
            if metrics is not None:
                metrics.record(
                    ControlledActionMetricEvent.REJECTED
                )

            return {
                "action_status": "rejected",
                "action_result": {
                    "confidence_threshold": (
                        config_service
                        .get_config()
                        .confidence_threshold
                    )
                },
            }

        except TrustedApprovalVerificationError as exc:
            if metrics is not None:
                metrics.record(
                    ControlledActionMetricEvent.DENIED
                )

            return {
                "action_status": "denied",
                "action_result": {
                    "confidence_threshold": (
                        config_service
                        .get_config()
                        .confidence_threshold
                    ),
                    "reason": str(exc),
                },
            }

        except Exception as exc:
            duration_ms = (
                                  perf_counter() - started_at
                          ) * 1000.0

            if metrics is not None:
                metrics.record(
                    ControlledActionMetricEvent.FAILED,
                    duration_ms=duration_ms,
                )

            if audit_enabled:
                audit_store.append(
                    AuditEvent.create(
                        event_type=(
                            AuditEventType
                            .CONTROLLED_ACTION_FAILED
                        ),
                        actor_type=(
                            AuditActorType.SYSTEM
                        ),
                        source_component=(
                            "approval_graph"
                        ),
                        request_id=request.request_id,
                        thread_id=state.get(
                            "thread_id"
                        ),
                        payload={
                            "action_name": (
                                request.action_name
                            ),
                            "arguments": dict(
                                request.arguments
                            ),
                            "error_type": (
                                type(exc).__name__
                            ),
                            "error_message": str(exc),
                        },
                    )
                )

            raise

        duration_ms = (
                              perf_counter() - started_at
                      ) * 1000.0

        if metrics is not None:
            metrics.record(
                ControlledActionMetricEvent.EXECUTED,
                duration_ms=duration_ms,
            )

        if audit_enabled:
            audit_store.append(
                AuditEvent.create(
                    event_type=(
                        AuditEventType
                        .CONTROLLED_ACTION_EXECUTED
                    ),
                    actor_type=AuditActorType.SYSTEM,
                    source_component="approval_graph",
                    request_id=request.request_id,
                    thread_id=state.get(
                        "thread_id"
                    ),
                    payload={
                        "action_name": (
                            request.action_name
                        ),
                        "arguments": dict(
                            request.arguments
                        ),
                        "result": {
                            "confidence_threshold": (
                                updated_config
                                .confidence_threshold
                            ),
                        },
                    },
                )
            )

        return {
            "action_status": "executed",
            "action_result": {
                "confidence_threshold": (
                    updated_config
                    .confidence_threshold
                )
            },
        }

    builder = StateGraph(
        ApprovalGraphState
    )

    builder.add_node(
        "prepare_request",
        prepare_request,
    )

    builder.add_node(
        "approval_gate",
        approval_gate,
    )

    builder.add_node(
        "execute_controlled_action",
        execute_controlled_action,
    )

    builder.add_edge(
        START,
        "prepare_request",
    )

    builder.add_edge(
        "prepare_request",
        "record_approval_required",
    )

    builder.add_edge(
        "record_approval_required",
        "approval_gate",
    )

    builder.add_edge(
        "approval_gate",
        "execute_controlled_action",
    )

    builder.add_edge(
        "execute_controlled_action",
        END,
    )
    builder.add_node(
        "record_approval_required",
        record_approval_required,
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
    )