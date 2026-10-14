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


class ApprovalGraphState(TypedDict, total=False):
    requested_threshold: float
    reason: str

    approval_request: dict[str, Any]

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

    def prepare_request(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request = ApprovalRequest(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": (
                    state["requested_threshold"]
                ),
            },
            reason=state.get(
                "reason",
                "Change inference confidence threshold",
            ),
        )

        return {
            "approval_request": _request_to_dict(
                request
            ),
        }

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

        try:
            verified_approval = (
                trusted_approval_verifier.verify(
                    request
                )
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

    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer,
    )