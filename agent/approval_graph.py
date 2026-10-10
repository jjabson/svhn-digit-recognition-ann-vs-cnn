from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config.inference_config_service import InferenceConfigService
from src.orchestration.action_policy import ActionPolicy
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)
from src.orchestration.controlled_executor import (
    ApprovalDeniedError,
    ControlledActionExecutor,
)


class ApprovalGraphState(TypedDict, total=False):
    requested_threshold: float
    reason: str

    approval_request: dict[str, Any]
    approval_response: dict[str, Any]

    action_status: str
    action_result: dict[str, Any]


def _request_to_dict(request: ApprovalRequest) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "action_name": request.action_name,
        "arguments": dict(request.arguments),
        "reason": request.reason,
    }


def _request_from_dict(data: dict[str, Any]) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=data["request_id"],
        action_name=data["action_name"],
        arguments=data["arguments"],
        reason=data["reason"],
    )


def build_approval_graph(
    config_service: InferenceConfigService,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """
    Build a LangGraph workflow for the controlled
    set_confidence_threshold action.

    LangGraph owns pause/resume behavior only.

    Authorization and execution remain owned by:
      - ActionPolicy
      - ApprovalRequest / ApprovalDecision
      - ControlledActionExecutor
      - InferenceConfigService
    """

    executor = ControlledActionExecutor(ActionPolicy())

    def prepare_request(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request = ApprovalRequest(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": state["requested_threshold"],
            },
            reason=state.get(
                "reason",
                "Change inference confidence threshold",
            ),
        )

        return {
            "approval_request": _request_to_dict(request),
        }

    def approval_gate(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request_data = state["approval_request"]
        current_config = config_service.get_config()

        response = interrupt(
            {
                "type": "approval_required",
                "request_id": request_data["request_id"],
                "action_name": request_data["action_name"],
                "arguments": request_data["arguments"],
                "reason": request_data["reason"],
                "current_confidence_threshold": (
                    current_config.confidence_threshold
                ),
                "message": (
                    "Approve changing the inference confidence "
                    f"threshold from "
                    f"{current_config.confidence_threshold:.2f} "
                    f"to "
                    f"{request_data['arguments']['confidence_threshold']:.2f}?"
                ),
            }
        )

        if not isinstance(response, dict):
            raise ValueError(
                "Approval response must be a dictionary"
            )

        if "approved" not in response:
            raise ValueError(
                "Approval response must contain 'approved'"
            )

        return {
            "approval_response": response,
        }

    def execute_controlled_action(
        state: ApprovalGraphState,
    ) -> dict[str, Any]:
        request = _request_from_dict(
            state["approval_request"]
        )

        response = state["approval_response"]

        approved = response["approved"]

        if not isinstance(approved, bool):
            raise ValueError(
                "'approved' must be a boolean"
            )

        decision = ApprovalDecision(
            request_id=request.request_id,
            status=(
                ApprovalStatus.APPROVED
                if approved
                else ApprovalStatus.REJECTED
            ),
            decided_by="human",
            reason=response.get("reason", ""),
        )

        try:
            updated_config = executor.execute(
                action_name=request.action_name,
                arguments=request.arguments,
                action=config_service.set_confidence_threshold,
                approval_request=request,
                approval_decision=decision,
            )

        except ApprovalDeniedError:
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

        return {
            "action_status": "executed",
            "action_result": {
                "confidence_threshold": (
                    updated_config.confidence_threshold
                )
            },
        }

    builder = StateGraph(ApprovalGraphState)

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
