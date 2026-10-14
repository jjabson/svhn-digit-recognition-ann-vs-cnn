from __future__ import annotations

import argparse

from langgraph.types import Command

from agent.approval_graph import (
    build_approval_graph,
)
from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.approval_checkpoint import (
    create_approval_checkpointer,
)
from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
)
from config.inference_config_service import (
    InferenceConfigService,
)
from config.inference_config_store import (
    InferenceConfigStore,
)
from src.orchestration.approval import (
    ApprovalRequest,
)
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
)
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
    TrustedApprovalService,
)


def _thread_config(
    thread_id: str,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Approve or reject a pending "
            "confidence-threshold change "
            "through the trusted local "
            "human channel."
        )
    )

    parser.add_argument(
        "thread_id",
        help=(
            "Pending LangGraph thread_id "
            "returned by the MCP request."
        ),
    )

    args = parser.parse_args()

    inference_store = (
        InferenceConfigStore()
    )

    config_service = (
        InferenceConfigService(
            DEFAULT_INFERENCE_CONFIG,
            store=inference_store,
        )
    )

    artifact_store = (
        ApprovalArtifactStore()
    )

    verifier = TrustedApprovalVerifier(
        artifact_store,
        trusted_issuers={
            DEFAULT_TRUSTED_APPROVAL_ISSUER,
        },
    )

    trusted_service = (
        TrustedApprovalService(
            artifact_store
        )
    )

    checkpointer = (
        create_approval_checkpointer()
    )

    graph = build_approval_graph(
        config_service=config_service,
        artifact_store=artifact_store,
        trusted_approval_verifier=verifier,
        checkpointer=checkpointer,
    )

    config = _thread_config(
        args.thread_id
    )

    snapshot = graph.get_state(
        config
    )

    state = snapshot.values

    if (
        not state
        or "approval_request" not in state
    ):
        raise RuntimeError(
            "No pending approval workflow "
            f"found for thread_id "
            f"{args.thread_id}."
        )

    if state.get("action_status"):
        print(
            "Workflow already completed: "
            f"{state['action_status']}"
        )
        return

    request_data = (
        state["approval_request"]
    )

    request = ApprovalRequest(
        request_id=(
            request_data["request_id"]
        ),
        action_name=(
            request_data["action_name"]
        ),
        arguments=(
            request_data["arguments"]
        ),
        reason=request_data["reason"],
    )

    existing_artifact = (
        artifact_store
        .get_by_request_id(
            request.request_id
        )
    )

    if existing_artifact is None:
        current_threshold = (
            config_service
            .get_config()
            .confidence_threshold
        )

        requested_threshold = (
            request.arguments[
                "confidence_threshold"
            ]
        )

        print()
        print("Trusted approval required")
        print("-------------------------")
        print(
            f"Thread ID: {args.thread_id}"
        )
        print(
            f"Request ID: "
            f"{request.request_id}"
        )
        print(
            f"Action: "
            f"{request.action_name}"
        )
        print(
            "Current threshold: "
            f"{current_threshold:.2f}"
        )
        print(
            "Requested threshold: "
            f"{requested_threshold:.2f}"
        )
        print(
            f"Reason: {request.reason}"
        )
        print()

        answer = input(
            "Approve? [y/N]: "
        ).strip().lower()

        approved = answer in {
            "y",
            "yes",
        }

        artifact = (
            trusted_service
            .record_decision(
                request,
                approved=approved,
            )
        )

        print()
        print(
            "Trusted decision recorded: "
            f"{'approved' if artifact.approved else 'rejected'}"
        )

    else:
        print(
            "Existing trusted decision found: "
            f"{'approved' if existing_artifact.approved else 'rejected'}"
        )

    final_result = graph.invoke(
        Command(
            resume={
                "source": (
                    "trusted_local_human_channel"
                )
            }
        ),
        config=config,
    )

    print()
    print("Workflow result")
    print("---------------")
    print(
        "Action status: "
        f"{final_result['action_status']}"
    )
    print(
        "Active confidence threshold: "
        f"{config_service.get_config().confidence_threshold:.2f}"
    )


if __name__ == "__main__":
    main()