from __future__ import annotations

from langgraph.types import Command

from agent.approval_graph import build_approval_graph
from config.inference_config import InferenceConfig
from config.inference_config_service import InferenceConfigService


def _thread_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def main() -> None:
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    graph = build_approval_graph(
        config_service=config_service,
    )

    config = _thread_config(
        "manual-threshold-approval"
    )

    requested_threshold = 0.95

    result = graph.invoke(
        {
            "requested_threshold": requested_threshold,
            "reason": (
                "Manual HITL verification for confidence "
                "threshold update"
            ),
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        raise RuntimeError(
            "Expected approval interrupt was not produced"
        )

    payload = interrupts[0].value

    print()
    print("Approval required")
    print("-----------------")
    print(f"Request ID: {payload['request_id']}")
    print(f"Action: {payload['action_name']}")
    print(
        "Current threshold: "
        f"{payload['current_confidence_threshold']:.2f}"
    )
    print(
        "Requested threshold: "
        f"{payload['arguments']['confidence_threshold']:.2f}"
    )
    print(f"Reason: {payload['reason']}")
    print()

    answer = input("Approve? [y/N]: ").strip().lower()

    approved = answer in {"y", "yes"}

    reason = (
        "Approved through CLI"
        if approved
        else "Rejected through CLI"
    )

    final_result = graph.invoke(
        Command(
            resume={
                "approved": approved,
                "reason": reason,
            }
        ),
        config=config,
    )

    print()
    print("Workflow result")
    print("---------------")
    print(
        f"Action status: "
        f"{final_result['action_status']}"
    )
    print(
        "Active confidence threshold: "
        f"{config_service.get_config().confidence_threshold:.2f}"
    )


if __name__ == "__main__":
    main()