from langgraph.types import Command

from agent.approval_graph import build_approval_graph
from config.inference_config import InferenceConfig
from config.inference_config_service import (
    InferenceConfigService,
)
from langgraph.checkpoint.memory import InMemorySaver


def _build_test_graph():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    graph = build_approval_graph(
        config_service=config_service,
    )

    return graph, config_service


def _thread_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_graph_interrupts_before_threshold_change():
    graph, config_service = _build_test_graph()

    config = _thread_config(
        "interrupt-before-change"
    )

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Use a stricter threshold",
        },
        config=config,
    )

    assert "__interrupt__" in result

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )


def test_interrupt_contains_approval_context():
    graph, _ = _build_test_graph()

    config = _thread_config(
        "interrupt-context"
    )

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Use a stricter threshold",
        },
        config=config,
    )

    interrupts = result["__interrupt__"]

    assert len(interrupts) == 1

    payload = interrupts[0].value

    assert payload["type"] == "approval_required"

    assert (
        payload["action_name"]
        == "set_confidence_threshold"
    )

    assert payload["arguments"] == {
        "confidence_threshold": 0.95,
    }

    assert (
        payload["current_confidence_threshold"]
        == 0.90
    )


def test_approved_resume_changes_threshold():
    graph, config_service = _build_test_graph()

    config = _thread_config(
        "approve-threshold-change"
    )

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Use a stricter threshold",
        },
        config=config,
    )

    assert "__interrupt__" in initial_result

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": "Approved by human",
            }
        ),
        config=config,
    )

    assert (
        resumed_result["action_status"]
        == "executed"
    )

    assert (
        resumed_result["action_result"]
        ["confidence_threshold"]
        == 0.95
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.95
    )


def test_rejected_resume_preserves_threshold():
    graph, config_service = _build_test_graph()

    config = _thread_config(
        "reject-threshold-change"
    )

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Use a stricter threshold",
        },
        config=config,
    )

    assert "__interrupt__" in initial_result

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": False,
                "reason": "Human rejected change",
            }
        ),
        config=config,
    )

    assert (
        resumed_result["action_status"]
        == "rejected"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )


def test_approval_request_id_survives_interrupt_resume():
    graph, _ = _build_test_graph()

    config = _thread_config(
        "request-id-survives"
    )

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Use a stricter threshold",
        },
        config=config,
    )

    interrupt_payload = (
        initial_result["__interrupt__"][0].value
    )

    request_id_before_resume = (
        interrupt_payload["request_id"]
    )

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": "Approved",
            }
        ),
        config=config,
    )

    assert (
        resumed_result["approval_request"]
        ["request_id"]
        == request_id_before_resume
    )

def test_approval_graph_accepts_injected_checkpointer():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    checkpointer = InMemorySaver()

    graph = build_approval_graph(
        config_service=config_service,
        checkpointer=checkpointer,
    )

    assert graph.checkpointer is checkpointer