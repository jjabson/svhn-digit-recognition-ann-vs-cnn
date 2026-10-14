from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

from agent.approval_graph import build_approval_graph
from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.inference_config import InferenceConfig
from config.inference_config_service import (
    InferenceConfigService,
)
from src.orchestration.approval import ApprovalRequest
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
)
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
    TrustedApprovalService,
)


def _build_test_graph(tmp_path):
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    verifier = TrustedApprovalVerifier(
        artifact_store,
        trusted_issuers={
            DEFAULT_TRUSTED_APPROVAL_ISSUER,
        },
    )

    trusted_service = TrustedApprovalService(
        artifact_store
    )

    graph = build_approval_graph(
        config_service=config_service,
        artifact_store=artifact_store,
        trusted_approval_verifier=verifier,
    )

    return (
        graph,
        config_service,
        artifact_store,
        trusted_service,
    )


def _thread_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def _request_from_interrupt(
    result,
) -> ApprovalRequest:
    payload = (
        result["__interrupt__"][0].value
    )

    return ApprovalRequest(
        request_id=payload["request_id"],
        action_name=payload["action_name"],
        arguments=payload["arguments"],
        reason=payload["reason"],
    )


def test_graph_interrupts_before_threshold_change(
    tmp_path,
):
    (
        graph,
        config_service,
        _,
        _,
    ) = _build_test_graph(tmp_path)

    config = _thread_config(
        "interrupt-before-change"
    )

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
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


def test_interrupt_contains_approval_context(
    tmp_path,
):
    graph, _, _, _ = (
        _build_test_graph(tmp_path)
    )

    config = _thread_config(
        "interrupt-context"
    )

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
        },
        config=config,
    )

    payload = (
        result["__interrupt__"][0].value
    )

    assert (
        payload["type"]
        == "approval_required"
    )

    assert (
        payload["action_name"]
        == "set_confidence_threshold"
    )

    assert payload["arguments"] == {
        "confidence_threshold": 0.95,
    }

    assert (
        payload[
            "current_confidence_threshold"
        ]
        == 0.90
    )


def test_trusted_approved_resume_changes_threshold(
    tmp_path,
):
    (
        graph,
        config_service,
        _,
        trusted_service,
    ) = _build_test_graph(tmp_path)

    config = _thread_config(
        "trusted-approve"
    )

    initial = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
        },
        config=config,
    )

    request = _request_from_interrupt(
        initial
    )

    trusted_service.record_decision(
        request,
        approved=True,
    )

    resumed = graph.invoke(
        Command(
            resume={
                "source": (
                    "trusted_local_human_channel"
                )
            }
        ),
        config=config,
    )

    assert (
        resumed["action_status"]
        == "executed"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.95
    )


def test_trusted_rejection_preserves_threshold(
    tmp_path,
):
    (
        graph,
        config_service,
        _,
        trusted_service,
    ) = _build_test_graph(tmp_path)

    config = _thread_config(
        "trusted-reject"
    )

    initial = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
        },
        config=config,
    )

    request = _request_from_interrupt(
        initial
    )

    trusted_service.record_decision(
        request,
        approved=False,
    )

    resumed = graph.invoke(
        Command(
            resume={
                "source": (
                    "trusted_local_human_channel"
                )
            }
        ),
        config=config,
    )

    assert (
        resumed["action_status"]
        == "rejected"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )


def test_raw_approved_resume_cannot_self_authorize(
    tmp_path,
):
    (
        graph,
        config_service,
        _,
        _,
    ) = _build_test_graph(tmp_path)

    config = _thread_config(
        "untrusted-self-approval"
    )

    graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
        },
        config=config,
    )

    resumed = graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": (
                    "Agent claims approval"
                ),
            }
        ),
        config=config,
    )

    assert (
        resumed["action_status"]
        == "denied"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )


def test_approval_request_id_survives_resume(
    tmp_path,
):
    (
        graph,
        _,
        _,
        trusted_service,
    ) = _build_test_graph(tmp_path)

    config = _thread_config(
        "request-id-survives"
    )

    initial = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Use a stricter threshold"
            ),
        },
        config=config,
    )

    request = _request_from_interrupt(
        initial
    )

    request_id = request.request_id

    trusted_service.record_decision(
        request,
        approved=True,
    )

    resumed = graph.invoke(
        Command(
            resume={
                "source": "trusted_cli",
            }
        ),
        config=config,
    )

    assert (
        resumed["approval_request"]
        ["request_id"]
        == request_id
    )


def test_approval_graph_accepts_injected_checkpointer(
    tmp_path,
):
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    verifier = TrustedApprovalVerifier(
        artifact_store,
        trusted_issuers={
            DEFAULT_TRUSTED_APPROVAL_ISSUER,
        },
    )

    checkpointer = InMemorySaver()

    graph = build_approval_graph(
        config_service=config_service,
        artifact_store=artifact_store,
        trusted_approval_verifier=verifier,
        checkpointer=checkpointer,
    )

    assert (
        graph.checkpointer
        is checkpointer
    )