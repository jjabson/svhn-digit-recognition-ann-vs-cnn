from langgraph.types import Command

from agent.approval_graph import (
    build_approval_graph,
)
from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.audit_event_store import (
    AuditEventStore,
)
from config.inference_config import (
    InferenceConfig,
)
from config.inference_config_service import (
    InferenceConfigService,
)
from src.orchestration.approval import (
    ApprovalRequest,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEventType,
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


def _build_environment(tmp_path):
    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    audit_store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    trusted_approval_verifier = (
        TrustedApprovalVerifier(
            artifact_store,
            trusted_issuers={
                DEFAULT_TRUSTED_APPROVAL_ISSUER,
            },
        )
    )

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    graph = build_approval_graph(
        config_service=config_service,
        artifact_store=artifact_store,
        trusted_approval_verifier=(
            trusted_approval_verifier
        ),
        audit_store=audit_store,
    )

    trusted_approval_service = (
        TrustedApprovalService(
            artifact_store,
            audit_store=audit_store,
        )
    )

    return (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
    )


def _request_from_graph_state(
    graph,
    config,
) -> ApprovalRequest:
    state = graph.get_state(config)

    request_data = state.values[
        "approval_request"
    ]

    return ApprovalRequest(
        request_id=request_data["request_id"],
        action_name=request_data["action_name"],
        arguments=request_data["arguments"],
        reason=request_data["reason"],
    )


def test_approved_controlled_action_records_complete_audit_lifecycle(
    tmp_path,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
    ) = _build_environment(tmp_path)

    thread_id = "audit-lifecycle-approved"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Full audit lifecycle test"
            ),
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=config,
    )

    assert "__interrupt__" in initial_result

    request = _request_from_graph_state(
        graph,
        config,
    )

    trusted_approval_service.record_decision(
        request,
        approved=True,
        thread_id=thread_id,
    )

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": (
                    "Trusted approval recorded"
                ),
            }
        ),
        config=config,
    )

    assert (
        resumed_result["action_status"]
        == "executed"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.95
    )

    events = (
        audit_store.list_by_request_id(
            request.request_id
        )
    )

    assert len(events) == 4

    event_types = [
        event.event_type
        for event in events
    ]

    assert event_types == [
        (
            AuditEventType
            .CONTROLLED_ACTION_REQUESTED
        ),
        AuditEventType.APPROVAL_REQUIRED,
        AuditEventType.APPROVAL_GRANTED,
        (
            AuditEventType
            .CONTROLLED_ACTION_EXECUTED
        ),
    ]

    assert all(
        event.request_id
        == request.request_id
        for event in events
    )

    assert all(
        event.thread_id
        == thread_id
        for event in events
    )

    assert (
        events[0].actor_type
        == AuditActorType.MCP_CLIENT
    )

    assert (
        events[0].source_component
        == "mcp_server"
    )

    assert (
        events[1].actor_type
        == AuditActorType.SYSTEM
    )

    assert (
        events[1].source_component
        == "approval_graph"
    )

    assert (
        events[2].actor_type
        == AuditActorType.TRUSTED_HUMAN
    )

    assert (
        events[2].source_component
        == "trusted_approval_service"
    )

    assert (
        events[3].actor_type
        == AuditActorType.SYSTEM
    )

    assert (
        events[3].source_component
        == "approval_graph"
    )

def test_rejected_controlled_action_records_complete_audit_lifecycle(
    tmp_path,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
    ) = _build_environment(tmp_path)

    thread_id = "audit-lifecycle-rejected"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Rejected audit lifecycle test"
            ),
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=config,
    )

    assert "__interrupt__" in initial_result

    request = _request_from_graph_state(
        graph,
        config,
    )

    trusted_approval_service.record_decision(
        request,
        approved=False,
        thread_id=thread_id,
    )

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": False,
                "reason": (
                    "Trusted rejection recorded"
                ),
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

    events = (
        audit_store.list_by_request_id(
            request.request_id
        )
    )

    assert len(events) == 3

    event_types = [
        event.event_type
        for event in events
    ]

    assert event_types == [
        (
            AuditEventType
            .CONTROLLED_ACTION_REQUESTED
        ),
        AuditEventType.APPROVAL_REQUIRED,
        AuditEventType.APPROVAL_REJECTED,
    ]

    assert all(
        event.request_id
        == request.request_id
        for event in events
    )

    assert all(
        event.thread_id
        == thread_id
        for event in events
    )

    assert not any(
        event.event_type
        == (
            AuditEventType
            .CONTROLLED_ACTION_EXECUTED
        )
        for event in events
    )

    assert not any(
        event.event_type
        == (
            AuditEventType
            .CONTROLLED_ACTION_FAILED
        )
        for event in events
    )