import pytest
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
from src.observability.controlled_action_metrics import (
    ControlledActionMetrics,
)


def _thread_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def _build_test_environment(tmp_path):
    audit_store = AuditEventStore(
        tmp_path / "audit_events.db"
    )

    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    metrics = ControlledActionMetrics()

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
        metrics=metrics,
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
        metrics,
    )


def test_successful_controlled_action_records_executed_event(
    tmp_path,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
        metrics,
    ) = _build_test_environment(tmp_path)

    thread_id = "execution-audit-success"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Execution audit success test"
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

    payload = (
        initial_result["__interrupt__"][0]
        .value
    )

    request_id = payload["request_id"]

    state = graph.get_state(config)

    request_data = state.values[
        "approval_request"
    ]

    from src.orchestration.approval import (
        ApprovalRequest,
    )

    request = ApprovalRequest(
        request_id=request_data["request_id"],
        action_name=request_data["action_name"],
        arguments=request_data["arguments"],
        reason=request_data["reason"],
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
                    "Trusted approval recorded for "
                    "execution audit test"
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
            request_id
        )
    )

    executed_events = [
        event
        for event in events
        if event.event_type
        == (
            AuditEventType
            .CONTROLLED_ACTION_EXECUTED
        )
    ]

    assert len(executed_events) == 1

    event = executed_events[0]

    assert (
        event.actor_type
        == AuditActorType.SYSTEM
    )

    assert (
        event.source_component
        == "approval_graph"
    )

    assert event.request_id == request_id

    assert event.thread_id == thread_id

    assert (
        event.payload["action_name"]
        == "set_confidence_threshold"
    )

    assert event.payload["arguments"] == {
        "confidence_threshold": 0.95,
    }

    assert (
        event.payload["result"][
            "confidence_threshold"
        ]
        == 0.95
    )

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1
    assert snapshot.approved == 1

    assert snapshot.rejected == 0
    assert snapshot.denied == 0

    assert snapshot.executed == 1
    assert snapshot.failed == 0

    assert snapshot.execution_latency_count == 1
    assert snapshot.total_execution_latency_ms >= 0.0
    assert snapshot.average_execution_latency_ms >= 0.0
    assert snapshot.max_execution_latency_ms >= 0.0

def test_failed_controlled_action_records_failed_event(
    tmp_path,
    monkeypatch,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
        metrics,
    ) = _build_test_environment(tmp_path)

    thread_id = "execution-audit-failure"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Execution audit failure test"
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

    payload = (
        initial_result["__interrupt__"][0]
        .value
    )

    request_id = payload["request_id"]

    state = graph.get_state(config)

    request_data = state.values[
        "approval_request"
    ]

    from src.orchestration.approval import (
        ApprovalRequest,
    )

    request = ApprovalRequest(
        request_id=request_data["request_id"],
        action_name=request_data["action_name"],
        arguments=request_data["arguments"],
        reason=request_data["reason"],
    )

    trusted_approval_service.record_decision(
        request,
        approved=True,
        thread_id=thread_id,
    )

    def fail_threshold_update(
        confidence_threshold: float,
    ):
        raise RuntimeError(
            "Injected controlled-action failure"
        )

    monkeypatch.setattr(
        config_service,
        "set_confidence_threshold",
        fail_threshold_update,
    )

    with pytest.raises(
        RuntimeError,
        match="Injected controlled-action failure",
    ):
        graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "reason": (
                        "Trusted approval recorded "
                        "for failure audit test"
                    ),
                }
            ),
            config=config,
        )

    events = (
        audit_store.list_by_request_id(
            request_id
        )
    )

    failed_events = [
        event
        for event in events
        if event.event_type
        == (
            AuditEventType
            .CONTROLLED_ACTION_FAILED
        )
    ]

    assert len(failed_events) == 1

    event = failed_events[0]

    assert (
        event.actor_type
        == AuditActorType.SYSTEM
    )

    assert (
        event.source_component
        == "approval_graph"
    )

    assert event.request_id == request_id

    assert event.thread_id == thread_id

    assert (
        event.payload["action_name"]
        == "set_confidence_threshold"
    )

    assert event.payload["arguments"] == {
        "confidence_threshold": 0.95,
    }

    assert (
        event.payload["error_type"]
        == "RuntimeError"
    )

    assert (
        event.payload["error_message"]
        == "Injected controlled-action failure"
    )

    executed_events = [
        event
        for event in events
        if event.event_type
        == (
            AuditEventType
            .CONTROLLED_ACTION_EXECUTED
        )
    ]

    assert executed_events == []

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1
    assert snapshot.approved == 1

    assert snapshot.rejected == 0
    assert snapshot.denied == 0

    assert snapshot.executed == 0
    assert snapshot.failed == 1

    assert snapshot.execution_latency_count == 1
    assert snapshot.total_execution_latency_ms >= 0.0
    assert snapshot.average_execution_latency_ms >= 0.0
    assert snapshot.max_execution_latency_ms >= 0.0

def test_rejected_controlled_action_records_rejected_metric(
    tmp_path,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
        metrics,
    ) = _build_test_environment(tmp_path)

    thread_id = "execution-metrics-rejected"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Controlled-action rejection metrics test"
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

    state = graph.get_state(config)

    request_data = state.values[
        "approval_request"
    ]

    from src.orchestration.approval import (
        ApprovalRequest,
    )

    request = ApprovalRequest(
        request_id=request_data["request_id"],
        action_name=request_data["action_name"],
        arguments=request_data["arguments"],
        reason=request_data["reason"],
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
                    "Trusted human rejected request"
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

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1

    assert snapshot.approved == 0
    assert snapshot.rejected == 1
    assert snapshot.denied == 0

    assert snapshot.executed == 0
    assert snapshot.failed == 0

    assert snapshot.execution_latency_count == 0


def test_missing_trusted_approval_records_denied_metric(
    tmp_path,
):
    (
        graph,
        config_service,
        trusted_approval_service,
        audit_store,
        metrics,
    ) = _build_test_environment(tmp_path)

    thread_id = "execution-metrics-denied"

    config = _thread_config(thread_id)

    initial_result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Controlled-action denial metrics test"
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

    # Deliberately do NOT call:
    #
    # trusted_approval_service.record_decision(...)
    #
    # Resume data itself must not authorize the action.

    resumed_result = graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": (
                    "Untrusted resume data must not authorize"
                ),
            }
        ),
        config=config,
    )

    assert (
        resumed_result["action_status"]
        == "denied"
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1

    assert snapshot.approved == 0
    assert snapshot.rejected == 0
    assert snapshot.denied == 1

    assert snapshot.executed == 0
    assert snapshot.failed == 0

    assert snapshot.execution_latency_count == 0