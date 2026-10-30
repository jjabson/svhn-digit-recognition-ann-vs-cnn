from config.audit_event_store import (
    AuditEventStore,
)
from config.inference_config import (
    InferenceConfig,
)
from config.inference_config_service import (
    InferenceConfigService,
)
from agent.approval_graph import (
    build_approval_graph,
)
from src.orchestration.audit_event import (
    AuditActorType,
    AuditEventType,
)
from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
)
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
)
from src.observability.controlled_action_metrics import (
    ControlledActionMetrics,
)

def _build_audited_test_graph(
    tmp_path,
):
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

    return (
        graph,
        config_service,
        audit_store,
        metrics
    )

def _thread_config(
    thread_id: str,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def test_controlled_request_records_initial_audit_events(
    tmp_path,
):
    graph, config_service, audit_store, metrics = (
        _build_audited_test_graph(tmp_path)
    )

    thread_id = "audit-request-test"

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Audit test",
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=_thread_config(thread_id),
    )

    assert "__interrupt__" in result

    events = audit_store.list_all()

    assert len(events) == 2

    assert (
        events[0].event_type
        == AuditEventType.CONTROLLED_ACTION_REQUESTED
    )

    assert (
        events[1].event_type
        == AuditEventType.APPROVAL_REQUIRED
    )

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1

    assert snapshot.approved == 0
    assert snapshot.rejected == 0
    assert snapshot.denied == 0
    assert snapshot.executed == 0
    assert snapshot.failed == 0

    assert snapshot.execution_latency_count == 0


def test_initial_audit_events_share_request_id(
    tmp_path,
):
    graph, config_service, audit_store, metrics = (
        _build_audited_test_graph(tmp_path)
    )

    thread_id = "audit-correlation-test"

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Correlation test",
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=_thread_config(thread_id),
    )

    request_id = (
        result["__interrupt__"][0]
        .value["request_id"]
    )

    events = audit_store.list_by_request_id(
        request_id
    )

    assert len(events) == 2

    assert all(
        event.request_id == request_id
        for event in events
    )

    assert all(
        event.thread_id == thread_id
        for event in events
    )


def test_requested_event_records_mcp_actor(
    tmp_path,
):
    graph, config_service, audit_store, metrics = (
        _build_audited_test_graph(tmp_path)
    )

    thread_id = "audit-actor-test"

    graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Actor test",
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=_thread_config(thread_id),
    )

    events = audit_store.list_all()

    requested = events[0]
    approval_required = events[1]

    assert (
        requested.actor_type
        == AuditActorType.MCP_CLIENT
    )

    assert (
        requested.source_component
        == "mcp_server"
    )

    assert (
        approval_required.actor_type
        == AuditActorType.SYSTEM
    )

    assert (
        approval_required.source_component
        == "approval_graph"
    )



def test_interrupt_does_not_duplicate_initial_audit_events(
    tmp_path,
):
    graph, config_service, audit_store, metrics = (
        _build_audited_test_graph(tmp_path)
    )

    thread_id = "audit-interrupt-test"

    result = graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Interrupt test",
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
        },
        config=_thread_config(thread_id),
    )

    assert "__interrupt__" in result

    events = audit_store.list_all()

    assert len(events) == 2