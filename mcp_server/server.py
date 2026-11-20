import base64
import binascii
from uuid import uuid4
import os
import argparse

from mcp_server.schemas import PredictionDecisionResult
from src.orchestration.errors import InvalidInferenceInputError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import (
    TransportSecuritySettings,
)
from agent.approval_graph import build_approval_graph
from config.inference_config import DEFAULT_INFERENCE_CONFIG
from config.inference_config_service import InferenceConfigService
from config.inference_config_store import InferenceConfigStore
from config.approval_checkpoint import (
    create_approval_checkpointer,
)

from mcp_server.schemas import (
    ClassMetricsResult,
    ControlledActionMetricsResult,
    EvaluationInsightsResult,
    EvaluationSummaryResult,
    InferenceConfigResult,
    InferenceMetricsResult,
    ModelSummaryResult,
)
from src.orchestration.service_factory import (
    create_orchestrated_inference_service,
)

from tools.evaluation.evaluation_service import (
    get_class_metrics,
    get_evaluation_insights,
    get_primary_evaluation,
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
from mcp_server.http_auth import (
    McpHttpAuthConfig,
    StaticBearerTokenVerifier,
    create_auth_settings,
)
from config.audit_event_store import (
    AuditEventStore,
)
from src.orchestration.audit_event import (
    AuditActorType,
)
from src.observability.metrics import InferenceMetrics
from src.observability.controlled_action_metrics import (
    ControlledActionMetrics,
)
from config.fallback_inference_config import (
    FallbackInferenceConfig,
)

MCP_HTTP_AUTH_CONFIG = (
    McpHttpAuthConfig.from_environment()
)

FALLBACK_INFERENCE_CONFIG = (
    FallbackInferenceConfig.from_environment()
)

def create_mcp_server() -> MCPServer:
    if not MCP_HTTP_AUTH_CONFIG.enabled:
        return MCPServer(
            "svhn-ml-system",
        )

    assert (
        MCP_HTTP_AUTH_CONFIG.bearer_token
        is not None
    )
    assert (
        MCP_HTTP_AUTH_CONFIG.resource_server_url
        is not None
    )

    token_verifier = StaticBearerTokenVerifier(
        expected_token=(
            MCP_HTTP_AUTH_CONFIG.bearer_token
        ),
        resource_server_url=(
            MCP_HTTP_AUTH_CONFIG
            .resource_server_url
        ),
        required_scope=(
            MCP_HTTP_AUTH_CONFIG
            .required_scope
        ),
    )

    return MCPServer(
        "svhn-ml-system",
        token_verifier=token_verifier,
        auth=create_auth_settings(
            MCP_HTTP_AUTH_CONFIG
        ),
    )


mcp = create_mcp_server()

RUNTIME_INSTANCE_ID = str(uuid4())

inference_metrics = InferenceMetrics()

controlled_action_metrics = (
    ControlledActionMetrics()
)

inference_config_store = InferenceConfigStore()

inference_config_service = InferenceConfigService(
    DEFAULT_INFERENCE_CONFIG,
    store=inference_config_store,
)

inference_service = create_orchestrated_inference_service(
    config_service=inference_config_service,
    metrics=inference_metrics,
    fallback_config=FALLBACK_INFERENCE_CONFIG,
)

approval_checkpointer = create_approval_checkpointer()

approval_artifact_store = ApprovalArtifactStore()

audit_event_store = AuditEventStore()

trusted_approval_verifier = TrustedApprovalVerifier(
    approval_artifact_store,
    trusted_issuers={
        DEFAULT_TRUSTED_APPROVAL_ISSUER,
    },
)

approval_graph = build_approval_graph(
    config_service=inference_config_service,
    artifact_store=approval_artifact_store,
    trusted_approval_verifier=trusted_approval_verifier,
    checkpointer=approval_checkpointer,
    metrics=controlled_action_metrics,
)

@mcp.tool(
    name="get_runtime_debug",
    description=(
        "Return runtime identity and active configuration details "
        "for diagnostic verification."
    ),
)
def get_runtime_debug() -> dict:
    config = inference_config_service.get_config()

    return {
        "runtime_instance_id": RUNTIME_INSTANCE_ID,
        "pid": os.getpid(),
        "config_service_id": id(inference_config_service),
        "approval_graph_id": id(approval_graph),
        "approval_checkpointer_id": id(approval_checkpointer),
        "confidence_threshold": config.confidence_threshold,
    }

@mcp.tool(
    name="get_inference_config",
    description="Return the active inference configuration used by the serving system.",
    structured_output=True,
)
def get_inference_config() -> InferenceConfigResult:
    """
    Return the active configuration of the inference service.
    """
    config = inference_service.get_config()

    return InferenceConfigResult(
        model_name=str(config["model_name"]),
        confidence_threshold=float(config["confidence_threshold"]),
    )

@mcp.tool(
    name="get_evaluation_summary",
    description="Return summary metrics for the primary model evaluation.",
    structured_output=True,
)
def get_evaluation_summary() -> EvaluationSummaryResult:
    """
    Return summary metrics for the primary model evaluation.
    """
    evaluation = get_primary_evaluation()

    return EvaluationSummaryResult(
        accuracy=evaluation.accuracy,
        macro_precision=evaluation.macro_precision,
        macro_recall=evaluation.macro_recall,
        macro_f1=evaluation.macro_f1,
        weighted_precision=evaluation.weighted_precision,
        weighted_recall=evaluation.weighted_recall,
        weighted_f1=evaluation.weighted_f1,
        total_support=evaluation.total_support,
        correct_predictions=evaluation.correct_predictions,
        incorrect_predictions=evaluation.incorrect_predictions,
        is_independent_evaluation=evaluation.is_independent_evaluation,
    )

@mcp.tool(
    name="get_evaluation_insights",
    description="Return derived insights for the primary model evaluation.",
    structured_output=True,
)
def get_primary_evaluation_insights() -> EvaluationInsightsResult:
    """
    Return derived insights for the primary model evaluation.
    """
    evaluation = get_primary_evaluation()
    insights = get_evaluation_insights(evaluation)

    return EvaluationInsightsResult(
        best_class=ClassMetricsResult(
            class_label=insights.best_class.class_label,
            precision=insights.best_class.precision,
            recall=insights.best_class.recall,
            f1_score=insights.best_class.f1_score,
            support=insights.best_class.support,
        ),
        worst_class=ClassMetricsResult(
            class_label=insights.worst_class.class_label,
            precision=insights.worst_class.precision,
            recall=insights.worst_class.recall,
            f1_score=insights.worst_class.f1_score,
            support=insights.worst_class.support,
        ),
        most_common_misclassification_true=(
            insights.most_common_misclassification_true
        ),
        most_common_misclassification_predicted=(
            insights.most_common_misclassification_predicted
        ),
        most_common_misclassification_count=(
            insights.most_common_misclassification_count
        ),
    )

@mcp.tool(
    name="get_digit_metrics",
    description="Return evaluation metrics for a specific digit in the primary model evaluation.",
    structured_output=True,
)
def get_digit_metrics(digit: int) -> ClassMetricsResult:
    """
    Return evaluation metrics for one digit.
    """
    evaluation = get_primary_evaluation()
    metrics = get_class_metrics(evaluation, digit)

    if metrics is None:
        raise ToolError(
            f"No evaluation metrics found for digit {digit}."
        )

    return ClassMetricsResult(
        class_label=metrics.class_label,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        support=metrics.support,
    )

@mcp.tool(
    name="get_model_summary",
    description="Return an architecture summary for the model currently used for inference.",
    structured_output=True,
)
def get_model_summary() -> ModelSummaryResult:
    """
    Return an architecture summary for the active serving model.
    """
    architecture = inference_service.get_model_architecture()

    return ModelSummaryResult(
        model_name=architecture.model_name,
        input_shape=architecture.input_shape,
        output_shape=architecture.output_shape,
        total_parameters=architecture.total_parameters,
        number_of_layers=architecture.number_of_layers,
    )

@mcp.tool(
    name="predict_digit",
    description=(
        "Classify a base64-encoded SVHN-style digit image using "
        "the production inference orchestration service."
    ),
    structured_output=True,
)
def predict_digit(image_data: str) -> PredictionDecisionResult:
    """
    Classify a base64-encoded digit image through the
    production inference orchestration service.
    """
    try:
        image_bytes = base64.b64decode(
            image_data,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ToolError(
            "image_data must be valid base64-encoded image data."
        ) from exc

    if not image_bytes:
        raise ToolError(
            "image_data must not decode to an empty image."
        )

    try:
        decision = inference_service.predict(image_bytes)
    except InvalidInferenceInputError as exc:
        raise ToolError(str(exc)) from exc

    return PredictionDecisionResult(
        selected_model=decision.selected_model,
        predicted_digit=decision.predicted_digit,
        confidence=decision.confidence,
        status=decision.status.value,
        decision_reason=decision.decision_reason,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        review_required=decision.review_required,
    )

@mcp.tool()
def set_confidence_threshold(
    confidence_threshold: float,
) -> dict:
    """
    Request a controlled update to the active inference confidence threshold.

    This operation does not change runtime configuration immediately.
    It starts a human-approval workflow and returns approval context.
    """

    validate_confidence_threshold_request(
        confidence_threshold
    )

    thread_id = f"threshold-{uuid4()}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = approval_graph.invoke(
        {
            "requested_threshold": confidence_threshold,
            "reason": (
                "MCP request to change inference "
                "confidence threshold"
            ),
            "thread_id": thread_id,
            "request_actor_type": (
                AuditActorType.MCP_CLIENT.value
            ),
            "request_source_component": (
                "mcp_server"
            ),
            "request_actor_id": None,
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        raise RuntimeError(
            "Expected approval interrupt was not produced"
        )

    payload = interrupts[0].value

    return {
        "status": "approval_required",
        "thread_id": thread_id,
        "request_id": payload["request_id"],
        "action_name": payload["action_name"],
        "arguments": payload["arguments"],
        "current_confidence_threshold": (
            payload["current_confidence_threshold"]
        ),
        "message": payload["message"],
    }

@mcp.tool()
def get_confidence_threshold_approval_status(
    thread_id: str,
) -> dict:
    """
    Inspect a confidence-threshold approval workflow.

    This tool is read-only. It cannot approve, reject,
    resume, or authorize controlled execution.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    snapshot = approval_graph.get_state(
        config
    )

    state = snapshot.values

    if (
        not state
        or "approval_request" not in state
    ):
        raise ToolError(
            "No confidence-threshold approval "
            f"workflow found for thread_id "
            f"{thread_id}."
        )

    request_data = state["approval_request"]

    request_id = (
        request_data["request_id"]
    )

    artifact = (
        approval_artifact_store
        .get_by_request_id(request_id)
    )

    action_status = state.get(
        "action_status"
    )

    if action_status is not None:
        status = action_status

    elif artifact is None:
        status = "approval_required"

    elif artifact.approved:
        status = (
            "approved_pending_execution"
        )

    else:
        status = (
            "rejected_pending_execution"
        )

    return {
        "status": status,
        "thread_id": thread_id,
        "request_id": request_id,
        "action_name": (
            request_data["action_name"]
        ),
        "arguments": (
            request_data["arguments"]
        ),
        "confidence_threshold": (
            inference_config_service
            .get_config()
            .confidence_threshold
        ),
    }

@mcp.tool(
    name="get_inference_metrics",
    description=(
        "Return process-local operational metrics for "
        "orchestrated inference."
    ),
    structured_output=True,
)
def get_inference_metrics() -> InferenceMetricsResult:
    """
    Return the current process-local inference metrics snapshot.

    These metrics are observational only and reset when the
    MCP server process restarts.
    """
    snapshot = inference_metrics.snapshot()

    return InferenceMetricsResult(
        total_attempts=snapshot.total_attempts,
        accepted=snapshot.accepted,
        uncertain=snapshot.uncertain,
        failed=snapshot.failed,
        fallback_used_count=snapshot.fallback_used_count,
        fallback_rate=snapshot.fallback_rate,
        total_latency_ms=snapshot.total_latency_ms,
        average_latency_ms=snapshot.average_latency_ms,
        max_latency_ms=snapshot.max_latency_ms,
    )

@mcp.tool(
    name="get_controlled_action_metrics",
    description=(
        "Return process-local operational metrics for "
        "controlled-action workflows."
    ),
    structured_output=True,
)
def get_controlled_action_metrics() -> ControlledActionMetricsResult:
    """
    Return the current process-local controlled-action metrics
    snapshot.

    These metrics are observational only and reset when the
    MCP server process restarts.
    """
    snapshot = controlled_action_metrics.snapshot()

    return ControlledActionMetricsResult(
        requested=snapshot.requested,
        approval_required=snapshot.approval_required,
        approved=snapshot.approved,
        rejected=snapshot.rejected,
        denied=snapshot.denied,
        executed=snapshot.executed,
        failed=snapshot.failed,
        execution_latency_count=(
            snapshot.execution_latency_count
        ),
        total_execution_latency_ms=(
            snapshot.total_execution_latency_ms
        ),
        average_execution_latency_ms=(
            snapshot.average_execution_latency_ms
        ),
        max_execution_latency_ms=(
            snapshot.max_execution_latency_ms
        ),
    )


def validate_http_bind_security(
    *,
    host: str,
    auth_enabled: bool,
) -> None:
    loopback_hosts = {
        "127.0.0.1",
        "localhost",
        "::1",
    }

    if (
        host not in loopback_hosts
        and not auth_enabled
    ):
        raise ValueError(
            "Refusing to start unauthenticated "
            "Streamable HTTP MCP server on "
            f"non-loopback host {host!r}. "
            "Enable MCP HTTP authentication "
            "before using a network-reachable bind."
        )

def validate_confidence_threshold_request(
    confidence_threshold: float,
) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ToolError(
            "confidence_threshold must be between 0.0 and 1.0 inclusive."
        )

def main() -> None:
    """
    Run the SVHN ML system MCP server.

    Default transport remains stdio for local MCP consumers
    such as Claude Code and VS Code.

    Streamable HTTP may be selected for network MCP clients
    such as n8n.
    """
    parser = argparse.ArgumentParser(
        description="Run the SVHN MCP server."
    )

    parser.add_argument(
        "--transport",
        choices=[
            "stdio",
            "streamable-http",
        ],
        default="stdio",
        help=(
            "MCP transport. Default: stdio."
        ),
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Host for Streamable HTTP transport. "
            "Default: 127.0.0.1."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help=(
            "Port for Streamable HTTP transport. "
            "Default: 8001."
        ),
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(
            transport="stdio",
        )
        return

    try:
        validate_http_bind_security(
            host=args.host,
            auth_enabled=(
                MCP_HTTP_AUTH_CONFIG.enabled
            ),
        )
    except ValueError as exc:
        parser.error(str(exc))

    security = TransportSecuritySettings(
        allowed_hosts=[
            "127.0.0.1",
            "127.0.0.1:*",
            "localhost",
            "localhost:*",
            "host.docker.internal",
            "host.docker.internal:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://host.docker.internal:*",
        ],
    )

    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        streamable_http_path="/mcp",
        transport_security=security,
    )


if __name__ == "__main__":
    main()