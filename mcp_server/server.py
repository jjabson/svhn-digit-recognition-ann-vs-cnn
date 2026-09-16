import base64
import binascii

from mcp_server.schemas import PredictionDecisionResult
from src.orchestration.errors import InvalidInferenceInputError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from mcp_server.schemas import (
    ClassMetricsResult,
    EvaluationInsightsResult,
    EvaluationSummaryResult,
    InferenceConfigResult,
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

mcp = MCPServer("svhn-ml-system")

inference_service = create_orchestrated_inference_service()

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


def main() -> None:
    """Run the SVHN ML system MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()