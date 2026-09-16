from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfigResult:
    """
    MCP result contract for the active inference configuration.
    """

    model_name: str
    confidence_threshold: float

@dataclass(frozen=True)
class EvaluationSummaryResult:
    """
    MCP result contract for the primary model evaluation summary.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_support: int
    correct_predictions: int
    incorrect_predictions: int
    is_independent_evaluation: bool

@dataclass(frozen=True)
class ClassMetricsResult:
    """
    MCP result contract for per-class evaluation metrics.
    """

    class_label: int
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class EvaluationInsightsResult:
    """
    MCP result contract for derived evaluation insights.
    """

    best_class: ClassMetricsResult
    worst_class: ClassMetricsResult
    most_common_misclassification_true: int
    most_common_misclassification_predicted: int
    most_common_misclassification_count: int

@dataclass(frozen=True)
class ModelSummaryResult:
    """
    MCP result contract for the serving model architecture summary.
    """

    model_name: str
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    total_parameters: int
    number_of_layers: int

@dataclass(frozen=True)
class PredictionDecisionResult:
    """
    MCP result contract for an orchestrated digit prediction decision.
    """

    selected_model: str | None
    predicted_digit: int | None
    confidence: float | None
    status: str
    decision_reason: str | None
    fallback_used: bool
    fallback_reason: str | None
    review_required: bool