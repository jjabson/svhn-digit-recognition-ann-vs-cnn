from pydantic import BaseModel

from tools.evaluation.evaluation_service import (
    EvaluationInsights,
    get_evaluation_insights,
    get_primary_evaluation,
)

from tools.evaluation.evaluate_model import ClassMetrics

from pydantic import BaseModel, ConfigDict

class ClassPerformanceResponse(BaseModel):
    """
    API representation of evaluation metrics for one digit class.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "digit": 0,
                "precision": 0.9787,
                "recall": 0.9571,
                "f1_score": 0.9678,
                "support": 2400,
            }
        }
    )

    digit: int
    precision: float
    recall: float
    f1_score: float
    support: int

class MisclassificationResponse(BaseModel):
    """
    API representation of a common classification error.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            }
        }
    )

    true_digit: int
    predicted_digit: int
    count: int

class EvaluationInsightsResponse(BaseModel):
    """
    API representation of derived evaluation intelligence.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "best_class": {
                    "digit": 0,
                    "precision": 0.9787,
                    "recall": 0.9571,
                    "f1_score": 0.9678,
                    "support": 2400,
                },
                "worst_class": {
                    "digit": 3,
                    "precision": 0.9615,
                    "recall": 0.9167,
                    "f1_score": 0.9386,
                    "support": 2400,
                },
                "most_common_misclassification": {
                    "true_digit": 3,
                    "predicted_digit": 5,
                    "count": 73,
                },
            }
        }
    )

    best_class: ClassPerformanceResponse
    worst_class: ClassPerformanceResponse
    most_common_misclassification: MisclassificationResponse

def class_metrics_to_response(
    metrics: ClassMetrics,
) -> ClassPerformanceResponse:
    """
    Convert internal class metrics into the public API representation.
    """
    return ClassPerformanceResponse(
        digit=metrics.class_label,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_score=metrics.f1_score,
        support=metrics.support,
    )

def evaluation_insights_to_response(
    insights: EvaluationInsights,
) -> EvaluationInsightsResponse:
    """
    Convert internal evaluation insights into the public API representation.
    """
    return EvaluationInsightsResponse(
        best_class=class_metrics_to_response(
            insights.best_class
        ),
        worst_class=class_metrics_to_response(
            insights.worst_class
        ),
        most_common_misclassification=(
            MisclassificationResponse(
                true_digit=(
                    insights.most_common_misclassification_true
                ),
                predicted_digit=(
                    insights.most_common_misclassification_predicted
                ),
                count=(
                    insights.most_common_misclassification_count
                ),
            )
        ),
    )

def main() -> None:
    evaluation = get_primary_evaluation()

    insights = get_evaluation_insights(
        evaluation
    )

    response = evaluation_insights_to_response(
        insights
    )

    print("Evaluation Insights API Response")
    print("--------------------------------")
    print(
        response.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()