import numpy as np

from dataclasses import dataclass

from tools.evaluation.evaluation_store import (
    load_evaluation_comparison,
)

from tools.evaluation.evaluate_model import (
    ClassMetrics,
    EvaluationComparison,
    EvaluationInfo,
)

@dataclass(frozen=True)
class EvaluationInsights:
    """
    Derived intelligence for a model evaluation.
    """

    best_class: ClassMetrics
    worst_class: ClassMetrics

    most_common_misclassification_true: int
    most_common_misclassification_predicted: int
    most_common_misclassification_count: int


def get_evaluation_comparison() -> EvaluationComparison:
    """
    Return the persisted evaluation comparison.
    """
    return load_evaluation_comparison()


def get_primary_evaluation() -> EvaluationInfo:
    """
    Return the independent historical holdout evaluation.
    """
    comparison = get_evaluation_comparison()
    return comparison.historical


def get_diagnostic_evaluation() -> EvaluationInfo:
    """
    Return the original HDF5 diagnostic evaluation.
    """
    comparison = get_evaluation_comparison()
    return comparison.original_test

def get_best_class(
    evaluation: EvaluationInfo,
) -> ClassMetrics:
    """
    Return the class with the highest F1 score.
    """
    return max(
        evaluation.class_metrics,
        key=lambda metrics: metrics.f1_score,
    )

def get_worst_class(
    evaluation: EvaluationInfo,
) -> ClassMetrics:
    """
    Return the class with the lowest F1 score.
    """
    return min(
        evaluation.class_metrics,
        key=lambda metrics: metrics.f1_score,
    )

def get_most_common_misclassification(
    evaluation: EvaluationInfo,
) -> tuple[int, int, int]:
    """
    Return the most common off-diagonal misclassification.

    Returns
    -------
    tuple[int, int, int]
        True class, predicted class, and number of occurrences.
    """
    matrix = evaluation.confusion_matrix.copy()

    np.fill_diagonal(
        matrix,
        0,
    )

    true_class, predicted_class = np.unravel_index(
        np.argmax(matrix),
        matrix.shape,
    )

    count = int(
        matrix[
            true_class,
            predicted_class,
        ]
    )

    return (
        int(true_class),
        int(predicted_class),
        count,
    )

def get_evaluation_insights(
    evaluation: EvaluationInfo,
) -> EvaluationInsights:
    """
    Derive high-level insights from an evaluation.
    """
    best_class = get_best_class(evaluation)
    worst_class = get_worst_class(evaluation)

    (
        true_class,
        predicted_class,
        count,
    ) = get_most_common_misclassification(
        evaluation
    )

    return EvaluationInsights(
        best_class=best_class,
        worst_class=worst_class,
        most_common_misclassification_true=true_class,
        most_common_misclassification_predicted=predicted_class,
        most_common_misclassification_count=count,
    )

def main() -> None:
    evaluation = get_primary_evaluation()
    insights = get_evaluation_insights(evaluation)

    print("Evaluation Intelligence")
    print("-----------------------")
    print(
        f"Best Class: "
        f"{insights.best_class.class_label} "
        f"(F1={insights.best_class.f1_score:.4f})"
    )
    print(
        f"Worst Class: "
        f"{insights.worst_class.class_label} "
        f"(F1={insights.worst_class.f1_score:.4f})"
    )
    print(
        "Most Common Misclassification: "
        f"{insights.most_common_misclassification_true} -> "
        f"{insights.most_common_misclassification_predicted} "
        f"({insights.most_common_misclassification_count})"
    )


if __name__ == "__main__":
    main()