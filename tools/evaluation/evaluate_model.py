from dataclasses import dataclass
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config.project_paths import DATA_FILE

from src.preprocessing import preprocess_image_batch

from tools.model.inspect_cnn import load_trained_model

@dataclass
class EvaluationProtocol:
    dataset_size: int
    test_size: float
    random_state: int
    stratified: bool


@dataclass
class ClassMetrics:
    class_label: int
    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass
class EvaluationInfo:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_support: int
    class_metrics: list[ClassMetrics]
    confusion_matrix: np.ndarray
    protocol: EvaluationProtocol

    @property
    def correct_predictions(self) -> int:
        """
        Return the number of correctly classified samples.
        """
        return int(np.trace(self.confusion_matrix))

    @property
    def incorrect_predictions(self) -> int:
        """
        Return the number of incorrectly classified samples.
        """
        return self.total_support - self.correct_predictions

def load_historical_evaluation_data() -> tuple[
    np.ndarray,
    np.ndarray,
    EvaluationProtocol,
]:
    """
    Reproduce the historical CNN evaluation split used in the notebook.

    The original train, validation, and test datasets are combined and then
    split into a new stratified 80/20 train/evaluation partition using
    random_state=42.
    """

    with h5py.File(DATA_FILE, "r") as h5_file:
        X_train = h5_file["X_train"][:]
        X_val = h5_file["X_val"][:]
        X_test = h5_file["X_test"][:]

        y_train = h5_file["y_train"][:]
        y_val = h5_file["y_val"][:]
        y_test = h5_file["y_test"][:]

    X_all = np.concatenate(
        [X_train, X_val, X_test],
        axis=0,
    )

    y_all = np.concatenate(
        [y_train, y_val, y_test],
        axis=0,
    )

    _, X_eval, _, y_eval = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all,
    )

    protocol = EvaluationProtocol(
        dataset_size=len(X_all),
        test_size=0.2,
        random_state=42,
        stratified=True,
    )

    return X_eval, y_eval, protocol

def evaluate_trained_model() -> EvaluationInfo:
    """
    Evaluate the trained SVHN CNN using the historical notebook protocol.
    """
    model = load_trained_model()

    X_eval, y_eval, protocol = load_historical_evaluation_data()

    X_eval_processed = preprocess_image_batch(
        X_eval
    )

    y_pred_probs = model.predict(
        X_eval_processed,
        verbose=0,
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1,
    )

    return evaluate_predictions(
        y_true=y_eval,
        y_pred=y_pred,
        protocol=protocol,
    )

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protocol: EvaluationProtocol,
) -> EvaluationInfo:
    """
    Evaluate predicted class labels against the true class labels.

    Args:
        y_true:
            Ground-truth class labels.

        y_pred:
            Predicted class labels.

        protocol:
            Description of the evaluation protocol used to produce
            the evaluation dataset.

    Returns:
        Structured evaluation results.
    """

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    class_metrics = []

    for class_label in sorted(np.unique(y_true)):
        metrics = report[str(class_label)]

        class_metrics.append(
            ClassMetrics(
                class_label=int(class_label),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1_score=float(metrics["f1-score"]),
                support=int(metrics["support"]),
            )
        )

    macro_metrics = report["macro avg"]
    weighted_metrics = report["weighted avg"]

    return EvaluationInfo(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_metrics["precision"]),
        macro_recall=float(macro_metrics["recall"]),
        macro_f1=float(macro_metrics["f1-score"]),
        weighted_precision=float(weighted_metrics["precision"]),
        weighted_recall=float(weighted_metrics["recall"]),
        weighted_f1=float(weighted_metrics["f1-score"]),
        total_support=len(y_true),
        class_metrics=class_metrics,
        confusion_matrix=confusion_matrix(
            y_true,
            y_pred,
        ),
        protocol=protocol,
    )

def main() -> None:
    """
    Verify the evaluation domain model with a small controlled example.
    """
    evaluation = evaluate_trained_model()

    print("Historical CNN Evaluation")
    print("-------------------------")
    print(f"Accuracy: {evaluation.accuracy:.4f}")
    print(f"Macro Precision: {evaluation.macro_precision:.4f}")
    print(f"Macro Recall: {evaluation.macro_recall:.4f}")
    print(f"Macro F1: {evaluation.macro_f1:.4f}")
    print(f"Weighted F1: {evaluation.weighted_f1:.4f}")
    print(f"Total Support: {evaluation.total_support}")
    print(f"Correct Predictions: {evaluation.correct_predictions}")
    print(f"Incorrect Predictions: {evaluation.incorrect_predictions}")

    # X_eval, y_eval, protocol = load_historical_evaluation_data()
    #
    # X_eval_processed = preprocess_image_batch(
    #     X_eval
    # )
    #
    # print("\nHistorical Evaluation Data")
    # print("--------------------------")
    # print(f"Raw shape: {X_eval.shape}")
    # print(f"Processed shape: {X_eval_processed.shape}")
    # print(f"Label shape: {y_eval.shape}")
    # print(f"Data type: {X_eval_processed.dtype}")
    # print(f"Pixel minimum: {X_eval_processed.min():.4f}")
    # print(f"Pixel maximum: {X_eval_processed.max():.4f}")
    # print(f"Combined dataset size: {protocol.dataset_size}")
    # print(f"Test size: {protocol.test_size}")
    # print(f"Random state: {protocol.random_state}")
    # print(f"Stratified: {protocol.stratified}")

    # y_true = np.array([0, 1, 2, 0, 1, 2])
    # y_pred = np.array([0, 1, 2, 0, 2, 2])
    #
    # protocol = EvaluationProtocol(
    #     dataset_size=len(y_true),
    #     test_size=1.0,
    #     random_state=42,
    #     stratified=False,
    # )
    #
    # evaluation = evaluate_predictions(
    #     y_true=y_true,
    #     y_pred=y_pred,
    #     protocol=protocol,
    # )
    #
    # print("Evaluation Summary")
    # print("------------------")
    # print(f"Accuracy: {evaluation.accuracy:.4f}")
    # print(f"Macro Precision: {evaluation.macro_precision:.4f}")
    # print(f"Macro Recall: {evaluation.macro_recall:.4f}")
    # print(f"Macro F1: {evaluation.macro_f1:.4f}")
    # print(f"Weighted F1: {evaluation.weighted_f1:.4f}")
    # print(f"Total Support: {evaluation.total_support}")
    # print(
    #     f"Correct Predictions: "
    #     f"{evaluation.correct_predictions}"
    # )
    # print(
    #     f"Incorrect Predictions: "
    #     f"{evaluation.incorrect_predictions}"
    # )
    #
    # print("\nPer-Class Metrics")
    # print("-----------------")
    #
    # for metrics in evaluation.class_metrics:
    #     print(
    #         f"Class {metrics.class_label}: "
    #         f"precision={metrics.precision:.4f}, "
    #         f"recall={metrics.recall:.4f}, "
    #         f"f1={metrics.f1_score:.4f}, "
    #         f"support={metrics.support}"
    #     )
    #
    # print("\nConfusion Matrix")
    # print("----------------")
    # print(evaluation.confusion_matrix)


if __name__ == "__main__":
    main()