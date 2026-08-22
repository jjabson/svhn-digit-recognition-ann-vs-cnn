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

from collections.abc import Callable


@dataclass
class EvaluationProtocol:
    protocol_name: str
    dataset_size: int
    test_size: float | None
    random_state: int | None
    stratified: bool | None
    independent_of_training: bool
    description: str

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

    @property
    def is_independent_evaluation(self) -> bool:
        return self.protocol.independent_of_training

    @property
    def accuracy_percent(self) -> float:
        return self.accuracy * 100

@dataclass
class EvaluationComparison:
    historical: EvaluationInfo
    original_test: EvaluationInfo

def load_original_test_data() -> tuple[
    np.ndarray,
    np.ndarray,
    EvaluationProtocol,
]:
    """
    Load the original held-out SVHN test split from the HDF5 dataset.
    """

    with h5py.File(DATA_FILE, "r") as h5_file:
        X_test = h5_file["X_test"][:]
        y_test = h5_file["y_test"][:]

    protocol = EvaluationProtocol(
        protocol_name="Original HDF5 Test Diagnostic",
        dataset_size=len(X_test),
        test_size=None,
        random_state=None,
        stratified=None,
        independent_of_training=False,
        description=(
            "Uses the original HDF5 test split. These samples were included "
            "in the combined dataset used by the historical CNN training split, "
            "so this evaluation is diagnostic rather than an independent "
            "held-out test."
        ),
    )

    return X_test, y_test, protocol

EvaluationDataLoader = Callable[
    [],
    tuple[
        np.ndarray,
        np.ndarray,
        EvaluationProtocol,
    ],
]

def evaluate_trained_model(
    data_loader: EvaluationDataLoader,
) -> EvaluationInfo:
    """
    Evaluate the trained SVHN CNN using the supplied evaluation protocol.
    """
    model = load_trained_model()

    X_eval, y_eval, protocol = data_loader()

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

def compare_evaluation_protocols() -> EvaluationComparison:
    """
    Evaluate the trained CNN using both supported evaluation protocols.
    """
    historical = evaluate_trained_model(
        load_historical_evaluation_data
    )

    original_test = evaluate_trained_model(
        load_original_test_data
    )

    return EvaluationComparison(
        historical=historical,
        original_test=original_test,
    )

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
        protocol_name="Historical Stratified Holdout",
        dataset_size=len(X_all),
        test_size=0.2,
        random_state=42,
        stratified=True,
        independent_of_training=True,
        description=(
            "Train, validation, and test datasets were combined and "
            "re-split into a stratified 80/20 training/evaluation split."
        ),
    )

    return X_eval, y_eval, protocol

def evaluate_trained_model(
    data_loader: EvaluationDataLoader,
) -> EvaluationInfo:
    """
    Evaluate the trained SVHN CNN using the supplied evaluation protocol.
    """
    model = load_trained_model()

    X_eval, y_eval, protocol = data_loader()

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
    comparison = compare_evaluation_protocols()

    print("Evaluation Protocol Comparison")
    print("------------------------------")

    for evaluation in (
            comparison.historical,
            comparison.original_test,
    ):
        print(f"\n{evaluation.protocol.protocol_name}")
        print(
            f"Independent of Training: "
            f"{evaluation.is_independent_evaluation}"
        )
        print(f"Accuracy: {evaluation.accuracy:.4f}")
        print(f"Accuracy Percent: {evaluation.accuracy_percent:.2f}%")
        print(f"Macro F1: {evaluation.macro_f1:.4f}")
        print(f"Total Support: {evaluation.total_support}")
        print(f"Correct Predictions: {evaluation.correct_predictions}")
        print(f"Incorrect Predictions: {evaluation.incorrect_predictions}")

if __name__ == "__main__":
    main()