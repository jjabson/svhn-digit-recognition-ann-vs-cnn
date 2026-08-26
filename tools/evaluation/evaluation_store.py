import json
import sqlite3
import numpy as np

from config.project_paths import DATABASE_FILE
from tools.evaluation.evaluate_model import (
    ClassMetrics,
    EvaluationComparison,
    EvaluationInfo,
    EvaluationProtocol,
)

from tools.evaluation.evaluate_model import (
    compare_evaluation_protocols,
)

def initialize_database() -> None:
    """
    Create the evaluation persistence database and required tables.
    """
    DATABASE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_name TEXT NOT NULL UNIQUE,
                dataset_size INTEGER NOT NULL,
                test_size REAL,
                random_state INTEGER,
                stratified INTEGER,
                independent_of_training INTEGER NOT NULL,
                description TEXT NOT NULL,

                accuracy REAL NOT NULL,
                macro_precision REAL NOT NULL,
                macro_recall REAL NOT NULL,
                macro_f1 REAL NOT NULL,
                weighted_precision REAL NOT NULL,
                weighted_recall REAL NOT NULL,
                weighted_f1 REAL NOT NULL,

                total_support INTEGER NOT NULL,
                correct_predictions INTEGER NOT NULL,
                incorrect_predictions INTEGER NOT NULL,

                confusion_matrix TEXT NOT NULL,

                created_at TEXT NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS class_metric (
                evaluation_run_id INTEGER NOT NULL,
                class_label INTEGER NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1_score REAL NOT NULL,
                support INTEGER NOT NULL,

                PRIMARY KEY (
                    evaluation_run_id,
                    class_label
                ),

                FOREIGN KEY (evaluation_run_id)
                    REFERENCES evaluation_run(id)
                    ON DELETE CASCADE
            )
            """
        )

def save_evaluation(
    evaluation: EvaluationInfo,
) -> None:
    """
    Persist one evaluation result and its per-class metrics.
    """
    protocol = evaluation.protocol

    confusion_matrix_json = json.dumps(
        evaluation.confusion_matrix.tolist()
    )

    with sqlite3.connect(DATABASE_FILE) as connection:
        existing = connection.execute(
            """
            SELECT id
            FROM evaluation_run
            WHERE protocol_name = ?
            """,
            (protocol.protocol_name,),
        ).fetchone()

        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_run (
                    protocol_name,
                    dataset_size,
                    test_size,
                    random_state,
                    stratified,
                    independent_of_training,
                    description,
                    accuracy,
                    macro_precision,
                    macro_recall,
                    macro_f1,
                    weighted_precision,
                    weighted_recall,
                    weighted_f1,
                    total_support,
                    correct_predictions,
                    incorrect_predictions,
                    confusion_matrix
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
                """,
                (
                    protocol.protocol_name,
                    protocol.dataset_size,
                    protocol.test_size,
                    protocol.random_state,
                    protocol.stratified,
                    protocol.independent_of_training,
                    protocol.description,
                    evaluation.accuracy,
                    evaluation.macro_precision,
                    evaluation.macro_recall,
                    evaluation.macro_f1,
                    evaluation.weighted_precision,
                    evaluation.weighted_recall,
                    evaluation.weighted_f1,
                    evaluation.total_support,
                    evaluation.correct_predictions,
                    evaluation.incorrect_predictions,
                    confusion_matrix_json,
                ),
            )

            evaluation_run_id = cursor.lastrowid

        else:
            evaluation_run_id = existing[0]

            connection.execute(
                """
                UPDATE evaluation_run
                SET
                    dataset_size = ?,
                    test_size = ?,
                    random_state = ?,
                    stratified = ?,
                    independent_of_training = ?,
                    description = ?,
                    accuracy = ?,
                    macro_precision = ?,
                    macro_recall = ?,
                    macro_f1 = ?,
                    weighted_precision = ?,
                    weighted_recall = ?,
                    weighted_f1 = ?,
                    total_support = ?,
                    correct_predictions = ?,
                    incorrect_predictions = ?,
                    confusion_matrix = ?,
                    created_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    protocol.dataset_size,
                    protocol.test_size,
                    protocol.random_state,
                    protocol.stratified,
                    protocol.independent_of_training,
                    protocol.description,
                    evaluation.accuracy,
                    evaluation.macro_precision,
                    evaluation.macro_recall,
                    evaluation.macro_f1,
                    evaluation.weighted_precision,
                    evaluation.weighted_recall,
                    evaluation.weighted_f1,
                    evaluation.total_support,
                    evaluation.correct_predictions,
                    evaluation.incorrect_predictions,
                    confusion_matrix_json,
                    evaluation_run_id,
                ),
            )

        connection.execute(
            """
            DELETE FROM class_metric
            WHERE evaluation_run_id = ?
            """,
            (evaluation_run_id,),
        )

        for metrics in evaluation.class_metrics:
            connection.execute(
                """
                INSERT INTO class_metric (
                    evaluation_run_id,
                    class_label,
                    precision,
                    recall,
                    f1_score,
                    support
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    metrics.class_label,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1_score,
                    metrics.support,
                ),
            )

def save_evaluation_comparison(
    comparison: EvaluationComparison,
) -> None:
    """
    Persist all evaluation protocols in an EvaluationComparison.
    """
    initialize_database()

    save_evaluation(
        comparison.historical
    )

    save_evaluation(
        comparison.original_test
    )

def load_evaluation(
    protocol_name: str,
) -> EvaluationInfo:
    """
    Load one persisted evaluation result from SQLite.
    """

    with sqlite3.connect(DATABASE_FILE) as connection:
        connection.row_factory = sqlite3.Row

        run_row = connection.execute(
            """
            SELECT *
            FROM evaluation_run
            WHERE protocol_name = ?
            """,
            (protocol_name,),
        ).fetchone()

        if run_row is None:
            raise ValueError(
                f"Evaluation protocol not found: {protocol_name}"
            )

        metric_rows = connection.execute(
            """
            SELECT
                class_label,
                precision,
                recall,
                f1_score,
                support
            FROM class_metric
            WHERE evaluation_run_id = ?
            ORDER BY class_label
            """,
            (run_row["id"],),
        ).fetchall()

    protocol = EvaluationProtocol(
        protocol_name=run_row["protocol_name"],
        dataset_size=run_row["dataset_size"],
        test_size=run_row["test_size"],
        random_state=run_row["random_state"],
        stratified=(
            bool(run_row["stratified"])
            if run_row["stratified"] is not None
            else None
        ),
        independent_of_training=bool(
            run_row["independent_of_training"]
        ),
        description=run_row["description"],
    )

    class_metrics = [
        ClassMetrics(
            class_label=row["class_label"],
            precision=row["precision"],
            recall=row["recall"],
            f1_score=row["f1_score"],
            support=row["support"],
        )
        for row in metric_rows
    ]

    confusion_matrix = np.array(
        json.loads(
            run_row["confusion_matrix"]
        ),
        dtype=int,
    )

    return EvaluationInfo(
        accuracy=run_row["accuracy"],
        macro_precision=run_row["macro_precision"],
        macro_recall=run_row["macro_recall"],
        macro_f1=run_row["macro_f1"],
        weighted_precision=run_row["weighted_precision"],
        weighted_recall=run_row["weighted_recall"],
        weighted_f1=run_row["weighted_f1"],
        total_support=run_row["total_support"],
        class_metrics=class_metrics,
        confusion_matrix=confusion_matrix,
        protocol=protocol,
    )

def load_evaluation_comparison() -> EvaluationComparison:
    """
    Load the persisted evaluation comparison from SQLite.
    """

    historical = load_evaluation(
        "Historical Stratified Holdout"
    )

    original_test = load_evaluation(
        "Original HDF5 Test Diagnostic"
    )

    return EvaluationComparison(
        historical=historical,
        original_test=original_test,
    )

def main() -> None:
    comparison = load_evaluation_comparison()

    print("Persisted Evaluation Comparison")
    print("-------------------------------")

    for evaluation in (
        comparison.historical,
        comparison.original_test,
    ):
        print(f"\n{evaluation.protocol.protocol_name}")
        print(
            f"Independent of Training: "
            f"{evaluation.is_independent_evaluation}"
        )
        print(
            f"Accuracy: "
            f"{evaluation.accuracy_percent:.2f}%"
        )
        print(
            f"Macro F1: "
            f"{evaluation.macro_f1 * 100:.2f}%"
        )
        print(
            f"Total Support: "
            f"{evaluation.total_support:,}"
        )
        print(
            f"Correct Predictions: "
            f"{evaluation.correct_predictions:,}"
        )
        print(
            f"Incorrect Predictions: "
            f"{evaluation.incorrect_predictions:,}"
        )
        print(
            f"Confusion Matrix Shape: "
            f"{evaluation.confusion_matrix.shape}"
        )
        print(
            f"Class Metrics: "
            f"{len(evaluation.class_metrics)}"
        )


if __name__ == "__main__":
    main()