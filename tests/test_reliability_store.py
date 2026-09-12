import sqlite3
import pytest

from tools.reliability.confidence_analysis import (
    ConfidenceObservation,
)
from tools.reliability.reliability_store import (
    initialize_reliability_database,
    save_reliability_observations,
    load_reliability_observations, _connect,
)


def test_initialize_reliability_database_creates_table(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with sqlite3.connect(database_file) as connection:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'reliability_observation'
            """
        ).fetchone()

    assert row is not None

def test_save_reliability_observations_persists_rows(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with sqlite3.connect(database_file) as connection:
        connection.execute(
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
                "Test Protocol",
                2,
                0.2,
                42,
                1,
                1,
                "Test evaluation protocol.",
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                2,
                1,
                1,
                "[[1, 0], [1, 0]]",
            ),
        )

    observations = [
        ConfidenceObservation(
            true_label=3,
            predicted_label=3,
            confidence=0.95,
        ),
        ConfidenceObservation(
            true_label=5,
            predicted_label=8,
            confidence=0.72,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=observations,
        database_file=database_file,
    )

    with sqlite3.connect(database_file) as connection:
        rows = connection.execute(
            """
            SELECT
                observation_index,
                true_label,
                predicted_label,
                confidence
            FROM reliability_observation
            ORDER BY observation_index
            """
        ).fetchall()

    assert rows == [
        (0, 3, 3, 0.95),
        (1, 5, 8, 0.72),
    ]

def test_save_reliability_observations_replaces_existing_rows(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with sqlite3.connect(database_file) as connection:
        connection.execute(
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
                "Test Protocol",
                2,
                0.2,
                42,
                1,
                1,
                "Test evaluation protocol.",
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                2,
                1,
                1,
                "[[1, 0], [1, 0]]",
            ),
        )

    first_observations = [
        ConfidenceObservation(
            true_label=3,
            predicted_label=3,
            confidence=0.95,
        ),
        ConfidenceObservation(
            true_label=5,
            predicted_label=8,
            confidence=0.72,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=first_observations,
        database_file=database_file,
    )

    replacement_observations = [
        ConfidenceObservation(
            true_label=7,
            predicted_label=7,
            confidence=0.99,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=replacement_observations,
        database_file=database_file,
    )

    with sqlite3.connect(database_file) as connection:
        rows = connection.execute(
            """
            SELECT
                observation_index,
                true_label,
                predicted_label,
                confidence
            FROM reliability_observation
            ORDER BY observation_index
            """
        ).fetchall()

    assert rows == [
        (0, 7, 7, 0.99),
    ]

def test_load_reliability_observations_returns_domain_objects(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with sqlite3.connect(database_file) as connection:
        connection.execute(
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
                "Test Protocol",
                2,
                0.2,
                42,
                1,
                1,
                "Test evaluation protocol.",
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                2,
                1,
                1,
                "[[1, 0], [1, 0]]",
            ),
        )

    observations = [
        ConfidenceObservation(
            true_label=3,
            predicted_label=3,
            confidence=0.95,
        ),
        ConfidenceObservation(
            true_label=5,
            predicted_label=8,
            confidence=0.72,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=observations,
        database_file=database_file,
    )

    loaded_observations = load_reliability_observations(
        protocol_name="Test Protocol",
        database_file=database_file,
    )

    assert loaded_observations == observations

def test_save_reliability_observations_rejects_unknown_protocol(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    observations = [
        ConfidenceObservation(
            true_label=3,
            predicted_label=3,
            confidence=0.95,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Evaluation protocol not found: Missing Protocol",
    ):
        save_reliability_observations(
            protocol_name="Missing Protocol",
            observations=observations,
            database_file=database_file,
        )

def test_load_reliability_observations_rejects_unknown_protocol(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    with pytest.raises(
        ValueError,
        match="Evaluation protocol not found: Missing Protocol",
    ):
        load_reliability_observations(
            protocol_name="Missing Protocol",
            database_file=database_file,
        )

def test_reliability_observation_requires_existing_evaluation_run(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with _connect(database_file) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO reliability_observation (
                    evaluation_run_id,
                    observation_index,
                    true_label,
                    predicted_label,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    999,
                    0,
                    3,
                    3,
                    0.95,
                ),
            )

def test_reliability_store_round_trip_replaces_and_loads(
    tmp_path,
):
    database_file = tmp_path / "test_reliability.db"

    initialize_reliability_database(
        database_file=database_file
    )

    with sqlite3.connect(database_file) as connection:
        connection.execute(
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
                "Test Protocol",
                2,
                0.2,
                42,
                1,
                1,
                "Test evaluation protocol.",
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                2,
                1,
                1,
                "[[1, 0], [1, 0]]",
            ),
        )

    initial_observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.91,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=3,
            confidence=0.66,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=initial_observations,
        database_file=database_file,
    )

    replacement_observations = [
        ConfidenceObservation(
            true_label=7,
            predicted_label=7,
            confidence=0.99,
        ),
        ConfidenceObservation(
            true_label=8,
            predicted_label=6,
            confidence=0.88,
        ),
    ]

    save_reliability_observations(
        protocol_name="Test Protocol",
        observations=replacement_observations,
        database_file=database_file,
    )

    loaded_observations = load_reliability_observations(
        protocol_name="Test Protocol",
        database_file=database_file,
    )

    assert loaded_observations == replacement_observations