import sqlite3
from pathlib import Path

from config.project_paths import DATABASE_FILE
from tools.evaluation.evaluation_store import (
    initialize_database,
)

from tools.reliability.confidence_analysis import (
    ConfidenceObservation,
)

def _connect(
    database_file: Path,
) -> sqlite3.Connection:
    connection = sqlite3.connect(database_file)
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )
    return connection


def initialize_reliability_database(
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Create reliability persistence tables.
    """
    initialize_database(database_file)

    with _connect(database_file) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS reliability_observation (
                evaluation_run_id INTEGER NOT NULL,
                observation_index INTEGER NOT NULL,
                true_label INTEGER NOT NULL,
                predicted_label INTEGER NOT NULL,
                confidence REAL NOT NULL,

                PRIMARY KEY (
                    evaluation_run_id,
                    observation_index
                ),

                FOREIGN KEY (evaluation_run_id)
                    REFERENCES evaluation_run(id)
                    ON DELETE CASCADE
            )
            """
        )

def save_reliability_observations(
    protocol_name: str,
    observations: list[ConfidenceObservation],
    database_file: Path = DATABASE_FILE,
) -> None:
    """
    Persist reliability observations for one evaluation protocol.
    """
    initialize_reliability_database(database_file)

    with sqlite3.connect(database_file) as connection:
        run_row = connection.execute(
            """
            SELECT id
            FROM evaluation_run
            WHERE protocol_name = ?
            """,
            (protocol_name,),
        ).fetchone()

        if run_row is None:
            raise ValueError(
                f"Evaluation protocol not found: {protocol_name}"
            )

        evaluation_run_id = run_row[0]

        connection.execute(
            """
            DELETE FROM reliability_observation
            WHERE evaluation_run_id = ?
            """,
            (evaluation_run_id,),
        )

        connection.executemany(
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
            [
                (
                    evaluation_run_id,
                    observation_index,
                    observation.true_label,
                    observation.predicted_label,
                    observation.confidence,
                )
                for observation_index, observation
                in enumerate(observations)
            ],
        )

def load_reliability_observations(
    protocol_name: str,
    database_file: Path = DATABASE_FILE,
) -> list[ConfidenceObservation]:
    """
    Load persisted reliability observations for one evaluation protocol.
    """
    initialize_reliability_database(database_file)

    with sqlite3.connect(database_file) as connection:
        connection.row_factory = sqlite3.Row

        run_row = connection.execute(
            """
            SELECT id
            FROM evaluation_run
            WHERE protocol_name = ?
            """,
            (protocol_name,),
        ).fetchone()

        if run_row is None:
            raise ValueError(
                f"Evaluation protocol not found: {protocol_name}"
            )

        rows = connection.execute(
            """
            SELECT
                true_label,
                predicted_label,
                confidence
            FROM reliability_observation
            WHERE evaluation_run_id = ?
            ORDER BY observation_index
            """,
            (run_row["id"],),
        ).fetchall()

    return [
        ConfidenceObservation(
            true_label=row["true_label"],
            predicted_label=row["predicted_label"],
            confidence=row["confidence"],
        )
        for row in rows
    ]