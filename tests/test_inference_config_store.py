from config.inference_config import InferenceConfig
from config.inference_config_store import InferenceConfigStore


def test_load_returns_none_when_no_config_exists(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    assert store.load() is None


def test_save_and_load_config(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    config = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.95,
    )

    store.save(config)

    loaded = store.load()

    assert loaded is not None
    assert loaded.model_name == "cnn"
    assert loaded.confidence_threshold == 0.95


def test_save_replaces_existing_config(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    store.save(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    store.save(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.95,
        )
    )

    loaded = store.load()

    assert loaded is not None
    assert loaded.confidence_threshold == 0.95


def test_loaded_config_is_validated(tmp_path):
    database_file = tmp_path / "runtime_state.db"

    store = InferenceConfigStore(
        database_file=database_file,
    )

    store.initialize()

    import sqlite3

    with sqlite3.connect(database_file) as connection:
        connection.execute(
            """
            INSERT INTO inference_config (
                id,
                model_name,
                confidence_threshold
            )
            VALUES (1, ?, ?)
            """,
            (
                "cnn",
                1.50,
            ),
        )

    import pytest

    with pytest.raises(ValueError):
        store.load()