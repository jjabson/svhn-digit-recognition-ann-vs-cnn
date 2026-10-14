from config.inference_config import InferenceConfig
from config.inference_config_service import (
    InferenceConfigService,
)
from config.inference_config_store import (
    InferenceConfigStore,
)


def create_service(database_file):
    store = InferenceConfigStore(
        database_file
    )

    return InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )


def test_existing_service_observes_external_config_update(
    tmp_path,
):
    database_file = (
        tmp_path / "runtime_state.db"
    )

    first_service = create_service(
        database_file
    )

    second_service = create_service(
        database_file
    )

    assert (
        first_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

    assert (
        second_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

    first_service.set_confidence_threshold(
        0.95
    )

    # second_service was already alive before the write.
    # Its next read must refresh from persistent storage.
    assert (
        second_service
        .get_config()
        .confidence_threshold
        == 0.95
    )


def test_cross_process_updates_are_bidirectional(
    tmp_path,
):
    database_file = (
        tmp_path / "runtime_state.db"
    )

    first_service = create_service(
        database_file
    )

    second_service = create_service(
        database_file
    )

    first_service.set_confidence_threshold(
        0.95
    )

    assert (
        second_service
        .get_config()
        .confidence_threshold
        == 0.95
    )

    second_service.set_confidence_threshold(
        0.90
    )

    assert (
        first_service
        .get_config()
        .confidence_threshold
        == 0.90
    )


def test_mutation_uses_latest_persisted_config(
    tmp_path,
):
    database_file = (
        tmp_path / "runtime_state.db"
    )

    first_service = create_service(
        database_file
    )

    second_service = create_service(
        database_file
    )

    first_service.set_confidence_threshold(
        0.95
    )

    updated = (
        second_service
        .set_confidence_threshold(
            0.92
        )
    )

    assert (
        updated.confidence_threshold
        == 0.92
    )

    assert (
        first_service
        .get_config()
        .confidence_threshold
        == 0.92
    )


def test_service_without_store_remains_in_memory():
    service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    updated = (
        service
        .set_confidence_threshold(
            0.95
        )
    )

    assert (
        updated.confidence_threshold
        == 0.95
    )

    assert (
        service
        .get_config()
        .confidence_threshold
        == 0.95
    )