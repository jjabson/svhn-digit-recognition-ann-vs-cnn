import pytest

from config.inference_config import InferenceConfig
from config.inference_config_service import InferenceConfigService
from config.inference_config_store import InferenceConfigStore


def test_get_config_returns_active_config():
    config = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.90,
    )

    service = InferenceConfigService(config)

    assert service.get_config() == config


def test_set_confidence_threshold_replaces_config():
    original = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.90,
    )

    service = InferenceConfigService(original)

    updated = service.set_confidence_threshold(0.95)

    assert updated.confidence_threshold == 0.95
    assert service.get_config().confidence_threshold == 0.95


def test_set_confidence_threshold_preserves_model_name():
    original = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.90,
    )

    service = InferenceConfigService(original)

    updated = service.set_confidence_threshold(0.95)

    assert updated.model_name == "cnn"


def test_set_confidence_threshold_creates_new_config_object():
    original = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.90,
    )

    service = InferenceConfigService(original)

    updated = service.set_confidence_threshold(0.95)

    assert updated is not original


def test_invalid_threshold_does_not_replace_active_config():
    original = InferenceConfig(
        model_name="cnn",
        confidence_threshold=0.90,
    )

    service = InferenceConfigService(original)

    with pytest.raises(ValueError):
        service.set_confidence_threshold(1.50)

    assert service.get_config() is original
    assert service.get_config().confidence_threshold == 0.90

def test_service_uses_persisted_config_when_available(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    store.save(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.95,
        )
    )

    service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    config = service.get_config()

    assert config.confidence_threshold == 0.95

def test_service_uses_default_when_no_persisted_config_exists(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    config = service.get_config()

    assert config.confidence_threshold == 0.90


def test_threshold_update_is_persisted(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    service.set_confidence_threshold(0.95)

    persisted = store.load()

    assert persisted is not None
    assert persisted.confidence_threshold == 0.95

def test_new_service_recovers_persisted_threshold(tmp_path):
    store = InferenceConfigStore(
        database_file=tmp_path / "runtime_state.db",
    )

    first_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    first_service.set_confidence_threshold(0.95)

    second_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    assert (
        second_service.get_config().confidence_threshold
        == 0.95
    )

def test_failed_persistence_does_not_replace_active_config():
    class FailingStore:
        def load(self):
            return None

        def save(self, config):
            raise RuntimeError("database unavailable")

    service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=FailingStore(),
    )

    import pytest

    with pytest.raises(
        RuntimeError,
        match="database unavailable",
    ):
        service.set_confidence_threshold(0.95)

    assert (
        service.get_config().confidence_threshold
        == 0.90
    )