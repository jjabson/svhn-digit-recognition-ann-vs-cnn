import pytest

from config.fallback_inference_config import (
    FALLBACK_ENABLED_ENV,
    FALLBACK_MODEL_NAME_ENV,
    FALLBACK_MODEL_PATH_ENV,
    FallbackInferenceConfig,
    FallbackInferenceConfigurationError,
)


def test_fallback_is_disabled_by_default(
    monkeypatch,
):
    monkeypatch.delenv(
        FALLBACK_ENABLED_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        FALLBACK_MODEL_NAME_ENV,
        raising=False,
    )
    monkeypatch.delenv(
        FALLBACK_MODEL_PATH_ENV,
        raising=False,
    )

    config = FallbackInferenceConfig.from_environment()

    assert config.enabled is False
    assert config.model_name is None
    assert config.model_path is None


def test_enabled_fallback_requires_model_name(
    monkeypatch,
    tmp_path,
):
    model_path = tmp_path / "fallback.keras"
    model_path.write_bytes(b"fake-model")

    monkeypatch.setenv(
        FALLBACK_ENABLED_ENV,
        "true",
    )
    monkeypatch.delenv(
        FALLBACK_MODEL_NAME_ENV,
        raising=False,
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_PATH_ENV,
        str(model_path),
    )

    with pytest.raises(
        FallbackInferenceConfigurationError,
        match=(
            f"{FALLBACK_MODEL_NAME_ENV} is required"
        ),
    ):
        FallbackInferenceConfig.from_environment()


def test_enabled_fallback_requires_model_path(
    monkeypatch,
):
    monkeypatch.setenv(
        FALLBACK_ENABLED_ENV,
        "true",
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_NAME_ENV,
        "ann",
    )
    monkeypatch.delenv(
        FALLBACK_MODEL_PATH_ENV,
        raising=False,
    )

    with pytest.raises(
        FallbackInferenceConfigurationError,
        match=(
            f"{FALLBACK_MODEL_PATH_ENV} is required"
        ),
    ):
        FallbackInferenceConfig.from_environment()


def test_enabled_fallback_requires_existing_model_file(
    monkeypatch,
    tmp_path,
):
    missing_path = (
        tmp_path / "missing.keras"
    )

    monkeypatch.setenv(
        FALLBACK_ENABLED_ENV,
        "true",
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_NAME_ENV,
        "ann",
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_PATH_ENV,
        str(missing_path),
    )

    with pytest.raises(
        FallbackInferenceConfigurationError,
        match="Fallback model file was not found",
    ):
        FallbackInferenceConfig.from_environment()


def test_enabled_fallback_loads_configuration(
    monkeypatch,
    tmp_path,
):
    model_path = (
        tmp_path / "fallback.keras"
    )
    model_path.write_bytes(b"fake-model")

    monkeypatch.setenv(
        FALLBACK_ENABLED_ENV,
        "true",
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_NAME_ENV,
        "ann",
    )
    monkeypatch.setenv(
        FALLBACK_MODEL_PATH_ENV,
        str(model_path),
    )

    config = FallbackInferenceConfig.from_environment()

    assert config.enabled is True
    assert config.model_name == "ann"
    assert config.model_path == model_path