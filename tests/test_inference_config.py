import pytest
from config.inference_config import (
    DEFAULT_INFERENCE_CONFIG,
    InferenceConfig,
)


def test_default_inference_config_values():
    assert isinstance(
        DEFAULT_INFERENCE_CONFIG,
        InferenceConfig,
    )

    assert DEFAULT_INFERENCE_CONFIG.confidence_threshold == 0.90
    assert DEFAULT_INFERENCE_CONFIG.model_name == "cnn"


def test_inference_config_rejects_threshold_below_zero():
    with pytest.raises(
        ValueError,
        match="confidence_threshold must be between",
    ):
        InferenceConfig(
            confidence_threshold=-0.01,
        )


def test_inference_config_rejects_threshold_above_one():
    with pytest.raises(
        ValueError,
        match="confidence_threshold must be between",
    ):
        InferenceConfig(
            confidence_threshold=1.01,
        )


def test_inference_config_rejects_empty_model_name():
    with pytest.raises(
        ValueError,
        match="model_name must not be empty",
    ):
        InferenceConfig(
            model_name="",
        )