from src.orchestration.execution import execute_model_attempt


def test_execute_model_attempt_returns_successful_attempt():
    def prediction():
        return 7, 0.982

    attempt = execute_model_attempt(
        model_name="cnn",
        predict_fn=prediction,
    )

    assert attempt.model_name == "cnn"
    assert attempt.predicted_digit == 7
    assert attempt.confidence == 0.982
    assert attempt.succeeded is True
    assert attempt.error_message is None


def test_execute_model_attempt_converts_exception_to_failed_attempt():
    def prediction():
        raise RuntimeError("Model unavailable")

    attempt = execute_model_attempt(
        model_name="ann",
        predict_fn=prediction,
    )

    assert attempt.model_name == "ann"
    assert attempt.predicted_digit is None
    assert attempt.confidence is None
    assert attempt.succeeded is False
    assert attempt.error_message == "Model unavailable"