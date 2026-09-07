from collections.abc import Callable

from src.schemas.orchestration import InferenceAttempt
from src.orchestration.errors import InvalidInferenceInputError


def execute_model_attempt(
    model_name: str,
    predict_fn: Callable[[], tuple[int, float]],
) -> InferenceAttempt:
    """
        Execute one model prediction and convert the result into
        a normalized InferenceAttempt.

        The prediction function must return:
            (predicted_digit, confidence)
        """
    try:
        predicted_digit, confidence = predict_fn()

        return InferenceAttempt(
            model_name=model_name,
            predicted_digit=predicted_digit,
            confidence=confidence,
            succeeded=True,
            error_message=None,
        )

    except InvalidInferenceInputError:
        raise

    except Exception as exc:
        return InferenceAttempt(
            model_name=model_name,
            predicted_digit=None,
            confidence=None,
            succeeded=False,
            error_message=str(exc),
        )