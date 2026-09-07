from collections.abc import Callable

from src.inference import SVHNPredictor
from src.orchestration.errors import InvalidInferenceInputError
from src.preprocessing import InvalidImageError


def create_svhn_predict_fn(
    predictor: SVHNPredictor,
    image_bytes: bytes,
) -> Callable[[], tuple[int, float]]:
    """
    Adapt SVHNPredictor.predict() to the generic prediction
    callable expected by the orchestration execution layer.
    """

    def predict() -> tuple[int, float]:
        try:
            result = predictor.predict(image_bytes)

        except InvalidImageError as exc:
            raise InvalidInferenceInputError(
                str(exc)
            ) from exc

        return (
            result.predicted_digit,
            result.confidence,
        )

    return predict