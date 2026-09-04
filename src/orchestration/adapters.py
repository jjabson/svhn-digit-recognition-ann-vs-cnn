from collections.abc import Callable

from src.inference import SVHNPredictor


def create_svhn_predict_fn(
    predictor: SVHNPredictor,
    image_bytes: bytes,
) -> Callable[[], tuple[int, float]]:
    """
    Adapt SVHNPredictor.predict() to the generic prediction
    callable expected by the orchestration execution layer.
    """

    def predict() -> tuple[int, float]:
        result = predictor.predict(image_bytes)

        return (
            int(result["predicted_digit"]),
            float(result["confidence"]),
        )

    return predict