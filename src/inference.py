from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_image_bytes


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "svhn_cnn.keras"
)


class SVHNPredictor:
    """Load an SVHN CNN and perform image classification."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found: {model_path}"
            )

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_bytes: bytes) -> dict:
        image_tensor = preprocess_image_bytes(image_bytes)

        probabilities = self.model.predict(
            image_tensor,
            verbose=0
        )[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])

        return {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "confidence_percent": f"{confidence * 100:.2f}%",
            "probabilities": {
                str(digit): float(probability)
                for digit, probability in enumerate(probabilities)
            }
        }