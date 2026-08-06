from pathlib import Path

from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "svhn_cnn.keras"

def load_trained_model() -> keras.Model:
    """
    Load the trained SVHN CNN from the saved Keras model file.

    Returns
    -------
    keras.Model
        The deserialized trained model.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}"
        )

    model = keras.models.load_model(MODEL_FILE)

    return model

def inspect_model_summary(
    model: keras.Model,
) -> dict[str, object]:
    """
    Extract high-level metadata from the trained CNN.
    """
    return {
        "model_name": model.name,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_parameters": model.count_params(),
        "number_of_layers": len(model.layers),
    }

def main() -> None:
    """
    Verify that the trained model can be loaded and inspected.
    """
    model = load_trained_model()
    summary = inspect_model_summary(model)

    print(f"Model loaded successfully: {MODEL_FILE}")

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()