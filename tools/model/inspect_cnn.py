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

def inspect_layer_config(
    layer: keras.layers.Layer,
) -> dict[str, object]:
    """
    Extract architecture-relevant configuration for a model layer.
    """
    config = layer.get_config()

    layer_config = {}

    if isinstance(layer, keras.layers.Conv2D):
        layer_config["filters"] = config["filters"]
        layer_config["kernel_size"] = config["kernel_size"]
        layer_config["strides"] = config["strides"]
        layer_config["padding"] = config["padding"]

    elif isinstance(layer, keras.layers.MaxPooling2D):
        layer_config["pool_size"] = config["pool_size"]
        layer_config["strides"] = config["strides"]
        layer_config["padding"] = config["padding"]

    elif isinstance(layer, keras.layers.BatchNormalization):
        layer_config["axis"] = config["axis"]

    elif isinstance(layer, keras.layers.Dense):
        layer_config["units"] = config["units"]
        layer_config["activation"] = config["activation"]

    elif isinstance(layer, keras.layers.Dropout):
        layer_config["rate"] = config["rate"]

    elif isinstance(layer, keras.layers.LeakyReLU):
        layer_config["negative_slope"] = config["negative_slope"]

    return layer_config

def inspect_model_layers(
    model: keras.Model,
) -> list[dict[str, object]]:
    """
    Extract layer-by-layer architecture details from the trained CNN.
    """
    layers = []

    for index, layer in enumerate(model.layers, start=1):
        layers.append(
            {
                "layer_number": index,
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "output_shape": layer.output.shape,
                "parameters": layer.count_params(),
                "config": inspect_layer_config(layer),
            }
        )

    return layers

def group_model_layers(
    layers: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Group raw CNN layers into higher-level architectural components.
    """
    return [
        {
            "group_name": "Feature Extraction Block 1",
            "layers": layers[0:6],
        },
        {
            "group_name": "Feature Extraction Block 2",
            "layers": layers[6:12],
        },
        {
            "group_name": "Classification Head",
            "layers": layers[12:17],
        },
    ]

def summarize_model_groups(
    groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    """
    Build summary statistics for each architectural model group.
    """
    summaries = []

    for group in groups:
        group_layers = group["layers"]

        summaries.append(
            {
                "group_name": group["group_name"],
                "number_of_layers": len(group_layers),
                "parameters": sum(
                    layer["parameters"]
                    for layer in group_layers
                ),
                "output_shape": group_layers[-1]["output_shape"],
            }
        )

    return summaries

def inspect_model(
    model: keras.Model,
) -> dict[str, object]:
    """
    Build the complete inspected architecture representation
    for the trained CNN.
    """
    summary = inspect_model_summary(model)
    layers = inspect_model_layers(model)
    groups = group_model_layers(layers)
    group_summaries = summarize_model_groups(groups)

    return {
        "summary": summary,
        "layers": layers,
        "groups": groups,
        "group_summaries": group_summaries,
        "model_name": summary["model_name"],
        "input_shape": summary["input_shape"],
        "output_shape": summary["output_shape"],
        "total_parameters": summary["total_parameters"],
        "number_of_layers": summary["number_of_layers"],
    }

def main() -> None:
    """
    Verify that the trained model can be loaded and inspected.
    """
    model = load_trained_model()
    architecture = inspect_model(model)

    print(f"Model loaded successfully: {MODEL_FILE}")

    print("\nModel Summary")
    print("-------------")

    for key, value in architecture["summary"].items():
        print(f"{key}: {value}")

    print("\nModel Architecture Groups")
    print("-------------------------")

    for group in architecture["groups"]:
        print(f"\n{group['group_name']}")

        for layer in group["layers"]:
            print(
                f"{layer['layer_number']}: "
                f"{layer['layer_type']} "
                f"{layer['output_shape']}"
            )

    print("\nModel Group Summary")
    print("-------------------")

    for group in architecture["group_summaries"]:
        print(group)


if __name__ == "__main__":
    main()