from pathlib import Path

from tensorflow import keras

from dataclasses import dataclass, field


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = PROJECT_ROOT / "models" / "svhn_cnn.keras"

@dataclass
class LayerInfo:
    layer_number: int
    layer_name: str
    layer_type: str
    output_shape: tuple
    parameters: int
    config: dict[str, object] = field(default_factory=dict)


@dataclass
class GroupInfo:
    group_name: str
    layers: list[LayerInfo]


@dataclass
class GroupSummary:
    group_name: str
    number_of_layers: int
    parameters: int
    output_shape: tuple


@dataclass
class ArchitectureInfo:
    model_name: str
    input_shape: tuple
    output_shape: tuple
    total_parameters: int
    number_of_layers: int
    layers: list[LayerInfo]
    groups: list[GroupInfo]
    group_summaries: list[GroupSummary]


@dataclass
class ModelSummary:
    model_name: str
    input_shape: tuple
    output_shape: tuple
    total_parameters: int
    number_of_layers: int

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
) -> ModelSummary:
    """
    Extract high-level metadata from the trained CNN.
    """
    return ModelSummary(
        model_name=model.name,
        input_shape=model.input_shape,
        output_shape=model.output_shape,
        total_parameters=model.count_params(),
        number_of_layers=len(model.layers),
    )

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
) -> list[LayerInfo]:
    """
    Extract layer-by-layer architecture details from the trained CNN.
    """
    layers = []

    for index, layer in enumerate(model.layers, start=1):
        layers.append(
            LayerInfo(
                layer_number=index,
                layer_name=layer.name,
                layer_type=layer.__class__.__name__,
                output_shape=tuple(layer.output.shape),
                parameters=layer.count_params(),
                config=inspect_layer_config(layer),
            )
        )

    return layers

def group_model_layers(
    layers: list[LayerInfo],
) -> list[GroupInfo]:
    """
    Group raw CNN layers into higher-level architectural components.
    """
    return [
        GroupInfo(
            group_name="Feature Extraction Block 1",
            layers=layers[0:6],
        ),
        GroupInfo(
            group_name="Feature Extraction Block 2",
            layers=layers[6:12],
        ),
        GroupInfo(
            group_name="Classification Head",
            layers=layers[12:17],
        ),
    ]

def summarize_model_groups(
    groups: list[GroupInfo],
) -> list[GroupSummary]:
    """
    Build summary statistics for each architectural model group.
    """
    summaries = []

    for group in groups:
        summaries.append(
            GroupSummary(
                group_name=group.group_name,
                number_of_layers=len(group.layers),
                parameters=sum(
                    layer.parameters
                    for layer in group.layers
                ),
                output_shape=group.layers[-1].output_shape,
            )
        )

    return summaries

def inspect_model(
    model: keras.Model,
) -> ArchitectureInfo:
    """
    Build the complete typed architecture representation
    for the trained CNN.
    """
    summary = inspect_model_summary(model)
    layers = inspect_model_layers(model)
    groups = group_model_layers(layers)
    group_summaries = summarize_model_groups(groups)

    return ArchitectureInfo(
        model_name=summary.model_name,
        input_shape=summary.input_shape,
        output_shape=summary.output_shape,
        total_parameters=summary.total_parameters,
        number_of_layers=summary.number_of_layers,
        layers=layers,
        groups=groups,
        group_summaries=group_summaries,
    )

def main() -> None:
    """
    Load the trained CNN and display its inspected architecture.
    """
    model = load_trained_model()
    architecture = inspect_model(model)

    print(f"Model loaded successfully: {MODEL_FILE}")

    print("\nModel Summary")
    print("-------------")
    print(f"Model name: {architecture.model_name}")
    print(f"Input shape: {architecture.input_shape}")
    print(f"Output shape: {architecture.output_shape}")
    print(f"Total parameters: {architecture.total_parameters:,}")
    print(f"Number of layers: {architecture.number_of_layers}")

    print("\nArchitecture Groups")
    print("-------------------")

    for group, summary in zip(
        architecture.groups,
        architecture.group_summaries,
    ):
        print(
            f"{group.group_name}: "
            f"{summary.number_of_layers} layers, "
            f"{summary.parameters:,} parameters, "
            f"output={summary.output_shape}"
        )


if __name__ == "__main__":
    main()