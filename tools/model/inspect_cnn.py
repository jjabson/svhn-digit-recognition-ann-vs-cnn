from tensorflow import keras

from dataclasses import dataclass, field
from config.project_paths import MODEL_FILE

@dataclass
class LayerInfo:
    layer_number: int
    layer_name: str
    layer_type: str
    output_shape: tuple
    parameters: int
    config: dict[str, object] = field(default_factory=dict)

    @property
    def units(self) -> int | None:
        """
        Return the number of units for layers that define them.
        """
        value = self.config.get("units")
        return int(value) if value is not None else None

    @property
    def dropout_rate(self) -> float | None:
        """
        Return the dropout rate for Dropout layers.
        """
        value = self.config.get("rate")
        return float(value) if value is not None else None

@dataclass
class GroupInfo:
    group_name: str
    layers: list[LayerInfo]

    @property
    def number_of_layers(self) -> int:
        """
        Return the number of layers in this architecture group.
        """
        return len(self.layers)

    @property
    def parameters(self) -> int:
        """
        Return the total parameter count for this architecture group.
        """
        return sum(
            layer.parameters
            for layer in self.layers
        )

    @property
    def output_shape(self) -> tuple:
        """
        Return the output shape of the final layer in this group.
        """
        return self.layers[-1].output_shape



@dataclass
class ArchitectureInfo:
    model_name: str
    input_shape: tuple
    output_shape: tuple
    total_parameters: int
    number_of_layers: int
    layers: list[LayerInfo]
    groups: list[GroupInfo]

    @property
    def first_conv(self) -> LayerInfo:
        return next(
            layer
            for layer in self.layers
            if layer.layer_type == "Conv2D"
        )

    @property
    def flatten_layer(self) -> LayerInfo:
        return next(
            layer
            for layer in self.layers
            if layer.layer_type == "Flatten"
        )

    @property
    def dropout_layer(self) -> LayerInfo:
        return next(
            layer
            for layer in self.layers
            if layer.layer_type == "Dropout"
        )

    @property
    def output_layer(self) -> LayerInfo:
        return next(
            layer
            for layer in reversed(self.layers)
            if layer.layer_type == "Dense"
        )

    @property
    def hidden_dense_layer(self) -> LayerInfo:
        dense_layers = [
            layer
            for layer in self.layers
            if layer.layer_type == "Dense"
        ]

        return dense_layers[0]

    @property
    def feature_extraction_block1(self) -> GroupInfo:
        return self.groups[0]

    @property
    def feature_extraction_block2(self) -> GroupInfo:
        return self.groups[1]

    @property
    def classification_head(self) -> GroupInfo:
        return self.groups[2]

    @property
    def final_feature_group(self) -> GroupInfo:
        return self.groups[-2]


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

    return ArchitectureInfo(
        model_name=summary.model_name,
        input_shape=summary.input_shape,
        output_shape=summary.output_shape,
        total_parameters=summary.total_parameters,
        number_of_layers=summary.number_of_layers,
        layers=layers,
        groups=groups,
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

    for group in architecture.groups:
        print(
            f"{group.group_name}: "
            f"{group.number_of_layers} layers, "
            f"{group.parameters:,} parameters, "
            f"output={group.output_shape}"
        )
    print("\nSemantic Layer Access")
    print("---------------------")
    print(f"First Conv2D: {architecture.first_conv.layer_name}")
    print(f"Flatten: {architecture.flatten_layer.layer_name}")
    print(f"Hidden Dense: {architecture.hidden_dense_layer.layer_name}")
    print(f"Dropout: {architecture.dropout_layer.layer_name}")
    print(f"Output Dense: {architecture.output_layer.layer_name}")



if __name__ == "__main__":
    main()