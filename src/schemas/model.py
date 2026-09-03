from pydantic import BaseModel, ConfigDict

from tools.model.inspect_cnn import ArchitectureInfo


class ModelSummaryResponse(BaseModel):
    """
    Public API representation of the trained model summary.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "sequential",
                "input_shape": "32 × 32 × 1",
                "output_classes": 10,
                "number_of_layers": 17,
                "total_parameters": 164362,
                "final_feature_shape": "8 × 8 × 64",
                "flattened_features": 4096,
                "hidden_dense_units": 32,
                "dropout_rate": 0.5,
            }
        }
    )

    model_name: str
    input_shape: str
    output_classes: int
    number_of_layers: int
    total_parameters: int
    final_feature_shape: str
    flattened_features: int
    hidden_dense_units: int
    dropout_rate: float


def format_model_shape(
    shape: tuple,
) -> str:
    """
    Convert a Keras tensor shape into a human-readable model shape,
    excluding the variable batch dimension.
    """
    dimensions = shape[1:]

    return " × ".join(
        str(dimension)
        for dimension in dimensions
    )


def model_architecture_to_response(
    architecture: ArchitectureInfo,
) -> ModelSummaryResponse:
    """
    Convert the internal architecture representation
    into the public API model summary.
    """
    return ModelSummaryResponse(
        model_name=architecture.model_name,
        input_shape=format_model_shape(
            architecture.input_shape
        ),
        output_classes=architecture.output_layer.units,
        number_of_layers=architecture.number_of_layers,
        total_parameters=architecture.total_parameters,
        final_feature_shape=format_model_shape(
            architecture.final_feature_group.output_shape
        ),
        flattened_features=architecture.flatten_layer.output_shape[-1],
        hidden_dense_units=architecture.hidden_dense_layer.units,
        dropout_rate=architecture.dropout_layer.dropout_rate,
    )