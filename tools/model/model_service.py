from tensorflow import keras

from tools.model.inspect_cnn import ArchitectureInfo, inspect_model


def get_model_architecture(
    model: keras.Model,
) -> ArchitectureInfo:
    """
    Return the semantic architecture representation
    for a trained Keras model.
    """
    return inspect_model(model)