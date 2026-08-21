import h5py
import numpy as np

from config.report_config import REPORT_METADATA

from tools.model.inspect_cnn import (
    inspect_model,
    load_trained_model,
)

from config.project_paths import (
    DATA_FILE,
    GENERATED_DIR,
    REPORT_DATA_FILE,
)

def build_model_variables() -> dict[str, str]:
    """
    Build Typst report variables from the trained CNN architecture.
    """
    model = load_trained_model()
    architecture = inspect_model(model)

    return {
        "model-name": architecture.model_name,
        "model-total-parameters": f"{architecture.total_parameters:,}",
        "model-number-of-layers": str(architecture.number_of_layers),

        "model-input-shape": (
            f"{architecture.input_shape[1]} × "
            f"{architecture.input_shape[2]} × "
            f"{architecture.input_shape[3]}"
        ),

        "model-final-feature-shape": (
            f"{architecture.final_feature_group.output_shape[1]} × "
            f"{architecture.final_feature_group.output_shape[2]} × "
            f"{architecture.final_feature_group.output_shape[3]}"
        ),

        "model-flattened-features": (
            f"{architecture.flatten_layer.output_shape[-1]:,}"
        ),

        "model-hidden-dense-units": (
            str(architecture.hidden_dense_layer.config["units"])
        ),

        "model-output-classes": (
            str(architecture.output_layer.config["units"])
        ),

        "model-kernel-size": (
            f'{architecture.first_conv.config["kernel_size"][0]} × '
            f'{architecture.first_conv.config["kernel_size"][1]}'
        ),

        "model-dropout-rate": (
            str(architecture.dropout_layer.config["rate"])
        ),
    }

def load_dataset_metadata() -> dict[str, int]:
    """
    Load dataset sizes and class information from the HDF5 file.
    """
    with h5py.File(DATA_FILE, "r") as h5_file:
        x_train = h5_file["X_train"]
        x_val = h5_file["X_val"]
        x_test = h5_file["X_test"]
        y_train = np.array(h5_file["y_train"])

        class_counts = np.bincount(y_train, minlength=10)

        metadata = {
            "training_images": x_train.shape[0],
            "validation_images": x_val.shape[0],
            "testing_images": x_test.shape[0],
            "image_height": x_train.shape[1],
            "image_width": x_train.shape[2],
            "image_channels": 1,
            "number_of_classes": len(np.unique(y_train)),
            "smallest_class_count": int(class_counts.min()),
            "largest_class_count": int(class_counts.max()),
            "class_count_difference": int(class_counts.max() - class_counts.min()),
            "average_class_count": float(class_counts.mean()),
        }

    return metadata

def format_integer(value: int) -> str:
     """
     Format an integer with thousands separators.

     Example:
         42000 -> "42,000"
     """
     return f"{value:,}"

def build_dataset_variables(
    metadata: dict[str, int | float],
) -> dict[str, str]:
    """
    Build presentation-ready variables for the Dataset Description section.
    """
    return {
        "training-images": format_integer(
            int(metadata["training_images"])
        ),
        "validation-images": format_integer(
            int(metadata["validation_images"])
        ),
        "testing-images": format_integer(
            int(metadata["testing_images"])
        ),
        "image-dimensions": (
            f'{int(metadata["image_width"])} × '
            f'{int(metadata["image_height"])} pixels'
        ),
        "class-summary": (
            f'{int(metadata["number_of_classes"])} '
            "(digits 0–9)"
        ),
        "image-channels": REPORT_METADATA["image_channels"],
        "data-type": REPORT_METADATA["data_type"],
        "learning-task": REPORT_METADATA["learning_task"],
    }

def build_eda_variables(
    metadata: dict[str, int | float],
) -> dict[str, str]:
    """
    Build presentation-ready variables for the EDA section.
    """
    return {
        "smallest-class-count": format_integer(
            int(metadata["smallest_class_count"])
        ),
        "largest-class-count": format_integer(
            int(metadata["largest_class_count"])
        ),
        "class-count-difference": format_integer(
            int(metadata["class_count_difference"])
        ),
        "average-class-count": format_integer(
            round(float(metadata["average_class_count"]))
        ),
    }

def build_preprocessing_variables(
    metadata: dict[str, int | float],
) -> dict[str, str]:
    """
    Build presentation-ready variables for the Data Preprocessing section.
    """
    return {
        "training-tensor-shape": (
            "("
            f"{format_integer(int(metadata['training_images']))}, "
            f"{int(metadata['image_height'])}, "
            f"{int(metadata['image_width'])}, "
            f"{int(metadata['image_channels'])}"
            ")"
        ),
        "original-pixel-range": "0–255",
        "normalized-pixel-range": "0.0–1.0",
        "normalization-divisor": "255",
        "one-hot-vector-length": format_integer(
            int(metadata["number_of_classes"])
        ),
    }

def build_report_variables(
    metadata: dict[str, int | float],
) -> dict[str, str]:
    """
    Combine all report-section variables into one public Typst interface.
    """
    return {
        **build_dataset_variables(metadata),
        **build_eda_variables(metadata),
        **build_preprocessing_variables(metadata),
        **build_model_variables(),
    }

def create_typst_report_data(metadata: dict[str, int]) -> None:
    """
    Write dataset metadata as reusable Typst variables.
    """
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    report_variables = build_report_variables(metadata)

    lines = [
        "// This file is generated automatically.",
        "// Do not edit it manually.",
        "",
    ]

    for variable_name, variable_value in report_variables.items():
        lines.append(
            f'#let {variable_name} = "{variable_value}"'
        )

    typst_content = "\n".join(lines) + "\n"

    """
    The generated report_data.typ file should expose only presentation-ready values required by the report. 
    Raw dataset values remain internal to Python unless the report needs them directly.
    """

    REPORT_DATA_FILE.write_text(
        typst_content,
        encoding="utf-8",
    )

    print(f"Report data created successfully: {REPORT_DATA_FILE}")


def main() -> None:
    metadata = load_dataset_metadata()

    create_typst_report_data(metadata)


if __name__ == "__main__":
    main()
