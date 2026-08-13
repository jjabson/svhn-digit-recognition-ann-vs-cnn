from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from tools.model.inspect_cnn import (
    LayerInfo,
    inspect_model,
    load_trained_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"

CNN_ARCHITECTURE_FILE = (
    FIGURES_DIR / "figure5_cnn_architecture.png"
)

CNN_DETAILED_ARCHITECTURE_FILE = (
    FIGURES_DIR / "figure6_cnn_detailed_architecture.png"
)

DETAILED_CNN_ARCHITECTURE_FILE = (
    FIGURES_DIR / "cnn_architecture_detailed.png"
)

def format_layer_details(
    layer: LayerInfo,
) -> str:
    """
    Build a concise human-readable description of a CNN layer.
    """
    layer_type = layer.layer_type
    config = layer.config

    if layer_type == "Conv2D":
        return (
            f'{config["filters"]} filters, '
            f'{config["kernel_size"][0]} × {config["kernel_size"][1]}, '
            f'padding={config["padding"]}'
        )

    if layer_type == "LeakyReLU":
        return (
            f'negative slope={config["negative_slope"]}'
        )

    if layer_type == "MaxPooling2D":
        return (
            f'pool={config["pool_size"][0]} × '
            f'{config["pool_size"][1]}'
        )

    if layer_type == "BatchNormalization":
        return "Batch normalization"

    if layer_type == "Flatten":
        return "Flatten feature maps"

    if layer_type == "Dense":
        return (
            f'{config["units"]} units, '
            f'activation={config["activation"]}'
        )

    if layer_type == "Dropout":
        return (
            f'rate={config["rate"]}'
        )

    return ""

def create_cnn_architecture_figure() -> None:
    """
    Generate a high-level diagram of the trained CNN architecture.
    """

    # ---------------------------------
    # Load and inspect model
    # ---------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_trained_model()
    architecture = inspect_model(model)

    group_summaries = architecture.group_summaries
    total_parameters = architecture.total_parameters

    # ---------------------------------
    # Prepare diagram data
    # ---------------------------------

    diagram_items = [
        {
            "title": "Input",
            "details": [
                "32 × 32 × 1",
                "Grayscale image",
            ],
        },
    ]

    for group in group_summaries:
        output_shape = group.output_shape

        if group.group_name == "Classification Head":
            shape_text = f"{output_shape[-1]} classes"
        else:
            shape_text = (
                f"{output_shape[1]} × "
                f"{output_shape[2]} × "
                f"{output_shape[3]}"
            )

        diagram_items.append(
            {
                "title": group.group_name,
                "details": [
                    f'{group.number_of_layers} layers',
                    f'{group.parameters:,} parameters',
                    f"Output: {shape_text}",
                ],
            }
        )

    # ---------------------------------
    # Create visualization
    # ---------------------------------

    figure, axis = plt.subplots(
        figsize=(10, 9),
    )

    axis.set_xlim(0, 10)
    axis.set_ylim(0, 14)
    axis.axis("off")

    figure.suptitle(
    "SVHN CNN Architecture",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    axis.text(
    5,
    13.25,
    f"{total_parameters:,} trainable parameters",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
    )

    box_width = 8.2
    box_height = 2.25
    box_x = (10 - box_width) / 2

    box_specs = [
        {
            "y": 10.3,
            "fill": "#EAF2FF",
            "edge": "#2F5EA8",
            "title": "Input",
            "subtitle": None,
            "details": [
            "32 × 32 × 1",
            "Grayscale image",
            ],
        },
        {
            "y": 7.2,
            "fill": "#EEF8EC",
            "edge": "#3D7A3A",
            "title": group_summaries[0].group_name,
            "subtitle": "Convolution + Pooling",
            "details": [
                f'{group_summaries[0].number_of_layers} layers',
                f'{group_summaries[0].parameters:,} parameters',
                (
                    "Tensor Shape: "
                    f'{group_summaries[0].output_shape[1]} × '
                    f'{group_summaries[0].output_shape[2]} × '
                    f'{group_summaries[0].output_shape[3]}'
                ),
            ],
        },
        {
            "y": 4.1,
            "fill": "#EEF8EC",
            "edge": "#3D7A3A",
            "title": group_summaries[1].group_name,
            "subtitle": "Convolution + Pooling",
            "details": [
                f'{group_summaries[1].number_of_layers} layers',
                f'{group_summaries[1].parameters:,} parameters',
                (
                    "Tensor Shape: "
                    f'{group_summaries[1].output_shape[1]} × '
                    f'{group_summaries[1].output_shape[2]} × '
                    f'{group_summaries[1].output_shape[3]}'
                ),
            ],
        },
        {
            "y": 1.0,
            "fill": "#FFF6DD",
            "edge": "#C58B11",
            "title": group_summaries[2].group_name,
            "subtitle": "Dense Neural Network",
            "details": [
                f'{group_summaries[2].number_of_layers} layers',
                f'{group_summaries[2].parameters:,} parameters',
                (
                    "Output: "
                    f'{group_summaries[2].output_shape[-1]} classes'
                ),
            ],
        },
    ]

    for index, spec in enumerate(box_specs):
        box = FancyBboxPatch(
            (box_x, spec["y"]),
            box_width,
            box_height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.8,
            edgecolor=spec["edge"],
            facecolor=spec["fill"],
        )

        axis.add_patch(box)

        title_y = spec["y"] + 1.72

        axis.text(
            5,
            title_y,
            spec["title"],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

        current_y = title_y - 0.55

        if spec["subtitle"] is not None:
            axis.text(
                5,
                current_y,
                spec["subtitle"],
                ha="center",
                va="center",
                fontsize=11,
                style="italic",
            )
            current_y -= 0.70

        axis.text(
            5,
            current_y,
            "\n".join(spec["details"]),
            ha="center",
            va="center",
            fontsize=10.5,
        )

        if index < len(box_specs) - 1:
            current_box_bottom = spec["y"]
            next_box_top = (
                    box_specs[index + 1]["y"]
                    + box_height
            )

            axis.annotate(
                "",
                xy=(5, next_box_top + 0.08),
                xytext=(5, current_box_bottom - 0.08),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.7,
                },
            )

    # ---------------------------------
    # Save artifact
    # ---------------------------------

    figure.savefig(
        CNN_ARCHITECTURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure created successfully: "
        f"{CNN_ARCHITECTURE_FILE}"
    )

def create_detailed_cnn_architecture_figure() -> None:
    """
    Generate a detailed layer-by-layer diagram of the trained CNN.
    """

    # ---------------------------------
    # Load and inspect model
    # ---------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_trained_model()
    architecture = inspect_model(model)

    groups = architecture.groups

    # ---------------------------------
    # Layout configuration
    # ---------------------------------

    LAYER_VERTICAL_SPACING = 1.05
    GROUP_TITLE_SPACING = 1.05
    GROUP_SPACING = 0.55
    TOP_MARGIN = 1.5
    BOTTOM_MARGIN = 1.0

    total_layers = sum(
        len(group.layers)
        for group in groups
    )

    total_groups = len(groups)

    content_height = (
            total_layers * LAYER_VERTICAL_SPACING
            + total_groups * GROUP_TITLE_SPACING
            + (total_groups - 1) * GROUP_SPACING
    )

    axis_height = (
            content_height
            + TOP_MARGIN
            + BOTTOM_MARGIN
    )

    group_styles = [
        {
            "fill": "#EEF8EC",
            "edge": "#3D7A3A",
        },
        {
            "fill": "#E2F2E5",
            "edge": "#3D7A3A",
        },
        {
            "fill": "#FFF6DD",
            "edge": "#C58B11",
        },
    ]

    # ---------------------------------
    # Create visualization
    # ---------------------------------

    figure, axis = plt.subplots(
        figsize=(12, 24),
    )

    axis.set_xlim(0, 12)
    axis.set_ylim(0, axis_height)
    axis.axis("off")

    figure.suptitle(
        "Detailed SVHN CNN Architecture",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    box_width = 9.5
    box_height = 0.90
    box_x = (12 - box_width) / 2

    current_y = axis_height - TOP_MARGIN

    for group_index, (group, style) in enumerate(
            zip(groups, group_styles)
    ):

        # ---------------------------------
        # Group heading
        # ---------------------------------

        # Draw the group title
        axis.text(
            6,
            current_y,
            group.group_name,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=style["edge"],
        )

        # Add breathing room
        current_y -= GROUP_TITLE_SPACING

        # Draw first box

        # ---------------------------------
        # Group layers
        # ---------------------------------

        for layer in group.layers:
            layer_details = format_layer_details(layer)

            output_shape = layer.output_shape

            if len(output_shape) == 4:
                shape_text = (
                    f"{output_shape[1]} × "
                    f"{output_shape[2]} × "
                    f"{output_shape[3]}"
                )
            else:
                shape_text = str(output_shape[-1])

            box = FancyBboxPatch(
                (box_x, current_y),
                box_width,
                box_height,
                boxstyle="round,pad=0.03,rounding_size=0.05",
                linewidth=1.2,
                edgecolor=style["edge"],
                facecolor=style["fill"],
            )

            axis.add_patch(box)

            axis.text(
                box_x + 0.25,
                current_y + 0.60,
                f'{layer.layer_number}. {layer.layer_type}',
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

            axis.text(
                box_x + 0.25,
                current_y + 0.26,
                layer_details,
                ha="left",
                va="center",
                fontsize=9.2,
            )

            axis.text(
                box_x + box_width - 0.25,
                current_y + 0.42,
                f"Output: {shape_text}",
                ha="right",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )

            current_y -= LAYER_VERTICAL_SPACING


        # Add spacing only between groups
        if group_index < len(groups) - 1:
            current_y -= GROUP_SPACING

    figure.tight_layout(
        rect=(0, 0, 1, 0.96),
    )

    # ---------------------------------
    # Save artifact
    # ---------------------------------

    figure.savefig(
        DETAILED_CNN_ARCHITECTURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure created successfully: "
        f"{DETAILED_CNN_ARCHITECTURE_FILE}"
    )

def create_model_figures() -> None:
    """
    Generate all figures used by the CNN architecture report section.
    """
    create_cnn_architecture_figure()
    create_detailed_cnn_architecture_figure()


def main() -> None:
    create_model_figures()


if __name__ == "__main__":
    main()