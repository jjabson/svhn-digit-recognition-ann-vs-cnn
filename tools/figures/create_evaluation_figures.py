from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.project_paths import FIGURES_DIR
from tools.evaluation.evaluation_store import (
    load_evaluation_comparison,
)

def create_confusion_matrix_figure(
    output_path: Path,
) -> None:
    """
    Create the confusion matrix figure for the independent
    historical holdout evaluation.
    """
    comparison = load_evaluation_comparison()
    evaluation = comparison.historical

    confusion_matrix = evaluation.confusion_matrix

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        confusion_matrix,
        cmap="Blues",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Number of Predictions",
    )

    class_labels = range(
        confusion_matrix.shape[0]
    )

    ax.set_xticks(class_labels)
    ax.set_yticks(class_labels)

    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_title(
        "CNN Confusion Matrix — Historical Holdout"
    )

    threshold = confusion_matrix.max() / 2

    for true_class in class_labels:
        for predicted_class in class_labels:
            value = confusion_matrix[
                true_class,
                predicted_class
            ]

            ax.text(
                predicted_class,
                true_class,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def main() -> None:
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURES_DIR /
        "confusion_matrix_historical_holdout.png"
    )

    create_confusion_matrix_figure(
        output_path
    )

    print(
        f"Figure created successfully: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()