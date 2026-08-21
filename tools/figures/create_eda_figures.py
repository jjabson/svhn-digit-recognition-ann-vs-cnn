from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from config.project_paths import (
    DATA_FILE,
    FIGURES_DIR,
    SAMPLE_IMAGES_DIR,
)

OUTPUT_FILE = FIGURES_DIR / "figure1_sample_digits.png"

SAMPLE_DIGITS_FILE = FIGURES_DIR / "figure1_sample_digits.png"
CLASS_DISTRIBUTION_FILE = (
    FIGURES_DIR / "figure2_class_distribution.png"
)

def load_training_labels():
    """
    Load the training labels from the SVHN dataset.
    """

    with h5py.File(DATA_FILE, "r") as h5_file:
        y_train = np.array(h5_file["y_train"])

    return y_train

def find_digit_image(digit: int) -> Path:
    """
    Find the sample image corresponding to a particular digit.

    Expected filename pattern:
        digit_0_true0.png
        digit_1_true1.png
        ...
        digit_9_true9.png
    """
    matches = sorted(SAMPLE_IMAGES_DIR.glob(f"digit_{digit}_*.png"))

    if not matches:
        raise FileNotFoundError(
            f"No sample image was found for digit {digit} in "
            f"{SAMPLE_IMAGES_DIR}"
        )

    return matches[0]

def create_sample_digits_figure() -> None:
    """
    Create a 2-by-5 figure containing one representative image
    for each digit class from 0 through 9.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(10, 5.5),
    )

    for digit, axis in enumerate(axes.flat):
        image_path = find_digit_image(digit)

        with Image.open(image_path) as image:
            axis.imshow(image, cmap="gray")

        axis.set_title(f"Digit {digit}", fontsize=11)
        axis.axis("off")

    figure.suptitle(
        "Representative Images from the SVHN Dataset",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.91),
        h_pad=2.0,
        w_pad=1.0,
    )

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Figure created successfully: {OUTPUT_FILE}")

def create_class_distribution_figure() -> None:
    """
    Create a bar chart showing the number of training images
    for each digit class.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    y_train = load_training_labels()

    counts = np.bincount(y_train, minlength=10)
    digits = np.arange(10)

    figure, axis = plt.subplots(figsize=(9, 5.5))

    bars = axis.bar(digits, counts)

    axis.set_title(
        "Distribution of Training Images by Digit Class",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    axis.set_xlabel("Digit Class", fontsize=11)
    axis.set_ylabel("Number of Training Images", fontsize=11)

    axis.set_xticks(digits)

    axis.set_ylim(0, counts.max() * 1.12)

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.35,
    )

    for bar, count in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 25,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.tight_layout()

    figure.savefig(
        CLASS_DISTRIBUTION_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure created successfully: "
        f"{CLASS_DISTRIBUTION_FILE}"
    )

def create_eda_figures() -> None:
    """
    Generate all figures used by the EDA report section.
    """
    create_sample_digits_figure()
    create_class_distribution_figure()


def main() -> None:
    create_eda_figures()


if __name__ == "__main__":
    main()