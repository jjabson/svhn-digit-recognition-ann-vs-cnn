from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "SVHN_single_grey1.h5"
FIGURES_DIR = PROJECT_ROOT / "figures"

NORMALIZATION_COMPARISON_FILE = (
    FIGURES_DIR / "figure3_normalization_comparison.png"
)

PIXEL_DISTRIBUTION_FILE = (
    FIGURES_DIR / "figure4_pixel_distribution.png"
)

def load_sample_training_image(
    image_index: int = 0,
) -> np.ndarray:
    """
    Load one image from the SVHN training dataset.

    Parameters
    ----------
    image_index:
        Position of the image in X_train.

    Returns
    -------
    np.ndarray
        A single 32 x 32 grayscale image.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with h5py.File(DATA_FILE, "r") as h5_file:
        images = h5_file["X_train"]

        if not 0 <= image_index < len(images):
            raise IndexError(
                f"Image index {image_index} is outside the valid "
                f"range 0 through {len(images) - 1}."
            )

        image = np.asarray(images[image_index])

    return image

def load_training_images() -> np.ndarray:
    """
    Load all training images from the SVHN dataset.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    with h5py.File(DATA_FILE, "r") as h5_file:
        images = np.asarray(h5_file["X_train"])

    return images

def format_millions(
    value: float,
    _: int,
) -> str:
    """
    Format axis tick labels using millions.

    Example:
        1,500,000 -> 1.5 M
    """
    return f"{value / 1_000_000:.1f} M"

def create_pixel_distribution_figure() -> None:
    """
    Compare pixel-value distributions before and after normalization.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_images = load_training_images()

    # ---------------------------------
    # Prepare data
    # ---------------------------------

    original_pixels = training_images.reshape(-1)

    normalized_pixels = (
            original_pixels.astype(np.float32) / 255.0
    )

    # ---------------------------------
    # Create visualization
    # ---------------------------------

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11, 4.8),
    )

    axes[0].hist(
        original_pixels,
        bins=50,
    )
    axes[0].set_title(
        "Before Normalization",
        fontsize=12,
    )
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Number of Pixels")
    axes[0].set_xlim(0, 255)

    axes[1].hist(
        normalized_pixels,
        bins=50,
    )
    axes[1].set_title(
        "After Normalization",
        fontsize=12,
    )
    axes[1].set_xlabel("Normalized Pixel Value")
    axes[1].set_ylabel("Number of Pixels")
    axes[1].set_xlim(0.0, 1.0)

    formatter = FuncFormatter(format_millions)

    axes[0].yaxis.set_major_formatter(formatter)
    axes[1].yaxis.set_major_formatter(formatter)

    figure.suptitle(
        "Pixel-Value Distribution Before and After Normalization",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.9),
        w_pad=3.0,
    )
    # ---------------------------------
    # Save artifact
    # ---------------------------------

    figure.savefig(
        PIXEL_DISTRIBUTION_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure created successfully: "
        f"{PIXEL_DISTRIBUTION_FILE}"
    )


def create_normalization_comparison_figure(
    image_index: int = 0,) -> None:
    """
    Compare an original SVHN image with its normalized version.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    original_image = load_sample_training_image(image_index)

    normalized_image = original_image.astype(np.float32) / 255.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9, 4.8),
    )

    axes[0].imshow(
        original_image,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[0].set_title(
        "Original Image\nPixel range: 0–255",
        fontsize=12,
    )
    axes[0].axis("off")

    axes[1].imshow(
        normalized_image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(
        "Normalized Image\nPixel range: 0.0–1.0",
        fontsize=12,
    )
    axes[1].axis("off")

    figure.suptitle(
        "Effect of Pixel-Value Normalization",
        fontsize=15,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.9),
        w_pad=2.5,
    )

    figure.savefig(
        NORMALIZATION_COMPARISON_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        "Figure created successfully: "
        f"{NORMALIZATION_COMPARISON_FILE}"
    )

def create_preprocessing_figures() -> None:
    """
    Generate all figures used by the preprocessing report section.
    """
    create_normalization_comparison_figure()
    create_pixel_distribution_figure()

def main() -> None:
    create_preprocessing_figures()


if __name__ == "__main__":
    main()