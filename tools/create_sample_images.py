"""Create one sample PNG image for each SVHN digit class.

This utility reads the SVHN HDF5 dataset, selects one test image
for each label from 0 through 9, and saves the images as PNG files.
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
from PIL import Image


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image array into values suitable for a PNG file.

    The HDF5 dataset may contain either:
    - pixel values from 0 to 255, or
    - normalized pixel values from 0 to 1.
    """
    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim not in (2, 3):
        raise ValueError(
            f"Expected a 2D grayscale or 3D color image, "
            f"but received shape {image.shape}."
        )

    # Handle normalized floating-point images.
    if np.issubdtype(image.dtype, np.floating):
        if image.min() >= 0 and image.max() <= 1:
            image = image * 255

    # Keep all pixel values in the valid PNG range.
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Convert labels into a flat integer array."""
    labels = np.asarray(labels).squeeze().astype(int)

    # Some versions of SVHN use label 10 to represent digit 0.
    labels = np.where(labels == 10, 0, labels)

    return labels


def save_sample_images(
    dataset_path: Path,
    output_dir: Path,
    image_key: str = "X_test",
    label_key: str = "y_test",
) -> None:
    """Save one test image for each digit label from 0 through 9."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Copy the HDF5 file into the data directory or provide "
            "the correct path with --dataset."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as h5_file:
        available_keys = list(h5_file.keys())

        print(f"Available HDF5 keys: {available_keys}")

        if image_key not in h5_file:
            raise KeyError(
                f"Image key '{image_key}' was not found. "
                f"Available keys: {available_keys}"
            )

        if label_key not in h5_file:
            raise KeyError(
                f"Label key '{label_key}' was not found. "
                f"Available keys: {available_keys}"
            )

        images = h5_file[image_key][:]
        labels = normalize_labels(h5_file[label_key][:])

    print(f"Image array shape: {images.shape}")
    print(f"Label array shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    if len(images) != len(labels):
        raise ValueError(
            f"Number of images ({len(images)}) does not match "
            f"number of labels ({len(labels)})."
        )

    saved_count = 0

    for digit in range(10):
        matching_indices = np.where(labels == digit)[0]

        if len(matching_indices) == 0:
            print(f"Warning: no test image found for digit {digit}.")
            continue

        image_index = int(matching_indices[0])
        image = convert_to_uint8(images[image_index])

        output_path = output_dir / f"digit_{digit}_true{digit}.png"

        Image.fromarray(image).save(output_path)

        print(
            f"Saved digit {digit}: "
            f"test index {image_index} -> {output_path}"
        )

        saved_count += 1

    print(f"\nFinished. Saved {saved_count} sample images to {output_dir}.")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create one PNG test image for each SVHN digit."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/SVHN_single_grey1.h5"),
        help="Path to the SVHN HDF5 dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_images"),
        help="Directory where the PNG files will be saved.",
    )

    parser.add_argument(
        "--image-key",
        default="X_test",
        help="HDF5 key containing test images.",
    )

    parser.add_argument(
        "--label-key",
        default="y_test",
        help="HDF5 key containing test labels.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the sample-image creation utility."""
    args = parse_arguments()

    save_sample_images(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        image_key=args.image_key,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()