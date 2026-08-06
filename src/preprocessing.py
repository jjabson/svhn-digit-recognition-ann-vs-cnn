# The API performs best on tightly cropped, SVHN-like digit images.
# Results may be unreliable for handwritten digits, multi-digit images,
# uncropped photographs, or images with substantially different contrast.

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SIZE = (32, 32)


class InvalidImageError(ValueError):
    """Raised when uploaded content cannot be decoded as an image."""


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes into a normalized CNN input tensor.

    Returns
    -------
    np.ndarray
        Array with shape (1, 32, 32, 1) and float32 values
        in the range [0, 1].
    """

    if not image_bytes:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc

    image = image.convert("L")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    return image_array.reshape(1, 32, 32, 1)