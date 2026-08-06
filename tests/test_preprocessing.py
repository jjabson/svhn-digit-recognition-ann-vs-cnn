from io import BytesIO

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image_bytes


def create_test_image() -> bytes:
    image = Image.new(
        mode="L",
        size=(32, 32),
        color=128
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def test_preprocessing_shape():
    image_bytes = create_test_image()
    result = preprocess_image_bytes(image_bytes)

    assert result.shape == (1, 32, 32, 1)


def test_preprocessing_dtype():
    image_bytes = create_test_image()
    result = preprocess_image_bytes(image_bytes)

    assert result.dtype == np.float32


def test_preprocessing_range():
    image_bytes = create_test_image()
    result = preprocess_image_bytes(image_bytes)

    assert result.min() >= 0.0
    assert result.max() <= 1.0