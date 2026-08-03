"""
Image decode/resize/normalize helpers for this project's model.

This model expects a 640x640 input (IMAGE_SIZE below — kept in sync
with app.config.settings.image_size), RGB channel order, pixel values
scaled to [0, 1], and NCHW layout with a batch dimension. Unlike a
plain classifier resize, YOLOv8 was trained on letterboxed inputs
(aspect ratio preserved, padded with grey) rather than a stretched
resize — stretching here would distort box shapes at inference time.
"""

import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 640  # must match app.config.settings.image_size
PAD_COLOR = (114, 114, 114)  # standard YOLOv8 letterbox padding color


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR image array (OpenCV convention)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)[:, :, ::-1]  # RGB -> BGR


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size, preserving aspect ratio, padding with grey."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = round(h * scale), round(w * scale)

    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    pad_h, pad_w = size - new_h, size - new_w
    top, left = pad_h // 2, pad_w // 2

    padded = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    padded[top : top + new_h, left : left + new_w] = resized
    return padded


def preprocess_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Letterbox-resize a BGR image array and prepare it for the model.

    Converts BGR -> RGB, resizes with aspect-ratio-preserving padding,
    scales pixels to [0, 1], and returns an NCHW float32 tensor with a
    batch dimension — ready to feed straight to the ONNX session.
    """
    rgb = image[:, :, ::-1]  # BGR (OpenCV/read order) -> RGB
    padded = _letterbox(rgb, size)

    normalized = padded.astype(np.float32) / 255.0
    chw = normalized.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(chw, axis=0)  # add batch dim -> NCHW


def scale_boxes(boxes: np.ndarray, model_size: int, original_shape: tuple[int, int]) -> np.ndarray:
    """Map boxes from letterboxed model-input space back to the original image.

    original_shape is (height, width) of the source image before preprocessing.
    """
    orig_h, orig_w = original_shape
    scale = min(model_size / orig_h, model_size / orig_w)
    pad_h, pad_w = model_size - round(orig_h * scale), model_size - round(orig_w * scale)
    top, left = pad_h // 2, pad_w // 2

    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - left) / scale
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - top) / scale

    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, orig_w)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, orig_h)
    return scaled