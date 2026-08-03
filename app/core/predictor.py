"""
Inference logic for the YOLOv8 custom object detection API.

All prediction happens through ``Predictor.predict()`` — routes call
this, never run the model directly (docs/CODING_STANDARDS.md, rule 2).
Resize/normalize preprocessing lives in ``utils/image_utils.py``; this
module owns everything specific to running the model and turning raw
output into detections: session invocation, confidence filtering, NMS,
and rescaling boxes back to the original image size.
"""

from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.core.model import get_model, input_name, output_names
from app.utils.image_utils import preprocess_image, scale_boxes
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object, in original-image pixel coordinates."""

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


class Predictor:
    """Runs the active task's ONNX model against an input image."""

    def __init__(self) -> None:
        self.session = get_model()
        self.input_name = input_name()
        self.output_names = output_names()
        self.class_names = settings.class_names
        self.conf_threshold = settings.confidence_threshold
        self.image_size = settings.image_size

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR image array, return sorted detections."""
        original_shape = image.shape[:2]  # (h, w)

        try:
            input_tensor = preprocess_image(image, size=self.image_size)
        except Exception:
            logger.exception("Preprocessing failed for input image")
            raise

        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception:
            logger.exception("ONNX Runtime inference failed")
            raise

        detections = self._postprocess(outputs[0], original_shape)
        logger.info("Predicted %d detection(s) above threshold %.2f", len(detections), self.conf_threshold)
        return detections

    def _postprocess(self, raw_output: np.ndarray, original_shape: tuple[int, int]) -> list[Detection]:
        """Filter by confidence, apply NMS, and rescale boxes to the original image."""
        predictions = np.squeeze(raw_output).T  # (num_boxes, 4 + num_classes)
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        keep = confidences >= self.conf_threshold
        boxes_xywh, class_ids, confidences = boxes_xywh[keep], class_ids[keep], confidences[keep]
        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)
        keep_idx = self._nms(boxes_xyxy, confidences)
        scaled_boxes = scale_boxes(boxes_xyxy[keep_idx], self.image_size, original_shape)

        return [
            Detection(
                class_name=self.class_names[class_ids[i]],
                confidence=float(confidences[i]),
                box=tuple(scaled_boxes[j].tolist()),
            )
            for j, i in enumerate(keep_idx)
        ]

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convert center-format boxes (cx, cy, w, h) to corner-format (x1, y1, x2, y2)."""
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xyxy

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
        """Greedy non-max suppression; returns indices to keep, highest confidence first."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]
        return keep