import logging
from pathlib import Path

from src.detection.labels import resolve_class_names

logger = logging.getLogger(__name__)


def result_to_boxes(result, class_names):
    """Convert one Ultralytics result into the project's box dicts.

    Shared by YoloDetector.detect() and VehicleTracker.track() so the
    two cannot drift apart. `track_id` is None unless the result came
    from model.track() and the tracker assigned IDs.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    out = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        class_id = int(classes[i])
        out.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "label": class_names.get(class_id, f"class_{class_id}"),
            "confidence": float(confs[i]),
            "track_id": int(ids[i]) if ids is not None else None,
            "class_id": class_id,
        })
    return out


class YoloDetector:
    """Wraps Ultralytics YOLO. detect(frame) -> list of box dicts."""

    def __init__(self, config):
        # Imported here so classical mode and the unit tests do not need
        # torch/ultralytics installed.
        from ultralytics import YOLO

        det_cfg = config["detection"]
        paths_cfg = config["paths"]

        weights = Path(paths_cfg["model_weights"])
        if weights.exists():
            weights_path = weights
        elif det_cfg.get("allow_pretrained_fallback", False):
            weights_path = Path(paths_cfg["pretrained_weights"])
            logger.warning(
                "Fine-tuned weights not found at %s; falling back to %s. "
                "Results will NOT reflect the fine-tuned model "
                "(no rickshaw class).", weights, weights_path)
        else:
            raise FileNotFoundError(
                f"Fine-tuned weights not found at '{weights}'. Download "
                "best.pt from the GitHub release into that path, or set "
                "detection.allow_pretrained_fallback: true to use COCO "
                "weights deliberately.")

        self.weights_path = str(weights_path)
        self.model = YOLO(self.weights_path)
        self.class_names = resolve_class_names(
            self.model.names, config["classes"]["names"])

        self.conf_threshold = det_cfg["confidence_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.input_size = det_cfg["input_size"]
        self.device = det_cfg["device"]

    def detect(self, frame):
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        return result_to_boxes(results[0], self.class_names)
