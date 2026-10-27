from src.detection.yolo_detector import result_to_boxes


class VehicleTracker:
    """Adds persistent per-vehicle IDs using Ultralytics' built-in tracker.

    `tracker_cfg` is "bytetrack.yaml" by default; point it at your own
    YAML (e.g. configs/bytetrack_tuned.yaml) to tune track_buffer /
    match_thresh, or at "botsort.yaml" to compare trackers.
    """

    def __init__(self, config, yolo_detector):
        det_cfg = config["detection"]
        self.model = yolo_detector.model
        self.class_names = yolo_detector.class_names
        self.conf_threshold = det_cfg["confidence_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.input_size = det_cfg["input_size"]
        self.device = det_cfg["device"]
        self.tracker_cfg = config.get("tracking", {}).get(
            "tracker", "bytetrack.yaml")

    def track(self, frame):
        # persist=True keeps tracker state between calls; without it every
        # frame restarts IDs and tracking silently becomes plain detection.
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )
        # Boxes without a confirmed track ID are dropped: the counter only
        # works on tracked objects.
        return [b for b in result_to_boxes(results[0], self.class_names)
                if b["track_id"] is not None]
