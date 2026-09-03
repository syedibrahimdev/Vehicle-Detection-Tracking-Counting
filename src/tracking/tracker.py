class VehicleTracker:
    """
    Adds persistent per-vehicle IDs on top of an existing YoloDetector's
    model using Ultralytics' built-in ByteTrack integration.

    We don't pull in a separate bytetrack/deep_sort package. Ultralytics
    already ships ByteTrack via model.track(tracker="bytetrack.yaml") --
    same weights, same call pattern as detect(), one extra argument.
    """

    def __init__(self, config, yolo_detector):
        det_cfg = config["detection"]
        self.model = yolo_detector.model
        self.conf_threshold = det_cfg["confidence_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.input_size = det_cfg["input_size"]
        self.device = det_cfg["device"]
        self.class_names = yolo_detector.class_names
        self.using_finetuned = yolo_detector.using_finetuned

    def track(self, frame):
        """
        Runs detection + ByteTrack association on one frame.
        persist=True keeps the tracker's internal state (track history,
        Kalman filters, ID assignment) between calls -- that state IS
        the tracker. Without it, every frame gets a fresh ID and
        tracking silently degrades into plain detection.
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        boxes = []
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            # No confirmed tracks this frame -- expected on frame 1 or
            # when nothing was detected, not an error.
            return boxes

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            track_id = int(box.id[0])

            if self.using_finetuned:
                label = self.class_names.get(class_id, f"class_{class_id}")
            else:
                label = self.model.names.get(class_id, f"class_{class_id}")

            boxes.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "label": label,
                "confidence": confidence,
                "track_id": track_id,
                "class_id": class_id,
            })

        return boxes