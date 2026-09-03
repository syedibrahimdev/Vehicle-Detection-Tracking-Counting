from ultralytics import YOLO


class YoloDetector:
    """
    Wraps Ultralytics YOLOv8 for vehicle detection.

    WHY A WRAPPER CLASS INSTEAD OF CALLING YOLO() DIRECTLY IN THE SCRIPT:
    Same reason as ClassicalDetector — keeps run_yolo.py clean, and lets
    us swap the underlying model (pretrained -> fine-tuned) without
    touching the script that calls it. Also gives classical and YOLO
    detectors the same detect(frame) -> boxes interface, so counter.py
    and tracker.py can work with either one later.
    """

    def __init__(self, config):
        det_cfg = config["detection"]
        paths_cfg = config["paths"]

        # Use fine-tuned weights if they exist, otherwise fall back to
        # the plain pretrained model. This means this exact file keeps
        # working before AND after you finish fine-tuning in Kaggle.
        import os
        weights_path = paths_cfg["model_weights"]
        if not os.path.exists(weights_path):
            print(f"[YoloDetector] Fine-tuned weights not found at "
                  f"{weights_path}, falling back to pretrained "
                  f"{paths_cfg['pretrained_weights']}")
            weights_path = paths_cfg["pretrained_weights"]

        self.model = YOLO(weights_path)

        self.conf_threshold = det_cfg["confidence_threshold"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.input_size = det_cfg["input_size"]
        self.device = det_cfg["device"]

        # class-index -> name mapping, only used when we're on the
        # fine-tuned model (pretrained COCO model has its own names,
        # accessed via self.model.names instead — see detect() below)
        self.class_names = config["classes"]["names"]
        self.using_finetuned = "best" in os.path.basename(weights_path)

    def detect(self, frame):
        """
        Runs YOLOv8 inference on a single frame.
        Returns a list of boxes in the SAME dict format ClassicalDetector
        uses, so the rest of the pipeline (visualizer, counter, tracker)
        doesn't need to know which detector produced them.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_size,
            device=self.device,
            verbose=False   # suppress ultralytics' own per-frame console spam
        )

        boxes = []
        result = results[0]  # predict() on a single frame returns a list of length 1

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            # If using the fine-tuned model, its own names (from data.yaml)
            # are authoritative. If still on pretrained COCO, use the
            # model's built-in COCO names instead of our custom class list,
            # since COCO doesn't have "rickshaw" etc.
            if self.using_finetuned:
                label = self.class_names.get(class_id, f"class_{class_id}")
            else:
                label = self.model.names.get(class_id, f"class_{class_id}")

            boxes.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "label": label,
                "confidence": confidence,
                "track_id": None,   # tracking comes in Phase 3
                "class_id": class_id
            })

        return boxes