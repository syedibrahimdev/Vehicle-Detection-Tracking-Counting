from collections import defaultdict


class LineCounter:
    """
    Counts vehicles crossing a single horizontal virtual line, once per
    track ID, split by class and direction.

    We compare previous vs current y-center against the line instead of
    checking "is the box on the line right now" -- a box's center only
    touches the exact line pixel for a frame or two, so a single-frame
    check would miss most crossings. Comparing the two sides catches
    every crossing exactly once, regardless of frame rate.
    """

    def __init__(self, config):
        cnt_cfg = config["counting"]
        self.line_ratio = cnt_cfg["line_position"]
        self.direction_up_label = cnt_cfg["direction_up"]
        self.direction_down_label = cnt_cfg["direction_down"]

        self.counts = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self._last_y = {}          # track_id -> last known y-center
        self._counted_ids = set()  # track_ids already counted once

    def update(self, boxes, frame_height):
        """
        Call once per frame with the current frame's tracked detections
        (each box needs a non-None track_id -- this consumes
        VehicleTracker output, not raw detector output).
        """
        line_y = int(frame_height * self.line_ratio)

        for box in boxes:
            track_id = box["track_id"]
            if track_id is None:
                continue

            y_center = (box["y1"] + box["y2"]) / 2
            previous_y = self._last_y.get(track_id)
            self._last_y[track_id] = y_center

            if previous_y is None or track_id in self._counted_ids:
                continue

            crossed_upward = previous_y > line_y >= y_center
            crossed_downward = previous_y < line_y <= y_center

            if crossed_upward or crossed_downward:
                direction = self.direction_up_label if crossed_upward else self.direction_down_label
                self.counts[box["label"]][direction] += 1
                self._counted_ids.add(track_id)

    def get_counts(self):
        return dict(self.counts)

    def total(self):
        return sum(v["IN"] + v["OUT"] for v in self.counts.values())