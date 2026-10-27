from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class _Track:
    start_side: int            # +1 = below the line, -1 = above/on it
    frames_seen: int = 1
    last_seen: int = 0
    counted: bool = False
    labels: Counter = field(default_factory=Counter)


class LineCounter:
    """Counts vehicles crossing one horizontal line, once per track ID.

    Differences from a naive "previous y vs current y" check:
    - Only classes in config `classes.count_classes` are counted
      (pedestrians are detected and drawn but never counted).
    - A track must be seen for `min_track_frames` before it can count, and
      the crossing is judged against the side the track STARTED on, so
      the delay never causes a missed crossing.
    - The class is the majority vote over the whole track, not the label
      of the single frame where it crossed (labels flicker, e.g. truck/car).
    - Per-track state is dropped after `stale_after_frames` unseen, so
      memory stays bounded on long streams. ByteTrack IDs are monotonic
      and never reused, so this cannot cause double counting.
    """

    def __init__(self, config):
        cnt_cfg = config["counting"]
        names = config["classes"]["names"]

        self.line_ratio = cnt_cfg["line_position"]
        self.label_up = cnt_cfg["direction_up"]
        self.label_down = cnt_cfg["direction_down"]
        self.min_track_frames = cnt_cfg.get("min_track_frames", 3)
        self.stale_after = cnt_cfg.get("stale_after_frames", 150)
        self.count_labels = {names[i] for i in config["classes"]["count_classes"]}

        self.counts = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self._tracks = {}
        self._frame = 0

    def update(self, boxes, frame_height):
        """Call once per frame with tracked boxes (track_id not None)."""
        self._frame += 1
        line_y = frame_height * self.line_ratio

        for box in boxes:
            track_id = box.get("track_id")
            label = box["label"]
            if track_id is None or label not in self.count_labels:
                continue

            y_center = (box["y1"] + box["y2"]) / 2
            side = 1 if y_center > line_y else -1

            track = self._tracks.get(track_id)
            if track is None:
                track = _Track(start_side=side)
                self._tracks[track_id] = track
            else:
                track.frames_seen += 1
            track.last_seen = self._frame
            track.labels[label] += 1

            if (track.counted or side == track.start_side
                    or track.frames_seen < self.min_track_frames):
                continue

            # started below, now above => moved up
            direction = self.label_up if track.start_side == 1 else self.label_down
            self.counts[track.labels.most_common(1)[0][0]][direction] += 1
            track.counted = True

        self._prune()

    def _prune(self):
        if self._frame % 100:
            return
        cutoff = self._frame - self.stale_after
        for tid in [t for t, s in self._tracks.items() if s.last_seen < cutoff]:
            del self._tracks[tid]

    def get_counts(self):
        return {k: dict(v) for k, v in self.counts.items()}

    def total(self):
        return sum(v["IN"] + v["OUT"] for v in self.counts.values())
