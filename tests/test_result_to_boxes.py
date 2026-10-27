import numpy as np

from src.detection.yolo_detector import result_to_boxes


class _T:  # minimal stand-in for a torch tensor
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
        self.id = _T(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy.a)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


NAMES = {0: "bus", 1: "car"}


def test_detection_result_has_no_track_ids():
    r = _Result(_Boxes([[1, 2, 30, 40]], [0.9], [1]))
    (b,) = result_to_boxes(r, NAMES)
    assert b["label"] == "car" and b["track_id"] is None
    assert (b["x1"], b["y1"], b["x2"], b["y2"]) == (1, 2, 30, 40)


def test_tracked_result_carries_ids():
    r = _Result(_Boxes([[0, 0, 5, 5], [1, 1, 6, 6]], [0.8, 0.7], [0, 1], ids=[7, 8]))
    assert [b["track_id"] for b in result_to_boxes(r, NAMES)] == [7, 8]


def test_empty_and_none_results():
    assert result_to_boxes(_Result(None), NAMES) == []
    assert result_to_boxes(_Result(_Boxes(np.empty((0, 4)), [], [])), NAMES) == []


def test_unknown_class_id_gets_fallback_label():
    r = _Result(_Boxes([[0, 0, 5, 5]], [0.8], [9]))
    assert result_to_boxes(r, NAMES)[0]["label"] == "class_9"
