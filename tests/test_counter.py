import pytest

from src.counting.counter import LineCounter

NAMES = {0: "bus", 1: "car", 2: "motorcycle", 3: "person", 4: "rickshaw", 5: "truck"}
H = 100  # frame height; line at y=50


def make_counter(**overrides):
    cnt = {"line_position": 0.5, "direction_up": "IN", "direction_down": "OUT",
           "min_track_frames": 2, "stale_after_frames": 10}
    cnt.update(overrides)
    return LineCounter({"counting": cnt,
                        "classes": {"names": NAMES, "count_classes": [0, 1, 2, 4, 5]}})


def box(track_id, y, label="car"):
    return {"track_id": track_id, "label": label,
            "x1": 0, "x2": 10, "y1": y - 5, "y2": y + 5}


def feed(counter, track_id, ys, label="car"):
    for y in ys:
        counter.update([box(track_id, y, label)], H)


def test_upward_crossing_counts_once_as_in():
    c = make_counter()
    feed(c, 1, [70, 60, 40, 30, 20])
    assert c.get_counts() == {"car": {"IN": 1, "OUT": 0}}


def test_downward_crossing_counts_as_out():
    c = make_counter()
    feed(c, 1, [30, 40, 60, 70])
    assert c.get_counts() == {"car": {"IN": 0, "OUT": 1}}


def test_jitter_back_and_forth_counts_only_once():
    c = make_counter()
    feed(c, 1, [70, 40, 60, 40, 60, 40])
    assert c.total() == 1


def test_track_that_never_crosses_is_not_counted():
    c = make_counter()
    feed(c, 1, [90, 80, 70, 60])
    assert c.total() == 0


def test_person_is_never_counted():
    c = make_counter()
    feed(c, 1, [70, 60, 40, 30], label="person")
    assert c.total() == 0


def test_class_is_majority_vote_not_crossing_frame_label():
    c = make_counter()
    for y, label in [(80, "truck"), (70, "truck"), (60, "truck"), (40, "car")]:
        c.update([box(1, y, label)], H)
    assert c.get_counts() == {"truck": {"IN": 1, "OUT": 0}}


def test_crossing_before_min_track_frames_is_counted_once_track_matures():
    c = make_counter(min_track_frames=3)
    feed(c, 1, [60, 40])          # crossed on frame 2, too young
    assert c.total() == 0
    feed(c, 1, [30])              # frame 3: now eligible
    assert c.get_counts() == {"car": {"IN": 1, "OUT": 0}}


def test_two_tracks_counted_independently():
    c = make_counter()
    for y1, y2 in [(70, 30), (60, 40), (40, 60), (30, 70)]:
        c.update([box(1, y1), box(2, y2)], H)
    assert c.get_counts() == {"car": {"IN": 1, "OUT": 1}}


def test_untracked_boxes_ignored():
    c = make_counter()
    c.update([{"track_id": None, "label": "car", "x1": 0, "x2": 1, "y1": 0, "y2": 1}], H)
    assert c.total() == 0


def test_stale_tracks_are_pruned():
    c = make_counter(stale_after_frames=10)
    feed(c, 1, [70, 60, 40])
    assert 1 in c._tracks
    for _ in range(120):
        c.update([], H)
    assert 1 not in c._tracks
