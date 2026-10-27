from scripts.evaluate_counts import evaluate


def test_per_class_and_overall_error():
    gt = {"car": {"IN": 10, "OUT": 10}, "truck": {"IN": 5, "OUT": 0}}
    pred = {"car": {"IN": 9, "OUT": 9}, "truck": {"IN": 8, "OUT": 0}}
    rows, overall = evaluate(pred, gt)
    by = {r["label"]: r for r in rows}
    assert by["car"]["err"] == -2 and by["car"]["abs_pct"] == 10.0
    assert by["truck"]["err"] == 3
    assert overall["weighted_abs_pct"] == (2 + 3) / 25 * 100


def test_opposite_errors_do_not_cancel():
    gt = {"car": {"IN": 10, "OUT": 0}, "bus": {"IN": 10, "OUT": 0}}
    pred = {"car": {"IN": 5, "OUT": 0}, "bus": {"IN": 15, "OUT": 0}}
    _, overall = evaluate(pred, gt)
    assert overall["pred"] == overall["gt"]
    assert overall["weighted_abs_pct"] == 50.0


def test_class_missing_from_gt_has_no_percentage():
    rows, _ = evaluate({"person": {"IN": 3, "OUT": 0}}, {})
    assert rows[0]["abs_pct"] is None
