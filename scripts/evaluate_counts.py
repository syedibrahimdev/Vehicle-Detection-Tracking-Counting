"""Score predicted counts against hand-counted ground truth.

    python scripts/evaluate_counts.py \
        --pred outputs/traffic_full_output_counts.json \
        --gt   data/ground_truth/traffic.json

Ground truth is a JSON file of the same shape as the pipeline's "counts":
    {"car": {"IN": 41, "OUT": 37}, "rickshaw": {"IN": 12, "OUT": 9}, ...}
Count by watching the clip and tallying each vehicle that crosses the line.
"""
import argparse
import json


def load_pred(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("counts", data)


def load_gt(path):
    with open(path) as f:
        return json.load(f)


def totals(counts, label):
    c = counts.get(label, {})
    return c.get("IN", 0) + c.get("OUT", 0)


def evaluate(pred, gt):
    """Returns (rows, overall). Each row: label, gt, pred, signed error, abs % error."""
    rows = []
    for label in sorted(set(pred) | set(gt)):
        g, p = totals(gt, label), totals(pred, label)
        pct = abs(p - g) / g * 100 if g else None
        rows.append({"label": label, "gt": g, "pred": p, "err": p - g, "abs_pct": pct})
    sum_gt = sum(r["gt"] for r in rows)
    sum_abs = sum(abs(r["err"]) for r in rows)
    overall = {
        "gt": sum_gt,
        "pred": sum(r["pred"] for r in rows),
        # sum of per-class absolute errors / total ground truth: an error in
        # one class cannot be hidden by an opposite error in another
        "weighted_abs_pct": sum_abs / sum_gt * 100 if sum_gt else None,
    }
    return rows, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    rows, overall = evaluate(load_pred(args.pred), load_gt(args.gt))

    print("| Class | Ground truth | Predicted | Error | Abs error % |")
    print("|---|---|---|---|---|")
    for r in rows:
        pct = f"{r['abs_pct']:.1f}%" if r["abs_pct"] is not None else "n/a"
        print(f"| {r['label']} | {r['gt']} | {r['pred']} | {r['err']:+d} | {pct} |")
    w = overall["weighted_abs_pct"]
    print(f"| **Overall** | {overall['gt']} | {overall['pred']} | "
          f"{overall['pred'] - overall['gt']:+d} | "
          f"{f'{w:.1f}%' if w is not None else 'n/a'} |")


if __name__ == "__main__":
    main()
