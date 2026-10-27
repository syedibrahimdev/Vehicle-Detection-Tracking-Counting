"""Hand-count ground truth for a clip, with the same counting line the pipeline uses.

    python scripts/tally_counts.py --video data/videos/clip1.mp4 --out data/ground_truth/clip1.json

Watch the clip and press a key each time a vehicle's centre crosses the line:

    lowercase = IN (crosses upward)        UPPERCASE = OUT (crosses downward)
    c / C car   m / M motorcycle   r / R rickshaw   t / T truck   b / B bus
    space  pause / resume        .  next frame (while paused)
    j      jump back 2 s (tallies from the skipped seconds are discarded, so you can recount them)
    u      undo last tally       q  save and quit

Turn Caps Lock OFF. Pedestrians are not counted. Writes <out> (input for evaluate_counts.py)
and <out>.events.json (one record per tally, with timestamps).
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.video_utils import get_video_capture, load_config, resize_to_width  # noqa: E402

CLASS_KEYS = {"c": "car", "m": "motorcycle", "r": "rickshaw", "t": "truck", "b": "bus"}
WINDOW = "tally (q = save and quit)"


def key_to_event(key):
    """cv2.waitKey code -> (class, 'IN'|'OUT') or None."""
    if key in (-1, 255):
        return None
    ch = chr(key & 0xFF)
    if ch.lower() not in CLASS_KEYS:
        return None
    return CLASS_KEYS[ch.lower()], ("OUT" if ch.isupper() else "IN")


def summarize(events):
    counts = {}
    for e in events:
        counts.setdefault(e["cls"], {"IN": 0, "OUT": 0})[e["dir"]] += 1
    return counts


def render(frame, line_ratio, events, seconds, paused):
    view = frame.copy()
    h, w = view.shape[:2]
    y = int(h * line_ratio)
    cv2.line(view, (0, y), (w, y), (0, 255, 0), 2)
    lines = [f"t={seconds:6.1f}s {'PAUSED' if paused else ''}",
             "lower=IN(up)  UPPER=OUT(down)  c m r t b  u=undo j=back q=quit"]
    for cls, c in sorted(summarize(events).items()):
        lines.append(f"{cls}: IN={c['IN']} OUT={c['OUT']}")
    cv2.rectangle(view, (0, 0), (560, 12 + 24 * len(lines)), (0, 0, 0), cv2.FILLED)
    for i, text in enumerate(lines):
        cv2.putText(view, text, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return view


def run(video, out_path, speed, line_ratio, width):
    cap = get_video_capture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps / speed))
    events, paused, step, frame, frame_idx = [], False, False, None, 0

    while True:
        if (not paused) or step or frame is None:
            ok, raw = cap.read()
            if not ok:
                break
            frame, frame_idx, step = resize_to_width(raw, width), frame_idx + 1, False

        cv2.imshow(WINDOW, render(frame, line_ratio, events, frame_idx / fps, paused))
        key = cv2.waitKey(30 if paused else delay)
        if key in (-1, 255):
            continue
        ch = chr(key & 0xFF)
        if ch == "q":
            break
        if ch == " ":
            paused = not paused
        elif ch == "." and paused:
            step = True
        elif ch == "j":
            frame_idx = max(0, frame_idx - int(2 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            events = [e for e in events if e["frame"] <= frame_idx]
            frame = None
        elif ch == "u":
            if events:
                events.pop()
        else:
            ev = key_to_event(key)
            if ev:
                events.append({"frame": frame_idx, "t": round(frame_idx / fps, 2),
                               "cls": ev[0], "dir": ev[1]})

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    counts = summarize(events)
    with open(out_path, "w") as f:
        json.dump(counts, f, indent=2)
    with open(out_path + ".events.json", "w") as f:
        json.dump({"video": video, "fps": fps, "events": events}, f, indent=1)
    print(f"Stopped at {frame_idx / fps:.1f}s. Saved {out_path}: {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, default=0.5, help="playback speed (default 0.5)")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(args.video, args.out, args.speed, cfg["counting"]["line_position"],
        cfg["video"]["resize_width"])


if __name__ == "__main__":
    main()