"""Single entry point for all three stages.

    python scripts/run_pipeline.py --mode classical --video data/videos/traffic.mp4
    python scripts/run_pipeline.py --mode yolo      --video data/videos/traffic.mp4
    python scripts/run_pipeline.py --mode full      --video data/videos/traffic.mp4

`full` = YOLO detection + tracking + line counting, and also writes a
counts JSON next to the video that scripts/evaluate_counts.py can score.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.video_utils import (  # noqa: E402
    load_config, get_video_capture, get_video_properties, get_video_writer,
    resize_to_width, release_resources, get_output_path,
)
from src.utils.visualizer import (  # noqa: E402
    draw_detection, draw_counting_line, draw_count_display, draw_fps,
)

logger = logging.getLogger("pipeline")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["classical", "yolo", "full"], default="full")
    p.add_argument("--video", default=None,
                   help="Input video (default: paths.video_input in the config)")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--no-video", action="store_true",
                   help="Skip writing the output video (faster; for benchmarking)")
    return p.parse_args()


def build_stage(mode, config):
    """Returns (step_fn, counter_or_None). step_fn(frame) -> list of boxes."""
    if mode == "classical":
        from src.classical.detector import ClassicalDetector
        detector = ClassicalDetector(config)
        return (lambda frame: detector.detect(frame)[0]), None

    from src.detection.yolo_detector import YoloDetector
    detector = YoloDetector(config)
    logger.info("Weights: %s", detector.weights_path)
    if mode == "yolo":
        return detector.detect, None

    from src.tracking.tracker import VehicleTracker
    from src.counting.counter import LineCounter
    return VehicleTracker(config, detector).track, LineCounter(config)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    video_path = args.video or config["paths"]["video_input"]
    vis = config["visualization"]
    colors = vis["colors"]
    width = config["video"]["resize_width"]

    cap = get_video_capture(video_path)
    props = get_video_properties(cap)
    logger.info("Input %s: %dx%d @ %.1f fps, %d frames", video_path,
                props["width"], props["height"], props["fps"], props["total_frames"])

    step, counter = build_stage(args.mode, config)

    out_path = get_output_path(config["paths"]["output_dir"], video_path,
                               f"{args.mode}_output")
    writer = None
    frames = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            frame = resize_to_width(frame, width)
            height = frame.shape[0]

            boxes = step(frame)
            if counter is not None:
                counter.update(boxes, height)

            if not args.no_video:
                if writer is None:  # size known only after the first resize
                    writer = get_video_writer(
                        out_path, props["fps"] if props["fps"] > 0 else 30,
                        frame.shape[1], height)
                for b in boxes:
                    draw_detection(
                        frame, b["x1"], b["y1"], b["x2"], b["y2"],
                        label=b["label"], confidence=b["confidence"],
                        track_id=b["track_id"],
                        color=tuple(colors.get(b["label"], [0, 255, 0])),
                        show_confidence=vis["show_confidence"] and args.mode != "classical",
                        show_track_id=vis["show_track_id"] and counter is not None)
                if args.mode != "classical":
                    draw_counting_line(frame, config["counting"]["line_position"])
                if counter is not None:
                    draw_count_display(frame, counter.get_counts())
                draw_fps(frame, frames / (time.perf_counter() - t0))
                writer.write(frame)

            if frames % 30 == 0:
                logger.info("frame %d/%d | %.1f fps", frames, props["total_frames"],
                            frames / (time.perf_counter() - t0))
    except KeyboardInterrupt:
        logger.warning("Interrupted at frame %d; finalising output", frames)
    finally:
        release_resources(cap, writer)

    elapsed = time.perf_counter() - t0
    fps = frames / elapsed if elapsed else 0.0
    logger.info("Done: %d frames in %.1fs (%.1f fps)", frames, elapsed, fps)
    if not args.no_video:
        logger.info("Video: %s", out_path)

    if counter is not None:
        result = {
            "video": video_path, "mode": args.mode, "frames": frames,
            "processing_fps": round(fps, 2),
            "total": counter.total(), "counts": counter.get_counts(),
        }
        json_path = out_path.replace(".mp4", "_counts.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Counts: %s -> %s", result["counts"], json_path)


if __name__ == "__main__":
    main()
