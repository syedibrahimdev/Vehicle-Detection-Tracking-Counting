import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.video_utils import (
    load_config, get_video_capture, get_video_properties,
    get_video_writer, resize_frame, release_resources,
    get_video_path_from_args, get_output_path
)
from src.utils.visualizer import (
    draw_detection, draw_counting_line, draw_count_display, draw_fps
)
from src.detection.yolo_detector import YoloDetector
from src.tracking.tracker import VehicleTracker
from src.counting.counter import LineCounter


def main():
    config = load_config("configs/config.yaml")
    video_path = get_video_path_from_args(config)

    cap = get_video_capture(video_path)
    props = get_video_properties(cap)
    print(f"Input video: {video_path}")
    print(f"Original size: {props['width']}x{props['height']}, "
          f"FPS: {props['fps']:.1f}, Total frames: {props['total_frames']}")

    detector = YoloDetector(config)
    tracker = VehicleTracker(config, detector)
    counter = LineCounter(config)
    colors = config["visualization"]["colors"]

    out_path = get_output_path(config["paths"]["output_dir"], video_path, "full_pipeline_output")
    writer = get_video_writer(
        out_path,
        fps=props["fps"] if props["fps"] > 0 else 30,
        width=config["video"]["resize_width"],
        height=config["video"]["resize_height"]
    )

    frame_count = 0
    last_boxes = []
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = resize_frame(frame, config["video"]["resize_width"], config["video"]["resize_height"])

            boxes = tracker.track(frame)
            counter.update(boxes, frame.shape[0])
            last_boxes = boxes

            for box in last_boxes:
                color = tuple(colors.get(box["label"], [0, 255, 0]))
                draw_detection(
                    frame,
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    label=box["label"],
                    confidence=box["confidence"],
                    track_id=box["track_id"],
                    color=color,
                    show_confidence=config["visualization"]["show_confidence"],
                    show_track_id=config["visualization"]["show_track_id"]
                )

            draw_counting_line(frame, config["counting"]["line_position"])
            draw_count_display(frame, counter.get_counts())

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            draw_fps(frame, current_fps)

            writer.write(frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{props['total_frames']} frames "
                      f"| Active tracks: {len(last_boxes)} "
                      f"| Total counted: {counter.total()} "
                      f"| Processing FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at frame {frame_count}. Saving what we have so far...")

    finally:
        release_resources(cap, writer)

    total_time = time.time() - start_time
    print(f"\nDone. Processed {frame_count} frames in {total_time:.1f} seconds "
          f"({frame_count/total_time:.1f} FPS average)")
    print(f"Final counts: {counter.get_counts()}")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()