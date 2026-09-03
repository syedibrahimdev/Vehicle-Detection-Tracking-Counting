import sys
import os
import time

# Add project root to Python's search path so "from src.xxx import yyy" works
# no matter where this script is called from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.video_utils import (
    load_config, get_video_capture, get_video_properties,
    get_video_writer, resize_frame, release_resources,
    get_video_path_from_args, get_output_path
)
from src.utils.visualizer import (
    draw_detection, draw_counting_line, draw_count_display, draw_fps
)
from src.classical.detector import ClassicalDetector


def main():
    # 1. Load settings
    # Everything (paths, thresholds, resize dims) comes from config.yaml
    # so we never hardcode values here.
    config = load_config("configs/config.yaml")

    video_path = get_video_path_from_args(config)

    # 2. Open input video
    cap = get_video_capture(video_path)
    props = get_video_properties(cap)
    print(f"Input video: {video_path}")
    print(f"Original size: {props['width']}x{props['height']}, "
          f"FPS: {props['fps']:.1f}, Total frames: {props['total_frames']}")

    # 3. Initialize the classical detector
    # This sets up MOG2, morphology kernels, and area thresholds
    # using the parameters we defined in config.yaml under "classical".
    detector = ClassicalDetector(config)

    # 4. Prepare output video writer
    # We write the resized dimensions (not original) because every
    # frame we process and draw on is resized first.
    out_path = get_output_path(config["paths"]["output_dir"], video_path, "classical_output")
    writer = get_video_writer(
        out_path,
        fps=props["fps"] if props["fps"] > 0 else 30,
        width=config["video"]["resize_width"],
        height=config["video"]["resize_height"]
    )

    # 5. Main processing loop
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = resize_frame(frame, config["video"]["resize_width"], config["video"]["resize_height"])

            boxes, mask = detector.detect(frame)

            for box in boxes:
                draw_detection(
                    frame,
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    label=box["label"],
                    confidence=box["confidence"],
                    track_id=box["track_id"],
                    color=(0, 255, 0),
                    show_confidence=False,
                    show_track_id=False
                )

            draw_counting_line(frame, config["counting"]["line_position"])

            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            draw_fps(frame, current_fps)

            writer.write(frame)

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{props['total_frames']} frames "
                      f"| Detections this frame: {len(boxes)} "
                      f"| Processing FPS: {current_fps:.1f}")

    except KeyboardInterrupt:
        print(f"\nInterrupted by user at frame {frame_count}. Saving what we have so far...")

    finally:
        # This block ALWAYS runs — whether the loop finished normally,
        # crashed, or was interrupted with Ctrl+C. Without this, an
        # interrupted run leaves a corrupted, unplayable mp4 because
        # VideoWriter never got to finalize the file.
        release_resources(cap, writer)

    total_time = time.time() - start_time
    print(f"\nDone. Processed {frame_count} frames in {total_time:.1f} seconds "
          f"({frame_count/total_time:.1f} FPS average)")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    # This guard means the code only runs when you execute this file
    # directly (python scripts/run_classical.py), not when it's imported
    # elsewhere. Standard Python convention for entry-point scripts.
    main()