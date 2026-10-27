import cv2
import yaml
import os
import argparse

def load_config(config_path="configs/config.yaml"):
    """
    Loads the YAML config file and returns it as a Python dictionary.
    We use a config file instead of hardcoding values so we can change
    settings (like confidence threshold or line position) without touching
    the actual code — just edit the YAML and re-run.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_video_capture(video_path):
    """
    Opens a video file using OpenCV and returns the capture object.
    cv2.VideoCapture is OpenCV's way of reading video — it gives us
    access to individual frames one by one, which is how all video
    processing works: frame by frame, not the whole video at once.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    return cap


def get_video_properties(cap):
    """
    Extracts basic properties from the video capture object.
    We need these to:
    - fps: know the real speed of the video (frames per second)
    - width/height: know frame dimensions for drawing lines and boxes
    - total_frames: show progress while processing
    """
    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return props


def get_video_writer(output_path, fps, width, height):
    """
    Creates a VideoWriter object to save the processed video to disk.
    We use the 'mp4v' codec which produces a standard .mp4 file.
    The writer needs to know fps and frame size upfront because video
    files store this metadata in their header before any frames.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Could not create video writer at: {output_path}")

    return writer


def resize_frame(frame, width, height):
    """
    Resizes a frame to the target dimensions.
    We resize for two reasons:
    1. Speed: smaller frames = faster processing per frame
    2. Consistency: all frames same size regardless of source video resolution
    cv2.INTER_LINEAR is a standard interpolation method — good balance
    between quality and speed for downscaling.
    """
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def release_resources(*resources):
    """
    Releases video capture and writer objects when we are done.
    This is important — if you don't release, the video file stays
    locked/open and the output file may be corrupted or unplayable.
    Always call this at the end of any video processing script.
    """
    for resource in resources:
        if resource is not None:
            resource.release()
    cv2.destroyAllWindows()

def get_video_path_from_args(config, default_key="video_input"):
    """
    Lets us pick which video to process from the command line, e.g.:
        python scripts/run_classical.py --video data/videos/traffic2.mp4
    If no --video is passed, it falls back to config.yaml's default path.

    WHY WE NEED THIS:
    You have two test videos (traffic.mp4, traffic2.mp4) and you'll want
    to re-run every phase (classical, YOLO, tracking, counting) on both.
    Editing config.yaml every time you switch videos is slow and easy to
    forget. A CLI flag lets you run both in two commands without touching
    any file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video. Overrides config.yaml's paths.video_input"
    )
    args, _ = parser.parse_known_args()
    # parse_known_args (not parse_args) so this doesn't break if a script
    # adds its own extra CLI args later — unknown ones are just ignored here.

    if args.video:
        return args.video
    return config["paths"][default_key]


def get_output_path(output_dir, video_path, suffix):
    """
    Builds an output filename that includes the input video's name, so
    results from traffic.mp4 and traffic2.mp4 never overwrite each other.

    Example:
        video_path = "data/videos/traffic2.mp4", suffix = "classical_output"
        -> "outputs/traffic2_classical_output.mp4"
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"{video_name}_{suffix}.mp4"
    return os.path.join(output_dir, filename)

def resize_to_width(frame, width):
    """
    Resizes to a target width while KEEPING the aspect ratio (height is
    rounded down to an even number, which mp4 encoders require). The old
    fixed 1280x720 resize stretched any non-16:9 video.
    """
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(round(h * width / w))
    new_h -= new_h % 2
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, new_h), interpolation=interp)
