import cv2
import yaml
import os


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