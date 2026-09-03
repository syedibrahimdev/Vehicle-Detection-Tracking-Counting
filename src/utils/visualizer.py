import cv2
import numpy as np


def draw_bounding_box(frame, x1, y1, x2, y2, color, thickness=2):
    """
    Draws a rectangle around a detected vehicle.
    (x1,y1) is top-left corner, (x2,y2) is bottom-right corner.
    These coordinates come from the detector — either contours
    (Phase 1) or YOLOv8 (Phase 2). Same drawing function works
    for both, which is why it's here in utils.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame, text, x1, y1, color, font_scale=0.6, thickness=2):
    """
    Draws a filled rectangle behind the text first, then puts the
    text on top. The background rectangle makes text readable on
    any background — without it, white text on light road = invisible.
    We calculate the text size first so the background fits perfectly.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get the pixel width and height of the text before drawing
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # draw filled rectangle as background behind text
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w + 4, y1),
        color,
        cv2.FILLED
    )

    # draw the text itself in white on top of colored background
    cv2.putText(
        frame, text,
        (x1 + 2, y1 - baseline - 2),
        font, font_scale,
        (255, 255, 255),  # white text always readable on any color bg
        thickness,
        cv2.LINE_AA  # anti-aliased — smoother edges on text
    )


def draw_detection(frame, x1, y1, x2, y2, label, confidence,
                   track_id=None, color=(0, 255, 0),
                   show_confidence=True, show_track_id=True):
    """
    Combines draw_bounding_box + draw_label into one call.
    Builds the label string based on what we want to show:
    - Class name always shown (car, rickshaw, etc.)
    - Confidence score optional (e.g. 0.87)
    - Track ID optional (e.g. #12) — only available in Phase 3+

    Example label output: "rickshaw 0.87 #12"
    """
    # draw the box
    draw_bounding_box(frame, x1, y1, x2, y2, color)

    # build label string piece by piece
    display_text = label
    if show_confidence:
        display_text += f" {confidence:.2f}"
    if show_track_id and track_id is not None:
        display_text += f" #{track_id}"

    draw_label(frame, display_text, x1, y1, color)


def draw_counting_line(frame, line_position_ratio, color=(0, 255, 0),
                       thickness=2):
    """
    Draws a horizontal virtual line across the frame.
    We use a ratio (0.0 to 1.0) instead of a fixed pixel value so
    the line position adapts automatically to any video resolution.
    0.5 = middle of frame, 0.7 = 70% down from top, etc.
    This line is what vehicles must cross to be counted.
    """
    height, width = frame.shape[:2]

    # convert ratio to actual pixel y-coordinate
    line_y = int(height * line_position_ratio)

    cv2.line(
        frame,
        (0, line_y),  # left edge of frame
        (width, line_y),  # right edge of frame
        color,
        thickness
    )

    return line_y  # return the y pixel value — counter logic needs it


def draw_count_display(frame, counts):
    """
    Draws a semi-transparent black panel in the top-left corner
    showing live vehicle counts per class.

    Semi-transparent overlay works by:
    1. Copy the frame region
    2. Draw a filled black rectangle on the copy
    3. Blend the copy back with the original using cv2.addWeighted
    This is called alpha blending — the 'alpha' controls transparency.

    'counts' is a dict like:
    {"car": {"IN": 5, "OUT": 3}, "rickshaw": {"IN": 2, "OUT": 1}}
    """
    if not counts:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 28
    padding = 10

    # calculate how tall the panel needs to be based on number of classes
    panel_height = padding * 2 + line_height * (len(counts) + 1)
    panel_width = 280

    # create overlay copy for transparency effect
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),  # black background
        cv2.FILLED
    )

    # blend overlay with original frame — 0.6 opacity
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # draw header text
    cv2.putText(
        frame, "Vehicle Count",
        (padding, padding + line_height - 8),
        font, font_scale,
        (255, 255, 0),  # yellow for header
        thickness, cv2.LINE_AA
    )

    # draw each class count on its own line
    for i, (class_name, direction_counts) in enumerate(counts.items()):
        in_count = direction_counts.get("IN", 0)
        out_count = direction_counts.get("OUT", 0)
        text = f"{class_name}: IN={in_count}  OUT={out_count}"

        cv2.putText(
            frame, text,
            (padding, padding + line_height * (i + 2) - 8),
            font, 0.5,
            (255, 255, 255),  # white for class rows
            1, cv2.LINE_AA
        )


def draw_fps(frame, fps):
    """
    Draws current processing FPS in the top-right corner.
    Useful during testing to see if our pipeline is fast enough.
    Real-time needs at least 15-20 FPS minimum.
    We calculate FPS ourselves in the main loop using time.time()
    and pass it here just for display.
    """
    height, width = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, _), _ = cv2.getTextSize(text, font, 0.6, 2)

    cv2.putText(
        frame, text,
        (width - text_w - 10, 30),
        font, 0.6,
        (0, 255, 255),  # cyan color — visible on most backgrounds
        2, cv2.LINE_AA
    )