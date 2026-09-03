import cv2
import numpy as np


class ClassicalDetector:
    """
    Detects moving vehicles using classical Digital Image Processing.
    No deep learning, no training data — purely pixel-level operations.

    This is our Phase 1 baseline. It will work reasonably well on
    static camera footage but will struggle with:
    - Moving camera (our Karachi video has this)
    - Shadows (MOG2 partially handles this)
    - Overlapping vehicles (contours merge into one blob)
    These limitations are intentional to show in the report why
    we then move to YOLOv8 in Phase 2.
    """

    def __init__(self, config):
        """
        Initialize the detector with settings from config.yaml.
        We read all parameters from config so nothing is hardcoded.
        """
        cfg = config["classical"]

        # Gaussian blur kernel size — must be odd number (3,5,7...)
        # Larger kernel = more blur = more noise removed but edges get softer
        self.blur_kernel = cfg["gaussian_blur_kernel"]

        # MOG2 = Mixture of Gaussians v2
        # It models each pixel's color over time as a mix of Gaussian distributions
        # 'history' = how many past frames it uses to build the background model
        # More history = more stable background model but slower to adapt
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg["mog2_history"],
            varThreshold=cfg["mog2_var_threshold"],
            detectShadows=cfg["mog2_detect_shadows"]
            # detectShadows=True marks shadows as gray (127) instead of white (255)
            # so we can remove them separately without losing vehicle pixels
        )

        # Morphological operation parameters
        self.dilate_iter = cfg["dilate_iterations"]
        self.erode_iter = cfg["erode_iterations"]

        # Contour area thresholds — filters out objects too small or too large
        # min: removes pedestrians, birds, noise blobs
        # max: removes entire frame artifacts (camera shake on our Karachi video)
        self.min_area = cfg["min_contour_area"]
        self.max_area = cfg["max_contour_area"]

        # Morphological kernels — these are the "stamps" used for erosion/dilation
        # 5x5 ellipse kernel is good for vehicle-shaped blobs (rounded rectangles)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

    def preprocess(self, frame):
        """
        Converts frame to grayscale and applies Gaussian blur.

        WHY GRAYSCALE:
        MOG2 works on single-channel images. Color (3 channels) adds
        complexity without much benefit for motion detection — we care
        about WHERE things move, not what color they are.

        WHY GAUSSIAN BLUR:
        Video from cameras always has pixel-level noise (tiny random
        color variations per frame). Without blur, MOG2 sees this noise
        as "motion" and generates thousands of tiny false positives.
        Blur averages neighboring pixels, smoothing out the noise.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel, self.blur_kernel),
            0   # sigmaX=0 means OpenCV calculates it automatically from kernel size
        )
        return blurred

    def apply_background_subtraction(self, preprocessed_frame):
        """
        Applies MOG2 background subtraction to get the foreground mask.

        OUTPUT: a binary-like mask where:
        - White (255) = moving foreground (vehicles we want)
        - Gray  (127) = shadows (we will remove these)
        - Black (0)   = static background (road, buildings)

        HOW MOG2 WORKS (simplified):
        For each pixel, MOG2 maintains a statistical model of what
        "normal" looks like based on past frames. If the current pixel
        value is too different from its normal range, it's flagged as
        foreground (moving object). This is smarter than simple frame
        differencing because it handles gradual lighting changes.
        """
        fg_mask = self.bg_subtractor.apply(preprocessed_frame)
        return fg_mask

    def remove_shadows(self, fg_mask):
        """
        Removes shadow pixels from the foreground mask.
        MOG2 marks shadows as gray (pixel value 127).
        We threshold to keep ONLY pure white (255) pixels — actual moving objects.
        Shadows would otherwise create ghost contours attached to vehicles,
        making bounding boxes much larger than the actual vehicle.
        """
        # keep only pixels with value > 200 (essentially just the 255 whites)
        _, clean_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        return clean_mask

    def apply_morphology(self, mask):
        """
        Cleans up the binary mask using morphological operations.

        WHY EROSION FIRST:
        Erosion shrinks all white regions. This removes tiny isolated
        noise blobs (1-3 pixels wide) that survived blurring and MOG2.
        Think of it as "eating away" the edges of white regions.

        WHY DILATION AFTER:
        Dilation expands white regions. After erosion removed noise,
        dilation restores the size of real vehicle blobs and also
        fills small gaps/holes inside vehicle regions (e.g. a vehicle
        with a transparent windshield appears as a donut shape —
        dilation fills it into a solid blob).

        EROSION then DILATION together = OPENING operation
        It removes noise while preserving the shape of larger objects.
        """
        # erosion — shrink blobs, kill tiny noise
        eroded = cv2.erode(
            mask, self.kernel,
            iterations=self.erode_iter
        )

        # dilation — expand surviving blobs, fill internal gaps
        dilated = cv2.dilate(
            eroded, self.kernel,
            iterations=self.dilate_iter
        )

        return dilated

    def find_vehicle_contours(self, clean_mask):
        """
        Finds contours (outlines) of white blobs in the mask and
        filters them by area to keep only vehicle-sized blobs.

        WHAT IS A CONTOUR:
        A contour is a list of (x,y) points that form the boundary
        of a connected white region. Think of it as tracing the outline
        of each white blob with a pen.

        cv2.RETR_EXTERNAL: only outer contours (ignores holes inside blobs)
        cv2.CHAIN_APPROX_SIMPLE: compresses contour points — only stores
        corners instead of every single point (saves memory)

        AREA FILTER:
        Small area  → noise, pedestrian, bird → skip
        Large area  → camera artifact, full frame blob → skip
        Mid range   → actual vehicle → keep
        """
        contours, _ = cv2.findContours(
            clean_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        vehicle_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # skip if too small or too large
            if area < self.min_area or area > self.max_area:
                continue

            # cv2.boundingRect returns (x, y, width, height) of the
            # smallest rectangle that fits around the contour
            x, y, w, h = cv2.boundingRect(contour)

            vehicle_boxes.append({
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "area": area,
                "label": "vehicle",       # classical cant classify type
                "confidence": 1.0,        # no confidence score in classical
                "track_id": None          # no tracking in Phase 1
            })

        return vehicle_boxes

    def detect(self, frame):
        """
        Main method — runs the full classical DIP pipeline on one frame.
        Call this once per frame in the main loop.

        Returns a list of detected vehicle bounding boxes.
        Each box is a dict with x1, y1, x2, y2, label, confidence.
        """
        preprocessed = self.preprocess(frame)
        fg_mask = self.apply_background_subtraction(preprocessed)
        clean_mask = self.remove_shadows(fg_mask)
        final_mask = self.apply_morphology(clean_mask)
        boxes = self.find_vehicle_contours(final_mask)

        return boxes, final_mask
        # we return final_mask too so we can display it side by side
        # with the original frame during debugging — very useful to
        # see what MOG2 is actually "seeing"