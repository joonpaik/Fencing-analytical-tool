__all__ = ['HomographyTracker', 'StripDetector', 'StripGeometry']

import cv2
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Strip geometry constants
# ---------------------------------------------------------------------------

STRIP_REAL = (14.0, 2.0)                       # metres: (length, width) of a fencing strip
PX_PER_M   = 50                                 # output canvas resolution (px / m)
STRIP_W    = int(STRIP_REAL[0] * PX_PER_M)     # 700 px
STRIP_H    = int(STRIP_REAL[1] * PX_PER_M)     # 100 px
_MARGIN    = 50                                  # canvas padding around the strip
CANVAS_W   = STRIP_W + 2 * _MARGIN              # 800 px
CANVAS_H   = STRIP_H + 2 * _MARGIN              # 200 px


# ---------------------------------------------------------------------------
# StripGeometry dataclass
# ---------------------------------------------------------------------------

@dataclass
class StripGeometry:
    """Homography + derived geometry for one frame."""
    H: np.ndarray           # 3×3 float64 pixel→canvas homography
    corners_px: np.ndarray  # (4, 2) float32 strip corners in frame coords
    confidence: float       # detection confidence [0, 1]


# ---------------------------------------------------------------------------
# Strip detection
# ---------------------------------------------------------------------------

class StripDetector:
    """
    Detects the fencing strip in a single frame using colour segmentation.

    The strip surface is bright and near-neutral (white/grey piste mat).
    Works in LAB colour space:
      L > 140               — bright enough
      |a − 128| < 20        — near neutral red-green axis (128 = neutral in OpenCV LAB)
      |b − 128| < 20        — near neutral blue-yellow axis
    Only the bottom 70% of the frame is searched (strip is always on the floor).
    A candidate region must cover at least 3% of the ROI area to be accepted.
    """

    MIN_CONF  = 0.4    # minimum contour-fill ratio accepted as a detection
    _L_THRESH = 140    # LAB L channel minimum
    _AB_DIFF  = 20     # maximum deviation from neutral on a and b channels

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the strip and return 4 corners (4, 2) float32 in full-frame pixel
        coordinates (TL, TR, BR, BL order), or None if no confident detection.
        """
        h, w = frame.shape[:2]

        # Bottom mask: only search below 70% of frame height
        bottom_y = int(h * 0.70)
        roi = frame[bottom_y:, :]

        # LAB colour segmentation
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L   = lab[:, :, 0].astype(np.int16)
        a   = lab[:, :, 1].astype(np.int16)
        b   = lab[:, :, 2].astype(np.int16)

        neutral_a, neutral_b = 128, 128
        mask = (
            (L > self._L_THRESH) &
            (np.abs(a - neutral_a) < self._AB_DIFF) &
            (np.abs(b - neutral_b) < self._AB_DIFF)
        ).astype(np.uint8) * 255

        # Morphological cleanup — close small gaps, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Minimum area = 3% of the ROI
        roi_h = h - bottom_y
        min_area = w * roi_h * 0.03

        best_corners = None
        best_area    = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(cnt)
            box  = cv2.boxPoints(rect)   # (4, 2) float32

            _, (rw, rh), _ = rect
            rect_area = rw * rh
            if rect_area < 1:
                continue

            conf = area / rect_area
            if conf < self.MIN_CONF:
                continue

            if area > best_area:
                best_area    = area
                best_corners = box.copy()

        if best_corners is None:
            return None

        # Translate from ROI coordinates back to full-frame coordinates
        best_corners[:, 1] += bottom_y
        return self._order_corners(best_corners)

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Order 4 corners as: top-left, top-right, bottom-right, bottom-left.
        Robust to any initial ordering from minAreaRect.
        """
        pts     = pts.reshape(4, 2).astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)       # top-left has smallest sum, bottom-right has largest
        d = np.diff(pts, axis=1)  # top-right has smallest diff, bottom-left has largest

        ordered[0] = pts[np.argmin(s)]   # top-left
        ordered[2] = pts[np.argmax(s)]   # bottom-right
        ordered[1] = pts[np.argmin(d)]   # top-right
        ordered[3] = pts[np.argmax(d)]   # bottom-left

        return ordered


# ---------------------------------------------------------------------------
# Homography tracking
# ---------------------------------------------------------------------------

class HomographyTracker:
    """
    Maintains a temporal homography from camera pixels to a canonical strip canvas.

    On each update():
      1. StripDetector locates strip corners in the frame.
      2. cv2.findHomography maps those corners to the canonical strip canvas.
      3. Between detections, Lucas-Kanade optical flow keeps corners alive.

    The homography expires after MAX_AGE consecutive frames without a new detection.
    """

    MAX_AGE = 45

    # Canonical destination corners on the output canvas (TL, TR, BR, BL).
    _MARGIN = _MARGIN   # same value as module-level constant
    _DST = np.array([
        [_MARGIN,              _MARGIN],
        [_MARGIN + STRIP_W,    _MARGIN],
        [_MARGIN + STRIP_W,    _MARGIN + STRIP_H],
        [_MARGIN,              _MARGIN + STRIP_H],
    ], dtype=np.float32)

    def __init__(self):
        self._detector:   StripDetector         = StripDetector()
        self._H:          Optional[np.ndarray]  = None
        self._corners_px: Optional[np.ndarray]  = None
        self._age:        int                   = 0
        self._prev_gray:  Optional[np.ndarray]  = None

    @property
    def H(self) -> Optional[np.ndarray]:
        """Current pixel→canvas homography (3×3 float64), or None."""
        return self._H

    def update(self, frame: np.ndarray) -> Optional[StripGeometry]:
        """
        Process one frame.
        Returns StripGeometry when a valid homography exists, otherwise None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Attempt fresh detection
        corners = self._detector.detect(frame)

        if corners is not None:
            self._corners_px = corners
            self._age        = 0
        elif self._corners_px is not None and self._prev_gray is not None:
            # Keep corners alive with optical flow
            refined = self._refine_optical_flow(gray, self._corners_px)
            if refined is not None:
                self._corners_px = refined
            self._age += 1
        else:
            self._age += 1

        self._prev_gray = gray

        # Expire homography when too old or no corners
        if self._age > self.MAX_AGE or self._corners_px is None:
            self._H = None
            return None

        H, mask = cv2.findHomography(self._corners_px, self._DST, cv2.RANSAC, 5.0)
        if H is None:
            self._H = None
            return None

        self._H    = H
        inliers    = int(mask.sum()) if mask is not None else 0
        confidence = inliers / max(len(self._corners_px), 1)

        return StripGeometry(H=H, corners_px=self._corners_px.copy(), confidence=confidence)

    def _refine_optical_flow(
        self, gray: np.ndarray, corners: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Refine corner positions between detections using Lucas-Kanade optical flow.
        Returns updated (4, 2) float32 corners, or None if tracking is unreliable.
        At least 3 of 4 corners must track successfully.
        """
        if self._prev_gray is None:
            return None

        pts = corners.reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if status is None or status.sum() < 3:
            return None

        out = corners.copy()
        for i, (ok, npt) in enumerate(zip(status.flatten(), next_pts.reshape(4, 2))):
            if ok:
                out[i] = npt
        return out

    def pixel_to_meters(
        self, pt: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Map a frame pixel coordinate (x, y) to strip-relative metres
        (along_strip, across_strip).  Returns None if no homography exists.
        """
        if self._H is None:
            return None
        p   = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, self._H)[0, 0]
        mx  = (float(dst[0]) - self._MARGIN) / PX_PER_M
        my  = (float(dst[1]) - self._MARGIN) / PX_PER_M
        return (mx, my)
