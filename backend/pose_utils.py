__all__ = [
    'YOLO_POSE_CONNECTIONS', 'YOLO_COCO_NAMES',
    '_angle', '_ANGLE_CONF_GATE', '_safe_angle',
    '_extract_angles_coco', '_draw_skeleton_coco',
    '_segments_intersect', '_legs_are_crossed',
    '_arms_are_crossed', '_torso_is_crossed',
]

import cv2
import math
import numpy as np
from typing import List, Tuple

__all__ = [
    'YOLO_POSE_CONNECTIONS', 'YOLO_COCO_NAMES',
    '_angle', '_ANGLE_CONF_GATE', '_safe_angle',
    '_extract_angles_coco', '_draw_skeleton_coco',
    '_segments_intersect', '_legs_are_crossed',
    '_arms_are_crossed', '_torso_is_crossed',
]

# COCO-17 keypoint indices (used by YOLO pose models)
YOLO_POSE_CONNECTIONS = [
    (5,  6),               # shoulders
    (5,  7), (7,  9),      # left arm
    (6,  8), (8, 10),      # right arm
    (5, 11), (6, 12),      # torso
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
]

YOLO_COCO_NAMES = [
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]


def _angle(p1, p2, p3) -> float:
    """Angle at p2 in the triangle p1-p2-p3 (degrees)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return 0.0
    return math.degrees(math.acos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))


_ANGLE_CONF_GATE = 0.2   # keypoints below this confidence yield 180.0 (neutral)


def _safe_angle(kp: np.ndarray, i1: int, i2: int, i3: int) -> float:
    """Return _angle(i1, i2, i3) or 180.0 if any keypoint is low-confidence."""
    if kp[i1, 2] < _ANGLE_CONF_GATE or kp[i2, 2] < _ANGLE_CONF_GATE or kp[i3, 2] < _ANGLE_CONF_GATE:
        return 180.0
    return _angle((kp[i1, 0], kp[i1, 1]),
                  (kp[i2, 0], kp[i2, 1]),
                  (kp[i3, 0], kp[i3, 1]))


def _extract_angles_coco(kp: np.ndarray) -> dict:
    """
    Derive joint angles from COCO-17 keypoints.
    Index mapping:  5=L-shoulder  6=R-shoulder
                    7=L-elbow     8=R-elbow
                    9=L-wrist    10=R-wrist
                   11=L-hip      12=R-hip
                   13=L-knee     14=R-knee
                   15=L-ankle    16=R-ankle
    """
    left_arm  = _safe_angle(kp, 5,  7,  9)
    right_arm = _safe_angle(kp, 6,  8,  10)
    left_leg  = _safe_angle(kp, 11, 13, 15)
    right_leg = _safe_angle(kp, 12, 14, 16)

    hip_cx = (kp[11][0] + kp[12][0]) / 2
    hip_cy = (kp[11][1] + kp[12][1]) / 2

    front = 'left' if kp[15][0] < kp[16][0] else 'right'
    front_knee = left_leg if front == 'left' else right_leg

    # Weapon arm: whichever wrist is further laterally from hip centre
    # (works for both left- and right-handed fencers)
    l_wrist_dist = abs(kp[9,  0] - hip_cx) if kp[9,  2] > _ANGLE_CONF_GATE else 0.0
    r_wrist_dist = abs(kp[10, 0] - hip_cx) if kp[10, 2] > _ANGLE_CONF_GATE else 0.0
    weapon_arm = left_arm if l_wrist_dist > r_wrist_dist else right_arm

    return {
        'left_arm_angle':   left_arm,
        'right_arm_angle':  right_arm,
        'left_leg_angle':   left_leg,
        'right_leg_angle':  right_leg,
        'hip_center_x':     hip_cx,
        'hip_center_y':     hip_cy,
        'front_knee_angle': front_knee,
        'weapon_arm_angle': weapon_arm,
    }


def _segments_intersect(p1: np.ndarray, p2: np.ndarray,
                         p3: np.ndarray, p4: np.ndarray) -> bool:
    """True if line segment p1→p2 and p3→p4 intersect (excluding endpoints)."""
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return False   # parallel / collinear
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def _legs_are_crossed(kp: np.ndarray, conf_thresh: float = 0.15) -> bool:
    """
    Return True if the left-leg line (left_hip→left_ankle) and right-leg line
    (right_hip→right_ankle) geometrically intersect — an anatomically impossible
    state that indicates a persistent label-swap error rather than a real crossing.

    Uses hip (11,12) and ankle (15,16) only; knees aren't needed for this check
    and are often the noisy joint during occlusion.
    """
    for i in (11, 12, 15, 16):
        if kp[i, 2] < conf_thresh:
            return False
    return _segments_intersect(kp[11, :2], kp[15, :2],
                                kp[12, :2], kp[16, :2])


def _arms_are_crossed(kp: np.ndarray, conf_thresh: float = 0.15) -> bool:
    """
    Return True if the left-arm line (left_shoulder→left_wrist) and right-arm
    line (right_shoulder→right_wrist) geometrically intersect.

    Uses shoulders (5,6) and wrists (9,10); elbows are skipped for the same
    reason knees are — they're noisier and not needed for the crossing check.
    """
    for i in (5, 6, 9, 10):
        if kp[i, 2] < conf_thresh:
            return False
    return _segments_intersect(kp[5, :2], kp[9, :2],
                                kp[6, :2], kp[10, :2])


def _torso_is_crossed(kp: np.ndarray, conf_thresh: float = 0.15) -> bool:
    """
    Return True if the left-torso line (left_shoulder→left_hip) and right-torso
    line (right_shoulder→right_hip) geometrically intersect — indicating that
    either the shoulder or hip labels (or both) have been persistently swapped.
    """
    for i in (5, 6, 11, 12):
        if kp[i, 2] < conf_thresh:
            return False
    return _segments_intersect(kp[5, :2], kp[11, :2],
                                kp[6, :2], kp[12, :2])


def _draw_skeleton_coco(frame, kp: np.ndarray,
                        dot_colour=(0, 0, 255),
                        line_colour=(0, 255, 0)):
    # Head: nose-only at higher confidence threshold (fencing masks confuse multi-point centroid)
    head_centre = None
    if kp[0, 2] > 0.25:
        head_centre = (int(kp[0, 0]), int(kp[0, 1]))
        cv2.circle(frame, head_centre, 5, dot_colour, -1)

    # Connect head centroid to each visible shoulder (indices 5, 6)
    if head_centre is not None:
        for shoulder_idx in (5, 6):
            if kp[shoulder_idx][2] > 0.05:
                cv2.line(frame, head_centre,
                         (int(kp[shoulder_idx][0]), int(kp[shoulder_idx][1])),
                         line_colour, 2)

    # Draw body keypoints (indices 5+)
    for x, y, conf in kp[5:]:
        if conf > 0.05:
            cv2.circle(frame, (int(x), int(y)), 5, dot_colour, -1)

    for i, j in YOLO_POSE_CONNECTIONS:
        if kp[i][2] > 0.05 and kp[j][2] > 0.05:
            cv2.line(frame,
                     (int(kp[i][0]), int(kp[i][1])),
                     (int(kp[j][0]), int(kp[j][1])),
                     line_colour, 2)
