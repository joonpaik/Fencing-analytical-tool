"""
YoloPoseEngine — body-pose marking for the Fencing Analysis tool.

Two operating modes (set via `mode` argument):
  'yolo_pose'      — single-pass: YOLO pose model outputs boxes + keypoints directly.
                     Uses models/yolo26n-pose.pt (fast) or yolo26s-pose.pt (accurate).
  'yolo_mediapipe' — two-pass (reference architecture): YOLO detects persons, then
                     MediaPipe PoseLandmarker runs on each cropped region.
                     Uses models/pose_landmarker_full.task (or lite).
"""

import cv2
import math
import time
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

from config import Config

# ---------------------------------------------------------------------------
# Skeleton connections
# ---------------------------------------------------------------------------

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

# MediaPipe 33-point connections (same mapping as reference code)
MP_POSE_CONNECTIONS = [
    (0, 11), (0, 12),        # head to shoulders
    (11, 13), (13, 15),      # left arm
    (12, 14), (14, 16),      # right arm
    (11, 23), (12, 24),      # torso
    (23, 25), (25, 27),      # left leg
    (24, 26), (26, 28),      # right leg
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PersonPose:
    """Holds detection box and keypoints for one person in a frame."""
    track_id: int
    box: Tuple[int, int, int, int]       # (x1, y1, x2, y2) in full-frame coords
    keypoints: List[Tuple[float, float, float]]  # (x, y, conf/visibility) per point
    left_arm_angle: float = 0.0
    right_arm_angle: float = 0.0
    left_leg_angle: float = 0.0
    right_leg_angle: float = 0.0
    hip_center_x: float = 0.0
    hip_center_y: float = 0.0
    front_knee_angle: float = 180.0
    weapon_arm_angle: float = 180.0

    @property
    def kp(self):
        """Shorthand: keypoints as numpy (N, 3) array."""
        return np.array(self.keypoints, dtype=np.float32)


# ---------------------------------------------------------------------------
# Swap detection + temporal smoother
# ---------------------------------------------------------------------------

# Independent symmetric pairs — checked and swapped individually.
_INDEPENDENT_PAIRS = [
    (5,  6),   # shoulders
    (11, 12),  # hips
]

# Linked limb chains — proximal and distal joints of each arm/leg must swap
# together to prevent half-swap artifacts (e.g. elbow flips but wrist doesn't,
# producing an arm that crosses the body).
# Each entry: (left_proximal, right_proximal, left_distal, right_distal).
_LIMB_CHAINS = [
    (7,  8,  9,  10),  # elbows (proximal) + wrists (distal)
    (13, 14, 15, 16),  # knees  (proximal) + ankles (distal)
]


def _fix_pair(kp: np.ndarray, kp_prev: np.ndarray,
              l: int, r: int, conf_thresh: float) -> bool:
    """
    Check one symmetric pair and swap in-place if that reduces distance cost.
    Returns True if both joints are visible (swap check was performed).
    """
    if (kp[l, 2] < conf_thresh or kp[r, 2] < conf_thresh or
            kp_prev[l, 2] < conf_thresh or kp_prev[r, 2] < conf_thresh):
        return False
    cost_keep = (np.linalg.norm(kp[l, :2] - kp_prev[l, :2]) +
                 np.linalg.norm(kp[r, :2] - kp_prev[r, :2]))
    cost_swap = (np.linalg.norm(kp[r, :2] - kp_prev[l, :2]) +
                 np.linalg.norm(kp[l, :2] - kp_prev[r, :2]))
    if cost_swap < cost_keep:
        kp[[l, r]] = kp[[r, l]]
    return True


def _detect_and_fix_swaps(kp_new: np.ndarray, kp_prev: np.ndarray,
                           conf_thresh: float = 0.15) -> np.ndarray:
    """
    Choose left/right assignments that minimise total Euclidean distance from
    the previous smoothed positions.

    Shoulders and hips are checked independently.  Arms (elbow+wrist) and legs
    (knee+ankle) are treated as linked chains so both segments always swap
    together — preventing half-swap artifacts where one end flips but the other
    doesn't.

    Partial-visibility fallback: if not all four joints of a chain are above
    conf_thresh, the proximal pair drives the swap decision and the result is
    applied to any visible distal joints.  This ensures that a low-confidence
    back-ankle or weapon-wrist never blocks the swap check for the whole limb.
    """
    kp = kp_new.copy()

    # --- independent pairs ---
    for l, r in _INDEPENDENT_PAIRS:
        _fix_pair(kp, kp_prev, l, r, conf_thresh)

    # --- linked limb chains with proximal-fallback ---
    for lp, rp, ld, rd in _LIMB_CHAINS:
        prox_vis = _pair_visible(kp, kp_prev, lp, rp, conf_thresh)
        dist_vis = _pair_visible(kp, kp_prev, ld, rd, conf_thresh)

        if prox_vis and dist_vis:
            # Full 4-joint chain comparison
            cost_keep = sum(np.linalg.norm(kp[i, :2] - kp_prev[i, :2])
                            for i in (lp, rp, ld, rd))
            cost_swap = (np.linalg.norm(kp[rp, :2] - kp_prev[lp, :2]) +
                         np.linalg.norm(kp[lp, :2] - kp_prev[rp, :2]) +
                         np.linalg.norm(kp[rd, :2] - kp_prev[ld, :2]) +
                         np.linalg.norm(kp[ld, :2] - kp_prev[rd, :2]))
            if cost_swap < cost_keep:
                kp[[lp, rp]] = kp[[rp, lp]]
                kp[[ld, rd]] = kp[[rd, ld]]

        elif prox_vis:
            # Proximal pair drives decision; apply to distal if visible
            cost_keep_p = (np.linalg.norm(kp[lp, :2] - kp_prev[lp, :2]) +
                           np.linalg.norm(kp[rp, :2] - kp_prev[rp, :2]))
            cost_swap_p = (np.linalg.norm(kp[rp, :2] - kp_prev[lp, :2]) +
                           np.linalg.norm(kp[lp, :2] - kp_prev[rp, :2]))
            if cost_swap_p < cost_keep_p:
                kp[[lp, rp]] = kp[[rp, lp]]
                if dist_vis:
                    kp[[ld, rd]] = kp[[rd, ld]]

        elif dist_vis:
            # Only distal visible — independent check on distal pair
            _fix_pair(kp, kp_prev, ld, rd, conf_thresh)

    return kp


def _pair_visible(kp: np.ndarray, kp_prev: np.ndarray,
                  l: int, r: int, conf_thresh: float) -> bool:
    """True when both joints of a pair are above conf_thresh in both frames."""
    return (kp[l, 2] >= conf_thresh and kp[r, 2] >= conf_thresh and
            kp_prev[l, 2] >= conf_thresh and kp_prev[r, 2] >= conf_thresh)


class PoseSmoother:
    """
    Per-track temporal smoother with swap correction and per-joint velocity limits.

    Pipeline per frame:
      1. Swap correction        — undo left/right label flips before anything else.
      2. Velocity clamp         — per-joint cap (raised for wrists/ankles to match
                                  real lunge speeds at 1080p).
      3. Velocity-adaptive EMA  — alpha scales up during fast motion so lunges are
                                  tracked responsively; slow guard posture stays smooth.
      4. Confidence gate        — below threshold, extrapolate from last known velocity
                                  (decaying each frame) instead of hard-freezing, so
                                  occluded weapon-arm detections don't snap on return.
    """

    # Max pixels a joint may move per frame — raised to match 1080p lunge speeds
    _MAX_VEL: dict = {
        # core — stable anchor
        5: 20,  6: 20,  11: 20, 12: 20,
        # mid-limb
        7: 45,  8: 45,  13: 100, 14: 100,
        # extremities — fast fencing action
        0: 80,  1: 80,  2: 80,  3: 80,  4: 80,
        9: 160, 10: 160,        # wrists: 80 → 160
        15: 130, 16: 130,       # ankles: 70 → 130
    }
    _DEFAULT_MAX_VEL = 45.0

    # Velocity-adaptive alpha range
    _ALPHA_SLOW  = 0.25   # nearly stationary (was 0.15 → ~4 frame lag vs ~7)
    _ALPHA_FAST  = 0.70   # full-speed lunge  (was 0.55 → tracks in ~1.4 frames vs ~2)
    _VEL_SLOW    = 5.0    # px/frame below which we use ALPHA_SLOW
    _VEL_FAST    = 50.0   # px/frame above which we use ALPHA_FAST (was 60 → engages sooner)

    # Velocity-decay prediction when confidence is low
    _VEL_DECAY   = 0.7    # multiply carried velocity by this each frame

    # Temporal crossing correction — legs are allowed to be in a geometrically
    # crossed state for up to this many consecutive frames (~267ms at 30fps)
    # before a force-swap is triggered.  Brief real crossings stay; stuck errors
    # get corrected.
    _CROSS_CORRECT_FRAMES = 8

    def __init__(self, conf_gate: float = 0.2):
        self.conf_gate  = conf_gate
        self._state: dict  = {}   # track_id -> smoothed (N, 3) np.ndarray
        self._veloc: dict  = {}   # track_id -> last velocity (N, 2) np.ndarray
        self._leg_cross_count: dict = {}    # track_id -> consecutive crossed-leg frames
        self._arm_cross_count: dict = {}    # track_id -> consecutive crossed-arm frames
        self._torso_cross_count: dict = {}  # track_id -> consecutive crossed-torso frames

    def smooth(self, track_id: int, kp: np.ndarray) -> np.ndarray:
        if track_id not in self._state:
            self._state[track_id] = kp.copy()
            self._veloc[track_id] = np.zeros((len(kp), 2), dtype=np.float32)
            return kp

        prev     = self._state[track_id]
        prev_vel = self._veloc[track_id]

        # 1. Fix left/right label swaps
        kp = _detect_and_fix_swaps(kp, prev, self.conf_gate)

        out     = prev.copy()
        new_vel = np.zeros_like(prev_vel)

        for idx in range(len(kp)):
            max_vel = self._MAX_VEL.get(idx, self._DEFAULT_MAX_VEL)

            if kp[idx, 2] > self.conf_gate:
                # 2. Velocity clamp on the new detection
                raw_delta = kp[idx, :2] - prev[idx, :2]
                dist      = np.linalg.norm(raw_delta)
                if dist > max_vel:
                    raw_delta = raw_delta * (max_vel / (dist + 1e-6))
                clamped = prev[idx, :2] + raw_delta

                # 3. Velocity-adaptive alpha
                alpha = np.interp(dist,
                                  [self._VEL_SLOW, self._VEL_FAST],
                                  [self._ALPHA_SLOW, self._ALPHA_FAST])
                out[idx, :2] = alpha * clamped + (1 - alpha) * prev[idx, :2]
                new_vel[idx] = out[idx, :2] - prev[idx, :2]
            else:
                # 4. Velocity-decay prediction — extrapolate rather than freeze
                predicted    = prev[idx, :2] + prev_vel[idx]
                out[idx, :2] = predicted
                new_vel[idx] = prev_vel[idx] * self._VEL_DECAY

            out[idx, 2] = kp[idx, 2]

        # 5. Temporal crossing correction
        #    For each limb pair, count consecutive frames where the smoothed
        #    chains are geometrically crossed (anchor→tip lines intersect).
        #    Brief real crossings stay; after _CROSS_CORRECT_FRAMES consecutive
        #    crossed frames the labels are force-swapped and the counter resets.

        leg_count = self._leg_cross_count.get(track_id, 0)
        if _legs_are_crossed(out, conf_thresh=0.15):
            leg_count += 1
            if leg_count >= self._CROSS_CORRECT_FRAMES:
                out[[13, 14]] = out[[14, 13]]
                out[[15, 16]] = out[[16, 15]]
                leg_count = 0
        else:
            leg_count = 0
        self._leg_cross_count[track_id] = leg_count

        arm_count = self._arm_cross_count.get(track_id, 0)
        if _arms_are_crossed(out, conf_thresh=0.15):
            arm_count += 1
            if arm_count >= self._CROSS_CORRECT_FRAMES:
                out[[7, 8]]  = out[[8, 7]]
                out[[9, 10]] = out[[10, 9]]
                arm_count = 0
        else:
            arm_count = 0
        self._arm_cross_count[track_id] = arm_count

        torso_count = self._torso_cross_count.get(track_id, 0)
        if _torso_is_crossed(out, conf_thresh=0.15):
            torso_count += 1
            if torso_count >= self._CROSS_CORRECT_FRAMES:
                out[[5, 6]]   = out[[6, 5]]
                out[[11, 12]] = out[[12, 11]]
                torso_count = 0
        else:
            torso_count = 0
        self._torso_cross_count[track_id] = torso_count

        self._state[track_id] = out
        self._veloc[track_id] = new_vel
        return out

    def extrapolate(self, track_id: int) -> Optional[np.ndarray]:
        """
        Advance the smoother one step using velocity-decay prediction only,
        without consuming a new YOLO detection.  Used when a detection is
        rejected (e.g. bounding-box jump).  Confidence values are kept from
        the last real detection so cross/swap logic stays valid next frame.
        """
        if track_id not in self._state:
            return None
        prev     = self._state[track_id]
        prev_vel = self._veloc[track_id]
        out      = prev.copy()
        new_vel  = np.zeros_like(prev_vel)
        for idx in range(len(prev)):
            out[idx, :2] = prev[idx, :2] + prev_vel[idx]
            new_vel[idx] = prev_vel[idx] * self._VEL_DECAY
        self._state[track_id] = out
        self._veloc[track_id] = new_vel
        return out

    def prune(self, active_ids: set):
        """Remove state for track IDs no longer present in the frame."""
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]
                self._veloc.pop(tid, None)
                self._leg_cross_count.pop(tid, None)
                self._arm_cross_count.pop(tid, None)
                self._torso_cross_count.pop(tid, None)


# ---------------------------------------------------------------------------
# Crossing-state helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

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


def _extract_angles_mp(landmarks, w: int, h: int) -> dict:
    """
    Derive joint angles from MediaPipe 33-point landmarks (NormalizedLandmark list).
    Uses the same joint groups as Config.ANGLE_CALCULATION_JOINTS.
    """
    def xy(i):
        lm = landmarks[i]
        return (lm.x * w, lm.y * h)

    # arm: shoulder(11/12) – elbow(13/14) – wrist(15/16)
    left_arm  = _angle(xy(11), xy(13), xy(15))
    right_arm = _angle(xy(12), xy(14), xy(16))
    # leg: hip(23/24) – knee(25/26) – ankle(27/28)
    left_leg  = _angle(xy(23), xy(25), xy(27))
    right_leg = _angle(xy(24), xy(26), xy(28))

    l_ankle_x = landmarks[27].x * w
    r_ankle_x = landmarks[28].x * w
    front = 'left' if l_ankle_x < r_ankle_x else 'right'
    front_knee = left_leg if front == 'left' else right_leg

    hip_cx = (landmarks[23].x + landmarks[24].x) * w / 2
    hip_cy = (landmarks[23].y + landmarks[24].y) * h / 2

    return {
        'left_arm_angle':   left_arm,
        'right_arm_angle':  right_arm,
        'left_leg_angle':   left_leg,
        'right_leg_angle':  right_leg,
        'hip_center_x':     hip_cx,
        'hip_center_y':     hip_cy,
        'front_knee_angle': front_knee,
        'weapon_arm_angle': right_arm,
    }


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

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




# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class YoloPoseEngine:
    """
    Encapsulates YOLO-based body pose marking.

    Parameters
    ----------
    mode : 'yolo_pose' | 'yolo_mediapipe'
    model_size : 'n' (nano/fast) | 's' (small/accurate) — only for yolo_pose
    mp_model : 'full' | 'lite' — only for yolo_mediapipe
    """

    MODELS_DIR = Config.MODELS_DIR

    def __init__(
        self,
        mode: str = 'yolo_mediapipe',
        model_size: str = 'n',
        mp_model: str = 'full',
    ):
        if mode not in ('yolo_pose', 'yolo_mediapipe'):
            raise ValueError(f"Unknown mode '{mode}'. Use 'yolo_pose' or 'yolo_mediapipe'.")
        self.mode = mode

        if mode == 'yolo_pose':
            pt_name = f"yolo26{model_size}-pose.pt"
            pt_path = self.MODELS_DIR / pt_name
            if not pt_path.exists():
                raise FileNotFoundError(f"YOLO pose model not found: {pt_path}")
            self._yolo = YOLO(str(pt_path))
            self._pose_detector = None
            self._smoother = PoseSmoother()
            print(f"[YoloPoseEngine] mode=yolo_pose  model={pt_name}")

        else:  # yolo_mediapipe
            # YOLO for person detection (reuse the nano pose model as a detector)
            pt_path = self.MODELS_DIR / f"yolo26n-pose.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {pt_path}")
            self._yolo = YOLO(str(pt_path))

            # MediaPipe PoseLandmarker
            task_name = f"pose_landmarker_{mp_model}.task"
            task_path = self.MODELS_DIR / task_name
            if not task_path.exists():
                raise FileNotFoundError(f"MediaPipe model not found: {task_path}")

            BaseOptions       = mp.tasks.BaseOptions
            PoseLandmarker    = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOpts = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode       = mp.tasks.vision.RunningMode

            opts = PoseLandmarkerOpts(
                base_options=BaseOptions(model_asset_path=str(task_path)),
                running_mode=RunningMode.IMAGE,
                num_poses=2,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self._pose_detector = PoseLandmarker.create_from_options(opts)
            self._smoother = PoseSmoother()
            print(f"[YoloPoseEngine] mode=yolo_mediapipe  det=yolo26n-pose.pt  pose={task_name}")

        # Canonical two-fencer state. IDs are fixed as 1 and 2.
        # We do not expose raw YOLO IDs because they can flip on occlusion.
        self._fencer_state = {
            1: {'raw_id': None, 'center': None, 'misses': 0},
            2: {'raw_id': None, 'center': None, 'misses': 0},
        }
        self._max_state_misses = 45

        # Box-jump guard — if a detection's box centre moves more than this many
        # pixels from the previous accepted centre, it likely grabbed a background
        # object.  The detection is rejected and the smoother extrapolates instead.
        self._BOX_JUMP_THRESH = 200  # px
        self._prev_boxes: dict = {}  # slot -> last accepted (x1,y1,x2,y2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Run pose detection on a single BGR frame.
        Returns a list of PersonPose objects (one per detected person).
        """
        if self.mode == 'yolo_pose':
            return self._process_yolo_pose(frame)
        return self._process_yolo_mediapipe(frame)

    def annotate_frame(self, frame: np.ndarray, poses: List[PersonPose]) -> np.ndarray:
        """Draw bounding boxes, skeletons, and angle labels on `frame` in-place."""
        for pose in poses:
            x1, y1, x2, y2 = pose.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(
                frame,
                f"Fencer {pose.track_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            kp = pose.kp
            if self.mode == 'yolo_pose':
                _draw_skeleton_coco(frame, kp)
            else:
                _draw_skeleton_mp_from_kp(frame, kp)
        return frame

    def process_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        skeleton_only: bool = False,
    ) -> List[dict]:
        """
        Full pipeline: read → detect → annotate → write.
        Returns per-frame analysis list compatible with main.py's format.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            if skeleton_only:
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_{self.mode}_skeleton.mp4"
                Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_{self.mode}.mp4"
                Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

        print(f"[YoloPoseEngine] {video_path.name}  {width}x{height} @ {fps:.1f}fps  ({total} frames)")

        frame_data  = []
        frame_id    = 0
        t_detect    = 0.0
        t_annotate  = 0.0
        t_write     = 0.0
        t_total_start = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            poses = self.process_frame(frame)
            t1 = time.perf_counter()
            canvas = np.zeros_like(frame) if skeleton_only else frame
            self.annotate_frame(canvas, poses)
            t2 = time.perf_counter()
            writer.write(canvas)
            t3 = time.perf_counter()

            t_detect   += t1 - t0
            t_annotate += t2 - t1
            t_write    += t3 - t2

            # Store analysis data (first detected person)
            p = poses[0] if poses else None
            frame_data.append({
                'frame':            frame_id,
                'persons_detected': len(poses),
                'left_arm_angle':   p.left_arm_angle  if p else None,
                'right_arm_angle':  p.right_arm_angle if p else None,
                'left_leg_angle':   p.left_leg_angle  if p else None,
                'right_leg_angle':  p.right_leg_angle if p else None,
                'hip_center_x':     p.hip_center_x    if p else None,
                'detect_ms':        round((t1 - t0) * 1000, 1),
            })

            frame_id += 1
            if frame_id % 60 == 0:
                elapsed = time.perf_counter() - t_total_start
                avg_fps = frame_id / elapsed
                print(f"  {frame_id}/{total} frames  |  {avg_fps:.1f} fps  |  "
                      f"detect {t_detect/frame_id*1000:.1f}ms  "
                      f"annotate {t_annotate/frame_id*1000:.1f}ms  "
                      f"write {t_write/frame_id*1000:.1f}ms  (avg/frame)")

        wall = time.perf_counter() - t_total_start
        avg_fps = frame_id / wall if wall > 0 else 0

        cap.release()
        writer.release()

        print(f"\n[YoloPoseEngine] done — {frame_id} frames → {output_path}")
        print(f"  Wall time   : {wall:.2f}s")
        print(f"  Avg FPS     : {avg_fps:.1f}")
        print(f"  Detection   : {t_detect:.2f}s  ({t_detect/wall*100:.1f}%)")
        print(f"  Annotation  : {t_annotate:.2f}s  ({t_annotate/wall*100:.1f}%)")
        print(f"  Video write : {t_write:.2f}s  ({t_write/wall*100:.1f}%)")
        return frame_data

    def close(self):
        """Release MediaPipe resources."""
        if self._pose_detector is not None:
            self._pose_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _box_area(box: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def _box_from_keypoints(
        kp: np.ndarray,
        ref_box: Tuple[int, int, int, int],
        conf_thresh: float = 0.1,
    ) -> Tuple[int, int, int, int]:
        """
        Build an updated reference box whose centre tracks the extrapolated
        keypoint centroid while preserving the width/height of ref_box.
        Used to advance _prev_boxes during background-grab frames so that a
        valid YOLO recovery is not falsely rejected as another jump.
        """
        visible = kp[kp[:, 2] > conf_thresh, :2]
        if len(visible) == 0:
            return ref_box
        cx = float(visible[:, 0].mean())
        cy = float(visible[:, 1].mean())
        x1, y1, x2, y2 = ref_box
        hw = (x2 - x1) / 2.0
        hh = (y2 - y1) / 2.0
        return (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))

    def _select_two_fencers(self, detections: List[dict]) -> List[dict]:
        """Pick at most two best person candidates (largest boxes)."""
        if len(detections) <= 2:
            return detections
        return sorted(detections, key=lambda d: self._box_area(d['box']), reverse=True)[:2]

    def _assign_canonical_ids(self, detections: List[dict]) -> List[dict]:
        """
        Assign stable canonical IDs {1, 2} to detections.

        Matching order:
          1) Raw YOLO ID continuity (if a known raw id reappears)
          2) Center proximity to each canonical slot's previous center
          3) Left-to-right fallback for any cold-start ambiguity
        """
        if not detections:
            for slot in (1, 2):
                st = self._fencer_state[slot]
                st['misses'] += 1
                st['raw_id'] = None
                if st['misses'] > self._max_state_misses:
                    st['center'] = None
            return []

        used_slots = set()
        unassigned = []

        # Pass 1: direct raw-id continuity.
        raw_to_slot = {
            st['raw_id']: slot
            for slot, st in self._fencer_state.items()
            if st['raw_id'] is not None
        }
        for d in detections:
            slot = raw_to_slot.get(d['raw_id'])
            if slot is not None and slot not in used_slots:
                d['track_id'] = slot
                used_slots.add(slot)
            else:
                unassigned.append(d)

        # Pass 2: center-distance assignment for remaining detections.
        open_slots = [s for s in (1, 2) if s not in used_slots]
        while unassigned and open_slots:
            best = None
            for d in unassigned:
                dc = self._box_center(d['box'])
                for slot in open_slots:
                    prev_c = self._fencer_state[slot]['center']
                    if prev_c is None:
                        continue
                    dist = math.hypot(dc[0] - prev_c[0], dc[1] - prev_c[1])
                    if best is None or dist < best[0]:
                        best = (dist, d, slot)
            if best is None:
                break
            _, d_best, slot_best = best
            d_best['track_id'] = slot_best
            used_slots.add(slot_best)
            open_slots.remove(slot_best)
            unassigned.remove(d_best)

        # Pass 3: deterministic left-to-right fallback.
        if unassigned:
            open_slots = [s for s in (1, 2) if s not in used_slots]
            unassigned_sorted = sorted(unassigned, key=lambda d: self._box_center(d['box'])[0])
            open_slots_sorted = sorted(open_slots)
            for d, slot in zip(unassigned_sorted, open_slots_sorted):
                d['track_id'] = slot
                used_slots.add(slot)

        # Update canonical slot states.
        slots_touched = set()
        for d in detections:
            slot = d.get('track_id')
            if slot is None:
                continue
            slots_touched.add(slot)
            st = self._fencer_state[slot]
            st['raw_id'] = d['raw_id']
            st['center'] = self._box_center(d['box'])
            st['misses'] = 0

        for slot in (1, 2):
            if slot not in slots_touched:
                st = self._fencer_state[slot]
                st['misses'] += 1
                st['raw_id'] = None
                if st['misses'] > self._max_state_misses:
                    st['center'] = None

        return [d for d in detections if d.get('track_id') in (1, 2)]

    def _process_yolo_pose(self, frame: np.ndarray) -> List[PersonPose]:
        """One-pass: YOLO pose model → boxes + COCO-17 keypoints. Uses persistent tracking."""
        results = self._yolo.track(frame, persist=True, conf=0.25, verbose=False,device='mps')
        poses = []
        raw_dets = []

        for r in results:
            if r.keypoints is None:
                continue
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                raw_id = int(r.boxes.id[i]) if r.boxes.id is not None else i
                raw_dets.append({
                    'box': (x1, y1, x2, y2),
                    'raw_id': raw_id,
                    'kp': r.keypoints.data[i].cpu().numpy(),
                })

        selected = self._select_two_fencers(raw_dets)
        assigned = self._assign_canonical_ids(selected)
        active_ids = set()

        for d in assigned:
            track_id = d['track_id']
            active_ids.add(track_id)

            box = d['box']
            prev_box = self._prev_boxes.get(track_id)
            if prev_box is not None:
                curr_c = self._box_center(box)
                prev_c = self._box_center(prev_box)
                if math.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1]) > self._BOX_JUMP_THRESH:
                    # Detection grabbed background — discard it entirely.
                    # The bad keypoints never touch the fencer's smoother state.
                    kp_data = self._smoother.extrapolate(track_id)
                    if kp_data is None:
                        continue
                    box = prev_box
                    # Advance the reference box so the next valid YOLO detection
                    # isn't falsely rejected because the fencer kept moving while
                    # _prev_boxes was frozen.
                    self._prev_boxes[track_id] = self._box_from_keypoints(kp_data, prev_box)
                else:
                    self._prev_boxes[track_id] = box
                    kp_data = self._smoother.smooth(track_id, d['kp'])
            else:
                self._prev_boxes[track_id] = box
                kp_data = self._smoother.smooth(track_id, d['kp'])

            angles = _extract_angles_coco(kp_data)
            pose = PersonPose(
                track_id=track_id,
                box=box,
                keypoints=kp_data.tolist(),
                **angles,
            )
            poses.append(pose)

        # Clean up prev_boxes for tracks that are no longer active.
        for slot in list(self._prev_boxes):
            if slot not in active_ids:
                del self._prev_boxes[slot]

        self._smoother.prune(active_ids)
        return sorted(poses, key=lambda p: p.track_id)

    def _process_yolo_mediapipe(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Two-pass (reference architecture):
          1. YOLO detects person bounding boxes.
          2. MediaPipe PoseLandmarker runs on each cropped region.
        """
        results = self._yolo.track(frame, persist=True, conf=0.25, verbose=False)
        poses = []
        raw_dets = []

        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                raw_id = int(r.boxes.id[i]) if r.boxes.id is not None else i
                raw_dets.append({'box': (x1, y1, x2, y2), 'raw_id': raw_id})

        selected = self._select_two_fencers(raw_dets)
        assigned = self._assign_canonical_ids(selected)
        active_ids = set()

        for d in assigned:
            track_id = d['track_id']
            box = d['box']
            x1, y1, x2, y2 = box

            prev_box = self._prev_boxes.get(track_id)
            if prev_box is not None:
                curr_c = self._box_center(box)
                prev_c = self._box_center(prev_box)
                if math.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1]) > self._BOX_JUMP_THRESH:
                    # Detection grabbed background — discard entirely; skip MediaPipe crop.
                    kp_arr = self._smoother.extrapolate(track_id)
                    if kp_arr is None:
                        continue
                    active_ids.add(track_id)
                    # Advance the reference box with the extrapolated position.
                    self._prev_boxes[track_id] = self._box_from_keypoints(kp_arr, prev_box)
                    angles = _extract_angles_coco(kp_arr)
                    poses.append(PersonPose(
                        track_id=track_id,
                        box=prev_box,
                        keypoints=kp_arr.tolist(),
                        **angles,
                    ))
                    continue
            self._prev_boxes[track_id] = box

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            det = self._pose_detector.detect(mp_image)

            if not det.pose_landmarks:
                continue

            active_ids.add(track_id)
            landmarks = det.pose_landmarks[0]   # first person in crop

            # Convert crop-relative landmarks to full-frame pixel coords
            kp_arr = np.array(
                [(lm.x * cw + x1, lm.y * ch + y1, lm.visibility) for lm in landmarks],
                dtype=np.float32,
            )
            kp_arr = self._smoother.smooth(track_id, kp_arr)

            angles = _extract_angles_mp(landmarks, cw, ch)
            # Translate hip_center_x to full-frame coords
            angles['hip_center_x'] += x1

            pose = PersonPose(
                track_id=track_id,
                box=box,
                keypoints=kp_arr.tolist(),
                **angles,
            )
            poses.append(pose)

        # Clean up prev_boxes for tracks that are no longer active.
        for slot in list(self._prev_boxes):
            if slot not in active_ids:
                del self._prev_boxes[slot]

        self._smoother.prune(active_ids)
        return sorted(poses, key=lambda p: p.track_id)


# ---------------------------------------------------------------------------
# Extra helper for MP keypoints stored as full-frame pixel tuples
# ---------------------------------------------------------------------------

def _draw_skeleton_mp_from_kp(frame, kp: np.ndarray):
    """Draw MediaPipe skeleton using full-frame keypoint coords."""
    for x, y, vis in kp:
        if vis > 0.05:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    for i, j in MP_POSE_CONNECTIONS:
        if i < len(kp) and j < len(kp) and kp[i][2] > 0.05 and kp[j][2] > 0.05:
            cv2.line(frame,
                     (int(kp[i][0]), int(kp[i][1])),
                     (int(kp[j][0]), int(kp[j][1])),
                     (0, 255, 0), 2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video = Config.PROJECT_ROOT / "sample" / "fencing.mp4"

    # --- Mode 1: YOLO pose model (one-pass, fastest) ---
    print("\n=== Mode: yolo_pose ===")
    with YoloPoseEngine(mode='yolo_pose', model_size='s') as engine:
        engine.process_video(video, skeleton_only=True)

    # # --- Mode 2: YOLO detection + MediaPipe pose (reference architecture) ---
    # print("\n=== Mode: yolo_mediapipe ===")
    # with YoloPoseEngine(mode='yolo_mediapipe', mp_model='full') as engine:
    #     engine.process_video(video)
