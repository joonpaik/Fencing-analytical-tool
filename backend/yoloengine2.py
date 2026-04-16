"""
YoloPoseEngine — body-pose marking for the Fencing Analysis tool.

Single operating mode: YOLO pose model outputs boxes + COCO-17 keypoints directly.
Uses models/yolo26n-pose.pt (fast) or yolo26s-pose.pt (accurate).
"""

import cv2
import math
import time
import numpy as np
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


_SWAP_INERTIA = 0.8   # swap only when cost_swap < cost_keep * this (20% hysteresis)


def _fix_pair(kp: np.ndarray, kp_prev: np.ndarray,
              l: int, r: int, conf_thresh: float) -> bool:
    """
    Check one symmetric pair and swap in-place if the swapped assignment is
    at least 20% cheaper (temporal inertia prevents oscillation on near-ties).
    Returns True if both joints are visible (swap check was performed).
    """
    if (kp[l, 2] < conf_thresh or kp[r, 2] < conf_thresh or
            kp_prev[l, 2] < conf_thresh or kp_prev[r, 2] < conf_thresh):
        return False
    cost_keep = (np.linalg.norm(kp[l, :2] - kp_prev[l, :2]) +
                 np.linalg.norm(kp[r, :2] - kp_prev[r, :2]))
    cost_swap = (np.linalg.norm(kp[r, :2] - kp_prev[l, :2]) +
                 np.linalg.norm(kp[l, :2] - kp_prev[r, :2]))
    if cost_swap < cost_keep * _SWAP_INERTIA:
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
            if cost_swap < cost_keep * _SWAP_INERTIA:
                kp[[lp, rp]] = kp[[rp, lp]]
                kp[[ld, rd]] = kp[[rd, ld]]

        elif prox_vis:
            # Proximal pair drives decision; apply to distal if visible
            cost_keep_p = (np.linalg.norm(kp[lp, :2] - kp_prev[lp, :2]) +
                           np.linalg.norm(kp[rp, :2] - kp_prev[rp, :2]))
            cost_swap_p = (np.linalg.norm(kp[rp, :2] - kp_prev[lp, :2]) +
                           np.linalg.norm(kp[lp, :2] - kp_prev[rp, :2]))
            if cost_swap_p < cost_keep_p * _SWAP_INERTIA:
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


class SwapCorrector:
    """
    Layer 4a — Left/right label swap correction.

    Stateless: all context comes from the caller (kp and kp_prev).
    Applies temporal inertia via _SWAP_INERTIA so near-tie assignments
    don't oscillate frame to frame.
    """

    _CONF_THRESH = 0.15

    def apply(self, kp: np.ndarray, kp_prev: np.ndarray) -> np.ndarray:
        """
        Returns a copy of kp with left/right labels corrected against kp_prev.
        Delegates to the module-level _detect_and_fix_swaps which already
        applies _SWAP_INERTIA at every comparison point.
        """
        return _detect_and_fix_swaps(kp, kp_prev, self._CONF_THRESH)


class CrossingCorrector:
    """
    Layer 4b — Temporal crossing correction + canonical post-correction state.

    Detects persistent geometric crossings (anatomical line segments that
    shouldn't intersect) and force-swaps after _CROSS_CORRECT_FRAMES
    consecutive crossed frames.  Brief real crossings are tolerated; stuck
    label errors are corrected.

    Also owns the canonical post-correction keypoint dict (_state) so that
    _estimate() can snapshot kp_prev from a single authoritative source.
    """

    _CROSS_CORRECT_FRAMES = 8

    def __init__(self):
        self._state: dict             = {}  # track_id -> post-correction (N, 3) kp
        self._leg_cross_count: dict   = {}  # track_id -> consecutive crossed-leg frames
        self._arm_cross_count: dict   = {}  # track_id -> consecutive crossed-arm frames
        self._torso_cross_count: dict = {}  # track_id -> consecutive crossed-torso frames

    def get_prev(self, track_id: int) -> Optional[np.ndarray]:
        """Post-correction (N, 3) kp from the previous frame, or None."""
        s = self._state.get(track_id)
        return s.copy() if s is not None else None

    def apply(self, track_id: int, kp: np.ndarray) -> np.ndarray:
        """
        Apply crossing correction, store corrected result as canonical state,
        and return the corrected (N, 3) kp.
        """
        out = kp.copy()

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

        # Canonical state — next frame's swap reference is always post-correction.
        self._state[track_id] = out
        return out

    def prune(self, active_ids: set):
        """Remove state for track IDs no longer present in the frame."""
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]
                self._leg_cross_count.pop(tid, None)
                self._arm_cross_count.pop(tid, None)
                self._torso_cross_count.pop(tid, None)


# ---------------------------------------------------------------------------
# Kalman filter — Layer 3 state estimation
# ---------------------------------------------------------------------------

class KalmanPoseFilter:
    """
    Per-joint constant-velocity Kalman filter for COCO-17 pose keypoints.

    State per joint: [x, y, vx, vy]
    Measurement:     [x, y]          (position observed directly from YOLO)
    Gain:            K = 0.2 + 0.6 * conf  (confidence-scaled; no covariance matrices)

    Call exactly one method per track per frame:
      update(track_id, kp)  — accepted detection: predict step + Kalman correction
      predict(track_id)     — rejected detection (box-jump): predict step only
    """

    def __init__(self):
        # track_id -> (N, 4) float32  [x, y, vx, vy]
        self._state: dict = {}

    # ------------------------------------------------------------------

    def predict(self, track_id: int) -> Optional[np.ndarray]:
        """
        Constant-velocity prediction (advance state by dt=1 frame).
        Mutates internal state.  Called when a detection is rejected so the
        skeleton keeps moving rather than freezing.

        Returns (N, 3) [x_pred, y_pred, conf=0], or None if no prior state.
        Caller should fill column 2 with confidence from last corrected state.
        """
        if track_id not in self._state:
            return None
        s = self._state[track_id]
        s[:, 0] += s[:, 2]   # x += vx
        s[:, 1] += s[:, 3]   # y += vy
        out = np.zeros((len(s), 3), dtype=np.float32)
        out[:, :2] = s[:, :2]
        return out

    def update(self, track_id: int, kp: np.ndarray) -> np.ndarray:
        """
        Predict + Kalman measurement update.

        kp   : (N, 3) float32  [x, y, conf]  — raw YOLO keypoints
        K    = clip(0.2 + 0.6 * conf, 0.2, 0.8) per joint  (normal path)
        Re-anchor override: joints with conf > 0.8 use
          x_new = 0.8 * z + 0.2 * x_pred  (K=0.8 hard snap to measurement)
          so a single strong detection immediately corrects long-running drift.

        Returns (N, 3) [x_filt, y_filt, conf].
        """
        if track_id not in self._state:
            # Bootstrap: initialize from first detection, zero velocity.
            s = np.zeros((len(kp), 4), dtype=np.float32)
            s[:, :2] = kp[:, :2]
            self._state[track_id] = s
            return kp.copy()

        s = self._state[track_id]
        x_prev = s[:, 0].copy()
        y_prev = s[:, 1].copy()

        # --- prediction step ---
        x_pred = x_prev + s[:, 2]   # x + vx
        y_pred = y_prev + s[:, 3]   # y + vy

        # --- measurement update (confidence-scaled Kalman gain) ---
        confs = np.clip(kp[:, 2], 0.0, 1.0).astype(np.float32)
        K     = np.clip(0.2 + 0.6 * confs, 0.2, 0.8)

        x_new = x_pred + K * (kp[:, 0] - x_pred)
        y_new = y_pred + K * (kp[:, 1] - y_pred)

        # --- strong re-anchor (5.3) ---
        # High-confidence joints snap harder to the measurement to prevent
        # accumulated Kalman drift from persisting through strong detections.
        strong = confs > 0.8
        if strong.any():
            x_new = np.where(strong, 0.8 * kp[:, 0] + 0.2 * x_pred, x_new)
            y_new = np.where(strong, 0.8 * kp[:, 1] + 0.2 * y_pred, y_new)

        # velocity = total corrected displacement over one frame
        s[:, 0] = x_new
        s[:, 1] = y_new
        s[:, 2] = x_new - x_prev
        s[:, 3] = y_new - y_prev

        out = np.empty((len(kp), 3), dtype=np.float32)
        out[:, 0] = x_new
        out[:, 1] = y_new
        out[:, 2] = kp[:, 2]
        return out

    def reset_velocity(self, track_id: int):
        """
        Zero vx/vy for a single track.  Called periodically (every ~300 frames)
        to prevent small systematic errors from accumulating into visible drift
        over long sequences.
        """
        if track_id in self._state:
            self._state[track_id][:, 2] = 0.0
            self._state[track_id][:, 3] = 0.0

    def get_state(self, track_id: int) -> Optional[np.ndarray]:
        """Current internal state (N, 4) [x, y, vx, vy], or None."""
        return self._state.get(track_id)

    def predict_preview(self, track_id: int) -> Optional[np.ndarray]:
        """
        Non-mutating constant-velocity prediction — for debug visualization only.
        Returns (N, 3) [x_pred, y_pred, conf=0] without advancing internal state,
        or None if no prior state exists for this track.
        """
        if track_id not in self._state:
            return None
        s = self._state[track_id]
        out = np.zeros((len(s), 3), dtype=np.float32)
        out[:, 0] = s[:, 0] + s[:, 2]  # x + vx
        out[:, 1] = s[:, 1] + s[:, 3]  # y + vy
        return out

    def prune(self, active_ids: set):
        """Remove state for tracks no longer active."""
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]


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
    model_size : 'n' (nano/fast) | 's' (small/accurate)
    """

    MODELS_DIR = Config.MODELS_DIR

    def __init__(
        self,
        model_size: str = 'n',
        debug: bool = False,
    ):
        pt_name = f"yolo26{model_size}-pose.pt"
        pt_path = self.MODELS_DIR / pt_name
        if not pt_path.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {pt_path}")
        self._yolo               = YOLO(str(pt_path))
        self._kalman             = KalmanPoseFilter()   # Layer 3: state estimation
        self._swap_corrector     = SwapCorrector()      # Layer 4a: swap correction
        self._crossing_corrector = CrossingCorrector()  # Layer 4b: crossing correction + state
        print(f"[YoloPoseEngine] mode=yolo_pose  model={pt_name}")

        # Canonical two-fencer state. IDs are fixed as 1 and 2.
        # Each slot tracks center, velocity (px/frame), color histogram, and
        # miss counter for occlusion tolerance.
        self._fencer_state = {
            1: {'raw_id': None, 'center': None, 'velocity': (0.0, 0.0), 'hist': None, 'misses': 0},
            2: {'raw_id': None, 'center': None, 'velocity': (0.0, 0.0), 'hist': None, 'misses': 0},
        }
        self._max_state_misses = 45

        # Pass-2 matching cost weights
        self._VEL_MATCH_LAMBDA = 0.5   # px — weight on velocity-difference penalty
        self._APP_MATCH_LAMBDA = 30.0  # scales Bhattacharyya [0,1] → px-equivalent
        self._use_appearance   = True  # toggle appearance matching

        # Role lock — left=1, right=2.  Only swap after this many consecutive
        # frames of inverted relative position to avoid crossing artefacts.
        self._role_mismatch_count: int = 0
        self._ROLE_LOCK_FRAMES: int    = 15   # ~0.5 s at 30 fps

        # Box-jump guard — adaptive threshold: 100 + 2 * slot_velocity_magnitude.
        # Faster-moving fencers get a proportionally larger allowed displacement
        # before a detection is treated as a background grab.
        self._BOX_JUMP_BASE  = 100.0   # px — minimum threshold (at rest)
        self._BOX_JUMP_SCALE =   2.0   # px per px/frame of velocity
        self._prev_boxes: dict = {}    # slot -> last accepted (x1,y1,x2,y2)
        self._reject_counts: dict = {} # slot -> consecutive box-jump rejections

        # Frame counter — used for periodic Kalman velocity reset (5.1)
        self._frame_id: int = 0
        self._VEL_RESET_INTERVAL: int = 300   # frames (~10 s at 30 fps)

        # Debug mode — visual overlays + per-event logging
        self._debug: bool = debug
        # Per-frame intermediates keyed by track_id:
        #   kp_raw        — raw YOLO keypoints before any filtering
        #   kp_predicted  — Kalman constant-velocity prediction (before measurement update)
        #   kp_filtered   — Kalman output after update
        #   velocities    — (N, 2) [vx, vy] from Kalman state after update
        self._debug_data: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Run pose detection on a single BGR frame.
        Returns a list of PersonPose objects (one per detected person).
        """
        return self._process_yolo_pose(frame)

    def annotate_frame(self, frame: np.ndarray, poses: List[PersonPose]) -> np.ndarray:
        """Draw bounding boxes, skeletons, and angle labels on `frame` in-place.

        Debug overlays (only when engine was constructed with debug=True):
          Red dots   — raw YOLO keypoints before Kalman filtering
          Blue dots  — Kalman constant-velocity prediction (where the joint was
                       expected to be this frame before the measurement arrived)
          Green      — Kalman-filtered output (normal skeleton, always drawn)
          Cyan arrows — velocity vectors scaled ×5 from each filtered joint
        """
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
            _draw_skeleton_coco(frame, kp)

            if self._debug:
                dbg = self._debug_data.get(pose.track_id, {})

                # Raw keypoints — red small dots
                kp_raw = dbg.get('kp_raw')
                if kp_raw is not None:
                    for x, y, c in kp_raw[5:]:
                        if c > 0.05:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 200), -1)

                # Predicted keypoints — blue small dots
                kp_pred = dbg.get('kp_predicted')
                if kp_pred is not None:
                    for x, y, c in kp_pred[5:]:
                        if c > 0.05:
                            cv2.circle(frame, (int(x), int(y)), 3, (200, 0, 0), -1)

                # Velocity vectors — cyan arrows (scale ×5 so 1 px/frame → 5 px arrow)
                kp_filt = dbg.get('kp_filtered')
                vels    = dbg.get('velocities')
                if kp_filt is not None and vels is not None:
                    _VEL_SCALE = 5.0
                    for idx in range(5, len(kp_filt)):
                        x, y, c = kp_filt[idx]
                        if c > 0.05:
                            vx, vy = float(vels[idx, 0]), float(vels[idx, 1])
                            tip = (int(x + vx * _VEL_SCALE), int(y + vy * _VEL_SCALE))
                            cv2.arrowedLine(frame, (int(x), int(y)), tip,
                                            (0, 255, 255), 1, tipLength=0.4)
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
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_yolo_pose_skeleton_v2.mp4"
            else:
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_yolo_pose_v2.mp4"
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
        pass

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

    @staticmethod
    def _compute_hist(
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        HSV color histogram (16×16 H×S bins) for the detection crop.
        Returns a (256, 1) float32 array suitable for cv2.compareHist, or
        None if the crop is too small.
        """
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

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

        # Pass 2: kinematic + appearance assignment for remaining detections.
        # Cost = position_distance
        #      + λ_vel * |observed_delta - slot_velocity|   (velocity consistency)
        #      + λ_app * Bhattacharyya(det_hist, slot_hist) (appearance similarity)
        # Gate is applied against the velocity-predicted position so fast-moving
        # fencers are not falsely rejected.
        _ID_REUSE_GATE = 150  # px from predicted position
        open_slots = [s for s in (1, 2) if s not in used_slots]
        while unassigned and open_slots:
            best = None
            for d in unassigned:
                dc = self._box_center(d['box'])
                for slot in open_slots:
                    st      = self._fencer_state[slot]
                    prev_c  = st['center']
                    if prev_c is None:
                        continue
                    vel = st['velocity']
                    pred_c = (prev_c[0] + vel[0], prev_c[1] + vel[1])

                    # Gate on predicted position
                    if math.hypot(dc[0] - pred_c[0], dc[1] - pred_c[1]) > _ID_REUSE_GATE:
                        continue

                    # Position cost (from actual previous position)
                    dist = math.hypot(dc[0] - prev_c[0], dc[1] - prev_c[1])

                    # Velocity consistency: penalise detections whose displacement
                    # differs from the slot's expected velocity vector
                    obs_dx   = dc[0] - prev_c[0]
                    obs_dy   = dc[1] - prev_c[1]
                    vel_diff = math.hypot(obs_dx - vel[0], obs_dy - vel[1])

                    # Appearance cost (Bhattacharyya; 0=identical, 1=very different)
                    app_cost = 0.0
                    if (self._use_appearance
                            and d.get('hist') is not None
                            and st['hist'] is not None):
                        app_cost = float(
                            cv2.compareHist(d['hist'], st['hist'],
                                            cv2.HISTCMP_BHATTACHARYYA))

                    cost = (dist
                            + self._VEL_MATCH_LAMBDA * vel_diff
                            + self._APP_MATCH_LAMBDA * app_cost)
                    if best is None or cost < best[0]:
                        best = (cost, d, slot)
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
            st      = self._fencer_state[slot]
            new_c   = self._box_center(d['box'])
            prev_c  = st['center']
            st['velocity'] = (
                (new_c[0] - prev_c[0], new_c[1] - prev_c[1])
                if prev_c is not None else (0.0, 0.0)
            )
            st['raw_id'] = d['raw_id']
            st['center'] = new_c
            st['misses'] = 0
            if d.get('hist') is not None:
                st['hist'] = d['hist']

        for slot in (1, 2):
            if slot not in slots_touched:
                st = self._fencer_state[slot]
                st['misses'] += 1
                st['raw_id']   = None
                st['velocity'] = (0.0, 0.0)  # reset on miss — don't extrapolate identity
                if st['misses'] > self._max_state_misses:
                    st['center'] = None

        return [d for d in detections if d.get('track_id') in (1, 2)]

    # ------------------------------------------------------------------
    # Five-layer pipeline
    # ------------------------------------------------------------------

    def _process_yolo_pose(self, frame: np.ndarray) -> List[PersonPose]:
        detections = self._detect(frame)
        tracked    = self._track(detections)
        estimated  = self._estimate(tracked)
        corrected  = self._correct(estimated)
        return self._build_output(corrected)

    # --- Layer 1: Detection ---

    def _detect(self, frame: np.ndarray) -> List[dict]:
        """
        Raw YOLO inference only. No filtering, no rejection.
        Returns one dict per detected person: box, kp, raw_id.
        """
        results = self._yolo.track(frame, persist=True, conf=0.25, verbose=False, device='mps')
        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                raw_id = int(r.boxes.id[i]) if r.boxes.id is not None else i
                hist = (self._compute_hist(frame, (x1, y1, x2, y2))
                        if self._use_appearance else None)
                detections.append({
                    'box':    (x1, y1, x2, y2),
                    'raw_id': raw_id,
                    'kp':     r.keypoints.data[i].cpu().numpy(),
                    'hist':   hist,
                })
        return detections

    # --- Layer 2: Tracking (ID assignment only) ---

    def _track(self, detections: List[dict]) -> List[dict]:
        """
        Assign stable canonical IDs {1, 2}. Does not touch keypoints.
        Applies role-lock after assignment to absorb crossing artefacts.
        """
        selected = self._select_two_fencers(detections)
        assigned = self._assign_canonical_ids(selected)
        return self._check_role_lock(assigned)

    def _check_role_lock(self, assigned: List[dict]) -> List[dict]:
        """
        Enforce left=1, right=2 convention with hysteresis.

        When both fencers are present and ID 1 is consistently to the RIGHT of
        ID 2 for _ROLE_LOCK_FRAMES consecutive frames, commit a role swap —
        renaming IDs and mirroring _fencer_state so the convention is restored.
        Brief crossings (< _ROLE_LOCK_FRAMES frames) do not trigger a swap,
        keeping identities stable while fencers pass each other.
        """
        centers = {
            d['track_id']: self._box_center(d['box'])
            for d in assigned
            if d.get('track_id') in (1, 2)
        }

        if 1 in centers and 2 in centers:
            if centers[1][0] > centers[2][0]:   # ID 1 is right of ID 2 — wrong side
                self._role_mismatch_count += 1
            else:
                self._role_mismatch_count = 0

            if self._role_mismatch_count >= self._ROLE_LOCK_FRAMES:
                self._role_mismatch_count = 0
                if self._debug:
                    print(f"[F{self._frame_id}] Role lock: IDs 1↔2 swapped after "
                          f"{self._ROLE_LOCK_FRAMES} consecutive inverted frames")
                # Swap track_id labels in the current result
                for d in assigned:
                    if d.get('track_id') == 1:
                        d['track_id'] = 2
                    elif d.get('track_id') == 2:
                        d['track_id'] = 1
                # Mirror internal state so future assignments stay consistent
                self._fencer_state[1], self._fencer_state[2] = (
                    self._fencer_state[2], self._fencer_state[1]
                )
        else:
            self._role_mismatch_count = 0

        return assigned

    # --- Layer 3: State Estimation ---

    def _estimate(self, tracked: List[dict]) -> List[dict]:
        """
        Box-jump guard + Kalman filter.

        Accepted detections  → kalman.update()  (predict + measurement correction)
        Rejected detections  → kalman.predict() (predict only; bad kp discarded)

        _prev_boxes only advances on accepted detections — never from a
        predicted frame — so the guard reference stays clean.

        Passes to _correct():
          kp_filtered  — Kalman output (x, y, conf)
          kp_prev      — snapshot of last post-correction state (for swap check)
          jumped       — True when detection was rejected
        """
        # 5.1 Periodic velocity reset: zero Kalman vx/vy every N frames to
        # prevent accumulated drift from compounding over long sequences.
        do_vel_reset = (self._frame_id % self._VEL_RESET_INTERVAL == 0
                        and self._frame_id > 0)

        if self._debug:
            self._debug_data.clear()

        estimated = []
        for d in tracked:
            track_id = d['track_id']
            box      = d['box']
            prev_box = self._prev_boxes.get(track_id)
            self._reject_counts.setdefault(track_id, 0)

            # --- 5.2 Adaptive box-jump threshold ---
            # Threshold scales with the slot's recent velocity so fast-moving
            # fencers are not unfairly penalised by a fixed pixel limit.
            slot_vel = self._fencer_state[track_id]['velocity']
            vel_mag  = math.hypot(slot_vel[0], slot_vel[1])
            box_jump_thresh = self._BOX_JUMP_BASE + self._BOX_JUMP_SCALE * vel_mag

            jump_detected = False
            _jump_dist = 0.0
            if prev_box is not None:
                curr_c = self._box_center(box)
                prev_c = self._box_center(prev_box)
                _jump_dist = math.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
                jump_detected = _jump_dist > box_jump_thresh

            if jump_detected:
                self._reject_counts[track_id] += 1
                if self._debug:
                    print(f"[F{self._frame_id}] Slot {track_id}: detection rejected "
                          f"(jump={_jump_dist:.1f}px > thresh={box_jump_thresh:.1f}px)")
            else:
                self._reject_counts[track_id] = 0

            # 5.4 Reject-count recovery (Phase 0): force-accept after 5 consecutive
            # rejections so the filter re-locks if the guard fires too aggressively.
            if self._reject_counts[track_id] > 5:
                jump_detected = False
                self._reject_counts[track_id] = 0
                if self._debug:
                    print(f"[F{self._frame_id}] Slot {track_id}: recovery triggered "
                          f"(forced accept after 5 consecutive rejections)")

            # Snapshot post-correction state for swap detection in _correct().
            kp_prev = self._crossing_corrector.get_prev(track_id)

            # Capture predicted position before state advances (debug visualization).
            if self._debug:
                kp_predicted = self._kalman.predict_preview(track_id)
                if kp_predicted is not None and kp_prev is not None:
                    kp_predicted[:, 2] = kp_prev[:, 2]
                self._debug_data[track_id] = {
                    'kp_raw':       d['kp'],
                    'kp_predicted': kp_predicted,
                }

            # --- Kalman step ---
            if jump_detected:
                # Predict only — bad detection never touches filter state.
                kp_filtered = self._kalman.predict(track_id)
                if kp_filtered is None:
                    continue   # no prior state → skip until first real detection
                # Carry confidence from last corrected frame so crossing-check
                # thresholds remain valid during rejected-detection runs.
                if kp_prev is not None:
                    kp_filtered[:, 2] = kp_prev[:, 2]
                out_box = prev_box
            else:
                self._prev_boxes[track_id] = box
                kp_filtered = self._kalman.update(track_id, d['kp'])
                out_box = box

            # 5.1 Periodic velocity reset applied after the Kalman step so the
            # current frame's output is unaffected; the reset takes effect next frame.
            if do_vel_reset:
                self._kalman.reset_velocity(track_id)

            if self._debug:
                kal_state = self._kalman.get_state(track_id)
                dbg = self._debug_data.setdefault(track_id, {})
                dbg['kp_filtered'] = kp_filtered
                if kal_state is not None:
                    dbg['velocities'] = kal_state[:, 2:4].copy()

            estimated.append({
                **d,
                'box':         out_box,
                'kp_filtered': kp_filtered,
                'kp_prev':     kp_prev,
                'jumped':      jump_detected,
            })

        return estimated

    # --- Layer 4: Pose Correction ---

    def _correct(self, estimated: List[dict]) -> List[dict]:
        """
        4a  Swap correction   — _swap_corrector.apply(kp, kp_prev)
            Only on accepted detections (jumped=False) with a prior reference.
            Skipped on Kalman-predict frames — no new measurement to compare.

        4b  Crossing correction — _crossing_corrector.apply(track_id, kp)
            Stores corrected result as canonical state so next frame's kp_prev
            is the post-correction reference.
        """
        corrected = []
        for d in estimated:
            track_id = d['track_id']
            kp       = d['kp_filtered']
            kp_prev  = d['kp_prev']

            if not d['jumped'] and kp_prev is not None:
                kp_orig = kp.copy() if self._debug else None
                kp = self._swap_corrector.apply(kp, kp_prev)
                if self._debug and not np.array_equal(kp[:, :2], kp_orig[:, :2]):
                    print(f"[F{self._frame_id}] Slot {track_id}: left/right swap applied")

            kp_pre_cross = kp.copy() if self._debug else None
            kp = self._crossing_corrector.apply(track_id, kp)
            if self._debug and not np.array_equal(kp[:, :2], kp_pre_cross[:, :2]):
                print(f"[F{self._frame_id}] Slot {track_id}: crossing correction applied")

            corrected.append({**d, 'kp_corrected': kp})

        return corrected

    # --- Layer 5: Output ---

    def _build_output(self, corrected: List[dict]) -> List[PersonPose]:
        """
        Build PersonPose objects and prune stale tracking state.
        """
        poses      = []
        active_ids = set()

        for d in corrected:
            track_id = d['track_id']
            active_ids.add(track_id)
            angles = _extract_angles_coco(d['kp_corrected'])
            poses.append(PersonPose(
                track_id=track_id,
                box=d['box'],
                keypoints=d['kp_corrected'].tolist(),
                **angles,
            ))

        for slot in list(self._prev_boxes):
            if slot not in active_ids:
                del self._prev_boxes[slot]
                self._reject_counts.pop(slot, None)

        self._kalman.prune(active_ids)
        self._crossing_corrector.prune(active_ids)
        # _swap_corrector is stateless — no pruning needed
        self._frame_id += 1
        return sorted(poses, key=lambda p: p.track_id)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video = Config.PROJECT_ROOT / "sample" / "fencing2.mp4"

    # --- Mode 1: YOLO pose model (one-pass, fastest) ---
    print("\n=== Mode: yolo_pose ===")
    with YoloPoseEngine(model_size='s') as engine:
        engine.process_video(video, skeleton_only=False)

