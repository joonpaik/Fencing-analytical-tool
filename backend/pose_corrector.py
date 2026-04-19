__all__ = [
    'SwapCorrector', 'CrossingCorrector',
    'LateralIdentityCorrector', 'KalmanPoseFilter',
]

import math
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

from pose_utils import (
    _segments_intersect, _legs_are_crossed,
    _arms_are_crossed, _torso_is_crossed,
)


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


class LateralIdentityCorrector:
    """
    Detects and corrects persistent left/right label flips in a body-part chain.

    Compares the lateral (x-axis) ordering of two *anchor* joints to two *check*
    joints over a rolling window.  If the orderings persistently disagree the
    listed swap_pairs are swapped to restore consistency.

    The anchor joints are assumed reliable and are NEVER modified — only the
    swap_pairs are touched.  This prevents the corrector from corrupting its own
    reference point and avoids oscillation with the frame-level swap corrector.

    Intended use — three instances, run in the order listed:
      torso  anchor=shoulders(5,6),  check=hips(11,12),
             swap_pairs=[(11,12)]                — fixes hip label flips
      legs   anchor=hips(11,12),     check=ankles(15,16),
             swap_pairs=[(13,14),(15,16)]        — fixes knee+ankle label flips
      arms   anchor=shoulders(5,6),  check=wrists(9,10),
             swap_pairs=[(7,8),(9,10)]           — fixes elbow+wrist label flips

    Running torso first ensures the hip anchor is correct before the leg
    corrector references it.

    Ambiguity guard (optional, via proximity parameter):
      When the two proximity joints are within ambiguity_dist pixels, history
      accumulation is paused and is_ambiguous=True is returned.  The caller
      should reduce Kalman gain for the affected joints that frame.  Once the
      joints separate, a lower mismatch threshold fires faster to catch any swap
      that occurred during contact before the history window fully refills.
    """

    def __init__(
        self,
        anchor          : Tuple[int, int],
        check           : Tuple[int, int],
        swap_pairs      : List[Tuple[int, int]],
        proximity       : Optional[Tuple[int, int]] = None,
        ambiguity_dist  : float = 50.0,
        history_len     : int   = 15,
        flip_ratio      : float = 0.70,
        post_cross_ratio: float = 0.50,
        conf_thresh     : float = 0.15,
    ):
        self._anchor          = anchor
        self._check           = check
        self._swap_pairs      = swap_pairs
        self._proximity       = proximity
        self._ambiguity_dist  = ambiguity_dist
        self._history_len     = history_len
        self._flip_ratio      = flip_ratio
        self._post_cross_ratio= post_cross_ratio
        self._conf_thresh     = conf_thresh
        self._history        : dict = {}  # track_id -> deque[bool]
        self._prev_ambiguous : dict = {}  # track_id -> bool

    def apply(self, track_id: int, kp: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Returns (corrected_kp, is_ambiguous).

        corrected_kp  — swap_pairs swapped when a persistent identity flip is
                        detected; otherwise identical to input.
        is_ambiguous  — True when the proximity joints are too close to reliably
                        determine correct labeling.
        """
        kp = kp.copy()
        al, ar = self._anchor
        cl, cr = self._check

        for i in (al, ar, cl, cr):
            if kp[i, 2] < self._conf_thresh:
                return kp, False

        # Proximity / ambiguity check
        is_ambiguous = False
        if self._proximity is not None:
            pl, pr = self._proximity
            if kp[pl, 2] >= self._conf_thresh and kp[pr, 2] >= self._conf_thresh:
                dist = math.hypot(float(kp[pl, 0] - kp[pr, 0]),
                                  float(kp[pl, 1] - kp[pr, 1]))
                is_ambiguous = dist < self._ambiguity_dist

        was_ambiguous = self._prev_ambiguous.get(track_id, False)
        self._prev_ambiguous[track_id] = is_ambiguous

        if not is_ambiguous:
            a_sign = np.sign(kp[al, 0] - kp[ar, 0])
            c_sign = np.sign(kp[cl, 0] - kp[cr, 0])
            signs_match = bool(a_sign == c_sign)

            hist = self._history.setdefault(track_id, deque(maxlen=self._history_len))
            hist.append(signs_match)

            # After ambiguity: shorter minimum window + lower threshold so a swap
            # is caught quickly before the history refills.
            min_frames = 3 if was_ambiguous else max(3, self._history_len // 3)
            if len(hist) >= min_frames:
                mismatch_frac = sum(1 for v in hist if not v) / len(hist)
                threshold = self._post_cross_ratio if was_ambiguous else self._flip_ratio

                if mismatch_frac >= threshold:
                    for lp, rp in self._swap_pairs:
                        kp[[lp, rp]] = kp[[rp, lp]]
                    hist.clear()   # reset so we don't trigger again next frame

        return kp, is_ambiguous

    def prune(self, active_ids: set):
        """Remove state for tracks no longer active."""
        for tid in list(self._history):
            if tid not in active_ids:
                del self._history[tid]
                self._prev_ambiguous.pop(tid, None)


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
        K     = np.clip(0.15 + 0.45 * confs, 0.2, 0.65)

        # --- measurement gating (5.3) ---
        # Gate calibrated from observed data: max legitimate per-frame displacement
        # is ~39 px (weapon-arm wrist strike); YOLO left/right leg swap errors are
        # always ≥105 px.  55 px sits firmly between the two populations.
        # Gated joints get K=0.1 so the filter coasts on its own prediction
        # instead of snapping to the erroneous measurement.
        _OUTLIER_GATE = 55.0
        meas_dist = np.sqrt((kp[:, 0] - x_pred) ** 2 + (kp[:, 1] - y_pred) ** 2)
        outlier   = meas_dist > _OUTLIER_GATE
        K = np.where(outlier, 0.1, K)

        x_new = x_pred + K * (kp[:, 0] - x_pred)
        y_new = y_pred + K * (kp[:, 1] - y_pred)

        # --- EMA velocity smoothing (5.4) ---
        # For accepted measurements, base velocity on the raw measurement
        # displacement so the prediction tracks actual speed (not the K-scaled
        # filtered displacement, which always undershoots during fast motion).
        # For gated (outlier) measurements, use 0 so the EMA decays velocity
        # to near-zero over ~3 frames. Using filtered displacement instead
        # would slowly drift the state toward the wrong position, causing the
        # correct measurement to appear as an outlier once YOLO recovers.
        _VEL_ALPHA = 0.5
        raw_vel_x = kp[:, 0] - x_prev
        raw_vel_y = kp[:, 1] - y_prev
        vel_src_x = np.where(outlier, 0.0, raw_vel_x)
        vel_src_y = np.where(outlier, 0.0, raw_vel_y)
        s[:, 0] = x_new
        s[:, 1] = y_new
        s[:, 2] = (1.0 - _VEL_ALPHA) * s[:, 2] + _VEL_ALPHA * vel_src_x
        s[:, 3] = (1.0 - _VEL_ALPHA) * s[:, 3] + _VEL_ALPHA * vel_src_y

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
