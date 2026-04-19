__all__ = ['FencerCoordinator', 'get_zone', 'classify_action', 'STRIP_ZONES']

import math
import numpy as np
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Strip zone map (metres from left end)
# ---------------------------------------------------------------------------

STRIP_ZONES: Dict[str, Tuple[float, float]] = {
    'left_end':   (0.0,   2.0),
    'left_warn':  (2.0,   3.0),
    'left_mid':   (3.0,   7.0),
    'right_mid':  (7.0,  11.0),
    'right_warn': (11.0, 12.0),
    'right_end':  (12.0, 14.0),
}


def get_zone(x_m: float) -> str:
    """Return the strip zone name for a given x position in metres."""
    if x_m < 0.0 or x_m > 14.0:
        return 'off_strip'
    for name, (lo, hi) in STRIP_ZONES.items():
        if lo <= x_m < hi:
            return name
    return 'right_end'   # catches x_m == 14.0 exactly


def classify_action(pose, prev_pose, x_vel: float, fps: float) -> str:
    """
    Classify the fencer's current action from pose angles and lateral velocity.

    x_vel is the raw frame-to-frame strip displacement (metres/frame).
    It is normalised to 30-fps equivalent before threshold comparisons so
    the same thresholds apply regardless of the source video frame-rate.

    Priority order: lunge > fleche > advance > retreat > guard (default).
    """
    x_vel_norm = x_vel * (30.0 / fps)

    knee   = pose.front_knee_angle
    weapon = pose.weapon_arm_angle

    if knee < 120 and weapon < 100 and abs(x_vel_norm) > 0.04:
        return 'lunge'
    if knee < 130 and abs(x_vel_norm) > 0.08:
        return 'fleche'
    if knee > 148 and x_vel_norm > 0.025:
        return 'advance'
    if knee > 148 and x_vel_norm < -0.025:
        return 'retreat'
    return 'guard'


# ---------------------------------------------------------------------------
# FencerCoordinator
# ---------------------------------------------------------------------------

_CORPS_DIST_M  = 0.6    # metres — strip-distance threshold for corps-à-corps
_ON_STRIP_X_LO = -0.5
_ON_STRIP_X_HI = 14.5
_ON_STRIP_Y_LO = -0.5
_ON_STRIP_Y_HI = 2.5


class FencerCoordinator:
    """
    Maps PersonPose objects onto the strip canvas and classifies actions.

    Maintains per-fencer previous x_m so that frame-to-frame lateral
    velocity can be computed for action classification.
    """

    def __init__(self, fps: float = 30.0):
        self.fps: float = fps
        # track_id -> previous strip x_m (None until first on-strip detection)
        self._prev_x: Dict[int, Optional[float]] = {1: None, 2: None}
        # track_id -> previous PersonPose
        self._prev_pose: Dict[int, object] = {}

    def process(self, poses, H_tracker, frame_id: int, engine) -> dict:
        """
        Process detected poses against the current strip homography.

        Returns a dict with:
          'fencer_1' / 'fencer_2'  — per-fencer data dicts (absent if not detected)
          'corps_a_corps'           — True when fencers are in close contact
        """
        result: dict = {}
        fencer_x: Dict[int, float] = {}   # track_id -> x_m, for corps distance check

        for pose in poses:
            tid = pose.track_id
            key = f'fencer_{tid}'

            x_m: Optional[float] = None
            y_m: Optional[float] = None
            zone:   Optional[str]   = None
            action: Optional[str]   = None

            hx = float(pose.hip_center_x)
            hy = float(pose.hip_center_y)

            if H_tracker.H is not None and hx > 0 and hy > 0:
                m = H_tracker.pixel_to_meters((hx, hy))
                if m is not None:
                    x_m = float(m[0])
                    y_m = float(m[1])

            # On-strip bounds check
            on_strip = (
                x_m is not None and y_m is not None
                and _ON_STRIP_X_LO <= x_m <= _ON_STRIP_X_HI
                and _ON_STRIP_Y_LO <= y_m <= _ON_STRIP_Y_HI
            )

            if on_strip:
                zone = get_zone(x_m)
                fencer_x[tid] = x_m

                # Lateral velocity (m/frame → normalised inside classify_action)
                prev_x    = self._prev_x.get(tid)
                x_vel     = (x_m - prev_x) if prev_x is not None else 0.0
                prev_pose = self._prev_pose.get(tid)
                action    = classify_action(pose, prev_pose, x_vel, self.fps)

                self._prev_x[tid]    = x_m
                self._prev_pose[tid] = pose
            else:
                self._prev_x[tid]    = None

            result[key] = {
                'left_arm_angle':   round(float(pose.left_arm_angle),   1),
                'right_arm_angle':  round(float(pose.right_arm_angle),  1),
                'left_leg_angle':   round(float(pose.left_leg_angle),   1),
                'right_leg_angle':  round(float(pose.right_leg_angle),  1),
                'hip_center_x':     round(hx, 1),
                'hip_center_y':     round(hy, 1),
                'front_knee_angle': round(float(pose.front_knee_angle), 1),
                'weapon_arm_angle': round(float(pose.weapon_arm_angle), 1),
                'x_m':    round(x_m, 4) if x_m is not None else None,
                'y_m':    round(y_m, 4) if y_m is not None else None,
                'zone':   zone,
                'action': action,
            }

        # Corps-à-corps: engine ambiguity flag OR strip distance < threshold
        corps = engine.is_corps_a_corps()
        if not corps and len(fencer_x) == 2:
            x1, x2 = list(fencer_x.values())
            corps = abs(x1 - x2) < _CORPS_DIST_M

        result['corps_a_corps'] = corps
        return result
