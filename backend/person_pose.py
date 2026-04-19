__all__ = ['PersonPose']

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


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
