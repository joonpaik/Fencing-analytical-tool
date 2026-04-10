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
# Temporal smoother
# ---------------------------------------------------------------------------

class PoseSmoother:
    """
    Per-track EMA smoother for keypoint (x, y) coordinates.
    Confidence values are kept raw so visibility thresholds are unaffected.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: dict = {}  # track_id -> smoothed (N, 3) np.ndarray

    def smooth(self, track_id: int, kp: np.ndarray) -> np.ndarray:
        if track_id not in self._state:
            self._state[track_id] = kp.copy()
            return kp
        prev = self._state[track_id]
        out = prev.copy()
        out[:, :2] = self.alpha * kp[:, :2] + (1 - self.alpha) * prev[:, :2]
        out[:, 2]  = kp[:, 2]          # keep raw confidence
        self._state[track_id] = out
        return out

    def prune(self, active_ids: set):
        """Remove state for track IDs no longer present in the frame."""
        for tid in list(self._state):
            if tid not in active_ids:
                del self._state[tid]


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
    def xy(i):
        return (kp[i][0], kp[i][1])

    left_arm  = _angle(xy(5),  xy(7),  xy(9))
    right_arm = _angle(xy(6),  xy(8),  xy(10))
    left_leg  = _angle(xy(11), xy(13), xy(15))
    right_leg = _angle(xy(12), xy(14), xy(16))

    hip_cx = (kp[11][0] + kp[12][0]) / 2
    hip_cy = (kp[11][1] + kp[12][1]) / 2

    front = 'left' if kp[15][0] < kp[16][0] else 'right'
    front_knee = left_leg if front == 'left' else right_leg

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

HEAD_KP_INDICES = [0, 1, 2, 3, 4]  # nose, left_eye, right_eye, left_ear, right_ear

def _draw_skeleton_coco(frame, kp: np.ndarray,
                        dot_colour=(0, 0, 255),
                        line_colour=(0, 255, 0)):
    # Draw a single centroid point for all visible head keypoints
    head_pts = [(kp[i][0], kp[i][1]) for i in HEAD_KP_INDICES if kp[i][2] > 0.05]
    head_centre = None
    if head_pts:
        cx = int(sum(p[0] for p in head_pts) / len(head_pts))
        cy = int(sum(p[1] for p in head_pts) / len(head_pts))
        head_centre = (cx, cy)
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
            Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = Config.OUTPUT_DIR / f"{video_path.stem}_{self.mode}.mp4"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

        print(f"[YoloPoseEngine] {video_path.name}  {width}x{height} @ {fps:.1f}fps  ({total} frames)")

        frame_data = []
        frame_id   = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            poses = self.process_frame(frame)
            self.annotate_frame(frame, poses)
            writer.write(frame)

            # Store analysis data (first detected person)
            p = poses[0] if poses else None
            frame_data.append({
                'frame':           frame_id,
                'persons_detected': len(poses),
                'left_arm_angle':  p.left_arm_angle  if p else None,
                'right_arm_angle': p.right_arm_angle if p else None,
                'left_leg_angle':  p.left_leg_angle  if p else None,
                'right_leg_angle': p.right_leg_angle if p else None,
                'hip_center_x':    p.hip_center_x    if p else None,
            })

            frame_id += 1
            if frame_id % 60 == 0:
                print(f"  {frame_id}/{total} frames …")

        cap.release()
        writer.release()
        print(f"[YoloPoseEngine] done — {frame_id} frames → {output_path}")
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

    def _process_yolo_pose(self, frame: np.ndarray) -> List[PersonPose]:
        """One-pass: YOLO pose model → boxes + COCO-17 keypoints. Uses persistent tracking."""
        results = self._yolo.track(frame, persist=True, conf=0.25, verbose=False)
        poses   = []

        active_ids = set()

        for r in results:
            if r.keypoints is None:
                continue
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(r.boxes.id[i]) if r.boxes.id is not None else i
                active_ids.add(track_id)

                kp_data = r.keypoints.data[i].cpu().numpy()   # (17, 3): x, y, conf
                kp_data = self._smoother.smooth(track_id, kp_data)

                angles = _extract_angles_coco(kp_data)
                pose   = PersonPose(
                    box=(x1, y1, x2, y2),
                    keypoints=kp_data.tolist(),
                    **angles,
                )
                poses.append(pose)

        self._smoother.prune(active_ids)
        return poses

    def _process_yolo_mediapipe(self, frame: np.ndarray) -> List[PersonPose]:
        """
        Two-pass (reference architecture):
          1. YOLO detects person bounding boxes.
          2. MediaPipe PoseLandmarker runs on each cropped region.
        """
        results = self._yolo.track(frame, persist=True, conf=0.25, verbose=False)
        poses      = []
        active_ids = set()

        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(r.boxes.id[i]) if r.boxes.id is not None else i
                active_ids.add(track_id)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                ch, cw = crop.shape[:2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                det      = self._pose_detector.detect(mp_image)

                if not det.pose_landmarks:
                    continue

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
                    box=(x1, y1, x2, y2),
                    keypoints=kp_arr.tolist(),
                    **angles,
                )
                poses.append(pose)

        self._smoother.prune(active_ids)
        return poses


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
    video = Config.PROJECT_ROOT / "sample" / "fencing2.mp4"

    # --- Mode 1: YOLO pose model (one-pass, fastest) ---
    print("\n=== Mode: yolo_pose ===")
    with YoloPoseEngine(mode='yolo_pose', model_size='n') as engine:
        engine.process_video(video)

    # --- Mode 2: YOLO detection + MediaPipe pose (reference architecture) ---
    print("\n=== Mode: yolo_mediapipe ===")
    with YoloPoseEngine(mode='yolo_mediapipe', mp_model='full') as engine:
        engine.process_video(video)
