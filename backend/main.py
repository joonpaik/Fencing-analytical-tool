"""
Fencing Analysis - Body Pose Detection and Movement Classification
Uses MediaPipe Pose to extract joint angles and classify fencing movements.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import math
from collections import deque, Counter

from config import Config

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_landmark_coords(landmarks, landmark_enum, w, h):
    """Return (x_px, y_px, visibility) for a given landmark."""
    lm = landmarks[landmark_enum.value]
    return (lm.x * w, lm.y * h, lm.visibility)


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (degrees) at joint p2 formed by p1-p2-p3.
    Corresponds to Config.ANGLE_CALCULATION_JOINTS: shoulder-elbow-wrist / hip-knee-ankle.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return 0.0
    cos_a = np.dot(v1, v2) / denom
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def is_blurry(frame):
    """Return True when Laplacian variance is below Config.BLUR_THRESHOLD."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < Config.BLUR_THRESHOLD


def preprocess_frame(frame):
    """Apply brightness/contrast adjustment per Config.BRIGHTNESS_ALPHA/BETA."""
    return cv2.convertScaleAbs(
        frame,
        alpha=Config.BRIGHTNESS_ALPHA,
        beta=Config.BRIGHTNESS_BETA
    )


def extract_pose_features(landmarks, w, h):
    """
    Extract angles and positions from MediaPipe landmark list.

    Returns a dict with:
      - joints: raw (x, y, visibility) per key joint
      - left/right arm angles  (shoulder-elbow-wrist per ANGLE_CALCULATION_JOINTS)
      - left/right leg angles  (hip-knee-ankle per ANGLE_CALCULATION_JOINTS)
      - hip_center_x / hip_center_y for tracking body position
      - front_knee_angle for lunge detection
      - weapon_arm_angle (right arm; assumed weapon side)
    """
    def c(lm_enum):
        return get_landmark_coords(landmarks, lm_enum, w, h)

    joints = {
        'left_shoulder':  c(mp_pose.PoseLandmark.LEFT_SHOULDER),
        'right_shoulder': c(mp_pose.PoseLandmark.RIGHT_SHOULDER),
        'left_elbow':     c(mp_pose.PoseLandmark.LEFT_ELBOW),
        'right_elbow':    c(mp_pose.PoseLandmark.RIGHT_ELBOW),
        'left_wrist':     c(mp_pose.PoseLandmark.LEFT_WRIST),
        'right_wrist':    c(mp_pose.PoseLandmark.RIGHT_WRIST),
        'left_hip':       c(mp_pose.PoseLandmark.LEFT_HIP),
        'right_hip':      c(mp_pose.PoseLandmark.RIGHT_HIP),
        'left_knee':      c(mp_pose.PoseLandmark.LEFT_KNEE),
        'right_knee':     c(mp_pose.PoseLandmark.RIGHT_KNEE),
        'left_ankle':     c(mp_pose.PoseLandmark.LEFT_ANKLE),
        'right_ankle':    c(mp_pose.PoseLandmark.RIGHT_ANKLE),
    }

    # --- Arm angles: shoulder -> elbow -> wrist ---
    left_arm_angle = calculate_angle(
        joints['left_shoulder'][:2],
        joints['left_elbow'][:2],
        joints['left_wrist'][:2],
    )
    right_arm_angle = calculate_angle(
        joints['right_shoulder'][:2],
        joints['right_elbow'][:2],
        joints['right_wrist'][:2],
    )

    # --- Leg angles: hip -> knee -> ankle ---
    left_leg_angle = calculate_angle(
        joints['left_hip'][:2],
        joints['left_knee'][:2],
        joints['left_ankle'][:2],
    )
    right_leg_angle = calculate_angle(
        joints['right_hip'][:2],
        joints['right_knee'][:2],
        joints['right_ankle'][:2],
    )

    # Hip centre for body-level tracking
    hip_center_x = (joints['left_hip'][0] + joints['right_hip'][0]) / 2
    hip_center_y = (joints['left_hip'][1] + joints['right_hip'][1]) / 2

    # Front leg = whichever ankle is further left in frame (fencer faces right)
    front_leg = 'left' if joints['left_ankle'][0] < joints['right_ankle'][0] else 'right'
    front_knee_angle = left_leg_angle if front_leg == 'left' else right_leg_angle

    # Weapon arm (right arm assumed)
    weapon_arm_angle = right_arm_angle

    return {
        'joints': joints,
        'left_arm_angle': left_arm_angle,
        'right_arm_angle': right_arm_angle,
        'left_leg_angle': left_leg_angle,
        'right_leg_angle': right_leg_angle,
        'hip_center_x': hip_center_x,
        'hip_center_y': hip_center_y,
        'front_knee_angle': front_knee_angle,
        'weapon_arm_angle': weapon_arm_angle,
    }


def classify_movement(pose_history):
    """
    Rule-based movement classifier using Config thresholds.

    Considers the last ≤10 valid frames from the pose_history deque.
    Movement types from Config.MOVEMENT_TYPES: advance, retreat, lunge, parry, riposte.
    """
    valid = [p for p in pose_history if p is not None]
    if len(valid) < 3:
        return None

    window = valid[-min(10, len(valid)):]
    current = window[-1]

    # Hip x-velocity (pixels/frame) — positive = moving right
    hip_velocity = (current['hip_center_x'] - window[0]['hip_center_x']) / len(window)

    # Weapon arm angular velocity (degrees/frame)
    arm_angles = [p['weapon_arm_angle'] for p in window]
    arm_velocity = abs(arm_angles[-1] - arm_angles[0]) / len(window)

    front_knee_angle = current['front_knee_angle']
    vel_thr = Config.VELOCITY_THRESHOLD

    # Lunge: front knee sharply bent AND forward hip drive
    if front_knee_angle < 110 and abs(hip_velocity) > vel_thr * 5:
        return 'lunge'

    # Parry / Riposte: rapid weapon-arm movement
    if arm_velocity > vel_thr * 10:
        return 'riposte' if hip_velocity > vel_thr else 'parry'

    # Footwork
    if hip_velocity > vel_thr * 3:
        return 'advance'
    if hip_velocity < -vel_thr * 3:
        return 'retreat'

    return None


def draw_annotations(frame, features, movement, frame_idx):
    """Overlay joint angles and movement label on a frame."""
    h, _ = frame.shape[:2]

    if features:
        joints = features['joints']

        # Arm angle labels near each elbow
        for side, elbow_key, angle_key, colour in [
            ('L-Arm', 'left_elbow',  'left_arm_angle',  (255, 255, 0)),
            ('R-Arm', 'right_elbow', 'right_arm_angle', (255, 255, 0)),
        ]:
            ex = int(joints[elbow_key][0])
            ey = int(joints[elbow_key][1])
            cv2.putText(
                frame, f"{side}:{features[angle_key]:.0f}d",
                (ex - 35, ey - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA
            )

        # Leg angle labels near each knee
        for side, knee_key, angle_key, colour in [
            ('L-Leg', 'left_knee',  'left_leg_angle',  (0, 255, 255)),
            ('R-Leg', 'right_knee', 'right_leg_angle', (0, 255, 255)),
        ]:
            kx = int(joints[knee_key][0])
            ky = int(joints[knee_key][1])
            cv2.putText(
                frame, f"{side}:{features[angle_key]:.0f}d",
                (kx - 35, ky + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA
            )

    # Movement classification banner
    if movement:
        colour_map = {
            'advance':  (0,   255,   0),
            'retreat':  (0,     0, 255),
            'lunge':    (0,   165, 255),
            'parry':    (255,   0,   0),
            'riposte':  (255,   0, 255),
        }
        colour = colour_map.get(movement, (255, 255, 255))
        cv2.putText(
            frame, f"MOVEMENT: {movement.upper()}",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3, cv2.LINE_AA
        )

    # Frame counter
    cv2.putText(
        frame, f"Frame: {frame_idx}",
        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA
    )

    return frame


def process_video(video_path, output_path):
    """
    Full pipeline: read → preprocess → pose detection → feature extraction
    → movement classification → annotate → write.
    Returns a list of per-frame analysis dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_w, target_h = map(int, Config.TARGET_RESOLUTION.split('x'))
    out_fps = min(fps, Config.TARGET_FPS)

    print(f"Input : {video_path.name}  {orig_w}x{orig_h} @ {fps:.1f} fps  ({total} frames)")
    print(f"Output: {output_path.name}  {target_w}x{target_h} @ {out_fps:.1f} fps")

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        out_fps,
        (target_w, target_h),
    )

    pose_history = deque(maxlen=Config.SEQUENCE_LENGTH)
    frame_data   = []

    with mp_pose.Pose(
        min_detection_confidence=Config.CONFIDENCE_THRESHOLD,
        min_tracking_confidence=Config.CONFIDENCE_THRESHOLD,
        model_complexity=1,
    ) as pose:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.resize(frame, (target_w, target_h))
            blurry   = is_blurry(frame)
            features = None
            movement = None

            if blurry:
                pose_history.append(None)
                cv2.putText(frame, "BLURRY", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                processed = preprocess_frame(frame.copy())
                rgb        = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                results    = pose.process(rgb)

                if results.pose_landmarks:
                    # Draw skeleton on original-brightness frame
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117,  66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,  66, 230), thickness=2, circle_radius=2),
                    )

                    features = extract_pose_features(
                        results.pose_landmarks.landmark, target_w, target_h
                    )
                    pose_history.append(features)

                    if len(pose_history) >= 3:
                        movement = classify_movement(list(pose_history))
                else:
                    pose_history.append(None)

            frame = draw_annotations(frame, features, movement, frame_idx)
            writer.write(frame)

            frame_data.append({
                'frame':           frame_idx,
                'blurry':          blurry,
                'movement':        movement,
                'left_arm_angle':  features['left_arm_angle']  if features else None,
                'right_arm_angle': features['right_arm_angle'] if features else None,
                'left_leg_angle':  features['left_leg_angle']  if features else None,
                'right_leg_angle': features['right_leg_angle'] if features else None,
                'hip_center_x':    features['hip_center_x']    if features else None,
            })

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"  {frame_idx}/{total} frames processed …")

    cap.release()
    writer.release()
    print(f"  {frame_idx} frames total.")
    return frame_data


def main():
    video_path = Config.PROJECT_ROOT / "sample" / "fencing2.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("=== Fencing Analysis — Pose Detection ===\n")

    output_video  = Config.OUTPUT_DIR / "fencing2_annotated.mp4"
    output_report = Config.OUTPUT_DIR / "fencing2_analysis.json"

    frame_data = process_video(video_path, output_video)
    print(f"\nAnnotated video: {output_video}")

    if Config.SAVE_INTERMEDIATE_FILES:
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_report, 'w') as f:
            json.dump(frame_data, f, indent=2)
        print(f"Analysis JSON  : {output_report}")

    # Summary
    movements = [d['movement'] for d in frame_data if d['movement']]
    if movements:
        print("\n=== Movement Summary ===")
        for mov, count in Counter(movements).most_common():
            pct = count / len(frame_data) * 100
            print(f"  {mov:<10} {count:>5} frames  ({pct:.1f}%)")
    else:
        print("\nNo movements classified — check VELOCITY_THRESHOLD in config.py.")

    print("\nDone.")


if __name__ == "__main__":
    main()
