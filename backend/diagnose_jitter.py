"""
Diagnostic script: trace raw YOLO keypoints and the full pipeline output
for frames 180-360 (covering 0:07-0:11 @ 30fps) and dump a CSV.

Run from backend/:
    python diagnose_jitter.py

Output: output/jitter_trace.csv
Columns (per frame, per joint):
    frame, slot, joint_idx, joint_name,
    raw_x, raw_y, raw_conf,       -- YOLO output before any filtering
    filt_x, filt_y,               -- after Kalman + all correctors
    delta_raw,                    -- distance from raw to filtered
    gate_fired                    -- 1 if Kalman gate fired on this joint
"""

import csv
import sys
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from yoloengine2 import YoloPoseEngine, YOLO_COCO_NAMES

# ---------- config ----------
VIDEO    = Config.PROJECT_ROOT / "sample" / "fencing2.mp4"
OUT_CSV  = Config.OUTPUT_DIR / "jitter_trace.csv"
# Frames ~0:06–0:12 at 30fps = 180–360; add a small margin.
START_FRAME = 180
END_FRAME   = 370
SLOT        = None   # None = both slots; or 1 / 2 to restrict

# Joint indices we care about most (upper body — where jitter is visible)
WATCH_JOINTS = list(range(5, 17))   # shoulders, elbows, wrists, hips, knees, ankles

# ---------- patched engine ----------

class TracingEngine(YoloPoseEngine):
    """Subclass that intercepts raw YOLO kp and the Kalman gate mask."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_rows = []

    # Override _estimate so we can capture raw kp + gate info per joint.
    def _estimate(self, tracked):
        from yoloengine2 import KalmanPoseFilter
        import math

        do_vel_reset = (self._frame_id % self._VEL_RESET_INTERVAL == 0
                        and self._frame_id > 0)

        estimated = []
        for d in tracked:
            track_id = d['track_id']
            box      = d['box']
            prev_box = self._prev_boxes.get(track_id)
            self._reject_counts.setdefault(track_id, 0)

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
            else:
                self._reject_counts[track_id] = 0

            if self._reject_counts[track_id] > 5:
                jump_detected = False
                self._reject_counts[track_id] = 0

            kp_prev = self._crossing_corrector.get_prev(track_id)

            # --- capture raw kp before Kalman ---
            raw_kp = d['kp'].copy()

            if jump_detected:
                kp_filtered = self._kalman.predict(track_id)
                if kp_filtered is None:
                    continue
                if kp_prev is not None:
                    kp_filtered[:, 2] = kp_prev[:, 2]
                out_box = prev_box
                gate_mask = np.zeros(len(raw_kp), dtype=bool)
                box_jumped = True
            else:
                self._prev_boxes[track_id] = box
                prev_smooth = self._smooth_boxes.get(track_id)
                if prev_smooth is None:
                    self._smooth_boxes[track_id] = box
                else:
                    a = self._BOX_EMA_ALPHA
                    self._smooth_boxes[track_id] = (
                        int(a * box[0] + (1 - a) * prev_smooth[0]),
                        int(a * box[1] + (1 - a) * prev_smooth[1]),
                        int(a * box[2] + (1 - a) * prev_smooth[2]),
                        int(a * box[3] + (1 - a) * prev_smooth[3]),
                    )
                kp_in = d['kp']
                arm_amb = self._arm_ambiguous.get(track_id, False)
                leg_amb = self._leg_ambiguous.get(track_id, False)
                if arm_amb or leg_amb:
                    kp_in = d['kp'].copy()
                    if arm_amb:
                        kp_in[7:11, 2] = np.minimum(kp_in[7:11, 2], 0.05)
                    if leg_amb:
                        kp_in[13:17, 2] = np.minimum(kp_in[13:17, 2], 0.05)

                # Compute gate mask the same way KalmanPoseFilter does
                kal_s = self._kalman._state.get(track_id)
                if kal_s is not None:
                    x_pred = kal_s[:, 0] + kal_s[:, 2]
                    y_pred = kal_s[:, 1] + kal_s[:, 3]
                    meas_dist = np.sqrt((kp_in[:, 0] - x_pred)**2 +
                                       (kp_in[:, 1] - y_pred)**2)
                    gate_mask = meas_dist > 55.0
                else:
                    gate_mask = np.zeros(len(raw_kp), dtype=bool)

                kp_filtered = self._kalman.update(track_id, kp_in)
                out_box = self._smooth_boxes.get(track_id, box)
                box_jumped = False

            if do_vel_reset:
                self._kalman.reset_velocity(track_id)

            # Store raw + filtered + gate info for this frame/slot
            self._trace_rows.append({
                'frame_id': self._frame_id,
                'slot': track_id,
                'box_jumped': int(box_jumped),
                'raw_kp': raw_kp,
                'filt_kp': kp_filtered,
                'gate_mask': gate_mask,
            })

            estimated.append({
                **d,
                'box':         out_box,
                'kp_filtered': kp_filtered,
                'kp_prev':     kp_prev,
                'jumped':      jump_detected,
            })

        return estimated


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"Cannot open {VIDEO}")
        return
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames @ {fps:.1f} fps")
    print(f"Tracing frames {START_FRAME}–{END_FRAME} ({START_FRAME/fps:.2f}s–{END_FRAME/fps:.2f}s)")

    engine = TracingEngine(model_size='s', debug=False)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id > END_FRAME:
            break
        if frame_id >= START_FRAME:
            engine.process_frame(frame)
        else:
            # Still need to advance engine state (tracking memory etc.)
            engine.process_frame(frame)
        frame_id += 1

    cap.release()

    # Write CSV
    rows = [r for r in engine._trace_rows
            if START_FRAME <= r['frame_id'] <= END_FRAME
            and (SLOT is None or r['slot'] == SLOT)]

    fieldnames = [
        'frame', 'time_s', 'slot', 'box_jumped',
        'joint_idx', 'joint_name',
        'raw_x', 'raw_y', 'raw_conf',
        'filt_x', 'filt_y',
        'delta_raw_to_filt',
        'gate_fired',
    ]
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            for ji in WATCH_JOINTS:
                rk = r['raw_kp'][ji]
                fk = r['filt_kp'][ji]
                delta = math.hypot(float(rk[0] - fk[0]), float(rk[1] - fk[1]))
                writer.writerow({
                    'frame':             r['frame_id'],
                    'time_s':            round(r['frame_id'] / fps, 3),
                    'slot':              r['slot'],
                    'box_jumped':        r['box_jumped'],
                    'joint_idx':         ji,
                    'joint_name':        YOLO_COCO_NAMES[ji],
                    'raw_x':             round(float(rk[0]), 1),
                    'raw_y':             round(float(rk[1]), 1),
                    'raw_conf':          round(float(rk[2]), 3),
                    'filt_x':            round(float(fk[0]), 1),
                    'filt_y':            round(float(fk[1]), 1),
                    'delta_raw_to_filt': round(delta, 1),
                    'gate_fired':        int(r['gate_mask'][ji]),
                })

    print(f"\nWrote {OUT_CSV}")

    # Print summary: frames where gate fired on any joint, per slot
    print("\n=== Gate-fire events (gate_fired=1) ===")
    gate_frames: dict = {}
    for r in rows:
        fid  = r['frame_id']
        slot = r['slot']
        gate_joints = [YOLO_COCO_NAMES[ji] for ji in WATCH_JOINTS
                       if r['gate_mask'][ji]]
        if gate_joints:
            gate_frames.setdefault((fid, slot), []).extend(gate_joints)

    for (fid, slot), joints in sorted(gate_frames.items()):
        print(f"  frame {fid:4d} ({fid/fps:.2f}s)  slot={slot}  gated={joints}")

    # Print frames where raw vs filtered displacement is large (potential jitter source)
    print("\n=== Large raw↔filt deltas (>15px, watch joints 9=l_wrist 10=r_wrist 5=l_sho 6=r_sho) ===")
    for r in rows:
        for ji in (5, 6, 9, 10):
            rk = r['raw_kp'][ji]
            fk = r['filt_kp'][ji]
            delta = math.hypot(float(rk[0] - fk[0]), float(rk[1] - fk[1]))
            if delta > 15:
                print(f"  frame {r['frame_id']:4d} ({r['frame_id']/fps:.2f}s)  "
                      f"slot={r['slot']}  {YOLO_COCO_NAMES[ji]:20s}  "
                      f"raw=({rk[0]:.0f},{rk[1]:.0f})  filt=({fk[0]:.1f},{fk[1]:.1f})  "
                      f"delta={delta:.1f}px  gate={int(r['gate_mask'][ji])}")

    # Also print consecutive filt-position jumps (frame-to-frame delta in filtered output)
    print("\n=== Consecutive filtered-output jumps (>10px/frame, watch joints) ===")
    prev_filt: dict = {}  # (slot, ji) -> prev filt pos
    for r in sorted(rows, key=lambda x: x['frame_id']):
        for ji in (5, 6, 7, 8, 9, 10):
            fk = r['filt_kp'][ji]
            key = (r['slot'], ji)
            if key in prev_filt:
                pf = prev_filt[key]
                jump = math.hypot(float(fk[0] - pf[0]), float(fk[1] - pf[1]))
                if jump > 10:
                    print(f"  frame {r['frame_id']:4d} ({r['frame_id']/fps:.2f}s)  "
                          f"slot={r['slot']}  {YOLO_COCO_NAMES[ji]:20s}  "
                          f"jump={jump:.1f}px  "
                          f"prev=({pf[0]:.1f},{pf[1]:.1f})→now=({fk[0]:.1f},{fk[1]:.1f})"
                          f"  gate={int(r['gate_mask'][ji])}")
            prev_filt[key] = fk[:2].copy()


if __name__ == '__main__':
    main()
