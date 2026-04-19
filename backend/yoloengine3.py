"""
YoloPoseEngine — body-pose marking for the Fencing Analysis tool.

Single operating mode: YOLO pose model outputs boxes + COCO-17 keypoints directly.
Uses models/yolo26n-pose.pt (fast) or yolo26s-pose.pt (accurate).
"""

import cv2
import math
import time
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

from config import Config

from person_pose import PersonPose
from pose_utils import (
    YOLO_POSE_CONNECTIONS, YOLO_COCO_NAMES,
    _angle, _ANGLE_CONF_GATE, _safe_angle,
    _extract_angles_coco, _draw_skeleton_coco,
    _segments_intersect, _legs_are_crossed,
    _arms_are_crossed, _torso_is_crossed,
)
from pose_corrector import (
    _INDEPENDENT_PAIRS, _LIMB_CHAINS, _SWAP_INERTIA,
    _fix_pair, _detect_and_fix_swaps, _pair_visible,
    SwapCorrector, CrossingCorrector,
    LateralIdentityCorrector, KalmanPoseFilter,
)


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
        self._yolo           = YOLO(str(pt_path))
        self._kalman         = KalmanPoseFilter()   # Layer 3: state estimation
        self._swap_corrector = SwapCorrector()       # Layer 4a: frame-level swap

        # Layer 4b–4d: persistent lateral-flip correctors.
        # Run in this order so each corrector's anchor joints are already clean
        # before the next one runs: torso fixes hips → legs use hips as anchor →
        # arms use shoulders as anchor (independent of the lower body).
        self._torso_corrector = LateralIdentityCorrector(
            anchor     = (5,  6),           # shoulders — reliable anchor
            check      = (11, 12),          # hips — potentially flipped
            swap_pairs = [(11, 12)],        # only swap hips; legs have their own corrector
            proximity  = None,              # same person's hips don't approach each other
        )
        self._leg_corrector = LateralIdentityCorrector(
            anchor     = (11, 12),          # hips (corrected by torso corrector above)
            check      = (15, 16),          # ankles — far end of leg chain
            swap_pairs = [(13, 14), (15, 16)],  # swap knees + ankles
            proximity  = (15, 16),          # ankles close together → ambiguous
        )
        self._arm_corrector = LateralIdentityCorrector(
            anchor     = (5,  6),           # shoulders — reliable anchor (NOT swapped)
            check      = (9,  10),          # wrists — far end of arm chain
            swap_pairs = [(7, 8), (9, 10)], # swap elbows + wrists only
            proximity  = (9,  10),          # wrists close together → ambiguous
        )
        self._crossing_corrector = CrossingCorrector()  # Layer 4e: geometric crossing + state

        # Ambiguity flags from the lateral correctors feed back into _estimate()
        # to clamp Kalman gain for the affected joints next frame.
        self._arm_ambiguous: dict = {}
        self._leg_ambiguous: dict = {}
        print(f"[YoloPoseEngine] mode=yolo_pose  model={pt_name}")

        from fencer_tracker import FencerTracker
        self._tracker = FencerTracker()

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
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_yolo_pose_skeleton_v3.mp4"
            else:
                output_path = Config.OUTPUT_DIR / f"{video_path.stem}_yolo_pose_v3.mp4"
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

    def get_slot_velocity(self, slot):
        return self._tracker.get_velocity(slot)

    def is_corps_a_corps(self):
        return (self._arm_ambiguous.get(1, False) and
                self._arm_ambiguous.get(2, False))

    def get_prediction_slots(self):
        return self._tracker.get_prediction_slots()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
                hist = (self._tracker._compute_hist(frame, (x1, y1, x2, y2))
                        if self._tracker._use_appearance else None)
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
        selected = self._tracker._select_two_fencers(detections)
        assigned = self._tracker._assign_canonical_ids(selected)
        return self._tracker._check_role_lock(assigned)

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
            prev_box = self._tracker._prev_boxes.get(track_id)
            self._tracker._reject_counts.setdefault(track_id, 0)

            # --- 5.2 Adaptive box-jump threshold ---
            # Threshold scales with the slot's recent velocity so fast-moving
            # fencers are not unfairly penalised by a fixed pixel limit.
            slot_vel = self._tracker._fencer_state[track_id]['velocity']
            vel_mag  = math.hypot(slot_vel[0], slot_vel[1])
            box_jump_thresh = self._tracker._BOX_JUMP_BASE + self._tracker._BOX_JUMP_SCALE * vel_mag

            jump_detected = False
            _jump_dist = 0.0
            if prev_box is not None:
                curr_c = self._tracker._box_center(box)
                prev_c = self._tracker._box_center(prev_box)
                _jump_dist = math.hypot(curr_c[0] - prev_c[0], curr_c[1] - prev_c[1])
                jump_detected = _jump_dist > box_jump_thresh

            if jump_detected:
                self._tracker._reject_counts[track_id] += 1
                if self._debug:
                    print(f"[F{self._frame_id}] Slot {track_id}: detection rejected "
                          f"(jump={_jump_dist:.1f}px > thresh={box_jump_thresh:.1f}px)")
            else:
                self._tracker._reject_counts[track_id] = 0

            # 5.4 Reject-count recovery (Phase 0): force-accept after 5 consecutive
            # rejections so the filter re-locks if the guard fires too aggressively.
            if self._tracker._reject_counts[track_id] > 5:
                jump_detected = False
                self._tracker._reject_counts[track_id] = 0
                # Reset Kalman state so the filter bootstraps from the new
                # detection position immediately.  Without this, the Kalman
                # state stays at the old (wrong) position and all joints remain
                # gated for ~25 more frames while K=0.1 slowly drifts toward
                # the correct position — producing visible skeleton displacement.
                self._kalman._state.pop(track_id, None)
                self._tracker._smooth_boxes.pop(track_id, None)
                if self._debug:
                    print(f"[F{self._frame_id}] Slot {track_id}: recovery triggered "
                          f"(forced accept + Kalman reset after 5 consecutive rejections)")

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
                self._tracker._prev_boxes[track_id] = box
                # EMA-smooth the box for output so the rectangle doesn't jump
                # with raw YOLO noise while the keypoints are Kalman-filtered.
                prev_smooth = self._tracker._smooth_boxes.get(track_id)
                if prev_smooth is None:
                    self._tracker._smooth_boxes[track_id] = box
                else:
                    a = self._tracker._BOX_EMA_ALPHA
                    self._tracker._smooth_boxes[track_id] = (
                        int(a * box[0] + (1 - a) * prev_smooth[0]),
                        int(a * box[1] + (1 - a) * prev_smooth[1]),
                        int(a * box[2] + (1 - a) * prev_smooth[2]),
                        int(a * box[3] + (1 - a) * prev_smooth[3]),
                    )
                # Pre-Kalman swap correction: fix YOLO left/right label swaps on
                # the raw measurement before the filter ingests them.
                # Running it only after Kalman (Layer 4a in _correct()) means the
                # filter has already pulled its state toward the swapped positions,
                # building velocity in the wrong direction.  On raw kp the cost
                # comparison is unambiguous (e.g. cost_keep=74 vs cost_swap=4),
                # whereas post-Kalman averaging can make the two costs nearly equal
                # and the inertia check fails to trigger.
                kp_in = d['kp']
                if kp_prev is not None:
                    kp_in = self._swap_corrector.apply(kp_in, kp_prev)

                # If a lateral corrector flagged ambiguity last frame, clamp the
                # affected joints' confidence to 0.05 so K stays near its minimum
                # (~0.23).  The filter trusts its own prediction over the unreliable
                # raw measurement during contact / close-proximity frames.
                arm_amb = self._arm_ambiguous.get(track_id, False)
                leg_amb = self._leg_ambiguous.get(track_id, False)
                if arm_amb or leg_amb:
                    if kp_in is d['kp']:   # ensure we hold a private copy
                        kp_in = kp_in.copy()
                    if arm_amb:
                        kp_in[7:11, 2] = np.minimum(kp_in[7:11, 2], 0.05)
                    if leg_amb:
                        kp_in[13:17, 2] = np.minimum(kp_in[13:17, 2], 0.05)
                kp_filtered = self._kalman.update(track_id, kp_in)
                out_box = self._tracker._smooth_boxes.get(track_id, box)

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
        4a  Swap correction         — frame-level left/right check vs kp_prev.
            Skipped on Kalman-predict frames (jumped=True).

        4b  Torso identity correction — fix persistent hip-label flips, anchored
            by shoulders.  Runs first so the leg corrector has a clean hip anchor.

        4c  Leg identity correction  — fix persistent knee+ankle label flips,
            anchored by (now-correct) hips.

        4d  Arm identity correction  — fix persistent elbow+wrist label flips,
            anchored by shoulders.

        4e  Geometric crossing correction — 8-frame temporal pressure on segment
            intersections.  Always last so it stores the final corrected kp as
            canonical state for next frame's kp_prev reference.
        """
        corrected = []
        for d in estimated:
            track_id = d['track_id']
            kp       = d['kp_filtered']
            kp_prev  = d['kp_prev']

            # 4a — frame-level swap
            if not d['jumped'] and kp_prev is not None:
                kp_orig = kp.copy() if self._debug else None
                kp = self._swap_corrector.apply(kp, kp_prev)
                if self._debug and not np.array_equal(kp[:, :2], kp_orig[:, :2]):
                    print(f"[F{self._frame_id}] Slot {track_id}: left/right swap applied")

            # 4b — torso (hips anchored by shoulders)
            kp_pre = kp.copy() if self._debug else None
            kp, _ = self._torso_corrector.apply(track_id, kp)
            if self._debug and not np.array_equal(kp[:, :2], kp_pre[:, :2]):
                print(f"[F{self._frame_id}] Slot {track_id}: torso identity correction applied")

            # 4c — legs (knees+ankles anchored by hips)
            kp_pre = kp.copy() if self._debug else None
            kp, leg_amb = self._leg_corrector.apply(track_id, kp)
            self._leg_ambiguous[track_id] = leg_amb
            if self._debug and not np.array_equal(kp[:, :2], kp_pre[:, :2]):
                print(f"[F{self._frame_id}] Slot {track_id}: leg identity correction applied")

            # 4d — arms (elbows+wrists anchored by shoulders)
            kp_pre = kp.copy() if self._debug else None
            kp, arm_amb = self._arm_corrector.apply(track_id, kp)
            self._arm_ambiguous[track_id] = arm_amb
            if self._debug and not np.array_equal(kp[:, :2], kp_pre[:, :2]):
                print(f"[F{self._frame_id}] Slot {track_id}: arm identity correction applied")

            # 4e — geometric crossing (writes canonical state)
            kp_pre = kp.copy() if self._debug else None
            kp = self._crossing_corrector.apply(track_id, kp)
            if self._debug and not np.array_equal(kp[:, :2], kp_pre[:, :2]):
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

        for slot in list(self._tracker._prev_boxes):
            if slot not in active_ids:
                del self._tracker._prev_boxes[slot]
                self._tracker._smooth_boxes.pop(slot, None)
                self._tracker._reject_counts.pop(slot, None)

        self._kalman.prune(active_ids)
        self._torso_corrector.prune(active_ids)
        self._leg_corrector.prune(active_ids)
        self._arm_corrector.prune(active_ids)
        self._crossing_corrector.prune(active_ids)
        # _swap_corrector is stateless — no pruning needed
        self._frame_id += 1
        return sorted(poses, key=lambda p: p.track_id)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video = Config.PROJECT_ROOT / "sample" / "inverson.mp4"

    # --- Mode 1: YOLO pose model (one-pass, fastest) ---
    print("\n=== Mode: yolo_pose ===")
    with YoloPoseEngine(model_size='s') as engine:
        engine.process_video(video, skeleton_only=False)

