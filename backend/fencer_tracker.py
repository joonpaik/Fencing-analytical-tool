__all__ = ['FencerTracker']

import cv2
import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class FencerTracker:
    """
    Identity assignment, box-jump state, and EMA smoothing for the
    canonical two-fencer slots {1, 2}.
    """

    def __init__(self):
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
        self._prev_boxes: dict = {}    # slot -> last accepted raw (x1,y1,x2,y2)
        self._smooth_boxes: dict = {}  # slot -> EMA-smoothed (x1,y1,x2,y2) for output
        self._BOX_EMA_ALPHA  = 0.35    # smoothing strength (lower = smoother box)
        self._reject_counts: dict = {} # slot -> consecutive box-jump rejections

        self._debug = False

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_velocity(self, slot: int) -> Tuple[float, float]:
        return self._fencer_state[slot]['velocity']

    def get_prediction_slots(self) -> set:
        return {s for s, c in self._reject_counts.items() if c > 0}

    def get_smooth_box(self, slot: int) -> Optional[tuple]:
        return self._smooth_boxes.get(slot)

    # ------------------------------------------------------------------
    # Static helpers
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

    # ------------------------------------------------------------------
    # Identity assignment
    # ------------------------------------------------------------------

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
                    print(f"[Role lock] IDs 1↔2 swapped after "
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
