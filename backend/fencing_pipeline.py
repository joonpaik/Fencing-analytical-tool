import cv2
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, List

from yoloengine3 import YoloPoseEngine
from strip_tracker import HomographyTracker
from fencer_coordinator import FencerCoordinator
from strip_analyzer import StripAnalyzer
from config import Config


class FencingAnalysisPipeline:
    """
    Top-level entry point that wires together all analysis layers:
      Layer 1-5  YoloPoseEngine   — detection, tracking, Kalman, correction, output
      Strip      HomographyTracker — strip detection + homography
      Coord      FencerCoordinator — strip-relative positions, zones, actions
      Analyzer   StripAnalyzer     — per-session statistics
    """

    def __init__(self, model_size: str = 'n', debug: bool = False):
        self._engine      = YoloPoseEngine(model_size, debug)
        self._strip       = HomographyTracker()
        self._coordinator = FencerCoordinator()
        self._analyzer    = StripAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path,
        output_path=None,
    ) -> dict:
        """
        Run the full analysis pipeline on a video file.

        Returns the StripAnalyzer report dict (JSON-serialisable).
        Writes an annotated output video alongside.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)   or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._coordinator.fps  = fps
        self._analyzer.fps     = fps

        if output_path is None:
            Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = Config.OUTPUT_DIR / f"{video_path.stem}_pipeline.mp4"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )

        print(f"[FencingAnalysisPipeline] {video_path.name}  "
              f"{width}x{height} @ {fps:.1f}fps  ({total} frames)")

        frame_id   = 0
        t_start    = time.perf_counter()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            poses      = self._engine.process_frame(frame)
            geom       = self._strip.update(frame)

            result     = {}
            if geom is not None and geom.confidence > 0.3:
                result = self._coordinator.process(
                    poses, self._strip, frame_id, self._engine)

            pred_slots = self._engine.get_prediction_slots()
            self._analyzer.record(frame_id, result, pred_slots)

            self._engine.annotate_frame(frame, poses)
            self._annotate_strip(frame, geom, result)
            writer.write(frame)

            frame_id += 1
            if frame_id % 60 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {frame_id}/{total} frames | "
                      f"{frame_id / elapsed:.1f} fps")

        cap.release()
        writer.release()

        wall = time.perf_counter() - t_start
        print(f"\n[FencingAnalysisPipeline] done — {frame_id} frames "
              f"in {wall:.1f}s → {output_path}")

        return self._analyzer.report()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _annotate_strip(
        self,
        frame: np.ndarray,
        geom,
        result: dict,
    ) -> None:
        """
        Overlay strip geometry and fencer positions on the frame in-place.

        Strip corners  — green quadrilateral
        Fencer 1 hip   — blue circle + label (left fencer)
        Fencer 2 hip   — red  circle + label (right fencer)
        """
        if geom is not None:
            corners = geom.corners_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [corners], isClosed=True,
                          color=(0, 255, 0), thickness=2)

        # fencer_1 = blue (BGR 255,0,0), fencer_2 = red (BGR 0,0,255)
        _SIDE_COLOUR = {
            'fencer_1': (255, 0,   0),
            'fencer_2': (0,   0, 255),
        }

        for key, colour in _SIDE_COLOUR.items():
            fdata = result.get(key)
            if fdata is None:
                continue
            hx = int(fdata.get('hip_center_x', 0) or 0)
            hy = int(fdata.get('hip_center_y', 0) or 0)
            if hx <= 0 or hy <= 0:
                continue

            cv2.circle(frame, (hx, hy), 9, colour, -1)

            zone  = fdata.get('zone') or ''
            label = f"{key}  {zone}"
            cv2.putText(
                frame, label,
                (hx + 12, hy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA,
            )
