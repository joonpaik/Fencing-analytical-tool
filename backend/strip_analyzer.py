__all__ = ['StripAnalyzer']

import math
from typing import Dict, List, Optional, Set


class StripAnalyzer:
    """
    Accumulates per-frame analysis records and produces a summary report.

    Filtering rules applied in record():
      - Frames where corps_a_corps is True are skipped (positions unreliable).
      - Fencer slots listed in prediction_slots are skipped (Kalman-predicted
        frames, not real detections).
    """

    def __init__(self):
        self.fps: float = 30.0
        # per-side lists of {x_m, zone, action}
        self._data: Dict[str, List[dict]] = {
            'fencer_1': [],
            'fencer_2': [],
        }

    def record(
        self,
        frame_id: int,
        result: dict,
        prediction_slots: Set[int],
    ) -> None:
        """Store one frame's analysis data (with filtering)."""
        if result.get('corps_a_corps', False):
            return

        for slot, key in ((1, 'fencer_1'), (2, 'fencer_2')):
            if slot in prediction_slots:
                continue
            fdata = result.get(key)
            if fdata is None:
                continue
            self._data[key].append({
                'x_m':    fdata.get('x_m'),
                'zone':   fdata.get('zone'),
                'action': fdata.get('action'),
            })

    def report(self) -> dict:
        """
        Return a summary dict that is fully JSON-serialisable.

        Per side:
          mean_x_m    — mean strip x position (m)
          std_x_m     — standard deviation of strip x position (m)
          time_in_zone — {zone_name: seconds}
          zone_actions — {zone_name: {action: count}}
          action_bio   — {action: total_count}
          x_trace      — [x_m, ...] for every recorded frame
        """
        out: dict = {}

        for side, records in self._data.items():
            xs = [float(r['x_m']) for r in records if r['x_m'] is not None]

            if xs:
                mean_x = float(sum(xs) / len(xs))
                var_x  = float(sum((x - mean_x) ** 2 for x in xs) / len(xs))
                std_x  = float(math.sqrt(var_x))
            else:
                mean_x = 0.0
                std_x  = 0.0

            time_in_zone: Dict[str, float] = {}
            zone_actions: Dict[str, Dict[str, int]] = {}
            action_bio:   Dict[str, int] = {}

            for r in records:
                zone   = r['zone']
                action = str(r['action']) if r['action'] is not None else 'guard'

                action_bio[action] = int(action_bio.get(action, 0)) + 1

                if zone is not None:
                    zone = str(zone)
                    time_in_zone[zone] = float(
                        time_in_zone.get(zone, 0.0) + 1.0 / float(self.fps)
                    )
                    if zone not in zone_actions:
                        zone_actions[zone] = {}
                    zone_actions[zone][action] = int(
                        zone_actions[zone].get(action, 0)
                    ) + 1

            out[side] = {
                'mean_x_m':     round(float(mean_x), 4),
                'std_x_m':      round(float(std_x),  4),
                'time_in_zone': {
                    k: round(float(v), 3) for k, v in time_in_zone.items()
                },
                'zone_actions': {
                    z: {a: int(c) for a, c in ac.items()}
                    for z, ac in zone_actions.items()
                },
                'action_bio':   {k: int(v) for k, v in action_bio.items()},
                'x_trace':      [round(float(x), 4) for x in xs],
            }

        return out
