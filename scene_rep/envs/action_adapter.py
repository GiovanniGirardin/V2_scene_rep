from __future__ import annotations

from typing import Dict, Tuple


class ActionAdapter:
    """
    Converts continuous SAC outputs into SMARTS-compatible actions.

    Input (from policy):
        action = (speed_norm, lane_change_continuous)
            speed_norm ∈ [-1, 1]
            lane_change_continuous ∈ [-1, 1]

    Output:
        speed in m/s
        lane_change ∈ {-1, 0, 1}
    """

    def __init__(self, config: Dict):
        action_cfg = config["action"]

        self.max_speed = action_cfg["max_speed_mps"]
        self.lane_bins = action_cfg["lane_change_bins"]
        self.lane_values = action_cfg.get(
            "lane_change_values",
            {
                "left": -1,
                "keep": 0,
                "right": 1,
            },
        )

    # ----------------------------
    # Main API
    # ----------------------------
    def adapt(self, action: Tuple[float, float]) -> Dict:
        """
        Convert policy output into SMARTS action dictionary.

        Returns
        -------
        dict:
            {
                "speed": float,
                "lane_change": int
            }
        """
        speed_norm, lane_raw = action

        speed = self._scale_speed(speed_norm)
        lane = self._discretize_lane(lane_raw)

        return {
            "speed": speed,
            "lane_change": lane,
        }

    # ----------------------------
    # Helpers
    # ----------------------------
    def _scale_speed(self, speed_norm: float) -> float:
        """
        Map [-1, 1] → [0, max_speed]
        """
        speed_norm = max(-1.0, min(1.0, speed_norm))
        return (speed_norm + 1.0) / 2.0 * self.max_speed

    def _discretize_lane(self, lane_raw: float) -> int:
        """
        Convert continuous lane signal into {-1, 0, 1}
        """
        if self.lane_bins["left"][0] <= lane_raw <= self.lane_bins["left"][1]:
            return int(self.lane_values["left"])
        elif self.lane_bins["right"][0] <= lane_raw <= self.lane_bins["right"][1]:
            return int(self.lane_values["right"])
        else:
            return int(self.lane_values.get("keep", 0))
