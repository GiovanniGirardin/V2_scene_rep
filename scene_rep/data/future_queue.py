from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


class FutureQueue:
    """
    Stores short state-action sequences for SLT training.

    This is separate from the SAC replay buffer.

    Each item is:
        {
            "obs": observation dict,
            "action": np.ndarray [2]
        }
    """

    def __init__(self, config: Dict[str, Any]):
        self.future_horizon = int(config["slt"]["future_horizon"])
        self.maxlen = self.future_horizon + 1
        self.queue: Deque[Dict[str, Any]] = deque(maxlen=self.maxlen)

    def reset(self) -> None:
        self.queue.clear()

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> None:
        self.queue.append(
            {
                "obs": obs,
                "action": np.asarray(action, dtype=np.float32),
            }
        )

    def is_ready(self) -> bool:
        return len(self.queue) == self.maxlen

    def get_sequence(self) -> Dict[str, np.ndarray]:
        """
        Return stacked sequence.

        Shapes:
            motion:
                [T, num_agents, history_len, motion_dim]

            waypoints:
                [T, num_agents, max_routes, waypoint_len, waypoint_dim]

            actions:
                [T, 2]
        """
        if not self.is_ready():
            raise RuntimeError(
                f"FutureQueue not ready: len={len(self.queue)}, required={self.maxlen}"
            )

        items: List[Dict[str, Any]] = list(self.queue)

        return {
            "motion": np.stack([x["obs"]["motion"] for x in items], axis=0),
            "waypoints": np.stack([x["obs"]["waypoints"] for x in items], axis=0),
            "agent_mask": np.stack([x["obs"]["agent_mask"] for x in items], axis=0),
            "route_mask": np.stack([x["obs"]["route_mask"] for x in items], axis=0),
            "actions": np.stack([x["action"] for x in items], axis=0),
        }

    def __len__(self) -> int:
        return len(self.queue)