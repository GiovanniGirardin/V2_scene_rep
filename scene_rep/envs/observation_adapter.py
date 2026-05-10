from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict

import numpy as np


class ObservationAdapter:
    """
    Converts simulator observations into Scene-Rep-Transformer tensors.

    Output format:
        motion:
            [num_agents, history_len, 5]
            features = x, y, vx, vy, heading

        waypoints:
            [num_agents, max_candidate_routes, waypoint_len, 3]
            features = x, y, heading

        agent_mask:
            [num_agents]

        route_mask:
            [num_agents, max_candidate_routes]

    For now this adapter supports dummy observations.
    Later we will connect this to real SMARTS observations.
    """

    def __init__(self, config: Dict[str, Any]):
        obs_cfg = config["observation"]

        self.history_len = obs_cfg["history_len"]
        self.max_neighbors = obs_cfg["max_neighbors"]
        self.num_agents = self.max_neighbors + 1

        self.max_candidate_routes = obs_cfg["max_candidate_routes"]
        self.waypoint_len = obs_cfg["waypoint_len"]

        self.motion_dim = config["model"]["motion_dim"]
        self.waypoint_dim = config["model"]["waypoint_dim"]

        self.history: Deque[np.ndarray] = deque(maxlen=self.history_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.history.clear()

    def adapt(self, raw_obs: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
        """
        Convert raw SMARTS observation into model-ready tensors.

        Currently uses placeholder data if raw_obs is None.
        """
        current_motion = self._extract_current_motion(raw_obs)

        self.history.append(current_motion)

        motion = self._build_motion_history()
        waypoints, route_mask = self._build_waypoints(raw_obs)
        agent_mask = self._build_agent_mask(current_motion)

        return {
            "motion": motion,
            "waypoints": waypoints,
            "agent_mask": agent_mask,
            "route_mask": route_mask,
        }

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def _extract_current_motion(self, raw_obs: Dict[str, Any] | None) -> np.ndarray:
        """
        Extract current agent states.

        Shape:
            [num_agents, 5]
        """
        current = np.zeros(
            (self.num_agents, self.motion_dim),
            dtype=np.float32,
        )

        # Placeholder:
        # ego exists at origin with zero velocity.
        current[0] = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return current

    def _build_motion_history(self) -> np.ndarray:
        """
        Build fixed-length history tensor.

        If history is shorter than history_len, pad at the beginning.
        """
        motion = np.zeros(
            (self.num_agents, self.history_len, self.motion_dim),
            dtype=np.float32,
        )

        if not self.history:
            return motion

        history_list = list(self.history)
        start = self.history_len - len(history_list)

        for i, frame in enumerate(history_list):
            motion[:, start + i, :] = frame

        return motion

    def _build_agent_mask(self, current_motion: np.ndarray) -> np.ndarray:
        """
        Valid agent mask.

        For now only ego is valid.
        Later:
            ego + detected neighbors.
        """
        mask = np.zeros((self.num_agents,), dtype=np.float32)
        mask[0] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Waypoints
    # ------------------------------------------------------------------
    def _build_waypoints(
        self,
        raw_obs: Dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build candidate route waypoint tensor.

        Shape:
            waypoints:
                [num_agents, max_candidate_routes, waypoint_len, 3]

            route_mask:
                [num_agents, max_candidate_routes]
        """
        waypoints = np.zeros(
            (
                self.num_agents,
                self.max_candidate_routes,
                self.waypoint_len,
                self.waypoint_dim,
            ),
            dtype=np.float32,
        )

        route_mask = np.zeros(
            (self.num_agents, self.max_candidate_routes),
            dtype=np.float32,
        )

        # Placeholder ego route: straight line along x.
        for t in range(self.waypoint_len):
            waypoints[0, 0, t, 0] = float(t)
            waypoints[0, 0, t, 1] = 0.0
            waypoints[0, 0, t, 2] = 0.0

        route_mask[0, 0] = 1.0

        return waypoints, route_mask