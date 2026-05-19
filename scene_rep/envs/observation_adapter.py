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
        current = np.zeros(
            (self.num_agents, self.motion_dim),
            dtype=np.float32,
        )

        if raw_obs is None:
            return current

        # Dummy mode
        if "ego" in raw_obs:
            ego = raw_obs["ego"]

            current[0] = np.array(
                [
                    ego["x"],
                    ego["y"],
                    ego["vx"],
                    ego["vy"],
                    ego["heading"],
                ],
                dtype=np.float32,
            )

            neighbors = raw_obs.get("neighbors", [])

            for i, neighbor in enumerate(neighbors[: self.max_neighbors]):
                current[i + 1] = np.array(
                    [
                        neighbor["x"],
                        neighbor["y"],
                        neighbor["vx"],
                        neighbor["vy"],
                        neighbor["heading"],
                    ],
                    dtype=np.float32,
                )

            return current

        # Real SMARTS mode
        smarts_obs = raw_obs["obs"]

        ego_state = smarts_obs.ego_vehicle_state

        ego_pos = ego_state.position
        ego_vel = ego_state.linear_velocity

        current[0] = np.array(
            [
                float(ego_pos[0]),
                float(ego_pos[1]),
                float(ego_vel[0]),
                float(ego_vel[1]),
                float(ego_state.heading),
            ],
            dtype=np.float32,
        )

        neighbors = getattr(smarts_obs, "neighborhood_vehicle_states", [])

        for i, vehicle in enumerate(neighbors[: self.max_neighbors]):
            pos = vehicle.position
            vel = vehicle.linear_velocity

            current[i + 1] = np.array(
                [
                    float(pos[0]),
                    float(pos[1]),
                    float(vel[0]),
                    float(vel[1]),
                    float(vehicle.heading),
                ],
                dtype=np.float32,
            )

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

        if raw_obs is None:
            return waypoints, route_mask

        # Dummy mode
        if "ego" in raw_obs:
            agents = [raw_obs["ego"]] + raw_obs.get("neighbors", [])[: self.max_neighbors]

            for agent_idx, agent in enumerate(agents):
                base_x = float(agent["x"])
                base_y = float(agent["y"])
                heading = float(agent["heading"])

                candidate_offsets = [0.0, -3.5, 3.5]

                for route_idx, lane_offset in enumerate(
                    candidate_offsets[: self.max_candidate_routes]
                ):
                    for t in range(self.waypoint_len):
                        waypoints[agent_idx, route_idx, t, 0] = base_x + float(t + 1)
                        waypoints[agent_idx, route_idx, t, 1] = base_y + lane_offset
                        waypoints[agent_idx, route_idx, t, 2] = heading

                    route_mask[agent_idx, route_idx] = 1.0

            return waypoints, route_mask

        # Real SMARTS mode
        smarts_obs = raw_obs["obs"]

        waypoint_paths = getattr(smarts_obs, "waypoint_paths", [])

        for route_idx, path in enumerate(waypoint_paths[: self.max_candidate_routes]):
            for t, wp in enumerate(path[: self.waypoint_len]):
                pos = wp.pos

                waypoints[0, route_idx, t, 0] = float(pos[0])
                waypoints[0, route_idx, t, 1] = float(pos[1])
                waypoints[0, route_idx, t, 2] = float(wp.heading)

            route_mask[0, route_idx] = 1.0

        # For neighbors, use simple forward pseudo-waypoints from their current state.
        neighbors = getattr(smarts_obs, "neighborhood_vehicle_states", [])

        for i, vehicle in enumerate(neighbors[: self.max_neighbors]):
            agent_idx = i + 1
            pos = vehicle.position
            heading = float(vehicle.heading)

            dx = np.cos(heading)
            dy = np.sin(heading)

            for t in range(self.waypoint_len):
                waypoints[agent_idx, 0, t, 0] = float(pos[0]) + dx * float(t + 1)
                waypoints[agent_idx, 0, t, 1] = float(pos[1]) + dy * float(t + 1)
                waypoints[agent_idx, 0, t, 2] = heading

            route_mask[agent_idx, 0] = 1.0

        return waypoints, route_mask
