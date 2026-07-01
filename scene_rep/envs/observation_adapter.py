from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


class ObservationAdapter:
    """
    Converts simulator observations into Scene-Rep-Transformer tensors.

    Output format:
        motion: [num_agents, history_len, 5]
        waypoints: [num_agents, max_candidate_routes, waypoint_len, 3]
        agent_mask: [num_agents]
        route_mask: [num_agents, max_candidate_routes]

    The SMARTS path supports two coordinate modes:
        ego: previous ego-relative representation.
        absolute: closer to the original Scene-Rep-Transformer SMARTS adapter,
                  using [x, y, heading, vx, vy] histories and stable neighbor IDs.
    """

    def __init__(self, config: Dict[str, Any]):
        obs_cfg = config["observation"]

        self.history_len = int(obs_cfg["history_len"])
        self.max_neighbors = int(obs_cfg["max_neighbors"])
        self.num_agents = self.max_neighbors + 1

        self.max_candidate_routes = int(obs_cfg["max_candidate_routes"])
        self.waypoint_len = int(obs_cfg["waypoint_len"])

        self.motion_dim = int(config["model"]["motion_dim"])
        self.waypoint_dim = int(config["model"]["waypoint_dim"])
        self.coordinate_frame = str(obs_cfg.get("coordinate_frame", "ego")).lower()

        self.history: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.ego_history: Deque[np.ndarray] = deque(maxlen=self.history_len)
        self.neighbor_histories: Dict[str, Deque[np.ndarray]] = {}
        self.current_neighbor_ids: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.history.clear()
        self.ego_history.clear()
        self.neighbor_histories.clear()
        self.current_neighbor_ids = []

    def adapt(self, raw_obs: Dict[str, Any] | None = None) -> Dict[str, np.ndarray]:
        if self.coordinate_frame == "absolute" and raw_obs is not None and "obs" in raw_obs:
            return self._adapt_absolute_smarts(
                raw_obs["obs"],
                raw_obs.get("neighbor_obs", {}),
            )

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
    # Original-style SMARTS adapter
    # ------------------------------------------------------------------
    def _adapt_absolute_smarts(
        self,
        smarts_obs: Any,
        neighbor_obs: Dict[str, Any] | None = None,
    ) -> Dict[str, np.ndarray]:
        ego_state = smarts_obs.ego_vehicle_state
        ego_feature = self._vehicle_absolute_feature(ego_state)
        self.ego_history.append(ego_feature)

        neighbors = self._get_sorted_neighbors(smarts_obs)
        self.current_neighbor_ids = []

        for vehicle in neighbors[: self.max_neighbors]:
            vehicle_id = str(getattr(vehicle, "id", len(self.current_neighbor_ids)))
            self.current_neighbor_ids.append(vehicle_id)
            if vehicle_id not in self.neighbor_histories:
                self.neighbor_histories[vehicle_id] = deque(maxlen=self.history_len)
            self.neighbor_histories[vehicle_id].append(
                self._vehicle_absolute_feature(vehicle)
            )

        active_ids = set(self.current_neighbor_ids)
        stale_ids = [
            vehicle_id
            for vehicle_id in self.neighbor_histories
            if vehicle_id not in active_ids
        ]
        for vehicle_id in stale_ids:
            # The original adapter only returns currently observed nearest vehicles.
            # Clearing stale IDs keeps empty slots genuinely empty after a vehicle leaves view.
            del self.neighbor_histories[vehicle_id]

        motion = np.zeros(
            (self.num_agents, self.history_len, self.motion_dim),
            dtype=np.float32,
        )
        motion[0] = self._pad_history(self.ego_history)

        agent_mask = np.zeros((self.num_agents,), dtype=np.float32)
        agent_mask[0] = 1.0

        for slot, vehicle_id in enumerate(self.current_neighbor_ids[: self.max_neighbors], start=1):
            motion[slot] = self._pad_history(self.neighbor_histories[vehicle_id])
            agent_mask[slot] = 1.0

        waypoints, route_mask = self._build_absolute_waypoints(
            smarts_obs,
            neighbor_obs or {},
        )

        return {
            "motion": motion,
            "waypoints": waypoints,
            "agent_mask": agent_mask,
            "route_mask": route_mask,
        }

    def _vehicle_absolute_feature(self, vehicle: Any) -> np.ndarray:
        pos = vehicle.position
        heading = float(vehicle.heading)
        speed = float(getattr(vehicle, "speed", 0.0))
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        linear_velocity = getattr(vehicle, "linear_velocity", None)
        if linear_velocity is not None:
            vx = float(linear_velocity[0])
            vy = float(linear_velocity[1])

        return np.array(
            [float(pos[0]), float(pos[1]), heading, vx, vy],
            dtype=np.float32,
        )

    def _pad_history(self, history: Deque[np.ndarray]) -> np.ndarray:
        padded = np.zeros((self.history_len, self.motion_dim), dtype=np.float32)
        values = list(history)[-self.history_len :]
        if not values:
            return padded
        start = self.history_len - len(values)
        for idx, value in enumerate(values):
            padded[start + idx] = value
        return padded

    def _build_absolute_waypoints(
        self,
        smarts_obs: Any,
        neighbor_obs: Dict[str, Any],
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

        waypoint_paths = getattr(smarts_obs, "waypoint_paths", [])
        for route_idx, path in enumerate(waypoint_paths[: self.max_candidate_routes]):
            self._copy_waypoint_path(
                waypoints=waypoints,
                route_mask=route_mask,
                agent_idx=0,
                route_idx=route_idx,
                path=path,
                is_ego=True,
            )

        neighbors = self._get_sorted_neighbors(smarts_obs)
        for slot, vehicle in enumerate(neighbors[: self.max_neighbors], start=1):
            vehicle_id = str(getattr(vehicle, "id", ""))
            vehicle_obs = neighbor_obs.get(vehicle_id)
            vehicle_paths = getattr(vehicle_obs, "waypoint_paths", None) if vehicle_obs is not None else None
            if vehicle_paths:
                for route_idx, path in enumerate(vehicle_paths[: self.max_candidate_routes]):
                    self._copy_waypoint_path(
                        waypoints=waypoints,
                        route_mask=route_mask,
                        agent_idx=slot,
                        route_idx=route_idx,
                        path=path,
                        is_ego=False,
                    )
            else:
                self._fill_neighbor_forward_waypoints(waypoints, route_mask, slot, vehicle)

        return waypoints, route_mask

    def _copy_waypoint_path(
        self,
        waypoints: np.ndarray,
        route_mask: np.ndarray,
        agent_idx: int,
        route_idx: int,
        path: Any,
        is_ego: bool,
    ) -> None:
        # The original adapter skips the current waypoint and uses the planned path ahead.
        path_ahead = list(path)[1 : 1 + self.waypoint_len]
        if not path_ahead:
            path_ahead = list(path)[: self.waypoint_len]

        for t, wp in enumerate(path_ahead[: self.waypoint_len]):
            pos = wp.pos
            waypoints[agent_idx, route_idx, t, 0] = float(pos[0])
            waypoints[agent_idx, route_idx, t, 1] = float(pos[1])
            waypoints[agent_idx, route_idx, t, 2] = float(wp.heading)
            if self.waypoint_dim >= 5:
                waypoints[agent_idx, route_idx, t, 3] = 1.0 if is_ego else 0.0
                waypoints[agent_idx, route_idx, t, 4] = 0.0 if is_ego else 1.0

        route_mask[agent_idx, route_idx] = 1.0

    def _fill_neighbor_forward_waypoints(
        self,
        waypoints: np.ndarray,
        route_mask: np.ndarray,
        agent_idx: int,
        vehicle: Any,
    ) -> None:
        pos = vehicle.position
        heading = float(vehicle.heading)
        speed = max(float(getattr(vehicle, "speed", 0.0)), 1.0)
        dx = np.cos(heading)
        dy = np.sin(heading)

        for t in range(self.waypoint_len):
            distance = 0.1 * speed * float(t + 1)
            waypoints[agent_idx, 0, t, 0] = float(pos[0]) + dx * distance
            waypoints[agent_idx, 0, t, 1] = float(pos[1]) + dy * distance
            waypoints[agent_idx, 0, t, 2] = heading
            if self.waypoint_dim >= 5:
                waypoints[agent_idx, 0, t, 3] = 0.0
                waypoints[agent_idx, 0, t, 4] = 1.0

        route_mask[agent_idx, 0] = 1.0

    # ------------------------------------------------------------------
    # Previous ego-relative adapter
    # ------------------------------------------------------------------
    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _to_ego_xy(self, xy: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
        delta = np.asarray(xy, dtype=np.float32)[:2] - np.asarray(ego_xy, dtype=np.float32)[:2]
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        return np.array(
            [
                cos_h * delta[0] - sin_h * delta[1],
                sin_h * delta[0] + cos_h * delta[1],
            ],
            dtype=np.float32,
        )

    def _to_ego_vec(self, vec: np.ndarray, ego_heading: float) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32)[:2]
        cos_h = np.cos(-ego_heading)
        sin_h = np.sin(-ego_heading)
        return np.array(
            [
                cos_h * vec[0] - sin_h * vec[1],
                sin_h * vec[0] + cos_h * vec[1],
            ],
            dtype=np.float32,
        )

    def _extract_current_motion(self, raw_obs: Dict[str, Any] | None) -> np.ndarray:
        current = np.zeros((self.num_agents, self.motion_dim), dtype=np.float32)

        if raw_obs is None:
            return current

        if "ego" in raw_obs:
            ego = raw_obs["ego"]
            ego_xy = np.array([float(ego["x"]), float(ego["y"])], dtype=np.float32)
            ego_heading = float(ego["heading"])
            ego_vel = self._to_ego_vec(
                np.array(
                    [float(ego.get("vx", 0.0)), float(ego.get("vy", 0.0))],
                    dtype=np.float32,
                ),
                ego_heading,
            )
            current[0] = np.array([0.0, 0.0, float(ego_vel[0]), float(ego_vel[1]), 0.0], dtype=np.float32)

            neighbors = raw_obs.get("neighbors", [])
            for i, neighbor in enumerate(neighbors[: self.max_neighbors]):
                neighbor_xy = np.array([float(neighbor["x"]), float(neighbor["y"])], dtype=np.float32)
                neighbor_vel = np.array(
                    [float(neighbor.get("vx", 0.0)), float(neighbor.get("vy", 0.0))],
                    dtype=np.float32,
                )
                rel_xy = self._to_ego_xy(neighbor_xy, ego_xy, ego_heading)
                rel_vel = self._to_ego_vec(neighbor_vel, ego_heading)
                current[i + 1] = np.array(
                    [
                        float(rel_xy[0]),
                        float(rel_xy[1]),
                        float(rel_vel[0]),
                        float(rel_vel[1]),
                        self._wrap_angle(float(neighbor["heading"]) - ego_heading),
                    ],
                    dtype=np.float32,
                )
            return current

        smarts_obs = raw_obs["obs"]
        ego_state = smarts_obs.ego_vehicle_state
        ego_pos = ego_state.position
        ego_vel = ego_state.linear_velocity
        ego_xy = np.array([float(ego_pos[0]), float(ego_pos[1])], dtype=np.float32)
        ego_heading = float(ego_state.heading)
        ego_vel_xy = self._to_ego_vec(
            np.array([float(ego_vel[0]), float(ego_vel[1])], dtype=np.float32),
            ego_heading,
        )
        current[0] = np.array([0.0, 0.0, float(ego_vel_xy[0]), float(ego_vel_xy[1]), 0.0], dtype=np.float32)

        neighbors = self._get_sorted_neighbors(smarts_obs)
        for i, vehicle in enumerate(neighbors[: self.max_neighbors]):
            pos = vehicle.position
            speed = float(getattr(vehicle, "speed", 0.0))
            heading = float(vehicle.heading)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            rel_xy = self._to_ego_xy(np.array(pos[:2], dtype=np.float32), ego_xy, ego_heading)
            rel_vel = self._to_ego_vec(np.array([vx, vy], dtype=np.float32), ego_heading)
            current[i + 1] = np.array(
                [
                    float(rel_xy[0]),
                    float(rel_xy[1]),
                    float(rel_vel[0]),
                    float(rel_vel[1]),
                    self._wrap_angle(heading - ego_heading),
                ],
                dtype=np.float32,
            )

        return current

    def _build_motion_history(self) -> np.ndarray:
        motion = np.zeros((self.num_agents, self.history_len, self.motion_dim), dtype=np.float32)
        if not self.history:
            return motion
        history_list = list(self.history)
        start = self.history_len - len(history_list)
        for i, frame in enumerate(history_list):
            motion[:, start + i, :] = frame
        return motion

    def _build_agent_mask(self, current_motion: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.num_agents,), dtype=np.float32)
        for i in range(self.num_agents):
            if i == 0 or np.any(current_motion[i] != 0.0):
                mask[i] = 1.0
        return mask

    def _build_waypoints(self, raw_obs: Dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        waypoints = np.zeros(
            (self.num_agents, self.max_candidate_routes, self.waypoint_len, self.waypoint_dim),
            dtype=np.float32,
        )
        route_mask = np.zeros((self.num_agents, self.max_candidate_routes), dtype=np.float32)

        if raw_obs is None:
            return waypoints, route_mask

        if "ego" in raw_obs:
            agents = [raw_obs["ego"]] + raw_obs.get("neighbors", [])[: self.max_neighbors]
            ego = raw_obs["ego"]
            ego_xy = np.array([float(ego["x"]), float(ego["y"])], dtype=np.float32)
            ego_heading = float(ego["heading"])
            for agent_idx, agent in enumerate(agents):
                base_xy = self._to_ego_xy(
                    np.array([float(agent["x"]), float(agent["y"])], dtype=np.float32),
                    ego_xy,
                    ego_heading,
                )
                heading = self._wrap_angle(float(agent["heading"]) - ego_heading)
                candidate_offsets = [0.0, -3.5, 3.5]
                for route_idx, lane_offset in enumerate(candidate_offsets[: self.max_candidate_routes]):
                    for t in range(self.waypoint_len):
                        waypoints[agent_idx, route_idx, t, 0] = base_xy[0] + float(t + 1)
                        waypoints[agent_idx, route_idx, t, 1] = base_xy[1] + lane_offset
                        waypoints[agent_idx, route_idx, t, 2] = heading
                    route_mask[agent_idx, route_idx] = 1.0
            return waypoints, route_mask

        smarts_obs = raw_obs["obs"]
        ego_state = smarts_obs.ego_vehicle_state
        ego_xy = np.array([float(ego_state.position[0]), float(ego_state.position[1])], dtype=np.float32)
        ego_heading = float(ego_state.heading)
        waypoint_paths = getattr(smarts_obs, "waypoint_paths", [])

        for route_idx, path in enumerate(waypoint_paths[: self.max_candidate_routes]):
            for t, wp in enumerate(path[: self.waypoint_len]):
                pos = wp.pos
                rel_xy = self._to_ego_xy(np.array(pos[:2], dtype=np.float32), ego_xy, ego_heading)
                waypoints[0, route_idx, t, 0] = float(rel_xy[0])
                waypoints[0, route_idx, t, 1] = float(rel_xy[1])
                waypoints[0, route_idx, t, 2] = self._wrap_angle(float(wp.heading) - ego_heading)
            route_mask[0, route_idx] = 1.0

        neighbors = self._get_sorted_neighbors(smarts_obs)
        for i, vehicle in enumerate(neighbors[: self.max_neighbors]):
            agent_idx = i + 1
            pos = vehicle.position
            heading = float(vehicle.heading)
            rel_xy = self._to_ego_xy(np.array(pos[:2], dtype=np.float32), ego_xy, ego_heading)
            rel_heading = self._wrap_angle(heading - ego_heading)
            dx = np.cos(heading)
            dy = np.sin(heading)
            for t in range(self.waypoint_len):
                waypoints[agent_idx, 0, t, 0] = float(rel_xy[0]) + dx * float(t + 1)
                waypoints[agent_idx, 0, t, 1] = float(rel_xy[1]) + dy * float(t + 1)
                waypoints[agent_idx, 0, t, 2] = rel_heading
            route_mask[agent_idx, 0] = 1.0

        return waypoints, route_mask

    def _get_sorted_neighbors(self, smarts_obs: Any) -> List[Any]:
        ego_pos = smarts_obs.ego_vehicle_state.position
        neighbors = list(getattr(smarts_obs, "neighborhood_vehicle_states", []))
        neighbors.sort(
            key=lambda v: np.linalg.norm(
                np.array(v.position[:2], dtype=np.float32)
                - np.array(ego_pos[:2], dtype=np.float32)
            )
        )
        return neighbors[: self.max_neighbors]
