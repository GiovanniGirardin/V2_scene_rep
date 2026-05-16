from __future__ import annotations

from typing import Any, Dict

import numpy as np


class SequenceBuffer:
    """
    Replay buffer for SLT sequence training.

    Stores fixed-length future sequences:

        obs_t, obs_t+1, ..., obs_t+Tf
        action_t, action_t+1, ..., action_t+Tf

    Shapes:
        motion:
            [capacity, T, A, H, motion_dim]

        waypoints:
            [capacity, T, A, R, W, waypoint_dim]

        actions:
            [capacity, T, 2]
    """

    def __init__(self, config: Dict[str, Any]):
        obs_cfg = config["observation"]
        model_cfg = config["model"]
        sac_cfg = config["sac"]
        slt_cfg = config["slt"]

        self.capacity = int(slt_cfg.get("sequence_replay_size", 1000))
        self.batch_size = int(slt_cfg.get("batch_size", sac_cfg["batch_size"]))
        self.future_horizon = int(slt_cfg["future_horizon"])
        self.seq_len = self.future_horizon + 1

        self.num_agents = obs_cfg["max_neighbors"] + 1
        self.history_len = obs_cfg["history_len"]
        self.max_routes = obs_cfg["max_candidate_routes"]
        self.waypoint_len = obs_cfg["waypoint_len"]

        self.motion_dim = model_cfg["motion_dim"]
        self.waypoint_dim = model_cfg["waypoint_dim"]

        self.ptr = 0
        self.size = 0

        self.motion = np.zeros(
            (
                self.capacity,
                self.seq_len,
                self.num_agents,
                self.history_len,
                self.motion_dim,
            ),
            dtype=np.float32,
        )

        self.waypoints = np.zeros(
            (
                self.capacity,
                self.seq_len,
                self.num_agents,
                self.max_routes,
                self.waypoint_len,
                self.waypoint_dim,
            ),
            dtype=np.float32,
        )

        self.agent_mask = np.zeros(
            (
                self.capacity,
                self.seq_len,
                self.num_agents,
            ),
            dtype=np.float32,
        )

        self.route_mask = np.zeros(
            (
                self.capacity,
                self.seq_len,
                self.num_agents,
                self.max_routes,
            ),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (
                self.capacity,
                self.seq_len,
                2,
            ),
            dtype=np.float32,
        )

    def add(self, sequence: Dict[str, np.ndarray]) -> None:
        idx = self.ptr

        self.motion[idx] = sequence["motion"]
        self.waypoints[idx] = sequence["waypoints"]
        self.agent_mask[idx] = sequence["agent_mask"]
        self.route_mask[idx] = sequence["route_mask"]
        self.actions[idx] = sequence["actions"]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self) -> bool:
        return self.size >= self.batch_size

    def sample(self) -> Dict[str, np.ndarray]:
        if not self.can_sample():
            raise RuntimeError(
                f"Not enough sequences: size={self.size}, batch_size={self.batch_size}"
            )

        idxs = np.random.randint(0, self.size, size=self.batch_size)

        return {
            "motion": self.motion[idxs],
            "waypoints": self.waypoints[idxs],
            "agent_mask": self.agent_mask[idxs],
            "route_mask": self.route_mask[idxs],
            "actions": self.actions[idxs],
        }

    def __len__(self) -> int:
        return self.size
