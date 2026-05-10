from __future__ import annotations

from typing import Any, Dict

import numpy as np


class ReplayBuffer:
    """
    Replay buffer for SAC.

    Stores:
        obs
        action
        reward
        next_obs
        done
    """

    def __init__(self, config: Dict[str, Any]):
        obs_cfg = config["observation"]
        model_cfg = config["model"]
        sac_cfg = config["sac"]

        self.capacity = int(sac_cfg["replay_size"])
        self.batch_size = int(sac_cfg["batch_size"])

        self.history_len = obs_cfg["history_len"]
        self.num_agents = obs_cfg["max_neighbors"] + 1
        self.max_candidate_routes = obs_cfg["max_candidate_routes"]
        self.waypoint_len = obs_cfg["waypoint_len"]

        self.motion_dim = model_cfg["motion_dim"]
        self.waypoint_dim = model_cfg["waypoint_dim"]

        self.ptr = 0
        self.size = 0

        self.motion = np.zeros(
            (self.capacity, self.num_agents, self.history_len, self.motion_dim),
            dtype=np.float32,
        )
        self.waypoints = np.zeros(
            (
                self.capacity,
                self.num_agents,
                self.max_candidate_routes,
                self.waypoint_len,
                self.waypoint_dim,
            ),
            dtype=np.float32,
        )
        self.agent_mask = np.zeros(
            (self.capacity, self.num_agents),
            dtype=np.float32,
        )
        self.route_mask = np.zeros(
            (self.capacity, self.num_agents, self.max_candidate_routes),
            dtype=np.float32,
        )

        self.next_motion = np.zeros_like(self.motion)
        self.next_waypoints = np.zeros_like(self.waypoints)
        self.next_agent_mask = np.zeros_like(self.agent_mask)
        self.next_route_mask = np.zeros_like(self.route_mask)

        self.actions = np.zeros((self.capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        idx = self.ptr

        self.motion[idx] = obs["motion"]
        self.waypoints[idx] = obs["waypoints"]
        self.agent_mask[idx] = obs["agent_mask"]
        self.route_mask[idx] = obs["route_mask"]

        self.next_motion[idx] = next_obs["motion"]
        self.next_waypoints[idx] = next_obs["waypoints"]
        self.next_agent_mask[idx] = next_obs["agent_mask"]
        self.next_route_mask[idx] = next_obs["route_mask"]

        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self) -> bool:
        return self.size >= self.batch_size

    def sample(self) -> Dict[str, np.ndarray]:
        if not self.can_sample():
            raise RuntimeError(
                f"Not enough samples: size={self.size}, batch_size={self.batch_size}"
            )

        idxs = np.random.randint(0, self.size, size=self.batch_size)

        return {
            "obs": {
                "motion": self.motion[idxs],
                "waypoints": self.waypoints[idxs],
                "agent_mask": self.agent_mask[idxs],
                "route_mask": self.route_mask[idxs],
            },
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": {
                "motion": self.next_motion[idxs],
                "waypoints": self.next_waypoints[idxs],
                "agent_mask": self.next_agent_mask[idxs],
                "route_mask": self.next_route_mask[idxs],
            },
            "dones": self.dones[idxs],
        }

    def __len__(self) -> int:
        return self.size