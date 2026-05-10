from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


from scene_rep.envs.action_adapter import ActionAdapter
from scene_rep.envs.observation_adapter import ObservationAdapter


class SMARTSSceneRepEnv:
    """
    Minimal SMARTS wrapper for the Scene-Rep-Transformer project.

    For now this class defines the interface we want:

        reset() -> observation
        step(action) -> observation, reward, done, info

    Later we will connect it to the real SMARTS simulator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.smarts_cfg = config["smarts"]
        self.obs_cfg = config["observation"]
        self.reward_cfg = config["reward"]

        self.action_adapter = ActionAdapter(config)
        self.observation_adapter = ObservationAdapter(config)

        self.step_count = 0
        self.max_episode_steps = self.smarts_cfg["max_episode_steps"]

        self.history_len = self.obs_cfg["history_len"]
        self.max_neighbors = self.obs_cfg["max_neighbors"]
        self.num_agents = self.max_neighbors + 1  # ego + neighbors

        self.max_candidate_routes = self.obs_cfg["max_candidate_routes"]
        self.waypoint_len = self.obs_cfg["waypoint_len"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Returns
        -------
        observation:
            {
                "motion": np.ndarray [num_agents, history_len, 5],
                "waypoints": np.ndarray [num_agents, max_routes, waypoint_len, 3],
                "agent_mask": np.ndarray [num_agents],
                "route_mask": np.ndarray [num_agents, max_routes],
            }
        """
        self.step_count = 0
        self.observation_adapter.reset()
        return self.observation_adapter.adapt(raw_obs=None)

    def step(
        self,
        action: Tuple[float, float],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Step the environment using a policy action.

        Parameters
        ----------
        action:
            Tuple from the policy:
                speed_norm in [-1, 1]
                lane_raw in [-1, 1]

        Returns
        -------
        observation, reward, done, info
        """
        self.step_count += 1

        smarts_action = self.action_adapter.adapt(action)

        observation = self.observation_adapter.adapt(raw_obs=None)
        reward = self._compute_dummy_reward()
        done = self.step_count >= self.max_episode_steps

        info = {
            "smarts_action": smarts_action,
            "step_count": self.step_count,
            "success": False,
            "collision": False,
            "off_route": False,
        }

        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Temporary placeholder logic
    # ------------------------------------------------------------------
    def _dummy_observation(self) -> Dict[str, np.ndarray]:
        """
        Temporary fake observation.

        This lets us develop the model, replay buffer, and trainer before
        fighting with SMARTS API details.
        """
        motion = np.zeros(
            (self.num_agents, self.history_len, 5),
            dtype=np.float32,
        )

        waypoints = np.zeros(
            (
                self.num_agents,
                self.max_candidate_routes,
                self.waypoint_len,
                3,
            ),
            dtype=np.float32,
        )

        agent_mask = np.zeros((self.num_agents,), dtype=np.float32)
        route_mask = np.zeros(
            (self.num_agents, self.max_candidate_routes),
            dtype=np.float32,
        )

        # Ego is always present.
        agent_mask[0] = 1.0

        # Give ego one valid dummy route for now.
        route_mask[0, 0] = 1.0

        return {
            "motion": motion,
            "waypoints": waypoints,
            "agent_mask": agent_mask,
            "route_mask": route_mask,
        }

    def _compute_dummy_reward(self) -> float:
        """
        Temporary reward.

        Later this will use SMARTS events:
            - reached goal
            - collision
            - off road/off route
            - timeout
        """
        return 0.0
