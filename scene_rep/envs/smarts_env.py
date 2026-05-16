from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from scene_rep.envs.action_adapter import ActionAdapter
from scene_rep.envs.observation_adapter import ObservationAdapter


class SMARTSSceneRepEnv:
    """
    SMARTS wrapper.

    Dummy mode now implements a simple 1D driving task:
        - ego moves forward along x
        - success if ego reaches goal_x
        - collision if ego reaches obstacle too aggressively
        - off-route if too many lane-change commands are accumulated
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.smarts_cfg = config["smarts"]
        self.reward_cfg = config["reward"]

        self.use_dummy = bool(self.smarts_cfg.get("use_dummy", True))

        self.action_adapter = ActionAdapter(config)
        self.observation_adapter = ObservationAdapter(config)

        self.step_count = 0
        self.max_episode_steps = int(self.smarts_cfg["max_episode_steps"])

        self.dt = float(self.smarts_cfg.get("dt", 0.1))
        self.goal_x = float(self.smarts_cfg.get("dummy_goal_x", 30.0))
        self.obstacle_x = float(self.smarts_cfg.get("dummy_obstacle_x", 15.0))
        self.collision_speed = float(self.smarts_cfg.get("dummy_collision_speed", 8.0))

        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_v = 0.0
        self.ego_heading = 0.0
        self.lane_index = 0

        self._smarts_env = None

        if not self.use_dummy:
            self._init_smarts()

    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0

        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_v = 0.0
        self.ego_heading = 0.0
        self.lane_index = 0

        self.observation_adapter.reset()

        raw_obs = self._make_dummy_raw_obs() if self.use_dummy else self._reset_smarts()
        return self.observation_adapter.adapt(raw_obs)

    def step(
        self,
        action: Tuple[float, float],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        self.step_count += 1

        smarts_action = self.action_adapter.adapt(action)

        if self.use_dummy:
            raw_obs, reward, done, info = self._dummy_step(smarts_action)
        else:
            raw_obs, reward, done, info = self._step_smarts(smarts_action)

        observation = self.observation_adapter.adapt(raw_obs)

        if self.step_count >= self.max_episode_steps and not done:
            done = True
            info["stagnation"] = True
            reward -= 1.0

        info["smarts_action"] = smarts_action
        info["step_count"] = self.step_count

        return observation, reward, done, info

    def close(self) -> None:
        if self._smarts_env is not None:
            self._smarts_env.close()

    def _dummy_step(self, smarts_action: Dict[str, Any]):
        target_speed = float(smarts_action["speed"])
        lane_change = int(smarts_action["lane_change"])

        self.ego_v = target_speed
        self.ego_x += self.ego_v * self.dt

        if lane_change != 0:
            self.lane_index += lane_change
            self.ego_y = float(self.lane_index) * 3.5

        success = self.ego_x >= self.goal_x

        near_obstacle = abs(self.ego_x - self.obstacle_x) < 1.0
        collision = near_obstacle and self.ego_v > self.collision_speed

        off_route = abs(self.lane_index) > 1

        done = success or collision or off_route

        reward = 0.0
        reward += 0.01 * self.ego_v

        if success:
            reward += 1.0
        if collision:
            reward -= 1.0
        if off_route:
            reward -= 1.0

        info = {
            "success": success,
            "collision": collision,
            "off_route": off_route,
            "stagnation": False,
            "ego_x": self.ego_x,
            "ego_y": self.ego_y,
            "ego_v": self.ego_v,
            "lane_index": self.lane_index,
        }

        return self._make_dummy_raw_obs(), reward, done, info

    def _make_dummy_raw_obs(self) -> Dict[str, Any]:
        return {
            "ego": {
                "x": self.ego_x,
                "y": self.ego_y,
                "vx": self.ego_v,
                "vy": 0.0,
                "heading": self.ego_heading,
            },
            "goal_x": self.goal_x,
            "obstacle_x": self.obstacle_x,
        }

    def _init_smarts(self) -> None:
        try:
            import smarts  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SMARTS is not installed or not available. "
                "Keep `smarts.use_dummy: true` for now."
            ) from exc

        raise NotImplementedError(
            "Real SMARTS initialization is not implemented yet."
        )

    def _reset_smarts(self):
        raise NotImplementedError

    def _step_smarts(self, smarts_action: Dict[str, Any]):
        raise NotImplementedError