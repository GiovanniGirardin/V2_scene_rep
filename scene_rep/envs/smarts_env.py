from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from scene_rep.envs.action_adapter import ActionAdapter
from scene_rep.envs.observation_adapter import ObservationAdapter

import gymnasium as gym

from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles


class SMARTSSceneRepEnv:
    """
    SMARTS wrapper for the Scene-Rep-Transformer project.

    It supports two modes:

    1. Dummy mode:
        use_dummy: true

        Used while developing the learning pipeline.

    2. SMARTS mode:
        use_dummy: false

        Later this connects to the real SMARTS simulator.
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

        self._smarts_env = None
        self.prev_distance_travelled = 0.0
        self.prev_ego_xy = None

        if not self.use_dummy:
            self._init_smarts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.prev_distance_travelled = 0.0
        self.observation_adapter.reset()
        self.prev_ego_xy = None

        if self.use_dummy:
            raw_obs = None
        else:
            raw_obs = self._reset_smarts()

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

        #print(
        #    "[obs debug]",
         #   "motion", observation["motion"].shape,
          #  "waypoints", observation["waypoints"].shape,
           # "agent_mask", observation["agent_mask"],
            #"route_mask", observation["route_mask"][0],
        #)

        if self.step_count >= self.max_episode_steps:
            done = True
            info["stagnation"] = True

        info["smarts_action"] = smarts_action
        info["step_count"] = self.step_count

        return observation, reward, done, info

    def close(self) -> None:
        if self._smarts_env is not None:
            self._smarts_env.close()

    # ------------------------------------------------------------------
    # Dummy mode
    # ------------------------------------------------------------------
    def _dummy_step(self, smarts_action: Dict[str, Any]):
        reward = 0.0
        done = False

        info = {
            "success": False,
            "collision": False,
            "off_route": False,
            "stagnation": False,
        }

        return None, reward, done, info

    # ------------------------------------------------------------------
    # SMARTS mode placeholders
    # ------------------------------------------------------------------
    def _init_smarts(self) -> None:
        scenario = self.smarts_cfg["scenario"]
        self.agent_id = self.smarts_cfg.get("agent_id", "Agent-007")

        agent_type_name = self.smarts_cfg.get("agent_type", "Laner")
        agent_type = getattr(AgentType, agent_type_name)

        agent_interfaces = {
            self.agent_id: AgentInterface.from_type(
            agent_type,
            max_episode_steps=self.max_episode_steps,
            neighborhood_vehicle_states=NeighborhoodVehicles(radius=50),
        )
                }

        self._smarts_env = gym.make(
            "smarts.env:hiway-v1",
            scenarios=[scenario],
            agent_interfaces=agent_interfaces,
            headless=bool(self.smarts_cfg.get("headless", True)),
            seed=int(self.config["project"]["seed"]),
        )

    def _reset_smarts(self):
        obs, info = self._smarts_env.reset()

        agent_info = info.get(self.agent_id, {})
        env_obs = agent_info.get("env_obs", obs.get(self.agent_id))

        return {
            "obs": env_obs,
            "info": agent_info,
        }

    def _step_smarts(self, smarts_action: Dict[str, Any]):
        speed = float(smarts_action["speed"])
        lane_change = int(smarts_action["lane_change"])

        # SMARTS LaneWithContinuousSpeed action:
        # usually [speed, lane_change]
        # lane_change: -1, 0, 1
        # speed: m/s
        action_value = (
            np.float32(speed),
            np.int8(lane_change),
        )

        action = {self.agent_id: action_value}

        obs, reward, terminated, truncated, info = self._smarts_env.step(action)

        agent_info = info.get(self.agent_id, {})
        agent_obs = agent_info.get("env_obs", obs.get(self.agent_id))

        ego_state = agent_obs.ego_vehicle_state
        ego_pos = ego_state.position
        ego_xy = np.array([float(ego_pos[0]), float(ego_pos[1])], dtype=np.float32)

        if self.prev_ego_xy is None:
            progress = 0.0
        else:
            progress = float(np.linalg.norm(ego_xy - self.prev_ego_xy))

        self.prev_ego_xy = ego_xy

        agent_terminated = bool(terminated[self.agent_id])
        agent_truncated = bool(truncated[self.agent_id])
        done = agent_terminated or agent_truncated

        events = getattr(agent_obs, "events", None)

        distance_travelled = 0.0
        
        if agent_obs is not None and hasattr(agent_obs, "distance_travelled"):
            distance_travelled = float(agent_obs.distance_travelled)
        
        progress = distance_travelled - self.prev_distance_travelled
        self.prev_distance_travelled = distance_travelled

        success = bool(events.reached_goal) if events is not None else False

        collision = bool(events.collisions) if events is not None else False
        off_route = bool(events.off_route) if events is not None else False
        off_road = bool(events.off_road) if events is not None else False
        stagnation = bool(events.not_moving) if events is not None else False

        

        out_info = {
            "success": success,
            "collision": collision,
            "off_route": off_route or off_road,
            "stagnation": stagnation,
            "raw_info": agent_info,
        }

        
        raw_reward = float(reward[self.agent_id])

        reward_mode = str(self.reward_cfg.get("mode", "progress"))

        if reward_mode == "sparse":
            agent_reward = 0.0
            if success:
                agent_reward += 1.0
            if collision or off_route or off_road:
                agent_reward -= 1.0

        elif reward_mode == "progress":
            agent_reward = 0.0
            agent_reward += 0.05 * progress

            if success:
                agent_reward += 1.0
            if collision or off_route or off_road:
                agent_reward -= 1.0

        else:
            agent_reward = raw_reward

            
        #print(
        #f"[debug] dist={distance_travelled:.2f}, "
        #f"success={success}, done={done}, reward={agent_reward}"
        #)


        return {
            "obs": agent_obs,
            "info": agent_info,
        }, agent_reward, done, out_info