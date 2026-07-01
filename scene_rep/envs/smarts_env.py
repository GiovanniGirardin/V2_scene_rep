from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from scene_rep.envs.action_adapter import ActionAdapter
from scene_rep.envs.observation_adapter import ObservationAdapter

import gymnasium as gym

from smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles, Waypoints
from smarts.core.controllers import ActionSpaceType


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
        self._neighbor_interface = None
        self._neighbor_sensor_vehicle_ids = set()
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
            info["timeout"] = True

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
            "timeout": False,
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

        # Suppress overly verbose SMARTS vehicle ID length warnings.
        warnings.filterwarnings(
            "ignore",
            message=r"`vehicle id` named `.*` is more than `50` characters long.*",
            category=UserWarning,
            module=r"smarts\.env\.utils\.observation_conversion",
        )

        self._neighbor_interface = AgentInterface(
            max_episode_steps=None,
            action=ActionSpaceType.Lane,
            waypoint_paths=Waypoints(int(self.config["observation"].get("waypoint_len", 10)) + 1),
        )

        self._smarts_env = gym.make(
            "smarts.env:hiway-v1",
            scenarios=[scenario],
            agent_interfaces=agent_interfaces,
            headless=bool(self.smarts_cfg.get("headless", True)),
            seed=int(self.config["project"]["seed"]),
        )


    def _collect_neighbor_observations(self, agent_obs):
        if agent_obs is None or self._neighbor_interface is None:
            return {}

        if not bool(self.config.get("observation", {}).get("use_neighbor_sensors", True)):
            return {}

        neighbor_ids = [
            str(vehicle.id)
            for vehicle in getattr(agent_obs, "neighborhood_vehicle_states", [])
        ]
        if not neighbor_ids:
            return {}

        smarts = getattr(getattr(self._smarts_env, "unwrapped", self._smarts_env), "_smarts", None)
        if smarts is None:
            return {}

        try:
            new_ids = [
                vehicle_id
                for vehicle_id in neighbor_ids
                if vehicle_id not in self._neighbor_sensor_vehicle_ids
            ]
            if new_ids:
                smarts.attach_sensors_to_vehicles(
                    self._neighbor_interface,
                    new_ids,
                    overwrite_sensors=False,
                    reset_sensors=False,
                )
                self._neighbor_sensor_vehicle_ids.update(new_ids)

            observations, _ = smarts.observe_from(neighbor_ids, self._neighbor_interface)
            return observations
        except Exception as exc:
            warnings.warn(
                f"Failed to collect SMARTS neighbor waypoint observations: {exc}",
                RuntimeWarning,
            )
            return {}

    def _reset_smarts(self):
        self._neighbor_sensor_vehicle_ids.clear()
        obs, info = self._smarts_env.reset()

        agent_info = info.get(self.agent_id, {})
        env_obs = agent_info.get("env_obs", obs.get(self.agent_id))

        if env_obs is not None:
            ego_pos = env_obs.ego_vehicle_state.position
            self.prev_ego_xy = np.array(
                [float(ego_pos[0]), float(ego_pos[1])],
                dtype=np.float32,
            )
            if hasattr(env_obs, "distance_travelled"):
                self.prev_distance_travelled = float(env_obs.distance_travelled)

        return {
            "obs": env_obs,
            "neighbor_obs": self._collect_neighbor_observations(env_obs),
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
            position_delta = 0.0
        else:
            position_delta = float(np.linalg.norm(ego_xy - self.prev_ego_xy))

        self.prev_ego_xy = ego_xy

        agent_terminated = bool(terminated[self.agent_id])
        agent_truncated = bool(truncated[self.agent_id])
        done = agent_terminated or agent_truncated

        events = getattr(agent_obs, "events", None)

        distance_travelled = 0.0

        if agent_obs is not None and hasattr(agent_obs, "distance_travelled"):
            distance_travelled = float(agent_obs.distance_travelled)

        progress = max(0.0, distance_travelled - self.prev_distance_travelled)
        self.prev_distance_travelled = distance_travelled

        success = bool(events.reached_goal) if events is not None else False

        collision = bool(events.collisions) if events is not None else False
        off_route = bool(events.off_route) if events is not None else False
        off_road = bool(events.off_road) if events is not None else False
        alive_done = bool(events.agents_alive_done) if events is not None else False
        max_episode = bool(events.reached_max_episode_steps) if events is not None else False
        stagnation = bool(events.not_moving) if events is not None else False
        timeout = bool(agent_truncated and not agent_terminated) or max_episode
        terminal_reason = "running"
        if success:
            terminal_reason = "success"
        elif collision:
            terminal_reason = "collision"
        elif off_route or off_road:
            terminal_reason = "off_route"
        elif alive_done:
            terminal_reason = "agents_alive_done"
        elif stagnation:
            terminal_reason = "stagnation"
        elif timeout:
            terminal_reason = "timeout"
        elif done:
            terminal_reason = "terminated"

        out_info = {
            "success": success,
            "collision": collision,
            "off_route": off_route or off_road,
            "agents_alive_done": alive_done,
            "stagnation": stagnation,
            "timeout": timeout,
            "terminal_reason": terminal_reason,
            "progress": progress,
            "position_delta": position_delta,
            "distance_travelled": distance_travelled,
            "raw_info": agent_info,
        }

        
        raw_reward = float(reward[self.agent_id])

        reward_mode = str(self.reward_cfg.get("mode", "progress"))
        success_reward = float(self.reward_cfg.get("success_reward", 1.0))
        collision_penalty = float(self.reward_cfg.get("collision_penalty", -1.0))
        off_route_penalty = float(self.reward_cfg.get("off_route_penalty", -1.0))
        step_penalty = float(self.reward_cfg.get("step_penalty", 0.0))
        progress_reward_scale = float(self.reward_cfg.get("progress_reward_scale", 0.05))
        position_delta_reward_scale = float(
            self.reward_cfg.get("position_delta_reward_scale", 0.0)
        )
        no_progress_penalty = float(self.reward_cfg.get("no_progress_penalty", 0.0))
        no_progress_threshold = float(self.reward_cfg.get("no_progress_threshold", 0.01))

        if reward_mode == "sparse":
            agent_reward = step_penalty
            if success:
                agent_reward += success_reward
            if collision:
                agent_reward += collision_penalty
            if off_route or off_road:
                agent_reward += off_route_penalty

        elif reward_mode == "progress":
            agent_reward = step_penalty
            agent_reward += progress_reward_scale * progress
            agent_reward += position_delta_reward_scale * position_delta
            if position_delta <= no_progress_threshold:
                agent_reward += no_progress_penalty

            if success:
                agent_reward += success_reward
            if collision:
                agent_reward += collision_penalty
            if off_route or off_road:
                agent_reward += off_route_penalty

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