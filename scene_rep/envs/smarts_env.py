from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from scene_rep.envs.action_adapter import ActionAdapter
from scene_rep.envs.observation_adapter import ObservationAdapter


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

        if not self.use_dummy:
            self._init_smarts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.observation_adapter.reset()

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
        """
        Initialize real SMARTS environment.

        We keep this isolated so import errors are easy to understand.
        """
        try:
            import smarts  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SMARTS is not installed or not available in this Python environment.\n"
                "For now, keep `smarts.use_dummy: true` in configs/default.yaml.\n"
                "Later, install SMARTS and set `smarts.use_dummy: false`."
            ) from exc

        raise NotImplementedError(
            "SMARTS real environment initialization is not implemented yet. "
            "Keep `smarts.use_dummy: true` for now."
        )

    def _reset_smarts(self):
        raise NotImplementedError

    def _step_smarts(self, smarts_action: Dict[str, Any]):
        raise NotImplementedError