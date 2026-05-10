from __future__ import annotations

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from scene_rep.data.replay_buffer import ReplayBuffer
from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.models.sac import SACAgent
from scene_rep.utils.torch_utils import (
    action_to_numpy,
    batch_to_torch,
    get_device,
    obs_to_torch,
)


class Trainer:
    """
    First training loop skeleton.

    For now it trains on the dummy SMARTS wrapper.
    Later we will connect the wrapper to the real SMARTS simulator.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.project_cfg = config["project"]
        self.training_cfg = config["training"]
        self.sac_cfg = config["sac"]

        self.device = get_device(self.project_cfg["device"])

        self.env = SMARTSSceneRepEnv(config)
        self.buffer = ReplayBuffer(config)
        self.agent = SACAgent(config).to(self.device)

        self.total_steps = int(self.training_cfg["total_steps"])
        self.warmup_steps = int(self.sac_cfg["warmup_steps"])
        self.log_every_steps = int(self.training_cfg["log_every_steps"])

    def train(self) -> None:
        obs = self.env.reset()

        last_metrics: Dict[str, float] = {}

        progress = tqdm(range(1, self.total_steps + 1), desc="Training")

        for step in progress:
            # --------------------------------------------------------
            # Select action
            # --------------------------------------------------------
            if step < self.warmup_steps:
                action = np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(2,),
                ).astype(np.float32)
            else:
                obs_torch = obs_to_torch(
                    obs,
                    device=self.device,
                    add_batch_dim=True,
                )
                action_torch = self.agent.act(obs_torch, deterministic=False)
                action = action_to_numpy(action_torch)

            # --------------------------------------------------------
            # Environment step
            # --------------------------------------------------------
            next_obs, reward, done, info = self.env.step(tuple(action))

            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs

            if done:
                obs = self.env.reset()

            # --------------------------------------------------------
            # SAC update
            # --------------------------------------------------------
            if step >= self.warmup_steps and self.buffer.can_sample():
                for _ in range(int(self.sac_cfg["updates_per_step"])):
                    batch_np = self.buffer.sample()
                    batch = batch_to_torch(batch_np, device=self.device)
                    last_metrics = self.agent.update(batch)

            # --------------------------------------------------------
            # Logging
            # --------------------------------------------------------
            if step % self.log_every_steps == 0:
                progress.set_postfix(
                    {
                        "buffer": len(self.buffer),
                        "reward": reward,
                        "alpha": round(last_metrics.get("alpha", 0.0), 4),
                        "critic": round(last_metrics.get("critic_loss", 0.0), 4),
                        "actor": round(last_metrics.get("actor_loss", 0.0), 4),
                    }
                )
