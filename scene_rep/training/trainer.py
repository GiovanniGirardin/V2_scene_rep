from __future__ import annotations

from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from scene_rep.data.future_queue import FutureQueue
from scene_rep.data.replay_buffer import ReplayBuffer
from scene_rep.data.sequence_buffer import SequenceBuffer
from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.models.sac import SACAgent
from scene_rep.training.checkpointing import save_checkpoint
from scene_rep.utils.seed import set_seed
from scene_rep.utils.torch_utils import (
    action_to_numpy,
    batch_to_torch,
    get_device,
    obs_to_torch,
    sequence_batch_to_torch,
)
from scene_rep.training.logger import EpisodeLogger

class Trainer:
    """
    Training loop for:
        MST encoder + SAC
        optional SLT auxiliary representation learning
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.project_cfg = config["project"]
        self.training_cfg = config["training"]
        self.sac_cfg = config["sac"]
        self.slt_cfg = config["slt"]

        set_seed(int(self.project_cfg["seed"]))

        self.device = get_device(self.project_cfg["device"])

        self.env = SMARTSSceneRepEnv(config)

        self.buffer = ReplayBuffer(config)

        self.slt_enabled = bool(self.slt_cfg.get("enabled", True))

        if self.slt_enabled:
            self.future_queue = FutureQueue(config)
            self.sequence_buffer = SequenceBuffer(config)
        else:
            self.future_queue = None
            self.sequence_buffer = None

        self.agent = SACAgent(config).to(self.device)

        self.total_steps = int(self.training_cfg["total_steps"])
        self.warmup_steps = int(self.sac_cfg["warmup_steps"])
        self.log_every_steps = int(self.training_cfg["log_every_steps"])
        self.save_every_steps = int(self.training_cfg["save_every_steps"])
        self.checkpoint_dir = str(self.training_cfg["checkpoint_dir"])

        
        self.slt_updates_per_step = int(self.slt_cfg.get("updates_per_step", 1))

    def train(self) -> None:
        obs = self.env.reset()
        if self.slt_enabled:
            self.future_queue.reset()
        
        episode_return = 0.0
        episode_length = 0
        last_episode_info = {}

        last_metrics: Dict[str, float] = {}
        last_slt_metrics: Dict[str, float] = {}

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
            # Store current state-action for SLT sequence construction
            # --------------------------------------------------------
            if self.slt_enabled:
                self.future_queue.add(obs=obs, action=action)

                if self.future_queue.is_ready():
                    sequence = self.future_queue.get_sequence()
                    self.sequence_buffer.add(sequence)

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
            episode_return += float(reward)
            episode_length += 1
            last_episode_info = info
            if done:
                self.logger.log_episode(
                    global_step=step,
                    episode_return=episode_return,
                    episode_length=episode_length,
                    info=last_episode_info,
                )
                episode_return = 0.0
                episode_length = 0
                last_episode_info = {}
                obs = self.env.reset()
                if self.slt_enabled:
                    self.future_queue.reset()

            # --------------------------------------------------------
            # SAC update
            # --------------------------------------------------------
            if step >= self.warmup_steps and self.buffer.can_sample():
                for _ in range(int(self.sac_cfg["updates_per_step"])):
                    batch_np = self.buffer.sample()
                    batch = batch_to_torch(batch_np, device=self.device)
                    last_metrics = self.agent.update(batch)

            # --------------------------------------------------------
            # SLT auxiliary update
            # --------------------------------------------------------
            if (
                self.slt_enabled
                and step >= self.warmup_steps
                and self.sequence_buffer is not None and self.sequence_buffer.can_sample()
            ):
                for _ in range(self.slt_updates_per_step):
                    seq_np = self.sequence_buffer.sample()
                    seq_batch = sequence_batch_to_torch(seq_np, device=self.device)
                    last_slt_metrics = self.agent.update_slt(seq_batch)

            # --------------------------------------------------------
            # Logging
            # --------------------------------------------------------
            if step % self.log_every_steps == 0:
                progress.set_postfix(
                    {
                        "buffer": len(self.buffer),
                        "seq": len(self.sequence_buffer) if self.sequence_buffer is not None else 0,
                        "reward": reward,
                        "alpha": round(last_metrics.get("alpha", 0.0), 4),
                        "critic": round(last_metrics.get("critic_loss", 0.0), 4),
                        "actor": round(last_metrics.get("actor_loss", 0.0), 4),
                        "slt": round(last_slt_metrics.get("slt_loss", 0.0), 4),
                    }
                )

            # --------------------------------------------------------
            # Checkpointing
            # --------------------------------------------------------
            if step % self.save_every_steps == 0:
                path = save_checkpoint(
                    agent=self.agent,
                    step=step,
                    checkpoint_dir=self.checkpoint_dir,
                    extra={
                        "last_metrics": last_metrics,
                        "last_slt_metrics": last_slt_metrics,
                    },
                )
                print(f"\nSaved checkpoint: {path}")