from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

from scene_rep.models.actor import Actor
from scene_rep.models.critic import Critic
from scene_rep.models.mst_encoder import MSTEncoder
from scene_rep.models.slt import SequentialLatentTransformer
from scene_rep.training.augmentation import augment_observation


class SACAgent(nn.Module):
    """
    SAC agent with MST scene encoder.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.sac_cfg = config["sac"]

        self.gamma = float(self.sac_cfg["gamma"])
        self.tau = float(self.sac_cfg["tau"])

        latent_dim = int(config["model"]["latent_dim"])
        action_dim = 2
        self.slt_cfg = config["slt"]
        self.slt_enabled = bool(self.slt_cfg.get("enabled", True))

        self.encoder = MSTEncoder(config)
        self.actor = Actor(latent_dim=latent_dim, action_dim=action_dim)
        self.critic1 = Critic(latent_dim=latent_dim, action_dim=action_dim)
        self.critic2 = Critic(latent_dim=latent_dim, action_dim=action_dim)
        self.slt = SequentialLatentTransformer(
            latent_dim=latent_dim,
            action_dim=action_dim,
            future_horizon=int(self.slt_cfg["future_horizon"]),
            nhead=int(config["model"]["nhead"]),
            dropout=float(config["model"]["dropout"]),
            projector_dim=int(self.slt_cfg["projector_dim"]),
            predictor_dim=int(self.slt_cfg["predictor_dim"]),
        )

        self.target_encoder = deepcopy(self.encoder)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)

        self._freeze_targets()

        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.target_entropy = float(self.sac_cfg["target_entropy"])

        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=float(self.sac_cfg["critic_lr"]),
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(self.sac_cfg["actor_lr"]),
        )

        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=float(self.sac_cfg["critic_lr"]),
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=float(self.sac_cfg["alpha_lr"]),
        )

        self.slt_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.slt.parameters()),
            lr=float(self.slt_cfg["lr"]),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> torch.Tensor:
        latent = self.encoder(obs)

        if deterministic:
            action = self.actor.deterministic(latent)
        else:
            action, _ = self.actor.sample(latent)

        return action

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]

        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        latent = self.encoder(obs)

        q1 = self.critic1(latent, actions)
        q2 = self.critic2(latent, actions)

        with torch.no_grad():
            next_latent = self.target_encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_latent)

            target_q1 = self.target_critic1(next_latent, next_action)
            target_q2 = self.target_critic2(next_latent, next_action)

            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha.detach() * next_log_prob

            y = rewards + self.gamma * (1.0 - dones) * target_q

        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.encoder_optimizer.step()
        self.critic_optimizer.step()

        with torch.no_grad():
            latent_detached = self.encoder(obs)

        new_action, log_prob = self.actor.sample(latent_detached)

        q1_pi = self.critic1(latent_detached, new_action)
        q2_pi = self.critic2(latent_detached, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets()

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
        }
    
    def update_slt(self, sequence_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
            """
            Update MST encoder + SLT using a batch of future sequences.

            sequence_batch shapes:
                motion:     [B, T, A, H, Dm]
                waypoints:  [B, T, A, R, W, Dw]
                agent_mask: [B, T, A]
                route_mask: [B, T, A, R]
                actions:    [B, T, 2]
            """
            if not self.slt_enabled:
                return {"slt_loss": 0.0}

            motion = sequence_batch["motion"]
            waypoints = sequence_batch["waypoints"]
            agent_mask = sequence_batch["agent_mask"]
            route_mask = sequence_batch["route_mask"]
            actions = sequence_batch["actions"]

            batch_size, seq_len = actions.shape[:2]

            # Flatten time into batch so MST can encode each state.
            obs = {
                "motion": motion.reshape(batch_size * seq_len, *motion.shape[2:]),
                "waypoints": waypoints.reshape(batch_size * seq_len, *waypoints.shape[2:]),
                "agent_mask": agent_mask.reshape(batch_size * seq_len, *agent_mask.shape[2:]),
                "route_mask": route_mask.reshape(batch_size * seq_len, *route_mask.shape[2:]),
            }

            obs = augment_observation(obs)

            latents = self.encoder(obs)
            latents = latents.reshape(batch_size, seq_len, -1)

            loss, metrics = self.slt.compute_loss(
                latents=latents,
                actions=actions,
            )

            loss = float(self.slt_cfg.get("loss_weight", 1.0)) * loss

            self.slt_optimizer.zero_grad()
            loss.backward()
            self.slt_optimizer.step()

            return metrics

    def soft_update_targets(self) -> None:
        self._soft_update(self.encoder, self.target_encoder)
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)

    def _freeze_targets(self) -> None:
        for net in [
            self.target_encoder,
            self.target_critic1,
            self.target_critic2,
        ]:
            for p in net.parameters():
                p.requires_grad = False