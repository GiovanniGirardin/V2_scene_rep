from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """
    SAC actor.

    Input:
        latent: [B, latent_dim]

    Output:
        action: [B, 2] in [-1, 1]
            action[:, 0] = normalized target speed
            action[:, 1] = lane-change signal
    """

    def __init__(self, latent_dim: int, action_dim: int = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(latent)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(latent)
        std = log_std.exp()

        dist = Normal(mean, std)

        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)

        # Tanh correction.
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, latent: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(latent)
        return torch.tanh(mean)