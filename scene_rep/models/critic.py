from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    """
    Single Q-network.

    Input:
        latent: [B, latent_dim]
        action: [B, action_dim]

    Output:
        q_value: [B, 1]
    """

    def __init__(self, latent_dim: int, action_dim: int = 2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, action], dim=-1)
        return self.net(x)