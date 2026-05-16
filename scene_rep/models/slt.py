from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class SequentialLatentTransformer(nn.Module):
    """
    Sequential Latent Transformer.

    Training-only auxiliary module.

    Input:
        latents:
            [B, T, latent_dim]

        actions:
            [B, T, action_dim]

    Output:
        predicted future latents:
            [B, T - 1, latent_dim]
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = 2,
        future_horizon: int = 5,
        nhead: int = 4,
        dropout: float = 0.1,
        projector_dim: int = 128,
        predictor_dim: int = 128,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.future_horizon = future_horizon

        self.input_proj = nn.Linear(latent_dim + action_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=nhead,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=1,
        )

        self.predictor_head = nn.Linear(latent_dim, latent_dim)

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, projector_dim),
            nn.ReLU(),
            nn.Linear(projector_dim, projector_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(projector_dim, predictor_dim),
            nn.ReLU(),
            nn.Linear(predictor_dim, projector_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next-step latent sequence.

        latents:
            [B, T, latent_dim]

        actions:
            [B, T, action_dim]
        """
        x = torch.cat([latents, actions], dim=-1)
        x = self.input_proj(x)

        x = self.transformer(x)

        pred = self.predictor_head(x)

        # pred[:, :-1] predicts latents[:, 1:]
        return pred[:, :-1, :]

    def similarity_loss(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        SimSiam-style negative cosine similarity.

        predicted_latents:
            [B, T - 1, latent_dim]

        target_latents:
            [B, T - 1, latent_dim]
        """
        z_target = self.projector(target_latents).detach()
        z_pred = self.predictor(self.projector(predicted_latents))

        z_target = F.normalize(z_target, dim=-1)
        z_pred = F.normalize(z_pred, dim=-1)

        loss = -(z_pred * z_target).sum(dim=-1).mean()
        return loss

    def compute_loss(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Convenience wrapper.

        latents:
            [B, T, latent_dim]

        actions:
            [B, T, action_dim]
        """
        predicted = self.forward(latents, actions)
        target = latents[:, 1:, :]

        loss = self.similarity_loss(predicted, target)

        metrics = {
            "slt_loss": float(loss.detach().cpu()),
        }

        return loss, metrics
