from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from scene_rep.models.transformer_blocks import (
    TransformerEncoderBlock,
    CrossAttentionBlock,
)


class MSTEncoder(nn.Module):
    """
    Multi-Stage Transformer encoder.

    Input:
        obs["motion"]:
            [B, A, H, 5]

        obs["waypoints"]:
            [B, A, R, W, 3]

        obs["agent_mask"]:
            [B, A]

        obs["route_mask"]:
            [B, A, R]

    Output:
        latent:
            [B, latent_dim]
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_cfg = config["model"]
        obs_cfg = config["observation"]

        self.motion_dim = model_cfg["motion_dim"]
        self.waypoint_dim = model_cfg["waypoint_dim"]

        self.d_model = model_cfg["d_model"]
        self.latent_dim = model_cfg["latent_dim"]
        self.nhead = model_cfg["nhead"]
        self.dropout = model_cfg["dropout"]

        self.num_agents = obs_cfg["max_neighbors"] + 1
        self.max_routes = obs_cfg["max_candidate_routes"]
        self.waypoint_len = obs_cfg["waypoint_len"]
        self.vehicle_embedding_dim = int(model_cfg.get("vehicle_embedding_dim", 64))

        # ------------------------------------------------------------
        # Stage 1: dynamic level
        # ------------------------------------------------------------
        self.motion_input = nn.Linear(self.motion_dim, self.d_model)
        self.waypoint_input = nn.Linear(self.waypoint_dim, self.d_model)

        self.motion_encoder = TransformerEncoderBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        self.waypoint_encoder = TransformerEncoderBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        # Fig. 3 of the paper uses MHA -> global max pooling -> MLP for
        # motion, and MHA -> flatten -> vehicle-type embedding -> MLP for a
        # candidate route.  The embedding differentiates ego routes from
        # neighbour routes before they are used by different cross-attention
        # stages.
        self.motion_post = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )
        self.vehicle_embedding = nn.Embedding(2, self.vehicle_embedding_dim)
        self.waypoint_post = nn.Sequential(
            nn.Linear(
                self.waypoint_len * self.d_model + self.vehicle_embedding_dim,
                self.d_model,
            ),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )

        # ------------------------------------------------------------
        # Stage 2: cross-modality level
        # agent motion attends to its own candidate routes
        # ------------------------------------------------------------
        self.agent_route_cross = CrossAttentionBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        # ------------------------------------------------------------
        # Stage 3: aggregation level
        # ego attends to all agents
        # ------------------------------------------------------------
        self.ego_agent_cross = CrossAttentionBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        # ------------------------------------------------------------
        # Stage 4: output level
        # aggregated ego attends to ego candidate routes
        # ------------------------------------------------------------
        self.ego_route_cross = CrossAttentionBlock(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
        )

        # Eq. (10) already produces h_t.  Keep an explicit projection only
        # when a caller requests a latent dimension different from d_model.
        self.output = (
            nn.Identity()
            if self.d_model == self.latent_dim
            else nn.Linear(self.d_model, self.latent_dim)
        )

    @staticmethod
    def _padding_mask(
        values: torch.Tensor,
        explicit_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a bool key-padding mask (True means padding)."""
        if explicit_mask is not None:
            return explicit_mask < 0.5
        # All adapters in this repository encode unavailable temporal tokens
        # as all-zero vectors.  Accepting an explicit mask above avoids that
        # convention being required by other data sources.
        return values.abs().sum(dim=-1) == 0

    @staticmethod
    def _masked_max(x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Max-pool tokens while making an entirely padded sequence zero."""
        masked = x.masked_fill(padding_mask.unsqueeze(-1), float("-inf"))
        pooled = masked.max(dim=1).values
        return torch.where(
            padding_mask.all(dim=1, keepdim=True),
            torch.zeros_like(pooled),
            pooled,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        motion = obs["motion"]
        waypoints = obs["waypoints"]
        agent_mask = obs["agent_mask"]
        route_mask = obs["route_mask"]

        batch_size = motion.shape[0]

        # ------------------------------------------------------------
        # Stage 1A: encode historical motion
        # motion: [B, A, H, 5]
        # ------------------------------------------------------------
        b, a, h, _ = motion.shape

        motion_padding_mask = self._padding_mask(
            motion,
            obs.get("motion_mask"),
        )
        motion_x = self.motion_input(motion)
        motion_x = motion_x.reshape(b * a, h, self.d_model)
        motion_padding_mask = motion_padding_mask.reshape(b * a, h)

        motion_encoded = self.motion_encoder(
            motion_x,
            key_padding_mask=motion_padding_mask,
        )

        # max pool over history
        motion_emb = self.motion_post(
            self._masked_max(motion_encoded, motion_padding_mask)
        )
        motion_emb = motion_emb.reshape(b, a, self.d_model)

        # ------------------------------------------------------------
        # Stage 1B: encode candidate route waypoints
        # waypoints: [B, A, R, W, 3]
        # ------------------------------------------------------------
        b, a, r, w, _ = waypoints.shape

        waypoint_padding_mask = self._padding_mask(
            waypoints,
            obs.get("waypoint_mask"),
        )
        wp_x = self.waypoint_input(waypoints)
        wp_x = wp_x.reshape(b * a * r, w, self.d_model)
        waypoint_padding_mask = waypoint_padding_mask.reshape(b * a * r, w)

        wp_encoded = self.waypoint_encoder(
            wp_x,
            key_padding_mask=waypoint_padding_mask,
        )

        # Invalid waypoint queries are excluded before flattening as well as
        # invalid keys being excluded from attention.
        wp_encoded = wp_encoded.masked_fill(
            waypoint_padding_mask.unsqueeze(-1), 0.0
        )
        route_flat = wp_encoded.reshape(b, a, r, w * self.d_model)
        vehicle_type = torch.ones((a,), device=waypoints.device, dtype=torch.long)
        vehicle_type[0] = 0  # 0=ego, 1=neighbour
        vehicle_emb = self.vehicle_embedding(vehicle_type)[None, :, None, :]
        vehicle_emb = vehicle_emb.expand(b, a, r, -1)
        route_emb = self.waypoint_post(torch.cat([route_flat, vehicle_emb], dim=-1))
        route_emb = route_emb.reshape(b, a, r, self.d_model)

        # ------------------------------------------------------------
        # Stage 2: each agent motion attends to its own routes
        # query:   [B*A, 1, D]
        # context: [B*A, R, D]
        # ------------------------------------------------------------
        agent_query = motion_emb.reshape(b * a, 1, self.d_model)
        route_context = route_emb.reshape(b * a, r, self.d_model)

        route_padding_mask = route_mask.reshape(b * a, r) < 0.5

        agent_route_emb = self.agent_route_cross(
            query=agent_query,
            context=route_context,
            context_key_padding_mask=route_padding_mask,
        )

        agent_route_emb = agent_route_emb.squeeze(1)
        agent_route_emb = agent_route_emb.reshape(b, a, self.d_model)

        # residual connection
        agent_emb = agent_route_emb + motion_emb

        # ------------------------------------------------------------
        # Stage 3: ego attends to all agents
        # ego query:     [B, 1, D] = ego motion only (Eq. 9)
        # agent context: [ego motion, neighbour cross-modal embeddings]
        # ------------------------------------------------------------
        ego_query = motion_emb[:, 0:1, :]
        aggregation_context = torch.cat(
            [motion_emb[:, 0:1, :], agent_emb[:, 1:, :]], dim=1
        )

        agent_padding_mask = agent_mask < 0.5

        aggregated_ego = self.ego_agent_cross(
            query=ego_query,
            context=aggregation_context,
            context_key_padding_mask=agent_padding_mask,
        )

        # ------------------------------------------------------------
        # Stage 4: aggregated ego attends to ego routes
        # ------------------------------------------------------------
        ego_routes = route_emb[:, 0, :, :]
        ego_route_mask = route_mask[:, 0, :] < 0.5

        ego_final = self.ego_route_cross(
            query=aggregated_ego,
            context=ego_routes,
            context_key_padding_mask=ego_route_mask,
        )

        # Eq. (10): preserve the interaction-aware aggregation through the
        # final residual connection.
        ego_final = ego_final + aggregated_ego
        ego_final = ego_final.squeeze(1)

        latent = self.output(ego_final)

        return latent
