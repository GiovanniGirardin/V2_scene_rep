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

        self.output = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
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

        motion_x = self.motion_input(motion)
        motion_x = motion_x.reshape(b * a, h, self.d_model)

        motion_encoded = self.motion_encoder(motion_x)

        # max pool over history
        motion_emb = motion_encoded.max(dim=1).values
        motion_emb = motion_emb.reshape(b, a, self.d_model)

        # ------------------------------------------------------------
        # Stage 1B: encode candidate route waypoints
        # waypoints: [B, A, R, W, 3]
        # ------------------------------------------------------------
        b, a, r, w, _ = waypoints.shape

        wp_x = self.waypoint_input(waypoints)
        wp_x = wp_x.reshape(b * a * r, w, self.d_model)

        wp_encoded = self.waypoint_encoder(wp_x)

        # max pool over waypoint sequence
        route_emb = wp_encoded.max(dim=1).values
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
        # ego query:     [B, 1, D]
        # agent context: [B, A, D]
        # ------------------------------------------------------------
        ego_query = agent_emb[:, 0:1, :]

        agent_padding_mask = agent_mask < 0.5

        aggregated_ego = self.ego_agent_cross(
            query=ego_query,
            context=agent_emb,
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

        ego_final = ego_final.squeeze(1)

        latent = self.output(ego_final)

        return latent