from __future__ import annotations

from typing import Dict

import torch


def random_rotation_2d(
    points_xy: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate 2D points.

    points_xy:
        [..., 2]

    angle:
        [B]
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    rot = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ],
        dim=-2,
    )

    while rot.ndim < points_xy.ndim + 1:
        rot = rot.unsqueeze(1)

    return torch.matmul(points_xy.unsqueeze(-2), rot).squeeze(-2)


def augment_observation(
    obs: Dict[str, torch.Tensor],
    max_rotation_rad: float = 1.57079632679,  # pi/2
) -> Dict[str, torch.Tensor]:
    """
    Apply simple state augmentation.

    Following the paper idea:
        1. ego-centric coordinates
        2. random rotations
    """
    motion = obs["motion"].clone()
    waypoints = obs["waypoints"].clone()

    batch_size = motion.shape[0]
    device = motion.device

    # ------------------------------------------------------------
    # Ego-centric transform
    # subtract ego current position
    # ------------------------------------------------------------
    ego_xy = motion[:, 0, -1, 0:2]

    motion[..., 0:2] -= ego_xy[:, None, None, :]
    waypoints[..., 0:2] -= ego_xy[:, None, None, None, :]

    # ------------------------------------------------------------
    # Random rotation
    # ------------------------------------------------------------
    angles = (
        torch.rand(batch_size, device=device) * 2.0 - 1.0
    ) * max_rotation_rad

    # motion positions
    motion_xy = motion[..., 0:2]
    motion[..., 0:2] = random_rotation_2d(motion_xy, angles)

    # motion velocities
    motion_v = motion[..., 2:4]
    motion[..., 2:4] = random_rotation_2d(motion_v, angles)

    # waypoint positions
    wp_xy = waypoints[..., 0:2]
    waypoints[..., 0:2] = random_rotation_2d(wp_xy, angles)

    # heading rotation
    motion[..., 4] += angles[:, None, None]
    waypoints[..., 2] += angles[:, None, None, None]

    return {
        "motion": motion,
        "waypoints": waypoints,
        "agent_mask": obs["agent_mask"],
        "route_mask": obs["route_mask"],
    }
