from __future__ import annotations

from typing import Dict

import torch


def random_rotation_2d(
    points_xy: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate 2D vectors.

    Args:
        points_xy: Tensor shaped [B, ... , 2].
        angle: Per-sample angle shaped [B].
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    rotation = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),
            torch.stack([sin_a, cos_a], dim=-1),
        ],
        dim=-2,
    )

    while rotation.ndim < points_xy.ndim + 1:
        rotation = rotation.unsqueeze(1)

    return torch.matmul(points_xy.unsqueeze(-2), rotation).squeeze(-2)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without changing gradients almost everywhere."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def augment_observation(
    obs: Dict[str, torch.Tensor],
    max_rotation_rad: float = 1.57079632679,  # pi / 2
    group_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Apply a rigid ego-centric transform and a random rotation.

    Motion features must use the canonical layout:
        [x, y, vx, vy, heading]

    When consecutive states of a trajectory are flattened into the batch
    dimension, pass ``group_size=T``. One angle is then sampled per trajectory
    and reused for all T states, preserving transition dynamics for the SLT.

    Existing zero-padding is restored after the transform so that padding does
    not turn into artificial agents or waypoint tokens.
    """
    motion = obs["motion"].clone()
    waypoints = obs["waypoints"].clone()

    batch_size = motion.shape[0]
    if batch_size % group_size != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by group_size {group_size}."
        )

    device = motion.device
    dtype = motion.dtype

    # Padding is represented by all-zero feature vectors in the current data
    # pipeline. Preserve it after applying translations/rotations.
    motion_valid = motion.abs().sum(dim=-1, keepdim=True) > 0
    waypoint_valid = waypoints.abs().sum(dim=-1, keepdim=True) > 0

    # Use the most recent ego state as the reference point. This works for the
    # absolute SMARTS frame and is a no-op for already ego-relative positions.
    ego_xy = motion[:, 0, -1, 0:2]
    motion[..., 0:2] -= ego_xy[:, None, None, :]
    waypoints[..., 0:2] -= ego_xy[:, None, None, None, :]

    trajectory_batch_size = batch_size // group_size
    trajectory_angles = (
        torch.rand(trajectory_batch_size, device=device, dtype=dtype) * 2.0 - 1.0
    ) * max_rotation_rad
    angles = trajectory_angles.repeat_interleave(group_size)

    motion[..., 0:2] = random_rotation_2d(motion[..., 0:2], angles)
    motion[..., 2:4] = random_rotation_2d(motion[..., 2:4], angles)
    waypoints[..., 0:2] = random_rotation_2d(waypoints[..., 0:2], angles)

    motion[..., 4] = _wrap_angle(motion[..., 4] + angles[:, None, None])
    waypoints[..., 2] = _wrap_angle(
        waypoints[..., 2] + angles[:, None, None, None]
    )

    motion = torch.where(motion_valid, motion, torch.zeros_like(motion))
    waypoints = torch.where(waypoint_valid, waypoints, torch.zeros_like(waypoints))

    return {
        "motion": motion,
        "waypoints": waypoints,
        "agent_mask": obs["agent_mask"],
        "route_mask": obs["route_mask"],
    }
