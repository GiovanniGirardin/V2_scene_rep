from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


def get_device(device_name: str = "auto") -> torch.device:
    """
    Select torch device.

    device_name:
        "auto" -> cuda if available else cpu
        "cpu"
        "cuda"
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def obs_to_torch(
    obs: Dict[str, np.ndarray],
    device: torch.device,
    add_batch_dim: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert one observation dict from numpy to torch.
    """
    torch_obs = {}

    for key, value in obs.items():
        tensor = torch.tensor(value, dtype=torch.float32, device=device)

        if add_batch_dim:
            tensor = tensor.unsqueeze(0)

        torch_obs[key] = tensor

    return torch_obs


def batch_to_torch(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Convert replay buffer batch from numpy to torch.
    """
    return {
        "obs": obs_to_torch(batch["obs"], device=device),
        "next_obs": obs_to_torch(batch["next_obs"], device=device),
        "actions": torch.tensor(batch["actions"], dtype=torch.float32, device=device),
        "rewards": torch.tensor(batch["rewards"], dtype=torch.float32, device=device),
        "dones": torch.tensor(batch["dones"], dtype=torch.float32, device=device),
    }


def action_to_numpy(action: torch.Tensor) -> np.ndarray:
    """
    Convert torch action to numpy vector.

    Expected input:
        [1, action_dim] or [action_dim]
    """
    action = action.detach().cpu()

    if action.ndim == 2:
        action = action.squeeze(0)

    return action.numpy().astype(np.float32)
