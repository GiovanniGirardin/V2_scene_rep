from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from scene_rep.models.sac import SACAgent


def save_checkpoint(
    agent: SACAgent,
    step: int,
    checkpoint_dir: str,
    extra: Dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    file_path = checkpoint_path / f"sac_step_{step}.pt"

    payload = {
        "step": step,
        "encoder": agent.encoder.state_dict(),
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "target_encoder": agent.target_encoder.state_dict(),
        "target_critic1": agent.target_critic1.state_dict(),
        "target_critic2": agent.target_critic2.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "extra": extra or {},
    }

    torch.save(payload, file_path)
    return file_path


def load_checkpoint(
    agent: SACAgent,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> int:
    payload = torch.load(checkpoint_path, map_location=map_location)

    agent.encoder.load_state_dict(payload["encoder"])
    agent.actor.load_state_dict(payload["actor"])
    agent.critic1.load_state_dict(payload["critic1"])
    agent.critic2.load_state_dict(payload["critic2"])
    agent.target_encoder.load_state_dict(payload["target_encoder"])
    agent.target_critic1.load_state_dict(payload["target_critic1"])
    agent.target_critic2.load_state_dict(payload["target_critic2"])

    agent.log_alpha.data = payload["log_alpha"].to(agent.log_alpha.device)

    return int(payload["step"])
