from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def compute_episode_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute evaluation metrics from a list of episode result dicts.

    Expected episode keys:
        total_reward
        steps
        success
        collision
        off_route

    Optional:
        stagnation
    """
    if len(episodes) == 0:
        raise ValueError("Cannot compute metrics from an empty episode list.")

    rewards = np.array([e["total_reward"] for e in episodes], dtype=np.float32)
    steps = np.array([e["steps"] for e in episodes], dtype=np.float32)

    success = np.array([e.get("success", False) for e in episodes], dtype=np.float32)
    collision = np.array([e.get("collision", False) for e in episodes], dtype=np.float32)
    off_route = np.array([e.get("off_route", False) for e in episodes], dtype=np.float32)
    stagnation = np.array([e.get("stagnation", False) for e in episodes], dtype=np.float32)

    return {
        "episodes": float(len(episodes)),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": float(steps.mean()),
        "success_rate": float(success.mean()),
        "collision_rate": float(collision.mean()),
        "off_route_rate": float(off_route.mean()),
        "stagnation_rate": float(stagnation.mean()),
    }
