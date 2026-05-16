from __future__ import annotations

from typing import Any, Dict

import numpy as np

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.models.sac import SACAgent
from scene_rep.utils.torch_utils import action_to_numpy, obs_to_torch
from scene_rep.evaluation.metrics import compute_episode_metrics


def run_episode(
    env: SMARTSSceneRepEnv,
    agent: SACAgent,
    device,
    deterministic: bool = True,
) -> Dict[str, Any]:
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    success = False
    collision = False
    off_route = False
    stagnation = False

    done = False

    while not done:
        obs_torch = obs_to_torch(
            obs,
            device=device,
            add_batch_dim=True,
        )

        action_torch = agent.act(
            obs_torch,
            deterministic=deterministic,
        )

        action = action_to_numpy(action_torch)

        obs, reward, done, info = env.step(tuple(action))

        total_reward += float(reward)
        steps += 1

        success = success or bool(info.get("success", False))
        collision = collision or bool(info.get("collision", False))
        off_route = off_route or bool(info.get("off_route", False))
        stagnation = stagnation or bool(info.get("stagnation", False))
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "collision": collision,
        "off_route": off_route,
        "stagnation": stagnation,
    }


def evaluate_policy(
    env: SMARTSSceneRepEnv,
    agent: SACAgent,
    device,
    episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, float]:
    results = [
        run_episode(
            env=env,
            agent=agent,
            device=device,
            deterministic=deterministic,
        )
        for _ in range(episodes)
    ]

    return compute_episode_metrics(results)