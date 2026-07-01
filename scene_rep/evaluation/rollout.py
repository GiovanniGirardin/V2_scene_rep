from __future__ import annotations

from typing import Any, Dict

import numpy as np

import torch

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.models.sac import SACAgent
from scene_rep.utils.torch_utils import action_to_numpy, obs_to_torch
from scene_rep.evaluation.metrics import compute_episode_metrics


def run_episode(
    env: SMARTSSceneRepEnv,
    agent: SACAgent,
    device,
    deterministic: bool = True,
    trace_actions: bool = False,
    episode_index: int = 0,
) -> Dict[str, Any]:
    obs = env.reset()

    total_reward = 0.0
    steps = 0

    success = False
    collision = False
    off_route = False
    stagnation = False
    timeout = False

    done = False

    while not done:
        obs_torch = obs_to_torch(
            obs,
            device=device,
            add_batch_dim=True,
        )

        with torch.no_grad():
            action_torch = agent.act(
                obs_torch,
                deterministic=deterministic,
            )

        action = action_to_numpy(action_torch)

        obs, reward, done, info = env.step(tuple(action))

        total_reward += float(reward)
        steps += 1

        if trace_actions:
            smarts_action = info.get("smarts_action", {})
            print(
                f"episode={episode_index} step={steps} "
                f"raw_action=({action[0]:.4f}, {action[1]:.4f}) "
                f"speed={float(smarts_action.get('speed', 0.0)):.3f} "
                f"lane_change={int(smarts_action.get('lane_change', 0))} "
                f"reward={float(reward):.5f} "
                f"progress={float(info.get('progress', 0.0)):.3f} "
                f"done={done}"
            )

        success = success or bool(info.get("success", False))
        collision = collision or bool(info.get("collision", False))
        off_route = off_route or bool(info.get("off_route", False))
        stagnation = stagnation or bool(info.get("stagnation", False))
        timeout = timeout or bool(info.get("timeout", False))
    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "collision": collision,
        "off_route": off_route,
        "stagnation": stagnation,
        "timeout": timeout,
    }


def evaluate_policy(
    env: SMARTSSceneRepEnv,
    agent: SACAgent,
    device,
    episodes: int = 10,
    deterministic: bool = True,
    trace_actions: bool = False,
) -> Dict[str, float]:
    results = [
        run_episode(
            env=env,
            agent=agent,
            device=device,
            deterministic=deterministic,
            trace_actions=trace_actions,
            episode_index=episode_idx + 1,
        )
        for episode_idx in range(episodes)
    ]

    return compute_episode_metrics(results)
