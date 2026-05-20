from __future__ import annotations

import argparse
import numpy as np

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["smarts"]["use_dummy"] = False

    env = SMARTSSceneRepEnv(cfg)
    obs = env.reset()

    print("Initial observation:")
    print("motion:", obs["motion"].shape)
    print("waypoints:", obs["waypoints"].shape)
    print("agent_mask:", obs["agent_mask"])
    print("route_mask ego:", obs["route_mask"][0])

    for step in range(args.steps):
        speed = np.random.uniform(2.0, 8.0)
        lane_change = np.random.choice([-1, 0, 1])

        obs, reward, done, info = env.step((speed, lane_change))

        if step % 20 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:.3f} "
                f"done={done} "
                f"mask={obs['agent_mask']} "
                f"ego_routes={obs['route_mask'][0].sum()} "
                f"neighbors={int(obs['agent_mask'][1:].sum())}"
            )

        if done:
            print("Episode finished. Resetting.")
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()