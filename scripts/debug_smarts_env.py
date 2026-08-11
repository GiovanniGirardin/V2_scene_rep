from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.utils.config import load_config


def normalized_action(speed_mps: float, max_speed_mps: float, lane_signal: float) -> Tuple[float, float]:
    """Convert a human-readable speed into the normalized SAC action domain."""
    normalized_speed = 2.0 * (speed_mps / max(max_speed_mps, 1e-6)) - 1.0
    return (
        float(np.clip(normalized_speed, -1.0, 1.0)),
        float(np.clip(lane_signal, -1.0, 1.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a SMARTS scenario with a deterministic lane-following action. "
            "Use --render to open SMARTS/Envision visualization."
        )
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=5.0,
        help="Target speed in m/s, converted to the normalized SAC action domain.",
    )
    parser.add_argument(
        "--lane-signal",
        type=float,
        default=0.0,
        help="Normalized lane signal: -1 left, 0 keep, +1 right.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Run SMARTS non-headless and enable the Envision client.",
    )
    parser.add_argument(
        "--random-actions",
        action="store_true",
        help="Use random normalized policy actions instead of keep-lane action.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help=(
            "Wall-clock delay after each simulator step. Use e.g. 0.1 with "
            "--render to inspect the live Envision stream."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["smarts"]["use_dummy"] = False
    if args.render:
        cfg["smarts"]["headless"] = False
        cfg["smarts"]["envision"] = True

    max_speed_mps = float(cfg["action"]["max_speed_mps"])
    keep_lane_action = normalized_action(
        speed_mps=args.speed_mps,
        max_speed_mps=max_speed_mps,
        lane_signal=args.lane_signal,
    )

    print(
        f"Scenario: {cfg['smarts']['scenario']}\n"
        f"Agent type: {cfg['smarts']['agent_type']}\n"
        f"Render: {args.render}\n"
        f"Policy action (normalized): {keep_lane_action}; "
        f"target speed: {args.speed_mps:.2f} m/s"
    )

    env = SMARTSSceneRepEnv(cfg)
    try:
        for episode in range(1, args.episodes + 1):
            obs = env.reset()
            episode_return = 0.0

            print(f"\nEpisode {episode}")
            print("motion:", obs["motion"].shape)
            print("waypoints:", obs["waypoints"].shape)
            print("agent_mask:", obs["agent_mask"])
            print("route_mask ego:", obs["route_mask"][0])

            for step in range(args.steps):
                if args.random_actions:
                    action = (
                        float(np.random.uniform(-1.0, 1.0)),
                        float(np.random.uniform(-1.0, 1.0)),
                    )
                else:
                    action = keep_lane_action

                obs, reward, done, info = env.step(action)
                episode_return += reward

                if step % 10 == 0 or done:
                    sent = info["smarts_action"]
                    print(
                        f"step={step:03d} reward={reward:+.3f} "
                        f"return={episode_return:+.3f} "
                        f"progress={info.get('progress', 0.0):.3f} "
                        f"distance={info.get('distance_travelled', 0.0):.1f} "
                        f"action=(speed={sent['speed']:.1f}, lane={sent['lane_change']:+d}) "
                        f"neighbors={int(obs['agent_mask'][1:].sum())} "
                        f"done={done}"
                    )

                if args.sleep_sec > 0.0:
                    time.sleep(args.sleep_sec)

                if done:
                    print(
                        "Terminal state: "
                        f"{info.get('terminal_reason', 'unknown')} | "
                        f"success={info['success']} collision={info['collision']} "
                        f"off_route={info['off_route']} stagnation={info['stagnation']} "
                        f"timeout={info['timeout']}"
                    )
                    break
            else:
                print(f"Stopped after requested {args.steps} steps; return={episode_return:+.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
