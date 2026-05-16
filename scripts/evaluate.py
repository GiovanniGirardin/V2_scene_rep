from __future__ import annotations

import argparse

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.evaluation.rollout import evaluate_policy
from scene_rep.models.sac import SACAgent
from scene_rep.training.checkpointing import load_checkpoint
from scene_rep.utils.config import load_config
from scene_rep.utils.torch_utils import get_device
from scene_rep.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])
    set_seed(int(cfg["project"]["seed"]))

    env = SMARTSSceneRepEnv(cfg)
    agent = SACAgent(cfg).to(device)

    step = load_checkpoint(
        agent=agent,
        checkpoint_path=args.checkpoint,
        map_location=device,
    )

    agent.eval()

    metrics = evaluate_policy(
        env=env,
        agent=agent,
        device=device,
        episodes=args.episodes,
        deterministic=True,
    )

    print(f"Loaded checkpoint step: {step}")
    print("Evaluation metrics:")

    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
