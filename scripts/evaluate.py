'''
EXAMPLE USAGE:
python3 scripts/evaluate.py \
  --config configs/scenarios/unprotected_left_turn.yaml \
  --checkpoint checkpoints/sac_step_3000.pt \
  --episodes 10 \
  --out logs/evaluation_results.csv

'''





from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.evaluation.rollout import evaluate_policy
from scene_rep.models.sac import SACAgent
from scene_rep.training.checkpointing import load_checkpoint
from scene_rep.utils.config import load_config
from scene_rep.utils.torch_utils import get_device
from scene_rep.utils.seed import set_seed


def append_metrics_csv(
    path: str,
    config_path: str,
    checkpoint_path: str,
    checkpoint_step: int,
    episodes: int,
    metrics: dict,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "episodes": episodes,
        **metrics,
    }

    file_exists = out_path.exists()

    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=str, default="logs/evaluation_results.csv")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SMARTS in non-headless mode so behavior can be viewed in the simulator GUI.",
    )
    parser.add_argument(
        "--trace-actions",
        action="store_true",
        help="Print per-step policy actions, adapted SMARTS actions, reward, and progress.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.gui:
        cfg["smarts"]["headless"] = False
        cfg["smarts"]["envision"] = True
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
        trace_actions=args.trace_actions,
    )

    env.close()

    append_metrics_csv(
        path=args.out,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint_step=step,
        episodes=args.episodes,
        metrics=metrics,
    )

    print(f"Loaded checkpoint step: {step}")
    print("Evaluation metrics:")

    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print(f"Saved evaluation results to: {args.out}")


if __name__ == "__main__":
    main()