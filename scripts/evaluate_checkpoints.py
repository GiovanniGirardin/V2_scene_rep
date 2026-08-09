"""
Evaluate a subset of checkpoints and plot the comparison.

Example:
python3 scripts/evaluate_checkpoints.py \
  --config configs/runpod_4090.yaml \
  --checkpoint-dir checkpoints/runpod_4090_seed42 \
  --episodes 10 \
  --stride 2 \
  --out logs/checkpoint_comparison.csv \
  --plot-out logs/plots/checkpoint_comparison.png
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scene_rep.envs.smarts_env import SMARTSSceneRepEnv
from scene_rep.evaluation.rollout import evaluate_policy
from scene_rep.models.sac import SACAgent
from scene_rep.training.checkpointing import load_checkpoint
from scene_rep.utils.config import load_config
from scene_rep.utils.seed import set_seed
from scene_rep.utils.torch_utils import get_device


CHECKPOINT_RE = re.compile(r"sac_step_(\d+)\.pt$")
RATE_COLUMNS = ["success_rate", "collision_rate", "off_route_rate", "timeout_rate"]


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Checkpoint name does not match sac_step_<step>.pt: {path}")
    return int(match.group(1))


def find_checkpoints(checkpoint_dir: Path, stride: int) -> List[Path]:
    checkpoints = sorted(
        checkpoint_dir.glob("sac_step_*.pt"),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No sac_step_*.pt checkpoints found in {checkpoint_dir}")
    return checkpoints[::stride]


def write_row(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_existing_checkpoints(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {row["checkpoint"] for row in reader if row.get("checkpoint")}


def plot_results(csv_path: Path, output_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    numeric_columns = ["checkpoint_step", "mean_reward", "std_reward", "mean_steps", *RATE_COLUMNS]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("checkpoint_step")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(df["checkpoint_step"], df["mean_reward"], marker="o", label="mean_reward")
    if "std_reward" in df.columns:
        axes[0].fill_between(
            df["checkpoint_step"],
            df["mean_reward"] - df["std_reward"],
            df["mean_reward"] + df["std_reward"],
            alpha=0.2,
            label="std_reward",
        )
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for column in RATE_COLUMNS:
        if column in df.columns:
            axes[1].plot(df["checkpoint_step"], df[column], marker="o", label=column)
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["checkpoint_step"], df["mean_steps"], marker="o", color="#455a64")
    axes[2].set_xlabel("Checkpoint step")
    axes[2].set_ylabel("Mean steps")
    axes[2].grid(True, alpha=0.3)

    episodes = df["episodes"].iloc[-1] if "episodes" in df.columns else "unknown"
    fig.suptitle(f"Checkpoint comparison - {episodes} episodes each")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def evaluate_checkpoints(
    checkpoint_paths: Iterable[Path],
    config_path: Path,
    out_csv: Path,
    episodes: int,
    log_every_steps: int,
    resume: bool,
) -> None:
    cfg = load_config(str(config_path))
    device = get_device(cfg["project"]["device"])
    set_seed(int(cfg["project"]["seed"]))

    completed = load_existing_checkpoints(out_csv) if resume else set()

    env = SMARTSSceneRepEnv(cfg)
    try:
        agent = SACAgent(cfg).to(device)
        agent.eval()

        selected = list(checkpoint_paths)
        for index, checkpoint_path in enumerate(selected, start=1):
            checkpoint_key = str(checkpoint_path)
            if checkpoint_key in completed:
                print(f"[compare] skipping existing checkpoint {checkpoint_path}")
                continue

            print(f"[compare] evaluating {index}/{len(selected)}: {checkpoint_path}", flush=True)
            loaded_step = load_checkpoint(
                agent=agent,
                checkpoint_path=str(checkpoint_path),
                map_location=device,
            )
            agent.eval()

            metrics = evaluate_policy(
                env=env,
                agent=agent,
                device=device,
                episodes=episodes,
                deterministic=True,
                trace_actions=False,
                log_every_steps=log_every_steps,
                log_episodes=True,
            )

            row = {
                "config": str(config_path),
                "checkpoint": checkpoint_key,
                "checkpoint_step": loaded_step,
                "episodes": episodes,
                **metrics,
            }
            write_row(out_csv, row)
            print(f"[compare] saved metrics for step {loaded_step} to {out_csv}", flush=True)
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/runpod_4090.yaml"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/runpod_4090_seed42"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2, help="Use one checkpoint every N files.")
    parser.add_argument("--out", type=Path, default=Path("logs/checkpoint_comparison.csv"))
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("logs/plots/checkpoint_comparison.png"),
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=0,
        help="Print per-episode heartbeat every N simulator steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints already present in --out.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected checkpoints without running SMARTS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    checkpoints = find_checkpoints(args.checkpoint_dir, args.stride)
    steps = [checkpoint_step(path) for path in checkpoints]
    print(f"Selected {len(checkpoints)} checkpoints: {steps}")

    if args.dry_run:
        return

    evaluate_checkpoints(
        checkpoint_paths=checkpoints,
        config_path=args.config,
        out_csv=args.out,
        episodes=args.episodes,
        log_every_steps=args.log_every_steps,
        resume=args.resume,
    )
    plot_results(args.out, args.plot_out)


if __name__ == "__main__":
    main()
