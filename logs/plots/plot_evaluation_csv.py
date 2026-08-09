from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


RATE_COLUMNS = [
    "success_rate",
    "collision_rate",
    "off_route_rate",
    "stagnation_rate",
    "timeout_rate",
]

RATE_LABELS = {
    "success_rate": "Success",
    "collision_rate": "Collision",
    "off_route_rate": "Off route",
    "stagnation_rate": "Stagnation",
    "timeout_rate": "Timeout",
}

RATE_COLORS = {
    "success_rate": "#2e7d32",
    "collision_rate": "#c62828",
    "off_route_rate": "#6a1b9a",
    "stagnation_rate": "#ef6c00",
    "timeout_rate": "#1565c0",
}


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    for column in ["checkpoint_step", "mean_reward", "std_reward", "mean_steps", *RATE_COLUMNS]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def format_rate_label(column: str, value: float) -> str:
    return f"{RATE_LABELS.get(column, column)}: {value * 100:.1f}%"


def plot_latest_summary(df: pd.DataFrame, out_dir: Path) -> None:
    latest = df.iloc[-1]
    rates = latest[[column for column in RATE_COLUMNS if column in df.columns]].dropna()
    checkpoint = latest.get("checkpoint_step", "unknown")
    episodes = latest.get("episodes", "unknown")

    fig, ax = plt.subplots(figsize=(9, 5))

    x_labels = [RATE_LABELS.get(column, column) for column in rates.index]
    colors = [RATE_COLORS.get(column, "#455a64") for column in rates.index]
    bars = ax.bar(x_labels, rates.values, color=colors)
    ax.bar_label(bars, labels=[f"{value * 100:.1f}%" for value in rates.values], padding=3)
    ax.set_title(f"Evaluation outcome rates - checkpoint {checkpoint}")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        bars,
        [format_rate_label(column, float(value)) for column, value in rates.items()],
        title=f"{episodes:g} episodes" if pd.notna(episodes) else "Episodes unknown",
        loc="upper right",
    )
    fig.tight_layout()

    output_path = out_dir / "evaluation_outcome_rates.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")

    reward = float(latest.get("mean_reward", 0.0))
    reward_std = float(latest.get("std_reward", 0.0))
    mean_steps = float(latest.get("mean_steps", 0.0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Mean reward", "Mean steps"], [reward, mean_steps], color=["#00838f", "#455a64"])
    ax.errorbar(["Mean reward"], [reward], yerr=[reward_std], fmt="none", color="black", capsize=5)
    ax.set_title(f"Latest run summary - checkpoint {checkpoint}")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    output_path = out_dir / "evaluation_reward_steps.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def plot_checkpoint_trends(df: pd.DataFrame, out_dir: Path) -> None:
    if "checkpoint_step" not in df.columns or len(df) < 2:
        return

    df = df.sort_values("checkpoint_step")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

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
            axes[1].plot(df["checkpoint_step"], df[column], marker="o", label=RATE_LABELS.get(column, column))

    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Evaluation trends")
    fig.tight_layout()

    output_path = out_dir / "evaluation_trends.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("logs/evaluation_results_final.csv"),
        help="Path to the evaluation CSV produced by scripts/evaluate.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/plots"),
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving them.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.csv)

    print(df.tail())
    plot_latest_summary(df, args.out_dir)
    plot_checkpoint_trends(df, args.out_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
