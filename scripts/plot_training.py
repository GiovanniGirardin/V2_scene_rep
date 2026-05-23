from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def save_curve(
    df: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    out_path: Path,
    window: int,
) -> None:
    x = df["global_step"]

    plt.figure()
    plt.plot(x, rolling(df[y], window))
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/training_episodes.csv")
    parser.add_argument("--out-dir", type=str, default="logs/plots")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    save_curve(
        df,
        y="episode_return",
        ylabel="Episode return",
        title="Training return",
        out_path=out_dir / "training_return.png",
        window=args.window,
    )

    save_curve(
        df,
        y="success",
        ylabel="Success rate",
        title="Training success rate",
        out_path=out_dir / "training_success.png",
        window=args.window,
    )

    save_curve(
        df,
        y="collision",
        ylabel="Collision rate",
        title="Training collision rate",
        out_path=out_dir / "training_collision.png",
        window=args.window,
    )

    save_curve(
        df,
        y="off_route",
        ylabel="Off-route rate",
        title="Training off-route rate",
        out_path=out_dir / "training_off_route.png",
        window=args.window,
    )

    save_curve(
        df,
        y="episode_length",
        ylabel="Episode length",
        title="Training episode length",
        out_path=out_dir / "training_episode_length.png",
        window=args.window,
    )

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()