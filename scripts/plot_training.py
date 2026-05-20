from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def rolling_mean(series, window: int):
    return series.rolling(window=window, min_periods=1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/training_episodes.csv")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--out", type=str, default="logs/training_plot.png")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)

    x = df["global_step"]

    plt.figure()
    plt.plot(x, rolling_mean(df["episode_return"], args.window))
    plt.xlabel("Training step")
    plt.ylabel("Episode return")
    plt.title("Training return")
    plt.savefig(args.out.replace(".png", "_return.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, rolling_mean(df["success"], args.window))
    plt.xlabel("Training step")
    plt.ylabel("Success rate")
    plt.title("Training success rate")
    plt.savefig(args.out.replace(".png", "_success.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(x, rolling_mean(df["collision"], args.window))
    plt.xlabel("Training step")
    plt.ylabel("Collision rate")
    plt.title("Training collision rate")
    plt.savefig(args.out.replace(".png", "_collision.png"), dpi=200)
    plt.close()

    print("Saved plots in:", Path(args.out).parent)


if __name__ == "__main__":
    main()