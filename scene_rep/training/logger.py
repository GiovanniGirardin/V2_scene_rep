from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict


class EpisodeLogger:
    def __init__(self, log_dir: str = "logs", filename: str = "training_episodes.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.path = self.log_dir / filename

        self.fieldnames = [
            "global_step",
            "episode",
            "episode_return",
            "episode_length",
            "success",
            "collision",
            "off_route",
            "stagnation",
        ]

        self.episode = 0

        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_episode(
        self,
        global_step: int,
        episode_return: float,
        episode_length: int,
        info: Dict[str, Any],
    ) -> None:
        self.episode += 1

        row = {
            "global_step": global_step,
            "episode": self.episode,
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
            "success": int(bool(info.get("success", False))),
            "collision": int(bool(info.get("collision", False))),
            "off_route": int(bool(info.get("off_route", False))),
            "stagnation": int(bool(info.get("stagnation", False))),
        }

        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)