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
            "agents_alive_done",
            "stagnation",
            "timeout",
            "terminal_reason",
        ]

        self.episode = 0

        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            self._ensure_header()

    def _ensure_header(self) -> None:
        with self.path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            rows = list(reader)

        if existing_fieldnames == self.fieldnames:
            return

        if "timeout" not in existing_fieldnames:
            for row in rows:
                row["timeout"] = self._infer_legacy_timeout(row)

        if "agents_alive_done" not in existing_fieldnames:
            for row in rows:
                row["agents_alive_done"] = "0"

        if "terminal_reason" not in existing_fieldnames:
            for row in rows:
                row["terminal_reason"] = self._infer_terminal_reason(row)

        backup_path = self.path.with_suffix(self.path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(self.path.read_text())

        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _infer_legacy_timeout(self, row: Dict[str, Any]) -> str:
        terminal_flags = ["success", "collision", "off_route", "agents_alive_done", "stagnation"]
        has_terminal_flag = any(str(row.get(flag, "0")) == "1" for flag in terminal_flags)

        try:
            episode_length = int(row.get("episode_length", 0))
        except (TypeError, ValueError):
            episode_length = 0

        return "1" if episode_length >= 299 and not has_terminal_flag else "0"

    def _infer_terminal_reason(self, row: Dict[str, Any]) -> str:
        for reason in ["success", "collision", "off_route", "agents_alive_done", "stagnation", "timeout"]:
            if str(row.get(reason, "0")) == "1":
                return reason
        return "terminated"

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
            "agents_alive_done": int(bool(info.get("agents_alive_done", False))),
            "stagnation": int(bool(info.get("stagnation", False))),
            "timeout": int(bool(info.get("timeout", False))),
            "terminal_reason": str(info.get("terminal_reason", "terminated")),
        }

        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)