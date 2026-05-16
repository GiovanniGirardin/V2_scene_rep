from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class ReplayBuffer:
    """
    Memory-safe replay buffer for SAC.

    Stores transition dictionaries without preallocating huge arrays.
    """

    def __init__(self, config: Dict[str, Any]):
        sac_cfg = config["sac"]

        self.capacity = int(sac_cfg["replay_size"])
        self.batch_size = int(sac_cfg["batch_size"])

        self.ptr = 0
        self.storage: List[Dict[str, Any]] = []

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        item = {
            "obs": {
                k: np.asarray(v, dtype=np.float32)
                for k, v in obs.items()
            },
            "actions": np.asarray(action, dtype=np.float32),
            "rewards": np.asarray([reward], dtype=np.float32),
            "next_obs": {
                k: np.asarray(v, dtype=np.float32)
                for k, v in next_obs.items()
            },
            "dones": np.asarray([float(done)], dtype=np.float32),
        }

        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item

        self.ptr = (self.ptr + 1) % self.capacity

    def can_sample(self) -> bool:
        return len(self.storage) >= self.batch_size

    def sample(self) -> Dict[str, Any]:
        if not self.can_sample():
            raise RuntimeError(
                f"Not enough samples: size={len(self.storage)}, "
                f"batch_size={self.batch_size}"
            )

        idxs = np.random.randint(0, len(self.storage), size=self.batch_size)
        batch = [self.storage[i] for i in idxs]

        return {
            "obs": {
                key: np.stack([b["obs"][key] for b in batch], axis=0)
                for key in batch[0]["obs"].keys()
            },
            "actions": np.stack([b["actions"] for b in batch], axis=0),
            "rewards": np.stack([b["rewards"] for b in batch], axis=0),
            "next_obs": {
                key: np.stack([b["next_obs"][key] for b in batch], axis=0)
                for key in batch[0]["next_obs"].keys()
            },
            "dones": np.stack([b["dones"] for b in batch], axis=0),
        }

    def __len__(self) -> int:
        return len(self.storage)