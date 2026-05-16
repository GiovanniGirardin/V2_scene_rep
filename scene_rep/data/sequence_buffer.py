from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class SequenceBuffer:
    """
    Memory-safe replay buffer for SLT sequence training.

    Unlike the previous version, this does NOT preallocate huge tensors.
    It stores complete sequence dictionaries in a circular Python list.
    """

    def __init__(self, config: Dict[str, Any]):
        sac_cfg = config["sac"]
        slt_cfg = config["slt"]

        self.capacity = int(slt_cfg.get("sequence_replay_size", 1000))
        self.batch_size = int(slt_cfg.get("batch_size", sac_cfg["batch_size"]))

        self.ptr = 0
        self.storage: List[Dict[str, np.ndarray]] = []

    def add(self, sequence: Dict[str, np.ndarray]) -> None:
        sequence = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in sequence.items()
        }

        if len(self.storage) < self.capacity:
            self.storage.append(sequence)
        else:
            self.storage[self.ptr] = sequence

        self.ptr = (self.ptr + 1) % self.capacity

    def can_sample(self) -> bool:
        return len(self.storage) >= self.batch_size

    def sample(self) -> Dict[str, np.ndarray]:
        if not self.can_sample():
            raise RuntimeError(
                f"Not enough sequences: size={len(self.storage)}, "
                f"batch_size={self.batch_size}"
            )

        idxs = np.random.randint(0, len(self.storage), size=self.batch_size)
        batch = [self.storage[i] for i in idxs]

        keys = batch[0].keys()

        return {
            key: np.stack([item[key] for item in batch], axis=0)
            for key in keys
        }

    def __len__(self) -> int:
        return len(self.storage)