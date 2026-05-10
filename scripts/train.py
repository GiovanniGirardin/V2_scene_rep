from __future__ import annotations

import argparse

from scene_rep.training.trainer import Trainer
from scene_rep.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
