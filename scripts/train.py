'''
EXAMPLE USAGE:
python3 scripts/train.py \
  --config configs/default.yaml

'''


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
