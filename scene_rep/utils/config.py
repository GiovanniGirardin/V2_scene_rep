from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")

    inherit_path = cfg.pop("inherit", None)

    if inherit_path is not None:
        inherit_path = Path(inherit_path)

        if not inherit_path.is_absolute():
            inherit_path = path.parent / inherit_path

        base_cfg = load_config(inherit_path)
        cfg = deep_update(base_cfg, cfg)

    return cfg

