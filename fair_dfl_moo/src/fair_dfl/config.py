"""Config loading utility for unified experiment execution."""

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "task" not in cfg or "training" not in cfg:
        raise ValueError("Config must contain top-level keys: 'task' and 'training'.")
    return cfg
