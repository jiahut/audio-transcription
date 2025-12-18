from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file root must be a mapping/object.")
        return data
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config file root must be a JSON object.")
        return data
    raise ValueError("Unsupported config extension (use .yaml/.yml/.json).")


def dump_config_file(path: Path, data: dict[str, Any]) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return
    if suffix == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return
    raise ValueError("Unsupported config extension (use .yaml/.yml/.json).")

