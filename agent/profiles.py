"""Per-model call settings. tuning/tune_qwen.py writes the tuned ones to
model_profiles.json; any model not listed (Claude, the fake) gets the defaults."""
from __future__ import annotations

import json
from pathlib import Path

from .loop import ModelSettings

PROFILES_FILE = Path(__file__).with_name("model_profiles.json")


def settings_for(model: str) -> ModelSettings:
    try:
        profiles = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ModelSettings()
    entry = profiles.get(model)
    if not isinstance(entry, dict):
        return ModelSettings()
    known = ModelSettings.__dataclass_fields__
    return ModelSettings(**{k: v for k, v in entry.get("settings", {}).items() if k in known})
