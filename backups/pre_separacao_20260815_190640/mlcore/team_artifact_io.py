"""Persistence helpers for team state artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def load_team_state_artifact(path: str | Path) -> Dict[str, Any]:
    """Load a team state artifact JSON from disk; returns {} on missing file or parse error."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_team_state_artifact(path: str | Path, artifact: Dict[str, Any]) -> None:
    """Atomically write artifact to disk via a .tmp file to avoid partial writes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            pass
    tmp.replace(p)
