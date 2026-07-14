"""Presets and aggressiveness slider for the Resumo tab.

Replaces 6+ individual sliders with a 2-axis control:

  axis 1: source preset  → which fonte dominates the Mix
          - "hist_first" : Hist 70, Mod 20, SC 10  (user's default — acerta mais com Hist)
          - "equilibrado": Hist 50, Mod 30, SC 20
          - "model_first": Mod 60, Hist 30, SC 10
          - "custom"     : do not overwrite — let the advanced sliders win

  axis 2: aggressiveness (0-10) → which picks survive the gates
          - 0-3 (conservador): odd_max=1.7, sample_liga=12, sample_h2h=5
          - 4-6 (neutro)     : odd_max=2.0, sample_liga=8,  sample_h2h=3
          - 7-10 (agressivo) : odd_max=3.0, sample_liga=5,  sample_h2h=1

Advanced users keep all the granular sliders in a collapsible block; this
module only writes the high-level dials.
"""
from __future__ import annotations

from typing import Any, MutableMapping


SOURCE_PRESETS: dict[str, dict[str, float]] = {
    "hist_first": {
        "resumo_mix_w_laplace": 70.0,
        "resumo_mix_w_model":   20.0,
        "resumo_mix_w_sc":      10.0,
    },
    "equilibrado": {
        "resumo_mix_w_laplace": 50.0,
        "resumo_mix_w_model":   30.0,
        "resumo_mix_w_sc":      20.0,
    },
    "model_first": {
        "resumo_mix_w_laplace": 30.0,
        "resumo_mix_w_model":   60.0,
        "resumo_mix_w_sc":      10.0,
    },
}

SOURCE_PRESET_LABELS: dict[str, str] = {
    "hist_first":  "Confia em Hist (recomendado)",
    "equilibrado": "Equilibrado",
    "model_first": "Confia em Modelo",
    "custom":      "Personalizado (avançado)",
}


def resolve_aggressiveness(level: int) -> dict[str, Any]:
    """Map a 0-10 slider to gate parameters.

    Returns a dict with: resumo_max_odd, resumo_min_sample_liga,
    resumo_min_sample_h2h.
    """
    lvl = int(max(0, min(10, level)))
    if lvl <= 3:
        return {
            "resumo_max_odd": 1.7,
            "resumo_min_sample_liga": 12,
            "resumo_min_sample_h2h": 5,
        }
    if lvl <= 6:
        return {
            "resumo_max_odd": 2.0,
            "resumo_min_sample_liga": 8,
            "resumo_min_sample_h2h": 3,
        }
    return {
        "resumo_max_odd": 3.0,
        "resumo_min_sample_liga": 5,
        "resumo_min_sample_h2h": 1,
    }


def apply_source_preset(session_state: MutableMapping[str, Any], preset_key: str) -> bool:
    """Apply the source weights from a preset. Returns True if applied.

    `custom` is a no-op so user keeps the manual sliders intact.
    """
    key = str(preset_key or "").strip().lower()
    if key == "custom":
        return False
    weights = SOURCE_PRESETS.get(key)
    if not weights:
        return False
    for k, v in weights.items():
        session_state[k] = float(v)
    return True


def apply_aggressiveness(session_state: MutableMapping[str, Any], level: int) -> dict[str, Any]:
    """Apply gate parameters from an aggressiveness level (0-10)."""
    params = resolve_aggressiveness(int(level))
    for k, v in params.items():
        session_state[k] = v
    return params


def aggressiveness_label(level: int) -> str:
    lvl = int(max(0, min(10, level)))
    if lvl <= 3:
        return f"Conservador ({lvl}/10)"
    if lvl <= 6:
        return f"Neutro ({lvl}/10)"
    return f"Agressivo ({lvl}/10)"


__all__ = [
    "SOURCE_PRESETS",
    "SOURCE_PRESET_LABELS",
    "aggressiveness_label",
    "apply_aggressiveness",
    "apply_source_preset",
    "resolve_aggressiveness",
]
