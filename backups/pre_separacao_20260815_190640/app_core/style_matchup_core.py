"""Backward-compat re-export shim. Implementation lives in mlcore.style_matchup_core."""
from __future__ import annotations
from mlcore.style_matchup_core import (
    STYLE_MATCHUP_B_MODEL_FEATURES,
    should_apply_style_matchup_b_model_gate,
    build_style_matchup_b_model_features,
    apply_style_matchup_b_model,
    apply_style_matchup_b_lite,
    apply_style_matchup_b_live,
)

__all__ = [
    "STYLE_MATCHUP_B_MODEL_FEATURES",
    "apply_style_matchup_b_lite",
    "apply_style_matchup_b_live",
    "apply_style_matchup_b_model",
    "build_style_matchup_b_model_features",
    "should_apply_style_matchup_b_model_gate",
]
