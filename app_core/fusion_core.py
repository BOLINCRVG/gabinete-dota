"""Backward-compat re-export shim. Implementation lives in mlcore.fusion_core."""
from __future__ import annotations
from mlcore.fusion_core import (
    fuse_with_coherence_guard,
    season_form_blend,
    current_season_override,
    hybrid_momentum_blend,
)

__all__ = [
    "current_season_override",
    "fuse_with_coherence_guard",
    "hybrid_momentum_blend",
    "season_form_blend",
]
