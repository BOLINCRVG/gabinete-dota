"""Backward-compat re-export shim. Implementation lives in mlcore.series_bo_core."""
from __future__ import annotations
from mlcore.series_bo_core import (
    apply_series_bo_dplite_live,
    enforce_series_economic_coherence,
)

__all__ = ["apply_series_bo_dplite_live", "enforce_series_economic_coherence"]
