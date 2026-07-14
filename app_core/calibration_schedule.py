"""Backward-compat re-export shim. Implementation lives in mlcore.calibration_schedule."""
from __future__ import annotations
from mlcore.calibration_schedule import (
    build_patch_snapshot,
    recommend_recalibration,
)

__all__ = ["build_patch_snapshot", "recommend_recalibration"]
