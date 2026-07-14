"""Post-fusion calibration helper for the live ML pipeline.

Applies Microseg + Platt re-calibration on the fused probability (p_fused), AFTER
team+players fusion. The ablation study (tools/run_ablation_cal_position.py) showed
that calibrating both BEFORE and AFTER fusion yields:
  - +0.97pp logloss vs calibrating only before fusion (current default behavior)
  - ECE reduced by 48% (0.085 -> 0.044 on 1313-sample holdout)

Feature-flagged via session_state["ml_postfusion_cal_enabled"] (default False).
Enable in production after validating in your specific deployment context.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from app_core.prob_math_core import safe_float as _f


def apply_postfusion_calibration(
    p_fused: float,
    *,
    enabled: bool,
    league_hint: str,
    bo: int,
    split_hint: str = "",
    phase_hint: str = "regular",
    side: str = "blue",
    microseg_table: Optional[Mapping[str, Any]] = None,
    microseg_min_n: int = 30,
    microseg_strength: float = 0.60,
    microseg_cap_pp: float = 8.0,
    microseg_hierarchical: bool = True,
    microseg_hier_decay: float = 0.80,
    apply_microseg_fn: Optional[Callable[..., Tuple[float, Dict[str, Any]]]] = None,
    apply_fused_postcal_fn: Optional[Callable[..., Tuple[float, Dict[str, Any]]]] = None,
    use_fitted_postcal: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Apply Microseg + Platt to a fused probability and return (p_out, trace_info).

    If enabled is False, returns (p_fused, info_with_enabled_false) unchanged.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "p_in": float(p_fused) if isinstance(p_fused, (int, float)) and math.isfinite(float(p_fused)) else None,
        "p_out": None,
        "microseg": {"applied": False},
        "postcal": {"applied": False},
    }
    if not bool(enabled):
        info["reason"] = "disabled"
        info["p_out"] = info["p_in"]
        return float(p_fused), info

    p_in = _f(p_fused, float("nan"))
    if not math.isfinite(p_in) or p_in <= 0.0 or p_in >= 1.0:
        info["reason"] = "invalid_p_in"
        info["p_out"] = info["p_in"]
        return float(p_fused), info

    p_out = float(p_in)
    bo_hint = str(int(bo)) if int(bo) in (1, 3, 5) else "1"

    # Step 1: Microseg on p_fused
    if callable(apply_microseg_fn) and microseg_table:
        try:
            p_after_ms, ms_info = apply_microseg_fn(
                float(p_out),
                league=str(league_hint or ""),
                split=str(split_hint or ""),
                side=str(side or "blue"),
                bo=bo_hint,
                phase=str(phase_hint or "regular"),
                table=microseg_table,
                min_n=int(microseg_min_n),
                strength=float(microseg_strength),
                cap_pp=float(microseg_cap_pp),
                hierarchical=bool(microseg_hierarchical),
                hierarchy_decay=float(microseg_hier_decay),
            )
            if isinstance(ms_info, dict):
                info["microseg"] = dict(ms_info)
                info["microseg"]["applied"] = bool(ms_info.get("applied", abs(float(p_after_ms) - p_out) > 1e-9))
            if math.isfinite(float(p_after_ms)):
                p_out = float(p_after_ms)
        except Exception as exc:
            info["microseg"]["error"] = f"{type(exc).__name__}: {exc}"

    # Step 2: Fused post-calibration (Platt) on p_fused (or microseg'd p_fused)
    if callable(apply_fused_postcal_fn):
        try:
            p_after_pc, pc_info = apply_fused_postcal_fn(
                float(p_out),
                league_hint=str(league_hint or ""),
                use_fitted=bool(use_fitted_postcal),
            )
            if isinstance(pc_info, dict):
                info["postcal"] = dict(pc_info)
            if math.isfinite(float(p_after_pc)):
                p_out = float(p_after_pc)
        except Exception as exc:
            info["postcal"]["error"] = f"{type(exc).__name__}: {exc}"

    info["p_out"] = float(p_out)
    info["applied"] = bool(abs(float(p_out) - p_in) > 1e-9)
    info["diff_pp"] = float(100.0 * (p_out - p_in))
    return float(p_out), info
