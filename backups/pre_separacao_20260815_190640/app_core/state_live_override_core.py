"""Helpers to promote shadow state layers into live pipeline once."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, MutableMapping

from app_core.prob_math_core import safe_float as _f


def apply_shadow_state_live_override(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    enabled: bool,
    shadow_trace: Dict[str, Any] | None,
    prob_win_series_fn: Callable[[float, int], float],
    stage_name: str,
    source_name: str,
    promoted_reason: str,
    not_applied_reason: str,
    session_state: MutableMapping[str, Any] | None = None,
    include_qwl_trace: bool = False,
    apply_confidence_governor_fn: Callable[..., Mapping[str, Any]] | None = None,
    governor_enabled: bool = True,
    governor_budget_pp: float = 8.0,
    governor_stage_name: str | None = None,
    governor_stage_cap_pp: float | None = None,
) -> Dict[str, Any]:
    p_map = _f(p_map_used, float("nan"))
    p_series = _f(p_series_used, float("nan"))
    trace: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": ("disabled_by_flag" if not bool(enabled) else ""),
        "p_map_before": (float(p_map) if math.isfinite(float(p_map)) else None),
        "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
        "p_map_after": None,
        "p_series_after": None,
    }
    pipeline_step = None

    try:
        shadow = shadow_trace if isinstance(shadow_trace, dict) else {}
        p_shadow_map = _f(shadow.get("p_map_new"), float("nan"))
        p_shadow_series = _f(shadow.get("p_series_new"), float("nan"))

        if bool(enabled) and bool(shadow.get("applied", False)) and math.isfinite(float(p_shadow_map)):
            p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            ps_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
            gov_out: Dict[str, Any] = {}

            p_candidate = float(p_shadow_map)
            if callable(apply_confidence_governor_fn) and math.isfinite(float(p_before)):
                try:
                    gv = apply_confidence_governor_fn(
                        stage=str(governor_stage_name or stage_name),
                        p_before=float(p_before),
                        p_candidate=float(p_candidate),
                        session_state=(session_state if isinstance(session_state, MutableMapping) else None),
                        enabled=bool(governor_enabled),
                        max_total_shrink_pp=float(governor_budget_pp),
                        reset=False,
                        stage_cap_pp=(float(governor_stage_cap_pp) if governor_stage_cap_pp is not None else None),
                    )
                    gov_out = dict(gv) if isinstance(gv, Mapping) else {}
                except Exception:
                    gov_out = {}
            p_gov = _f((gov_out or {}).get("p_out"), float("nan"))
            p_map = float(p_gov) if math.isfinite(float(p_gov)) else float(p_candidate)
            use_shadow_series = (
                math.isfinite(float(p_shadow_series))
                and math.isfinite(float(p_map))
                and abs(float(p_map) - float(p_candidate)) <= 1e-12
            )
            if use_shadow_series:
                p_series = float(p_shadow_series)
            else:
                try:
                    p_series = float(prob_win_series_fn(float(p_map), int(bo)))
                except Exception:
                    p_series = float("nan")

            if isinstance(session_state, MutableMapping):
                try:
                    session_state["_p_map_used"] = float(p_map)
                    session_state["_p_series_used"] = float(p_series)
                except Exception:
                    pass

            p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            ps_after = float(p_series) if math.isfinite(float(p_series)) else float("nan")
            applied_live = bool(
                math.isfinite(float(p_before))
                and math.isfinite(float(p_after))
                and abs(float(p_after) - float(p_before)) > 1e-12
            )
            trace = {
                "enabled": True,
                "applied": bool(applied_live),
                "reason": str(promoted_reason or ""),
                "p_map_before": (float(p_before) if math.isfinite(float(p_before)) else None),
                "p_series_before": (float(ps_before) if math.isfinite(float(ps_before)) else None),
                "p_map_after": (float(p_after) if math.isfinite(float(p_after)) else None),
                "p_series_after": (float(ps_after) if math.isfinite(float(ps_after)) else None),
                "diff_pp_map": (
                    float(100.0 * (float(p_after) - float(p_before)))
                    if math.isfinite(float(p_before)) and math.isfinite(float(p_after))
                    else None
                ),
                "diff_pp_series": (
                    float(100.0 * (float(ps_after) - float(ps_before)))
                    if math.isfinite(float(ps_before)) and math.isfinite(float(ps_after))
                    else None
                ),
            }
            if gov_out:
                trace["governor"] = {
                    "enabled": bool(gov_out.get("enabled", False)),
                    "limited": bool(gov_out.get("limited", False)),
                    "requested_shrink_pp": gov_out.get("requested_shrink_pp"),
                    "applied_shrink_pp": gov_out.get("applied_shrink_pp"),
                    "used_shrink_pp": gov_out.get("used_shrink_pp"),
                    "remaining_shrink_pp": gov_out.get("remaining_shrink_pp"),
                }
            if bool(include_qwl_trace) and isinstance(shadow.get("qwl_trace"), dict):
                trace["qwl_trace"] = dict(shadow.get("qwl_trace"))

            pipeline_step = {
                "stage": str(stage_name),
                "before": (float(p_before) if math.isfinite(float(p_before)) else None),
                "after": (float(p_after) if math.isfinite(float(p_after)) else None),
                "applied": bool(applied_live),
                "enabled": True,
                "source": str(source_name),
                "governor_limited": bool((trace.get("governor") or {}).get("limited", False)),
                "governor_applied_shrink_pp": (
                    float((trace.get("governor") or {}).get("applied_shrink_pp"))
                    if (trace.get("governor") or {}).get("applied_shrink_pp") is not None
                    else None
                ),
            }
        elif bool(enabled) and (not bool(shadow.get("applied", False))):
            trace["reason"] = str(not_applied_reason or "")
    except Exception as exc:
        trace = {
            "enabled": bool(enabled),
            "applied": False,
            "reason": f"error:{type(exc).__name__}",
            "error": str(exc),
            "p_map_before": (float(p_map) if math.isfinite(float(p_map)) else None),
            "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
            "p_map_after": None,
            "p_series_after": None,
        }

    return {
        "p_map_used": float(p_map) if math.isfinite(float(p_map)) else float("nan"),
        "p_series_used": float(p_series) if math.isfinite(float(p_series)) else float("nan"),
        "trace": trace,
        "pipeline_step": pipeline_step,
    }
