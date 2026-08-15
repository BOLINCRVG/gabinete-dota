"""Helpers to apply live map stages with consistent tracing/session updates."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional

from app_core.pipeline_trace_core import append_pipeline_step


def _finite_or_nan(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return float(x) if math.isfinite(x) else float("nan")


def _opt_float(v: Any) -> Optional[float]:
    x = _finite_or_nan(v)
    return float(x) if math.isfinite(x) else None


def build_live_stage_trace(
    *,
    enabled: bool,
    p_map_used: Any,
    p_series_used: Any,
    map_before_key: str = "p_map_before",
    map_after_key: str = "p_map_after",
    series_before_key: str = "p_series_before",
    series_after_key: str = "p_series_after",
    reason_disabled: str = "disabled_by_flag",
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "applied": False,
        "reason": (str(reason_disabled) if not bool(enabled) else ""),
        str(map_before_key): _opt_float(p_map_used),
        str(series_before_key): _opt_float(p_series_used),
        str(map_after_key): None,
        str(series_after_key): None,
    }


def build_live_stage_error_trace(
    *,
    enabled: bool,
    error: Exception | str,
    p_map_used: Any,
    p_series_used: Any,
    map_before_key: str = "p_map_before",
    map_after_key: str = "p_map_after",
    series_before_key: str = "p_series_before",
    series_after_key: str = "p_series_after",
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "applied": False,
        "reason": f"error:{type(error).__name__}" if isinstance(error, Exception) else f"error:{error!s}",
        "error": str(error),
        str(map_before_key): _opt_float(p_map_used),
        str(series_before_key): _opt_float(p_series_used),
        str(map_after_key): None,
        str(series_after_key): None,
    }


def apply_live_stage_result(
    *,
    out: Mapping[str, Any] | None,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    stage_name: str,
    trace_pipeline_steps: List[Dict[str, Any]],
    prob_win_series_fn: Callable[[float, int], float],
    session_state: MutableMapping[str, Any] | None = None,
    source_key: str = "method",
    source_default: str = "",
    map_before_key: str = "p_map_before",
    map_after_key: str = "p_map_after",
    series_before_key: str = "p_series_before",
    series_after_key: str = "p_series_after",
    p_after_key: str = "p_after",
    out_series_key: str | None = None,
    apply_confidence_governor_fn: Callable[..., Mapping[str, Any]] | None = None,
    governor_enabled: bool = True,
    governor_budget_pp: float = 8.0,
    governor_stage_name: str | None = None,
) -> Dict[str, Any]:
    o: Dict[str, Any] = dict(out or {})
    o.setdefault("enabled", True)

    p_before = _finite_or_nan(p_map_used)
    ps_before = _finite_or_nan(p_series_used)

    p_after_candidate = _finite_or_nan(o.get(str(p_after_key), float("nan")))
    applied_raw = bool(o.get("applied", False)) and math.isfinite(p_after_candidate)
    gov_out: Dict[str, Any] = {}
    if applied_raw and callable(apply_confidence_governor_fn) and math.isfinite(p_before):
        try:
            gv = apply_confidence_governor_fn(
                stage=str(governor_stage_name or stage_name),
                p_before=float(p_before),
                p_candidate=float(p_after_candidate),
                session_state=(session_state if isinstance(session_state, MutableMapping) else None),
                enabled=bool(governor_enabled),
                max_total_shrink_pp=float(governor_budget_pp),
                reset=False,
            )
            gov_out = dict(gv) if isinstance(gv, Mapping) else {}
            p_gov = _finite_or_nan(gov_out.get("p_out", float("nan")))
            if math.isfinite(p_gov):
                p_after_candidate = float(p_gov)
        except Exception:
            gov_out = {}
    if gov_out:
        o["governor"] = {
            "enabled": bool(gov_out.get("enabled", False)),
            "limited": bool(gov_out.get("limited", False)),
            "requested_shrink_pp": gov_out.get("requested_shrink_pp"),
            "applied_shrink_pp": gov_out.get("applied_shrink_pp"),
            "used_shrink_pp": gov_out.get("used_shrink_pp"),
            "remaining_shrink_pp": gov_out.get("remaining_shrink_pp"),
        }
    applied = bool(applied_raw) and (
        (not math.isfinite(p_before)) or (abs(float(p_after_candidate) - float(p_before)) > 1e-12)
    )

    p_new = float(p_before)
    ps_new = float(ps_before)
    if applied:
        p_new = float(p_after_candidate)
        if out_series_key is not None:
            ps_out = _finite_or_nan(o.get(str(out_series_key), float("nan")))
            if math.isfinite(ps_out):
                ps_new = float(ps_out)
            else:
                try:
                    ps_new = float(prob_win_series_fn(float(p_new), int(bo)))
                except Exception:
                    ps_new = float("nan")
        else:
            try:
                ps_new = float(prob_win_series_fn(float(p_new), int(bo)))
            except Exception:
                ps_new = float("nan")

        if isinstance(session_state, MutableMapping):
            try:
                session_state["_p_map_used"] = float(p_new)
                session_state["_p_series_used"] = float(ps_new)
            except Exception:
                pass

        o[str(map_before_key)] = _opt_float(p_before)
        o[str(map_after_key)] = _opt_float(p_new)
        o[str(series_before_key)] = _opt_float(ps_before)
        o[str(series_after_key)] = _opt_float(ps_new)
        o["applied"] = True
        if math.isfinite(p_before) and math.isfinite(p_new):
            o["diff_pp"] = float(100.0 * (float(p_new) - float(p_before)))
        if math.isfinite(ps_before) and math.isfinite(ps_new):
            o["diff_pp_series"] = float(100.0 * (float(ps_new) - float(ps_before)))

        gov_trace = o.get("governor") if isinstance(o.get("governor"), Mapping) else {}
        append_pipeline_step(
            steps=trace_pipeline_steps,
            stage=str(stage_name),
            before=p_before,
            after=p_new,
            applied=True,
            enabled=True,
            source=str(o.get(str(source_key)) or str(source_default)),
            governor_limited=bool(gov_trace.get("limited", False)),
            governor_applied_shrink_pp=(
                float(gov_trace.get("applied_shrink_pp"))
                if gov_trace.get("applied_shrink_pp") is not None
                else None
            ),
        )
    else:
        o["applied"] = False

    return {
        "out": o,
        "p_map_used": float(p_new) if math.isfinite(p_new) else float("nan"),
        "p_series_used": float(ps_new) if math.isfinite(ps_new) else float("nan"),
    }
