"""Unified confidence governor for shrink-style probability adjustments.

This module centralizes shrink budgeting so multiple uncertainty guards do not
over-shrink the same matchup in sequence.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping

from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import safe_float as _f


def _edge_pp(p: float) -> float:
    return float(abs(float(p) - 0.5) * 100.0)


def _empty_state(*, enabled: bool, max_total_shrink_pp: float) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "max_total_shrink_pp": float(max(0.0, max_total_shrink_pp)),
        "used_shrink_pp": 0.0,
        "stages": [],
    }


def apply_confidence_governor(
    *,
    stage: str,
    p_before: Any,
    p_candidate: Any,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    enabled: bool,
    max_total_shrink_pp: float,
    reset: bool = False,
    state_key: str = "_ml_confidence_governor",
    stage_cap_pp: float | None = None,
) -> Dict[str, Any]:
    """Apply/track a global shrink budget across guards.

    Shrink is defined as reduction in edge vs 50% (in percentage points).
    Moves away from 50% do not consume budget and are never capped.

    stage_cap_pp: optional per-stage maximum allocation in pp. Prevents any
    single stage from monopolising the shared budget, guaranteeing headroom
    for later, higher-priority guards (e.g. marginal_trust_guard).
    """

    stw = session_state if isinstance(session_state, MutableMapping) else None
    en = bool(enabled)
    budget = float(max(0.0, _f(max_total_shrink_pp, 0.0)))
    p0 = _f(p_before, float("nan"))
    p1 = _f(p_candidate, float("nan"))
    p0_ok = math.isfinite(p0)
    p1_ok = math.isfinite(p1)

    if bool(reset):
        state = _empty_state(enabled=en, max_total_shrink_pp=budget)
        if isinstance(stw, MutableMapping):
            stw[str(state_key)] = dict(state)
        return {
            "enabled": bool(en),
            "reset": True,
            "limited": False,
            "requested_shrink_pp": 0.0,
            "applied_shrink_pp": 0.0,
            "used_shrink_pp": 0.0,
            "remaining_shrink_pp": float(budget),
            "p_out": (float(_clip_prob(p1)) if p1_ok else (float(_clip_prob(p0)) if p0_ok else None)),
            "state_key": str(state_key),
        }

    if (not p0_ok) or (not p1_ok):
        return {
            "enabled": bool(en),
            "reset": False,
            "limited": False,
            "requested_shrink_pp": None,
            "applied_shrink_pp": None,
            "used_shrink_pp": None,
            "remaining_shrink_pp": None,
            "p_out": (float(_clip_prob(p1)) if p1_ok else None),
            "state_key": str(state_key),
            "reason": "invalid_inputs",
        }

    p0 = float(_clip_prob(p0))
    p1 = float(_clip_prob(p1))
    requested = float(max(0.0, _edge_pp(p0) - _edge_pp(p1)))
    limited = False

    raw_state = (session_state or {}).get(str(state_key), {}) if isinstance(session_state, Mapping) else {}
    cur_state = dict(raw_state) if isinstance(raw_state, dict) else {}
    if (
        cur_state.get("enabled") is None
        or cur_state.get("max_total_shrink_pp") is None
        or cur_state.get("used_shrink_pp") is None
    ):
        cur_state = _empty_state(enabled=en, max_total_shrink_pp=budget)
    used_prev = float(max(0.0, _f(cur_state.get("used_shrink_pp"), 0.0)))
    max_pp = float(budget)
    if bool(en):
        max_pp = float(max(0.0, _f(cur_state.get("max_total_shrink_pp"), budget)))
        if not math.isfinite(max_pp) or max_pp <= 0.0:
            max_pp = float(budget)
        allowed = float(max(0.0, max_pp - used_prev))
        if stage_cap_pp is not None:
            cap = float(max(0.0, stage_cap_pp))
            allowed = float(min(allowed, cap))
        applied = float(min(requested, allowed))
        if requested > applied + 1e-12:
            limited = True
            edge_after = float(max(0.0, _edge_pp(p0) - applied))
            sign = 1.0 if float(p0) >= 0.5 else -1.0
            p_out = float(_clip_prob(0.5 + sign * (edge_after / 100.0)))
        else:
            p_out = float(p1)
        used_new = float(used_prev + applied)
        rem = float(max(0.0, max_pp - used_new))
    else:
        applied = float(requested)
        used_new = float(used_prev)
        rem = None
        p_out = float(p1)

    out = {
        "enabled": bool(en),
        "reset": False,
        "stage": str(stage or ""),
        "limited": bool(limited),
        "requested_shrink_pp": float(requested),
        "applied_shrink_pp": float(applied),
        "used_shrink_pp": float(used_new),
        "remaining_shrink_pp": (float(rem) if rem is not None else None),
        "max_total_shrink_pp": (float(max_pp) if bool(en) else float(budget)),
        "stage_cap_pp": (float(stage_cap_pp) if stage_cap_pp is not None else None),
        "p_out": float(p_out),
        "state_key": str(state_key),
    }

    if isinstance(stw, MutableMapping):
        stages = list(cur_state.get("stages") or [])
        stages.append(
            {
                "stage": str(stage or ""),
                "requested_shrink_pp": float(requested),
                "applied_shrink_pp": float(applied),
                "limited": bool(limited),
            }
        )
        if len(stages) > 24:
            stages = stages[-24:]
        stw[str(state_key)] = {
            "enabled": bool(en),
            "max_total_shrink_pp": (float(max_pp) if bool(en) else float(budget)),
            "used_shrink_pp": float(used_new),
            "stages": stages,
        }

    return out


def read_governor_budget_summary(
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    *,
    state_key: str = "_ml_confidence_governor",
) -> Dict[str, Any]:
    """Return current budget consumption state for tracing/display.

    Reads the running governor state without modifying it.
    """
    raw = (session_state or {}).get(str(state_key), {}) if isinstance(session_state, Mapping) else {}
    if not isinstance(raw, dict) or not raw:
        return {"available": False, "budget_exhausted": False}
    max_pp = _f(raw.get("max_total_shrink_pp"), float("nan"))
    used_pp = _f(raw.get("used_shrink_pp"), float("nan"))
    enabled = bool(raw.get("enabled", False))
    remaining = (
        float(max(0.0, max_pp - used_pp))
        if math.isfinite(max_pp) and math.isfinite(used_pp)
        else None
    )
    exhausted = bool(
        enabled
        and math.isfinite(max_pp)
        and math.isfinite(used_pp)
        and used_pp >= max_pp - 1e-9
    )
    stages = list(raw.get("stages") or [])
    n_limited = sum(1 for s in stages if bool(s.get("limited", False)))
    return {
        "available": True,
        "enabled": enabled,
        "max_total_shrink_pp": (float(max_pp) if math.isfinite(max_pp) else None),
        "used_shrink_pp": (float(used_pp) if math.isfinite(used_pp) else None),
        "remaining_shrink_pp": remaining,
        "budget_exhausted": exhausted,
        "n_stages_applied": len(stages),
        "n_stages_limited": n_limited,
        "stages": stages,
        "state_key": str(state_key),
    }
