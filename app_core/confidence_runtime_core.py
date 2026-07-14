# -*- coding: utf-8 -*-
"""Runtime confidence governor, series gamma, and coherence budget functions."""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

from mlcore.league_context import league_macro_key as _shared_league_macro_key
from app_core.ml_session_runtime_core import get_fit_runtime_snapshot as _get_fit_runtime_snapshot
from app_core.runtime_tuning_core import (
    compute_confidence_governor_budget_pp as _compute_confidence_governor_budget_pp,
    compute_series_coherence_max_dev_pp as _compute_series_coherence_max_dev_pp,
)


def _league_macro_key(league: Any) -> str:
    return str(_shared_league_macro_key(league, unknown="OTHER"))


def load_confidence_baseline_runtime(*, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    _ML_CONFIDENCE_BASELINE_PATH = ctx.get("_ML_CONFIDENCE_BASELINE_PATH")
    _load_ml_confidence_baseline_cached = ctx.get("_load_ml_confidence_baseline_cached")
    _file_sig = ctx.get("_file_sig")
    try:
        if _ML_CONFIDENCE_BASELINE_PATH.exists():
            out = _load_ml_confidence_baseline_cached(_file_sig(str(_ML_CONFIDENCE_BASELINE_PATH))) or {}
            return out if isinstance(out, dict) else {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


def series_gamma_for_league(league_hint: str, bo: int, *, st: Any, ctx: Mapping[str, Any]) -> tuple:
    try:
        b = int(bo)
    except (TypeError, ValueError):
        b = 1
    if b not in (3, 5):
        return 1.0, ""
    k = f"bo{b}"
    _fit_rt = _get_fit_runtime_snapshot(st.session_state)
    cfg = _fit_rt.get("_fit_series_gamma_cfg") if isinstance(_fit_rt.get("_fit_series_gamma_cfg"), dict) else {}
    lg = str(league_hint or "").strip()

    by_lg = cfg.get("by_league") if isinstance(cfg, dict) else {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            rr = row.get(k)
            if isinstance(rr, dict) and rr.get("gamma") is not None:
                try:
                    return max(0.25, float(rr.get("gamma"))), f"league:{lg}"
                except (TypeError, ValueError):
                    pass
    by_mk = cfg.get("by_macro") if isinstance(cfg, dict) else {}
    mk = _league_macro_key(lg)
    if isinstance(by_mk, dict):
        row = by_mk.get(mk)
        if isinstance(row, dict):
            rr = row.get(k)
            if isinstance(rr, dict) and rr.get("gamma") is not None:
                try:
                    return max(0.25, float(rr.get("gamma"))), f"macro:{mk}"
                except (TypeError, ValueError):
                    pass
    gg = cfg.get("global") if isinstance(cfg, dict) else {}
    if isinstance(gg, dict):
        rr = gg.get(k)
        if isinstance(rr, dict) and rr.get("gamma") is not None:
            try:
                return max(0.25, float(rr.get("gamma"))), "global"
            except (TypeError, ValueError):
                pass
    return 1.0, ""


def runtime_confidence_governor_budget_for_league(
    league_hint: str,
    confidence_baseline: Optional[Dict[str, Any]] = None,
    *,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    _fit_cfg = ctx.get("_fit_cfg", {})
    lg = str(league_hint or "").strip().upper()
    base_pp = float(st.session_state.get("ml_confidence_governor_max_total_shrink_pp", 8.0) or 8.0)
    enabled = bool(st.session_state.get("ml_confidence_governor_enabled", True))
    use_fit = bool(st.session_state.get("ml_confidence_governor_use_fitted_by_league", True))
    cb = confidence_baseline if isinstance(confidence_baseline, dict) else load_confidence_baseline_runtime(ctx=ctx)
    out = _compute_confidence_governor_budget_pp(
        base_budget_pp=float(base_pp),
        league_hint=str(lg),
        enabled=bool(enabled),
        use_fitted_by_league=bool(use_fit),
        params_fit=(_fit_cfg if isinstance(_fit_cfg, dict) else {}),
        confidence_baseline=(cb if isinstance(cb, dict) else {}),
    )
    return out if isinstance(out, dict) else {
        "enabled": bool(enabled),
        "league": str(lg),
        "base_budget_pp": float(base_pp),
        "budget_pp": float(base_pp),
        "reason": "fallback",
    }


def runtime_series_coherence_post_cap_for_league(
    *,
    league_hint: str,
    bo: int,
    confidence_baseline: Optional[Dict[str, Any]] = None,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    _fit_cfg = ctx.get("_fit_cfg", {})
    lg = str(league_hint or "").strip().upper()
    use_fit = bool(st.session_state.get("ml_series_coherence_post_use_fitted_by_league", True))
    cb = confidence_baseline if isinstance(confidence_baseline, dict) else load_confidence_baseline_runtime(ctx=ctx)
    out = _compute_series_coherence_max_dev_pp(
        bo=int(bo),
        base_bo3_pp=float(st.session_state.get("ml_series_coherence_post_max_dev_pp_bo3", 16.0) or 16.0),
        base_bo5_pp=float(st.session_state.get("ml_series_coherence_post_max_dev_pp_bo5", 20.0) or 20.0),
        league_hint=str(lg),
        enabled=True,
        use_fitted_by_league=bool(use_fit),
        params_fit=(_fit_cfg if isinstance(_fit_cfg, dict) else {}),
        confidence_baseline=(cb if isinstance(cb, dict) else {}),
    )
    return out if isinstance(out, dict) else {
        "enabled": True,
        "bo": int(bo),
        "league": str(lg),
        "max_dev_pp": None,
        "reason": "fallback",
    }
