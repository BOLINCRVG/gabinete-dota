"""Runtime league-aware tuning helpers for ML safeguards.

These helpers keep matchup-level auto-tuning deterministic and auditable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from app_core.prob_math_core import safe_float as _f
from mlcore.utils import clip as _clip
from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)


def _norm_key(v: Any) -> str:
    return str(v or "").strip().upper()


def _is_explicit_interregional_league(v: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(v))


def _macro_key(league: str) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _row_by_league(table: Any, league_hint: str) -> Tuple[Dict[str, Any], str]:
    lg = _norm_key(league_hint)
    if not isinstance(table, Mapping) or not lg:
        return {}, ""
    if isinstance(table.get(lg), Mapping):
        return dict(table.get(lg) or {}), f"league:{lg}"
    mk = _macro_key(lg)
    if isinstance(table.get(mk), Mapping):
        return dict(table.get(mk) or {}), f"macro:{mk}"
    return {}, ""


def _confidence_band_from_baseline(
    *,
    league_hint: str,
    confidence_baseline: Optional[Mapping[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    lg = _norm_key(league_hint)
    row: Dict[str, Any] = {}
    src = ""
    if isinstance(confidence_baseline, Mapping):
        by_lg = confidence_baseline.get("by_league")
        if isinstance(by_lg, Mapping):
            row, src = _row_by_league(by_lg, lg)

    qb = str((row or {}).get("quality_band", "") or "").strip().lower()
    # Normalize baseline quality bands to an operational 4-band contract.
    if qb in {"strong", "excellent"}:
        band = "strong"
    elif qb in {"good", "core"}:
        band = "good"
    elif qb in {"ok", "medium"}:
        band = "ok"
    elif qb in {"low", "low_sample", "warn", "weak"}:
        band = "low_sample"
    else:
        top = {"LCK", "LPL", "LEC", "LCS"}
        core = {"CBLOL", "AMERICAS", "LCP", "PCS", "VCS", "LLA", "MSI", "WORLDS"}
        if lg in top:
            band = "strong"
            src = "heuristic:top"
        elif lg in core:
            band = "good"
            src = "heuristic:core"
        elif lg:
            band = "ok"
            src = "heuristic:other"
        else:
            band = "ok"
            src = "heuristic:unknown"

    evidence_n = int(
        max(
            0,
            _f((row or {}).get("lv_n"), 0.0),
            _f((row or {}).get("wf_n"), 0.0),
        )
    )
    info = {
        "league": lg,
        "band": str(band),
        "band_source": str(src or ""),
        "evidence_n": int(evidence_n),
        "row": dict(row or {}),
    }
    return str(band), info


def compute_confidence_governor_budget_pp(
    *,
    base_budget_pp: Any,
    league_hint: str,
    enabled: bool = True,
    use_fitted_by_league: bool = True,
    params_fit: Optional[Mapping[str, Any]] = None,
    confidence_baseline: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base = _clip(_f(base_budget_pp, 8.0), 0.0, 30.0)
    band, band_info = _confidence_band_from_baseline(
        league_hint=str(league_hint or ""),
        confidence_baseline=(confidence_baseline if isinstance(confidence_baseline, Mapping) else None),
    )
    mult = {
        "strong": 0.72,
        "good": 0.88,
        "ok": 1.00,
        "low_sample": 1.22,
    }.get(str(band), 1.00)

    evidence_n = int(band_info.get("evidence_n", 0) or 0)
    if evidence_n > 0:
        if evidence_n < 180:
            mult *= 1.08
        elif evidence_n >= 1200:
            mult *= 0.96

    fit_src = ""
    fit_n = 0
    fit_imp = None
    if bool(use_fitted_by_league) and isinstance(params_fit, Mapping):
        by_lg = params_fit.get("best_by_league")
        if isinstance(by_lg, Mapping):
            row, fit_src = _row_by_league(by_lg, str(league_hint or ""))
            fit_n = int(max(0, _f((row or {}).get("n_total"), 0.0)))
            fit_imp_f = _f((row or {}).get("improvement_logloss_vs_baseline"), float("nan"))
            fit_imp = float(fit_imp_f) if math.isfinite(fit_imp_f) else None
            if fit_n >= 120 and fit_imp is not None:
                if fit_imp >= 0.008:
                    mult *= 0.90
                elif fit_imp >= 0.003:
                    mult *= 0.95
                elif fit_imp <= -0.004:
                    mult *= 1.14
                elif fit_imp <= -0.001:
                    mult *= 1.06

    budget_pp = _clip(base * float(mult), 3.0, 18.0) if bool(enabled) else float(base)
    return {
        "enabled": bool(enabled),
        "league": _norm_key(league_hint),
        "base_budget_pp": float(base),
        "budget_pp": float(budget_pp),
        "multiplier": float(mult),
        "band": str(band),
        "band_info": dict(band_info),
        "use_fitted_by_league": bool(use_fitted_by_league),
        "fit_source": str(fit_src or ""),
        "fit_n": int(fit_n),
        "fit_improvement_logloss_vs_baseline": (float(fit_imp) if fit_imp is not None else None),
    }


def compute_series_coherence_max_dev_pp(
    *,
    bo: Any,
    base_bo3_pp: Any,
    base_bo5_pp: Any,
    league_hint: str,
    enabled: bool = True,
    use_fitted_by_league: bool = True,
    params_fit: Optional[Mapping[str, Any]] = None,
    confidence_baseline: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        bo_i = int(float(bo))
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    if bo_i not in (3, 5):
        return {
            "enabled": bool(enabled),
            "bo": int(bo_i),
            "league": _norm_key(league_hint),
            "max_dev_pp": None,
            "reason": "bo1_noop",
        }

    base = _f(base_bo5_pp if bo_i == 5 else base_bo3_pp, 20.0 if bo_i == 5 else 16.0)
    base = _clip(base, 2.0, 35.0)
    band, band_info = _confidence_band_from_baseline(
        league_hint=str(league_hint or ""),
        confidence_baseline=(confidence_baseline if isinstance(confidence_baseline, Mapping) else None),
    )
    mult = {
        "strong": 1.12,
        "good": 1.03,
        "ok": 0.95,
        "low_sample": 0.82,
    }.get(str(band), 1.00)

    evidence_n = int(band_info.get("evidence_n", 0) or 0)
    if evidence_n > 0:
        if evidence_n < 180:
            mult *= 0.94
        elif evidence_n >= 1200:
            mult *= 1.04

    fit_src = ""
    fit_n = 0
    fit_imp = None
    if bool(use_fitted_by_league) and isinstance(params_fit, Mapping):
        by_lg = params_fit.get("series_gamma_by_league")
        if isinstance(by_lg, Mapping):
            row, fit_src = _row_by_league(by_lg, str(league_hint or ""))
            bo_node = row.get(f"bo{int(bo_i)}") if isinstance(row.get(f"bo{int(bo_i)}"), Mapping) else {}
            fit_n = int(max(0, _f((bo_node or {}).get("n"), _f((row or {}).get("rows"), 0.0))))
            fit_imp_f = _f((bo_node or {}).get("improvement_logloss_vs_gamma1"), float("nan"))
            fit_imp = float(fit_imp_f) if math.isfinite(fit_imp_f) else None
            if fit_n >= 80 and fit_imp is not None:
                if fit_imp >= 0.010:
                    mult *= 1.08
                elif fit_imp >= 0.003:
                    mult *= 1.03
                elif fit_imp <= -0.008:
                    mult *= 0.90
                elif fit_imp <= -0.002:
                    mult *= 0.95

    cap = float(base * float(mult)) if bool(enabled) else float(base)
    if bo_i == 3:
        cap = _clip(cap, 8.0, 28.0)
    else:
        cap = _clip(cap, 10.0, 32.0)
    return {
        "enabled": bool(enabled),
        "bo": int(bo_i),
        "league": _norm_key(league_hint),
        "base_max_dev_pp": float(base),
        "max_dev_pp": float(cap),
        "multiplier": float(mult),
        "band": str(band),
        "band_info": dict(band_info),
        "use_fitted_by_league": bool(use_fitted_by_league),
        "fit_source": str(fit_src or ""),
        "fit_n": int(fit_n),
        "fit_improvement_logloss_vs_gamma1": (float(fit_imp) if fit_imp is not None else None),
        "reason": "ok",
    }
