"""Summarise the quality of a fitted params configuration.

Pure functions — no I/O, no Streamlit. Consumes the dict stored in
params_fitted.json and produces a structured quality report suitable for
display, monitoring, or programmatic gating.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError):
        return default


def _gate_info(cfg: Any) -> Tuple[str, List[str]]:
    """Extract (decision, reasons) from a gate sub-dict."""
    if not isinstance(cfg, dict):
        return "unknown", []
    gate = cfg.get("gate") if isinstance(cfg.get("gate"), dict) else {}
    decision = str(gate.get("decision") or "unknown").strip().lower()
    reasons = [str(r) for r in (gate.get("reasons") or []) if r]
    return decision, reasons


# ---------------------------------------------------------------------------
# Component-level summaries
# ---------------------------------------------------------------------------

def _summarise_overall_fit(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    baseline = fit_cfg.get("baseline") or {}
    best = fit_cfg.get("best") or {}
    n_total = int(_f(best.get("n_total"), 0) or _f(baseline.get("n_total"), 0))
    baseline_ll = _f(baseline.get("logloss"))
    best_ll = _f(best.get("logloss"))
    baseline_br = _f(baseline.get("brier"))
    best_br = _f(best.get("brier"))
    gain_ll = (baseline_ll - best_ll) if math.isfinite(baseline_ll) and math.isfinite(best_ll) else float("nan")
    gain_br = (baseline_br - best_br) if math.isfinite(baseline_br) and math.isfinite(best_br) else float("nan")
    return {
        "n_total": n_total,
        "baseline_logloss": (float(baseline_ll) if math.isfinite(baseline_ll) else None),
        "best_logloss": (float(best_ll) if math.isfinite(best_ll) else None),
        "gain_logloss": (float(gain_ll) if math.isfinite(gain_ll) else None),
        "baseline_brier": (float(baseline_br) if math.isfinite(baseline_br) else None),
        "best_brier": (float(best_br) if math.isfinite(best_br) else None),
        "gain_brier": (float(gain_br) if math.isfinite(gain_br) else None),
        "team_scale": _f(best.get("team_scale"), 1.0),
        "players_scale": _f(best.get("players_scale"), 1.0),
    }


def _summarise_form(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    best_form = fit_cfg.get("best_form") or {}
    enabled = bool(best_form.get("enabled", False))
    gain = _f(best_form.get("improvement_logloss_vs_no_form"))
    return {
        "enabled": enabled,
        "gain_logloss_vs_no_form": (float(gain) if math.isfinite(gain) else None),
        "n_total": int(_f(best_form.get("n_total"), 0)),
        "settings": dict(best_form.get("settings") or {}),
    }


def _summarise_players_context(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pcf = fit_cfg.get("players_context_fit") or {}
    decision, reasons = _gate_info(pcf)
    settings = dict(pcf.get("settings") or {})
    meta = dict(pcf.get("meta") or {})
    n = int(_f((pcf.get("best_on") or {}).get("n_total"), 0))
    return {
        "gate_decision": decision,
        "gate_reasons": reasons,
        "n_total": n,
        "settings": settings,
        "cache_entries": int(_f(meta.get("cache_entries"), 0)),
        "candidates_tested": int(_f(meta.get("candidates_tested"), 0)),
    }


def _summarise_fused_calibration(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fc = fit_cfg.get("fused_post_calibration") or {}
    decision, reasons = _gate_info(fc)
    n = int(_f((fc.get("meta") or {}).get("n_total"), _f((fc.get("gate") or {}).get("n_total"), 0)))
    return {
        "gate_decision": decision,
        "gate_reasons": reasons,
        "n_total": n,
        "global_model": dict(fc.get("promoted_model_global") or {}),
    }


def _summarise_microseg(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ms = fit_cfg.get("microseg_calibration") or {}
    table = ms.get("table") or {}
    shrink_k = _f(ms.get("shrink_k"), float("nan"))
    n_pred_rows = int(_f(ms.get("pred_rows_n"), 0))
    n_segments = len(table) if isinstance(table, dict) else 0
    # Detect global (wildcard) segment to know if a fallback exists
    has_global_fallback = isinstance(table, dict) and "*|*|*|*|*" in table
    return {
        "n_pred_rows": n_pred_rows,
        "n_segments": n_segments,
        "shrink_k": (float(shrink_k) if math.isfinite(shrink_k) else None),
        "has_global_fallback": has_global_fallback,
        "source": str(ms.get("source") or ""),
    }


def _summarise_series_gamma(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sg_global = fit_cfg.get("series_gamma_global") or {}
    n_leagues = len(fit_cfg.get("series_gamma_by_league") or {})
    n_macros = len(fit_cfg.get("series_gamma_by_macro") or {})
    return {
        "global_gamma": _f(sg_global.get("gamma"), float("nan")),
        "n_leagues_tuned": n_leagues,
        "n_macros_tuned": n_macros,
    }


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _quality_score(
    components: Dict[str, Any],
    overall: Dict[str, Any],
) -> Tuple[float, List[str]]:
    """Heuristic 0–100 quality score with explanation list."""
    score = 100.0
    flags: List[str] = []

    # Overall model improvement
    gain_ll = _f(overall.get("gain_logloss"), float("nan"))
    if not math.isfinite(gain_ll):
        score -= 10.0
        flags.append("overall_gain_not_evaluated")
    elif gain_ll < 0.0002:
        score -= 8.0
        flags.append(f"overall_gain_very_small:{gain_ll:.4f}")
    elif gain_ll < 0.0010:
        score -= 3.0
        flags.append(f"overall_gain_small:{gain_ll:.4f}")

    n_total = int(overall.get("n_total") or 0)
    if n_total < 300:
        score -= 15.0
        flags.append(f"low_n_total:{n_total}")
    elif n_total < 600:
        score -= 5.0
        flags.append(f"moderate_n_total:{n_total}")

    # Players context gate
    pc = components.get("players_context") or {}
    pc_decision = str(pc.get("gate_decision") or "unknown")
    if pc_decision == "shadow_only":
        reasons = pc.get("gate_reasons") or []
        if any("regression" in r for r in reasons):
            score -= 12.0
            flags.append(f"players_context_shadow_regression:{','.join(reasons)}")
        else:
            score -= 5.0
            flags.append(f"players_context_shadow_only:{','.join(reasons)}")
    elif pc_decision in ("unknown", "not_evaluated"):
        score -= 3.0
        flags.append("players_context_not_evaluated")

    # Fused calibration gate
    fc = components.get("fused_calibration") or {}
    fc_decision = str(fc.get("gate_decision") or "unknown")
    if fc_decision == "shadow_only":
        reasons = fc.get("gate_reasons") or []
        if any("regression" in r for r in reasons):
            score -= 8.0
            flags.append(f"fused_cal_shadow_regression:{','.join(reasons)}")
        elif "not_evaluated" not in reasons:
            score -= 3.0
            flags.append(f"fused_cal_shadow:{','.join(reasons)}")

    # Microseg coverage
    ms = components.get("microseg") or {}
    if int(ms.get("n_segments") or 0) == 0:
        score -= 5.0
        flags.append("microseg_no_segments")
    elif not ms.get("has_global_fallback"):
        score -= 2.0
        flags.append("microseg_no_global_fallback")

    return float(max(0.0, min(100.0, score))), flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarise_fit_quality(fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured quality report for a fitted params dict.

    Keys:
    - quality_score (float 0-100): composite heuristic score
    - quality_flags (List[str]): specific deductions with reasons
    - generated_at (str): from fit_cfg
    - n_samples (int): total training samples
    - components (dict): per-component quality summaries
    - overall (dict): baseline/best logloss, brier, gains
    """
    if not isinstance(fit_cfg, dict):
        return {
            "quality_score": 0.0,
            "quality_flags": ["invalid_fit_cfg"],
            "generated_at": "",
            "n_samples": 0,
            "components": {},
            "overall": {},
        }

    overall = _summarise_overall_fit(fit_cfg)
    components = {
        "form": _summarise_form(fit_cfg),
        "players_context": _summarise_players_context(fit_cfg),
        "fused_calibration": _summarise_fused_calibration(fit_cfg),
        "microseg": _summarise_microseg(fit_cfg),
        "series_gamma": _summarise_series_gamma(fit_cfg),
    }

    score, flags = _quality_score(components, overall)

    return {
        "quality_score": float(score),
        "quality_flags": flags,
        "generated_at": str(fit_cfg.get("generated_at") or ""),
        "n_samples": int(_f(fit_cfg.get("samples_total"), 0)),
        "min_train_games": int(_f(fit_cfg.get("min_train_games"), 0)),
        "speed_profile": str(fit_cfg.get("speed_profile") or ""),
        "components": components,
        "overall": overall,
    }


__all__ = ["summarise_fit_quality"]
