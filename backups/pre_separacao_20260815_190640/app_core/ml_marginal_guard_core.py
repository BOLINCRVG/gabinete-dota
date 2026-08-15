"""Post-layer marginal trust guard orchestration."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, MutableMapping, Tuple

from app_core.prob_math_core import safe_float as _f


def run_marginal_trust_guard(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    ml_info: Dict[str, Any] | None,
    league_hint: str,
    trace_n_team_eff: Any,
    lv_last_report: Any,
    ml_trace_last: Any,
    confidence_baseline: Dict[str, Any] | None,
    session_state: Mapping[str, Any] | MutableMapping[str, Any],
    ml_confidence_fn: Callable[..., Dict[str, Any]],
    marginal_adjust_fn: Callable[..., Tuple[float, Dict[str, Any]]],
    prob_win_series_fn: Callable[[float, int], float],
    apply_confidence_governor_fn: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    p_map = _f(p_map_used, float("nan"))
    p_series = _f(p_series_used, float("nan"))
    bo_i = int(bo)
    trace_marginal_trust_info: Dict[str, Any] = {}
    gov_enabled = bool((session_state or {}).get("ml_confidence_governor_enabled", True))
    gov_budget_pp = float(
        (session_state or {}).get(
            "_ml_confidence_governor_budget_runtime_pp",
            (session_state or {}).get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        )
        or 8.0
    )

    p_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
    p_after = float(p_before)
    nte = _f(trace_n_team_eff, float("nan"))

    conf = ml_confidence_fn(
        lv_last_report,
        league_hint=str(league_hint or ""),
        n_team_eff=(float(nte) if math.isfinite(nte) else float("nan")),
        ml_trace=ml_trace_last,
        confidence_baseline=(confidence_baseline if isinstance(confidence_baseline, dict) else {}),
    )
    score = _f((conf or {}).get("score"), float("nan"))
    p_mt, mt_i = marginal_adjust_fn(
        float(p_map),
        ml_info if isinstance(ml_info, dict) else None,
        league=str(league_hint or ""),
        enabled=bool((session_state or {}).get("ml_marginal_trust_enabled", True)),
        strength=float((session_state or {}).get("ml_marginal_trust_strength", 0.70) or 0.70),
        prior_games=float((session_state or {}).get("ml_marginal_trust_prior_games", 28.0) or 28.0),
        max_shrink=float((session_state or {}).get("ml_marginal_trust_max_shrink", 0.72) or 0.72),
        cap_prob=float((session_state or {}).get("ml_marginal_trust_cap_prob", 0.84) or 0.84),
        confidence_score=(score if math.isfinite(score) else None),
        score_soft=float((session_state or {}).get("ml_marginal_trust_score_soft", 68.0) or 68.0),
        score_hard=float((session_state or {}).get("ml_marginal_trust_score_hard", 46.0) or 46.0),
        exclude_leagues=(session_state or {}).get("ml_marginal_trust_exclude_leagues", ""),
        band_tuning=bool((session_state or {}).get("ml_marginal_trust_band_tuning", True)),
        band_coeffs=(
            (session_state or {}).get("ml_marginal_trust_band_coeffs", {})
            if isinstance((session_state or {}).get("ml_marginal_trust_band_coeffs", {}), dict)
            else {}
        ),
    )
    gov_mt: Dict[str, Any] = {}
    if math.isfinite(_f(p_mt)):
        if callable(apply_confidence_governor_fn):
            try:
                gov_mt = apply_confidence_governor_fn(
                    stage="marginal_trust_guard",
                    p_before=float(p_map),
                    p_candidate=float(p_mt),
                    session_state=(session_state if isinstance(session_state, MutableMapping) else None),
                    enabled=bool(gov_enabled),
                    max_total_shrink_pp=float(gov_budget_pp),
                    reset=False,
                )
            except Exception:
                gov_mt = {}
        p_mt_used = (
            float(gov_mt.get("p_out"))
            if isinstance(gov_mt, dict) and math.isfinite(_f(gov_mt.get("p_out")))
            else float(p_mt)
        )
        p_mt = float(p_mt_used)
        p_map = float(p_mt)
        p_after = float(p_mt)
        try:
            p_series = float(prob_win_series_fn(float(p_map), int(bo_i)))
        except Exception:
            p_series = float("nan")
    if isinstance(mt_i, dict):
        mt_i["confidence_score"] = float(score) if math.isfinite(score) else None
        mt_i["confidence_grade"] = str((conf or {}).get("grade", "") or "")
        mt_i["confidence_source"] = str((conf or {}).get("source", "") or "")
        if isinstance(gov_mt, dict) and gov_mt:
            mt_i["governor"] = {
                "enabled": bool(gov_mt.get("enabled", False)),
                "limited": bool(gov_mt.get("limited", False)),
                "requested_shrink_pp": gov_mt.get("requested_shrink_pp"),
                "applied_shrink_pp": gov_mt.get("applied_shrink_pp"),
                "used_shrink_pp": gov_mt.get("used_shrink_pp"),
                "remaining_shrink_pp": gov_mt.get("remaining_shrink_pp"),
            }
        trace_marginal_trust_info = dict(mt_i)

    step = {
        "stage": "marginal_trust_guard",
        "before": (float(p_before) if math.isfinite(float(p_before)) else None),
        "after": (float(p_after) if math.isfinite(float(p_after)) else None),
        "applied": bool(trace_marginal_trust_info.get("applied", False))
        or (
            math.isfinite(float(p_before))
            and math.isfinite(float(p_after))
            and abs(float(p_after) - float(p_before)) > 1e-12
        ),
        "enabled": bool(trace_marginal_trust_info.get("enabled", False)),
        "source": "marginal_trust_guard",
        "league": str(trace_marginal_trust_info.get("league", "") or ""),
        "tier": str(trace_marginal_trust_info.get("tier", "") or ""),
        "confidence_score": float(trace_marginal_trust_info.get("confidence_score"))
        if trace_marginal_trust_info.get("confidence_score") is not None
        else None,
        "shrink": float(trace_marginal_trust_info.get("shrink"))
        if trace_marginal_trust_info.get("shrink") is not None
        else None,
        "cap_hi": float(trace_marginal_trust_info.get("cap_hi"))
        if trace_marginal_trust_info.get("cap_hi") is not None
        else None,
        "reason": str(trace_marginal_trust_info.get("reason", "") or ""),
        "governor_limited": bool(((trace_marginal_trust_info or {}).get("governor") or {}).get("limited", False)),
        "governor_applied_shrink_pp": (
            float(((trace_marginal_trust_info or {}).get("governor") or {}).get("applied_shrink_pp"))
            if ((trace_marginal_trust_info or {}).get("governor") or {}).get("applied_shrink_pp") is not None
            else None
        ),
    }

    caption_applied = ""
    caption_not_applied = ""
    if bool(trace_marginal_trust_info.get("applied", False)):
        caption_applied = (
            "Guardião marginal: "
            f"liga={trace_marginal_trust_info.get('league', '-') or '-'!s} | "
            f"score={float(trace_marginal_trust_info.get('confidence_score') or 0.0):.1f} | "
            f"shrink={(100.0*float(trace_marginal_trust_info.get('shrink', 0.0) or 0.0)):.1f}% | "
            f"cap={(100.0*float(trace_marginal_trust_info.get('cap_hi', 0.5) or 0.5)):.1f}% | "
            f"p_final={float(trace_marginal_trust_info.get('p_out', p_map) or p_map):.4f}"
        )
    elif str(trace_marginal_trust_info.get("reason", "")) in {"excluded_league", "gate_strong_band", "disabled"}:
        caption_not_applied = (
            "Guardião marginal: "
            f"não aplicado ({trace_marginal_trust_info.get('reason', '-')!s}) "
            f"na liga {trace_marginal_trust_info.get('league', '-') or '-'!s}."
        )

    return {
        "p_map_used": float(p_map) if math.isfinite(float(p_map)) else float("nan"),
        "p_series_used": float(p_series) if math.isfinite(float(p_series)) else float("nan"),
        "trace_marginal_trust_info": trace_marginal_trust_info,
        "pipeline_step": step,
        "caption_applied": str(caption_applied),
        "caption_not_applied": str(caption_not_applied),
    }
