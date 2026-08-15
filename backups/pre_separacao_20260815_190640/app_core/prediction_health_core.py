"""Runtime health checks for live ML prediction pipeline output.

Pure functions — no I/O, no side effects. Consumes the dict returned by
run_ml_post_layers and produces structured health signals for monitoring,
display, and alerting.
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


# ---------------------------------------------------------------------------
# p_map / p_series sanity
# ---------------------------------------------------------------------------

def _check_probabilities(
    p_map: float,
    p_series: float,
    bo: int,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not math.isfinite(p_map):
        errors.append("p_map_not_finite")
    elif p_map <= 0.0 or p_map >= 1.0:
        errors.append("p_map_out_of_range")
    elif p_map < 0.05 or p_map > 0.95:
        warnings.append("p_map_extreme")

    if bo in (3, 5):
        if not math.isfinite(p_series):
            errors.append("p_series_not_finite")
        elif p_series <= 0.0 or p_series >= 1.0:
            errors.append("p_series_out_of_range")
        else:
            # Series probability must be on the same side of 0.5 as p_map
            if (p_map > 0.5) != (p_series > 0.5):
                errors.append("p_series_flipped_vs_p_map")
            # Series must amplify the map edge for the favourite
            if p_map > 0.5 and p_series < p_map:
                warnings.append("p_series_weaker_than_p_map_for_favourite")
            if p_map < 0.5 and p_series > p_map:
                warnings.append("p_series_weaker_than_p_map_for_favourite")
    return warnings, errors


# ---------------------------------------------------------------------------
# Governor budget
# ---------------------------------------------------------------------------

def _check_governor(budget: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not isinstance(budget, dict) or not budget.get("available"):
        return warnings, errors
    if budget.get("budget_exhausted"):
        warnings.append("governor_budget_exhausted")
    n_limited = int(budget.get("n_stages_limited") or 0)
    if n_limited > 0:
        warnings.append(f"governor_limited_{n_limited}_stage(s)")
    remaining = _f(budget.get("remaining_shrink_pp"), float("nan"))
    max_pp = _f(budget.get("max_total_shrink_pp"), float("nan"))
    if math.isfinite(remaining) and math.isfinite(max_pp) and max_pp > 0.0:
        pct_remaining = remaining / max_pp
        if pct_remaining < 0.10:
            warnings.append("governor_less_than_10pct_remaining")
    return warnings, errors


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

def _check_series_coherence(info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not isinstance(info, dict):
        return warnings, errors
    reason = str(info.get("reason") or "")
    if reason.startswith("error:"):
        errors.append(f"series_coherence_{reason}")
    return warnings, errors


# ---------------------------------------------------------------------------
# Pipeline stage movement
# ---------------------------------------------------------------------------

def _check_stage_movements(
    steps: List[Dict[str, Any]],
    *,
    large_move_pp: float = 8.0,
) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    for step in (steps or []):
        if not isinstance(step, dict):
            continue
        before = _f(step.get("before"), float("nan"))
        after = _f(step.get("after"), float("nan"))
        if not (math.isfinite(before) and math.isfinite(after)):
            continue
        move_pp = abs(after - before) * 100.0
        if move_pp >= large_move_pp:
            stage = str(step.get("stage") or "unknown")
            warnings.append(f"large_move_{stage}_{move_pp:.1f}pp")
    return warnings, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_prediction_health(
    pipeline_out: Dict[str, Any],
    *,
    large_move_pp: float = 8.0,
) -> Dict[str, Any]:
    """Return a structured health summary for a run_ml_post_layers output dict.

    Keys in the returned dict:
    - healthy (bool): no errors detected
    - warnings (List[str]): non-fatal issues
    - errors (List[str]): fatal / data-quality failures
    - p_map_used (float or None)
    - p_series_used (float or None)
    - bo (int)
    - n_warnings (int)
    - n_errors (int)
    """
    if not isinstance(pipeline_out, dict):
        return {
            "healthy": False,
            "warnings": [],
            "errors": ["invalid_pipeline_output"],
            "n_warnings": 0,
            "n_errors": 1,
            "p_map_used": None,
            "p_series_used": None,
            "bo": 0,
        }

    p_map = _f(pipeline_out.get("p_map_used"), float("nan"))
    p_series = _f(pipeline_out.get("p_series_used"), float("nan"))
    steps = pipeline_out.get("trace_pipeline_steps") or []
    budget = pipeline_out.get("trace_governor_budget_final") or {}
    coherence_info = pipeline_out.get("trace_series_coherence_post_info") or {}

    # Try to extract bo from coherence info (it's always stored there)
    bo = int(_f(coherence_info.get("bo"), 0))

    all_warnings: List[str] = []
    all_errors: List[str] = []

    w, e = _check_probabilities(p_map, p_series, bo)
    all_warnings.extend(w)
    all_errors.extend(e)

    w, e = _check_governor(budget)
    all_warnings.extend(w)
    all_errors.extend(e)

    w, e = _check_series_coherence(coherence_info)
    all_warnings.extend(w)
    all_errors.extend(e)

    w, e = _check_stage_movements(steps, large_move_pp=large_move_pp)
    all_warnings.extend(w)
    all_errors.extend(e)

    return {
        "healthy": len(all_errors) == 0,
        "warnings": all_warnings,
        "errors": all_errors,
        "n_warnings": len(all_warnings),
        "n_errors": len(all_errors),
        "p_map_used": (float(p_map) if math.isfinite(p_map) else None),
        "p_series_used": (float(p_series) if math.isfinite(p_series) else None),
        "bo": bo,
    }


def score_prediction_confidence(
    pipeline_out: Dict[str, Any],
) -> float:
    """Return a composite confidence score 0.0–1.0 for a pipeline output.

    Higher means the pipeline ran cleanly with budget to spare.
    Returns 0.5 when confidence cannot be assessed (no governor state).
    """
    if not isinstance(pipeline_out, dict):
        return 0.0

    health = check_prediction_health(pipeline_out)
    if not health["healthy"]:
        return 0.0

    score = 1.0
    budget = pipeline_out.get("trace_governor_budget_final") or {}
    if isinstance(budget, dict) and budget.get("available"):
        max_pp = _f(budget.get("max_total_shrink_pp"), float("nan"))
        used_pp = _f(budget.get("used_shrink_pp"), float("nan"))
        if math.isfinite(max_pp) and math.isfinite(used_pp) and max_pp > 0.0:
            pct_used = used_pp / max_pp
            # Deduct up to 0.30 based on governor consumption
            score -= 0.30 * pct_used

    n_warnings = health["n_warnings"]
    score -= 0.05 * float(n_warnings)

    return float(max(0.0, min(1.0, score)))


__all__ = [
    "check_prediction_health",
    "score_prediction_confidence",
]
