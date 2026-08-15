"""Dynamic state-strength shadow helpers (A-lite).

This module implements a pure, testable shadow estimator that combines:
- structural strength prior (team model output / diff points)
- recent-form directional signal
- uncertainty (Glicko-like rating deviation proxy)

It is designed for shadow mode first: compare old vs new probabilities without
changing the production odds path.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import isfinite_num as _isfinite
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _safe_logit
from app_core.prob_math_core import safe_sigmoid as _safe_sigmoid
from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)


def _odd_from_p(p: Optional[float]) -> Optional[float]:
    if p is None or not _isfinite(p):
        return None
    pp = _f(p)
    if pp <= 0.0 or pp >= 1.0:
        return None
    return float(1.0 / pp)


def _prob_win_series(p_map: float, bo: int) -> float:
    p = _clip_prob(float(p_map))
    b = int(bo or 1)
    if b <= 1:
        return p
    if b == 3:
        # win 2+ in Bo3
        return float((p * p * p) + (3.0 * p * p * (1.0 - p)))
    if b == 5:
        # win 3+ in Bo5
        q = 1.0 - p
        return float((p**5) + (5.0 * (p**4) * q) + (10.0 * (p**3) * (q**2)))
    # Fallback: constant p Bernoulli, exact binomial sum.
    need = (b // 2) + 1
    out = 0.0
    for k in range(need, b + 1):
        out += math.comb(b, k) * (p**k) * ((1.0 - p) ** (b - k))
    return float(out)


def _is_explicit_interregional_league(league: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(league))


def _macro_key(league: str) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _glicko_like_rd(*, games_played: float, shrink_factor: float, roster_changed: bool, patch_status: str, patch_shrink: float) -> float:
    n = max(0.0, _f(games_played, 0.0))
    sf = _f(shrink_factor, 1.0 if n > 0 else 0.0)
    sf = max(0.0, min(1.0, sf if _isfinite(sf) else 0.0))
    rd = 0.30 + (1.40 / math.sqrt(n + 1.0)) + (0.85 * (1.0 - sf))
    if roster_changed:
        rd *= 1.15
    if patch_status in {"refit", "refit_needed", "changed"}:
        rd *= (1.0 + 0.20 * max(0.0, min(1.0, _f(patch_shrink, 0.0))))
    return float(max(0.18, min(2.60, rd)))


def _strength_prob_from_diff(diff_points: float, scale_points: float) -> float:
    sp = max(1e-9, abs(_f(scale_points, 400.0)))
    d = _f(diff_points, 0.0)
    return float(_clip_prob(1.0 / (1.0 + (10.0 ** (-d / sp)))))


def compute_state_shadow_alite(
    *,
    p_map_old: Any,
    p_series_old: Any,
    bo: int,
    ml_info: Optional[Dict[str, Any]],
    trace_form_info: Optional[Dict[str, Any]],
    trace_fusion_info: Optional[Dict[str, Any]],
    trace_patch_guard_info: Optional[Dict[str, Any]],
    trace_uncertainty_info: Optional[Dict[str, Any]],
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    consistency_guard_triggered: bool = False,
    consistency_guard_drift_pp: Any = None,
) -> Dict[str, Any]:
    """Build an A-lite shadow probability using state + uncertainty.

    Returns a trace-ready dict. If inputs are insufficient, returns applied=False
    with a reason and carries the old probabilities.
    """
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "method": "state_shadow_alite_v1",
        "reason": "",
        "bo": int(bo or 1),
        "p_map_old": (_f(p_map_old) if _isfinite(p_map_old) else None),
        "p_series_old": (_f(p_series_old) if _isfinite(p_series_old) else None),
        "p_map_new": None,
        "p_series_new": None,
        "diff_pp_map": None,
        "diff_pp_series": None,
    }

    if not _isfinite(p_map_old):
        out["reason"] = "invalid_p_map_old"
        return out

    p_old = _clip_prob(_f(p_map_old))
    z_old = _safe_logit(p_old)

    mi = ml_info if isinstance(ml_info, dict) else {}
    blue = mi.get("blue") if isinstance(mi.get("blue"), dict) else {}
    red = mi.get("red") if isinstance(mi.get("red"), dict) else {}
    tf = trace_form_info if isinstance(trace_form_info, dict) else {}
    fu = trace_fusion_info if isinstance(trace_fusion_info, dict) else {}
    pg = trace_patch_guard_info if isinstance(trace_patch_guard_info, dict) else {}
    tu = trace_uncertainty_info if isinstance(trace_uncertainty_info, dict) else {}

    p_struct = None
    if _isfinite(mi.get("p_cal")):
        p_struct = _clip_prob(_f(mi.get("p_cal")))
    elif _isfinite(mi.get("p_raw")):
        p_struct = _clip_prob(_f(mi.get("p_raw")))
    elif _isfinite(mi.get("diff_points")) and _isfinite(mi.get("scale_points")):
        p_struct = _strength_prob_from_diff(_f(mi.get("diff_points")), _f(mi.get("scale_points"), 400.0))
    else:
        p_struct = p_old
    z_struct = _safe_logit(p_struct)

    league_a = str(blue.get("league") or "").strip()
    league_b = str(red.get("league") or "").strip()
    macro_a = _macro_key(league_a)
    macro_b = _macro_key(league_b)
    interregional = bool(
        (macro_a and macro_b and macro_a != macro_b)
        or _is_explicit_interregional_league(league_a)
        or _is_explicit_interregional_league(league_b)
    )
    unknown_macro = bool(macro_a == "OTHER" or macro_b == "OTHER")

    patch_status = str(pg.get("status") or "").strip().lower()
    patch_shrink = max(0.0, min(1.0, _f(pg.get("shrink"), 0.0)))
    roster_changed = bool(fu.get("roster_changed", False))

    games_b = max(0.0, _f(blue.get("games_played"), 0.0))
    games_r = max(0.0, _f(red.get("games_played"), 0.0))
    shrink_b = _f(blue.get("shrink_factor"), 1.0 if games_b > 0 else 0.0)
    shrink_r = _f(red.get("shrink_factor"), 1.0 if games_r > 0 else 0.0)
    rd_b = _glicko_like_rd(
        games_played=games_b,
        shrink_factor=shrink_b,
        roster_changed=roster_changed,
        patch_status=patch_status,
        patch_shrink=patch_shrink,
    )
    rd_r = _glicko_like_rd(
        games_played=games_r,
        shrink_factor=shrink_r,
        roster_changed=roster_changed,
        patch_status=patch_status,
        patch_shrink=patch_shrink,
    )
    rd_pair = float(math.sqrt((rd_b * rd_b) + (rd_r * rd_r)))

    n_eff_unc = max(0.0, _f(tu.get("n_eff"), 0.0))
    prior_unc = max(1.0, _f(tu.get("prior_games"), 22.0))
    w_unc = max(0.18, min(1.0, n_eff_unc / (n_eff_unc + prior_unc))) if n_eff_unc > 0 else 0.18

    ev_team = max(0.0, _f(n_team_eff, min(games_b, games_r)))
    ev_players = max(0.0, _f(n_players_eff, 0.0))
    ev_total = float(ev_team + 0.65 * ev_players)
    w_evidence = max(0.30, min(1.0, ev_total / (ev_total + 14.0))) if ev_total > 0 else 0.30

    # Convert pair RD into a confidence-like weight.
    w_rd = float(1.0 / (1.0 + (rd_pair / 1.10) ** 2))
    w_rd = max(0.20, min(1.0, w_rd))

    w_patch = 1.0
    if patch_status in {"refit_needed", "refit", "changed"}:
        w_patch = max(0.55, min(1.0, 1.0 - 0.65 * patch_shrink))
    elif bool(pg.get("applied", False)):
        w_patch = max(0.65, min(1.0, 1.0 - 0.45 * patch_shrink))

    w_consistency = 1.0
    drift_pp = _f(consistency_guard_drift_pp)
    if bool(consistency_guard_triggered) and _isfinite(drift_pp):
        w_consistency = max(0.55, min(1.0, 1.0 - min(0.45, abs(drift_pp) / 45.0)))

    # Build dynamic target in logit space: structural prior + directional recent form.
    p_form = _f(tf.get("p_form"))
    p_form_model = _f(tf.get("p_model"))
    w_form_used = max(0.0, min(1.0, _f(tf.get("w_form"), 0.0)))
    n_form_a = _f(tf.get("n_a"))
    n_form_b = _f(tf.get("n_b"))
    n_form_min = min(n_form_a, n_form_b) if _isfinite(n_form_a) and _isfinite(n_form_b) else float("nan")
    form_applied = bool(tf.get("applied", False))
    recent_evidence = 0.0
    if _isfinite(n_form_min) and n_form_min > 0:
        recent_evidence = max(0.0, min(1.0, float(n_form_min) / (float(n_form_min) + 6.0)))

    z_target = z_struct
    w_recent_dir = 0.0
    recent_delta_logit = 0.0
    interregional_relief = 0.0
    if form_applied and _isfinite(p_form):
        z_form = _safe_logit(_f(p_form))
        z_form_base = _safe_logit(_f(p_form_model)) if _isfinite(p_form_model) else z_struct
        recent_delta_logit = float(z_form - z_form_base)
        w_recent_dir = float(max(0.0, min(0.75, (0.08 + 0.72 * w_form_used) * recent_evidence)))
        if roster_changed:
            w_recent_dir = float(min(0.80, w_recent_dir * 1.12))
        if interregional:
            interregional_relief = float(
                max(
                    0.0,
                    min(
                        0.22,
                        (0.10 * recent_evidence)
                        + (0.07 * (ev_team / (ev_team + 12.0) if ev_team > 0.0 else 0.0))
                        + (0.05 * (ev_players / (ev_players + 10.0) if ev_players > 0.0 else 0.0)),
                    ),
                )
            )
            if unknown_macro:
                interregional_relief *= 0.65
            if roster_changed and patch_status in {"refit", "refit_needed", "changed"}:
                interregional_relief *= 0.80
            w_recent_dir = float(min(0.32, w_recent_dir * (0.55 + interregional_relief)))
        elif unknown_macro:
            w_recent_dir = float(min(0.20, w_recent_dir * 0.75))
        # Cap recent directional delta (logit), stronger cap on low sample or unknown context.
        delta_cap = float(0.50 + 0.70 * recent_evidence)
        if _isfinite(n_form_min):
            if n_form_min < 6.0:
                delta_cap *= 0.55
            elif n_form_min < 10.0:
                delta_cap *= 0.75
        if interregional:
            delta_cap *= float(0.60 + (0.18 * interregional_relief))
        elif unknown_macro:
            delta_cap *= 0.80
        # CN/KR same-region top-vs-bottom can legitimately have huge gaps; allow more movement.
        if macro_a in {"CN", "KR"} and macro_a == macro_b and _isfinite(p_old) and p_old >= 0.75:
            delta_cap *= 1.18
        delta_cap = float(max(0.20, min(1.40, delta_cap)))
        recent_delta_logit = float(max(-delta_cap, min(delta_cap, recent_delta_logit)))
        z_target = float(z_target + (w_recent_dir * recent_delta_logit))

    # A-lite final weight: uncertainty + evidence + RD + patch + consistency.
    w_floor = 0.24
    if recent_evidence > 0.0:
        w_floor = float(min(0.72, w_floor + 0.16 * recent_evidence))
    if ev_team > 0.0:
        w_floor = float(min(0.80, w_floor + 0.16 * (ev_team / (ev_team + 10.0))))
    if unknown_macro and ev_team > 0.0 and not interregional:
        w_floor = float(min(0.84, w_floor + 0.04 * (ev_team / (ev_team + 25.0))))

    w_final = float(max(w_floor, min(1.0, w_unc * w_evidence * w_rd * w_patch * w_consistency)))

    z_delta_to_target = float(z_target - z_old)
    # Conservative cap on total move in one pass.
    move_cap = 1.20 if (macro_a in {"CN", "KR"} and macro_a == macro_b and p_old >= 0.78) else 0.95
    if interregional:
        move_cap *= float(0.75 + (0.10 * interregional_relief))
    elif unknown_macro:
        move_cap *= 0.85
    z_delta_to_target = float(max(-move_cap, min(move_cap, z_delta_to_target)))

    z_new = float(z_old + (w_final * z_delta_to_target))
    p_new = float(_clip_prob(_safe_sigmoid(z_new)))
    try:
        p_series_new = float(_prob_win_series(p_new, int(bo or 1)))
    except Exception:
        p_series_new = float("nan")

    out.update(
        {
            "applied": True,
            "reason": "state+uncertainty+recent_form",
            "p_map_new": p_new,
            "p_series_new": (p_series_new if _isfinite(p_series_new) else None),
            "diff_pp_map": float(100.0 * (p_new - p_old)),
            "diff_pp_series": (
                float(100.0 * (p_series_new - _f(p_series_old)))
                if _isfinite(p_series_old) and _isfinite(p_series_new)
                else None
            ),
            "weights": {
                "uncertainty": float(w_unc),
                "evidence": float(w_evidence),
                "rd_conf": float(w_rd),
                "patch": float(w_patch),
                "consistency": float(w_consistency),
                "floor": float(w_floor),
                "final": float(w_final),
                "recent_direction": float(w_recent_dir),
                "interregional_relief": float(interregional_relief),
            },
            "uncertainty": {
                "rd_blue": float(rd_b),
                "rd_red": float(rd_r),
                "rd_pair": float(rd_pair),
                "games_blue": float(games_b),
                "games_red": float(games_r),
                "shrink_blue": float(max(0.0, min(1.0, _f(shrink_b, 0.0)))),
                "shrink_red": float(max(0.0, min(1.0, _f(shrink_r, 0.0)))),
            },
            "inputs": {
                "league_a": str(league_a),
                "league_b": str(league_b),
                "macro_a": str(macro_a),
                "macro_b": str(macro_b),
                "interregional": bool(interregional),
                "unknown_macro": bool(unknown_macro),
                "roster_changed": bool(roster_changed),
                "patch_status": str(patch_status),
                "patch_shrink": float(patch_shrink),
                "n_eff_uncertainty": float(n_eff_unc),
                "prior_games_uncertainty": float(prior_unc),
                "n_team_eff": float(ev_team),
                "n_players_eff": float(ev_players),
                "season_form_applied": bool(form_applied),
                "season_form_w": float(w_form_used),
                "season_form_n_min": (float(n_form_min) if _isfinite(n_form_min) else None),
                "season_form_p_form": (float(_f(p_form)) if _isfinite(p_form) else None),
                "season_form_p_model": (float(_f(p_form_model)) if _isfinite(p_form_model) else None),
                "recent_evidence": float(recent_evidence),
                "recent_delta_logit": float(recent_delta_logit),
            },
            "targets": {
                "p_struct": float(p_struct),
                "z_old": float(z_old),
                "z_struct": float(z_struct),
                "z_target": float(z_target),
                "z_delta_to_target": float(z_delta_to_target),
                "z_new": float(z_new),
            },
            "odds": {
                "teamA_map_old": _odd_from_p(p_old),
                "teamA_map_new": _odd_from_p(p_new),
                "teamA_series_old": _odd_from_p(_f(p_series_old)) if _isfinite(p_series_old) else None,
                "teamA_series_new": _odd_from_p(p_series_new) if _isfinite(p_series_new) else None,
            },
        }
    )
    return out
