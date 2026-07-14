"""Roster continuity / team identity adjustment layer for ML map probability.

Goal:
- reduce over-reliance on org historical strength when current lineup continuity is low
- keep the move conservative and auditable
- use existing runtime fusion signals (lineup_continuity/freshness/coverage/roster_changed)

This is a runtime-only layer (no separate training artifact for now).
"""

from __future__ import annotations

import math
from typing import Any, Dict

from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _logit
from app_core.prob_math_core import safe_sigmoid as _sigmoid


def _norm_playoffs(v: Any) -> bool:
    s = str(v or "").strip().lower()
    return s in {"1", "true", "playoffs", "sim", "yes"}


def build_roster_live_inputs(
    *,
    fusion_info: Dict[str, Any] | None = None,
    shadow_state_areal: Dict[str, Any] | None = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    playoffs: Any = None,
    roster_gate: Any = None,
) -> Dict[str, Any]:
    """Build stable runtime inputs for roster continuity layer.

    Prefers players/fusion signals when available, then falls back to A-real
    shadow state continuity metadata.
    """

    out: Dict[str, Any] = {
        "lineup_continuity": None,
        "freshness_avg": None,
        "coverage_eff": None,
        "transfer_signal": None,
        "roster_changed": False,
        "roster_changed_source": "default_false",
        "season_phase": ("playoffs" if _norm_playoffs(playoffs) else "auto"),
        "signal_source": "none",
        "fallback_used": False,
        "source_parts": [],
    }
    src_parts: list[str] = []
    fz = fusion_info if isinstance(fusion_info, dict) else {}
    if str(fz.get("season_phase", "") or "").strip():
        out["season_phase"] = str(fz.get("season_phase") or "").strip()

    if "roster_changed" in fz:
        out["roster_changed"] = bool(fz.get("roster_changed", False))
        out["roster_changed_source"] = "fusion"
        src_parts.append("fusion.roster_changed")

    pp_summary = fz.get("player_portability_summary") if isinstance(fz.get("player_portability_summary"), dict) else {}
    id_sig_a = _f(pp_summary.get("identity_successor_signal_a"))
    id_sig_b = _f(pp_summary.get("identity_successor_signal_b"))
    id_match_a = _f(pp_summary.get("identity_match_ratio_a"))
    id_match_b = _f(pp_summary.get("identity_match_ratio_b"))
    id_pair = [v for v in [id_sig_a, id_sig_b] if math.isfinite(v)]
    id_match_pair = [v for v in [id_match_a, id_match_b] if math.isfinite(v)]
    identity_continuity = None
    if id_pair:
        identity_continuity = float(max(0.0, min(1.0, sum(id_pair) / len(id_pair))))
        out["identity_successor_continuity"] = float(identity_continuity)
        src_parts.append("fusion.player_portability_summary.identity_successor_signal")
    if id_match_pair:
        out["identity_match_ratio"] = float(max(0.0, min(1.0, sum(id_match_pair) / len(id_match_pair))))
        src_parts.append("fusion.player_portability_summary.identity_match_ratio")

    def _set_prob_field(name: str, value: Any, source: str) -> None:
        x = _f(value)
        if math.isfinite(x):
            out[name] = float(max(0.0, min(1.0, x)))
            src_parts.append(source)

    _set_prob_field("lineup_continuity", fz.get("lineup_continuity"), "fusion.lineup_continuity")
    _set_prob_field("freshness_avg", fz.get("freshness_avg"), "fusion.freshness_avg")
    _set_prob_field("coverage_eff", fz.get("coverage_eff"), "fusion.coverage_eff")
    _set_prob_field("transfer_signal", fz.get("transfer_signal"), "fusion.transfer_signal")
    if out.get("lineup_continuity") is not None and identity_continuity is not None:
        out["lineup_continuity"] = float(max(0.0, min(1.0, (0.82 * float(out["lineup_continuity"])) + (0.18 * identity_continuity))))
        src_parts.append("blend.lineup_continuity_identity")
    elif out.get("lineup_continuity") is None and identity_continuity is not None:
        out["lineup_continuity"] = float(identity_continuity)
        src_parts.append("identity_successor_continuity")

    sar = shadow_state_areal if isinstance(shadow_state_areal, dict) else {}
    sar_state = sar.get("state") if isinstance(sar.get("state"), dict) else {}
    b_cont = _f(sar_state.get("blue_roster_cont_avg"))
    r_cont = _f(sar_state.get("red_roster_cont_avg"))
    b_n = _f(sar_state.get("blue_roster_cont_n"))
    r_n = _f(sar_state.get("red_roster_cont_n"))
    stale_days = _f(sar_state.get("stale_days_max"))

    used_shadow = False
    if out.get("lineup_continuity") is None and math.isfinite(b_cont) and math.isfinite(r_cont):
        out["lineup_continuity"] = float(max(0.0, min(1.0, (float(b_cont) + float(r_cont)) * 0.5)))
        src_parts.append("shadow_state_areal.roster_cont_avg")
        used_shadow = True

    if out.get("coverage_eff") is None:
        if math.isfinite(b_n) and math.isfinite(r_n) and b_n > 0.0 and r_n > 0.0:
            n_min = float(max(0.0, min(float(b_n), float(r_n))))
            out["coverage_eff"] = float(max(0.0, min(1.0, n_min / (n_min + 8.0))))
            src_parts.append("shadow_state_areal.roster_cont_n")
            used_shadow = True
        else:
            npl = _f(n_players_eff)
            ntm = _f(n_team_eff)
            if math.isfinite(npl) and npl > 0.0:
                out["coverage_eff"] = float(max(0.0, min(1.0, npl / (npl + 6.0))))
                src_parts.append("runtime.n_players_eff")
            elif math.isfinite(ntm) and ntm > 0.0:
                out["coverage_eff"] = float(max(0.0, min(1.0, (ntm / (ntm + 24.0)) * 0.45)))
                src_parts.append("runtime.n_team_eff")

    if out.get("freshness_avg") is None:
        if math.isfinite(stale_days):
            days = float(max(0.0, stale_days))
            out["freshness_avg"] = float(max(0.25, min(1.0, 1.0 / (1.0 + (days / 18.0) ** 1.20))))
            src_parts.append("shadow_state_areal.stale_days")
            used_shadow = True
        elif out.get("lineup_continuity") is not None:
            lc = float(out.get("lineup_continuity") or 0.5)
            out["freshness_avg"] = float(max(0.25, min(1.0, 0.62 + (0.28 * lc))))
            src_parts.append("derived_from_lineup_continuity")

    if out.get("transfer_signal") is None and out.get("lineup_continuity") is not None:
        lc = float(out.get("lineup_continuity") or 0.5)
        out["transfer_signal"] = float(max(0.0, min(1.0, 1.0 - lc)))
        src_parts.append("derived_from_lineup_continuity")
    elif out.get("transfer_signal") is not None and identity_continuity is not None:
        out["transfer_signal"] = float(max(0.0, min(1.0, (0.78 * float(out["transfer_signal"])) + (0.22 * (1.0 - identity_continuity)))))
        src_parts.append("blend.transfer_signal_identity")

    if str(out.get("roster_changed_source", "")) != "fusion":
        thr = _f(roster_gate, 0.55)
        thr = float(max(0.30, min(0.90, thr if math.isfinite(thr) else 0.55)))
        if out.get("lineup_continuity") is not None:
            out["roster_changed"] = bool(float(out.get("lineup_continuity") or 1.0) < thr)
            out["roster_changed_source"] = "lineup_vs_gate"
        else:
            out["roster_changed"] = False
            out["roster_changed_source"] = "default_false"

    has_fusion = any(str(s).startswith("fusion.") for s in src_parts)
    has_shadow = any(str(s).startswith("shadow_state_areal.") for s in src_parts)
    if has_fusion and has_shadow:
        out["signal_source"] = "fusion+areal_fallback"
    elif has_fusion:
        out["signal_source"] = "fusion"
    elif has_shadow:
        out["signal_source"] = "areal_fallback"
    elif src_parts:
        out["signal_source"] = "runtime_fallback"
    out["fallback_used"] = bool(used_shadow or out["signal_source"] in {"areal_fallback", "fusion+areal_fallback", "runtime_fallback"})
    out["source_parts"] = list(dict.fromkeys(src_parts))
    return out


def apply_roster_continuity_live(
    *,
    p_map_in: Any,
    p_players: Any = None,
    p_team: Any = None,
    lineup_continuity: Any = None,
    freshness_avg: Any = None,
    coverage_eff: Any = None,
    roster_changed: bool = False,
    season_phase: str = "",
    bo: Any = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    transfer_signal: Any = None,
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "roster_continuity_v2",
        "p_before": None,
        "p_after": None,
        "diff_pp": None,
        "weight": None,
        "target_p": None,
        "lineup_continuity": None,
        "freshness_avg": None,
        "coverage_eff": None,
        "roster_changed": bool(roster_changed),
        "season_phase": str(season_phase or ""),
        "bo": None,
        "n_team_eff": None,
        "n_players_eff": None,
        "transfer_signal": None,
        "hist_weight": None,
        "players_evidence": None,
        "cfg": {},
    }
    p0 = _f(p_map_in)
    if not math.isfinite(p0):
        out["reason"] = "invalid_p_in"
        return out
    out["p_before"] = float(p0)
    cfg = cfg if isinstance(cfg, dict) else {}
    c_strength = float(max(0.0, min(2.0, _f(cfg.get("strength", 1.0), 1.0))))
    c_team_shrink = float(max(0.0, min(2.0, _f(cfg.get("team_shrink_mult", 1.0), 1.0))))
    c_players_alpha = float(max(0.0, min(2.0, _f(cfg.get("players_alpha_mult", 1.0), 1.0))))
    c_min_cov = float(max(0.0, min(0.8, _f(cfg.get("min_coverage", 0.15), 0.15))))
    c_w_cap = float(max(0.02, min(0.45, _f(cfg.get("weight_cap", 0.28), 0.28))))
    out["cfg"] = {
        "strength": c_strength,
        "team_shrink_mult": c_team_shrink,
        "players_alpha_mult": c_players_alpha,
        "min_coverage": c_min_cov,
        "weight_cap": c_w_cap,
    }

    p_pl = _f(p_players)
    p_tm = _f(p_team)
    lc = _f(lineup_continuity)
    fr = _f(freshness_avg)
    cov = _f(coverage_eff)
    n_team = _f(n_team_eff)
    n_players = _f(n_players_eff)
    transfer = _f(transfer_signal)
    out["lineup_continuity"] = (float(lc) if math.isfinite(lc) else None)
    out["freshness_avg"] = (float(fr) if math.isfinite(fr) else None)
    out["coverage_eff"] = (float(cov) if math.isfinite(cov) else None)
    out["n_team_eff"] = (float(n_team) if math.isfinite(n_team) else None)
    out["n_players_eff"] = (float(n_players) if math.isfinite(n_players) else None)
    out["transfer_signal"] = (float(transfer) if math.isfinite(transfer) else None)
    try:
        bo_i = int(float(bo)) if bo is not None else 1
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    out["bo"] = int(bo_i)

    # Need some evidence from lineup/players to say anything useful.
    if not math.isfinite(lc) and not math.isfinite(fr):
        out["reason"] = "missing_roster_signals"
        return out

    # Normalize missing values conservatively.
    lc_n = float(min(1.0, max(0.0, lc))) if math.isfinite(lc) else 0.5
    fr_n = float(min(1.0, max(0.0, fr))) if math.isfinite(fr) else 0.5
    cov_n = float(min(1.0, max(0.0, cov))) if math.isfinite(cov) else 0.0

    # How much we distrust the org historical identity.
    low_cont = (1.0 - lc_n)
    low_fresh = (1.0 - fr_n)
    roster_pressure = 0.55 * low_cont + 0.30 * low_fresh + (0.15 if bool(roster_changed) else 0.0)
    roster_pressure = float(max(0.0, min(1.0, roster_pressure)))

    # If coverage is weak, avoid overreacting.
    evidence = float(max(0.0, min(1.0, cov_n)))
    if evidence < c_min_cov:
        out["reason"] = "low_lineup_coverage"
        return out

    # Team-history strength proxy (how much historical identity should matter).
    # Higher n_team means stronger org-historical anchor.
    n_team_n = max(0.0, float(n_team)) if math.isfinite(n_team) else 0.0
    k_hist = 18.0
    hist_weight = float(max(0.0, min(1.0, n_team_n / (n_team_n + k_hist))))
    out["hist_weight"] = float(hist_weight)

    # Players evidence proxy.
    n_players_n = max(0.0, float(n_players)) if math.isfinite(n_players) else 0.0
    k_players = 7.0
    players_ev = float(max(0.0, min(1.0, n_players_n / (n_players_n + k_players))))
    if math.isfinite(transfer):
        players_ev = float(max(0.0, min(1.0, players_ev + 0.18 * max(0.0, min(1.0, float(transfer))))))
    out["players_evidence"] = float(players_ev)

    # Build roster-aware target:
    # 1) shrink team-only identity toward 50% under roster pressure
    # 2) if players signal exists, blend toward players according to evidence
    if math.isfinite(p_tm):
        zt = _logit(float(_clip_prob(p_tm)))
        shrink_team = (0.40 + 0.45 * hist_weight) * roster_pressure * c_team_shrink
        team_adj = float(_sigmoid((1.0 - shrink_team) * zt))
    else:
        z0 = _logit(float(_clip_prob(p0)))
        shrink_cur = (0.45 + 0.35 * hist_weight) * roster_pressure * c_team_shrink
        team_adj = float(_sigmoid((1.0 - shrink_cur) * z0))

    if math.isfinite(p_pl):
        ppl = float(_clip_prob(p_pl))
        alpha_players = float(max(0.0, min(1.0, players_ev * (0.55 + 0.45 * roster_pressure) * c_players_alpha)))
        ztgt = (1.0 - alpha_players) * _logit(team_adj) + alpha_players * _logit(ppl)
        target = float(_sigmoid(ztgt))
        target_source = "team_shrunk_plus_players"
    else:
        target = float(team_adj)
        target_source = "team_shrunk_only"
    out["target_p"] = float(target)
    out["target_source"] = str(target_source)

    # Weight is conservative and grows only with evidence and roster-change signal.
    w = 0.0
    if roster_pressure > 0.05:
        # More impact when team-history anchor is strong and players evidence exists.
        w = (0.03 + 0.24 * roster_pressure * math.sqrt(evidence) * (0.55 + 0.45 * hist_weight)) * c_strength
        if math.isfinite(p_pl):
            w *= float(0.85 + 0.35 * players_ev)
    # Early season: allow a bit more because roster identity mismatch is common.
    sp = str(season_phase or "").strip().lower()
    if sp in {"early", "start", "inicio"}:
        w *= 1.15
    # BO1 more conservative; BO5 slightly less.
    if bo_i == 1:
        w *= 0.85
    elif bo_i >= 5:
        w *= 1.08
    # If target strongly disagrees and evidence is modest, reduce.
    disagree = abs(_logit(target) - _logit(p0))
    if disagree > 0.9 and evidence < 0.45:
        w *= 0.65
    w = float(max(0.0, min(c_w_cap, w)))
    out["weight"] = float(w)
    if w <= 0.0:
        out["reason"] = "weight_zero"
        return out

    z1 = (1.0 - w) * _logit(p0) + w * _logit(target)
    p1 = float(_sigmoid(z1))
    out["applied"] = True
    out["reason"] = "ok"
    out["p_after"] = float(p1)
    out["diff_pp"] = float(100.0 * (p1 - p0))
    out["roster_pressure"] = float(roster_pressure)
    out["evidence"] = float(evidence)
    return out
