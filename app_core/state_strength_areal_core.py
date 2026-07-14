"""A-real (team state artifact) runtime helpers.

Applies a state-based correction using a persisted team-state artifact
(`mu`, `sigma`, `gp`) and blends it in logit space with the current ML map
probability.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date
from typing import Any, Dict, Optional

from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import isfinite_num as _isfinite
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _safe_logit
from app_core.prob_math_core import safe_sigmoid as _safe_sigmoid
from mlcore.tiers import league_strength_score, tier_for_league


def _norm_team_key(v: Any) -> str:
    s = str(v or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _prob_win_series(p_map: float, bo: int) -> float:
    p = _clip_prob(float(p_map))
    b = int(bo or 1)
    if b <= 1:
        return p
    if b == 3:
        return float((p * p * p) + (3.0 * p * p * (1.0 - p)))
    if b == 5:
        q = 1.0 - p
        return float((p**5) + (5.0 * (p**4) * q) + (10.0 * (p**3) * (q**2)))
    need = (b // 2) + 1
    out = 0.0
    for k in range(need, b + 1):
        out += math.comb(b, k) * (p**k) * ((1.0 - p) ** (b - k))
    return float(out)


def _odd_from_p(p: Any) -> Optional[float]:
    if not _isfinite(p):
        return None
    pp = _f(p)
    if pp <= 0.0 or pp >= 1.0:
        return None
    return float(1.0 / pp)


def _safe_date(v: Any) -> Optional[date]:
    s = str(v or "").strip()
    if not s:
        return None
    s = s[:10]
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except Exception:
        return None


def _tier_rank(league: Any) -> int:
    tier = str(tier_for_league(str(league or "")) or "OTHER").upper()
    return {
        "T1": 0,
        "T2": 1,
        "ERL": 2,
        "MINOR": 3,
        "OTHER": 4,
    }.get(tier, 4)


def _strength_gap(league_a: Any, league_b: Any) -> float:
    return float(
        abs(
            float(max(0.0, min(1.0, league_strength_score(str(league_a or "")))))
            - float(max(0.0, min(1.0, league_strength_score(str(league_b or "")))))
        )
    )


def compute_state_shadow_areal(
    *,
    p_map_old: Any,
    p_series_old: Any,
    bo: int,
    team_blue: str,
    team_red: str,
    league_blue: str = "",
    league_red: str = "",
    state_artifact: Optional[Dict[str, Any]],
    trace_fusion_info: Optional[Dict[str, Any]] = None,
    trace_patch_guard_info: Optional[Dict[str, Any]] = None,
    trace_uncertainty_info: Optional[Dict[str, Any]] = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    consistency_guard_triggered: bool = False,
    consistency_guard_drift_pp: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "method": "state_shadow_areal_v9",
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
    if not isinstance(state_artifact, dict):
        out["reason"] = "missing_state_artifact"
        return out
    teams = state_artifact.get("teams")
    if not isinstance(teams, dict) or not teams:
        out["reason"] = "invalid_state_artifact_teams"
        return out

    out["artifact_version"] = str(state_artifact.get("version") or "")
    _art_meta = state_artifact.get("meta") if isinstance(state_artifact.get("meta"), dict) else {}
    if isinstance(_art_meta, dict):
        out["artifact_meta"] = {
            "update_mode": str(_art_meta.get("update_mode") or ""),
            "seed_used": bool(_art_meta.get("seed_used", False)),
            "backfill_detected": bool(_art_meta.get("backfill_detected", False)),
            "last_processed_game": _art_meta.get("last_processed_game"),
            "online_journal_tail_count": int(_f(_art_meta.get("online_journal_tail_count"), 0.0)),
            "quality_adjust_avg": (_f(_art_meta.get("quality_adjust_avg"), float("nan")) if _isfinite(_art_meta.get("quality_adjust_avg")) else None),
            "quality_early_avg": (_f(_art_meta.get("quality_early_avg"), float("nan")) if _isfinite(_art_meta.get("quality_early_avg")) else None),
            "quality_objective_avg": (_f(_art_meta.get("quality_objective_avg"), float("nan")) if _isfinite(_art_meta.get("quality_objective_avg")) else None),
            "quality_kill_avg": (_f(_art_meta.get("quality_kill_avg"), float("nan")) if _isfinite(_art_meta.get("quality_kill_avg")) else None),
            "quality_tempo_avg": (_f(_art_meta.get("quality_tempo_avg"), float("nan")) if _isfinite(_art_meta.get("quality_tempo_avg")) else None),
            "quality_conversion_avg": (_f(_art_meta.get("quality_conversion_avg"), float("nan")) if _isfinite(_art_meta.get("quality_conversion_avg")) else None),
            "quality_throw_avg": (_f(_art_meta.get("quality_throw_avg"), float("nan")) if _isfinite(_art_meta.get("quality_throw_avg")) else None),
            "quality_comeback_avg": (_f(_art_meta.get("quality_comeback_avg"), float("nan")) if _isfinite(_art_meta.get("quality_comeback_avg")) else None),
        }
    _art_params = state_artifact.get("params") if isinstance(state_artifact.get("params"), dict) else {}
    if isinstance(_art_params, dict) and _art_params:
        def _qwl_param(name: str) -> Any:
            v = _f(_art_params.get(name))
            return float(v) if math.isfinite(v) else None

        def _qwl_map_len(name: str) -> int:
            vv = _art_params.get(name)
            if isinstance(vv, (list, tuple)):
                return len(vv)
            if isinstance(vv, dict):
                return len(vv)
            return 0

        out["qwl_trace"] = {
            "enabled": True,
            "params": {
                "qwl_expected_base": _qwl_param("qwl_expected_base"),
                "qwl_expected_slope": _qwl_param("qwl_expected_slope"),
                "qwl_surplus_mix": _qwl_param("qwl_surplus_mix"),
                "qwl_tempo_weight": _qwl_param("qwl_tempo_weight"),
                "qwl_conversion_weight": _qwl_param("qwl_conversion_weight"),
                "qwl_throw_weight": _qwl_param("qwl_throw_weight"),
                "qwl_comeback_weight": _qwl_param("qwl_comeback_weight"),
                "qwl_phase_dynamic_strength": _qwl_param("qwl_phase_dynamic_strength"),
                "qwl_opponent_context_mix": _qwl_param("qwl_opponent_context_mix"),
            },
            "map_overrides_count": {
                "qwl_surplus_mix_by_league": _qwl_map_len("qwl_surplus_mix_by_league"),
                "qwl_surplus_mix_by_patch": _qwl_map_len("qwl_surplus_mix_by_patch"),
                "qwl_throw_weight_by_league": _qwl_map_len("qwl_throw_weight_by_league"),
                "qwl_throw_weight_by_patch": _qwl_map_len("qwl_throw_weight_by_patch"),
                "qwl_comeback_weight_by_league": _qwl_map_len("qwl_comeback_weight_by_league"),
                "qwl_comeback_weight_by_patch": _qwl_map_len("qwl_comeback_weight_by_patch"),
                "qwl_phase_dynamic_strength_by_league": _qwl_map_len("qwl_phase_dynamic_strength_by_league"),
                "qwl_phase_dynamic_strength_by_patch": _qwl_map_len("qwl_phase_dynamic_strength_by_patch"),
                "qwl_opponent_context_mix_by_league": _qwl_map_len("qwl_opponent_context_mix_by_league"),
                "qwl_opponent_context_mix_by_patch": _qwl_map_len("qwl_opponent_context_mix_by_patch"),
            },
            "artifact_meta": {
                "quality_adjust_avg": (_f(_art_meta.get("quality_adjust_avg"), float("nan")) if _isfinite(_art_meta.get("quality_adjust_avg")) else None),
                "quality_expected_avg": (_f(_art_meta.get("quality_expected_avg"), float("nan")) if _isfinite(_art_meta.get("quality_expected_avg")) else None),
                "quality_surplus_avg": (_f(_art_meta.get("quality_surplus_avg"), float("nan")) if _isfinite(_art_meta.get("quality_surplus_avg")) else None),
                "quality_context_multiplier_avg": (
                    _f(_art_meta.get("quality_context_multiplier_avg"), float("nan"))
                    if _isfinite(_art_meta.get("quality_context_multiplier_avg"))
                    else None
                ),
                "quality_context_boost_games": int(_f(_art_meta.get("quality_context_boost_games"), 0.0)),
                "quality_context_cut_games": int(_f(_art_meta.get("quality_context_cut_games"), 0.0)),
                "quality_tempo_avg": (_f(_art_meta.get("quality_tempo_avg"), float("nan")) if _isfinite(_art_meta.get("quality_tempo_avg")) else None),
                "quality_conversion_avg": (_f(_art_meta.get("quality_conversion_avg"), float("nan")) if _isfinite(_art_meta.get("quality_conversion_avg")) else None),
                "quality_throw_avg": (_f(_art_meta.get("quality_throw_avg"), float("nan")) if _isfinite(_art_meta.get("quality_throw_avg")) else None),
                "quality_comeback_avg": (_f(_art_meta.get("quality_comeback_avg"), float("nan")) if _isfinite(_art_meta.get("quality_comeback_avg")) else None),
            },
        }

    kb = _norm_team_key(team_blue)
    kr = _norm_team_key(team_red)
    if not kb or not kr:
        out["reason"] = "empty_team_keys"
        return out
    sb = teams.get(kb) if isinstance(teams.get(kb), dict) else None
    sr = teams.get(kr) if isinstance(teams.get(kr), dict) else None
    if not sb or not sr:
        out["reason"] = "team_not_found_in_state_artifact"
        out["lookup"] = {"blue_key": kb, "red_key": kr, "has_blue": bool(sb), "has_red": bool(sr)}
        return out

    p_old = _clip_prob(_f(p_map_old))
    z_old = _safe_logit(p_old)
    mu_b = _f(sb.get("mu"), 0.0)
    mu_r = _f(sr.get("mu"), 0.0)
    sig_b = max(0.05, _f(sb.get("sigma"), 1.0))
    sig_r = max(0.05, _f(sr.get("sigma"), 1.0))
    gp_b = max(0.0, _f(sb.get("gp"), 0.0))
    gp_r = max(0.0, _f(sr.get("gp"), 0.0))

    def _derive_qwl_from_journal(team_key: str) -> Dict[str, float]:
        tail = state_artifact.get("online_journal_tail") if isinstance(state_artifact.get("online_journal_tail"), list) else []
        if not team_key or not tail:
            return {}
        n = 0
        s_adj = 0.0
        s_sur = 0.0
        s_tempo = 0.0
        s_conv = 0.0
        s_throw = 0.0
        s_cb = 0.0
        s_rel = 0.0
        for row in tail:
            if not isinstance(row, dict):
                continue
            k_blue = _norm_team_key(row.get("blue"))
            k_red = _norm_team_key(row.get("red"))
            side = None
            if k_blue == team_key:
                side = "blue"
            elif k_red == team_key:
                side = "red"
            if side is None:
                continue
            y_blue = int(_f(row.get("y_blue"), 0.0) >= 0.5)
            q_adj = float(max(0.0, min(1.0, _f(row.get("q_adjusted"), 0.0))))
            q_sur = float(max(0.0, min(1.0, _f(row.get("q_surplus"), 0.0))))
            q_tempo = float(max(-1.0, min(1.0, _f(row.get("q_tempo_term"), 0.0))))
            q_conv = float(max(-1.0, min(1.0, _f(row.get("q_conversion_term"), 0.0))))
            q_throw = float(max(-1.0, min(1.0, _f(row.get("q_throw_term"), 0.0))))
            q_cb = float(max(-1.0, min(1.0, _f(row.get("q_comeback_term"), 0.0))))
            q_rel = float(max(0.70, min(1.00, _f(row.get("q_data_reliability"), 1.0))))

            sign_blue = 1.0 if y_blue == 1 else -1.0
            sign = sign_blue if side == "blue" else -sign_blue
            side_sign = 1.0 if side == "blue" else -1.0
            s_adj += float(sign * q_adj)
            s_sur += float(sign * q_sur)
            s_tempo += float(side_sign * q_tempo)
            s_conv += float(side_sign * q_conv)
            s_throw += float(side_sign * q_throw)
            s_cb += float(side_sign * q_cb)
            s_rel += float(q_rel)
            n += 1
        if n <= 0:
            return {}
        inv_n = 1.0 / float(n)
        return {
            "qwl_n": float(n),
            "qwl_signed_avg": float(s_adj * inv_n),
            "qwl_surplus_signed_avg": float(s_sur * inv_n),
            "qwl_tempo_signed_avg": float(s_tempo * inv_n),
            "qwl_conversion_signed_avg": float(s_conv * inv_n),
            "qwl_throw_signed_avg": float(s_throw * inv_n),
            "qwl_comeback_signed_avg": float(s_cb * inv_n),
            "qwl_data_reliability_avg": float(max(0.70, min(1.00, s_rel * inv_n))),
        }

    # QWL runtime edge from team-level signed quality aggregates in artifact.
    qwl_adj_b = float(_f(sb.get("qwl_signed_avg"), 0.0))
    qwl_adj_r = float(_f(sr.get("qwl_signed_avg"), 0.0))
    qwl_sur_b = float(_f(sb.get("qwl_surplus_signed_avg"), 0.0))
    qwl_sur_r = float(_f(sr.get("qwl_surplus_signed_avg"), 0.0))
    qwl_tempo_b = float(_f(sb.get("qwl_tempo_signed_avg"), 0.0))
    qwl_tempo_r = float(_f(sr.get("qwl_tempo_signed_avg"), 0.0))
    qwl_conv_b = float(_f(sb.get("qwl_conversion_signed_avg"), 0.0))
    qwl_conv_r = float(_f(sr.get("qwl_conversion_signed_avg"), 0.0))
    qwl_throw_b = float(_f(sb.get("qwl_throw_signed_avg"), 0.0))
    qwl_throw_r = float(_f(sr.get("qwl_throw_signed_avg"), 0.0))
    qwl_cb_b = float(_f(sb.get("qwl_comeback_signed_avg"), 0.0))
    qwl_cb_r = float(_f(sr.get("qwl_comeback_signed_avg"), 0.0))
    qwl_rel_b = float(max(0.70, min(1.00, _f(sb.get("qwl_data_reliability_avg"), 1.0))))
    qwl_rel_r = float(max(0.70, min(1.00, _f(sr.get("qwl_data_reliability_avg"), 1.0))))
    qwl_n_b = float(max(0.0, _f(sb.get("qwl_n"), 0.0)))
    qwl_n_r = float(max(0.0, _f(sr.get("qwl_n"), 0.0)))
    # Backward compatibility: derive per-team QWL from journal when artifact was built
    # before team-level QWL fields existed.
    if qwl_n_b <= 0.0 or qwl_n_r <= 0.0:
        d_b = _derive_qwl_from_journal(kb)
        d_r = _derive_qwl_from_journal(kr)
        if qwl_n_b <= 0.0 and isinstance(d_b, dict) and d_b:
            qwl_adj_b = float(_f(d_b.get("qwl_signed_avg"), qwl_adj_b))
            qwl_sur_b = float(_f(d_b.get("qwl_surplus_signed_avg"), qwl_sur_b))
            qwl_tempo_b = float(_f(d_b.get("qwl_tempo_signed_avg"), qwl_tempo_b))
            qwl_conv_b = float(_f(d_b.get("qwl_conversion_signed_avg"), qwl_conv_b))
            qwl_throw_b = float(_f(d_b.get("qwl_throw_signed_avg"), qwl_throw_b))
            qwl_cb_b = float(_f(d_b.get("qwl_comeback_signed_avg"), qwl_cb_b))
            qwl_rel_b = float(max(0.70, min(1.00, _f(d_b.get("qwl_data_reliability_avg"), qwl_rel_b))))
            qwl_n_b = float(max(0.0, _f(d_b.get("qwl_n"), qwl_n_b)))
        if qwl_n_r <= 0.0 and isinstance(d_r, dict) and d_r:
            qwl_adj_r = float(_f(d_r.get("qwl_signed_avg"), qwl_adj_r))
            qwl_sur_r = float(_f(d_r.get("qwl_surplus_signed_avg"), qwl_sur_r))
            qwl_tempo_r = float(_f(d_r.get("qwl_tempo_signed_avg"), qwl_tempo_r))
            qwl_conv_r = float(_f(d_r.get("qwl_conversion_signed_avg"), qwl_conv_r))
            qwl_throw_r = float(_f(d_r.get("qwl_throw_signed_avg"), qwl_throw_r))
            qwl_cb_r = float(_f(d_r.get("qwl_comeback_signed_avg"), qwl_cb_r))
            qwl_rel_r = float(max(0.70, min(1.00, _f(d_r.get("qwl_data_reliability_avg"), qwl_rel_r))))
            qwl_n_r = float(max(0.0, _f(d_r.get("qwl_n"), qwl_n_r)))
    qwl_n_h = float((2.0 * qwl_n_b * qwl_n_r) / (qwl_n_b + qwl_n_r)) if qwl_n_b > 0.0 and qwl_n_r > 0.0 else float(max(qwl_n_b, qwl_n_r))
    w_qwl_n = float(max(0.0, min(1.0, qwl_n_h / (qwl_n_h + 10.0)))) if qwl_n_h > 0.0 else 0.0
    w_qwl_rel = float(max(0.70, min(1.00, math.sqrt(max(0.0, qwl_rel_b) * max(0.0, qwl_rel_r)))))
    qwl_surplus_mix = float(max(0.0, min(1.0, _f(_art_params.get("qwl_surplus_mix"), 0.70))))
    qwl_quality_k = float(max(0.0, min(0.65, _f(_art_params.get("quality_k_boost"), 0.22))))
    qwl_edge_core = float((qwl_adj_b - qwl_adj_r))
    qwl_edge_surplus = float((qwl_sur_b - qwl_sur_r))
    qwl_edge_secondary = float(
        0.35 * (qwl_tempo_b - qwl_tempo_r)
        + 0.35 * (qwl_conv_b - qwl_conv_r)
        + 0.20 * (qwl_cb_b - qwl_cb_r)
        + 0.10 * (qwl_throw_b - qwl_throw_r)
    )
    qwl_edge = float(
        max(
            -2.0,
            min(
                2.0,
                ((1.0 - qwl_surplus_mix) * qwl_edge_core)
                + (qwl_surplus_mix * qwl_edge_surplus)
                + (0.25 * qwl_edge_secondary),
            ),
        )
    )
    qwl_logit_shift_raw = float(0.80 * qwl_quality_k * w_qwl_n * w_qwl_rel * qwl_edge)
    qwl_logit_shift = float(max(-0.45, min(0.45, qwl_logit_shift_raw)))
    qwl_state_boost = float(1.0 + min(0.25, abs(qwl_logit_shift)))

    if isinstance(out.get("qwl_trace"), dict):
        out["qwl_trace"]["team_edge"] = {
            "edge_core": float(qwl_edge_core),
            "edge_surplus": float(qwl_edge_surplus),
            "edge_secondary": float(qwl_edge_secondary),
            "edge_total": float(qwl_edge),
            "n_blue": int(qwl_n_b),
            "n_red": int(qwl_n_r),
            "n_harmonic": float(qwl_n_h),
            "w_n": float(w_qwl_n),
            "w_reliability": float(w_qwl_rel),
            "quality_k_boost": float(qwl_quality_k),
            "surplus_mix": float(qwl_surplus_mix),
            "logit_shift": float(qwl_logit_shift),
        }

    den = float(max(1.0, math.sqrt(1.0 + sig_b * sig_b + sig_r * sig_r)))
    z_state = float(((mu_b - mu_r) / den) + qwl_logit_shift)
    p_state = float(_clip_prob(_safe_sigmoid(z_state)))

    gp_h = float((2.0 * gp_b * gp_r) / (gp_b + gp_r)) if gp_b > 0 and gp_r > 0 else float(max(gp_b, gp_r))
    w_gp = float(max(0.10, min(1.0, gp_h / (gp_h + 18.0)))) if gp_h > 0 else 0.10
    rd_pair = float(math.sqrt(sig_b * sig_b + sig_r * sig_r))
    w_sigma = float(max(0.10, min(1.0, 1.0 / (1.0 + (rd_pair / 1.10) ** 2))))
    w_state_conf = float(max(0.10, min(1.0, w_gp * w_sigma)))

    # League coverage confidence from artifact (data-driven):
    # sparse leagues should not let A-real dominate map odds.
    by_league = state_artifact.get("by_league") if isinstance(state_artifact.get("by_league"), dict) else {}
    cur_lg_b = str(league_blue or "").strip().upper()
    cur_lg_r = str(league_red or "").strip().upper()
    # Prefer same-league matchups; cross-league uses the lower coverage side.
    cov_b = by_league.get(cur_lg_b) if isinstance(by_league, dict) and cur_lg_b else None
    cov_r = by_league.get(cur_lg_r) if isinstance(by_league, dict) and cur_lg_r else None
    def _league_cov_weight(c: Any) -> float:
        if not isinstance(c, dict):
            return 0.82
        teams_n = max(0.0, _f(c.get("teams"), 0.0))
        games_n = max(0.0, _f(c.get("games"), 0.0))
        if teams_n <= 0.0 or games_n <= 0.0:
            return 0.80
        games_per_team = float(games_n / max(1.0, teams_n))
        base = float(max(0.25, min(1.0, games_per_team / (games_per_team + 10.0))))
        # Very small league coverage should be treated carefully.
        if teams_n < 4.0:
            base = min(base, 0.72)
        return float(max(0.25, min(1.0, base)))

    if cur_lg_b and cur_lg_r and cur_lg_b == cur_lg_r:
        w_league_cov = _league_cov_weight(cov_b)
    else:
        w_league_cov = min(_league_cov_weight(cov_b), _league_cov_weight(cov_r))

    # State recency confidence: stale states should lose influence (Glicko-like intuition).
    art_last_date = None
    if isinstance(_art_meta, dict):
        _lpg = _art_meta.get("last_processed_game")
        if isinstance(_lpg, dict):
            art_last_date = _safe_date(_lpg.get("date"))
    d_b = _safe_date(sb.get("last_date"))
    d_r = _safe_date(sr.get("last_date"))
    days_b = float((art_last_date - d_b).days) if (art_last_date and d_b) else float("nan")
    days_r = float((art_last_date - d_r).days) if (art_last_date and d_r) else float("nan")
    stale_days = max(days_b if _isfinite(days_b) else 0.0, days_r if _isfinite(days_r) else 0.0)
    # 0-14d: near full trust; 30d+: significant decay; floor keeps some signal.
    w_recency_state = float(max(0.35, min(1.0, 1.0 / (1.0 + (max(0.0, stale_days) / 18.0) ** 1.35))))

    tu = trace_uncertainty_info if isinstance(trace_uncertainty_info, dict) else {}
    n_eff_unc = max(0.0, _f(tu.get("n_eff"), 0.0))
    prior_unc = max(1.0, _f(tu.get("prior_games"), 22.0))
    w_unc = max(0.20, min(1.0, n_eff_unc / (n_eff_unc + prior_unc))) if n_eff_unc > 0 else 0.20

    ev_team = max(0.0, _f(n_team_eff, 0.0))
    ev_players = max(0.0, _f(n_players_eff, 0.0))
    ev_total = float(ev_team + 0.65 * ev_players)
    w_evidence = max(0.25, min(1.0, ev_total / (ev_total + 14.0))) if ev_total > 0 else 0.25

    fu = trace_fusion_info if isinstance(trace_fusion_info, dict) else {}
    roster_changed = bool(fu.get("roster_changed", False))
    pg = trace_patch_guard_info if isinstance(trace_patch_guard_info, dict) else {}
    patch_status = str(pg.get("status") or "").strip().lower()
    patch_shrink = max(0.0, min(1.0, _f(pg.get("shrink"), 0.0)))
    w_patch = 1.0
    if patch_status in {"refit", "refit_needed", "changed"}:
        w_patch = max(0.55, min(1.0, 1.0 - 0.65 * patch_shrink))
    elif bool(pg.get("applied", False)):
        w_patch = max(0.70, min(1.0, 1.0 - 0.45 * patch_shrink))

    w_roster = 0.80 if roster_changed else 1.0

    # League-drift confidence:
    # if a team state was built mostly in another league, reduce state influence in this matchup.
    # In interregional spots, a shared recent international event (for example both teams coming
    # from the same short tournament) should not be allowed to compress a clearly stronger team
    # all the way back toward even.
    sb_last_lg = str(sb.get("last_league") or "").strip().upper()
    sr_last_lg = str(sr.get("last_league") or "").strip().upper()
    interregional = bool(cur_lg_b and cur_lg_r and cur_lg_b != cur_lg_r)
    blue_league_mismatch = bool(sb_last_lg and cur_lg_b and sb_last_lg != cur_lg_b)
    red_league_mismatch = bool(sr_last_lg and cur_lg_r and sr_last_lg != cur_lg_r)
    mismatch_count = int(blue_league_mismatch) + int(red_league_mismatch)
    shared_last_league = bool(sb_last_lg and sr_last_lg and sb_last_lg == sr_last_lg)
    shared_foreign_last_league = bool(interregional and shared_last_league and sb_last_lg not in {cur_lg_b, cur_lg_r})
    tier_gap = abs(_tier_rank(cur_lg_b) - _tier_rank(cur_lg_r)) if interregional else 0
    strength_gap = _strength_gap(cur_lg_b, cur_lg_r) if interregional else 0.0
    cross_tier_interregional = bool(interregional and tier_gap >= 1)
    cross_strength_interregional = bool(interregional and strength_gap >= 0.16)
    art_last_lg = ""
    if isinstance(_art_meta, dict):
        _lpg = _art_meta.get("last_processed_game")
        if isinstance(_lpg, dict):
            art_last_lg = str(_lpg.get("league") or "").strip().upper()
    recent_shared_event = bool(shared_foreign_last_league and art_last_lg and art_last_lg == sb_last_lg)
    if not interregional:
        if mismatch_count <= 0:
            w_league = 1.0
        elif mismatch_count == 1:
            w_league = 0.84
        else:
            w_league = 0.72
    else:
        if mismatch_count <= 0:
            w_league = 0.82
        elif mismatch_count == 1:
            w_league = 0.60
        else:
            w_league = 0.44
        if cross_tier_interregional:
            w_league *= (0.72 if tier_gap == 1 else 0.60)
        if cross_strength_interregional:
            w_league *= (0.78 if strength_gap < 0.22 else 0.66)
        if shared_foreign_last_league:
            w_league *= 0.74
        if recent_shared_event:
            w_league *= 0.85
    w_league = float(max(0.18, min(1.0, w_league)))

    w_cons = 1.0
    drift_pp = _f(consistency_guard_drift_pp)
    if bool(consistency_guard_triggered) and _isfinite(drift_pp):
        w_cons = max(0.60, min(1.0, 1.0 - min(0.40, abs(drift_pp) / 45.0)))

    # Team state reliability from roster continuity inside artifact.
    # Low continuity over enough observations means the stored team-state is noisier.
    rb_avg = float(max(0.0, min(1.0, _f(sb.get("roster_cont_avg"), 1.0))))
    rr_avg = float(max(0.0, min(1.0, _f(sr.get("roster_cont_avg"), 1.0))))
    rb_n = max(0.0, _f(sb.get("roster_cont_n"), 0.0))
    rr_n = max(0.0, _f(sr.get("roster_cont_n"), 0.0))
    rb_conf = float(max(0.0, min(1.0, rb_n / (rb_n + 6.0))))
    rr_conf = float(max(0.0, min(1.0, rr_n / (rr_n + 6.0))))
    rb_eff = float((1.0 - rb_conf) + (rb_conf * rb_avg))
    rr_eff = float((1.0 - rr_conf) + (rr_conf * rr_avg))
    roster_pair = float(max(0.0, min(1.0, math.sqrt(max(0.0, rb_eff) * max(0.0, rr_eff)))))
    w_roster_state = float(max(0.55, min(1.0, 0.55 + 0.45 * roster_pair)))

    z_target = float(_safe_logit(p_state))
    z_delta_raw = float(z_target - z_old)

    # v8: precision-like blend in logit space (closer to a Bayesian fusion intuition).
    # Model precision proxies the confidence in current p_old; state precision proxies confidence in state estimate.
    model_prec = float(max(0.05, min(4.0, 0.35 + 1.90 * w_unc * w_evidence * w_cons)))
    state_prec = float(
        max(
            0.02,
            min(
                4.5,
                0.20
                + 2.40
                * w_state_conf
                * w_patch
                * w_roster
                * w_recency_state
                * w_league
                * w_league_cov
                * w_roster_state,
            ),
        )
    )
    state_interregional_damp = 1.0
    if interregional:
        state_interregional_damp *= 0.82
    if cross_tier_interregional:
        state_interregional_damp *= (0.74 if tier_gap == 1 else 0.62)
    if cross_strength_interregional:
        state_interregional_damp *= (0.76 if strength_gap < 0.22 else 0.64)
    if shared_foreign_last_league:
        state_interregional_damp *= 0.72
    if recent_shared_event:
        state_interregional_damp *= 0.88
    state_prec = float(max(0.02, min(4.5, state_prec * state_interregional_damp)))
    state_prec = float(max(0.02, min(4.5, state_prec * qwl_state_boost)))
    disagree_abs = abs(z_delta_raw)
    disagree_pen = 1.0
    if disagree_abs > 0.90:
        disagree_pen *= max(0.55, min(1.0, 1.0 - ((disagree_abs - 0.90) / 3.0)))
    if w_unc < 0.45 or w_evidence < 0.45:
        disagree_pen *= 0.90
    state_prec_eff = float(max(0.02, min(4.5, state_prec * disagree_pen)))
    total_prec = float(max(1e-6, model_prec + state_prec_eff))
    w_prec_state = float(max(0.02, min(0.92, state_prec_eff / total_prec)))

    # Keep a recency-aware floor so the state signal never vanishes completely when the artifact is valid.
    w_floor = 0.16
    if gp_h > 0:
        w_floor = float(min(0.50, w_floor + 0.16 * (gp_h / (gp_h + 16.0))))
    w_floor_eff = float(max(0.06, min(0.50, w_floor * (0.55 + 0.45 * w_recency_state))))
    if interregional:
        w_floor_eff *= 0.78
    if cross_tier_interregional:
        w_floor_eff *= (0.78 if tier_gap == 1 else 0.66)
    if cross_strength_interregional:
        w_floor_eff *= (0.82 if strength_gap < 0.22 else 0.72)
    if shared_foreign_last_league:
        w_floor_eff *= 0.62
    if recent_shared_event:
        w_floor_eff *= 0.88
    w_floor_eff = float(max(0.04, min(0.50, w_floor_eff)))
    w_final = float(max(w_floor_eff, w_prec_state))

    z_delta = float(z_delta_raw)
    move_cap = 0.90
    if gp_h >= 40 and rd_pair <= 0.75:
        move_cap = 1.20
    if interregional:
        move_cap *= 0.82
    if cross_tier_interregional:
        move_cap *= (0.72 if tier_gap == 1 else 0.60)
    if cross_strength_interregional:
        move_cap *= (0.78 if strength_gap < 0.22 else 0.66)
    if shared_foreign_last_league:
        move_cap *= 0.75
    if roster_changed:
        move_cap *= 0.85
    z_delta = float(max(-move_cap, min(move_cap, z_delta)))
    z_new = float(z_old + (w_final * z_delta))
    p_new = float(_clip_prob(_safe_sigmoid(z_new)))
    if cross_strength_interregional:
        stronger_side = "blue" if float(league_strength_score(str(league_blue or ""))) >= float(league_strength_score(str(league_red or ""))) else "red"
        before_stronger = float(p_old if stronger_side == "blue" else (1.0 - p_old))
        after_stronger = float(p_new if stronger_side == "blue" else (1.0 - p_new))
        adverse_pp = float(max(0.0, (before_stronger - after_stronger) * 100.0))
        adverse_cap_pp = 1.1 if strength_gap < 0.18 else 0.9
        if adverse_pp > adverse_cap_pp:
            clipped_stronger = float(before_stronger - (adverse_cap_pp / 100.0))
            clipped_stronger = max(1e-6, min(1.0 - 1e-6, clipped_stronger))
            p_new = float(clipped_stronger if stronger_side == "blue" else (1.0 - clipped_stronger))
    try:
        p_series_new = float(_prob_win_series(p_new, int(bo or 1)))
    except Exception:
        p_series_new = float("nan")

    out.update(
        {
            "applied": True,
            "reason": "team_state_artifact_mu_sigma",
            "p_map_new": p_new,
            "p_series_new": (p_series_new if _isfinite(p_series_new) else None),
            "diff_pp_map": float(100.0 * (p_new - p_old)),
            "diff_pp_series": (
                float(100.0 * (p_series_new - _f(p_series_old)))
                if _isfinite(p_series_old) and _isfinite(p_series_new)
                else None
            ),
            "weights": {
                "state_conf": float(w_state_conf),
                "uncertainty": float(w_unc),
                "evidence": float(w_evidence),
                "patch": float(w_patch),
                "roster": float(w_roster),
                "league": float(w_league),
                "interregional_state_damp": float(state_interregional_damp),
                "league_coverage": float(w_league_cov),
                "roster_state": float(w_roster_state),
                "qwl_n": float(w_qwl_n),
                "qwl_reliability": float(w_qwl_rel),
                "qwl_state_boost": float(qwl_state_boost),
                "qwl_logit_shift": float(qwl_logit_shift),
                "consistency": float(w_cons),
                "state_recency": float(w_recency_state),
                "model_precision": float(model_prec),
                "state_precision": float(state_prec),
                "state_precision_effective": float(state_prec_eff),
                "disagree_penalty": float(disagree_pen),
                "precision_state_weight": float(w_prec_state),
                "floor": float(w_floor),
                "floor_effective": float(w_floor_eff),
                "final": float(w_final),
            },
            "state": {
                "blue": {"team": str(sb.get("team") or team_blue), "mu": float(mu_b), "sigma": float(sig_b), "gp": int(gp_b)},
                "red": {"team": str(sr.get("team") or team_red), "mu": float(mu_r), "sigma": float(sig_r), "gp": int(gp_r)},
                "rd_pair": float(rd_pair),
                "gp_harmonic": float(gp_h),
                "stale_days_max": float(stale_days),
                "blue_last_league": str(sb_last_lg),
                "red_last_league": str(sr_last_lg),
                "blue_league_mismatch": bool(blue_league_mismatch),
                "red_league_mismatch": bool(red_league_mismatch),
                "interregional": bool(interregional),
                "cross_tier_interregional": bool(cross_tier_interregional),
                "cross_strength_interregional": bool(cross_strength_interregional),
                "tier_gap": int(tier_gap),
                "strength_gap": float(strength_gap),
                "shared_last_league": bool(shared_last_league),
                "shared_foreign_last_league": bool(shared_foreign_last_league),
                "recent_shared_event": bool(recent_shared_event),
                "blue_roster_cont_avg": float(rb_avg),
                "red_roster_cont_avg": float(rr_avg),
                "blue_roster_cont_n": int(rb_n),
                "red_roster_cont_n": int(rr_n),
                "p_state": float(p_state),
                "artifact_version": str(state_artifact.get("version") or ""),
                "artifact_update_mode": str(_art_meta.get("update_mode") or "") if isinstance(_art_meta, dict) else "",
            },
            "inputs": {
                "team_blue": str(team_blue or ""),
                "team_red": str(team_red or ""),
                "team_blue_key": str(kb),
                "team_red_key": str(kr),
                "league_blue": str(league_blue or ""),
                "league_red": str(league_red or ""),
                "roster_changed": bool(roster_changed),
                "patch_status": str(patch_status),
                "patch_shrink": float(patch_shrink),
                "n_team_eff": float(ev_team),
                "n_players_eff": float(ev_players),
                "n_eff_uncertainty": float(n_eff_unc),
            },
            "odds": {
                "teamA_map_old": _odd_from_p(p_old),
                "teamA_map_state": _odd_from_p(p_state),
                "teamA_map_new": _odd_from_p(p_new),
                "teamA_series_old": _odd_from_p(_f(p_series_old)) if _isfinite(p_series_old) else None,
                "teamA_series_new": _odd_from_p(p_series_new) if _isfinite(p_series_new) else None,
            },
        }
    )
    return out
