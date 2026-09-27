# -*- coding: utf-8 -*-
"""Matematica de estado do time (Glicko-lite + Bayes) — fatiado de
state_team_artifact.py em 16/08/2026.

Por que separar: o arquivo tinha 1477 linhas e UMA funcao de 928
(build_team_state_artifact, complexidade ciclomatica 236 — a pior do projeto).
Os 19 helpers aqui sao logica pura e coesa: inflacao de sigma por tempo/regime/
roster, update bayesiano, recentragem por troca de patch/liga/elenco, k efetivo.
Nenhum deles conhece o builder (dependencia num sentido so, verificado por AST),
entao o corte e limpo.

Sem Streamlit, sem I/O: da para testar cada peca isolada. O builder continua em
state_team_artifact.py e importa daqui.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


from .players_layer import POS_CANON
from .utils import safe_float as _f, clip as _clip, sigmoid as _sigmoid
from .team_artifact_normalize import _norm_team_key, _norm_upper_key
from .team_artifact_validate import _param_map_sig

from .team_artifact_utils import (
    _days_between_iso,
    _league_k_multiplier,
    _running_avg,
    _safe_nonneg_int,
)



def _lineup_continuity(prev_lineup: Optional[Dict[str, str]], cur_lineup: Optional[Dict[str, str]]) -> Optional[float]:
    if not isinstance(prev_lineup, dict) or not isinstance(cur_lineup, dict):
        return None
    overlap = 0
    seen = 0
    for pos in POS_CANON:
        p0 = str(prev_lineup.get(pos, "") or "").strip()
        p1 = str(cur_lineup.get(pos, "") or "").strip()
        if not p0 or not p1:
            continue
        seen += 1
        if p0 == p1:
            overlap += 1
    if seen <= 0:
        return None
    return float(overlap / float(seen))


def _update_team_qwl_stats(
    st: "TeamState",
    *,
    qwl_signed: float,
    qwl_surplus_signed: float,
    qwl_tempo_signed: float,
    qwl_conversion_signed: float,
    qwl_throw_signed: float,
    qwl_comeback_signed: float,
    qwl_data_reliability: float,
) -> None:
    n0 = int(max(0, int(st.qwl_n)))
    st.qwl_signed_avg = float(_running_avg(float(st.qwl_signed_avg), n0, float(qwl_signed)))
    st.qwl_surplus_signed_avg = float(_running_avg(float(st.qwl_surplus_signed_avg), n0, float(qwl_surplus_signed)))
    st.qwl_tempo_signed_avg = float(_running_avg(float(st.qwl_tempo_signed_avg), n0, float(qwl_tempo_signed)))
    st.qwl_conversion_signed_avg = float(_running_avg(float(st.qwl_conversion_signed_avg), n0, float(qwl_conversion_signed)))
    st.qwl_throw_signed_avg = float(_running_avg(float(st.qwl_throw_signed_avg), n0, float(qwl_throw_signed)))
    st.qwl_comeback_signed_avg = float(_running_avg(float(st.qwl_comeback_signed_avg), n0, float(qwl_comeback_signed)))
    st.qwl_data_reliability_avg = float(
        _clip(_running_avg(float(st.qwl_data_reliability_avg), n0, float(qwl_data_reliability)), 0.70, 1.00)
    )
    st.qwl_n = int(n0 + 1)


@dataclass
class TeamState:
    team: str
    key: str
    mu: float = 0.0
    sigma: float = 1.00
    gp: int = 0
    wins: int = 0
    losses: int = 0
    last_date: Optional[str] = None
    last_patch: str = ""
    last_league: str = ""
    last_split: str = ""
    last_playoffs: Optional[bool] = None
    last_lineup: Optional[Dict[str, str]] = None
    roster_cont_avg: float = 1.0
    roster_cont_n: int = 0
    qwl_signed_avg: float = 0.0
    qwl_surplus_signed_avg: float = 0.0
    qwl_tempo_signed_avg: float = 0.0
    qwl_conversion_signed_avg: float = 0.0
    qwl_throw_signed_avg: float = 0.0
    qwl_comeback_signed_avg: float = 0.0
    qwl_data_reliability_avg: float = 1.0
    qwl_n: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "key": self.key,
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "gp": int(self.gp),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "last_date": self.last_date,
            "last_patch": str(self.last_patch or ""),
            "last_league": str(self.last_league or ""),
            "last_split": str(self.last_split or ""),
            "last_playoffs": (bool(self.last_playoffs) if self.last_playoffs is not None else None),
            "last_lineup": (dict(self.last_lineup) if isinstance(self.last_lineup, dict) else None),
            "roster_cont_avg": float(self.roster_cont_avg),
            "roster_cont_n": int(self.roster_cont_n),
            "qwl_signed_avg": float(self.qwl_signed_avg),
            "qwl_surplus_signed_avg": float(self.qwl_surplus_signed_avg),
            "qwl_tempo_signed_avg": float(self.qwl_tempo_signed_avg),
            "qwl_conversion_signed_avg": float(self.qwl_conversion_signed_avg),
            "qwl_throw_signed_avg": float(self.qwl_throw_signed_avg),
            "qwl_comeback_signed_avg": float(self.qwl_comeback_signed_avg),
            "qwl_data_reliability_avg": float(self.qwl_data_reliability_avg),
            "qwl_n": int(self.qwl_n),
        }


def _team_state_from_dict(d: Dict[str, Any]) -> Optional[TeamState]:
    if not isinstance(d, dict):
        return None
    key = _norm_team_key(d.get("key") or d.get("team"))
    if not key:
        return None
    last_lineup_raw = d.get("last_lineup") if isinstance(d.get("last_lineup"), dict) else None
    last_lineup = None
    if isinstance(last_lineup_raw, dict):
        last_lineup = {str(k): str(v) for k, v in last_lineup_raw.items() if str(v or "").strip()}
    return TeamState(
        team=str(d.get("team") or "").strip() or key,
        key=key,
        mu=float(_f(d.get("mu"), 0.0)),
        sigma=float(_clip(_f(d.get("sigma"), 1.0), 0.05, 10.0)),
        gp=_safe_nonneg_int(d.get("gp"), 0),
        wins=_safe_nonneg_int(d.get("wins"), 0),
        losses=_safe_nonneg_int(d.get("losses"), 0),
        last_date=(str(d.get("last_date")) if d.get("last_date") else None),
        last_patch=str(d.get("last_patch") or ""),
        last_league=str(d.get("last_league") or ""),
        last_split=str(d.get("last_split") or ""),
        last_playoffs=(None if d.get("last_playoffs") is None else bool(d.get("last_playoffs"))),
        last_lineup=last_lineup,
        roster_cont_avg=float(_clip(_f(d.get("roster_cont_avg"), 1.0), 0.0, 1.0)),
        roster_cont_n=_safe_nonneg_int(d.get("roster_cont_n"), 0),
        qwl_signed_avg=float(_clip(_f(d.get("qwl_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_surplus_signed_avg=float(_clip(_f(d.get("qwl_surplus_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_tempo_signed_avg=float(_clip(_f(d.get("qwl_tempo_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_conversion_signed_avg=float(_clip(_f(d.get("qwl_conversion_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_throw_signed_avg=float(_clip(_f(d.get("qwl_throw_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_comeback_signed_avg=float(_clip(_f(d.get("qwl_comeback_signed_avg"), 0.0), -1.0, 1.0)),
        qwl_data_reliability_avg=float(_clip(_f(d.get("qwl_data_reliability_avg"), 1.0), 0.70, 1.0)),
        qwl_n=_safe_nonneg_int(d.get("qwl_n"), 0),
    )


def _params_match_seed(
    seed_artifact: Optional[Dict[str, Any]],
    *,
    base_k: float,
    base_sigma: float,
    min_sigma: float,
    max_sigma: float,
    sigma_decay: float,
    inactivity_sigma_per_day: float,
    patch_sigma_bump: float,
    roster_sigma_bump: float,
    patch_mu_recenter: float,
    roster_mu_recenter: float,
    league_switch_mu_recenter: float,
    roster_cont_hard_reset: float,
    roster_hard_reset_extra_sigma: float,
    quality_k_boost: float,
    qwl_expected_base: float,
    qwl_expected_slope: float,
    qwl_surplus_mix: float,
    qwl_tempo_weight: float,
    qwl_conversion_weight: float,
    qwl_throw_weight: float,
    qwl_comeback_weight: float,
    qwl_phase_dynamic_strength: float,
    qwl_opponent_context_mix: float,
    mu_update_mode: str,
    sigma_update_mode: str,
    bayes_info_scale: float,
    bayes_mu_q_strength: float,
    bayes_info_scale_by_league: Optional[Dict[str, float]],
    bayes_info_scale_by_patch: Optional[Dict[str, float]],
    quality_k_boost_by_league: Optional[Dict[str, float]],
    quality_k_boost_by_patch: Optional[Dict[str, float]],
    qwl_surplus_mix_by_league: Optional[Dict[str, float]],
    qwl_surplus_mix_by_patch: Optional[Dict[str, float]],
    qwl_throw_weight_by_league: Optional[Dict[str, float]],
    qwl_throw_weight_by_patch: Optional[Dict[str, float]],
    qwl_comeback_weight_by_league: Optional[Dict[str, float]],
    qwl_comeback_weight_by_patch: Optional[Dict[str, float]],
    qwl_phase_dynamic_strength_by_league: Optional[Dict[str, float]],
    qwl_phase_dynamic_strength_by_patch: Optional[Dict[str, float]],
    qwl_opponent_context_mix_by_league: Optional[Dict[str, float]],
    qwl_opponent_context_mix_by_patch: Optional[Dict[str, float]],
    include_players_for_roster: bool,
) -> bool:
    if not isinstance(seed_artifact, dict):
        return False
    sp = seed_artifact.get("params") if isinstance(seed_artifact.get("params"), dict) else {}
    if not isinstance(sp, dict):
        return False
    checks = {
        "base_k": float(base_k),
        "base_sigma": float(base_sigma),
        "min_sigma": float(min_sigma),
        "max_sigma": float(max_sigma),
        "sigma_decay": float(sigma_decay),
        "inactivity_sigma_per_day": float(inactivity_sigma_per_day),
        "patch_sigma_bump": float(patch_sigma_bump),
        "roster_sigma_bump": float(roster_sigma_bump),
        "patch_mu_recenter": float(patch_mu_recenter),
        "roster_mu_recenter": float(roster_mu_recenter),
        "league_switch_mu_recenter": float(league_switch_mu_recenter),
        "roster_cont_hard_reset": float(roster_cont_hard_reset),
        "roster_hard_reset_extra_sigma": float(roster_hard_reset_extra_sigma),
        "quality_k_boost": float(quality_k_boost),
        "qwl_expected_base": float(qwl_expected_base),
        "qwl_expected_slope": float(qwl_expected_slope),
        "qwl_surplus_mix": float(qwl_surplus_mix),
        "qwl_tempo_weight": float(qwl_tempo_weight),
        "qwl_conversion_weight": float(qwl_conversion_weight),
        "qwl_throw_weight": float(qwl_throw_weight),
        "qwl_comeback_weight": float(qwl_comeback_weight),
        "qwl_phase_dynamic_strength": float(qwl_phase_dynamic_strength),
        "qwl_opponent_context_mix": float(qwl_opponent_context_mix),
        "bayes_info_scale": float(bayes_info_scale),
        "bayes_mu_q_strength": float(bayes_mu_q_strength),
    }
    for k, v in checks.items():
        sv = _f(sp.get(k), float("nan"))
        if not math.isfinite(sv):
            return False
        if not math.isfinite(sv) or abs(sv - v) > 1e-9:
            return False
    if bool(sp.get("include_players_for_roster", False)) != bool(include_players_for_roster):
        return False
    if str(sp.get("mu_update_mode") or "legacy_k_v1").strip().lower() != str(mu_update_mode or "").strip().lower():
        return False
    if str(sp.get("sigma_update_mode") or "legacy_decay_v1").strip().lower() != str(sigma_update_mode or "").strip().lower():
        return False
    if list(sp.get("bayes_info_scale_by_league") or []) != [list(x) for x in _param_map_sig(bayes_info_scale_by_league)]:
        return False
    if list(sp.get("bayes_info_scale_by_patch") or []) != [list(x) for x in _param_map_sig(bayes_info_scale_by_patch)]:
        return False
    if list(sp.get("quality_k_boost_by_league") or []) != [list(x) for x in _param_map_sig(quality_k_boost_by_league)]:
        return False
    if list(sp.get("quality_k_boost_by_patch") or []) != [list(x) for x in _param_map_sig(quality_k_boost_by_patch)]:
        return False
    if list(sp.get("qwl_surplus_mix_by_league") or []) != [list(x) for x in _param_map_sig(qwl_surplus_mix_by_league)]:
        return False
    if list(sp.get("qwl_surplus_mix_by_patch") or []) != [list(x) for x in _param_map_sig(qwl_surplus_mix_by_patch)]:
        return False
    if list(sp.get("qwl_throw_weight_by_league") or []) != [list(x) for x in _param_map_sig(qwl_throw_weight_by_league)]:
        return False
    if list(sp.get("qwl_throw_weight_by_patch") or []) != [list(x) for x in _param_map_sig(qwl_throw_weight_by_patch)]:
        return False
    if list(sp.get("qwl_comeback_weight_by_league") or []) != [list(x) for x in _param_map_sig(qwl_comeback_weight_by_league)]:
        return False
    if list(sp.get("qwl_comeback_weight_by_patch") or []) != [list(x) for x in _param_map_sig(qwl_comeback_weight_by_patch)]:
        return False
    if list(sp.get("qwl_phase_dynamic_strength_by_league") or []) != [list(x) for x in _param_map_sig(qwl_phase_dynamic_strength_by_league)]:
        return False
    if list(sp.get("qwl_phase_dynamic_strength_by_patch") or []) != [list(x) for x in _param_map_sig(qwl_phase_dynamic_strength_by_patch)]:
        return False
    if list(sp.get("qwl_opponent_context_mix_by_league") or []) != [list(x) for x in _param_map_sig(qwl_opponent_context_mix_by_league)]:
        return False
    if list(sp.get("qwl_opponent_context_mix_by_patch") or []) != [list(x) for x in _param_map_sig(qwl_opponent_context_mix_by_patch)]:
        return False
    return True


def _get_or_create_team(states: Dict[str, TeamState], team_name: str) -> TeamState:
    key = _norm_team_key(team_name)
    if not key:
        raise ValueError("empty_team_key")
    st = states.get(key)
    if st is None:
        st = TeamState(team=str(team_name or "").strip(), key=key)
        states[key] = st
    else:
        # Keep the most recent display name seen.
        if str(team_name or "").strip():
            st.team = str(team_name).strip()
    return st


def _inflate_sigma_for_time_and_regime(
    st: TeamState,
    *,
    cur_date_iso: str,
    cur_patch_key: str,
    inactivity_sigma_per_day: float,
    patch_sigma_bump: float,
    patch_sigma_mult: float,
    max_sigma: float,
) -> Dict[str, Any]:
    out = {"days_inactive": 0, "patch_switched": False}
    days_inactive = _days_between_iso(st.last_date, cur_date_iso)
    out["days_inactive"] = int(max(0, days_inactive))
    if days_inactive > 0:
        add = float(max(0.0, inactivity_sigma_per_day) * float(days_inactive))
        st.sigma = float(min(max_sigma, math.sqrt(float(st.sigma) ** 2 + add * add)))
    if st.last_patch and cur_patch_key and str(st.last_patch) != str(cur_patch_key):
        addp = float(max(0.0, patch_sigma_bump) * max(0.75, float(patch_sigma_mult)))
        st.sigma = float(min(max_sigma, math.sqrt(float(st.sigma) ** 2 + addp * addp)))
        out["patch_switched"] = True
    return out


def _inflate_sigma_for_roster(
    st: TeamState,
    *,
    cur_lineup: Optional[Dict[str, str]],
    roster_sigma_bump: float,
    max_sigma: float,
) -> Optional[float]:
    cont = _lineup_continuity(st.last_lineup, cur_lineup)
    if cont is None:
        return None
    miss = float(max(0.0, 1.0 - cont))
    if miss > 0.0:
        add = float(max(0.0, roster_sigma_bump) * miss)
        st.sigma = float(min(max_sigma, math.sqrt(float(st.sigma) ** 2 + add * add)))
    st.roster_cont_avg = float(
        ((st.roster_cont_avg * st.roster_cont_n) + float(cont)) / float(st.roster_cont_n + 1)
    )
    st.roster_cont_n = int(st.roster_cont_n) + 1
    return float(cont)


def _update_sigma_after_game(
    st: TeamState,
    *,
    base_sigma: float,
    sigma_decay: float,
    min_sigma: float,
    max_sigma: float,
    surprise: float = 0.0,
    sigma_update_mode: str = "legacy_decay_v1",
    info_term: float = 0.0,
    bayes_info_scale: float = 1.0,
) -> None:
    mode = str(sigma_update_mode or "legacy_decay_v1").strip().lower()
    if mode == "bayes_logit_v1":
        # Bayesian posterior variance update in logit space:
        # var_post = 1 / (1/var_prior + lambda * info_term)
        var_prior = float(max(1e-6, float(st.sigma) ** 2))
        info = float(max(0.0, info_term))
        lam = float(max(0.05, min(4.0, bayes_info_scale)))
        den = (1.0 / var_prior) + (lam * info)
        if den > 0.0 and math.isfinite(den):
            var_post = float(max(1e-6, 1.0 / den))
            st.sigma = float(_clip(math.sqrt(var_post), min_sigma, max_sigma))
        else:
            st.sigma = float(_clip(float(st.sigma), min_sigma, max_sigma))
        # Keep mild re-open for shocks/upsets, preserving adaptation speed.
        s = float(max(0.0, min(1.0, surprise)))
        if s > 0.0:
            shock = 0.05 * s
            st.sigma = float(min(max_sigma, math.sqrt((float(st.sigma) ** 2) + (shock * shock))))
        return

    # Legacy safe fallback (default behavior before probabilistic update).
    gp_next = int(st.gp) + 1
    target = float(max(min_sigma, min(max_sigma, float(base_sigma) / math.sqrt(gp_next + 1.0))))
    d = _clip(float(sigma_decay), 0.0, 1.0)
    st.sigma = float(_clip((float(st.sigma) * (1.0 - d)) + (target * d), min_sigma, max_sigma))
    s = float(max(0.0, min(1.0, surprise)))
    if s > 0.0:
        shock = 0.07 * s
        st.sigma = float(min(max_sigma, math.sqrt((float(st.sigma) ** 2) + (shock * shock))))


def _update_bayes_joint_after_game(
    blue: TeamState,
    red: TeamState,
    *,
    y_blue: float,
    p_blue: float,
    g_blue: float,
    g_red: float,
    denom: float,
    min_sigma: float,
    max_sigma: float,
    bayes_info_scale: float,
    surprise: float = 0.0,
    q_adjusted: float = 0.0,
    quality_k_boost_eff: float = 0.22,
    update_sigma: bool = True,
) -> Dict[str, float]:
    # Bayesian joint update (mean + variance) in logit space using one game
    # observation. This avoids heuristic K-only updates and keeps mu/sigma
    # coupled by information content of the same match.
    var_b = float(max(1e-6, float(blue.sigma) ** 2))
    var_r = float(max(1e-6, float(red.sigma) ** 2))
    den = float(max(1e-6, float(denom)))
    sens = float(((float(g_blue) + float(g_red)) * 0.5) / den)
    h_b = float(sens)
    h_r = float(-sens)

    p = float(_clip(float(p_blue), 1e-6, 1.0 - 1e-6))
    y = float(_clip(float(y_blue), 0.0, 1.0))
    info = float(max(1e-6, p * (1.0 - p)))
    lam = float(max(0.05, min(4.0, float(bayes_info_scale))))

    qk = float(_clip(float(quality_k_boost_eff), 0.0, 0.65))
    q_adj = float(_clip(float(q_adjusted), 0.0, 1.0))
    q_mul = float(1.0 + (qk * q_adj))
    meas_prec = float(lam * info * q_mul)

    proj_var = float((h_b * h_b * var_b) + (h_r * h_r * var_r))
    s_den = float(max(1e-6, 1.0 + (meas_prec * proj_var)))
    innov = float(y - p)

    k_b = float((var_b * h_b * meas_prec) / s_den)
    k_r = float((var_r * h_r * meas_prec) / s_den)
    delta_b = float(k_b * innov)
    delta_r = float(k_r * innov)

    blue.mu = float(blue.mu + delta_b)
    red.mu = float(red.mu + delta_r)

    var_b_post = float(var_b)
    var_r_post = float(var_r)
    if bool(update_sigma):
        var_b_post = float(max(1e-6, var_b - (((var_b * h_b) ** 2) * meas_prec / s_den)))
        var_r_post = float(max(1e-6, var_r - (((var_r * h_r) ** 2) * meas_prec / s_den)))

        s = float(_clip(float(surprise), 0.0, 1.0))
        if s > 0.0:
            shock = float(0.05 * s)
            var_b_post = float(var_b_post + (shock * shock))
            var_r_post = float(var_r_post + (shock * shock))

        blue.sigma = float(_clip(math.sqrt(var_b_post), min_sigma, max_sigma))
        red.sigma = float(_clip(math.sqrt(var_r_post), min_sigma, max_sigma))

    k_equiv = float(abs(delta_b) / max(1e-6, abs(innov)))
    return {
        "innov": float(innov),
        "k_blue": float(k_b),
        "k_red": float(k_r),
        "delta_blue": float(delta_b),
        "delta_red": float(delta_r),
        "k_equiv": float(k_equiv),
        "info_term": float(info),
        "meas_prec": float(meas_prec),
        "q_mul": float(q_mul),
        "var_blue_post": float(var_b_post),
        "var_red_post": float(var_r_post),
    }


def _league_prior_mu(states: Dict[str, TeamState], league: str) -> float:
    lg = str(league or "").strip().upper()
    if not lg:
        return 0.0
    vals: List[float] = []
    for st in states.values():
        if str(st.last_league or "").strip().upper() != lg:
            continue
        if int(st.gp or 0) <= 0:
            continue
        vals.append(float(st.mu))
    if not vals:
        return 0.0
    return float(sum(vals) / max(1, len(vals)))


def _recenter_mu(st: TeamState, *, prior_mu: float, shrink: float) -> float:
    sh = float(_clip(float(shrink), 0.0, 0.95))
    if sh <= 0.0:
        return 0.0
    prev = float(st.mu)
    st.mu = float(prior_mu + (prev - prior_mu) * (1.0 - sh))
    return float(prev - st.mu)


def _resolve_scoped_param(
    global_v: float,
    by_league: Dict[str, float],
    by_patch: Dict[str, float],
    league_v: str,
    patch_v: str,
) -> float:
    """Resolve um parâmetro global -> override por liga -> mistura por patch.

    Extraído de build_team_state_artifact (era nested, 19/07 refactor guiado por
    equivalence_check). Puro: liga sobrescreve o global; patch mistura 65/35 por
    cima. Chamado ~10x com escopos diferentes (info_scale, quality_k, qwl_*)."""
    out_v = float(global_v)
    lgk = _norm_upper_key(league_v)
    ptk = _norm_upper_key(patch_v)
    if lgk and lgk in by_league:
        out_v = float(by_league[lgk])
    if ptk and ptk in by_patch:
        out_v = float((0.65 * out_v) + (0.35 * float(by_patch[ptk])))
    return float(out_v)


def _glicko_lite_p_blue(*, mu_b: float, mu_r: float, sigma_b: float, sigma_r: float) -> Tuple[float, float, float, float]:
    """Fatores Glicko-lite do jogo: retorna (g_b, g_r, denom, p_blue).

    Extraído do loop de build_team_state_artifact (20/07, refactor supervisionado
    guiado por equivalence_check). É o CORE de cada jogo — a probabilidade que
    vira o erro que atualiza o estado. g_b/g_r encolhem a sensibilidade conforme
    o adversário é incerto e ALIMENTAM a atualização de precisão/mu depois (por
    isso são devolvidos, não só p_blue); denom normaliza pela variância conjunta."""
    g_r = float(1.0 / math.sqrt(1.0 + (3.0 * (float(sigma_r) ** 2) / (math.pi ** 2))))
    g_b = float(1.0 / math.sqrt(1.0 + (3.0 * (float(sigma_b) ** 2) / (math.pi ** 2))))
    denom = float(max(1e-6, math.sqrt(1.0 + (float(sigma_b) ** 2) + (float(sigma_r) ** 2))))
    p_blue = float(_sigmoid(((float(mu_b) - float(mu_r)) / denom) * ((g_b + g_r) * 0.5)))
    return g_b, g_r, denom, p_blue


def _effective_k(*, base_k: float, league: str, sigma_b: float, sigma_r: float, base_sigma: float) -> float:
    """Taxa de aprendizado efetiva do jogo (quanto o resultado move o estado).

    Extraído do loop. Ligas mais fortes movem mais (league_k_multiplier); alta
    incerteza combinada também permite passos maiores (unc_mul). Limitado a
    [0.04, 0.55]. Puro; equivalence_check prova 856 times idênticos."""
    lm = float(_league_k_multiplier(league))
    unc_mul = float(0.85 + 0.35 * min(1.0, (float(sigma_b) + float(sigma_r)) / max(1e-6, (2.0 * float(base_sigma)))))
    return float(max(0.04, min(0.55, float(base_k) * lm * unc_mul)))


def _resolve_effective_params(
    specs: Dict[str, Tuple[float, Dict[str, float], Dict[str, float], Optional[float], Optional[float]]],
    league: str,
    patch_k: str,
) -> Dict[str, float]:
    """Resolve TODOS os params efetivos (por liga/patch) de um jogo de uma vez.

    Extraído de build_team_state_artifact (era ~80 linhas de `x_eff =
    clip(resolve(...))` repetidas no loop). Data-driven: cada spec é
    (global, by_league, by_patch, lo, hi); lo=None -> sem clip. Puro; a rede
    de equivalência (equivalence_check) prova mu/sigma idênticos em 856 times."""
    out: Dict[str, float] = {}
    for name, (global_v, by_league, by_patch, lo, hi) in specs.items():
        v = _resolve_scoped_param(float(global_v), by_league, by_patch, league, patch_k)
        if lo is not None and hi is not None:
            v = _clip(v, float(lo), float(hi))
        out[name] = float(v)
    return out


def _apply_patch_switch_recenter(
    st: "TeamState",
    *,
    patch_switched: bool,
    patch_mu_recenter: float,
    patch_sigma_mult: float,
    league_prior_mu: float,
) -> Tuple[int, int]:
    """Recentra o mu do time ao trocar de patch. Retorna (switches, recenters).

    Extraído do loop de build (20/07, blocos stateful; equivalence_check prova
    856 times). Muta `st` via _recenter_mu — a ramificação sai do build."""
    if not bool(patch_switched):
        return 0, 0
    sh = float(_clip(float(patch_mu_recenter) * float(patch_sigma_mult), 0.0, 0.65))
    if sh > 0.0:
        _recenter_mu(st, prior_mu=float(league_prior_mu), shrink=sh)
        return 1, 1
    return 1, 0


def _apply_league_switch_recenter(
    st: "TeamState",
    *,
    league: str,
    league_switch_mu_recenter: float,
    league_prior_mu: float,
) -> int:
    """Recentra o mu ao time mudar de liga. Retorna o incremento do contador."""
    if not (st.last_league and league and str(st.last_league).strip().upper() != str(league).strip().upper()):
        return 0
    sh = float(_clip(float(league_switch_mu_recenter), 0.0, 0.55))
    if sh > 0.0:
        _recenter_mu(st, prior_mu=float(league_prior_mu), shrink=sh)
        return 1
    return 0


def _apply_roster_recenter(
    st: "TeamState",
    cc: Optional[float],
    *,
    roster_mu_recenter: float,
    roster_cont_hard_reset: float,
    roster_hard_reset_extra_sigma: float,
    max_sigma: float,
    league_prior_mu: float,
) -> Tuple[int, int, int, int]:
    """Recentra mu / infla sigma conforme a mudança de roster (continuidade cc).

    Retorna (events, events_with_change, hard_reset, mu_recenter). cc é a
    continuidade do lineup (1.0 = igual). Muta `st`."""
    if cc is None:
        return 0, 0, 0, 0
    if float(cc) >= 0.999:
        return 1, 0, 0, 0
    hard = 0
    murec = 0
    _miss = float(max(0.0, 1.0 - float(cc)))
    _gp_mul = float(1.0 + min(0.35, float(max(0.0, _f(st.gp, 0.0))) / 80.0))
    _ro_sh = float(_clip(float(roster_mu_recenter) * _miss * _gp_mul, 0.0, 0.80))
    if float(cc) <= float(roster_cont_hard_reset):
        _ro_sh = float(max(_ro_sh, _clip(float(roster_mu_recenter) + 0.20, 0.0, 0.85)))
        _extra = float(max(0.0, float(roster_hard_reset_extra_sigma)))
        if _extra > 0.0:
            st.sigma = float(min(float(max_sigma), math.sqrt(float(st.sigma) ** 2 + _extra * _extra)))
        hard = 1
    if _ro_sh > 0.0:
        _recenter_mu(st, prior_mu=float(league_prior_mu), shrink=_ro_sh)
        murec = 1
    return 1, 1, hard, murec


