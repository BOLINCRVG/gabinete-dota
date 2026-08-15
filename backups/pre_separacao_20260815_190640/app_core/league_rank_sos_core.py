"""League-internal SOS-adjusted ranking prior for ML map probability.

Runtime-only layer (no training artifact) that builds a compact same-league prior
from team_games and blends it into the current p_map. Goal: reduce distortions
from raw W/L when a team mostly loses to the league's top team(s).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd
from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _logit
from app_core.prob_math_core import safe_sigmoid as _sigmoid
from mlcore.utils import clip_safe as _clip


def _norm_margin(v: Any, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    x = _f(v, float("nan"))
    if not math.isfinite(x):
        return 0.0
    return float(max(-1.0, min(1.0, x / float(scale))))


def _pair_diff_value(row: pd.Series, stem: str) -> float:
    vf = _f(row.get(f"{stem}_for"), float("nan"))
    va = _f(row.get(f"{stem}_against"), float("nan"))
    if math.isfinite(vf) and math.isfinite(va):
        # Some inputs already store team-perspective deltas (opp ~= -for).
        # Prefer that representation when detected to avoid doubling the margin.
        if abs(vf + va) <= (0.12 * (abs(vf) + abs(va) + 1.0)):
            return float(vf)
        return float(vf - va)
    if math.isfinite(vf):
        return float(vf)
    if math.isfinite(va):
        return float(-va)
    return float("nan")


def _data_completeness_weight(v: Any) -> float:
    s = str(v or "").strip().lower()
    if s in {"complete", "full", "1", "true", "yes"}:
        return 1.0
    if s in {"partial", "partially_complete", "incomplete", "0", "false", "no"}:
        return 0.82
    return 0.92


def _first_objective_diff(row: pd.Series, name: str) -> float:
    ff = _f(row.get(f"{name}_for"), float("nan"))
    fa = _f(row.get(f"{name}_against"), float("nan"))
    if math.isfinite(ff) and math.isfinite(fa):
        return float(max(-1.0, min(1.0, ff - fa)))
    if math.isfinite(ff):
        if ff in (0.0, 1.0):
            return 1.0 if ff >= 0.5 else -1.0
        return float(max(-1.0, min(1.0, ff)))
    if math.isfinite(fa):
        if fa in (0.0, 1.0):
            return -1.0 if fa >= 0.5 else 1.0
        return float(max(-1.0, min(1.0, -fa)))
    return 0.0


def _recency_weights_from_dates(
    dates: pd.Series,
    *,
    half_life_days: float = 160.0,
    reference_date: Any = None,
) -> pd.Series:
    if dates is None or len(dates) <= 0:
        return pd.Series(dtype=float)
    d = pd.to_datetime(dates, errors="coerce")
    if d.notna().any():
        ref = pd.to_datetime(reference_date, errors="coerce")
        if pd.isna(ref):
            ref = d.max()
        age_days = (ref - d).dt.total_seconds() / (24.0 * 3600.0)
        age_days = age_days.clip(lower=0.0)
        w = age_days.map(
            lambda a: float(0.5 ** (float(a) / float(max(30.0, half_life_days))))
            if math.isfinite(float(a))
            else 1.0
        )
        return pd.to_numeric(w, errors="coerce").fillna(1.0).clip(lower=0.03, upper=1.0)
    return pd.Series([1.0] * len(dates), index=dates.index, dtype=float)


def _compute_h2h_quality_pairwise_signal(
    h2h_a_rows: pd.DataFrame,
    *,
    reference_date: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False,
        "reason": "",
        "n_maps": 0,
        "wins_a": 0,
        "wins_b": 0,
        "n_eff": 0.0,
        "p_laplace": 0.5,
        "p_quality": 0.5,
        "weight": 0.0,
        "edge_win": 0.0,
        "edge_quality": 0.0,
        "edge_surplus": 0.0,
        "edge_conversion": 0.0,
        "tempo_edge": 0.0,
        "coverage": 0.0,
        "freshness": 0.0,
        "data_reliability": 0.0,
        "age_days_mean": None,
    }
    if h2h_a_rows is None or not isinstance(h2h_a_rows, pd.DataFrame) or h2h_a_rows.empty:
        out["reason"] = "no_h2h_rows"
        return out

    h = h2h_a_rows.copy()
    h["win"] = pd.to_numeric(h.get("win"), errors="coerce").fillna(0.0)
    h = h[h["win"].isin([0.0, 1.0])].copy()
    if h.empty:
        out["reason"] = "invalid_h2h_rows"
        return out

    out["n_maps"] = len(h)
    out["wins_a"] = int((h["win"] >= 0.5).sum())
    out["wins_b"] = int(max(0, out["n_maps"] - out["wins_a"]))

    if "date" in h.columns:
        w_rec = _recency_weights_from_dates(
            h["date"],
            half_life_days=160.0,
            reference_date=reference_date,
        )
        d = pd.to_datetime(h["date"], errors="coerce")
        if d.notna().any():
            ref = pd.to_datetime(reference_date, errors="coerce")
            if pd.isna(ref):
                ref = d.max()
            age_days = (ref - d).dt.total_seconds() / (24.0 * 3600.0)
            age_days = age_days.clip(lower=0.0).fillna(0.0)
        else:
            age_days = pd.Series([0.0] * len(h), index=h.index, dtype=float)
    else:
        w_rec = pd.Series([1.0] * len(h), index=h.index, dtype=float)
        age_days = pd.Series([0.0] * len(h), index=h.index, dtype=float)

    sum_w = 0.0
    sum_age = 0.0
    sum_sign = 0.0
    sum_q = 0.0
    sum_conv = 0.0
    sum_tempo = 0.0
    sum_cov = 0.0
    sum_rel = 0.0

    for idx, row in h.iterrows():
        w_i = _f(w_rec.loc[idx], 1.0)
        if not math.isfinite(w_i) or w_i <= 0.0:
            w_i = 1.0
        sign = 1.0 if _f(row.get("win"), 0.0) >= 0.5 else -1.0

        kd = _pair_diff_value(row, "kills")
        td = _pair_diff_value(row, "towers")
        dd = _pair_diff_value(row, "dragons")
        ed = _pair_diff_value(row, "elementaldrakes")
        if (not math.isfinite(dd)) and math.isfinite(ed):
            dd = float(ed)
        bd = _pair_diff_value(row, "barons")
        hd = _pair_diff_value(row, "heralds")
        vg = _pair_diff_value(row, "voidgrubs")
        ak = _pair_diff_value(row, "atakhans")
        pl = _pair_diff_value(row, "turretplates")
        idf = _pair_diff_value(row, "inhibitors")

        obj_total_w = 0.0
        obj_acc = 0.0
        if math.isfinite(td):
            obj_acc += 0.27 * _norm_margin(td, 9.0)
            obj_total_w += 0.27
        if math.isfinite(dd):
            obj_acc += 0.18 * _norm_margin(dd, 4.0)
            obj_total_w += 0.18
        if math.isfinite(bd):
            obj_acc += 0.18 * _norm_margin(bd, 2.0)
            obj_total_w += 0.18
        if math.isfinite(hd):
            obj_acc += 0.07 * _norm_margin(hd, 2.0)
            obj_total_w += 0.07
        if math.isfinite(vg):
            obj_acc += 0.05 * _norm_margin(vg, 6.0)
            obj_total_w += 0.05
        if math.isfinite(ak):
            obj_acc += 0.03 * _norm_margin(ak, 1.0)
            obj_total_w += 0.03
        if math.isfinite(pl):
            obj_acc += 0.07 * _norm_margin(pl, 10.0)
            obj_total_w += 0.07
        if math.isfinite(idf):
            obj_acc += 0.15 * _norm_margin(idf, 3.0)
            obj_total_w += 0.15
        obj_score = (obj_acc / obj_total_w) if obj_total_w > 0.0 else 0.0
        kill_score = _norm_margin(kd, 16.0) if math.isfinite(kd) else 0.0
        pressure = _clip((0.62 * obj_score) + (0.30 * kill_score), -1.0, 1.0)

        # First-objective control as an early execution proxy.
        first_terms = [
            0.12 * _first_objective_diff(row, "firstblood"),
            0.16 * _first_objective_diff(row, "firstdragon"),
            0.22 * _first_objective_diff(row, "firstbaron"),
            0.08 * _first_objective_diff(row, "firstherald"),
            0.16 * _first_objective_diff(row, "firsttower"),
            0.10 * _first_objective_diff(row, "firstmidtower"),
            0.16 * _first_objective_diff(row, "firsttothreetowers"),
        ]
        first_score = _clip(sum(first_terms), -1.0, 1.0)
        pressure = _clip(pressure + (0.08 * first_score), -1.0, 1.0)

        gl = _f(row.get("gamelength"), float("nan"))
        tempo_raw = _norm_margin(2100.0 - gl, 1200.0) if math.isfinite(gl) and gl > 300.0 else 0.0
        tempo_signed = float(sign * tempo_raw)

        # Conversion quality from early lead (gold/xp/cs) when available.
        lead = float("nan")
        lead_scale = 1.0
        gd20 = _pair_diff_value(row, "golddiffat20")
        gd15 = _pair_diff_value(row, "golddiffat15")
        gd10 = _pair_diff_value(row, "golddiffat10")
        xp20 = _pair_diff_value(row, "xpdiffat20")
        xp15 = _pair_diff_value(row, "xpdiffat15")
        xp10 = _pair_diff_value(row, "xpdiffat10")
        cs20 = _pair_diff_value(row, "csdiffat20")
        cs15 = _pair_diff_value(row, "csdiffat15")
        cs10 = _pair_diff_value(row, "csdiffat10")
        if math.isfinite(gd20):
            lead, lead_scale = float(gd20), 6000.0
        elif math.isfinite(gd15):
            lead, lead_scale = float(gd15), 4500.0
        elif math.isfinite(gd10):
            lead, lead_scale = float(gd10), 3000.0
        elif math.isfinite(xp20):
            lead, lead_scale = float(xp20), 4500.0
        elif math.isfinite(xp15):
            lead, lead_scale = float(xp15), 3500.0
        elif math.isfinite(xp10):
            lead, lead_scale = float(xp10), 2500.0
        elif math.isfinite(cs20):
            lead, lead_scale = float(cs20), 75.0
        elif math.isfinite(cs15):
            lead, lead_scale = float(cs15), 55.0
        elif math.isfinite(cs10):
            lead, lead_scale = float(cs10), 35.0
        q_conv = 0.0
        if math.isfinite(lead):
            ln = _norm_margin(lead, lead_scale)
            la = abs(float(ln))
            if la > 0.0:
                if (ln * sign) > 0.0:
                    q_conv = la
                elif sign > 0.0:
                    q_conv = 0.65 * la
                else:
                    q_conv = -1.20 * la
        q_conv = _clip(q_conv, -1.0, 1.0)

        rel_for = _data_completeness_weight(row.get("datacompleteness_for"))
        rel_against = _data_completeness_weight(row.get("datacompleteness_against"))
        rel = float(max(0.70, min(1.0, min(rel_for, rel_against))))

        # Pairwise quality signal from A perspective:
        # - sign captures W/L
        # - win_align rewards decisive wins and softens "throw-like" losses
        win_align = float(sign * pressure)
        q_signed = _clip(sign + (0.24 * win_align) + (0.10 * tempo_signed) + (0.14 * q_conv), -1.0, 1.0)
        q_signed *= rel

        obj_cov = _clip(obj_total_w / 1.0, 0.0, 1.0)
        kill_cov = 1.0 if math.isfinite(kd) else 0.0
        tempo_cov = 1.0 if (math.isfinite(gl) and gl > 300.0) else 0.0
        lead_cov = 1.0 if math.isfinite(lead) else 0.0
        first_cov = 0.0
        first_seen = 0
        for fstem in (
            "firstblood",
            "firstdragon",
            "firstbaron",
            "firstherald",
            "firsttower",
            "firstmidtower",
            "firsttothreetowers",
        ):
            if math.isfinite(_f(row.get(f"{fstem}_for"), float("nan"))) or math.isfinite(_f(row.get(f"{fstem}_against"), float("nan"))):
                first_seen += 1
        if first_seen > 0:
            first_cov = float(first_seen / 7.0)
        cov_i = _clip(
            ((0.38 * obj_cov) + (0.22 * kill_cov) + (0.16 * tempo_cov) + (0.16 * lead_cov) + (0.08 * first_cov))
            * rel,
            0.0,
            1.0,
        )

        age_i = _f(age_days.loc[idx], 0.0)
        if not math.isfinite(age_i):
            age_i = 0.0

        sum_w += float(w_i)
        sum_age += float(w_i * age_i)
        sum_sign += float(w_i * sign)
        sum_q += float(w_i * q_signed)
        sum_conv += float(w_i * q_conv)
        sum_tempo += float(w_i * tempo_signed)
        sum_cov += float(w_i * cov_i)
        sum_rel += float(w_i * rel)

    if sum_w <= 0.0:
        out["reason"] = "zero_weight_h2h"
        return out

    n_maps = int(out["n_maps"])
    wins_a = int(out["wins_a"])
    p_lap = float((wins_a + 1.0) / (max(1.0, float(n_maps)) + 2.0))
    edge_win = float(sum_sign / sum_w)
    edge_quality = float(sum_q / sum_w)
    edge_surplus = float(edge_quality - edge_win)
    edge_conv = float(sum_conv / sum_w)
    tempo_edge = float(sum_tempo / sum_w)
    cov = float(_clip(sum_cov / sum_w, 0.0, 1.0))
    rel_avg = float(_clip(sum_rel / sum_w, 0.70, 1.0))
    age_mean = float(max(0.0, sum_age / sum_w))
    fresh = float(max(0.30, min(1.0, 1.0 / (1.0 + (age_mean / 160.0) ** 1.30))))
    n_eff = float(sum_w)
    n_conf = float(max(0.0, min(1.0, n_eff / (n_eff + 2.5))))

    z_lap = _logit(p_lap)
    z_q = float(max(-2.2, min(2.2, z_lap + (0.95 * edge_surplus) + (0.15 * tempo_edge))))
    p_q = _clip(_sigmoid(z_q), 1e-6, 1.0 - 1e-6)

    w_h2h = float(0.02 + 0.18 * math.sqrt(max(0.0, n_conf * cov * fresh)))
    if n_maps <= 1:
        w_h2h *= 0.65
    w_h2h = float(max(0.0, min(0.16, w_h2h)))

    out.update(
        {
            "available": bool(n_maps >= 2),
            "reason": "ok",
            "n_eff": float(n_eff),
            "p_laplace": float(p_lap),
            "p_quality": float(p_q),
            "weight": float(w_h2h),
            "edge_win": float(edge_win),
            "edge_quality": float(edge_quality),
            "edge_surplus": float(edge_surplus),
            "edge_conversion": float(edge_conv),
            "tempo_edge": float(tempo_edge),
            "coverage": float(cov),
            "freshness": float(fresh),
            "data_reliability": float(rel_avg),
            "age_days_mean": float(age_mean),
        }
    )
    return out


def _coerce_playoffs_mask(series: pd.Series, playoffs_opt: Any) -> pd.Series:
    z = str(playoffs_opt or "").strip().lower()
    if not z or z in {"all", "todos"}:
        return pd.Series([True] * len(series), index=series.index)
    s = series.fillna("").astype(str).str.strip().str.lower()
    if z in {"1", "true", "playoffs", "sim", "yes"}:
        return s.isin({"1", "true", "playoffs", "sim", "yes"})
    if z in {"0", "false", "regular", "nao", "no"}:
        return ~s.isin({"1", "true", "playoffs", "sim", "yes"})
    return pd.Series([True] * len(series), index=series.index)


def _resolve_team_name_from_pool(
    team_name: str,
    *,
    aliases: list[str] | None,
    team_pool: list[str],
) -> str:
    pool_map = {str(item).strip().casefold(): str(item) for item in list(team_pool or []) if str(item or "").strip()}
    candidates = [str(team_name or "").strip(), *(list(aliases or []))]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate or "").strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        if key in pool_map:
            return str(pool_map[key])
    return str(team_name or "").strip()


def apply_league_rank_sos_live(
    *,
    p_map_in: Any,
    team_games: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    team_aliases_a: list[str] | None = None,
    team_aliases_b: list[str] | None = None,
    league: str,
    year_opt: Any = None,
    split_opt: Any = None,
    playoffs_opt: Any = None,
    bo: Any = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "league_rank_sos_h2h_quality_v2",
        "p_before": None,
        "p_after": None,
        "diff_pp": None,
        "p_rank": None,
        "p_rank_base": None,
        "weight": None,
        "n_team_a": None,
        "n_team_b": None,
        "n_teams_league": None,
        "n_games_league": None,
        "same_league": False,
        "h2h_maps": 0,
        "h2h_a_wins": 0,
        "h2h_b_wins": 0,
        "h2h_n_eff": None,
        "h2h_p_laplace": None,
        "h2h_p_quality": None,
        "h2h_quality_weight": None,
        "h2h_quality_edge": None,
        "h2h_quality_surplus": None,
        "h2h_quality_conversion_edge": None,
        "h2h_quality_tempo_edge": None,
        "h2h_quality_coverage": None,
        "h2h_quality_freshness": None,
        "h2h_quality_data_reliability": None,
        "h2h_quality_age_days_mean": None,
        "team_a_found": False,
        "team_b_found": False,
    }
    p0 = _f(p_map_in)
    if not math.isfinite(p0):
        out["reason"] = "invalid_p_in"
        return out
    out["p_before"] = float(p0)
    lg = str(league or "").strip()
    ta = str(team_a or "").strip()
    tb = str(team_b or "").strip()
    if not lg or not ta or not tb:
        out["reason"] = "missing_identity"
        return out
    if team_games is None or not isinstance(team_games, pd.DataFrame) or team_games.empty:
        out["reason"] = "no_team_games"
        return out
    req = {"team", "opponent", "league", "win"}
    if not req.issubset(set(team_games.columns)):
        out["reason"] = "missing_cols"
        out["missing_cols"] = sorted(list(req - set(team_games.columns)))
        return out

    tg = team_games.copy()
    tg = tg[tg["league"].fillna("").astype(str).str.strip() == lg].copy()
    out["same_league"] = True
    if tg.empty:
        out["reason"] = "league_empty"
        return out
    if year_opt is not None and "year" in tg.columns and str(year_opt).strip():
        try:
            yy = int(float(year_opt))
            tg = tg[pd.to_numeric(tg["year"], errors="coerce") == yy].copy()
        except Exception:
            pass
    if split_opt is not None and "split" in tg.columns:
        sz = str(split_opt).strip()
        if sz and sz.lower() not in {"all", "todos"}:
            tg = tg[tg["split"].fillna("").astype(str).str.strip() == sz].copy()
    if playoffs_opt is not None and "playoffs" in tg.columns:
        tg = tg[_coerce_playoffs_mask(tg["playoffs"], playoffs_opt)].copy()
    if tg.empty:
        out["reason"] = "league_filtered_empty"
        return out

    # keep only valid rows
    tg["team"] = tg["team"].fillna("").astype(str).str.strip()
    tg["opponent"] = tg["opponent"].fillna("").astype(str).str.strip()
    tg = tg[(tg["team"] != "") & (tg["opponent"] != "")].copy()
    tg = tg[tg["team"] != tg["opponent"]].copy()
    tg["win"] = pd.to_numeric(tg["win"], errors="coerce")
    tg = tg[tg["win"].isin([0, 1])].copy()
    if tg.empty:
        out["reason"] = "invalid_rows"
        return out

    teams = sorted(set(tg["team"].tolist()) | set(tg["opponent"].tolist()))
    out["n_teams_league"] = len(teams)
    out["n_games_league"] = len(tg)
    if len(teams) < 4:
        out["reason"] = "too_few_teams"
        return out

    ta_lookup = _resolve_team_name_from_pool(
        ta,
        aliases=team_aliases_a,
        team_pool=teams,
    )
    tb_lookup = _resolve_team_name_from_pool(
        tb,
        aliases=team_aliases_b,
        team_pool=teams,
    )
    out["team_a_lookup"] = str(ta_lookup or "")
    out["team_b_lookup"] = str(tb_lookup or "")

    # Team raw record
    grp = tg.groupby("team", as_index=False).agg(
        n_games=("win", "count"),
        wins=("win", "sum"),
    )
    grp["losses"] = grp["n_games"] - grp["wins"]
    grp["raw_wp"] = (grp["wins"] + 1.5) / (grp["n_games"] + 3.0)
    raw_map = {str(r["team"]): float(r["raw_wp"]) for _, r in grp.iterrows()}
    n_map = {str(r["team"]): float(r["n_games"]) for _, r in grp.iterrows()}
    w_map = {str(r["team"]): float(r["wins"]) for _, r in grp.iterrows()}

    # Opponent strength of schedule (average opponent raw_wp weighted by meetings)
    opp_meets = tg.groupby(["team", "opponent"], as_index=False).agg(m=("win", "count"))
    opp_sos: Dict[str, float] = {}
    for team_name, g in opp_meets.groupby("team"):
        num = 0.0
        den = 0.0
        for _, rr in g.iterrows():
            opp = str(rr["opponent"])
            m = float(rr["m"] or 0.0)
            if m <= 0:
                continue
            num += m * float(raw_map.get(opp, 0.5))
            den += m
        opp_sos[str(team_name)] = (num / den) if den > 0 else 0.5

    # Optional dominance proxy from kill diff (if present)
    dom_map: Dict[str, float] = {}
    if {"kills", "opp_kills"}.issubset(set(tg.columns)):
        tk = pd.to_numeric(tg["kills"], errors="coerce")
        ok = pd.to_numeric(tg["opp_kills"], errors="coerce")
        den = (tk.abs() + ok.abs() + 1.0)
        tg["_dom_ratio"] = ((tk - ok) / den).fillna(0.0)
        dom_g = tg.groupby("team", as_index=False).agg(dom=("_dom_ratio", "mean"))
        dom_map = {str(r["team"]): float(r["dom"]) for _, r in dom_g.iterrows()}

    score_map: Dict[str, float] = {}
    debug_rows = []
    for tname in teams:
        n = float(n_map.get(tname, 0.0))
        raw = float(raw_map.get(tname, 0.5))
        sos = float(opp_sos.get(tname, 0.5))
        # SOS-adjusted win probability (centered around 0.5)
        p_adj = raw + 0.70 * (sos - 0.5)
        p_adj = float(max(0.03, min(0.97, p_adj)))
        dom = float(dom_map.get(tname, 0.0))
        # shrink the score under small sample
        shrink = float(n / (n + 8.0)) if n > 0 else 0.0
        score = shrink * (_logit(p_adj) + 0.90 * dom)
        score_map[tname] = float(score)
        debug_rows.append(
            {
                "team": tname,
                "n_games": int(n),
                "wins": int(w_map.get(tname, 0.0)),
                "raw_wp": raw,
                "opp_sos_wp": sos,
                "dom": dom,
                "score": score,
                "shrink": shrink,
            }
        )

    out["team_a_found"] = bool(ta_lookup in score_map)
    out["team_b_found"] = bool(tb_lookup in score_map)
    if (not out["team_a_found"]) or (not out["team_b_found"]):
        out["reason"] = "team_not_in_league_pool"
        return out

    nA = float(n_map.get(ta_lookup, 0.0))
    nB = float(n_map.get(tb_lookup, 0.0))
    out["n_team_a"] = int(nA)
    out["n_team_b"] = int(nB)
    if min(nA, nB) < 3.0:
        out["reason"] = "low_team_sample"
        return out

    # H2H summary in league sample
    h2h = tg[
        ((tg["team"] == ta_lookup) & (tg["opponent"] == tb_lookup))
        | ((tg["team"] == tb_lookup) & (tg["opponent"] == ta_lookup))
    ].copy()
    if not h2h.empty:
        h2h_a = h2h[h2h["team"] == ta_lookup]
        h2h_b = h2h[h2h["team"] == tb_lookup]
        out["h2h_maps"] = len(h2h_a)
        out["h2h_a_wins"] = int(pd.to_numeric(h2h_a["win"], errors="coerce").fillna(0).sum())
        out["h2h_b_wins"] = int(pd.to_numeric(h2h_b["win"], errors="coerce").fillna(0).sum())
        tg_ref_date = pd.to_datetime(tg["date"], errors="coerce").max() if "date" in tg.columns else None
        h2h_sig = _compute_h2h_quality_pairwise_signal(h2h_a, reference_date=tg_ref_date)
        if isinstance(h2h_sig, dict):
            out["h2h_n_eff"] = (
                float(h2h_sig.get("n_eff")) if math.isfinite(_f(h2h_sig.get("n_eff"), float("nan"))) else None
            )
            out["h2h_p_laplace"] = (
                float(h2h_sig.get("p_laplace"))
                if math.isfinite(_f(h2h_sig.get("p_laplace"), float("nan")))
                else None
            )
            out["h2h_p_quality"] = (
                float(h2h_sig.get("p_quality"))
                if math.isfinite(_f(h2h_sig.get("p_quality"), float("nan")))
                else None
            )
            out["h2h_quality_weight"] = (
                float(h2h_sig.get("weight"))
                if math.isfinite(_f(h2h_sig.get("weight"), float("nan")))
                else None
            )
            out["h2h_quality_edge"] = (
                float(h2h_sig.get("edge_quality"))
                if math.isfinite(_f(h2h_sig.get("edge_quality"), float("nan")))
                else None
            )
            out["h2h_quality_surplus"] = (
                float(h2h_sig.get("edge_surplus"))
                if math.isfinite(_f(h2h_sig.get("edge_surplus"), float("nan")))
                else None
            )
            out["h2h_quality_conversion_edge"] = (
                float(h2h_sig.get("edge_conversion"))
                if math.isfinite(_f(h2h_sig.get("edge_conversion"), float("nan")))
                else None
            )
            out["h2h_quality_tempo_edge"] = (
                float(h2h_sig.get("tempo_edge"))
                if math.isfinite(_f(h2h_sig.get("tempo_edge"), float("nan")))
                else None
            )
            out["h2h_quality_coverage"] = (
                float(h2h_sig.get("coverage"))
                if math.isfinite(_f(h2h_sig.get("coverage"), float("nan")))
                else None
            )
            out["h2h_quality_freshness"] = (
                float(h2h_sig.get("freshness"))
                if math.isfinite(_f(h2h_sig.get("freshness"), float("nan")))
                else None
            )
            out["h2h_quality_data_reliability"] = (
                float(h2h_sig.get("data_reliability"))
                if math.isfinite(_f(h2h_sig.get("data_reliability"), float("nan")))
                else None
            )
            out["h2h_quality_age_days_mean"] = (
                float(h2h_sig.get("age_days_mean"))
                if math.isfinite(_f(h2h_sig.get("age_days_mean"), float("nan")))
                else None
            )
            out["h2h_quality_signal"] = h2h_sig

    z_rank = float(score_map[ta_lookup] - score_map[tb_lookup])
    # modest scale; this is a prior, not a full model
    p_rank_base = _sigmoid(0.95 * z_rank)
    p_rank = float(p_rank_base)
    out["p_rank_base"] = float(p_rank_base)

    # Pairwise H2H quality refinement (conservative):
    # blend rank prior with pair-specific H2H quality posterior in logit space.
    h2h_w = _f(out.get("h2h_quality_weight"), 0.0)
    h2h_pq = _f(out.get("h2h_p_quality"), float("nan"))
    h2h_nm = int(max(0, _f(out.get("h2h_maps"), 0.0)))
    if h2h_w > 0.0 and math.isfinite(h2h_pq) and h2h_nm >= 2:
        z_rank_base = _logit(p_rank)
        z_h2h = _logit(_clip_prob(h2h_pq))
        disagree_h = abs(z_h2h - z_rank_base)
        if disagree_h > 1.2 and h2h_nm < 3:
            h2h_w *= 0.60
        elif disagree_h > 1.8:
            h2h_w *= 0.75
        h2h_w = float(max(0.0, min(0.16, h2h_w)))
        if h2h_w > 0.0:
            z_rank_blend = (1.0 - h2h_w) * z_rank_base + h2h_w * z_h2h
            p_rank = float(_sigmoid(z_rank_blend))
            out["h2h_quality_weight"] = float(h2h_w)
            out["h2h_quality_disagree"] = float(disagree_h)

    out["p_rank"] = float(p_rank)

    # Context weight / gate
    try:
        bo_i = int(float(bo)) if bo is not None else 1
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    n_min = min(nA, nB)
    w = 0.06 + 0.24 * math.sqrt(max(0.0, n_min / (n_min + 8.0)))
    if bo_i == 1:
        w *= 0.85
    elif bo_i >= 5:
        w *= 1.10
    # if current model already very extreme and rank prior opposes it, be conservative
    disagree = abs(_logit(p_rank) - _logit(p0))
    if disagree > 1.0 and n_min < 8.0:
        w *= 0.65
    w = float(max(0.0, min(0.28, w)))
    out["weight"] = float(w)
    if w <= 0.0:
        out["reason"] = "weight_zero"
        return out

    z0 = _logit(p0)
    z1 = (1.0 - w) * z0 + w * _logit(p_rank)
    p1 = _sigmoid(z1)
    out["applied"] = True
    out["reason"] = "applied"
    out["p_after"] = float(p1)
    out["diff_pp"] = float(100.0 * (p1 - p0))
    # compact audit payload
    try:
        dbg = pd.DataFrame(debug_rows).sort_values("score", ascending=False)
        out["league_rank_top"] = dbg.head(6).to_dict("records")
        out["team_a_rank_row"] = dbg.loc[dbg["team"] == ta_lookup].head(1).to_dict("records")
        out["team_b_rank_row"] = dbg.loc[dbg["team"] == tb_lookup].head(1).to_dict("records")
    except Exception:
        pass
    return out
