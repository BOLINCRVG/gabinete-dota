"""League-transfer builder — profile builders.

Team, player, lineup, and interleague-pair profile computation.
All inputs come from the loader layer (league_transfer_loader_core).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mlcore.identity_registry_core import normalize_identity_key
from mlcore.league_context import league_macro_key
from mlcore.tiers import (
    canonical_league,
    is_international_league,
    league_strength_band,
    league_strength_score,
    tier_for_league,
)
from mlcore.league_transfer_loader_core import (
    ROLE_CANON,
    _shrink_prob,
    _weighted_mean,
    _weighted_rate,
)
from mlcore.utils import clip_safe as _clip, sigmoid as _sigmoid


def _year_weight(year: Any) -> float:
    try:
        yy = int(float(year))
    except Exception:
        return 0.35
    if yy >= 2026:
        return 1.00
    if yy == 2025:
        return 0.70
    if yy == 2024:
        return 0.45
    return 0.35


def _playoffs_weight(value: Any) -> float:
    text = str(value or "").strip().lower()
    return 1.12 if text in {"1", "true", "playoffs", "sim", "yes"} else 1.0


def _build_match_pairs(team_rows: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in team_rows.columns if col in {"gameid", "participantid", "teamname", "team_entity", "team_current_name", "team_canonical_name", "league", "domestic_league", "year"}]
    meta = team_rows[cols].copy()
    pairs = team_rows.merge(meta, on="gameid", suffixes=("", "_opp"))
    pairs = pairs[pd.to_numeric(pairs["participantid"], errors="coerce") != pd.to_numeric(pairs["participantid_opp"], errors="coerce")].copy()
    pairs = pairs.drop_duplicates(subset=["gameid", "participantid"]).reset_index(drop=True)
    return pairs


def _prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["w_row"] = out.apply(lambda row: _year_weight(row.get("year")) * _playoffs_weight(row.get("playoffs")), axis=1)
    out["combined_kills"] = pd.to_numeric(out.get("kills"), errors="coerce").fillna(0.0) + pd.to_numeric(out.get("deaths"), errors="coerce").fillna(0.0)
    out["tower_diff"] = pd.to_numeric(out.get("towers"), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("opp_towers"), errors="coerce").fillna(0.0)
    out["dragon_diff"] = pd.to_numeric(out.get("dragons"), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("opp_dragons"), errors="coerce").fillna(0.0)
    out["baron_diff"] = pd.to_numeric(out.get("barons"), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("opp_barons"), errors="coerce").fillna(0.0)
    out["inhib_diff"] = pd.to_numeric(out.get("inhibitors"), errors="coerce").fillna(0.0) - pd.to_numeric(out.get("opp_inhibitors"), errors="coerce").fillna(0.0)
    out["stomp_win"] = (
        (pd.to_numeric(out.get("result"), errors="coerce").fillna(0.0) > 0.5)
        & (
            (pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0) >= 1800.0)
            | (out["tower_diff"] >= 4.0)
            | ((pd.to_numeric(out.get("gamelength"), errors="coerce").fillna(0.0) <= 1800.0) & (out["tower_diff"] >= 3.0))
        )
    )
    out["contested_game"] = pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0).abs() <= 1000.0
    out["nonlead_game"] = pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0) <= 0.0
    out["no_firsts_game"] = (
        (pd.to_numeric(out.get("firstblood"), errors="coerce").fillna(0.0) <= 0.0)
        & (pd.to_numeric(out.get("firsttower"), errors="coerce").fillna(0.0) <= 0.0)
    )
    return out


def _team_profile(group: pd.DataFrame) -> dict[str, Any]:
    w = pd.to_numeric(group["w_row"], errors="coerce").fillna(0.0)
    games = len(group)
    win_rate = _weighted_mean(group["result"], w, 0.5)
    lead_score = _sigmoid(_weighted_mean(group["golddiffat15"], w, 0.0) / 900.0)
    obj_mix = (
        (_weighted_mean(group["tower_diff"], w, 0.0) / 1.8)
        + (0.55 * _weighted_mean(group["dragon_diff"], w, 0.0))
        + (0.85 * _weighted_mean(group["baron_diff"], w, 0.0))
    )
    objective_score = _sigmoid(obj_mix)
    lead_mask = pd.to_numeric(group["golddiffat15"], errors="coerce").fillna(0.0) > 0.0
    closeout = _shrink_prob(
        _weighted_mean(group.loc[lead_mask, "result"], w.loc[lead_mask], 0.73),
        float(w.loc[lead_mask].sum()),
        prior=0.73,
        k=6.0,
    )
    behind_mask = group["nonlead_game"].fillna(False)
    no_firsts_mask = group["no_firsts_game"].fillna(False)
    contested_mask = group["contested_game"].fillna(False)
    wr_behind = _shrink_prob(
        _weighted_mean(group.loc[behind_mask, "result"], w.loc[behind_mask], 0.30),
        float(w.loc[behind_mask].sum()),
        prior=0.30,
        k=6.0,
    )
    wr_no_firsts = _shrink_prob(
        _weighted_mean(group.loc[no_firsts_mask, "result"], w.loc[no_firsts_mask], 0.42),
        float(w.loc[no_firsts_mask].sum()),
        prior=0.42,
        k=6.0,
    )
    wr_contested = _shrink_prob(
        _weighted_mean(group.loc[contested_mask, "result"], w.loc[contested_mask], 0.50),
        float(w.loc[contested_mask].sum()),
        prior=0.50,
        k=6.0,
    )
    out_of_script = _clip((0.45 * wr_behind) + (0.20 * wr_no_firsts) + (0.35 * wr_contested), 0.0, 1.0)
    stomp_share = _weighted_rate(group["stomp_win"], w, default=0.0)
    nonstomp_mask = ~group["stomp_win"].fillna(False)
    nonstomp_wr = _shrink_prob(
        _weighted_mean(group.loc[nonstomp_mask, "result"], w.loc[nonstomp_mask], 0.50),
        float(w.loc[nonstomp_mask].sum()),
        prior=0.50,
        k=6.0,
    )
    stomp_dependency = _clip((0.65 * stomp_share) + (0.35 * (1.0 - nonstomp_wr)), 0.0, 1.0)
    raw_domestic_power = _clip(
        (0.45 * win_rate)
        + (0.18 * closeout)
        + (0.15 * lead_score)
        + (0.12 * objective_score)
        + (0.10 * out_of_script),
        0.0,
        1.0,
    )
    raw_domestic_power = _shrink_prob(raw_domestic_power, float(games), prior=0.50, k=6.0)
    latest_year = int(pd.to_numeric(group.get("year"), errors="coerce").dropna().max()) if not group.empty else 0
    latest_rows = group[pd.to_numeric(group.get("year"), errors="coerce").fillna(-1).astype(int) == latest_year].copy() if latest_year > 0 else group
    league_counts = latest_rows.get("domestic_league", pd.Series(dtype=str)).astype(str).value_counts()
    latest_league = str(league_counts.index[0]) if not league_counts.empty else ""
    return {
        "sample_games": games,
        "win_rate": float(win_rate),
        "team_closeout_efficiency": float(closeout),
        "team_out_of_script_strength": float(out_of_script),
        "team_stomp_dependency": float(stomp_dependency),
        "raw_domestic_power": float(raw_domestic_power),
        "latest_year": int(latest_year),
        "domestic_league": str(latest_league or ""),
        "team_current_name": str(latest_rows["team_current_name"].iloc[-1] if "team_current_name" in latest_rows.columns and not latest_rows.empty else group["teamname"].iloc[-1]),
        "team_canonical_name": str(latest_rows["team_canonical_name"].iloc[-1] if "team_canonical_name" in latest_rows.columns and not latest_rows.empty else group["teamname"].iloc[-1]),
    }


def _patch_stability(df: pd.DataFrame) -> float:
    if df.empty or "patch" not in df.columns:
        return 0.50
    use = df.copy()
    use["patch"] = use["patch"].astype(str).str.strip()
    use = use[use["patch"] != ""]
    if use.empty:
        return 0.50
    overall = {
        "kills": _weighted_mean(use["combined_kills"], use["w_row"], 26.0),
        "time": _weighted_mean(use["gamelength"], use["w_row"], 1900.0),
        "lead15": _weighted_mean(use["golddiffat15"].abs(), use["w_row"], 1200.0),
        "towers": _weighted_mean((use["towers"] + use["opp_towers"]), use["w_row"], 11.0),
    }
    total_w = float(use["w_row"].sum())
    dev = 0.0
    for _patch, grp in use.groupby("patch"):
        w_patch = float(grp["w_row"].sum())
        if w_patch <= 0.0:
            continue
        patch_dev = 0.0
        patch_dev += abs(_weighted_mean(grp["combined_kills"], grp["w_row"], overall["kills"]) - overall["kills"]) / 12.0
        patch_dev += abs(_weighted_mean(grp["gamelength"], grp["w_row"], overall["time"]) - overall["time"]) / 900.0
        patch_dev += abs(_weighted_mean(grp["golddiffat15"].abs(), grp["w_row"], overall["lead15"]) - overall["lead15"]) / 1800.0
        patch_dev += abs(_weighted_mean((grp["towers"] + grp["opp_towers"]), grp["w_row"], overall["towers"]) - overall["towers"]) / 6.0
        dev += (w_patch / max(1e-9, total_w)) * patch_dev
    return float(max(0.0, min(1.0, math.exp(-dev))))


def _build_team_and_league_profiles(team_rows: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    pairs = _build_match_pairs(team_rows)
    domestic = pairs[
        (pairs["domestic_league"].astype(str).str.strip() != "")
        & (pairs["domestic_league"].astype(str) == pairs["domestic_league_opp"].astype(str))
        & (pairs["league"].astype(str) == pairs["domestic_league"].astype(str))
        & (~pairs["league"].map(is_international_league))
    ].copy()
    domestic = _prepare_metrics(domestic)
    if domestic.empty:
        return {}, {}, domestic

    team_profiles_raw: dict[str, Any] = {}
    for entity_key, grp in domestic.groupby("team_entity", dropna=False):
        team_profiles_raw[str(entity_key)] = _team_profile(grp)

    raw_power_map = {key: float((row or {}).get("raw_domestic_power") or 0.50) for key, row in team_profiles_raw.items()}
    sos_map: dict[str, float] = {}
    for entity_key, grp in domestic.groupby("team_entity", dropna=False):
        opp_power = grp["team_entity_opp"].map(lambda value: raw_power_map.get(str(value), 0.50))
        sos_map[str(entity_key)] = float(_weighted_mean(opp_power, grp["w_row"], 0.50))

    league_profiles: dict[str, Any] = {}
    for league, grp in domestic.groupby("domestic_league", dropna=False):
        lg = str(canonical_league(str(league or "")) or str(league or "")).strip()
        team_keys = grp["team_entity"].astype(str).unique().tolist()
        team_scores = sorted((raw_power_map.get(key, 0.50) for key in team_keys), reverse=True)
        top1 = float(team_scores[0]) if team_scores else 0.50
        next_slice = team_scores[1:6] if len(team_scores) > 1 else team_scores[:1]
        next_mean = float(sum(next_slice) / len(next_slice)) if next_slice else 0.50
        gap_top1_top6 = float(max(0.0, top1 - next_mean))
        league_depth = float(max(0.0, min(1.0, next_mean - (0.5 * gap_top1_top6))))
        lead_mask = pd.to_numeric(grp["golddiffat15"], errors="coerce").fillna(0.0) > 0.0
        trail_mask = pd.to_numeric(grp["golddiffat15"], errors="coerce").fillna(0.0) <= 0.0
        tower_mask = pd.to_numeric(grp["firsttower"], errors="coerce").fillna(0.0) > 0.0
        lead_conv = _shrink_prob(_weighted_mean(grp.loc[lead_mask, "result"], grp.loc[lead_mask, "w_row"], 0.73), float(grp.loc[lead_mask, "w_row"].sum()), prior=0.73, k=8.0)
        trail_recovery = _shrink_prob(_weighted_mean(grp.loc[trail_mask, "result"], grp.loc[trail_mask, "w_row"], 0.28), float(grp.loc[trail_mask, "w_row"].sum()), prior=0.28, k=8.0)
        first_obj_conv = _shrink_prob(_weighted_mean(grp.loc[tower_mask, "result"], grp.loc[tower_mask, "w_row"], 0.66), float(grp.loc[tower_mask, "w_row"].sum()), prior=0.66, k=8.0)
        predictability = _clip((0.50 * lead_conv) + (0.25 * (1.0 - trail_recovery)) + (0.25 * first_obj_conv), 0.0, 1.0)
        patch_stability = _patch_stability(grp)
        sample_reliability = float(max(0.0, min(1.0, float(len(grp)) / 120.0)))
        completeness_cols = [col for col in ("result", "gamelength", "kills", "deaths", "towers", "golddiffat15") if col in grp.columns]
        data_completeness = float(np.mean([grp[col].notna().mean() for col in completeness_cols])) if completeness_cols else 1.0
        game_level = grp.groupby("gameid", dropna=False).agg(abs_lead15=("golddiffat15", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())), w_row=("w_row", "max"))
        stomp_rate = _weighted_rate(game_level["abs_lead15"] >= 1800.0, game_level["w_row"], default=0.0) if not game_level.empty else 0.0
        league_profiles[lg] = {
            "league_strength": float(_clip(league_strength_score(lg), 0.0, 1.0)),
            "league_strength_band": str(league_strength_band(lg) or "local"),
            "league_depth": float(league_depth),
            "league_patch_stability": float(patch_stability),
            "league_reliability": float(_clip((0.45 * predictability) + (0.25 * patch_stability) + (0.20 * sample_reliability) + (0.10 * data_completeness), 0.0, 1.0)),
            "league_translation_score": 0.50,
            "league_domestic_inflation": 0.50,
            "sample_games": int(game_level.shape[0]),
            "sample_team_rows": len(grp),
            "sample_teams": len(team_keys),
            "patch_count": int(grp["patch"].astype(str).nunique()) if "patch" in grp.columns else 0,
            "gap_top1_top6": float(gap_top1_top6),
            "stomp_rate": float(stomp_rate),
            "predictability": float(predictability),
            "data_completeness": float(data_completeness),
        }

    cross = pairs[
        (pairs["domestic_league"].astype(str).str.strip() != "")
        & (pairs["domestic_league_opp"].astype(str).str.strip() != "")
        & (pairs["domestic_league"].astype(str) != pairs["domestic_league_opp"].astype(str))
    ].copy()
    cross = _prepare_metrics(cross)
    if not cross.empty:
        cross["expected"] = cross.apply(lambda row: _sigmoid(4.0 * (league_strength_score(row.get("domestic_league")) - league_strength_score(row.get("domestic_league_opp")))), axis=1)
        cross["residual"] = pd.to_numeric(cross["result"], errors="coerce").fillna(0.0) - pd.to_numeric(cross["expected"], errors="coerce").fillna(0.5)
        for league, grp in cross.groupby("domestic_league", dropna=False):
            lg = str(canonical_league(str(league or "")) or str(league or "")).strip()
            resid = _weighted_mean(grp["residual"], grp["w_row"], 0.0)
            shrink = float(max(0.0, min(1.0, float(len(grp)) / (float(len(grp)) + 18.0))))
            translation = float(max(0.20, min(0.80, 0.50 + (0.90 * resid * shrink))))
            if lg not in league_profiles:
                league_profiles[lg] = {
                    "league_strength": float(_clip(league_strength_score(lg), 0.0, 1.0)),
                    "league_strength_band": str(league_strength_band(lg) or "local"),
                    "league_depth": 0.50,
                    "league_patch_stability": 0.50,
                    "league_reliability": 0.50,
                    "sample_games": 0,
                    "sample_team_rows": 0,
                    "sample_teams": 0,
                    "patch_count": 0,
                    "gap_top1_top6": 0.0,
                    "stomp_rate": 0.0,
                    "predictability": 0.50,
                    "data_completeness": 1.0,
                }
            league_profiles[lg]["league_translation_score"] = float(translation)
            league_profiles[lg]["translation_sample_games"] = int(grp["gameid"].nunique())
            league_profiles[lg]["translation_residual"] = float(resid)

    for _league, row in league_profiles.items():
        gap = float(row.get("gap_top1_top6") or 0.0)
        stomp_rate = float(row.get("stomp_rate") or 0.0)
        depth = float(row.get("league_depth") or 0.50)
        translation = float(row.get("league_translation_score") or 0.50)
        inflation = _sigmoid((4.0 * gap) + (2.2 * stomp_rate) - (2.6 * depth) - (2.0 * (translation - 0.50)))
        row["league_domestic_inflation"] = float(_clip(inflation, 0.0, 1.0))

    team_profiles: dict[str, Any] = {}
    for entity_key, row in team_profiles_raw.items():
        league = str(row.get("domestic_league") or "")
        league_row = league_profiles.get(league) or {}
        sos = float(sos_map.get(entity_key, 0.50))
        domestic_power = _clip((0.88 * float(row.get("raw_domestic_power") or 0.50)) + (0.12 * sos), 0.0, 1.0)
        rel = float(league_row.get("league_reliability") or 0.50)
        depth = float(league_row.get("league_depth") or 0.50)
        translation = float(league_row.get("league_translation_score") or 0.50)
        inflation = float(league_row.get("league_domestic_inflation") or 0.50)
        out_script = float(row.get("team_out_of_script_strength") or 0.50)
        closeout = float(row.get("team_closeout_efficiency") or 0.50)
        stomp = float(row.get("team_stomp_dependency") or 0.50)
        transfer_raw = (
            domestic_power * (0.70 + (0.12 * rel) + (0.10 * translation) + (0.08 * depth))
            + (0.10 * out_script)
            + (0.06 * closeout)
            + (0.05 * sos)
            - (0.12 * inflation)
            - (0.09 * stomp)
        )
        transfer_power = _shrink_prob(_clip(transfer_raw, 0.0, 1.0), float(row.get("sample_games") or 0.0), prior=0.50, k=8.0)
        team_profiles[entity_key] = {
            **dict(row),
            "team_strength_of_schedule": float(_clip(sos, 0.0, 1.0)),
            "team_domestic_power": float(domestic_power),
            "team_transfer_power": float(_clip(transfer_power, 0.0, 1.0)),
            "league_reliability": float(rel),
            "league_translation_score": float(translation),
            "league_domestic_inflation": float(inflation),
        }
    return team_profiles, league_profiles, domestic


def _prepare_player_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    years = pd.to_numeric(out.get("year"), errors="coerce")
    year_weight = pd.Series(0.35, index=out.index, dtype=float)
    year_weight = year_weight.mask(years == 2024, 0.45)
    year_weight = year_weight.mask(years == 2025, 0.70)
    year_weight = year_weight.mask(years >= 2026, 1.00)
    playoffs_text = out.get("playoffs", "").astype(str).str.strip().str.lower()
    playoff_weight = np.where(playoffs_text.isin({"1", "true", "playoffs", "sim", "yes"}), 1.12, 1.0)
    out["w_row"] = year_weight.astype(float) * pd.Series(playoff_weight, index=out.index, dtype=float)
    out["contested_game"] = pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0).abs() <= 1000.0
    out["nonlead_game"] = pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0) <= 0.0
    out["no_firsts_game"] = (
        (pd.to_numeric(out.get("firstblood"), errors="coerce").fillna(0.0) <= 0.0)
        & (pd.to_numeric(out.get("firsttower"), errors="coerce").fillna(0.0) <= 0.0)
    )
    out["stomp_game"] = pd.to_numeric(out.get("golddiffat15"), errors="coerce").fillna(0.0) >= 1800.0
    out["nonstomp_game"] = ~out["stomp_game"].fillna(False)
    return out


def _append_player_perf_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    metrics = [
        ("dpm", 0.22),
        ("damageshare", 0.12),
        ("earnedgoldshare", 0.10),
        ("visionscore", 0.08),
        ("vspm", 0.08),
        ("cspm", 0.10),
        ("golddiffat15", 0.12),
        ("xpdiffat15", 0.10),
        ("csdiffat15", 0.08),
    ]
    score = pd.Series(0.0, index=out.index, dtype=float)
    for col, weight in metrics:
        base = pd.to_numeric(out.get(col), errors="coerce")
        mu = base.groupby([out["domestic_league"], out["pos_can"]]).transform("mean")
        sd = base.groupby([out["domestic_league"], out["pos_can"]]).transform("std").replace(0.0, np.nan)
        z = ((base - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)
        out[f"z_{col}"] = z.astype(float)
        score = score + (float(weight) * z.astype(float))
    out["player_perf_index"] = score.astype(float)
    return out


def _build_player_profiles(
    player_rows: pd.DataFrame,
    *,
    league_profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], pd.DataFrame]:
    domestic = player_rows[
        (player_rows["domestic_league"].astype(str).str.strip() != "")
        & (player_rows["league"].astype(str) == player_rows["domestic_league"].astype(str))
        & (~player_rows["league"].map(is_international_league))
        & (player_rows["pos_can"].astype(str).isin(ROLE_CANON))
    ].copy()
    domestic = _prepare_player_metrics(domestic)
    domestic = _append_player_perf_index(domestic)
    if domestic.empty:
        return {}, {}, domestic

    domestic["w"] = pd.to_numeric(domestic["w_row"], errors="coerce").fillna(0.0).astype(float)
    domestic["perf"] = pd.to_numeric(domestic["player_perf_index"], errors="coerce").fillna(0.0).astype(float)
    domestic["result_num"] = pd.to_numeric(domestic.get("result"), errors="coerce").fillna(0.0).astype(float)
    domestic["w_perf"] = domestic["w"] * domestic["perf"]
    domestic["w_perf_sq"] = domestic["w"] * domestic["perf"] * domestic["perf"]

    flag_cols = {
        "contested": "contested_game",
        "nonstomp": "nonstomp_game",
        "nonlead": "nonlead_game",
        "no_firsts": "no_firsts_game",
    }
    for prefix, col in flag_cols.items():
        mask = domestic[col].fillna(False).astype(bool)
        domestic[f"w_{prefix}"] = domestic["w"].where(mask, 0.0)
        domestic[f"w_perf_{prefix}"] = domestic["w_perf"].where(mask, 0.0)
        domestic[f"w_result_{prefix}"] = (domestic["w"] * domestic["result_num"]).where(mask, 0.0)

    grouped = domestic.groupby("player_entity", dropna=False)
    player_agg = grouped.agg(
        sample_games=("gameid", "nunique"),
        sample_leagues=("domestic_league", "nunique"),
        latest_year=("year", "max"),
        w_sum=("w", "sum"),
        w_perf_sum=("w_perf", "sum"),
        w_perf_sq_sum=("w_perf_sq", "sum"),
        w_contested=("w_contested", "sum"),
        w_perf_contested=("w_perf_contested", "sum"),
        w_nonstomp=("w_nonstomp", "sum"),
        w_perf_nonstomp=("w_perf_nonstomp", "sum"),
        w_nonlead=("w_nonlead", "sum"),
        w_result_nonlead=("w_result_nonlead", "sum"),
        w_no_firsts=("w_no_firsts", "sum"),
        w_result_no_firsts=("w_result_no_firsts", "sum"),
    ).reset_index()

    player_agg["perf_mean"] = np.where(
        player_agg["w_sum"] > 0.0,
        player_agg["w_perf_sum"] / player_agg["w_sum"],
        0.0,
    )
    perf_var = np.where(
        player_agg["w_sum"] > 0.0,
        (player_agg["w_perf_sq_sum"] / player_agg["w_sum"]) - (player_agg["perf_mean"] ** 2),
        1.0,
    )
    player_agg["perf_std"] = np.sqrt(np.maximum(perf_var, 0.0))

    dom_raw = 1.0 / (1.0 + np.exp(-0.72 * player_agg["perf_mean"].astype(float)))
    dom_k = 10.0
    player_agg["player_domestic_rating"] = (
        (player_agg["sample_games"].astype(float) / (player_agg["sample_games"].astype(float) + dom_k)) * dom_raw
        + (dom_k / (player_agg["sample_games"].astype(float) + dom_k)) * 0.50
    )

    def _shrunk_sigmoid(mean_col: str, weight_col: str, prior_col: str | float, *, k: float) -> pd.Series:
        mean_val = np.where(
            player_agg[weight_col] > 0.0,
            player_agg[mean_col] / player_agg[weight_col],
            0.0,
        )
        raw = 1.0 / (1.0 + np.exp(-0.72 * mean_val.astype(float)))
        prior_val = player_agg[prior_col].astype(float) if isinstance(prior_col, str) else float(prior_col)
        return (
            (player_agg[weight_col].astype(float) / (player_agg[weight_col].astype(float) + float(k))) * raw
            + (float(k) / (player_agg[weight_col].astype(float) + float(k))) * prior_val
        )

    def _shrunk_rate(sum_col: str, weight_col: str, *, prior: float, k: float) -> pd.Series:
        raw = np.where(
            player_agg[weight_col] > 0.0,
            player_agg[sum_col] / player_agg[weight_col],
            float(prior),
        )
        return (
            (player_agg[weight_col].astype(float) / (player_agg[weight_col].astype(float) + float(k))) * raw
            + (float(k) / (player_agg[weight_col].astype(float) + float(k))) * float(prior)
        )

    player_agg["contested_strength"] = _shrunk_sigmoid("w_perf_contested", "w_contested", "player_domestic_rating", k=6.0)
    player_agg["nonstomp_strength"] = _shrunk_sigmoid("w_perf_nonstomp", "w_nonstomp", "player_domestic_rating", k=6.0)
    player_agg["nonlead_win_rate"] = _shrunk_rate("w_result_nonlead", "w_nonlead", prior=0.35, k=6.0)
    player_agg["no_firsts_win_rate"] = _shrunk_rate("w_result_no_firsts", "w_no_firsts", prior=0.42, k=6.0)
    player_agg["player_consistency"] = np.clip(np.exp(-np.maximum(player_agg["perf_std"].astype(float), 0.0)), 0.0, 1.0)

    league_perf = (
        domestic.groupby(["player_entity", "domestic_league"], dropna=False)
        .agg(w_sum=("w", "sum"), w_perf_sum=("w_perf", "sum"))
        .reset_index()
    )
    league_perf["perf_mean"] = np.where(
        league_perf["w_sum"] > 0.0,
        league_perf["w_perf_sum"] / league_perf["w_sum"],
        0.0,
    )
    league_perf["perf_sq"] = league_perf["perf_mean"] * league_perf["perf_mean"]
    league_perf_agg = league_perf.groupby("player_entity", dropna=False).agg(
        lg_count=("domestic_league", "nunique"),
        lg_perf_mean=("perf_mean", "mean"),
        lg_perf_sq=("perf_sq", "mean"),
    ).reset_index()
    league_perf_agg["translation_std"] = np.sqrt(
        np.maximum(league_perf_agg["lg_perf_sq"].astype(float) - (league_perf_agg["lg_perf_mean"].astype(float) ** 2), 0.0)
    )
    league_perf_agg["player_translation_score"] = np.where(
        league_perf_agg["lg_count"].astype(int) <= 1,
        0.50,
        np.clip(np.exp(-league_perf_agg["translation_std"].astype(float)), 0.0, 1.0),
    )
    player_agg = player_agg.merge(
        league_perf_agg[["player_entity", "player_translation_score"]],
        on="player_entity",
        how="left",
    )
    player_agg["player_translation_score"] = player_agg["player_translation_score"].fillna(0.50).astype(float)

    role_counts = (
        domestic.groupby(["player_entity", "pos_can"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["player_entity", "n", "pos_can"], ascending=[True, False, True])
        .drop_duplicates(subset=["player_entity"], keep="first")
        .rename(columns={"pos_can": "role"})
    )
    player_agg = player_agg.merge(role_counts[["player_entity", "role"]], on="player_entity", how="left")

    latest_year_map = player_agg.set_index("player_entity")["latest_year"]
    latest_rows = domestic[
        pd.to_numeric(domestic["year"], errors="coerce").fillna(-1).astype(int)
        == domestic["player_entity"].map(latest_year_map).fillna(-1).astype(int)
    ].copy()
    latest_league = (
        latest_rows.groupby(["player_entity", "domestic_league"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["player_entity", "n", "domestic_league"], ascending=[True, False, True])
        .drop_duplicates(subset=["player_entity"], keep="first")
        .rename(columns={"domestic_league": "latest_domestic_league"})
    )
    latest_names = (
        latest_rows.sort_values(["player_entity", "date", "gameid", "player_current_name", "player_canonical_name"])
        .drop_duplicates(subset=["player_entity"], keep="last")
        [["player_entity", "player_current_name", "player_canonical_name"]]
    )
    player_agg = player_agg.merge(latest_league[["player_entity", "latest_domestic_league"]], on="player_entity", how="left")
    player_agg = player_agg.merge(latest_names, on="player_entity", how="left")
    player_agg["domestic_league"] = player_agg["latest_domestic_league"].fillna("").astype(str)

    league_rel_map = {str(k): float((v or {}).get("league_reliability") or 0.50) for k, v in (league_profiles or {}).items()}
    league_infl_map = {str(k): float((v or {}).get("league_domestic_inflation") or 0.50) for k, v in (league_profiles or {}).items()}
    player_agg["league_reliability"] = player_agg["domestic_league"].map(lambda v: league_rel_map.get(str(v), 0.50)).astype(float)
    player_agg["league_domestic_inflation"] = player_agg["domestic_league"].map(lambda v: league_infl_map.get(str(v), 0.50)).astype(float)

    player_agg["role_portability"] = np.clip(
        (0.50 * player_agg["player_consistency"].astype(float))
        + (0.28 * player_agg["contested_strength"].astype(float))
        + (0.22 * player_agg["player_translation_score"].astype(float)),
        0.0,
        1.0,
    )
    portability_raw = (
        player_agg["player_domestic_rating"].astype(float)
        * (
            0.58
            + (0.18 * player_agg["role_portability"].astype(float))
            + (0.12 * player_agg["nonstomp_strength"].astype(float))
            + (0.08 * player_agg["contested_strength"].astype(float))
        )
        + (0.07 * player_agg["nonlead_win_rate"].astype(float))
        + (0.05 * player_agg["no_firsts_win_rate"].astype(float))
        + (0.04 * player_agg["league_reliability"].astype(float))
        - (0.08 * player_agg["league_domestic_inflation"].astype(float))
    )
    portability_clip = np.clip(portability_raw.astype(float), 0.0, 1.0)
    port_k = 10.0
    sample_games = player_agg["sample_games"].astype(float)
    player_agg["player_portability"] = (
        (sample_games / (sample_games + port_k)) * portability_clip
        + (port_k / (sample_games + port_k)) * 0.50
    )

    by_player: dict[str, Any] = {}
    alias_index: dict[str, str] = {}
    player_map = domestic.attrs.get("identity_player_map") or {}
    for row in player_agg.itertuples(index=False):
        entity_key = str(getattr(row, "player_entity", "") or "")
        role = str(getattr(row, "role", "") or "")
        sample_leagues = int(getattr(row, "sample_leagues", 0) or 0)
        domestic_league = str(getattr(row, "domestic_league", "") or "")
        current_name = str(getattr(row, "player_current_name", "") or "")
        canonical_name = str(getattr(row, "player_canonical_name", "") or current_name)
        meta = player_map.get(str(current_name)) or player_map.get(str(canonical_name)) or {}
        aliases: list[str] = []
        seen: set[str] = set()
        for alias in [*list(meta.get("aliases") or []), current_name, canonical_name]:
            text = str(alias or "").strip()
            if text and text not in seen:
                seen.add(text)
                aliases.append(text)
        for alias in aliases:
            alias_index[normalize_identity_key(alias)] = str(entity_key)
        by_player[str(entity_key)] = {
            "entity_key": str(entity_key),
            "canonical_name": canonical_name,
            "current_name": current_name,
            "aliases": aliases,
            "role": role,
            "domestic_league": domestic_league,
            "sample_games": int(getattr(row, "sample_games", 0) or 0),
            "sample_leagues": int(sample_leagues),
            "player_domestic_rating": float(_clip(getattr(row, "player_domestic_rating", 0.5), 0.0, 1.0)),
            "player_portability": float(_clip(getattr(row, "player_portability", 0.5), 0.0, 1.0)),
            "role_portability": float(_clip(getattr(row, "role_portability", 0.5), 0.0, 1.0)),
            "player_consistency": float(_clip(getattr(row, "player_consistency", 0.5), 0.0, 1.0)),
            "player_translation_score": float(_clip(getattr(row, "player_translation_score", 0.5), 0.0, 1.0)),
            "contested_strength": float(_clip(getattr(row, "contested_strength", 0.5), 0.0, 1.0)),
            "nonstomp_strength": float(_clip(getattr(row, "nonstomp_strength", 0.5), 0.0, 1.0)),
            "nonlead_win_rate": float(_clip(getattr(row, "nonlead_win_rate", 0.35), 0.0, 1.0)),
            "no_firsts_win_rate": float(_clip(getattr(row, "no_firsts_win_rate", 0.42), 0.0, 1.0)),
            "league_reliability": float(_clip(getattr(row, "league_reliability", 0.5), 0.0, 1.0)),
            "league_domestic_inflation": float(_clip(getattr(row, "league_domestic_inflation", 0.5), 0.0, 1.0)),
        }
    return by_player, alias_index, domestic


def _build_lineup_profiles(
    player_rows: pd.DataFrame,
    *,
    by_player: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    domestic = player_rows[
        (player_rows["domestic_league"].astype(str).str.strip() != "")
        & (player_rows["league"].astype(str) == player_rows["domestic_league"].astype(str))
        & (~player_rows["league"].map(is_international_league))
        & (player_rows["pos_can"].astype(str).isin(ROLE_CANON))
    ].copy()
    if domestic.empty:
        return {}
    role_weights = {"Top": 0.18, "Jungle": 0.22, "Mid": 0.24, "Bot": 0.22, "Support": 0.14}
    years = pd.to_numeric(domestic["year"], errors="coerce").fillna(-1).astype(int)
    latest_year = domestic.groupby("team_entity", dropna=False)["year"].transform("max").fillna(-1).astype(int)
    latest = domestic[years == latest_year].copy()
    if latest.empty:
        return {}

    latest_league = (
        latest.groupby(["team_entity", "domestic_league"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["team_entity", "n", "domestic_league"], ascending=[True, False, True])
        .drop_duplicates(subset=["team_entity"], keep="first")
        .rename(columns={"domestic_league": "team_domestic_league"})
    )
    latest = latest.merge(latest_league[["team_entity", "team_domestic_league"]], on="team_entity", how="left")
    latest = latest[latest["domestic_league"].astype(str) == latest["team_domestic_league"].astype(str)].copy()
    if latest.empty:
        return {}

    team_games = (
        latest.groupby("team_entity", dropna=False)["gameid"]
        .nunique()
        .rename("team_games")
        .reset_index()
    )

    chosen = (
        latest.groupby(["team_entity", "pos_can", "player_entity"], dropna=False)["gameid"]
        .nunique()
        .rename("n_games")
        .reset_index()
        .sort_values(["team_entity", "pos_can", "n_games", "player_entity"], ascending=[True, True, False, True])
        .drop_duplicates(subset=["team_entity", "pos_can"], keep="first")
    )
    if chosen.empty:
        return {}

    chosen = chosen.merge(team_games, on="team_entity", how="left")
    chosen["role_weight"] = chosen["pos_can"].map(lambda v: float(role_weights.get(str(v), 0.20)))
    chosen["role_stability"] = np.where(
        chosen["team_games"].astype(float) > 0.0,
        chosen["n_games"].astype(float) / chosen["team_games"].astype(float),
        0.0,
    )

    player_feature_rows = []
    for entity_key, row in (by_player or {}).items():
        player_feature_rows.append(
            {
                "player_entity": str(entity_key),
                "player_domestic_rating": float((row or {}).get("player_domestic_rating") or 0.50),
                "player_portability": float((row or {}).get("player_portability") or 0.50),
            }
        )
    player_feature_df = pd.DataFrame(
        player_feature_rows,
        columns=["player_entity", "player_domestic_rating", "player_portability"],
    )
    chosen = chosen.merge(player_feature_df, on="player_entity", how="left")
    chosen["player_domestic_rating"] = chosen["player_domestic_rating"].fillna(0.50).astype(float)
    chosen["player_portability"] = chosen["player_portability"].fillna(0.50).astype(float)

    chosen_names = (
        latest.sort_values(["team_entity", "pos_can", "date", "gameid", "player_current_name"])
        .drop_duplicates(subset=["team_entity", "pos_can", "player_entity"], keep="last")
        [["team_entity", "pos_can", "player_entity", "player_current_name"]]
    )
    chosen = chosen.merge(chosen_names, on=["team_entity", "pos_can", "player_entity"], how="left")

    game_lineups = (
        latest.groupby(["team_entity", "gameid", "pos_can"], dropna=False)["player_entity"]
        .first()
        .unstack("pos_can")
        .reindex(columns=list(ROLE_CANON), fill_value="")
    )
    game_lineups = game_lineups.fillna("").astype(str)
    game_lineups["lineup_signature"] = game_lineups[list(ROLE_CANON)].agg("|".join, axis=1)
    game_lineups = game_lineups.reset_index()

    chosen_lineup = (
        chosen.pivot(index="team_entity", columns="pos_can", values="player_entity")
        .reindex(columns=list(ROLE_CANON), fill_value="")
        .fillna("")
        .astype(str)
    )
    chosen_lineup["chosen_signature"] = chosen_lineup[list(ROLE_CANON)].agg("|".join, axis=1)
    chosen_lineup = chosen_lineup.reset_index()

    game_lineups = game_lineups.merge(chosen_lineup[["team_entity", "chosen_signature"]], on="team_entity", how="left")
    game_lineups["lineup_match"] = game_lineups["lineup_signature"].astype(str) == game_lineups["chosen_signature"].astype(str)
    synergy = (
        game_lineups.groupby("team_entity", dropna=False)["lineup_match"]
        .mean()
        .rename("lineup_synergy_raw")
        .reset_index()
        .merge(team_games, on="team_entity", how="left")
    )
    synergy["lineup_synergy"] = (
        (synergy["team_games"].astype(float) / (synergy["team_games"].astype(float) + 6.0)) * synergy["lineup_synergy_raw"].astype(float)
        + (6.0 / (synergy["team_games"].astype(float) + 6.0)) * 0.50
    )

    latest_years = (
        latest.groupby("team_entity", dropna=False)["year"]
        .max()
        .rename("latest_year")
        .reset_index()
    )
    lineup_base = (
        chosen.groupby("team_entity", dropna=False)
        .agg(
            team_games=("team_games", "max"),
            coverage=("pos_can", "nunique"),
            weight_sum=("role_weight", "sum"),
            dom_score=("player_domestic_rating", lambda s: 0.0),
        )
        .drop(columns=["dom_score"])
        .reset_index()
    )
    weighted_scores = chosen.copy()
    weighted_scores["dom_weighted"] = weighted_scores["role_weight"] * weighted_scores["player_domestic_rating"]
    weighted_scores["transfer_weighted"] = weighted_scores["role_weight"] * weighted_scores["player_portability"]
    weighted_scores["stability_weighted"] = weighted_scores["role_stability"]
    score_agg = (
        weighted_scores.groupby("team_entity", dropna=False)
        .agg(
            dom_weighted=("dom_weighted", "sum"),
            transfer_weighted=("transfer_weighted", "sum"),
            stability_mean=("role_stability", "mean"),
        )
        .reset_index()
    )
    lineup_base = lineup_base.merge(score_agg, on="team_entity", how="left")
    lineup_base = lineup_base.merge(synergy[["team_entity", "lineup_synergy"]], on="team_entity", how="left")
    lineup_base = lineup_base.merge(latest_league[["team_entity", "team_domestic_league"]], on="team_entity", how="left")
    lineup_base = lineup_base.merge(latest_years, on="team_entity", how="left")
    lineup_base["coverage"] = lineup_base["coverage"].astype(float) / float(len(ROLE_CANON))
    lineup_base["lineup_strength"] = np.where(
        lineup_base["weight_sum"].astype(float) > 0.0,
        lineup_base["dom_weighted"].astype(float) / lineup_base["weight_sum"].astype(float),
        0.50,
    )
    lineup_base["transfer_strength"] = np.where(
        lineup_base["weight_sum"].astype(float) > 0.0,
        lineup_base["transfer_weighted"].astype(float) / lineup_base["weight_sum"].astype(float),
        0.50,
    )
    lineup_base["lineup_stability"] = np.clip(lineup_base["stability_mean"].fillna(0.0).astype(float), 0.0, 1.0)
    lineup_base["lineup_synergy"] = np.clip(lineup_base["lineup_synergy"].fillna(0.50).astype(float), 0.0, 1.0)
    lineup_base["lineup_transfer_power"] = np.clip(
        lineup_base["transfer_strength"].astype(float)
        * (
            0.68
            + (0.18 * lineup_base["lineup_stability"].astype(float))
            + (0.14 * lineup_base["lineup_synergy"].astype(float))
        ),
        0.0,
        1.0,
    )

    player_entities_by_pos = (
        chosen.pivot(index="team_entity", columns="pos_can", values="player_entity")
        .reindex(columns=list(ROLE_CANON), fill_value="")
        .fillna("")
        .astype(str)
    )
    player_names_by_pos = (
        chosen.pivot(index="team_entity", columns="pos_can", values="player_current_name")
        .reindex(columns=list(ROLE_CANON), fill_value="")
        .fillna("")
        .astype(str)
    )

    out: dict[str, Any] = {}
    for row in lineup_base.itertuples(index=False):
        team_entity = str(getattr(row, "team_entity", "") or "")
        entities = {pos: str(player_entities_by_pos.at[team_entity, pos]) for pos in ROLE_CANON if team_entity in player_entities_by_pos.index and str(player_entities_by_pos.at[team_entity, pos])}
        names = {pos: str(player_names_by_pos.at[team_entity, pos]) for pos in ROLE_CANON if team_entity in player_names_by_pos.index and str(player_names_by_pos.at[team_entity, pos])}
        out[team_entity] = {
            "entity_key": team_entity,
            "domestic_league": str(getattr(row, "team_domestic_league", "") or ""),
            "latest_year": int(getattr(row, "latest_year", 0) or 0),
            "sample_games": int(getattr(row, "team_games", 0) or 0),
            "coverage": float(_clip(getattr(row, "coverage", 0.0), 0.0, 1.0)),
            "lineup_strength": float(_clip(getattr(row, "lineup_strength", 0.5), 0.0, 1.0)),
            "lineup_stability": float(_clip(getattr(row, "lineup_stability", 0.0), 0.0, 1.0)),
            "lineup_synergy": float(_clip(getattr(row, "lineup_synergy", 0.5), 0.0, 1.0)),
            "lineup_transfer_power": float(_clip(getattr(row, "lineup_transfer_power", 0.5), 0.0, 1.0)),
            "player_entities_by_pos": entities,
            "player_names_by_pos": names,
        }
    return out


def _league_strength_class(league: Any, league_profiles: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    lg = str(canonical_league(str(league or "")) or str(league or "")).strip()
    row = dict((league_profiles or {}).get(lg) or {})
    band = str(row.get("league_strength_band") or league_strength_band(lg) or "local").strip().lower()
    rel = float(row.get("league_reliability") or 0.50)
    if band in {"elite", "major", "upper"} and rel >= 0.48:
        return "major"
    return "minor"


def _build_interleague_pair_profiles(
    team_rows: pd.DataFrame,
    *,
    team_profiles: Mapping[str, Mapping[str, Any]],
    league_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = _build_match_pairs(team_rows)
    cross = pairs[
        (pairs["domestic_league"].astype(str).str.strip() != "")
        & (pairs["domestic_league_opp"].astype(str).str.strip() != "")
        & (pairs["domestic_league"].astype(str) != pairs["domestic_league_opp"].astype(str))
    ].copy()
    if cross.empty:
        return {}
    cross = _prepare_metrics(cross)
    cross["league_a"] = cross["domestic_league"].astype(str)
    cross["league_b"] = cross["domestic_league_opp"].astype(str)
    cross["tier_a"] = cross["league_a"].map(lambda v: str(tier_for_league(v) or "OTHER"))
    cross["tier_b"] = cross["league_b"].map(lambda v: str(tier_for_league(v) or "OTHER"))
    cross["macro_a"] = cross["league_a"].map(lambda v: league_macro_key(v, unknown="OTHER"))
    cross["macro_b"] = cross["league_b"].map(lambda v: league_macro_key(v, unknown="OTHER"))
    cross["strength_band_a"] = cross["league_a"].map(lambda v: str(league_strength_band(v) or "local"))
    cross["strength_band_b"] = cross["league_b"].map(lambda v: str(league_strength_band(v) or "local"))
    cross["class_a"] = cross["league_a"].map(lambda v: _league_strength_class(v, league_profiles))
    cross["class_b"] = cross["league_b"].map(lambda v: _league_strength_class(v, league_profiles))
    cross["class_relation"] = cross["class_a"].astype(str) + "->" + cross["class_b"].astype(str)

    team_tp_map = {str(k): float((row or {}).get("team_transfer_power") or 0.50) for k, row in (team_profiles or {}).items()}
    team_dp_map = {str(k): float((row or {}).get("team_domestic_power") or 0.50) for k, row in (team_profiles or {}).items()}
    lg_rel_map = {str(k): float((row or {}).get("league_reliability") or 0.50) for k, row in (league_profiles or {}).items()}
    lg_depth_map = {str(k): float((row or {}).get("league_depth") or 0.50) for k, row in (league_profiles or {}).items()}
    lg_trans_map = {str(k): float((row or {}).get("league_translation_score") or 0.50) for k, row in (league_profiles or {}).items()}
    lg_infl_map = {str(k): float((row or {}).get("league_domestic_inflation") or 0.50) for k, row in (league_profiles or {}).items()}

    cross["team_transfer_edge"] = cross["team_entity"].map(lambda v: team_tp_map.get(str(v), 0.50)) - cross["team_entity_opp"].map(lambda v: team_tp_map.get(str(v), 0.50))
    cross["team_domestic_edge"] = cross["team_entity"].map(lambda v: team_dp_map.get(str(v), 0.50)) - cross["team_entity_opp"].map(lambda v: team_dp_map.get(str(v), 0.50))
    cross["league_rel_edge"] = cross["league_a"].map(lambda v: lg_rel_map.get(str(v), 0.50)) - cross["league_b"].map(lambda v: lg_rel_map.get(str(v), 0.50))
    cross["league_depth_edge"] = cross["league_a"].map(lambda v: lg_depth_map.get(str(v), 0.50)) - cross["league_b"].map(lambda v: lg_depth_map.get(str(v), 0.50))
    cross["league_trans_edge"] = cross["league_a"].map(lambda v: lg_trans_map.get(str(v), 0.50)) - cross["league_b"].map(lambda v: lg_trans_map.get(str(v), 0.50))
    cross["league_infl_edge"] = cross["league_b"].map(lambda v: lg_infl_map.get(str(v), 0.50)) - cross["league_a"].map(lambda v: lg_infl_map.get(str(v), 0.50))
    cross["result_edge"] = (pd.to_numeric(cross["result"], errors="coerce").fillna(0.0) - 0.5) * 2.0
    cross["strength_gap_abs"] = (cross["league_a"].map(lambda v: league_strength_score(v)) - cross["league_b"].map(lambda v: league_strength_score(v))).abs()
    cross["league_edge"] = (
        (0.45 * cross["league_trans_edge"].astype(float))
        + (0.25 * cross["league_rel_edge"].astype(float))
        + (0.20 * cross["league_depth_edge"].astype(float))
        + (0.10 * cross["league_infl_edge"].astype(float))
    )

    cross["league_pair_key"] = "league_pair:" + cross["league_a"].astype(str) + "->" + cross["league_b"].astype(str)
    cross["tier_pair_key"] = "tier_pair:" + cross["tier_a"].astype(str) + "->" + cross["tier_b"].astype(str)
    cross["strength_pair_key"] = "strength_pair:" + cross["strength_band_a"].astype(str) + "->" + cross["strength_band_b"].astype(str)
    cross["macro_pair_key"] = "macro_pair:" + cross["macro_a"].astype(str) + "->" + cross["macro_b"].astype(str)
    cross["class_pair_key"] = "class_pair:" + cross["class_a"].astype(str) + "->" + cross["class_b"].astype(str)

    def _aggregate_pair_table(key_col: str, scope: str) -> dict[str, Any]:
        records: dict[str, Any] = {}
        grouped = cross.groupby(key_col, dropna=False)
        for key, grp in grouped:
            pair_key = str(key or "").strip()
            if not pair_key:
                continue
            w = pd.to_numeric(grp["w_row"], errors="coerce").fillna(0.0)
            w_sum = float(w.sum())
            if w_sum <= 0.0:
                continue
            n_games = int(grp["gameid"].astype(str).nunique())
            result_edge = _weighted_mean(grp["result_edge"], w, 0.0)
            transfer_edge = _weighted_mean(grp["team_transfer_edge"], w, 0.0)
            domestic_edge = _weighted_mean(grp["team_domestic_edge"], w, 0.0)
            league_edge = _weighted_mean(grp["league_edge"], w, 0.0)
            avg_rel = _weighted_mean(grp["league_a"].map(lambda v: lg_rel_map.get(str(v), 0.50)), w, 0.50)
            avg_trans = _weighted_mean(grp["league_a"].map(lambda v: lg_trans_map.get(str(v), 0.50)), w, 0.50)
            avg_infl = _weighted_mean(grp["league_a"].map(lambda v: lg_infl_map.get(str(v), 0.50)), w, 0.50)
            strength_gap = _weighted_mean(grp["strength_gap_abs"], w, 0.0)
            n_support = float(n_games) / (float(n_games) + 12.0)
            pair_conf = _clip((0.45 * n_support) + (0.30 * avg_rel) + (0.25 * avg_trans), 0.0, 1.0)
            raw_edge = (
                (0.38 * transfer_edge)
                + (0.26 * result_edge)
                + (0.22 * league_edge)
                + (0.14 * domestic_edge)
            )
            portability_edge = float(max(-1.0, min(1.0, ((0.55 * pair_conf) + 0.45) * raw_edge)))
            first = grp.iloc[0]
            records[pair_key] = {
                "scope": scope,
                "sample_games": int(n_games),
                "sample_rows": len(grp),
                "pair_confidence": float(pair_conf),
                "pair_portability_edge": float(portability_edge),
                "pair_result_edge": float(result_edge),
                "pair_transfer_edge": float(transfer_edge),
                "pair_domestic_edge": float(domestic_edge),
                "pair_league_edge": float(league_edge),
                "avg_league_reliability": float(_clip(avg_rel, 0.0, 1.0)),
                "avg_league_translation": float(_clip(avg_trans, 0.0, 1.0)),
                "avg_league_inflation": float(_clip(avg_infl, 0.0, 1.0)),
                "strength_gap": float(max(0.0, min(1.0, strength_gap))),
                "league_a": str(first.get("league_a") or ""),
                "league_b": str(first.get("league_b") or ""),
                "tier_a": str(first.get("tier_a") or ""),
                "tier_b": str(first.get("tier_b") or ""),
                "strength_band_a": str(first.get("strength_band_a") or ""),
                "strength_band_b": str(first.get("strength_band_b") or ""),
                "macro_a": str(first.get("macro_a") or ""),
                "macro_b": str(first.get("macro_b") or ""),
                "class_a": str(first.get("class_a") or ""),
                "class_b": str(first.get("class_b") or ""),
                "class_relation": str(first.get("class_relation") or ""),
            }
        return records

    out: dict[str, Any] = {}
    for key_col, scope in (
        ("league_pair_key", "league_pair"),
        ("tier_pair_key", "tier_pair"),
        ("strength_pair_key", "strength_pair"),
        ("macro_pair_key", "macro_pair"),
        ("class_pair_key", "class_pair"),
    ):
        out.update(_aggregate_pair_table(key_col, scope))
    return out
