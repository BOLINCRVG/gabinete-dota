from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import pandas as pd

from app_core.champion_knowledge_core import get_champion_knowledge
from app_core.live_bet_shared_core import (
    PLAYER_PROP_META,
    ROLE_ORDER,
    parse_compact_draft_tokens,
    _metric_team_mean,
    _value_confidence_label,
    _weighted_value,
    _confidence_rank_from_score,
)
from app_core.lane_matchup_core import evaluate_lane_matchup
from app_core.comp_archetype_core import build_comp_profile, compare_comp_profiles
from mlcore.utils import safe_float_or_none as _safe_float
from app_core.live_bet_stats_core import (
    _draft_wr_parts,
    _laplace_win_pct,
    _league_alert_label,
    _mean_per_min,
    _safe_ratio,
    _scale_0_100,
    _series_mean_safe,
    _wr_dual_html,
)
from app_core.live_bet_champ_profile_core import (
    _champ_meta_profile,
    _champ_trait,
    _player_tactical_profile,
)
from app_core.live_bet_combo_core import (
    _build_combo_summary,
    _combo_metric_mean,
    _combo_support_summary,
    _market_call_from_combo_means,
    _weighted_combo_edge,
)


def _resolve_pick_names(
    *,
    tokens: list[str],
    vocab: list[str],
    fuzzy_match_one_fn: Callable[[str, list[str], float], tuple[str | None, list[str]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    resolved: list[str] = []
    unmatched: list[dict[str, Any]] = []
    for token in tokens:
        match, suggestions = fuzzy_match_one_fn(token, vocab, 0.6)
        if match is None:
            unmatched.append({"entrada": token, "sugestoes": suggestions[:5]})
            resolved.append(token)
        else:
            resolved.append(str(match))
    return resolved, unmatched


def resolve_compact_draft_sides(
    *,
    text: str,
    vocab: list[str],
    blue_team: str,
    team_a: str,
    team_b: str,
    fuzzy_match_one_fn: Callable[[str, list[str], float], tuple[str | None, list[str]]],
) -> dict[str, Any]:
    compact_tokens = parse_compact_draft_tokens(text, vocab, max_picks=10)
    resolved, unmatched = _resolve_pick_names(tokens=compact_tokens[:10], vocab=vocab, fuzzy_match_one_fn=fuzzy_match_one_fn)
    blue_picks = list(resolved[:5])
    red_picks = list(resolved[5:10])
    blue_side_team = str(blue_team or team_a)
    if blue_side_team == str(team_b):
        picks_a = red_picks
        picks_b = blue_picks
    else:
        picks_a = blue_picks
        picks_b = red_picks
    return {
        "resolved": resolved,
        "unmatched": unmatched,
        "blue_side_team": blue_side_team,
        "red_side_team": str(team_b) if blue_side_team == str(team_a) else str(team_a),
        "picks_a": picks_a,
        "picks_b": picks_b,
    }


def _label_player_spot_row(row: Mapping[str, Any]) -> str:
    player = str(row.get("player") or "-")
    champ = str(row.get("champ") or "-")
    return f"{player} / {champ}"


def _label_player_meta_row(row: Mapping[str, Any], *, label_key: str) -> str:
    if not row:
        return "-"
    base = _label_player_spot_row(row)
    label = str(row.get(label_key) or "").strip()
    return f"{base} | {label}" if label else base


def _build_player_spot_rows(
    *,
    df_scope_players: pd.DataFrame | None,
    team_name: str,
    picks: list[str],
    players: list[str],
    side_label: str,
    registry: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df_scope_players is None or df_scope_players.empty:
        return rows
    for idx, role in enumerate(ROLE_ORDER):
        champ_name = str((picks or [""] * len(ROLE_ORDER))[idx] or "").strip()
        player_name = str((players or [""] * len(ROLE_ORDER))[idx] or "").strip()
        if not champ_name or not player_name:
            continue
        subset = df_scope_players[
            (df_scope_players["team"].astype(str) == str(team_name))
            & (df_scope_players["player"].astype(str) == str(player_name))
        ].copy()
        if subset.empty:
            continue
        champ_subset = subset[subset["champ"].astype(str) == str(champ_name)].copy()
        side_subset = subset[subset["side"].astype(str).str.title() == str(side_label).title()].copy() if "side" in subset.columns else pd.DataFrame([])
        recent_subset = subset.copy()
        if "date" in recent_subset.columns:
            recent_subset["date"] = pd.to_datetime(recent_subset["date"], errors="coerce")
            recent_subset = recent_subset.sort_values(["date", "gameid"], ascending=[False, False], na_position="last").head(5)
        wr_overall = _laplace_win_pct(subset)
        wr_champ = _laplace_win_pct(champ_subset) if not champ_subset.empty else None
        wr_recent = _laplace_win_pct(recent_subset) if not recent_subset.empty else None
        wr_side = _laplace_win_pct(side_subset) if not side_subset.empty else None
        bundle = _blend_prop_bundle(
            [
                (_player_prop_share_bundle(subset, "K+A"), 0.34),
                (_player_prop_share_bundle(champ_subset, "K+A"), 0.34),
                (_player_prop_share_bundle(side_subset, "K+A"), 0.12),
                (_player_prop_share_bundle(recent_subset, "K+A"), 0.20),
            ]
        )
        kp_pct = 100.0 * float(_safe_float(bundle.get("kp")) or 0.0)
        kill_share_pct = 100.0 * float(_safe_float(bundle.get("kill_share")) or 0.0)
        death_share_pct = 100.0 * float(_safe_float(bundle.get("death_share")) or 0.0)
        kda_mean = _weighted_value(
            [
                (_safe_ratio((_series_mean_safe(subset, "kills") or 0.0) + (_series_mean_safe(subset, "assists") or 0.0), max((_series_mean_safe(subset, "deaths") or 0.0), 1.0)), 0.30),
                (_safe_ratio((_series_mean_safe(champ_subset, "kills") or 0.0) + (_series_mean_safe(champ_subset, "assists") or 0.0), max((_series_mean_safe(champ_subset, "deaths") or 0.0), 1.0)), 0.35),
                (_safe_ratio((_series_mean_safe(recent_subset, "kills") or 0.0) + (_series_mean_safe(recent_subset, "assists") or 0.0), max((_series_mean_safe(recent_subset, "deaths") or 0.0), 1.0)), 0.35),
            ]
        )
        dpm_value = _weighted_value(
            [
                (_mean_per_min(subset, "damagetochampions"), 0.30),
                (_mean_per_min(champ_subset, "damagetochampions"), 0.45),
                (_mean_per_min(recent_subset, "damagetochampions"), 0.25),
            ]
        )
        vpm_value = _weighted_value(
            [
                (_mean_per_min(subset, "visionscore"), 0.35),
                (_mean_per_min(champ_subset, "visionscore"), 0.25),
                (_mean_per_min(recent_subset, "visionscore"), 0.40),
            ]
        )
        tactical = _player_tactical_profile(str(role), champ_name, registry)
        meta_profile = _champ_meta_profile(
            df_scope_players=df_scope_players,
            champ_name=champ_name,
            role=str(role),
            side_label=str(side_label),
            registry=registry,
        )
        role_key = str(role).lower()
        dpm_target = 700.0 if role_key in {"bot", "mid"} else (540.0 if role_key in {"top", "jng"} else 240.0)
        vpm_target = 2.3 if role_key == "sup" else (1.4 if role_key == "jng" else 0.9)
        dpm_score = _scale_0_100(dpm_value, target=dpm_target)
        vpm_score = _scale_0_100(vpm_value, target=vpm_target)
        kda_score = _scale_0_100(kda_mean, target=5.0 if role_key in {"bot", "mid"} else 4.0)
        if str(tactical.get("primary_label") or "") in {"inicia", "setup", "frontline"}:
            util_score = 0.42 * kp_pct + 0.33 * vpm_score + 0.25 * kda_score
        else:
            util_score = 0.44 * dpm_score + 0.26 * kp_pct + 0.18 * kill_share_pct + 0.12 * kda_score
        comfort_pct = _weighted_value(
            [
                (wr_champ, 0.5),
                (wr_side, 0.15),
                (wr_recent, 0.15),
                (wr_overall, 0.20),
            ]
        )
        impact_score = _weighted_value(
            [
                (comfort_pct, 0.34),
                (util_score, 0.30),
                (kp_pct, 0.14),
                (kda_score, 0.10),
                (_scale_0_100(tactical.get("engage_score") if str(tactical.get("primary_label") or "") in {"inicia", "setup", "frontline"} else tactical.get("carry_score"), target=3.0), 0.12),
            ]
        )
        carry_spot_score = _weighted_value(
            [
                (_scale_0_100(tactical.get("carry_score"), target=3.0), 0.42),
                (dpm_score, 0.22),
                (kill_share_pct, 0.14),
                (kp_pct, 0.10),
                (comfort_pct, 0.12),
            ]
        )
        rows.append(
            {
                "player": player_name,
                "champ": champ_name,
                "role": role_key,
                "primary_label": str(tactical.get("primary_label") or "-"),
                "risk_label": str(tactical.get("risk_label") or "-"),
                "impact_score": float(_safe_float(impact_score) or 0.0),
                "carry_spot_score": float(_safe_float(carry_spot_score) or 0.0),
                "engage_score": float(_safe_float(tactical.get("engage_score")) or 0.0),
                "frontline_score": float(_safe_float(tactical.get("frontline_score")) or 0.0),
                "exposure_score": float(_safe_float(tactical.get("exposure_score")) or 0.0),
                "wr_overall": wr_overall,
                "wr_champ": wr_champ,
                "wr_recent": wr_recent,
                "kp_pct": kp_pct,
                "kill_share_pct": kill_share_pct,
                "death_share_pct": death_share_pct,
                "dpm": dpm_value,
                "vpm": vpm_value,
                "meta_label": str(meta_profile.get("label") or "Meta neutro"),
                "meta_score": float(_safe_float(meta_profile.get("score")) or 50.0),
                "meta_sample": int(_safe_float(meta_profile.get("sample")) or 0),
                "global_meta_label": str(meta_profile.get("global_label") or "Situacional"),
                "global_meta_score": float(_safe_float(meta_profile.get("global_score")) or 50.0),
            }
        )
    return rows


def _team_player_leadership(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {"best_player": {}, "carry_spot": {}, "engage_engine": {}, "most_exposed": {}}
    return {
        "best_player": max(rows, key=lambda row: float(_safe_float(row.get("impact_score")) or 0.0)),
        "carry_spot": max(rows, key=lambda row: float(_safe_float(row.get("carry_spot_score")) or 0.0)),
        "engage_engine": max(rows, key=lambda row: float(_safe_float(row.get("engage_score")) or 0.0)),
        "most_exposed": max(rows, key=lambda row: float(_safe_float(row.get("exposure_score")) or 0.0)),
    }


def _team_meta_leadership(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {"best_meta": {}, "weak_meta": {}, "best_global_meta": {}, "weak_global_meta": {}}
    meta_rows = [row for row in rows if int(_safe_float(row.get("meta_sample")) or 0) > 0]
    global_rows = [row for row in rows if str(row.get("global_meta_label") or "") not in {"", "Sem base"}]
    return {
        "best_meta": max((meta_rows or rows), key=lambda row: float(_safe_float(row.get("meta_score")) or 0.0)),
        "weak_meta": min((meta_rows or rows), key=lambda row: float(_safe_float(row.get("meta_score")) or 50.0)),
        "best_global_meta": max((global_rows or rows), key=lambda row: float(_safe_float(row.get("global_meta_score")) or 0.0)),
        "weak_global_meta": min((global_rows or rows), key=lambda row: float(_safe_float(row.get("global_meta_score")) or 50.0)),
    }


def _build_pairs(df_scope_players: pd.DataFrame) -> pd.DataFrame:
    if df_scope_players is None or df_scope_players.empty:
        return pd.DataFrame([])
    req = {"gameid", "pos", "side", "champ", "result"}
    if not req.issubset(df_scope_players.columns):
        return pd.DataFrame([])
    d2 = df_scope_players[["gameid", "pos", "side", "champ", "result"]].copy()
    d2 = d2[(d2["pos"] != "") & (d2["champ"] != "")].copy()
    d2["side"] = d2["side"].astype(str).str.title()
    blue = d2[d2["side"] == "Blue"].copy()
    red = d2[d2["side"] == "Red"].copy()
    pairs = blue.merge(red, on=["gameid", "pos"], suffixes=("_blue", "_red"))
    pairs["blue_win"] = pd.to_numeric(pairs["result_blue"], errors="coerce").fillna(0).astype(int)
    pairs["red_win"] = pd.to_numeric(pairs["result_red"], errors="coerce").fillna(0).astype(int)
    return pairs


def _lane_h2h_wr(pairs: pd.DataFrame, champ_a: str, champ_b: str, pos: str) -> tuple[Optional[float], int]:
    if pairs is None or pairs.empty:
        return None, 0
    x1 = pairs[(pairs["pos"] == pos) & (pairs["champ_blue"] == champ_a) & (pairs["champ_red"] == champ_b)].copy()
    w1 = int(x1["blue_win"].sum())
    n1 = int(x1.shape[0])
    x2 = pairs[(pairs["pos"] == pos) & (pairs["champ_red"] == champ_a) & (pairs["champ_blue"] == champ_b)].copy()
    w2 = int(x2["red_win"].sum())
    n2 = int(x2.shape[0])
    n = int(n1 + n2)
    if n <= 0:
        return None, 0
    return float((w1 + w2) / float(n)), n


def _pct_value(value: Any) -> Optional[float]:
    text = str(value or "").strip().replace("%", "").replace(",", ".")
    if not text or text == "-":
        return None
    return _safe_float(text)


def _champion_tag(*, player_wr: Optional[float], league_wr: Optional[float], player_games: int) -> str:
    if player_games <= 0 or player_wr is None or league_wr is None:
        return "Sem base"
    delta = float(player_wr) - float(league_wr)
    if player_games >= 4 and delta >= 5.0:
        return "Conforto"
    if player_games >= 4 and delta <= -5.0:
        return "Fraco"
    return "Neutro"


def _lane_edge_text(edge_pp: Any, team_a: str, team_b: str) -> str:
    value = _safe_float(edge_pp)
    if value is None or abs(value) < 1.5:
        return "Equilibrado"
    return f"Lane {team_a}" if value > 0 else f"Lane {team_b}"


def _build_lane_cards(
    lane_view: pd.DataFrame,
    team_a: str,
    team_b: str,
    registry: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cards_a: list[dict[str, Any]] = []
    cards_b: list[dict[str, Any]] = []
    if lane_view is None or lane_view.empty:
        return cards_a, cards_b
    for _, row in lane_view.iterrows():
        pos = str(row.get("Pos") or "-").upper()
        edge_pp = _pct_value(row.get("Edge (p.p.)"))
        a_player_wr = _pct_value(row.get(f"WR Player {team_a}"))
        a_league_wr = _pct_value(row.get(f"WR Liga {team_a}"))
        b_player_wr = _pct_value(row.get(f"WR Player {team_b}"))
        b_league_wr = _pct_value(row.get(f"WR Liga {team_b}"))
        a_player_games = int(_safe_float(row.get(f"N Player {team_a}")) or 0)
        b_player_games = int(_safe_float(row.get(f"N Player {team_b}")) or 0)
        tactical_a = _player_tactical_profile(str(row.get("Pos") or ""), str(row.get(f"{team_a} Champ") or ""), registry)
        tactical_b = _player_tactical_profile(str(row.get("Pos") or ""), str(row.get(f"{team_b} Champ") or ""), registry)
        cards_a.append(
            {
                "pos": pos,
                "player": str(row.get(f"{team_a} Player") or "-"),
                "champ": str(row.get(f"{team_a} Champ") or "-"),
                "player_wr": row.get(f"WR Player {team_a}") or "-",
                "league_wr": row.get(f"WR Liga {team_a}") or "-",
                "games": a_player_games,
                "player_wr_blue": row.get(f"WR Player Blue {team_a}") or "-",
                "player_wr_red": row.get(f"WR Player Red {team_a}") or "-",
                "champ_wr_blue": row.get(f"WR Champ Blue {team_a}") or "-",
                "champ_wr_red": row.get(f"WR Champ Red {team_a}") or "-",
                "player_games_blue": int(_safe_float(row.get(f"N Player Blue {team_a}")) or 0),
                "player_games_red": int(_safe_float(row.get(f"N Player Red {team_a}")) or 0),
                "champ_games_blue": int(_safe_float(row.get(f"N Champ Blue {team_a}")) or 0),
                "champ_games_red": int(_safe_float(row.get(f"N Champ Red {team_a}")) or 0),
                "tag": _champion_tag(player_wr=a_player_wr, league_wr=a_league_wr, player_games=a_player_games),
                "edge_pp": edge_pp,
                "lane_read": str(row.get("Leitura Lane") or "-"),
                "early_team": str(row.get("Early") or "-"),
                "late_team": str(row.get("Late") or "-"),
                "lane_confidence": str(row.get("Confianca Lane") or "-"),
                "team_plan": str(row.get(f"Plano {team_a}") or "-"),
                "primary_label": str(tactical_a.get("primary_label") or "-"),
                "risk_label": str(tactical_a.get("risk_label") or "-"),
                "carry_score": float(_safe_float(tactical_a.get("carry_score")) or 0.0),
                "engage_score": float(_safe_float(tactical_a.get("engage_score")) or 0.0),
                "frontline_score": float(_safe_float(tactical_a.get("frontline_score")) or 0.0),
                "exposure_score": float(_safe_float(tactical_a.get("exposure_score")) or 0.0),
                "meta_label": str(row.get(f"Meta {team_a}") or "Meta neutro"),
                "meta_score": float(_safe_float(row.get(f"Meta Score {team_a}")) or 50.0),
                "meta_sample": int(_safe_float(row.get(f"Meta Sample {team_a}")) or 0),
                "global_meta_label": str(row.get(f"Meta Global {team_a}") or "Situacional"),
                "global_meta_score": float(_safe_float(row.get(f"Meta Global Score {team_a}")) or 50.0),
                "champ_wr_world_blue": row.get(f"WR Champ Mundo Blue {team_a}") or "-",
                "champ_wr_world_red": row.get(f"WR Champ Mundo Red {team_a}") or "-",
                "champ_games_world_blue": int(_safe_float(row.get(f"N Champ Mundo Blue {team_a}")) or 0),
                "champ_games_world_red": int(_safe_float(row.get(f"N Champ Mundo Red {team_a}")) or 0),
                "league_alert": str(row.get(f"Alerta Liga {team_a}") or "").strip(),
            }
        )
        cards_b.append(
            {
                "pos": pos,
                "player": str(row.get(f"{team_b} Player") or "-"),
                "champ": str(row.get(f"{team_b} Champ") or "-"),
                "player_wr": row.get(f"WR Player {team_b}") or "-",
                "league_wr": row.get(f"WR Liga {team_b}") or "-",
                "games": b_player_games,
                "player_wr_blue": row.get(f"WR Player Blue {team_b}") or "-",
                "player_wr_red": row.get(f"WR Player Red {team_b}") or "-",
                "champ_wr_blue": row.get(f"WR Champ Blue {team_b}") or "-",
                "champ_wr_red": row.get(f"WR Champ Red {team_b}") or "-",
                "player_games_blue": int(_safe_float(row.get(f"N Player Blue {team_b}")) or 0),
                "player_games_red": int(_safe_float(row.get(f"N Player Red {team_b}")) or 0),
                "champ_games_blue": int(_safe_float(row.get(f"N Champ Blue {team_b}")) or 0),
                "champ_games_red": int(_safe_float(row.get(f"N Champ Red {team_b}")) or 0),
                "tag": _champion_tag(player_wr=b_player_wr, league_wr=b_league_wr, player_games=b_player_games),
                "edge_pp": edge_pp,
                "lane_read": str(row.get("Leitura Lane") or "-"),
                "early_team": str(row.get("Early") or "-"),
                "late_team": str(row.get("Late") or "-"),
                "lane_confidence": str(row.get("Confianca Lane") or "-"),
                "team_plan": str(row.get(f"Plano {team_b}") or "-"),
                "primary_label": str(tactical_b.get("primary_label") or "-"),
                "risk_label": str(tactical_b.get("risk_label") or "-"),
                "carry_score": float(_safe_float(tactical_b.get("carry_score")) or 0.0),
                "engage_score": float(_safe_float(tactical_b.get("engage_score")) or 0.0),
                "frontline_score": float(_safe_float(tactical_b.get("frontline_score")) or 0.0),
                "exposure_score": float(_safe_float(tactical_b.get("exposure_score")) or 0.0),
                "meta_label": str(row.get(f"Meta {team_b}") or "Meta neutro"),
                "meta_score": float(_safe_float(row.get(f"Meta Score {team_b}")) or 50.0),
                "meta_sample": int(_safe_float(row.get(f"Meta Sample {team_b}")) or 0),
                "global_meta_label": str(row.get(f"Meta Global {team_b}") or "Situacional"),
                "global_meta_score": float(_safe_float(row.get(f"Meta Global Score {team_b}")) or 50.0),
                "champ_wr_world_blue": row.get(f"WR Champ Mundo Blue {team_b}") or "-",
                "champ_wr_world_red": row.get(f"WR Champ Mundo Red {team_b}") or "-",
                "champ_games_world_blue": int(_safe_float(row.get(f"N Champ Mundo Blue {team_b}")) or 0),
                "champ_games_world_red": int(_safe_float(row.get(f"N Champ Mundo Red {team_b}")) or 0),
                "league_alert": str(row.get(f"Alerta Liga {team_b}") or "").strip(),
            }
        )
    return cards_a, cards_b


def _player_role_function(role: str, champ_name: str, registry: Mapping[str, Any] | None) -> str:
    profile = _player_tactical_profile(role, champ_name, registry)
    label_map = {
        "carrega": "carry",
        "inicia": "abre luta",
        "setup": "peel/setup",
        "frontline": "frontline",
        "side": "side",
    }
    return str(label_map.get(str(profile.get("primary_label") or ""), profile.get("primary_label") or "-"))


def _team_role_snapshot(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {"carry": {}, "engage": {}, "exposed": {}, "frontline": {}}

    def _pick_best(score_key: str, *, extra_wr: float = 0.0) -> dict[str, Any]:
        return max(
            rows,
            key=lambda row: float(_safe_float(row.get(score_key)) or 0.0)
            + extra_wr * float(_safe_float(row.get("player_wr")) or 50.0)
            + (2.5 if str(row.get("tag") or "") == "Conforto" else 0.0),
        )

    return {
        "carry": _pick_best("carry_score", extra_wr=0.015),
        "engage": _pick_best("engage_score"),
        "exposed": _pick_best("exposure_score"),
        "frontline": _pick_best("frontline_score"),
    }


def _edge_share_text(delta: Any, *, team_a: str, team_b: str, scale: float = 8.0) -> str:
    value = float(_safe_float(delta) or 0.0)
    score = max(min(50.0 + (value * scale), 78.0), 22.0)
    other = 100.0 - score
    if abs(score - 50.0) < 2.0:
        return "50/50"
    leader = str(team_a) if score >= 50.0 else str(team_b)
    lead_value = score if score >= 50.0 else other
    trail_value = other if score >= 50.0 else score
    return f"{leader} {round(lead_value)}/{round(trail_value)}"


def build_live_bet_handicap_summary(
    *,
    df_scope: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    blue_side_team: str,
    draft: Mapping[str, Any],
    line_values: Mapping[str, Any],
    format_num_fn: Callable[[float, int], str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if df_scope is None or df_scope.empty:
        return {"rows": pd.DataFrame([]), "best_row": None}
    side_a = "Blue" if str(blue_side_team) == str(team_a) else "Red"
    side_b = "Red" if side_a == "Blue" else "Blue"
    verdict = dict(draft.get("verdict") or {})
    comp_compare = dict(draft.get("comp_compare") or {})
    definitions = [
        ("HC Kills T1", f"HC Kills {team_a}", "kills_for", "kills_against", float(_safe_float(line_values.get("HC Kills T1")) or 0.0), 0.9),
        ("HC Torres T1", f"HC Torres {team_a}", "towers_for", "towers_against", float(_safe_float(line_values.get("HC Torres T1")) or 0.0), 0.45),
    ]
    for market_key, market_label, for_col, against_col, line_value, strong_delta in definitions:
        if market_key not in line_values or _safe_float(line_values.get(market_key)) is None:
            continue
        if for_col not in df_scope.columns or against_col not in df_scope.columns:
            continue
        team_a_for = _weighted_value(
            [
                (_metric_team_mean(df_scope, team_name=str(team_a), column=for_col), 0.42),
                (_metric_team_mean(df_scope, team_name=str(team_b), column=against_col), 0.23),
                (_metric_team_mean(df_scope, team_name=str(team_a), column=for_col, side=side_a), 0.20),
                (_metric_team_mean(df_scope, team_name=str(team_a), column=for_col, opponent=str(team_b)), 0.15),
            ]
        )
        team_b_for = _weighted_value(
            [
                (_metric_team_mean(df_scope, team_name=str(team_b), column=for_col), 0.42),
                (_metric_team_mean(df_scope, team_name=str(team_a), column=against_col), 0.23),
                (_metric_team_mean(df_scope, team_name=str(team_b), column=for_col, side=side_b), 0.20),
                (_metric_team_mean(df_scope, team_name=str(team_b), column=for_col, opponent=str(team_a)), 0.15),
            ]
        )
        if team_a_for is None or team_b_for is None:
            continue
        fair_diff = float(team_a_for) - float(team_b_for)
        if market_key == "HC Kills T1":
            fair_diff += 0.08 * float(_safe_float(verdict.get("score_pp")) or 0.0)
        if market_key == "HC Torres T1":
            fair_diff += 0.18 * float(_safe_float(comp_compare.get("towers_delta")) or 0.0)
        delta = float(fair_diff) - float(line_value)
        if market_key == "HC Torres T1" and abs(delta) < 0.15:
            reading = f"ViÃ©s {team_a}" if delta >= 0 else f"ViÃ©s {team_b}"
        else:
            reading = "Muito perto da linha"
            if abs(delta) >= 0.15:
                reading = f"{team_a} cobre {format_num_fn(line_value, 1)}" if delta > 0 else f"{team_b} cobre {format_num_fn(abs(line_value), 1)}"
        rows.append(
            {
                "Mercado": market_label,
                "Linha": format_num_fn(line_value, 1),
                "Justo": format_num_fn(fair_diff, 1),
                "Leitura": reading,
                "Confianca": _value_confidence_label(abs_delta=abs(delta), evidence_count=4, strong_delta=strong_delta),
                "Drivers": (
                    f"{team_a} for {format_num_fn(float(team_a_for), 1)} | "
                    f"{team_b} for {format_num_fn(float(team_b_for), 1)}"
                ),
                "_abs_delta": abs(delta),
            }
        )
    df_rows = pd.DataFrame(rows)
    if df_rows.empty:
        return {"rows": df_rows, "best_row": None}
    best_row = max(rows, key=lambda row: float(row.get("_abs_delta") or 0.0))
    return {"rows": df_rows.drop(columns=["_abs_delta"], errors="ignore"), "best_row": best_row}


def _player_prop_value(series: pd.DataFrame, prop_name: str) -> pd.Series:
    meta = dict(PLAYER_PROP_META.get(prop_name) or {})
    columns = list(meta.get("columns") or [])
    if not columns:
        return pd.Series(dtype=float)
    out = pd.Series(0.0, index=series.index, dtype=float)
    for col in columns:
        if col in series.columns:
            out = out + pd.to_numeric(series.get(col), errors="coerce").fillna(0.0)
    return out


def _player_prop_share_bundle(df_in: pd.DataFrame, prop_name: str) -> dict[str, Optional[float]]:
    if df_in is None or df_in.empty:
        return {
            "team_kills": None,
            "team_deaths": None,
            "kp": None,
            "kill_share": None,
            "assist_share": None,
            "death_share": None,
            "context_prop": None,
        }

    team_kills = _series_mean_safe(df_in, "teamkills")
    team_deaths = _series_mean_safe(df_in, "teamdeaths")
    kills_mean = _series_mean_safe(df_in, "kills")
    deaths_mean = _series_mean_safe(df_in, "deaths")
    assists_mean = _series_mean_safe(df_in, "assists")
    ka_mean = _safe_float(_player_prop_value(df_in, "K+A").mean())
    kp = _safe_ratio((kills_mean or 0.0) + (assists_mean or 0.0), team_kills)
    kill_share = _safe_ratio(kills_mean, team_kills)
    assist_share = _safe_ratio(assists_mean, team_kills)
    death_share = _safe_ratio(deaths_mean, team_deaths)

    if prop_name == "Kills":
        context_prop = (team_kills * kill_share) if team_kills is not None and kill_share is not None else None
    elif prop_name == "Deaths":
        context_prop = (team_deaths * death_share) if team_deaths is not None and death_share is not None else None
    elif prop_name == "Assists":
        context_prop = (team_kills * assist_share) if team_kills is not None and assist_share is not None else None
    else:
        context_prop = (team_kills * kp) if team_kills is not None and kp is not None else ka_mean

    return {
        "team_kills": team_kills,
        "team_deaths": team_deaths,
        "kp": kp,
        "kill_share": kill_share,
        "assist_share": assist_share,
        "death_share": death_share,
        "context_prop": context_prop,
    }


def _blend_prop_bundle(parts: list[tuple[dict[str, Optional[float]], float]]) -> dict[str, Optional[float]]:
    return {
        "team_kills": _weighted_value([(part.get("team_kills"), weight) for part, weight in parts]),
        "team_deaths": _weighted_value([(part.get("team_deaths"), weight) for part, weight in parts]),
        "kp": _weighted_value([(part.get("kp"), weight) for part, weight in parts]),
        "kill_share": _weighted_value([(part.get("kill_share"), weight) for part, weight in parts]),
        "assist_share": _weighted_value([(part.get("assist_share"), weight) for part, weight in parts]),
        "death_share": _weighted_value([(part.get("death_share"), weight) for part, weight in parts]),
        "context_prop": _weighted_value([(part.get("context_prop"), weight) for part, weight in parts]),
    }


def _player_prop_spot_context(
    bundle: Mapping[str, Any] | None,
    *,
    prop_name: str,
    expected_team_kills: Any,
    expected_team_deaths: Any,
) -> Optional[float]:
    bundle_map = dict(bundle or {})
    team_kills = _safe_float(expected_team_kills)
    team_deaths = _safe_float(expected_team_deaths)
    kp = _safe_float(bundle_map.get("kp"))
    kill_share = _safe_float(bundle_map.get("kill_share"))
    assist_share = _safe_float(bundle_map.get("assist_share"))
    death_share = _safe_float(bundle_map.get("death_share"))
    if prop_name == "Kills":
        if team_kills is None or kill_share is None:
            return None
        return float(team_kills) * float(kill_share)
    if prop_name == "Deaths":
        if team_deaths is None or death_share is None:
            return None
        return float(team_deaths) * float(death_share)
    if prop_name == "Assists":
        if team_kills is None or assist_share is None:
            return None
        return float(team_kills) * float(assist_share)
    if team_kills is None or kp is None:
        return None
    return float(team_kills) * float(kp)


def _player_prop_semantic_shift(
    *,
    prop_name: str,
    role: str,
    champ: Mapping[str, Any] | None,
    comp_compare: Mapping[str, Any],
    team_name: str,
) -> float:
    role_key = str(role or "").strip().lower()
    late = _champ_trait(champ, "late")
    pick = _champ_trait(champ, "pick")
    dive = _champ_trait(champ, "dive")
    peel = _champ_trait(champ, "peel")
    frontline = _champ_trait(champ, "frontline")
    engage = _champ_trait(champ, "engage")
    mobility = _champ_trait(champ, "mobility")
    range_score = _champ_trait(champ, "range")
    teamfight = _champ_trait(champ, "teamfight")

    kills_delta = float(_safe_float(comp_compare.get("kills_delta")) or 0.0)
    engage_team = str(comp_compare.get("engage_team") or "")
    late_team = str(comp_compare.get("late_team") or "")

    team_sign = 1.0 if str(team_name) == engage_team else (-1.0 if engage_team and str(team_name) != engage_team and engage_team != "Equilibrado" else 0.0)
    late_sign = 1.0 if str(team_name) == late_team else (-1.0 if late_team and str(team_name) != late_team and late_team != "Equilibrado" else 0.0)

    if prop_name == "Kills":
        shift = 0.28 * late + 0.24 * pick + 0.18 * dive + 0.10 * range_score - 0.14 * peel
        if role_key == "sup":
            shift -= 0.55
        elif role_key == "bot":
            shift += 0.18 * range_score
        shift += 0.05 * kills_delta * late_sign
        return shift
    if prop_name == "Deaths":
        shift = 0.32 * frontline + 0.24 * engage + 0.18 * dive - 0.22 * mobility - 0.16 * peel
        if role_key in {"sup", "jng"}:
            shift += 0.18
        if role_key == "bot":
            shift += 0.10 * max(-team_sign, 0.0)
        shift += 0.06 * max(-kills_delta * late_sign, 0.0)
        return shift
    if prop_name == "Assists":
        shift = 0.34 * engage + 0.32 * peel + 0.16 * teamfight - 0.10 * late
        if role_key == "sup":
            shift += 0.45
        elif role_key == "jng":
            shift += 0.20
        elif role_key == "bot":
            shift -= 0.20
        shift += 0.05 * max(kills_delta * team_sign, 0.0)
        return shift
    shift = 0.22 * engage + 0.20 * teamfight + 0.18 * late + 0.10 * pick
    if role_key == "sup":
        shift += 0.20
    shift += 0.04 * kills_delta * (team_sign if team_sign != 0.0 else late_sign)
    return shift


def build_live_bet_player_prop_summary(
    *,
    df_scope_players: pd.DataFrame | None,
    team_name: str,
    opponent_name: str,
    player_name: str,
    champ_name: str,
    side_label: str,
    prop_name: str,
    line_value: Any,
    registry: Mapping[str, Any] | None,
    comp_compare: Mapping[str, Any],
    expected_team_kills: Any = None,
    expected_team_deaths: Any = None,
    format_num_fn: Callable[[float, int], str],
) -> dict[str, Any]:
    line_float = _safe_float(line_value)
    if df_scope_players is None or df_scope_players.empty or not str(player_name or "").strip() or not str(champ_name or "").strip() or line_float is None:
        return {"ok": False}
    subset = df_scope_players[
        (df_scope_players["team"].astype(str) == str(team_name))
        & (df_scope_players["player"].astype(str) == str(player_name))
    ].copy()
    if subset.empty:
        return {"ok": False}
    role = ""
    if "pos" in subset.columns:
        pos_series = subset["pos"].dropna().astype(str)
        if not pos_series.empty:
            role = str(pos_series.mode().iloc[0])
    champ_subset = subset[subset["champ"].astype(str) == str(champ_name)].copy()
    side_subset = subset[subset["side"].astype(str).str.title() == str(side_label).title()].copy() if "side" in subset.columns else pd.DataFrame([])
    recent_subset = subset.copy()
    if "date" in recent_subset.columns:
        recent_subset["date"] = pd.to_datetime(recent_subset["date"], errors="coerce")
        recent_subset = recent_subset.sort_values(["date", "gameid"], ascending=[False, False], na_position="last").head(5)
    overall_mean = _safe_float(_player_prop_value(subset, prop_name).mean())
    champ_mean = _safe_float(_player_prop_value(champ_subset, prop_name).mean()) if not champ_subset.empty else None
    side_mean = _safe_float(_player_prop_value(side_subset, prop_name).mean()) if not side_subset.empty else None
    recent_mean = _safe_float(_player_prop_value(recent_subset, prop_name).mean()) if not recent_subset.empty else None
    champ = get_champion_knowledge(registry, champ_name, role_hint=role)

    overall_bundle = _player_prop_share_bundle(subset, prop_name)
    champ_bundle = _player_prop_share_bundle(champ_subset, prop_name) if not champ_subset.empty else _player_prop_share_bundle(pd.DataFrame([]), prop_name)
    side_bundle = _player_prop_share_bundle(side_subset, prop_name) if not side_subset.empty else _player_prop_share_bundle(pd.DataFrame([]), prop_name)
    recent_bundle = _player_prop_share_bundle(recent_subset, prop_name) if not recent_subset.empty else _player_prop_share_bundle(pd.DataFrame([]), prop_name)
    spot_context = _weighted_value(
        [
            (_player_prop_spot_context(overall_bundle, prop_name=str(prop_name), expected_team_kills=expected_team_kills, expected_team_deaths=expected_team_deaths), 0.28),
            (_player_prop_spot_context(champ_bundle, prop_name=str(prop_name), expected_team_kills=expected_team_kills, expected_team_deaths=expected_team_deaths), 0.34),
            (_player_prop_spot_context(side_bundle, prop_name=str(prop_name), expected_team_kills=expected_team_kills, expected_team_deaths=expected_team_deaths), 0.14),
            (_player_prop_spot_context(recent_bundle, prop_name=str(prop_name), expected_team_kills=expected_team_kills, expected_team_deaths=expected_team_deaths), 0.24),
        ]
    )
    blended_bundle = _blend_prop_bundle(
        [
            (overall_bundle, 0.34),
            (champ_bundle, 0.34),
            (side_bundle, 0.14),
            (recent_bundle, 0.18),
        ]
    )
    share_context = _weighted_value(
        [
            (spot_context, 0.7),
            (blended_bundle.get("context_prop"), 0.3),
        ]
    )
    semantic_shift = _player_prop_semantic_shift(
        prop_name=str(prop_name),
        role=str(role),
        champ=champ,
        comp_compare=comp_compare,
        team_name=str(team_name),
    )
    fair_value = _weighted_value(
        [
            (overall_mean, 0.24),
            (champ_mean, 0.28),
            (side_mean, 0.10),
            (recent_mean, 0.14),
            (share_context, 0.24),
        ]
    )
    if fair_value is None:
        return {"ok": False}
    fair_value = float(fair_value) + float(semantic_shift)
    delta = float(fair_value) - float(line_float)
    meta = dict(PLAYER_PROP_META.get(prop_name) or {})
    reading = "Muito perto da linha"
    if abs(delta) >= 0.18:
        reading = f"{meta.get('high_label', 'Over')} {format_num_fn(float(line_float), 1)}" if delta > 0 else f"{meta.get('low_label', 'Under')} {format_num_fn(float(line_float), 1)}"
    overall_n = len(subset.index)
    champ_n = len(champ_subset.index)
    side_n = len(side_subset.index)
    recent_n = len(recent_subset.index)
    evidence_count = 0
    evidence_count += 1 if overall_mean is not None and overall_n >= 6 else 0
    evidence_count += 1 if champ_mean is not None and champ_n >= 3 else 0
    evidence_count += 1 if side_mean is not None and side_n >= 3 else 0
    evidence_count += 1 if recent_mean is not None and recent_n >= 3 else 0
    evidence_count += 1 if share_context is not None else 0
    confidence = _value_confidence_label(abs_delta=abs(delta), evidence_count=evidence_count, strong_delta=float(meta.get("delta_high") or 0.8))
    kp_text = format_num_fn(100.0 * float(blended_bundle["kp"]), 1) + "%" if blended_bundle.get("kp") is not None else "-"
    kill_share_text = format_num_fn(100.0 * float(blended_bundle["kill_share"]), 1) + "%" if blended_bundle.get("kill_share") is not None else "-"
    assist_share_text = format_num_fn(100.0 * float(blended_bundle["assist_share"]), 1) + "%" if blended_bundle.get("assist_share") is not None else "-"
    death_share_text = format_num_fn(100.0 * float(blended_bundle["death_share"]), 1) + "%" if blended_bundle.get("death_share") is not None else "-"
    expected_kills_text = format_num_fn(float(expected_team_kills), 1) if _safe_float(expected_team_kills) is not None else "-"
    expected_deaths_text = format_num_fn(float(expected_team_deaths), 1) if _safe_float(expected_team_deaths) is not None else "-"
    drivers = [
        f"Geral {format_num_fn(float(overall_mean), 1)} ({overall_n}j)" if overall_mean is not None else "",
        f"{champ_name} {format_num_fn(float(champ_mean), 1)} ({champ_n}j)" if champ_mean is not None else "",
        f"{side_label} {format_num_fn(float(side_mean), 1)} ({side_n}j)" if side_mean is not None else "",
        f"Recente {format_num_fn(float(recent_mean), 1)} ({recent_n}j)" if recent_mean is not None else "",
        f"Spot {format_num_fn(float(share_context), 1)}" if share_context is not None else "",
        f"Kills time {expected_kills_text}" if prop_name in {"Kills", "Assists", "K+A"} and expected_kills_text != "-" else "",
        f"Deaths esperadas {expected_deaths_text}" if prop_name == "Deaths" and expected_deaths_text != "-" else "",
        f"KP {kp_text}" if kp_text != "-" else "",
        f"Kill share {kill_share_text}" if prop_name in {"Kills", "K+A"} and kill_share_text != "-" else "",
        f"Assist share {assist_share_text}" if prop_name in {"Assists", "K+A"} and assist_share_text != "-" else "",
        f"Death share {death_share_text}" if prop_name == "Deaths" and death_share_text != "-" else "",
        f"Funcao {_player_role_function(role, champ_name, registry)}",
        f"Spot {team_name} vs {opponent_name}",
    ]
    return {
        "ok": True,
        "player": str(player_name),
        "champ": str(champ_name),
        "role": role or "-",
        "prop": str(prop_name),
        "line": format_num_fn(float(line_float), 1),
        "fair": format_num_fn(float(fair_value), 1),
        "reading": reading,
        "confidence": confidence,
        "drivers": " | ".join([item for item in drivers if item]),
        "overall_mean": overall_mean,
        "champ_mean": champ_mean,
        "side_mean": side_mean,
        "recent_mean": recent_mean,
        "share_context": share_context,
        "kp": blended_bundle.get("kp"),
        "kill_share": blended_bundle.get("kill_share"),
        "assist_share": blended_bundle.get("assist_share"),
        "death_share": blended_bundle.get("death_share"),
        "expected_team_kills": _safe_float(expected_team_kills),
        "expected_team_deaths": _safe_float(expected_team_deaths),
    }


def build_live_bet_tactical_notes(
    *,
    team_a: str,
    team_b: str,
    picks_a: list[str],
    picks_b: list[str],
    players_a: list[str],
    players_b: list[str],
    comp_compare: Mapping[str, Any],
    lane_view: pd.DataFrame | None,
    board_rows_a: list[dict[str, Any]] | None,
    board_rows_b: list[dict[str, Any]] | None,
    registry: Mapping[str, Any] | None,
    format_num_fn: Callable[[float, int], str],
) -> list[str]:
    notes: list[str] = []
    early_text = _edge_share_text(comp_compare.get("early_delta"), team_a=str(team_a), team_b=str(team_b), scale=7.0)
    late_text = _edge_share_text(comp_compare.get("late_delta"), team_a=str(team_a), team_b=str(team_b), scale=7.0)
    if early_text != "50/50":
        notes.append(f"Early estimado: {early_text}.")
    if late_text != "50/50":
        notes.append(f"Late estimado: {late_text}.")
    tower_total_reason = str(comp_compare.get("tower_total_reason") or "").strip()
    if tower_total_reason:
        notes.append(f"Torres: {tower_total_reason}")
    dragons_total_reason = str(comp_compare.get("dragons_total_reason") or "").strip()
    if dragons_total_reason:
        notes.append(f"Dragoes: {dragons_total_reason}")
    barons_total_reason = str(comp_compare.get("barons_total_reason") or "").strip()
    if barons_total_reason:
        notes.append(f"Baroes: {barons_total_reason}")
    snapshot_a = _team_role_snapshot(list(board_rows_a or []))
    snapshot_b = _team_role_snapshot(list(board_rows_b or []))
    meta_a = _team_meta_leadership(list(board_rows_a or []))
    meta_b = _team_meta_leadership(list(board_rows_b or []))

    def _row_label(row: Mapping[str, Any]) -> str:
        if not row:
            return ""
        player = str(row.get("player") or "").strip()
        champ = str(row.get("champ") or "").strip()
        if player and champ:
            return f"{player} ({champ})"
        return player or champ

    engage_a = _row_label(snapshot_a.get("engage") or {})
    engage_b = _row_label(snapshot_b.get("engage") or {})
    carry_a = _row_label(snapshot_a.get("carry") or {})
    carry_b = _row_label(snapshot_b.get("carry") or {})
    exposed_a = _row_label(snapshot_a.get("exposed") or {})
    exposed_b = _row_label(snapshot_b.get("exposed") or {})
    best_meta_a = dict(meta_a.get("best_meta") or {})
    best_meta_b = dict(meta_b.get("best_meta") or {})
    best_global_a = dict(meta_a.get("best_global_meta") or {})
    best_global_b = dict(meta_b.get("best_global_meta") or {})
    if str(best_meta_a.get("meta_label") or "") == "Meta forte":
        notes.append(f"Meta {team_a}: {_row_label(best_meta_a)} entra forte no recorte atual.")
    if str(best_meta_b.get("meta_label") or "") == "Meta forte":
        notes.append(f"Meta {team_b}: {_row_label(best_meta_b)} entra forte no recorte atual.")
    if str(best_global_a.get("global_meta_label") or "") == "Premium":
        notes.append(f"Meta global {team_a}: {_row_label(best_global_a)} entra como pick premium no recorte.")
    elif str(best_global_a.get("global_meta_label") or "") == "Contestado":
        notes.append(f"Meta global {team_a}: {_row_label(best_global_a)} entra como pick contestado.")
    if str(best_global_b.get("global_meta_label") or "") == "Premium":
        notes.append(f"Meta global {team_b}: {_row_label(best_global_b)} entra como pick premium no recorte.")
    elif str(best_global_b.get("global_meta_label") or "") == "Contestado":
        notes.append(f"Meta global {team_b}: {_row_label(best_global_b)} entra como pick contestado.")
    if carry_a:
        notes.append(f"Condicao de dano {team_a}: {carry_a}.")
    if carry_b:
        notes.append(f"Condicao de dano {team_b}: {carry_b}.")
    if engage_a and exposed_b:
        notes.append(f"Se {engage_a} achar acesso em {exposed_b}, {team_a} tende a ganhar a frente da fight.")
    if engage_b and exposed_a:
        notes.append(f"Se {engage_b} achar acesso em {exposed_a}, {team_b} tende a virar a fight.")
    if isinstance(lane_view, pd.DataFrame) and not lane_view.empty:
        row = lane_view.iloc[(pd.to_numeric(lane_view.get("Edge (p.p.)"), errors="coerce").abs()).idxmax()]
        lane_name = str(row.get("Pos") or "-").upper()
        lane_read = str(row.get("Leitura Lane") or "-")
        edge_value = abs(float(_safe_float(row.get("Edge (p.p.)")) or 0.0))
        notes.append(f"Lane-chave: {lane_name} | {lane_read} ({format_num_fn(edge_value, 1)} p.p.).")
    for reason in list(comp_compare.get("reasons") or []):
        if len(notes) >= 5:
            break
        text = str(reason or "").strip()
        if text and text not in notes:
            notes.append(text)
    return notes[:5]


def _draft_confidence(
    *,
    lane_view: pd.DataFrame,
    duos_a: pd.DataFrame,
    duos_b: pd.DataFrame,
    trios_a: pd.DataFrame,
    trios_b: pd.DataFrame,
    champion_registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lane_score = 0.0
    lane_rows = 0
    if isinstance(lane_view, pd.DataFrame) and not lane_view.empty:
        for _, row in lane_view.iterrows():
            lane_rows += 1
            conf_text = str(row.get("Confianca Lane") or "").strip().casefold()
            if conf_text == "alta":
                lane_score += 1.0
            elif conf_text == "media":
                lane_score += 0.65
            else:
                lane_score += 0.3 if conf_text else 0.1
    lane_ratio = (lane_score / float(lane_rows)) if lane_rows > 0 else 0.0

    combo_frames = [df for df in (duos_a, duos_b, trios_a, trios_b) if isinstance(df, pd.DataFrame) and not df.empty]
    combo_games = []
    for df in combo_frames:
        for _, row in df.head(2).iterrows():
            g = _safe_float(row.get("Games"))
            if g is not None:
                combo_games.append(float(g))
    combo_score = 0.0
    if combo_games:
        combo_score = min(sum(min(g, 8.0) for g in combo_games) / 20.0, 1.0)

    registry_size = len(dict((champion_registry or {}).get("champions") or {}))
    semantic_score = 1.0 if registry_size >= 20 else 0.7 if registry_size >= 10 else 0.4
    score = (0.5 * lane_ratio) + (0.35 * combo_score) + (0.15 * semantic_score)
    return {
        "score": float(score),
        "label": _confidence_rank_from_score(score),
        "lane_ratio": float(lane_ratio),
        "combo_score": float(combo_score),
        "semantic_score": float(semantic_score),
    }


def build_live_bet_draft_analysis(
    *,
    df_scope_players: pd.DataFrame | None,
    df_all_players: pd.DataFrame | None = None,
    df_scope_teams: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    blue_side_team: str | None = None,
    league_context: str | None = None,
    picks_a: list[str],
    picks_b: list[str],
    players_a: list[str],
    players_b: list[str],
    format_num_fn: Callable[[float, int], str],
    champion_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    empty = {
        "lane_view": pd.DataFrame([]),
        "duos_a": pd.DataFrame([]),
        "duos_b": pd.DataFrame([]),
        "trios_a": pd.DataFrame([]),
        "trios_b": pd.DataFrame([]),
        "player_spot_rows_a": [],
        "player_spot_rows_b": [],
        "player_leadership_a": {},
        "player_leadership_b": {},
        "market_calls": {},
        "market_means": {},
        "comp_a": {},
        "comp_b": {},
        "comp_compare": {},
        "draft_confidence": {"score": 0.0, "label": "Baixa"},
        "verdict": {"favored_team": "-", "score_pp": 0.0, "summary": "Sem base suficiente para avaliar o draft.", "reasons": []},
    }
    if df_scope_players is None or df_scope_players.empty or len(picks_a) < 5 or len(picks_b) < 5:
        return empty

    pairs = _build_pairs(df_scope_players)
    names_a = list(players_a[:5]) + [""] * max(0, 5 - len(players_a))
    names_b = list(players_b[:5]) + [""] * max(0, 5 - len(players_b))
    side_a_current = "Blue" if str(blue_side_team or team_a) == str(team_a) else "Red"
    side_b_current = "Blue" if str(blue_side_team or team_a) == str(team_b) else "Red"

    lane_rows: list[dict[str, Any]] = []
    lane_score_pp = 0.0
    strongest_edge = ("-", 0.0)
    for idx, pos in enumerate(ROLE_ORDER):
        champ_a = str(picks_a[idx]) if idx < len(picks_a) else ""
        champ_b = str(picks_b[idx]) if idx < len(picks_b) else ""
        player_a = str(names_a[idx] or "")
        player_b = str(names_b[idx] or "")
        if not champ_a or not champ_b:
            continue
        wr_a = _draft_wr_parts(df_scope_players, champ_a, player_a if player_a else None)
        wr_b = _draft_wr_parts(df_scope_players, champ_b, player_b if player_b else None)
        meta_a = _champ_meta_profile(
            df_scope_players=df_scope_players,
            champ_name=champ_a,
            role=str(pos),
            side_label=side_a_current,
            registry=champion_registry,
        )
        meta_b = _champ_meta_profile(
            df_scope_players=df_scope_players,
            champ_name=champ_b,
            role=str(pos),
            side_label=side_b_current,
            registry=champion_registry,
        )
        full_players_df = df_all_players if isinstance(df_all_players, pd.DataFrame) and not df_all_players.empty else df_scope_players
        wr_a_world = _draft_wr_parts(full_players_df, champ_a, None) if isinstance(full_players_df, pd.DataFrame) and not full_players_df.empty else wr_a
        wr_b_world = _draft_wr_parts(full_players_df, champ_b, None) if isinstance(full_players_df, pd.DataFrame) and not full_players_df.empty else wr_b
        league_alert_a = _league_alert_label(
            current_wr=(100.0 * float(wr_a["p_league"])) if int(wr_a["n_league"]) > 0 else None,
            current_sample=int(wr_a["n_league"]),
            other_wr=(100.0 * float(wr_a_world["p_league"])) if int(wr_a_world["n_league"]) > 0 else None,
            other_sample=int(wr_a_world["n_league"]),
        )
        league_alert_b = _league_alert_label(
            current_wr=(100.0 * float(wr_b["p_league"])) if int(wr_b["n_league"]) > 0 else None,
            current_sample=int(wr_b["n_league"]),
            other_wr=(100.0 * float(wr_b_world["p_league"])) if int(wr_b_world["n_league"]) > 0 else None,
            other_sample=int(wr_b_world["n_league"]),
        )
        h2h_a, h2h_n = _lane_h2h_wr(pairs, champ_a, champ_b, pos)
        h2h_edge = ((float(h2h_a) - 0.5) * 100.0) if h2h_a is not None else 0.0
        blend_edge = (float(wr_a["p_blend"]) - float(wr_b["p_blend"])) * 100.0
        lane_edge = (0.75 * blend_edge) + (0.25 * h2h_edge)
        lane_sem = evaluate_lane_matchup(
            registry=champion_registry,
            role=str(pos),
            champ_a=str(champ_a),
            champ_b=str(champ_b),
            team_a=str(team_a),
            team_b=str(team_b),
            empirical_edge_pp=lane_edge,
            h2h_games=h2h_n,
            player_games_a=int(wr_a["n_player"]),
            player_games_b=int(wr_b["n_player"]),
        )
        lane_score_pp += lane_edge
        if abs(lane_edge) > abs(strongest_edge[1]):
            strongest_edge = (pos, lane_edge)
        lane_rows.append(
            {
                "Pos": pos,
                f"{team_a} Player": player_a or "-",
                f"{team_a} Champ": champ_a,
                f"WR Liga {team_a}": format_num_fn(100.0 * wr_a["p_league"], 1),
                f"WR Player {team_a}": format_num_fn(100.0 * wr_a["p_player"], 1),
                f"N Liga {team_a}": int(wr_a["n_league"]),
                f"N Player {team_a}": int(wr_a["n_player"]),
                f"WR Champ Blue {team_a}": format_num_fn(100.0 * wr_a["p_league_blue"], 1),
                f"WR Champ Red {team_a}": format_num_fn(100.0 * wr_a["p_league_red"], 1),
                f"N Champ Blue {team_a}": int(wr_a["n_league_blue"]),
                f"N Champ Red {team_a}": int(wr_a["n_league_red"]),
                f"WR Player Blue {team_a}": format_num_fn(100.0 * wr_a["p_player_blue"], 1),
                f"WR Player Red {team_a}": format_num_fn(100.0 * wr_a["p_player_red"], 1),
                f"N Player Blue {team_a}": int(wr_a["n_player_blue"]),
                f"N Player Red {team_a}": int(wr_a["n_player_red"]),
                f"Meta {team_a}": str(meta_a.get("label") or "Meta neutro"),
                f"Meta Score {team_a}": format_num_fn(float(_safe_float(meta_a.get("score")) or 50.0), 1),
                f"Meta Sample {team_a}": int(_safe_float(meta_a.get("sample")) or 0),
                f"Meta Global {team_a}": str(meta_a.get("global_label") or "Situacional"),
                f"Meta Global Score {team_a}": format_num_fn(float(_safe_float(meta_a.get("global_score")) or 50.0), 1),
                f"WR Champ Mundo Blue {team_a}": format_num_fn(100.0 * wr_a_world["p_league_blue"], 1),
                f"WR Champ Mundo Red {team_a}": format_num_fn(100.0 * wr_a_world["p_league_red"], 1),
                f"N Champ Mundo Blue {team_a}": int(wr_a_world["n_league_blue"]),
                f"N Champ Mundo Red {team_a}": int(wr_a_world["n_league_red"]),
                f"Alerta Liga {team_a}": league_alert_a,
                f"{team_b} Player": player_b or "-",
                f"{team_b} Champ": champ_b,
                f"WR Liga {team_b}": format_num_fn(100.0 * wr_b["p_league"], 1),
                f"WR Player {team_b}": format_num_fn(100.0 * wr_b["p_player"], 1),
                f"N Liga {team_b}": int(wr_b["n_league"]),
                f"N Player {team_b}": int(wr_b["n_player"]),
                f"WR Champ Blue {team_b}": format_num_fn(100.0 * wr_b["p_league_blue"], 1),
                f"WR Champ Red {team_b}": format_num_fn(100.0 * wr_b["p_league_red"], 1),
                f"N Champ Blue {team_b}": int(wr_b["n_league_blue"]),
                f"N Champ Red {team_b}": int(wr_b["n_league_red"]),
                f"WR Player Blue {team_b}": format_num_fn(100.0 * wr_b["p_player_blue"], 1),
                f"WR Player Red {team_b}": format_num_fn(100.0 * wr_b["p_player_red"], 1),
                f"N Player Blue {team_b}": int(wr_b["n_player_blue"]),
                f"N Player Red {team_b}": int(wr_b["n_player_red"]),
                f"Meta {team_b}": str(meta_b.get("label") or "Meta neutro"),
                f"Meta Score {team_b}": format_num_fn(float(_safe_float(meta_b.get("score")) or 50.0), 1),
                f"Meta Sample {team_b}": int(_safe_float(meta_b.get("sample")) or 0),
                f"Meta Global {team_b}": str(meta_b.get("global_label") or "Situacional"),
                f"Meta Global Score {team_b}": format_num_fn(float(_safe_float(meta_b.get("global_score")) or 50.0), 1),
                f"WR Champ Mundo Blue {team_b}": format_num_fn(100.0 * wr_b_world["p_league_blue"], 1),
                f"WR Champ Mundo Red {team_b}": format_num_fn(100.0 * wr_b_world["p_league_red"], 1),
                f"N Champ Mundo Blue {team_b}": int(wr_b_world["n_league_blue"]),
                f"N Champ Mundo Red {team_b}": int(wr_b_world["n_league_red"]),
                f"Alerta Liga {team_b}": league_alert_b,
                "WR H2H Lane": format_num_fn(100.0 * float(h2h_a), 1) if h2h_a is not None else "-",
                "N H2H": int(h2h_n),
                "Edge (p.p.)": format_num_fn(lane_edge, 1),
                "Leitura Lane": str(lane_sem.get("summary") or "-"),
                "Early": str(lane_sem.get("early_team") or "-"),
                "Late": str(lane_sem.get("late_team") or "-"),
                "Confianca Lane": str(lane_sem.get("confidence") or "-"),
                "Counter": str(lane_sem.get("counter_team") or "-"),
                f"Plano {team_a}": str(lane_sem.get("team_a_plan") or "-"),
                f"Plano {team_b}": str(lane_sem.get("team_b_plan") or "-"),
            }
        )

    duos_a = _build_combo_summary(df_scope_players=df_scope_players, df_scope_teams=df_scope_teams, team_name=str(team_a), picks=picks_a, size=2, format_num_fn=format_num_fn)
    duos_b = _build_combo_summary(df_scope_players=df_scope_players, df_scope_teams=df_scope_teams, team_name=str(team_b), picks=picks_b, size=2, format_num_fn=format_num_fn)
    trios_a = _build_combo_summary(df_scope_players=df_scope_players, df_scope_teams=df_scope_teams, team_name=str(team_a), picks=picks_a, size=3, format_num_fn=format_num_fn)
    trios_b = _build_combo_summary(df_scope_players=df_scope_players, df_scope_teams=df_scope_teams, team_name=str(team_b), picks=picks_b, size=3, format_num_fn=format_num_fn)
    player_spot_rows_a = _build_player_spot_rows(
        df_scope_players=df_scope_players,
        team_name=str(team_a),
        picks=picks_a,
        players=players_a,
        side_label=side_a_current,
        registry=champion_registry,
    )
    player_spot_rows_b = _build_player_spot_rows(
        df_scope_players=df_scope_players,
        team_name=str(team_b),
        picks=picks_b,
        players=players_b,
        side_label=side_b_current,
        registry=champion_registry,
    )
    player_leadership_a = _team_player_leadership(player_spot_rows_a)
    player_leadership_b = _team_player_leadership(player_spot_rows_b)
    meta_leadership_a = _team_meta_leadership(player_spot_rows_a)
    meta_leadership_b = _team_meta_leadership(player_spot_rows_b)
    comp_a = build_comp_profile(registry=champion_registry, picks=picks_a, players=players_a, role_order=ROLE_ORDER)
    comp_b = build_comp_profile(registry=champion_registry, picks=picks_b, players=players_b, role_order=ROLE_ORDER)
    comp_compare = compare_comp_profiles(comp_a=comp_a, comp_b=comp_b, team_a=str(team_a), team_b=str(team_b))

    synergy_edge_pp = 0.5 * (_weighted_combo_edge(duos_a) - _weighted_combo_edge(duos_b)) + 0.5 * (_weighted_combo_edge(trios_a) - _weighted_combo_edge(trios_b))
    comp_edge_pp = (
        0.85 * (_safe_float(comp_compare.get("early_delta")) or 0.0)
        + 0.55 * (_safe_float(comp_compare.get("late_delta")) or 0.0)
        + 0.45 * (_safe_float(comp_compare.get("objective_delta")) or 0.0)
        + 0.35 * (_safe_float(comp_compare.get("towers_delta")) or 0.0)
    )
    total_score_pp = float(lane_score_pp) + float(synergy_edge_pp) + float(comp_edge_pp)
    if total_score_pp > 1.5:
        favored = str(team_a)
    elif total_score_pp < -1.5:
        favored = str(team_b)
    else:
        favored = "Equilibrado"
    reasons: list[str] = []
    if strongest_edge[0] != "-":
        direction = team_a if strongest_edge[1] >= 0 else team_b
        reasons.append(f"Maior edge de lane: {direction} em {strongest_edge[0]} ({format_num_fn(abs(strongest_edge[1]), 1)} p.p.).")
    if abs(synergy_edge_pp) >= 0.75:
        direction = team_a if synergy_edge_pp >= 0 else team_b
        reasons.append(f"Sinergia de duos/trios favorece {direction} ({format_num_fn(abs(synergy_edge_pp), 1)} p.p.).")
    for reason in list(comp_compare.get("reasons") or []):
        if reason not in reasons:
            reasons.append(str(reason))
    if not reasons:
        reasons.append("Draft sem edge estrutural claro pela base atual.")
    summary = (
        f"Melhor draft: {favored} ({format_num_fn(abs(total_score_pp), 1)} p.p.)."
        if favored != "Equilibrado"
        else f"Draft equilibrado ({format_num_fn(abs(total_score_pp), 1)} p.p. de diferenca)."
    )

    scope_kills_mean = _safe_float(pd.to_numeric(df_scope_teams.get("total_kills"), errors="coerce").mean()) if isinstance(df_scope_teams, pd.DataFrame) and "total_kills" in df_scope_teams.columns else None
    scope_towers_mean = _safe_float(pd.to_numeric(df_scope_teams.get("total_towers"), errors="coerce").mean()) if isinstance(df_scope_teams, pd.DataFrame) and "total_towers" in df_scope_teams.columns else None
    scope_dragons_mean = _safe_float(pd.to_numeric(df_scope_teams.get("total_dragons"), errors="coerce").mean()) if isinstance(df_scope_teams, pd.DataFrame) and "total_dragons" in df_scope_teams.columns else None
    scope_barons_mean = _safe_float(pd.to_numeric(df_scope_teams.get("total_nashors"), errors="coerce").mean()) if isinstance(df_scope_teams, pd.DataFrame) and "total_nashors" in df_scope_teams.columns else None
    scope_time_mean = _safe_float(pd.to_numeric(df_scope_teams.get("game_time_min"), errors="coerce").mean()) if isinstance(df_scope_teams, pd.DataFrame) and "game_time_min" in df_scope_teams.columns else None

    mean_kills_a = _combo_metric_mean(duos_a, "Avg Kills") or _combo_metric_mean(trios_a, "Avg Kills")
    mean_kills_b = _combo_metric_mean(duos_b, "Avg Kills") or _combo_metric_mean(trios_b, "Avg Kills")
    mean_towers_a = _combo_metric_mean(duos_a, "Avg Towers") or _combo_metric_mean(trios_a, "Avg Towers")
    mean_towers_b = _combo_metric_mean(duos_b, "Avg Towers") or _combo_metric_mean(trios_b, "Avg Towers")
    mean_dragons_a = _combo_metric_mean(duos_a, "Avg Dragons") or _combo_metric_mean(trios_a, "Avg Dragons")
    mean_dragons_b = _combo_metric_mean(duos_b, "Avg Dragons") or _combo_metric_mean(trios_b, "Avg Dragons")
    mean_barons_a = _combo_metric_mean(duos_a, "Avg Barons") or _combo_metric_mean(trios_a, "Avg Barons")
    mean_barons_b = _combo_metric_mean(duos_b, "Avg Barons") or _combo_metric_mean(trios_b, "Avg Barons")
    mean_time_a = _combo_metric_mean(duos_a, "Avg Time") or _combo_metric_mean(trios_a, "Avg Time")
    mean_time_b = _combo_metric_mean(duos_b, "Avg Time") or _combo_metric_mean(trios_b, "Avg Time")

    market_calls = {
        "Kills": _market_call_from_combo_means(kind="kills", mean_a=mean_kills_a, mean_b=mean_kills_b, scope_mean=scope_kills_mean),
        "Torres": _market_call_from_combo_means(kind="towers", mean_a=mean_towers_a, mean_b=mean_towers_b, scope_mean=scope_towers_mean),
        "DragÃµes": _market_call_from_combo_means(kind="dragons", mean_a=mean_dragons_a, mean_b=mean_dragons_b, scope_mean=scope_dragons_mean),
        "BarÃµes": _market_call_from_combo_means(kind="barons", mean_a=mean_barons_a, mean_b=mean_barons_b, scope_mean=scope_barons_mean),
        "Tempo": _market_call_from_combo_means(kind="time", mean_a=mean_time_a, mean_b=mean_time_b, scope_mean=scope_time_mean),
    }
    market_means = {
        "Kills": {"team_a": mean_kills_a, "team_b": mean_kills_b, "scope": scope_kills_mean},
        "Torres": {"team_a": mean_towers_a, "team_b": mean_towers_b, "scope": scope_towers_mean},
        "DragÃµes": {"team_a": mean_dragons_a, "team_b": mean_dragons_b, "scope": scope_dragons_mean},
        "BarÃµes": {"team_a": mean_barons_a, "team_b": mean_barons_b, "scope": scope_barons_mean},
        "Tempo": {"team_a": mean_time_a, "team_b": mean_time_b, "scope": scope_time_mean},
    }
    if market_calls.get("Kills") in {"Sem base", "Neutro"}:
        market_calls["Kills"] = str(comp_compare.get("kills_call") or market_calls.get("Kills") or "Neutro")
    if market_calls.get("Torres") in {"Sem base", "Neutro"}:
        market_calls["Torres"] = str(comp_compare.get("towers_call") or market_calls.get("Torres") or "Neutro")
    if market_calls.get("DragÃµes") in {"Sem base", "Neutro"}:
        market_calls["DragÃµes"] = str(comp_compare.get("dragons_call") or market_calls.get("DragÃµes") or "Neutro")
    if market_calls.get("BarÃµes") in {"Sem base", "Neutro"}:
        market_calls["BarÃµes"] = str(comp_compare.get("barons_call") or market_calls.get("BarÃµes") or "Neutro")
    if market_calls.get("Tempo") in {"Sem base", "Neutro"}:
        market_calls["Tempo"] = str(comp_compare.get("time_call") or market_calls.get("Tempo") or "Neutro")
    combo_support_a = _combo_support_summary(
        team_name=str(team_a),
        duos_df=duos_a if isinstance(duos_a, pd.DataFrame) else pd.DataFrame([]),
        trios_df=trios_a if isinstance(trios_a, pd.DataFrame) else pd.DataFrame([]),
        market_means=market_means,
        market_calls=market_calls,
        comp_compare=comp_compare,
        format_num_fn=format_num_fn,
    )
    combo_support_b = _combo_support_summary(
        team_name=str(team_b),
        duos_df=duos_b if isinstance(duos_b, pd.DataFrame) else pd.DataFrame([]),
        trios_df=trios_b if isinstance(trios_b, pd.DataFrame) else pd.DataFrame([]),
        market_means=market_means,
        market_calls=market_calls,
        comp_compare=comp_compare,
        format_num_fn=format_num_fn,
    )


    lane_view_df = pd.DataFrame(lane_rows)
    draft_confidence = _draft_confidence(
        lane_view=lane_view_df,
        duos_a=duos_a,
        duos_b=duos_b,
        trios_a=trios_a,
        trios_b=trios_b,
        champion_registry=champion_registry,
    )

    return {
        "lane_view": lane_view_df,
        "duos_a": duos_a.drop(columns=["_wr", "_games"], errors="ignore"),
        "duos_b": duos_b.drop(columns=["_wr", "_games"], errors="ignore"),
        "trios_a": trios_a.drop(columns=["_wr", "_games"], errors="ignore"),
        "trios_b": trios_b.drop(columns=["_wr", "_games"], errors="ignore"),
        "player_spot_rows_a": player_spot_rows_a,
        "player_spot_rows_b": player_spot_rows_b,
        "player_leadership_a": player_leadership_a,
        "player_leadership_b": player_leadership_b,
        "meta_leadership_a": meta_leadership_a,
        "meta_leadership_b": meta_leadership_b,
        "market_calls": market_calls,
        "market_means": market_means,
        "combo_support_a": combo_support_a,
        "combo_support_b": combo_support_b,
        "comp_a": comp_a,
        "comp_b": comp_b,
        "comp_compare": comp_compare,
        "draft_confidence": draft_confidence,
        "verdict": {"favored_team": favored, "score_pp": total_score_pp, "summary": summary, "reasons": reasons},
    }


__all__ = [
    "_build_combo_summary",
    "_build_lane_cards",
    "_build_pairs",
    "_build_player_spot_rows",
    "_champ_meta_profile",
    "_draft_wr_parts",
    "_edge_share_text",
    "_label_player_meta_row",
    "_label_player_spot_row",
    "_lane_edge_text",
    "_lane_h2h_wr",
    "_league_alert_label",
    "_player_role_function",
    "_resolve_pick_names",
    "_team_meta_leadership",
    "_team_player_leadership",
    "_team_role_snapshot",
    "_weighted_combo_edge",
    "_wr_dual_html",
    "build_live_bet_draft_analysis",
    "build_live_bet_handicap_summary",
    "build_live_bet_player_prop_summary",
    "build_live_bet_tactical_notes",
    "resolve_compact_draft_sides",
]
