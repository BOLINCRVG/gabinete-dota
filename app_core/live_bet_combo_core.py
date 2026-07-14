"""Champion combo analysis helpers for live-bet draft analysis.

Extracted from live_bet_draft_core.py. All functions are stateless.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from mlcore.utils import safe_float_or_none as _safe_float


def _build_combo_summary(
    *,
    df_scope_players: pd.DataFrame,
    df_scope_teams: pd.DataFrame,
    team_name: str,
    picks: list[str],
    size: int,
    format_num_fn: Callable[[float, int], str],
) -> pd.DataFrame:
    """Build a DataFrame of champion combo stats for all size-combinations from picks list."""
    if size <= 1 or len(picks) < size or df_scope_players is None or df_scope_players.empty:
        return pd.DataFrame([])
    req = {"gameid", "team", "champ", "result"}
    if not req.issubset(df_scope_players.columns):
        return pd.DataFrame([])
    pl = df_scope_players[df_scope_players["team"].astype(str) == str(team_name)].copy()
    if pl.empty:
        return pd.DataFrame([])
    grouped = (
        pl.groupby(["gameid", "team"], dropna=False)
        .agg(
            champs=("champ", lambda s: tuple(sorted([str(v) for v in s.dropna().astype(str).tolist() if str(v)]))),
            win=("result", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).iloc[0]) if len(s) > 0 else 0.0),
        )
        .reset_index()
    )
    team_metrics = pd.DataFrame([])
    if df_scope_teams is not None and not df_scope_teams.empty and {"gameid", "team"}.issubset(df_scope_teams.columns):
        keep_cols = [
            col
            for col in ["gameid", "team", "total_kills", "total_towers", "total_dragons", "total_nashors", "game_time_min"]
            if col in df_scope_teams.columns
        ]
        team_metrics = df_scope_teams[keep_cols].drop_duplicates(subset=["gameid", "team"]).copy()
    if not team_metrics.empty:
        grouped = grouped.merge(team_metrics, on=["gameid", "team"], how="left")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    min_games = 5
    for combo in itertools.combinations([str(p) for p in picks if str(p).strip()], size):
        combo_key = tuple(sorted(combo))
        if combo_key in seen:
            continue
        seen.add(combo_key)
        subset = grouped[grouped["champs"].map(lambda values: set(combo_key).issubset(set(values or ())))].copy()
        if subset.empty:
            continue
        games = int(subset.shape[0])
        wins = float(pd.to_numeric(subset["win"], errors="coerce").fillna(0).sum()) if games > 0 else 0.0
        win_rate = float(wins / float(games)) if games > 0 else 0.0
        eligible = bool(games >= min_games or (games >= 4 and wins >= games))
        if not eligible:
            continue
        score_wr = float((wins + 1.5) / (games + 3.0)) if games > 0 else 0.5
        avg_kills = _safe_float(pd.to_numeric(subset.get("total_kills"), errors="coerce").mean()) if "total_kills" in subset.columns else None
        avg_towers = _safe_float(pd.to_numeric(subset.get("total_towers"), errors="coerce").mean()) if "total_towers" in subset.columns else None
        avg_dragons = _safe_float(pd.to_numeric(subset.get("total_dragons"), errors="coerce").mean()) if "total_dragons" in subset.columns else None
        avg_barons = _safe_float(pd.to_numeric(subset.get("total_nashors"), errors="coerce").mean()) if "total_nashors" in subset.columns else None
        avg_time = _safe_float(pd.to_numeric(subset.get("game_time_min"), errors="coerce").mean()) if "game_time_min" in subset.columns else None
        confidence = "Alta" if games >= 10 else ("Media" if games >= 6 else "Excecao")
        rows.append(
            {
                "Combo": " + ".join(combo_key),
                "Games": games,
                "Win Rate": format_num_fn(100.0 * win_rate, 1),
                "WR Score": format_num_fn(100.0 * score_wr, 1),
                "Confianca": confidence,
                "Avg Kills": format_num_fn(float(avg_kills), 1) if avg_kills is not None else "-",
                "Avg Towers": format_num_fn(float(avg_towers), 1) if avg_towers is not None else "-",
                "Avg Dragons": format_num_fn(float(avg_dragons), 1) if avg_dragons is not None else "-",
                "Avg Barons": format_num_fn(float(avg_barons), 1) if avg_barons is not None else "-",
                "Avg Time": format_num_fn(float(avg_time), 1) if avg_time is not None else "-",
                "_wr": win_rate,
                "_score_wr": score_wr,
                "_games": games,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["_score_wr", "_games", "Combo"], ascending=[False, False, True]).reset_index(drop=True)
    return out


def _weighted_combo_edge(df_combo: pd.DataFrame) -> float:
    """Compute a weighted win-rate edge (vs 50%) across all combos, weighted by sample size."""
    if df_combo is None or df_combo.empty:
        return 0.0
    num = 0.0
    den = 0.0
    for _, row in df_combo.iterrows():
        wr = _safe_float(row.get("_score_wr"))
        games = _safe_float(row.get("_games"))
        if wr is None or games is None or games <= 0:
            continue
        num += (float(wr) - 0.5) * float(games)
        den += float(games)
    if den <= 0:
        return 0.0
    return 100.0 * (num / den)


def _combo_metric_mean(df_combo: pd.DataFrame, column: str) -> Optional[float]:
    """Return the average of a metric column across the top 2 combos in the DataFrame."""
    if df_combo is None or df_combo.empty or column not in df_combo.columns:
        return None
    values: list[float] = []
    for _, row in df_combo.head(2).iterrows():
        value = _safe_float(row.get(column))
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _market_call_from_combo_means(
    *,
    kind: str,
    mean_a: Optional[float],
    mean_b: Optional[float],
    scope_mean: Optional[float],
) -> str:
    """Return a market signal string (e.g. 'Favorece over') based on combo metric averages vs scope mean."""
    if mean_a is None or mean_b is None or scope_mean is None:
        return "Sem base"
    avg_value = (float(mean_a) + float(mean_b)) / 2.0
    if kind == "kills":
        if mean_a >= scope_mean + 1.5 and mean_b >= scope_mean + 1.5:
            return "Favorece over"
        if mean_a <= scope_mean - 1.5 and mean_b <= scope_mean - 1.5:
            return "Favorece under"
        if avg_value >= scope_mean + 1.0:
            return "Vies over"
        if avg_value <= scope_mean - 1.0:
            return "Vies under"
        return "Neutro"
    if kind == "towers":
        if mean_a >= scope_mean + 0.6 and mean_b >= scope_mean + 0.6:
            return "Mais torres"
        if mean_a <= scope_mean - 0.6 and mean_b <= scope_mean - 0.6:
            return "Menos torres"
        if avg_value >= scope_mean + 0.4:
            return "Vies mais torres"
        if avg_value <= scope_mean - 0.4:
            return "Vies menos torres"
        return "Neutro"
    if kind == "dragons":
        if mean_a >= scope_mean + 0.35 and mean_b >= scope_mean + 0.35:
            return "Mais dragoes"
        if mean_a <= scope_mean - 0.35 and mean_b <= scope_mean - 0.35:
            return "Menos dragoes"
        if avg_value >= scope_mean + 0.2:
            return "Vies mais dragoes"
        if avg_value <= scope_mean - 0.2:
            return "Vies menos dragoes"
        return "Neutro"
    if kind == "barons":
        if mean_a >= scope_mean + 0.2 and mean_b >= scope_mean + 0.2:
            return "Mais baroes"
        if mean_a <= scope_mean - 0.2 and mean_b <= scope_mean - 0.2:
            return "Menos baroes"
        if avg_value >= scope_mean + 0.1:
            return "Vies mais baroes"
        if avg_value <= scope_mean - 0.1:
            return "Vies menos baroes"
        return "Neutro"
    if kind == "time":
        if mean_a <= scope_mean - 0.8 and mean_b <= scope_mean - 0.8:
            return "Encurta"
        if mean_a >= scope_mean + 0.8 and mean_b >= scope_mean + 0.8:
            return "Alonga"
        if avg_value <= scope_mean - 0.5:
            return "Vies curto"
        if avg_value >= scope_mean + 0.5:
            return "Vies longo"
        return "Neutro"
    return "Neutro"


def _combo_market_supports(
    row: Mapping[str, Any],
    market_means: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, str, float]]:
    """Return list of (market, direction, delta) tuples where combo stats deviate from scope mean."""
    defs = (
        ("Kills", "Avg Kills", 2.0, "Mais kills", "Menos kills"),
        ("Torres", "Avg Towers", 0.8, "Mais torres", "Menos torres"),
        ("Dragões", "Avg Dragons", 0.45, "Mais dragoes", "Menos dragoes"),
        ("Barões", "Avg Barons", 0.22, "Mais baroes", "Menos baroes"),
        ("Tempo", "Avg Time", 1.6, "Mais longo", "Mais curto"),
    )
    out: list[tuple[str, str, float]] = []
    for market_name, col_name, threshold, high_label, low_label in defs:
        avg_value = _safe_float(row.get(col_name))
        scope_mean = _safe_float((market_means.get(market_name) or {}).get("scope"))
        if avg_value is None or scope_mean is None:
            continue
        diff = float(avg_value) - float(scope_mean)
        if abs(diff) < float(threshold):
            continue
        out.append((market_name, high_label if diff > 0 else low_label, abs(diff)))
    out.sort(key=lambda item: float(item[2]), reverse=True)
    return out


def _combo_alignment_bonus(
    support_rows: list[tuple[str, str, float]],
    *,
    market_calls: Mapping[str, Any],
    comp_compare: Mapping[str, Any],
) -> float:
    """Score bonus when combo market signals align with the current market call direction."""
    bonus = 0.0
    semantic_map = {
        "Kills": str(market_calls.get("Kills") or comp_compare.get("kills_call") or ""),
        "Torres": str(market_calls.get("Torres") or comp_compare.get("towers_call") or ""),
        "Dragões": str(market_calls.get("Dragões") or comp_compare.get("dragons_call") or ""),
        "Barões": str(market_calls.get("Barões") or comp_compare.get("barons_call") or ""),
        "Tempo": str(market_calls.get("Tempo") or comp_compare.get("time_call") or ""),
    }
    for market_name, support_text, _ in support_rows:
        semantic = semantic_map.get(str(market_name), "").casefold()
        support_norm = str(support_text).casefold()
        if "mais" in support_norm and any(token in semantic for token in ("over", "mais", "alonga", "setup", "pressiona", "acelera", "joga para", "joga cedo")):
            bonus += 0.8
        elif "menos" in support_norm and any(token in semantic for token in ("under", "menos", "encurta", "neutro")):
            bonus += 0.8
    return bonus


def _combo_support_summary(
    *,
    team_name: str,
    duos_df: pd.DataFrame,
    trios_df: pd.DataFrame,
    market_means: Mapping[str, Mapping[str, Any]],
    market_calls: Mapping[str, Any],
    comp_compare: Mapping[str, Any],
    format_num_fn: Callable[[float, int], str],
) -> dict[str, Any]:
    """Find the best combo (duo or trio) that supports a market call and return a summary dict."""
    candidates: list[dict[str, Any]] = []
    for df_combo, combo_kind in ((duos_df, "Duo"), (trios_df, "Trio")):
        if df_combo is None or df_combo.empty:
            continue
        for _, row in df_combo.head(3).iterrows():
            games = int(_safe_float(row.get("Games")) or 0)
            score_wr = float(_safe_float(row.get("_score_wr")) or 0.0)
            if games < 6 and not (games >= 4 and score_wr >= 0.75):
                continue
            support_rows = _combo_market_supports(row, market_means)
            if not support_rows and score_wr < 0.60:
                continue
            support_score = max(0.0, (score_wr - 0.50) * 100.0) + min(float(games), 12.0) * 0.9
            support_score += _combo_alignment_bonus(support_rows, market_calls=market_calls, comp_compare=comp_compare)
            support_score += min(len(support_rows), 2) * 1.1
            candidates.append(
                {
                    "team": str(team_name),
                    "kind": combo_kind,
                    "combo": str(row.get("Combo") or "-"),
                    "games": games,
                    "score_wr": float(score_wr),
                    "wr": str(row.get("Win Rate") or "-"),
                    "confidence": str(row.get("Confianca") or "-"),
                    "supports": support_rows[:2],
                    "support_score": support_score,
                }
            )
    if not candidates:
        return {}
    best = max(candidates, key=lambda item: float(item.get("support_score") or 0.0))
    supports = list(best.get("supports") or [])
    if supports:
        support_text = " + ".join(str(item[1]) for item in supports)
        reading = f"{best['kind']} {best['combo']} reforca {support_text.lower()}."
    else:
        reading = f"{best['kind']} {best['combo']} tem sinergia boa, mas sem mercado claro forte."
    return {
        "team": str(best.get("team") or team_name),
        "kind": str(best.get("kind") or "-"),
        "combo": str(best.get("combo") or "-"),
        "games": int(best.get("games") or 0),
        "score_wr": format_num_fn(100.0 * float(best.get("score_wr") or 0.0), 1),
        "confidence": str(best.get("confidence") or "-"),
        "supports": [str(item[1]) for item in supports],
        "reading": reading,
    }
