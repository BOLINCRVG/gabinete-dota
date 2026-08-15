from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import pandas as pd

from app_core.championship_stats_core import build_championship_stats_table, build_stats_table_view
from app_core.opponent_profile_core import build_opponent_reference_table
from app_core.live_bet_shared_core import (
    LIVE_MARKET_LABELS,
    VALUE_MARKET_META,
    _confidence_label,
    _sample_band,
    _profile_strength_text,
    _resolution_label,
    _value_confidence_label,
    _confidence_rank_from_score,
    _metric_team_mean,
    _weighted_value,
)
from mlcore.utils import safe_float_or_none as _safe_float


def _team_stats_lookup(df_scope: pd.DataFrame) -> pd.DataFrame:
    df_raw = build_championship_stats_table(df_scope, threshold_lines_by_metric={})
    if df_raw is None or df_raw.empty:
        return pd.DataFrame([])
    return df_raw.copy()


def build_live_bet_pregame_summary(
    *,
    df_scope: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
) -> dict[str, Any]:
    empty = {
        "stats_view": pd.DataFrame([]),
        "profile_view": pd.DataFrame([]),
        "profile_cards_a": [],
        "profile_cards_b": [],
        "profile_matrix": pd.DataFrame([]),
        "profile_lookup": {},
        "summary_line": "",
        "quick_cards": [],
    }
    if df_scope is None or df_scope.empty:
        return empty

    df_stats_raw = _team_stats_lookup(df_scope)
    if df_stats_raw.empty:
        return empty

    df_stats_view = build_stats_table_view(
        df_stats_raw,
        min_to_mmss_fn=min_to_mmss_fn,
        format_num_fn=format_num_fn,
        priority_teams=[str(team_a), str(team_b)],
    )
    df_stats_view = df_stats_view[df_stats_view["Team"].astype(str).isin([str(team_a), str(team_b)])].copy()

    profile_rows: list[dict[str, Any]] = []
    profile_map: dict[str, dict[str, str]] = {str(team_a): {}, str(team_b): {}}
    for market_label in LIVE_MARKET_LABELS:
        ref = build_opponent_reference_table(df_scope, market_label)
        if ref is None or ref.empty:
            continue
        league_mean = _safe_float(pd.to_numeric(ref.get("Metric Mean"), errors="coerce").mean())
        for team_name in (str(team_a), str(team_b)):
            row = ref[ref["Team"].astype(str) == team_name].head(1).copy()
            if row.empty:
                continue
            bucket = str(row.iloc[0].get("Bucket") or "-")
            metric_mean = _safe_float(row.iloc[0].get("Metric Mean"))
            metric_z = _safe_float(row.iloc[0].get("Metric Z"))
            games = int(_safe_float(row.iloc[0].get("Games")) or 0)
            delta_vs_scope = (float(metric_mean) - float(league_mean)) if metric_mean is not None and league_mean is not None else None
            confidence_label = _confidence_label(metric_z, games)
            sample_label = _sample_band(games)
            strength_text = _profile_strength_text(bucket)
            profile_rows.append(
                {
                    "Mercado": market_label,
                    "Team": team_name,
                    "Perfil": bucket,
                    "Perfil Leitura": strength_text,
                    "Jogos": games,
                    "Amostra": sample_label,
                    "Confianca": confidence_label,
                    "Media": metric_mean,
                    "Z": metric_z,
                    "Delta vs Escopo": delta_vs_scope,
                }
            )
            profile_map[team_name][market_label] = {
                "bucket": bucket,
                "strength": strength_text,
                "games": games,
                "sample": sample_label,
                "confidence": confidence_label,
                "z": metric_z,
                "mean": metric_mean,
                "delta_scope": delta_vs_scope,
            }

    profile_view = pd.DataFrame(profile_rows)
    if not profile_view.empty:
        for col in ("Media", "Z", "Delta vs Escopo"):
            if col in profile_view.columns:
                profile_view[col] = profile_view[col].map(
                    lambda value: format_num_fn(float(value), 2) if value is not None and pd.notna(value) else "-"
                )
    cards_a: list[dict[str, str]] = []
    cards_b: list[dict[str, str]] = []
    profile_matrix_rows: list[dict[str, str]] = []
    for market_label in LIVE_MARKET_LABELS:
        info_a = dict(profile_map.get(str(team_a), {}).get(market_label) or {})
        info_b = dict(profile_map.get(str(team_b), {}).get(market_label) or {})
        resolution = _resolution_label(str(info_a.get("bucket") or "-"), str(info_b.get("bucket") or "-"))
        cards_a.append(
            {
                "Mercado": market_label,
                "Resolucao": resolution,
                "Perfil": str(info_a.get("strength") or "-"),
                "Amostra": str(info_a.get("sample") or "-"),
                "Confianca": str(info_a.get("confidence") or "-"),
            }
        )
        cards_b.append(
            {
                "Mercado": market_label,
                "Resolucao": resolution,
                "Perfil": str(info_b.get("strength") or "-"),
                "Amostra": str(info_b.get("sample") or "-"),
                "Confianca": str(info_b.get("confidence") or "-"),
            }
        )
        profile_matrix_rows.append(
            {
                "Mercado": market_label,
                "Resolucao": resolution,
                str(team_a): f"{info_a.get('strength') or '-'!s} | {info_a.get('sample') or '-'!s} | {info_a.get('confidence') or '-'!s}",
                str(team_b): f"{info_b.get('strength') or '-'!s} | {info_b.get('sample') or '-'!s} | {info_b.get('confidence') or '-'!s}",
            }
        )
    kills_info_a = dict(profile_map.get(str(team_a), {}).get("Kills") or {})
    kills_info_b = dict(profile_map.get(str(team_b), {}).get("Kills") or {})
    kills_a = str(kills_info_a.get("bucket") or "-")

    towers_a = str((profile_map.get(str(team_a), {}).get("Torres") or {}).get("bucket") or "-")
    towers_b = str((profile_map.get(str(team_b), {}).get("Torres") or {}).get("bucket") or "-")

    time_a = str((profile_map.get(str(team_a), {}).get("Tempo") or {}).get("bucket") or "-")

    summary_line = f"Pre-jogo: kills {kills_a.lower() if kills_a != '-' else 'sem base'}, torres {towers_a.lower() if towers_a != '-' else 'sem base'} / {towers_b.lower() if towers_b != '-' else 'sem base'}, tempo {time_a.lower() if time_a != '-' else 'sem base'}."
    quick_cards = [
        {
            "Titulo": "Ritmo",
            "Valor": f"{team_a}: {kills_info_a.get('strength') or '-' !s} | {team_b}: {kills_info_b.get('strength') or '-'!s}",
            "Detalhe": f"Amostra {kills_info_a.get('sample', '-')} / {kills_info_b.get('sample', '-')}",
        },
        {
            "Titulo": "Torres",
            "Valor": f"{team_a}: {(profile_map.get(str(team_a), {}).get('Torres') or {}).get('strength') or '-' !s} | {team_b}: {(profile_map.get(str(team_b), {}).get('Torres') or {}).get('strength') or '-'!s}",
            "Detalhe": f"Confianca {(profile_map.get(str(team_a), {}).get('Torres') or {}).get('confidence') or '-' !s} / {(profile_map.get(str(team_b), {}).get('Torres') or {}).get('confidence') or '-'!s}",
        },
        {
            "Titulo": "Tempo",
            "Valor": f"{team_a}: {(profile_map.get(str(team_a), {}).get('Tempo') or {}).get('strength') or '-' !s} | {team_b}: {(profile_map.get(str(team_b), {}).get('Tempo') or {}).get('strength') or '-'!s}",
            "Detalhe": f"Confianca {(profile_map.get(str(team_a), {}).get('Tempo') or {}).get('confidence') or '-' !s} / {(profile_map.get(str(team_b), {}).get('Tempo') or {}).get('confidence') or '-'!s}",
        },
        {
            "Titulo": "Confianca",
            "Valor": f"{team_a}: {kills_info_a.get('confidence') or '-' !s} | {team_b}: {kills_info_b.get('confidence') or '-'!s}",
            "Detalhe": "Leitura de kills no recorte ativo.",
        },
    ]

    return {
        "stats_view": df_stats_view,
        "profile_view": profile_view,
        "profile_cards_a": cards_a,
        "profile_cards_b": cards_b,
        "profile_matrix": pd.DataFrame(profile_matrix_rows),
        "profile_lookup": profile_map,
        "summary_line": summary_line,
        "quick_cards": quick_cards,
    }






def _profile_bucket_score(bucket: str) -> float:
    mapping = {
        "Muito Under": -1.0,
        "Under": -0.5,
        "Neutro": 0.0,
        "Over": 0.5,
        "Muito Over": 1.0,
    }
    return float(mapping.get(str(bucket or "").strip(), 0.0))


def _draft_call_score(call: str) -> float:
    value = str(call or "").casefold()
    if "favorece over" in value or "mais " in value or "alonga" in value:
        return 1.0
    if "vies over" in value or "vies mais" in value or "vies longo" in value:
        return 0.5
    if "favorece under" in value or "menos " in value or "encurta" in value:
        return -1.0
    if "vies under" in value or "vies menos" in value or "vies curto" in value:
        return -0.5
    return 0.0






def build_live_bet_market_value_summary(
    *,
    df_scope: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    blue_side_team: str,
    pregame: Mapping[str, Any],
    draft: Mapping[str, Any],
    line_values: Mapping[str, Any],
    format_num_fn: Callable[[float, int], str],
    min_to_mmss_fn: Callable[[Any], str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    if df_scope is None or df_scope.empty:
        return {"rows": pd.DataFrame([]), "best_row": None}

    profile_lookup = dict(pregame.get("profile_lookup") or {})
    draft_market_calls = dict(draft.get("market_calls") or {})
    draft_market_means = dict(draft.get("market_means") or {})
    comp_compare = dict(draft.get("comp_compare") or {})
    side_a = "Blue" if str(blue_side_team) == str(team_a) else "Red"
    side_b = "Red" if side_a == "Blue" else "Blue"

    for market_label, meta in VALUE_MARKET_META.items():
        column = str(meta["column"])
        if column not in df_scope.columns:
            continue

        line_val = _safe_float(line_values.get(market_label))
        scope_mean = _safe_float(pd.to_numeric(df_scope.get(column), errors="coerce").mean())
        team_a_mean = _metric_team_mean(df_scope, team_name=str(team_a), column=column)
        team_b_mean = _metric_team_mean(df_scope, team_name=str(team_b), column=column)
        team_mean = _weighted_value([(team_a_mean, 0.5), (team_b_mean, 0.5)])
        side_mix = _weighted_value(
            [
                (_metric_team_mean(df_scope, team_name=str(team_a), column=column, side=side_a), 0.5),
                (_metric_team_mean(df_scope, team_name=str(team_b), column=column, side=side_b), 0.5),
            ]
        )
        h2h_mean = _metric_team_mean(df_scope, team_name=str(team_a), opponent=str(team_b), column=column)

        draft_mean_payload = dict(draft_market_means.get(market_label) or {})
        draft_mix = _weighted_value(
            [
                (_safe_float(draft_mean_payload.get("team_a")), 0.5),
                (_safe_float(draft_mean_payload.get("team_b")), 0.5),
            ]
        )
        blended = _weighted_value(
            [
                (team_mean, 0.34),
                (scope_mean, 0.16),
                (side_mix, 0.16),
                (h2h_mean, 0.12),
                (draft_mix, 0.22),
            ]
        )
        if blended is None:
            continue

        info_a = dict(profile_lookup.get(str(team_a), {}).get(market_label) or {})
        info_b = dict(profile_lookup.get(str(team_b), {}).get(market_label) or {})
        profile_shift = (
            (_profile_bucket_score(str(info_a.get("bucket") or "-")) + _profile_bucket_score(str(info_b.get("bucket") or "-"))) / 2.0
        ) * float(meta["profile_scale"])
        draft_shift = _draft_call_score(str(draft_market_calls.get(market_label) or "")) * float(meta["draft_scale"])
        semantic_shift = 0.0
        if market_label == "Torres":
            semantic_shift += float(_safe_float(comp_compare.get("tower_total_bias")) or 0.0)
        elif market_label == "DragÃµes":
            semantic_shift += float(_safe_float(comp_compare.get("dragons_total_bias")) or 0.0)
        elif market_label == "BarÃµes":
            semantic_shift += float(_safe_float(comp_compare.get("barons_total_bias")) or 0.0)
        fair_value = float(blended) + float(profile_shift) + float(draft_shift) + float(semantic_shift)

        fair_text = min_to_mmss_fn(fair_value) if str(meta["kind"]) == "time" else format_num_fn(float(fair_value), 1)

        lean = "Sem linha"
        abs_delta = 0.0
        if line_val is not None:
            delta = float(fair_value) - float(line_val)
            abs_delta = abs(delta)
            if market_label == "Torres" and abs_delta < 0.18:
                tower_total_call = str(comp_compare.get("tower_total_call") or "").strip()
                if tower_total_call == "Menos torres":
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif tower_total_call == "Mais torres":
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif delta >= 0.0:
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                else:
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
            elif market_label == "DragÃµes" and abs_delta < 0.12:
                dragons_total_call = str(comp_compare.get("dragons_total_call") or "").strip()
                if dragons_total_call == "Menos dragoes":
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif dragons_total_call == "Mais dragoes":
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif delta >= 0.0:
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                else:
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
            elif market_label == "BarÃµes" and abs_delta < 0.08:
                barons_total_call = str(comp_compare.get("barons_total_call") or "").strip()
                if barons_total_call == "Menos baroes":
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif barons_total_call == "Mais baroes":
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                elif delta >= 0.0:
                    lean = f"Vies {meta['high_label'].lower()} {format_num_fn(float(line_val), 1)}"
                else:
                    lean = f"Vies {meta['low_label'].lower()} {format_num_fn(float(line_val), 1)}"
            elif abs_delta < (0.25 * float(meta["delta_high"])):
                lean = "Muito perto da linha"
            elif delta > 0.0:
                lean = f"{meta['high_label']} {format_num_fn(float(line_val), 1) if meta['kind'] != 'time' else min_to_mmss_fn(line_val)}"
            else:
                lean = f"{meta['low_label']} {format_num_fn(float(line_val), 1) if meta['kind'] != 'time' else min_to_mmss_fn(line_val)}"

        evidence_parts = [
            ("Times", team_mean),
            ("Side", side_mix),
            ("H2H", h2h_mean),
            ("Draft", draft_mix),
            ("Escopo", scope_mean),
        ]
        evidence_count = sum(1 for _, value in evidence_parts if value is not None)
        confidence_label = _value_confidence_label(abs_delta=abs_delta, evidence_count=evidence_count, strong_delta=float(meta["delta_high"]))
        drivers_parts = [
            f"{label} {min_to_mmss_fn(value) if str(meta['kind']) == 'time' else format_num_fn(float(value), 1)}"
            for label, value in evidence_parts
            if value is not None
        ]
        if market_label == "Torres":
            tower_total_call = str(comp_compare.get("tower_total_call") or "").strip()
            tower_total_reason = str(comp_compare.get("tower_total_reason") or "").strip()
            if tower_total_call and tower_total_call != "Neutro":
                drivers_parts.append(f"Macro {tower_total_call}")
            if tower_total_reason:
                drivers_parts.append(tower_total_reason)
        elif market_label == "DragÃµes":
            dragons_total_call = str(comp_compare.get("dragons_total_call") or "").strip()
            dragons_total_reason = str(comp_compare.get("dragons_total_reason") or "").strip()
            if dragons_total_call and dragons_total_call != "Neutro":
                drivers_parts.append(f"Macro {dragons_total_call}")
            if dragons_total_reason:
                drivers_parts.append(dragons_total_reason)
        elif market_label == "BarÃµes":
            barons_total_call = str(comp_compare.get("barons_total_call") or "").strip()
            barons_total_reason = str(comp_compare.get("barons_total_reason") or "").strip()
            if barons_total_call and barons_total_call != "Neutro":
                drivers_parts.append(f"Macro {barons_total_call}")
            if barons_total_reason:
                drivers_parts.append(barons_total_reason)
        drivers = " | ".join(drivers_parts)
        rows.append(
            {
                "Mercado": market_label,
                "Linha": (
                    min_to_mmss_fn(line_val)
                    if line_val is not None and str(meta["kind"]) == "time"
                    else (format_num_fn(float(line_val), 1) if line_val is not None else "-")
                ),
                "Justo": fair_text,
                "Leitura": lean,
                "Confianca": confidence_label,
                "Draft": str(draft_market_calls.get(market_label) or "-"),
                "Resolucao": _resolution_label(str(info_a.get("bucket") or "-"), str(info_b.get("bucket") or "-")),
                "Drivers": drivers or "-",
                "_abs_delta": abs_delta,
            }
        )
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        best_row = max(rows, key=lambda row: float(row.get("_abs_delta") or 0.0))
        df_rows = df_rows.drop(columns=["_abs_delta"], errors="ignore")
    return {"rows": df_rows, "best_row": best_row}





def _window_label_from_minute(minute: float) -> str:
    if float(minute) <= 15.0:
        return "0-15"
    if float(minute) <= 20.0:
        return "15-20"
    return "20+"


def build_live_bet_live_assessment(
    *,
    team_a: str,
    team_b: str,
    df_scope: pd.DataFrame | None,
    minute: Any,
    gold_diff: Any,
    kills_a: Any,
    kills_b: Any,
    towers_a: Any,
    towers_b: Any,
    dragons_a: Any,
    dragons_b: Any,
    barons_a: Any,
    barons_b: Any,
    inhibitors_a: Any,
    inhibitors_b: Any,
    comp_compare: Mapping[str, Any],
    pregame_resolution: Mapping[str, Any] | None,
    baselines: Mapping[str, Any] | None,
    # ---- Phase 6 live state-to-win integration (optional) ----
    # When these are provided AND the feature flag is enabled in session_state,
    # the assessment includes a calibrated P(team_a wins) under the key 'live_p_win'.
    # All defaults preserve backwards compatibility with existing callers.
    session_state: Mapping[str, Any] | None = None,
    p_map_pre: Any = None,
    league: str = "",
    bo: int = 1,
    is_playoffs: bool = False,
    xp_diff: Any = None,
    cs_diff: Any = None,
    deaths_a: Any = None,
    deaths_b: Any = None,
    assists_a: Any = None,
    assists_b: Any = None,
) -> dict[str, Any]:
    minute_f = max(float(_safe_float(minute) or 0.0), 1.0)
    gold_diff_f = float(_safe_float(gold_diff) or 0.0)
    kills_a_f = float(_safe_float(kills_a) or 0.0)
    kills_b_f = float(_safe_float(kills_b) or 0.0)
    towers_a_f = float(_safe_float(towers_a) or 0.0)
    towers_b_f = float(_safe_float(towers_b) or 0.0)
    dragons_a_f = float(_safe_float(dragons_a) or 0.0)
    dragons_b_f = float(_safe_float(dragons_b) or 0.0)
    barons_a_f = float(_safe_float(barons_a) or 0.0)
    barons_b_f = float(_safe_float(barons_b) or 0.0)
    inhibitors_a_f = float(_safe_float(inhibitors_a) or 0.0)
    inhibitors_b_f = float(_safe_float(inhibitors_b) or 0.0)

    window = _window_label_from_minute(minute_f)
    if window == "0-15":
        window_team = str(comp_compare.get("early_team") or "Equilibrado")
    elif window == "15-20":
        window_team = str(comp_compare.get("mid_team") or comp_compare.get("early_team") or "Equilibrado")
    else:
        window_team = str(comp_compare.get("late_team") or "Equilibrado")

    state_edge = (
        1.00 * (gold_diff_f / 1000.0)
        + 0.80 * (towers_a_f - towers_b_f)
        + 1.05 * (dragons_a_f - dragons_b_f)
        + 1.65 * (barons_a_f - barons_b_f)
        + 1.10 * (inhibitors_a_f - inhibitors_b_f)
        + 0.12 * ((kills_a_f - kills_b_f))
    )
    if state_edge >= 0.75:
        leader = str(team_a)
    elif state_edge <= -0.75:
        leader = str(team_b)
    else:
        leader = "Equilibrado"

    late_team = str(comp_compare.get("late_team") or "Equilibrado")
    objective_team = str(comp_compare.get("objective_team") or "Equilibrado")
    engage_team = str(comp_compare.get("engage_team") or "Equilibrado")

    if leader == "Equilibrado":
        conversion = "Sem lado claramente na frente."
        reversal_risk = "Baixo"
    else:
        sustain_hits = int(leader == window_team) + int(leader == objective_team) + int(leader == engage_team)
        if late_team not in {"Equilibrado", leader} and minute_f >= 20.0:
            conversion = f"{leader} esta na frente, mas a janela atual favorece o outro lado."
            reversal_risk = "Alto"
        elif sustain_hits >= 2:
            conversion = f"Vantagem de {leader} conversa bem com a comp e a janela atual."
            reversal_risk = "Baixo"
        elif sustain_hits == 1:
            conversion = f"{leader} esta na frente, mas a conversao pede cuidado."
            reversal_risk = "Medio"
        else:
            conversion = f"{leader} esta na frente por estado, mas a comp nao ajuda tanto a sustentar."
            reversal_risk = "Medio"

    base = dict(baselines or {})
    base_kills = max(float(_safe_float(base.get("kills")) or 0.0), 1.0)
    base_towers = max(float(_safe_float(base.get("towers")) or 0.0), 1.0)
    base_time = max(float(_safe_float(base.get("time")) or 30.0), 10.0)
    base_dragons = max(float(_safe_float(base.get("dragons")) or 0.0), 0.1)
    base_inhibitors = max(float(_safe_float(base.get("inhibitors")) or 0.0), 0.1)

    kills_pace = (kills_a_f + kills_b_f) / minute_f
    kills_base_pace = base_kills / base_time
    towers_pace = (towers_a_f + towers_b_f) / minute_f
    towers_base_pace = base_towers / base_time

    if kills_pace >= kills_base_pace * 1.15:
        kills_live = "Acelerou para over"
    elif kills_pace <= kills_base_pace * 0.85:
        kills_live = "Travou para under"
    else:
        kills_live = str(comp_compare.get("kills_call") or "Neutro")

    if towers_pace >= towers_base_pace * 1.15:
        towers_live = "Pressao de torre alta"
    elif towers_pace <= towers_base_pace * 0.85:
        towers_live = "Torre abaixo do ritmo"
    else:
        towers_live = str(comp_compare.get("towers_call") or "Neutro")

    dragons_total = dragons_a_f + dragons_b_f
    if minute_f < 10.0:
        dragons_live = "Ainda cedo"
    elif dragons_total >= max(2.0, base_dragons * 0.75):
        dragons_live = "Pressao em dragoes"
    else:
        dragons_live = str(comp_compare.get("dragons_call") or "Neutro")

    if minute_f < 20.0:
        barons_live = "Ainda cedo"
    elif (barons_a_f + barons_b_f) >= 1.0:
        barons_live = "Barao ja apareceu"
    elif (dragons_a_f + dragons_b_f) >= 3.0 and objective_team != "Equilibrado":
        barons_live = f"Setup de {objective_team}"
    else:
        barons_live = str(comp_compare.get("barons_call") or "Neutro")

    inhibitors_total = inhibitors_a_f + inhibitors_b_f
    if minute_f < 18.0:
        inhibitors_live = "Ainda cedo"
    elif inhibitors_total >= max(1.0, base_inhibitors * 1.10):
        if inhibitors_a_f > inhibitors_b_f:
            inhibitors_live = f"{team_a} abriu base"
        elif inhibitors_b_f > inhibitors_a_f:
            inhibitors_live = f"{team_b} abriu base"
        else:
            inhibitors_live = "Bases abertas"
    elif inhibitors_total > 0.0:
        inhibitors_live = "Pressao de base"
    else:
        inhibitors_live = "Sem quebra de base"

    if dragons_a_f >= 3.0 and dragons_b_f >= 3.0:
        soul_point = "Soul point contestado"
    elif dragons_a_f >= 3.0:
        soul_point = f"{team_a} em soul point"
    elif dragons_b_f >= 3.0:
        soul_point = f"{team_b} em soul point"
    else:
        soul_point = "Sem soul point"

    if dragons_a_f >= 3.0 and dragons_b_f >= 3.0:
        soul_pressure = "Soul altamente contestado"
    elif dragons_a_f >= 3.0:
        soul_pressure = f"Soul iminente para {team_a}" if leader in {team_a, "Equilibrado"} or objective_team == team_a else f"{team_a} em soul point, mas sem controle pleno"
    elif dragons_b_f >= 3.0:
        soul_pressure = f"Soul iminente para {team_b}" if leader in {team_b, "Equilibrado"} or objective_team == team_b else f"{team_b} em soul point, mas sem controle pleno"
    elif dragons_a_f >= 2.0 or dragons_b_f >= 2.0:
        dragon_gap = dragons_a_f - dragons_b_f
        if dragon_gap >= 2.0:
            soul_pressure = f"Pressao de soul para {team_a}"
        elif dragon_gap <= -2.0:
            soul_pressure = f"Pressao de soul para {team_b}"
        else:
            soul_pressure = "Rota de soul ainda aberta"
    else:
        soul_pressure = "Soul ainda distante"

    if minute_f < 20.0:
        baron_pressure = "Barao indisponivel"
    elif (barons_a_f + barons_b_f) >= 1.0:
        if barons_a_f > barons_b_f and (inhibitors_a_f > inhibitors_b_f or towers_a_f > towers_b_f):
            baron_pressure = f"{team_a} ja converteu pressao de barao"
        elif barons_b_f > barons_a_f and (inhibitors_b_f > inhibitors_a_f or towers_b_f > towers_a_f):
            baron_pressure = f"{team_b} ja converteu pressao de barao"
        else:
            baron_pressure = "Barao ja impactou o mapa"
    else:
        baron_setup_a = (
            max(gold_diff_f, 0.0) / 1000.0
            + max(towers_a_f - towers_b_f, 0.0) * 0.55
            + max(dragons_a_f - dragons_b_f, 0.0) * 0.65
            + max(inhibitors_a_f - inhibitors_b_f, 0.0) * 0.80
            + (0.45 if objective_team == team_a else 0.0)
            + (0.25 if late_team == team_a else 0.0)
        )
        baron_setup_b = (
            max(-gold_diff_f, 0.0) / 1000.0
            + max(towers_b_f - towers_a_f, 0.0) * 0.55
            + max(dragons_b_f - dragons_a_f, 0.0) * 0.65
            + max(inhibitors_b_f - inhibitors_a_f, 0.0) * 0.80
            + (0.45 if objective_team == team_b else 0.0)
            + (0.25 if late_team == team_b else 0.0)
        )
        if abs(baron_setup_a - baron_setup_b) < 0.75:
            baron_pressure = "Setup de barao contestado"
        elif baron_setup_a > baron_setup_b:
            baron_pressure = f"Setup de barao para {team_a}"
        else:
            baron_pressure = f"Setup de barao para {team_b}"

    if minute_f >= base_time * 0.9 and (towers_a_f + towers_b_f) <= base_towers * 0.55:
        time_live = "Alonga"
    elif minute_f <= base_time * 0.55 and ((towers_a_f + towers_b_f) >= base_towers * 0.65 or kills_pace >= kills_base_pace * 1.25):
        time_live = "Encurta"
    else:
        time_live = str(comp_compare.get("time_call") or "Neutro")

    pregame_map = {str(k): str(v) for k, v in dict(pregame_resolution or {}).items() if str(k).strip()}
    shifts: list[str] = []
    live_map = {
        "Kills": kills_live,
        "Torres": towers_live,
        "DragÃµes": dragons_live,
        "BarÃµes": barons_live,
        "Inibidores": inhibitors_live,
        "Tempo": time_live,
    }
    for market, live_value in live_map.items():
        pre_value = str(pregame_map.get(market) or "").strip()
        if pre_value and pre_value != live_value:
            shifts.append(f"{market}: {pre_value} -> {live_value}")

    summary = f"Janela {window} | Lider atual: {leader} | {conversion}"
    conversion_a = _historical_conversion_from_state(
        df_scope=df_scope,
        team_name=str(team_a),
        minute=minute_f,
        team_gold_diff=gold_diff_f,
        team_dragons_diff=(dragons_a_f - dragons_b_f),
        team_towers_diff=(towers_a_f - towers_b_f),
    )
    conversion_b = _historical_conversion_from_state(
        df_scope=df_scope,
        team_name=str(team_b),
        minute=minute_f,
        team_gold_diff=-gold_diff_f,
        team_dragons_diff=(dragons_b_f - dragons_a_f),
        team_towers_diff=(towers_b_f - towers_a_f),
    )
    live_confidence = _live_confidence(
        minute=minute_f,
        gold_diff=gold_diff_f,
        conversion_games_a=int(conversion_a.get("games") or 0),
        conversion_games_b=int(conversion_b.get("games") or 0),
    )

    # ---- Phase 6: live state-to-win calibrated probability (optional) ----
    # Returns None when feature flag is off or any input is missing/invalid,
    # so the caller can fall back to existing heuristics seamlessly.
    live_p_win = _compute_live_p_win(
        session_state=session_state,
        minute=minute_f,
        p_map_pre=p_map_pre,
        league=str(league or ""),
        bo=int(bo),
        is_playoffs=bool(is_playoffs),
        team_a=str(team_a),
        gold_diff=gold_diff_f,
        xp_diff=xp_diff,
        cs_diff=cs_diff,
        kills_a=kills_a_f, kills_b=kills_b_f,
        deaths_a=deaths_a, deaths_b=deaths_b,
        assists_a=assists_a, assists_b=assists_b,
    )

    return {
        "window": window,
        "leader": leader,
        "state_edge": state_edge,
        "conversion": conversion,
        "reversal_risk": reversal_risk,
        "soul_point": soul_point,
        "soul_pressure": soul_pressure,
        "baron_pressure": baron_pressure,
        "summary": summary,
        "market_live": live_map,
        "shifts": shifts,
        "conversion_history_a": conversion_a,
        "conversion_history_b": conversion_b,
        "live_confidence": live_confidence,
        "live_p_win": live_p_win,  # P(team_a wins), or None if model not active
    }


def _compute_live_p_win(
    *,
    session_state: Mapping[str, Any] | None,
    minute: float,
    p_map_pre: Any,
    league: str,
    bo: int,
    is_playoffs: bool,
    team_a: str,
    gold_diff: float,
    xp_diff: Any,
    cs_diff: Any,
    kills_a: float, kills_b: float,
    deaths_a: Any, deaths_b: Any,
    assists_a: Any, assists_b: Any,
) -> float | None:
    """Call the Phase 6 live state model with built features. Returns None on any failure.

    All inputs are flexible: missing optional features default to 0. The function
    is intentionally defensive — any error or missing required input returns None
    so the caller continues using existing heuristics.
    """
    try:
        from app_core.live_state_runtime_core import predict_live_p_win
        # Snap to the nearest supported snapshot minute (10, 15, 20, 25).
        try:
            m = round(float(minute))
        except (TypeError, ValueError):
            return None
        supported = (10, 15, 20, 25)
        # Only run when we're within a reasonable window of a supported snapshot.
        nearest = min(supported, key=lambda x: abs(x - m))
        if abs(nearest - m) > 3:
            return None
        # Note: _safe_float in this module is safe_float_or_none (returns float|None).
        p_pre_safe = _safe_float(p_map_pre)
        if p_pre_safe is None:
            return None
        p_pre = float(p_pre_safe)
        import math as _math
        if not _math.isfinite(p_pre) or p_pre <= 0.0 or p_pre >= 1.0:
            return None

        def _f(v):
            fv = _safe_float(v)
            if fv is None or not _math.isfinite(float(fv)):
                return 0.0
            return float(fv)

        # Reconstruct symmetric "at minute" features. Gold/xp/cs are stored
        # as totals for blue and red separately; we have only the diff, so we
        # split it symmetrically around a neutral baseline. The model only
        # consumes diffs (in features.py), so symmetric split is mathematically
        # equivalent for the model's purposes.
        gold = max(0.0, _f(gold_diff))
        opp_gold = max(0.0, -_f(gold_diff))
        xp = max(0.0, _f(xp_diff))
        opp_xp = max(0.0, -_f(xp_diff))
        cs = max(0.0, _f(cs_diff))
        opp_cs = max(0.0, -_f(cs_diff))
        state_features = {
            f"goldat{nearest}": gold, f"opp_goldat{nearest}": opp_gold,
            f"xpat{nearest}": xp,     f"opp_xpat{nearest}": opp_xp,
            f"csat{nearest}": cs,     f"opp_csat{nearest}": opp_cs,
            f"killsat{nearest}": _f(kills_a),   f"opp_killsat{nearest}": _f(kills_b),
            f"deathsat{nearest}": _f(deaths_a), f"opp_deathsat{nearest}": _f(deaths_b),
            f"assistsat{nearest}": _f(assists_a), f"opp_assistsat{nearest}": _f(assists_b),
        }
        return predict_live_p_win(
            session_state=session_state,
            minute=nearest,
            gameid=str(team_a or "unknown_game"),
            side="Blue",
            p_map_pre=p_pre,
            league=str(league or ""),
            bo=int(bo),
            is_playoffs=bool(is_playoffs),
            state_features=state_features,
        )
    except Exception:
        return None

def _live_confidence(
    *,
    minute: float,
    gold_diff: float,
    conversion_games_a: int,
    conversion_games_b: int,
) -> dict[str, Any]:
    state_score = 0.25
    if float(minute) >= 10.0:
        state_score += 0.2
    if float(minute) >= 20.0:
        state_score += 0.15
    if abs(float(gold_diff)) >= 1000.0:
        state_score += 0.15
    if abs(float(gold_diff)) >= 2500.0:
        state_score += 0.1
    sample_score = min(max(int(conversion_games_a), int(conversion_games_b)) / 10.0, 1.0)
    score = (0.55 * state_score) + (0.45 * sample_score)
    return {
        "score": float(score),
        "label": _confidence_rank_from_score(score),
        "state_score": float(state_score),
        "sample_score": float(sample_score),
    }


def _historical_conversion_from_state(
    *,
    df_scope: pd.DataFrame | None,
    team_name: str,
    minute: float,
    team_gold_diff: float,
    team_dragons_diff: float = 0.0,
    team_towers_diff: float = 0.0,
) -> dict[str, Any]:
    empty = {"summary": "Sem base historica parecida.", "games": 0, "win_rate": None}
    if df_scope is None or df_scope.empty or not str(team_name or "").strip():
        return empty
    if "team" not in df_scope.columns:
        return empty
    team_df = df_scope[df_scope["team"].astype(str) == str(team_name)].copy()
    if team_df.empty:
        return empty
    minute_f = float(minute or 0.0)
    gold_diff_f = float(team_gold_diff or 0.0)
    if minute_f <= 15.0:
        source_col = "golddiffat15_for"
    else:
        source_col = "golddiffat20_for"
    if source_col not in team_df.columns:
        return empty
    edge = pd.to_numeric(team_df.get(source_col), errors="coerce")
    if edge.dropna().empty:
        return empty
    abs_diff = abs(gold_diff_f)
    if abs_diff >= 2500.0:
        band = 2500.0
    elif abs_diff >= 1000.0:
        band = 1000.0
    else:
        band = 0.0
    if band >= 2500.0:
        if gold_diff_f >= 0:
            mask = edge >= 2500.0
            lead_text = "lead forte"
        else:
            mask = edge <= -2500.0
            lead_text = "desvantagem forte"
    elif band >= 1000.0:
        if gold_diff_f >= 0:
            mask = (edge >= 1000.0) & (edge < 2500.0)
            lead_text = "lead moderada"
        else:
            mask = (edge <= -1000.0) & (edge > -2500.0)
            lead_text = "desvantagem moderada"
    else:
        if gold_diff_f >= 0:
            mask = (edge >= 0.0) & (edge < 1000.0)
            lead_text = "lead leve"
        else:
            mask = (edge < 0.0) & (edge > -1000.0)
            lead_text = "desvantagem leve"
    sample = team_df[mask].copy()
    source_games = int(sample.shape[0])
    signal_parts: list[str] = []
    if not sample.empty and minute_f >= 8.0 and "firstdragon" in sample.columns and abs(float(team_dragons_diff or 0.0)) >= 1.0:
        want_first_dragon = float(team_dragons_diff or 0.0) > 0.0
        dragon_mask = pd.to_numeric(sample.get("firstdragon"), errors="coerce").fillna(0.0) > 0.5
        sample_dragon = sample[dragon_mask] if want_first_dragon else sample[~dragon_mask]
        if int(sample_dragon.shape[0]) >= max(2, math.ceil(source_games * 0.35)):
            sample = sample_dragon.copy()
            signal_parts.append("com first dragon" if want_first_dragon else "sem first dragon")
    if not sample.empty and minute_f >= 10.0 and "firsttower" in sample.columns and abs(float(team_towers_diff or 0.0)) >= 1.0:
        want_first_tower = float(team_towers_diff or 0.0) > 0.0
        tower_mask = pd.to_numeric(sample.get("firsttower"), errors="coerce").fillna(0.0) > 0.5
        sample_tower = sample[tower_mask] if want_first_tower else sample[~tower_mask]
        if int(sample_tower.shape[0]) >= max(2, math.ceil(source_games * 0.35)):
            sample = sample_tower.copy()
            signal_parts.append("com first tower" if want_first_tower else "sem first tower")
    if sample.empty or "win" not in sample.columns:
        return empty
    win_rate = _safe_float(pd.to_numeric(sample.get("win"), errors="coerce").mean())
    if win_rate is None:
        return empty
    if minute_f <= 15.0:
        window = "aos 15"
    elif minute_f <= 20.0:
        window = "aos 20"
    else:
        window = "apos 20 (proxy aos 20)"
    signal_suffix = f" | {' + '.join(signal_parts)}" if signal_parts else ""
    return {
        "summary": f"{team_name}: {format(100.0 * win_rate, '.1f')}% em {int(sample.shape[0])} jogos com {lead_text} {window}{signal_suffix}",
        "games": int(sample.shape[0]),
        "win_rate": float(win_rate),
        "source_games": int(source_games),
        "signals": signal_parts,
    }


__all__ = [
    "build_live_bet_live_assessment",
    "build_live_bet_market_value_summary",
    "build_live_bet_pregame_summary",
]
