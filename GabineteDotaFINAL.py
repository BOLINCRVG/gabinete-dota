from __future__ import annotations

from html import escape
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from opendota_client import (
    auth_status,
    get_hero_stats,
    get_match,
    get_player_heroes,
    get_pro_matches,
    get_team_heroes,
    get_team_matches,
    get_team_players,
    get_team_summary,
)
from opendota_sync import backfill_legacy_matches, csv_health_report, import_match_ids, load_sync_state, sync_professional_leagues, sync_recent_pro_matches

from dota_core import (
    LinesConfig,
    TOTAL_MARKETS,
    analyze_handicap_market,
    analyze_total_market,
    analyze_winner_market,
    app_dir,
    build_overview,
    build_stats_table,
    build_summary_rows,
    clamp_prob,
    csv_path,
    detail_tables,
    fair_odds,
    fmt_mmss_from_minutes,
    load_saved_games,
    make_saved_key,
    save_saved_games,
    upsert_saved_game,
    DotaDataset,
)

st.set_page_config(page_title="Gabinete Dota", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.gabi-hero {padding: 1rem 1.25rem; border: 1px solid rgba(255,255,255,.08); border-radius: 14px; background: linear-gradient(180deg, rgba(26,32,44,.86), rgba(17,24,39,.92));}
.gabi-kicker {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #8fb3ff; font-weight: 700;}
.gabi-section {font-size: 1.6rem; font-weight: 800; margin: 1rem 0 .5rem 0;}
.gabi-caption {color: #9fb0c8;}
.gabi-chip {display:inline-block; padding:.16rem .55rem; border-radius:999px; border:1px solid rgba(255,255,255,.16); font-size:.78rem; margin-right:.35rem; margin-bottom:.25rem;}
.gabi-chip-muted {background: rgba(148,163,184,.12); color: #cbd5e1; border-color: rgba(148,163,184,.24);}
.gabi-chip-ml-strong {background: rgba(22,163,74,.18); color: #a7f3d0; border-color: rgba(74,222,128,.50);}
.gabi-chip-ml-moderate {background: rgba(245,158,11,.18); color: #fde68a; border-color: rgba(251,191,36,.48);}
.gabi-chip-ml-weak {background: rgba(14,165,233,.18); color: #bae6fd; border-color: rgba(56,189,248,.48);}
.gabi-chip-ml-heuristic {background: rgba(239,68,68,.16); color: #fecaca; border-color: rgba(248,113,113,.45);}
.gabi-chip-ml-off {background: rgba(107,114,128,.20); color: #d1d5db; border-color: rgba(156,163,175,.35);}
.gabi-card {border:1px solid rgba(255,255,255,.08); border-radius:12px; padding: .75rem 1rem; background: rgba(17,24,39,.55);}
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "dota_nav_page": "Plays",
        "dota_technical_view": "Atualização CSV",
        "dota_year": None,
        "dota_league": "(Todas)",
        "dota_team1": "",
        "dota_team2": "",
        "dota_event_name": "",
        "dota_match_format": "BO3",
        "dota_manual_team1": "",
        "dota_manual_team2": "",
        "dota_manual_event_name": "",
        "dota_manual_format": "BO3",
        "line_total_kills": "",
        "line_total_towers": "",
        "line_total_roshans": "",
        "line_total_barracks": "",
        "line_total_time": "",
        "line_hc_kills": "",
        "line_hc_towers": "",
        "line_hc_roshans": "",
        "line_hc_barracks": "",
        "dota_laplace": True,
        "dota_stats_championship": "",
        "dota_stats_side": "All",
        "dota_stats_result": "All",
        "dota_stats_market": "Kills",
        "dota_stats_custom_line": "",
        "dota_summary_max_odd": 2.50,
        "dota_resumo_req_liga": True,
        "dota_resumo_req_h2h": False,
        "dota_resumo_req_model": True,
        "dota_resumo_show_laplace_cols": True,
        "dota_consistency_market": "Kills",
        "dota_draft_team1": "",
        "dota_draft_team2": "",
        "dota_draft_blueside": "Time 1",
        "dota_saved_key": "",
        "dota_source_mode": "CSV local",
        "od_league_filter": "",
        "od_match_label": "",
        "od_match_id_manual": "",
        "dota_series_use_manual": False,
        "dota_series_manual_p1": 50.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_dataset() -> DotaDataset:
    source = csv_path()
    mtime = source.stat().st_mtime

    @st.cache_resource(show_spinner=False)
    def _cached(_mtime: float) -> DotaDataset:
        return DotaDataset(source)

    return _cached(mtime)


def _lines_from_state() -> LinesConfig:
    return LinesConfig(
        total_kills=_parse_line_text(st.session_state.get("line_total_kills", "")),
        total_towers=_parse_line_text(st.session_state.get("line_total_towers", "")),
        total_roshans=_parse_line_text(st.session_state.get("line_total_roshans", "")),
        total_barracks=_parse_line_text(st.session_state.get("line_total_barracks", "")),
        total_time_min=_parse_time_text(st.session_state.get("line_total_time", "")),
        hc_kills_t1=_parse_line_text(st.session_state.get("line_hc_kills", "")),
        hc_towers_t1=_parse_line_text(st.session_state.get("line_hc_towers", "")),
        hc_roshans_t1=_parse_line_text(st.session_state.get("line_hc_roshans", "")),
        hc_barracks_t1=_parse_line_text(st.session_state.get("line_hc_barracks", "")),
    )


def _apply_lines_to_state(lines: LinesConfig) -> None:
    st.session_state["line_total_kills"] = ", ".join(f"{x:g}" for x in lines.total_kills)
    st.session_state["line_total_towers"] = ", ".join(f"{x:g}" for x in lines.total_towers)
    st.session_state["line_total_roshans"] = ", ".join(f"{x:g}" for x in lines.total_roshans)
    st.session_state["line_total_barracks"] = ", ".join(f"{x:g}" for x in lines.total_barracks)
    st.session_state["line_total_time"] = ", ".join(fmt_mmss_from_minutes(x) for x in lines.total_time_min)
    st.session_state["line_hc_kills"] = ", ".join(f"{x:g}" for x in lines.hc_kills_t1)
    st.session_state["line_hc_towers"] = ", ".join(f"{x:g}" for x in lines.hc_towers_t1)
    st.session_state["line_hc_roshans"] = ", ".join(f"{x:g}" for x in lines.hc_roshans_t1)
    st.session_state["line_hc_barracks"] = ", ".join(f"{x:g}" for x in lines.hc_barracks_t1)


def _init_prefixed_lines(prefix: str, lines: LinesConfig) -> None:
    defaults = {
        prefix + "line_total_kills": ", ".join(f"{x:g}" for x in lines.total_kills),
        prefix + "line_total_towers": ", ".join(f"{x:g}" for x in lines.total_towers),
        prefix + "line_total_roshans": ", ".join(f"{x:g}" for x in lines.total_roshans),
        prefix + "line_total_barracks": ", ".join(f"{x:g}" for x in lines.total_barracks),
        prefix + "line_total_time": ", ".join(fmt_mmss_from_minutes(x) for x in lines.total_time_min),
        prefix + "line_hc_kills": ", ".join(f"{x:g}" for x in lines.hc_kills_t1),
        prefix + "line_hc_towers": ", ".join(f"{x:g}" for x in lines.hc_towers_t1),
        prefix + "line_hc_roshans": ", ".join(f"{x:g}" for x in lines.hc_roshans_t1),
        prefix + "line_hc_barracks": ", ".join(f"{x:g}" for x in lines.hc_barracks_t1),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _prefixed_lines_from_state(prefix: str) -> LinesConfig:
    return LinesConfig(
        total_kills=_parse_line_text(st.session_state.get(prefix + "line_total_kills", "")),
        total_towers=_parse_line_text(st.session_state.get(prefix + "line_total_towers", "")),
        total_roshans=_parse_line_text(st.session_state.get(prefix + "line_total_roshans", "")),
        total_barracks=_parse_line_text(st.session_state.get(prefix + "line_total_barracks", "")),
        total_time_min=_parse_time_text(st.session_state.get(prefix + "line_total_time", "")),
        hc_kills_t1=_parse_line_text(st.session_state.get(prefix + "line_hc_kills", "")),
        hc_towers_t1=_parse_line_text(st.session_state.get(prefix + "line_hc_towers", "")),
        hc_roshans_t1=_parse_line_text(st.session_state.get(prefix + "line_hc_roshans", "")),
        hc_barracks_t1=_parse_line_text(st.session_state.get(prefix + "line_hc_barracks", "")),
    )


def _parse_line_text(text: str) -> List[float]:
    from dota_core import parse_float_list

    return parse_float_list(text)


def _parse_time_text(text: str) -> List[float]:
    from dota_core import parse_time_line

    return parse_time_line(text)


def _fmt_pct(prob: float) -> str:
    return f"{clamp_prob(prob) * 100:.1f}%"


def _fmt_stat(value: float, minutes: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if minutes:
        return fmt_mmss_from_minutes(float(value))
    return f"{float(value):.1f}"


def _ml_signal_class(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized == "ml forte":
        return "gabi-chip gabi-chip-ml-strong"
    if normalized == "ml moderado":
        return "gabi-chip gabi-chip-ml-moderate"
    if normalized == "ml fraco":
        return "gabi-chip gabi-chip-ml-weak"
    if normalized == "usar heurístico":
        return "gabi-chip gabi-chip-ml-heuristic"
    if normalized == "ml indisponível":
        return "gabi-chip gabi-chip-ml-off"
    return "gabi-chip"


def _render_ml_signal(label: str, detail: str) -> None:
    parts = [f"<span class='{_ml_signal_class(label)}'>{escape(f'Sinal ML: {label}')}</span>"]
    if str(detail or "").strip():
        parts.append(f"<span class='gabi-chip gabi-chip-muted'>{escape(str(detail))}</span>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _lines_status_parts(lines: LinesConfig) -> tuple[int, int]:
    total_markets = sum(
        1
        for values in [
            lines.total_kills,
            lines.total_towers,
            lines.total_roshans,
            lines.total_barracks,
            lines.total_time_min,
        ]
        if values
    )
    handicaps = sum(
        1
        for values in [
            lines.hc_kills_t1,
            lines.hc_towers_t1,
            lines.hc_roshans_t1,
            lines.hc_barracks_t1,
        ]
        if values
    )
    return int(total_markets), int(handicaps)


def _lines_status_label(lines: LinesConfig) -> str:
    total_markets, handicaps = _lines_status_parts(lines)
    if total_markets <= 0 and handicaps <= 0:
        return "Sem linhas salvas"
    return f"{total_markets} mercados | {handicaps} handicaps"


def _lines_signature(lines: LinesConfig) -> str:
    try:
        return json.dumps(lines.to_dict(), sort_keys=True, ensure_ascii=True)
    except Exception:
        return ""


def _render_context_bar(year: Optional[int], league: str, team1: str, team2: str, extra: Optional[List[str]] = None) -> None:
    chips = [
        f"Confronto: {team1} x {team2}",
        f"Temporada: {year if year is not None else '-'}",
        f"Liga: {league}",
    ]
    for item in extra or []:
        if item:
            chips.append(str(item))
    st.markdown("".join(f"<span class='gabi-chip'>{chip}</span>" for chip in chips), unsafe_allow_html=True)


def _format_sync_timestamp(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "-"
    try:
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.tz_convert("America/Sao_Paulo").strftime("%d/%m/%Y %H:%M")
    except Exception:
        pass
    return raw


def _format_age_hours(value: object) -> str:
    try:
        hours = float(value)
    except Exception:
        return "-"
    if not math.isfinite(hours):
        return "-"
    if hours < 1.0:
        return f"{int(round(hours * 60.0))} min"
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} d"


def _render_tab_hero_box(title: str, subtitle: str, chips: Optional[List[str]] = None) -> None:
    chip_html = "".join(
        f"<span class='gabi-chip'>{escape(str(chip))}</span>"
        for chip in (chips or [])
        if str(chip or "").strip()
    )
    st.markdown(
        f"""
<div class='gabi-hero' style='margin-top:6px;'>
  <div class='gabi-kicker'>Leitura</div>
  <div style='font-size:1.1rem; font-weight:800; margin-top:.2rem;'>{escape(str(title))}</div>
  <div class='gabi-caption' style='margin-top:.2rem;'>{escape(str(subtitle))}</div>
  <div style='margin-top:.55rem;'>{chip_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_tab_section_head(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<div class='gabi-caption' style='margin-top:.18rem;'>{escape(str(subtitle))}</div>" if str(subtitle or "").strip() else ""
    st.markdown(
        f"""
<div style='margin-bottom:.55rem;'>
  <div class='gabi-kicker'>Secao</div>
  <div style='font-size:1.0rem; font-weight:800; margin-top:.1rem;'>{escape(str(title))}</div>
  {subtitle_html}
</div>
        """,
        unsafe_allow_html=True,
    )



def _normalize_ui_state() -> None:
    page_map = {"Tecnico": "Técnico"}
    tech_map = {"Atualizacao CSV": "Atualização CSV"}
    st.session_state["dota_nav_page"] = page_map.get(str(st.session_state.get("dota_nav_page") or ""), st.session_state.get("dota_nav_page", "Plays"))
    st.session_state["dota_technical_view"] = tech_map.get(str(st.session_state.get("dota_technical_view") or ""), st.session_state.get("dota_technical_view", "Atualização CSV"))


def _recent_form_caption(df: pd.DataFrame) -> str:
    if df is None or df.empty or "result" not in df.columns:
        return "Last 5: - | Last 10: -"
    last5 = df.head(5)
    last10 = df.head(10)
    wr5 = float(pd.to_numeric(last5["result"], errors="coerce").fillna(0).mean() * 100.0) if not last5.empty else float("nan")
    wr10 = float(pd.to_numeric(last10["result"], errors="coerce").fillna(0).mean() * 100.0) if not last10.empty else float("nan")
    return f"Last 5: {wr5:.1f}% | Last 10: {wr10:.1f}%"


def _overview_table(team_label: str, block: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metrica": "Win Rate", team_label: _fmt_pct(block["win_rate"]),},
            {"Metrica": "# Games", team_label: block["games"],},
            {"Metrica": "Record", team_label: block["record"],},
            {"Metrica": "Radiant", team_label: _fmt_pct(block["blue"]) if not pd.isna(block["blue"]) else "-",},
            {"Metrica": "Dire", team_label: _fmt_pct(block["red"]) if not pd.isna(block["red"]) else "-",},
            {"Metrica": "Avg Kills", team_label: _fmt_stat(block["avg_kills"]),},
            {"Metrica": "Avg Towers", team_label: _fmt_stat(block["avg_towers"]),},
            {"Metrica": "Avg Roshans", team_label: _fmt_stat(block["avg_roshans"]),},
            {"Metrica": "Avg Barracks", team_label: _fmt_stat(block["avg_barracks"]),},
            {"Metrica": "Avg Time", team_label: _fmt_stat(block["avg_time"], minutes=True),},
        ]
    )


def _render_overview(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, overview: dict) -> None:
    _render_context_bar(year, league, team1, team2, ["Escopo: recorte atual + ano todo"])
    winner = analyze_winner_market(dataset, year, league, team1, team2, bool(st.session_state.get("dota_laplace", True)))
    full_year_team1 = dataset.team_games(year, "(Todas)", team1)
    full_year_team2 = dataset.team_games(year, "(Todas)", team2)
    full_year_h2h = dataset.pair_games(year, "(Todas)", team1, team2)
    full_year_overview = {
        "team1": {
            "games": int(len(full_year_team1)),
            "win_rate": float(pd.to_numeric(full_year_team1["result"], errors="coerce").mean()) if not full_year_team1.empty else float("nan"),
            "record": f"{int(pd.to_numeric(full_year_team1['result'], errors='coerce').fillna(0).sum())}W-{int(len(full_year_team1) - pd.to_numeric(full_year_team1['result'], errors='coerce').fillna(0).sum())}L" if not full_year_team1.empty else "-",
            "blue": float(full_year_team1[full_year_team1["side"] == "radiant"]["result"].mean()) if not full_year_team1.empty else float("nan"),
            "red": float(full_year_team1[full_year_team1["side"] == "dire"]["result"].mean()) if not full_year_team1.empty else float("nan"),
            "avg_kills": float(full_year_team1["totalkills"].mean()) if not full_year_team1.empty else float("nan"),
            "avg_towers": float(full_year_team1["totaltowers"].mean()) if not full_year_team1.empty else float("nan"),
            "avg_roshans": float(full_year_team1["totalroshans"].mean()) if not full_year_team1.empty else float("nan"),
            "avg_barracks": float(full_year_team1["totalbarracks"].mean()) if not full_year_team1.empty else float("nan"),
            "avg_time": float(full_year_team1["time_min"].mean()) if not full_year_team1.empty else float("nan"),
        },
        "team2": {
            "games": int(len(full_year_team2)),
            "win_rate": float(pd.to_numeric(full_year_team2["result"], errors="coerce").mean()) if not full_year_team2.empty else float("nan"),
            "record": f"{int(pd.to_numeric(full_year_team2['result'], errors='coerce').fillna(0).sum())}W-{int(len(full_year_team2) - pd.to_numeric(full_year_team2['result'], errors='coerce').fillna(0).sum())}L" if not full_year_team2.empty else "-",
            "blue": float(full_year_team2[full_year_team2["side"] == "radiant"]["result"].mean()) if not full_year_team2.empty else float("nan"),
            "red": float(full_year_team2[full_year_team2["side"] == "dire"]["result"].mean()) if not full_year_team2.empty else float("nan"),
            "avg_kills": float(full_year_team2["totalkills"].mean()) if not full_year_team2.empty else float("nan"),
            "avg_towers": float(full_year_team2["totaltowers"].mean()) if not full_year_team2.empty else float("nan"),
            "avg_roshans": float(full_year_team2["totalroshans"].mean()) if not full_year_team2.empty else float("nan"),
            "avg_barracks": float(full_year_team2["totalbarracks"].mean()) if not full_year_team2.empty else float("nan"),
            "avg_time": float(full_year_team2["time_min"].mean()) if not full_year_team2.empty else float("nan"),
        },
    }

    with st.container(border=True):
        st.markdown("<div class='gabi-kicker'>Operação</div><div class='gabi-section'>ML da Partida</div>", unsafe_allow_html=True)
        st.caption("Resumo rápido do vencedor no recorte atual. Probabilidade final usa o blend entre heurístico e classificador supervisionado quando houver amostra.")
        _render_ml_signal(str(winner.get("ml_signal_label", "-")), str(winner.get("ml_signal_detail", "")))
        hero_cards = st.columns(5)
        hero_cards[0].metric(team1, _fmt_pct(float(winner["model_t1_prob"])))
        hero_cards[1].metric(team2, _fmt_pct(float(winner["model_t2_prob"])))
        hero_cards[2].metric("Sinal ML", str(winner.get("ml_signal_label", "-")))
        hero_cards[3].metric("Blend ML", f"{float(winner.get('ml_blend_weight', 0.0)) * 100:.0f}%")
        if winner.get("ml_holdout_auc") is not None:
            hero_cards[4].metric("AUC Holdout", f"{float(winner['ml_holdout_auc']):.3f}")
        else:
            hero_cards[4].metric("H2H", int(winner["h2h_games"]))
        winner_df = pd.DataFrame(
            [
                {
                    "Lado": team1,
                    "Prob. Final": _fmt_pct(float(winner["model_t1_prob"])),
                    "Odd Final": f"{fair_odds(float(winner['model_t1_prob'])):.2f}",
                    "Heur.": _fmt_pct(float(winner["heuristic_t1_prob"])),
                    "ML": _fmt_pct(float(winner["ml_t1_prob"])) if winner.get("ml_t1_prob") is not None else "-",
                    "Forma": _fmt_pct(float(winner["team1_win_rate"])),
                    "H2H": _fmt_pct(float(winner["h2h_win_rate_t1"])) if int(winner["h2h_games"]) > 0 else "-",
                },
                {
                    "Lado": team2,
                    "Prob. Final": _fmt_pct(float(winner["model_t2_prob"])),
                    "Odd Final": f"{fair_odds(float(winner['model_t2_prob'])):.2f}",
                    "Heur.": _fmt_pct(float(winner["heuristic_t2_prob"])),
                    "ML": _fmt_pct(float(winner["ml_t2_prob"])) if winner.get("ml_t2_prob") is not None else "-",
                    "Forma": _fmt_pct(float(winner["team2_win_rate"])),
                    "H2H": _fmt_pct(1.0 - float(winner["h2h_win_rate_t1"])) if int(winner["h2h_games"]) > 0 else "-",
                },
            ]
        )
        st.dataframe(winner_df, hide_index=True, use_container_width=True)
        ml_parts = [f"Fonte: {winner.get('model_source', 'Heuristico')}"]
        if winner.get("ml_holdout_brier") is not None:
            ml_parts.append(f"Brier holdout: {float(winner['ml_holdout_brier']):.3f}")
        if winner.get("ml_baseline_brier") is not None:
            ml_parts.append(f"Brier baseline: {float(winner['ml_baseline_brier']):.3f}")
        if winner.get("ml_model_label"):
            ml_parts.append(f"Modelo: {winner.get('ml_model_label')}")
        if winner.get("ml_reason") and str(winner.get("ml_reason")) != "ok":
            ml_parts.append(f"Status: {winner.get('ml_reason')}")
        st.caption(" | ".join(ml_parts))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Overall</div>", unsafe_allow_html=True)
        st.caption("Recorte atual")
        current_df = pd.merge(_overview_table(team1, overview["team1"]), _overview_table(team2, overview["team2"]), on="Metrica", how="outer")
        st.dataframe(current_df, hide_index=True, use_container_width=True)
        st.caption("Ano todo")
        full_df = pd.merge(_overview_table(team1, full_year_overview["team1"]), _overview_table(team2, full_year_overview["team2"]), on="Metrica", how="outer")
        st.dataframe(full_df, hide_index=True, use_container_width=True)
    with col2:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Recent Form</div>", unsafe_allow_html=True)
        st.write(f"**{team1}**")
        st.caption(_recent_form_caption(overview["recent_team1"]))
        for row in _recent_rows(overview["recent_team1"]):
            st.markdown(f"- {row}")
        st.divider()
        st.write(f"**{team2}**")
        st.caption(_recent_form_caption(overview["recent_team2"]))
        for row in _recent_rows(overview["recent_team2"]):
            st.markdown(f"- {row}")
    with col3:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Past Faceoffs</div>", unsafe_allow_html=True)
        st.metric("H2H Games", int(len(full_year_h2h)))
        if not full_year_h2h.empty:
            t1_h2h_wins = int(pd.to_numeric(full_year_h2h["result_t1"], errors="coerce").fillna(0).sum())
            st.markdown(f"<div class='gabi-card'><b>{team1}</b><br>{t1_h2h_wins}W-{int(len(full_year_h2h)-t1_h2h_wins)}L</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='gabi-card'><b>{team2}</b><br>{int(len(full_year_h2h)-t1_h2h_wins)}W-{t1_h2h_wins}L</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            for row in _h2h_rows(full_year_h2h, team1, team2):
                st.markdown(f"- {row}")
        else:
            st.info("Sem H2H no recorte anual completo.")


def _recent_rows(df: pd.DataFrame) -> List[str]:
    rows: List[str] = []
    if df is None or df.empty:
        return rows
    for _, row in df.head(10).iterrows():
        result = "W" if int(row.get("result", 0) or 0) == 1 else "L"
        side = "R" if str(row.get("side", "")).lower() == "radiant" else "D"
        date = str(row.get("date", "") or "")[:10] or "-"
        opp = str(row.get("opp_teamname", "-") or "-")
        league_name = str(row.get("league", "-") or "-")
        try:
            k = int(float(row.get("teamkills") or 0))
            ok = int(float(row.get("oppkills") or 0))
            score = f"{k}-{ok}"
        except Exception:
            score = ""
        rows.append(f"{date} | {side} vs {opp} | **{result}** {score} | {league_name}")
    return rows


def _h2h_rows(df: pd.DataFrame, team1: str, team2: str) -> List[str]:
    rows: List[str] = []
    if df is None or df.empty:
        return rows
    for _, row in df.head(10).iterrows():
        result = "W" if int(row.get("result_t1", 0) or 0) == 1 else "L"
        date = str(row.get("date", "") or "")[:10] or "-"
        league_name = str(row.get("league", "-") or "-")
        try:
            k1 = int(float(row.get("kills_t1") or 0))
            k2 = int(float(row.get("kills_t2") or 0))
            score = f"{k1}-{k2}"
        except Exception:
            score = ""
        try:
            t = float(row.get("time_min") or 0)
            time_str = fmt_mmss_from_minutes(t) if t > 0 else ""
        except Exception:
            time_str = ""
        rows.append(f"{date} | {team1} **{result}** {score} {time_str} | {league_name}")
    return rows


def _market_metric_display(value_col: str, value: object) -> str:
    try:
        val = float(value)
    except Exception:
        return "-"
    if not math.isfinite(val):
        return "-"
    if value_col == "time_min":
        return fmt_mmss_from_minutes(val)
    if value_col.startswith("diff_"):
        return f"{val:+.1f}" if abs(val - round(val)) > 1e-9 else f"{int(round(val)):+d}"
    return f"{val:.1f}" if abs(val - round(val)) > 1e-9 else f"{int(round(val))}"


def _market_block(df: pd.DataFrame, value_col: str) -> dict:
    if df is None or df.empty:
        return {"games": 0, "record": "-", "avg": float("nan")}
    wins = int(pd.to_numeric(df.get("result"), errors="coerce").fillna(0).sum()) if "result" in df.columns else 0
    games = int(len(df))
    losses = max(0, games - wins)
    avg = float(pd.to_numeric(df.get(value_col), errors="coerce").mean()) if value_col in df.columns else float("nan")
    return {"games": games, "record": f"{wins}W-{losses}L", "avg": avg}


def _market_recent_rows(df: pd.DataFrame, value_col: str, n: int = 10) -> List[str]:
    rows: List[str] = []
    if df is None or df.empty:
        return rows
    frame = df.sort_values("date", ascending=False).head(int(n))
    for _, row in frame.iterrows():
        opp = str(row.get("opp_teamname", row.get("opponent", "-")) or "-")
        league_name = str(row.get("league", "-") or "-")
        val_disp = _market_metric_display(value_col, row.get(value_col))
        rows.append(f"{opp} - {val_disp} | {league_name}")
    return rows


def _market_h2h_rows(df: pd.DataFrame, team1: str, team2: str, value_col: str, n: int = 10) -> List[str]:
    rows: List[str] = []
    if df is None or df.empty:
        return rows
    frame = df.sort_values("date", ascending=False).head(int(n))
    for _, row in frame.iterrows():
        league_name = str(row.get("league", "-") or "-")
        val_disp = _market_metric_display(value_col, row.get(value_col))
        rows.append(f"{team1} vs {team2} - {val_disp} | {league_name}")
    return rows


def _render_market_context_panel(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, title: str, value_col: str) -> None:
    current_team1 = dataset.team_games(year, league, team1)
    current_team2 = dataset.team_games(year, league, team2)
    full_year_team1 = dataset.team_games(year, "(Todas)", team1)
    full_year_team2 = dataset.team_games(year, "(Todas)", team2)
    full_year_h2h = dataset.pair_games(year, "(Todas)", team1, team2)

    current_a = _market_block(current_team1, value_col)
    current_b = _market_block(current_team2, value_col)
    full_a = _market_block(full_year_team1, value_col)
    full_b = _market_block(full_year_team2, value_col)

    def _overall_df(block_a: dict, block_b: dict) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"Métrica": f"Avg {title}", team1: _market_metric_display(value_col, block_a["avg"]), team2: _market_metric_display(value_col, block_b["avg"])},
                {"Métrica": "# Games", team1: int(block_a["games"]), team2: int(block_b["games"])},
                {"Métrica": "Record", team1: block_a["record"], team2: block_b["record"]},
            ]
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Overall</div>", unsafe_allow_html=True)
        st.caption("Recorte atual")
        st.dataframe(_overall_df(current_a, current_b), hide_index=True, use_container_width=True)
        st.caption("Ano todo")
        st.dataframe(_overall_df(full_a, full_b), hide_index=True, use_container_width=True)
    with col2:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Recent Form</div>", unsafe_allow_html=True)
        st.write(f"**{team1}**")
        st.caption(_recent_form_caption(current_team1))
        for row in _market_recent_rows(full_year_team1, value_col, 10):
            st.markdown(f"- {row}")
        st.divider()
        st.write(f"**{team2}**")
        st.caption(_recent_form_caption(current_team2))
        for row in _market_recent_rows(full_year_team2, value_col, 10):
            st.markdown(f"- {row}")
    with col3:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Past Faceoffs</div>", unsafe_allow_html=True)
        st.metric("H2H Games", int(len(full_year_h2h)))
        if not full_year_h2h.empty:
            avg_h2h = float(pd.to_numeric(full_year_h2h.get(value_col), errors="coerce").mean()) if value_col in full_year_h2h.columns else float("nan")
            st.markdown(f"<div class='gabi-card'><b>Avg {title}</b><br>{_market_metric_display(value_col, avg_h2h)}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            for row in _market_h2h_rows(full_year_h2h, team1, team2, value_col, 10):
                st.markdown(f"- {row}")
        else:
            st.info("Sem H2H no recorte anual completo.")


def _saved_game_label(item: dict) -> str:
    event_name = str(item.get("event_name") or item.get("league") or "").strip()
    match_format = str(item.get("match_format") or "").strip()
    parts = [str(item.get("updated_at", "") or "").strip()]
    if event_name:
        parts.append(event_name)
    matchup = f"{item.get('team1', '')} x {item.get('team2', '')}".strip()
    if matchup:
        parts.append(matchup)
    if match_format:
        parts.append(match_format)
    return " | ".join([p for p in parts if p])


def _render_saved_games(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    saved = load_saved_games()
    with st.expander("Selecionar jogo salvo (Agenda)", expanded=False):
        if saved:
            ordered = sorted(saved.values(), key=lambda item: item.get("updated_at", ""), reverse=True)
            options = {_saved_game_label(item): item["key"] for item in ordered}
            selected_label = st.selectbox("Jogo salvo", [""] + list(options.keys()), key="saved_game_label")
            selected_key = options.get(selected_label, "")
        else:
            selected_key = ""
            st.info("Nenhum confronto salvo ainda.")
        rec = saved.get(selected_key) if selected_key else None
        if rec:
            chips = []
            if rec.get("event_name"):
                chips.append(f"Campeonato: {rec.get('event_name')}")
            if rec.get("match_format"):
                chips.append(f"Formato: {rec.get('match_format')}")
            if rec.get("league"):
                chips.append(f"Liga analítica: {rec.get('league')}")
            if rec.get("year") is not None:
                chips.append(f"Temporada: {rec.get('year')}")
            chips.append(f"Linhas: {_lines_status_label(LinesConfig.from_dict(rec.get('lines') or {}))}")
            chips.append(f"Atualizado: {rec.get('updated_at') or '-'}")
            st.markdown("".join(f"<span class='gabi-chip'>{chip}</span>" for chip in chips), unsafe_allow_html=True)
            prefix = f"dota_ag_{str(selected_key)}_"
            _init_prefixed_lines(prefix, LinesConfig.from_dict(rec.get("lines") or {}))
            with st.expander("Linhas deste jogo (opcional) — salvar e/ou analisar", expanded=False):
                st.caption("As linhas salvas aqui acompanham o jogo na agenda e entram junto quando você usar ou analisar este confronto.")
                _render_prefixed_line_editor(prefix)
                s1, s2 = st.columns(2)
                save_lines = s1.button("Salvar linhas", use_container_width=True, key=f"{prefix}save")
                analyze = s2.button("Analisar confronto", type="primary", use_container_width=True, key=f"{prefix}analyze")
                if save_lines or analyze:
                    rec["lines"] = _prefixed_lines_from_state(prefix).to_dict()
                    saved[selected_key] = rec
                    save_saved_games(saved)
                    if save_lines:
                        st.success("Linhas salvas para este jogo.")
                        st.rerun()
                    if analyze:
                        lines_obj = LinesConfig.from_dict(rec.get("lines") or {})
                        st.session_state["dota_year"] = rec.get("year")
                        st.session_state["dota_league"] = rec.get("league") or "(Todas)"
                        st.session_state["dota_team1"] = rec.get("team1") or ""
                        st.session_state["dota_team2"] = rec.get("team2") or ""
                        st.session_state["dota_event_name"] = rec.get("event_name") or ""
                        st.session_state["dota_match_format"] = rec.get("match_format") or "BO3"
                        _apply_lines_to_state(lines_obj)
                        st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("Usar este jogo", use_container_width=True, disabled=not selected_key):
            rec = saved.get(selected_key)
            if rec:
                lines_obj = LinesConfig.from_dict(rec.get("lines") or {})
                st.session_state["dota_year"] = rec.get("year")
                st.session_state["dota_league"] = rec.get("league") or "(Todas)"
                st.session_state["dota_team1"] = rec.get("team1") or ""
                st.session_state["dota_team2"] = rec.get("team2") or ""
                st.session_state["dota_event_name"] = rec.get("event_name") or ""
                st.session_state["dota_match_format"] = rec.get("match_format") or "BO3"
                _apply_lines_to_state(lines_obj)
                st.rerun()
        if c2.button("Salvar confronto atual", use_container_width=True):
            if team1 and team2 and team1 != team2:
                records = upsert_saved_game(
                    saved,
                    year,
                    league,
                    team1,
                    team2,
                    _lines_from_state(),
                    event_name=st.session_state.get("dota_event_name", ""),
                    match_format=st.session_state.get("dota_match_format", ""),
                )
                save_saved_games(records)
                st.success("Confronto salvo.")
                st.rerun()
            else:
                st.warning("Selecione Time 1 e Time 2 antes de salvar.")
        if c3.button("Excluir selecionado", use_container_width=True, disabled=not selected_key):
            if selected_key in saved:
                saved.pop(selected_key, None)
                save_saved_games(saved)
                st.success("Confronto excluído.")
                st.rerun()
        if rec:
            current_lines = _prefixed_lines_from_state(prefix)
            st.caption(f"Status deste jogo salvo: {_lines_status_label(current_lines)}.")
        st.divider()
        st.markdown("**Adicionar jogo à agenda**")
        st.caption("Cadastre manualmente campeonato, times e formato para usar depois no fluxo principal.")
        all_teams = [""] + dataset.teams_for_scope(year, "(Todas)")
        m1, m2, m3, m4 = st.columns([1.4, 1.4, 1.2, 1])
        m1.text_input("Campeonato", key="dota_manual_event_name", placeholder="ex: DreamLeague Season 26")
        m2.selectbox("Time 1", all_teams, key="dota_manual_team1")
        m3.selectbox("Time 2", all_teams, key="dota_manual_team2")
        m4.selectbox("Formato", ["BO1", "BO2", "BO3", "BO5"], key="dota_manual_format")
        if st.button("Salvar jogo manual", use_container_width=True, key="dota_save_manual_game"):
            manual_team1 = str(st.session_state.get("dota_manual_team1") or "").strip()
            manual_team2 = str(st.session_state.get("dota_manual_team2") or "").strip()
            manual_event = str(st.session_state.get("dota_manual_event_name") or "").strip()
            manual_format = str(st.session_state.get("dota_manual_format") or "BO3").strip()
            if not manual_team1 or not manual_team2 or manual_team1 == manual_team2:
                st.warning("Selecione Time 1 e Time 2 válidos para salvar o jogo manual.")
            else:
                records = upsert_saved_game(
                    saved,
                    year,
                    "(Todas)",
                    manual_team1,
                    manual_team2,
                    LinesConfig(),
                    event_name=manual_event,
                    match_format=manual_format,
                )
                save_saved_games(records)
                st.success("Jogo manual salvo.")
                st.session_state["dota_team1"] = manual_team1
                st.session_state["dota_team2"] = manual_team2
                st.session_state["dota_event_name"] = manual_event
                st.session_state["dota_match_format"] = manual_format
                st.rerun()


def _render_line_editor() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Kills (ex: 45.5, 49.5)", key="line_total_kills")
        st.text_input("Torres (ex: 15.5, 16.5)", key="line_total_towers")
    with c2:
        st.text_input("Roshans (ex: 1.5, 2.5)", key="line_total_roshans")
        st.text_input("Barracks (ex: 3.5, 4.5)", key="line_total_barracks")
    with c3:
        st.text_input("Tempo (ex: 38, 38.5)", key="line_total_time")

    with st.expander("Extras (opcional)", expanded=False):
        st.caption("Handicap: usa Time 1 como lado base. Use .5 para evitar push.")
        st.text_input("HC Kills (Time 1) (ex: -4.5)", key="line_hc_kills")
        st.text_input("HC Torres (Time 1) (ex: -2.5)", key="line_hc_towers")
        st.text_input("HC Roshans (Time 1) (ex: -0.5)", key="line_hc_roshans")
        st.text_input("HC Barracks (Time 1) (ex: -1.5)", key="line_hc_barracks")


def _render_prefixed_line_editor(prefix: str) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Kills (ex: 45.5, 49.5)", key=prefix + "line_total_kills")
        st.text_input("Torres (ex: 15.5, 16.5)", key=prefix + "line_total_towers")
    with c2:
        st.text_input("Roshans (ex: 1.5, 2.5)", key=prefix + "line_total_roshans")
        st.text_input("Barracks (ex: 3.5, 4.5)", key=prefix + "line_total_barracks")
    with c3:
        st.text_input("Tempo (ex: 38, 38.5)", key=prefix + "line_total_time")

    with st.expander("Extras (opcional)", expanded=False):
        st.caption("Handicap: usa Time 1 como lado base. Use .5 para evitar push.")
        st.text_input("HC Kills (Time 1) (ex: -4.5)", key=prefix + "line_hc_kills")
        st.text_input("HC Torres (Time 1) (ex: -2.5)", key=prefix + "line_hc_towers")
        st.text_input("HC Roshans (Time 1) (ex: -0.5)", key=prefix + "line_hc_roshans")
        st.text_input("HC Barracks (Time 1) (ex: -1.5)", key=prefix + "line_hc_barracks")


def _available_lines(lines: LinesConfig, market: str) -> List[float]:
    mapping = {
        "Kills": lines.total_kills,
        "Torres": lines.total_towers,
        "Roshans": lines.total_roshans,
        "Barracks": lines.total_barracks,
        "Tempo": lines.total_time_min,
        "HC Kills": lines.hc_kills_t1,
        "HC Torres": lines.hc_towers_t1,
        "HC Roshans": lines.hc_roshans_t1,
        "HC Barracks": lines.hc_barracks_t1,
    }
    return mapping.get(market, [])


def _render_consistency(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, lines: LinesConfig) -> None:
    _render_context_bar(year, league, team1, team2, [f"Laplace: {'On' if st.session_state['dota_laplace'] else 'Off'}"])
    market = st.radio(
        "Mercado",
        ["Win Rate", "Kills", "Torres", "Roshans", "Barracks", "Tempo", "HC Kills", "HC Torres", "HC Roshans", "HC Barracks"],
        key="dota_consistency_market",
        horizontal=True,
    )
    if market == "Win Rate":
        result = analyze_winner_market(dataset, year, league, team1, team2, st.session_state["dota_laplace"])
        summary_cards = st.columns(5)
        summary_cards[0].metric(team1, _fmt_pct(float(result["model_t1_prob"])))
        summary_cards[1].metric(team2, _fmt_pct(float(result["model_t2_prob"])))
        summary_cards[2].metric("Sinal ML", str(result.get("ml_signal_label", "-")))
        summary_cards[3].metric("Jogos Time 1", int(result["team1_games"]))
        summary_cards[4].metric("H2H", int(result["h2h_games"]))
        signal_detail = str(result.get("ml_signal_detail", "") or "")
        left, right = st.columns(2)
        with left:
            st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Modelo</div>", unsafe_allow_html=True)
            st.caption(f"Probabilidade final do vencedor. Fonte ativa: {result.get('model_source', 'Heuristico')}.")
            _render_ml_signal(str(result.get("ml_signal_label", "-")), signal_detail)
            model_df = pd.DataFrame(
                [
                    {
                        "Lado": team1,
                        "Prob. Final": _fmt_pct(float(result["model_t1_prob"])),
                        "Odd Final": f"{fair_odds(float(result['model_t1_prob'])):.2f}",
                        "Heur.": _fmt_pct(float(result["heuristic_t1_prob"])),
                        "ML": _fmt_pct(float(result["ml_t1_prob"])) if result.get("ml_t1_prob") is not None else "-",
                        "Forma": _fmt_pct(float(result["team1_win_rate"])),
                        "H2H": _fmt_pct(float(result["h2h_win_rate_t1"])) if int(result["h2h_games"]) > 0 else "-",
                    },
                    {
                        "Lado": team2,
                        "Prob. Final": _fmt_pct(float(result["model_t2_prob"])),
                        "Odd Final": f"{fair_odds(float(result['model_t2_prob'])):.2f}",
                        "Heur.": _fmt_pct(float(result["heuristic_t2_prob"])),
                        "ML": _fmt_pct(float(result["ml_t2_prob"])) if result.get("ml_t2_prob") is not None else "-",
                        "Forma": _fmt_pct(float(result["team2_win_rate"])),
                        "H2H": _fmt_pct(1.0 - float(result["h2h_win_rate_t1"])) if int(result["h2h_games"]) > 0 else "-",
                    },
                ]
            )
            st.dataframe(model_df, hide_index=True, use_container_width=True)
            ml_auc = result.get("ml_holdout_auc")
            ml_brier = result.get("ml_holdout_brier")
            ml_baseline_brier = result.get("ml_baseline_brier")
            ml_parts = [
                f"ML pronto: {'sim' if result.get('ml_ready') else 'nao'}",
                f"Blend ML: {float(result.get('ml_blend_weight', 0.0)) * 100:.0f}%",
            ]
            if ml_auc is not None:
                ml_parts.append(f"AUC holdout: {float(ml_auc):.3f}")
            if ml_brier is not None:
                ml_parts.append(f"Brier holdout: {float(ml_brier):.3f}")
            if ml_baseline_brier is not None:
                ml_parts.append(f"Brier baseline: {float(ml_baseline_brier):.3f}")
            if result.get("ml_model_label"):
                ml_parts.append(f"Modelo: {result.get('ml_model_label')}")
            if result.get("ml_reason") and str(result.get("ml_reason")) != "ok":
                ml_parts.append(f"Status: {result.get('ml_reason')}")
            st.caption(" | ".join(ml_parts))
        with right:
            st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Histórico</div>", unsafe_allow_html=True)
            st.caption("Laplace por recorte para o vencedor nos mesmos recortes do confronto.")
            period_labels = ["Ano todo", "Últimos 15", "Últimos 10", "Últimos 5"]
            hist_rows = []
            for row_key, label in [
                ("league_t1", f"Liga - {team1}"),
                ("league_t2", f"Liga - {team2}"),
                ("h2h_t1", f"H2H - {team1}"),
                ("h2h_t2", f"H2H - {team2}"),
            ]:
                hist_rows.append(
                    {
                        "Recorte": label,
                        **{period_label: result["periods"].get(period_label, {}).get(row_key, "-") for period_label in period_labels},
                    }
                )
            st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)
        return
    market_panel_map = {
        "Kills": ("Kills", "totalkills"),
        "Torres": ("Torres", "totaltowers"),
        "Roshans": ("Roshans", "totalroshans"),
        "Barracks": ("Barracks", "totalbarracks"),
        "Tempo": ("Tempo", "time_min"),
        "HC Kills": ("Diff Kills", "diff_kills"),
        "HC Torres": ("Diff Torres", "diff_towers"),
        "HC Roshans": ("Diff Roshans", "diff_roshans"),
        "HC Barracks": ("Diff Barracks", "diff_barracks"),
    }
    panel_meta = market_panel_map.get(market)
    if panel_meta:
        _render_market_context_panel(dataset, year, league, team1, team2, panel_meta[0], panel_meta[1])
        st.divider()
    line_values = _available_lines(lines, market)
    if not line_values:
        st.info("Defina linhas para esse mercado no bloco 'Linhas de análise'.")
        return
    is_hc = market.startswith("HC ")
    base_market = market.replace("HC ", "")
    line_selected = st.selectbox(
        "Linha",
        line_values,
        format_func=lambda value: fmt_mmss_from_minutes(value) if market == "Tempo" else f"{value:g}",
        key=f"line_{market}",
    )
    line_label = fmt_mmss_from_minutes(line_selected) if market == "Tempo" else (f"{line_selected:+g}" if is_hc else f"{line_selected:g}")
    st.caption(f"Mercado ativo: {market} | Linha: {line_label}")
    if is_hc:
        result = analyze_handicap_market(dataset, year, league, team1, team2, base_market, line_selected, st.session_state["dota_laplace"])
        summary_cards = st.columns(4)
        summary_cards[0].metric("Linha", f"{line_selected:+g}")
        summary_cards[1].metric(team1, f"{result['model_t1_prob'] * 100:.1f}%")
        summary_cards[2].metric(team2, f"{result['model_t2_prob'] * 100:.1f}%")
        summary_cards[3].metric("Edge medio", f"{result['expected_edge']:.2f}")
        left, right = st.columns(2)
        with left:
            st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Modelo</div>", unsafe_allow_html=True)
            st.caption("Probabilidade justa do handicap no recorte ativo.")
            st.dataframe(pd.DataFrame([{
                "Linha": f"{line_selected:+g}",
                team1: f"{_fmt_pct(result['model_t1_prob'])} | {fair_odds(result['model_t1_prob']):.2f}",
                team2: f"{_fmt_pct(result['model_t2_prob'])} | {fair_odds(result['model_t2_prob']):.2f}",
                "Edge medio": f"{result['expected_edge']:.2f}",
                "SD": f"{result['sd']:.2f}",
            }]), hide_index=True, use_container_width=True)
        with right:
            st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Histórico</div>", unsafe_allow_html=True)
            st.caption("Cobertura histórica do handicap nos mesmos recortes.")
            period_labels = ["Ano todo", "Últimos 15", "Últimos 10", "Últimos 5"]
            hist_table = pd.DataFrame([
                {"Lado": team1, **{label: result["periods"][label]["t1_cover"] for label in period_labels}},
                {"Lado": team2, **{label: result["periods"][label]["t2_cover"] for label in period_labels}},
            ])
            st.dataframe(hist_table, hide_index=True, use_container_width=True)
        return

    result = analyze_total_market(dataset, year, league, team1, team2, base_market, line_selected, st.session_state["dota_laplace"])
    summary_cards = st.columns(4)
    summary_cards[0].metric("Media esperada", fmt_mmss_from_minutes(result["expected_mean"]) if base_market == "Tempo" else f"{result['expected_mean']:.2f}")
    summary_cards[1].metric("Linha", fmt_mmss_from_minutes(line_selected) if base_market == "Tempo" else f"{line_selected:g}")
    summary_cards[2].metric("Acima", f"{result['model_over_prob'] * 100:.1f}%")
    summary_cards[3].metric("Abaixo", f"{result['model_under_prob'] * 100:.1f}%")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Modelo</div>", unsafe_allow_html=True)
        st.caption("Media esperada e odds justas do motor para a linha ativa.")
        model_df = pd.DataFrame([
            {
                "Media esperada": fmt_mmss_from_minutes(result["expected_mean"]) if base_market == "Tempo" else f"{result['expected_mean']:.2f}",
                "SD": f"{result['sd']:.2f}",
                "Linha": fmt_mmss_from_minutes(line_selected) if base_market == "Tempo" else f"{line_selected:g}",
                "Acima (%)": _fmt_pct(result["model_over_prob"]),
                "Odd Acima": f"{fair_odds(result['model_over_prob']):.2f}",
                "Abaixo (%)": _fmt_pct(result["model_under_prob"]),
                "Odd Abaixo": f"{fair_odds(result['model_under_prob']):.2f}",
            }
        ])
        st.dataframe(model_df, hide_index=True, use_container_width=True)
    with right:
        st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Histórico</div>", unsafe_allow_html=True)
        st.caption("Laplace por recorte para as mesmas linhas, com foco em coerência e sanidade.")
        hist_rows = []
        period_labels = ["Ano todo", "Últimos 15", "Últimos 10", "Últimos 5"]
        for row_key, label in [
            ("league_over", "Liga - Over"),
            ("league_under", "Liga - Under"),
            ("h2h_over", "H2H - Over"),
            ("h2h_under", "H2H - Under"),
        ]:
            hist_rows.append({
                "Recorte": label,
                **{period_label: result["periods"][period_label][row_key] for period_label in period_labels},
            })
        st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)


def _render_stats(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Stats por campeonato"])
    championship_options = dataset.championships_for_matchup(year, team1, team2)
    if league and league != "(Todas)" and league not in championship_options:
        championship_options.insert(0, league)
    if "(Todas)" not in championship_options:
        championship_options = ["(Todas)"] + championship_options
    if not st.session_state.get("dota_stats_championship"):
        st.session_state["dota_stats_championship"] = league if league and league != "(Todas)" else championship_options[0]
    st.markdown("<div class='gabi-kicker'>Leitura</div><div class='gabi-section'>Stats por campeonato</div>", unsafe_allow_html=True)
    st.caption("Tabela agregada por time no recorte ativo, com campeonato selecionável e linha dinâmica.")
    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1])
    championship = c1.selectbox("Campeonato", championship_options, key="dota_stats_championship")
    c2.selectbox("Side", ["All", "Radiant", "Dire"], key="dota_stats_side")
    c3.selectbox("Win/loss", ["All", "Win", "Loss"], key="dota_stats_result")
    c4.selectbox("Linha dinamica", list(TOTAL_MARKETS.keys()), key="dota_stats_market")
    custom_col1, custom_col2 = st.columns([1, 3])
    custom_col1.text_input("Linha custom", key="dota_stats_custom_line", placeholder="ex: 50.5")
    custom_line = None
    try:
        if st.session_state.get("dota_stats_custom_line"):
            custom_line = float(str(st.session_state["dota_stats_custom_line"]).replace(",", "."))
    except ValueError:
        custom_line = None
    line_origin = f"Linha custom: {custom_line:g}" if custom_line is not None else "Linha do confronto: padrao do mercado"
    _render_context_bar(
        year,
        league,
        team1,
        team2,
        [
            f"Campeonato: {championship}",
            f"Side: {st.session_state['dota_stats_side']}",
            f"Win/loss: {st.session_state['dota_stats_result']}",
            f"Mercado: {st.session_state['dota_stats_market']}",
            line_origin,
        ],
    )
    stats_df = build_stats_table(
        dataset,
        year,
        None if championship == "(Todas)" else championship,
        st.session_state["dota_stats_side"],
        st.session_state["dota_stats_result"],
        st.session_state["dota_stats_market"],
        custom_line,
    )
    if stats_df.empty:
        st.info("Sem dados para esse campeonato/recorte.")
        return
    stats_df = stats_df.copy()
    if "Team" in stats_df.columns:
        stats_df.insert(
            0,
            "Spot",
            stats_df["Team"].map({team1: "Time 1", team2: "Time 2"}).fillna(""),
        )
    if "Game Time" in stats_df.columns:
        stats_df["Game Time"] = stats_df["Game Time"].apply(lambda v: fmt_mmss_from_minutes(v / 60.0) if pd.notna(v) else "-")
    stat_cards = st.columns(4)
    stat_cards[0].metric("Times", int(len(stats_df)))
    if "# Games" in stats_df.columns:
        stat_cards[1].metric("Jogos agregados", int(pd.to_numeric(stats_df["# Games"], errors="coerce").fillna(0).sum()))
    if team1 in stats_df.get("Team", pd.Series([], dtype=str)).tolist():
        t1_row = stats_df[stats_df["Team"] == team1].head(1)
        stat_cards[2].metric(team1, f"{float(t1_row['Win Rate'].iloc[0]):.1f}%" if not t1_row.empty and 'Win Rate' in t1_row.columns else "-")
    if team2 in stats_df.get("Team", pd.Series([], dtype=str)).tolist():
        t2_row = stats_df[stats_df["Team"] == team2].head(1)
        stat_cards[3].metric(team2, f"{float(t2_row['Win Rate'].iloc[0]):.1f}%" if not t2_row.empty and 'Win Rate' in t2_row.columns else "-")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)


def _render_details(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Detalhes dos jogos"])
    st.markdown("<div class='gabi-kicker'>Auditoria</div><div class='gabi-section'>Detalhes dos Jogos</div>", unsafe_allow_html=True)
    st.caption("Use o modo simples para decisão rápida e auditoria para inspeção completa.")
    details = detail_tables(dataset, year, league, team1, team2)
    view = st.radio("Modo", ["Simples", "Auditoria"], horizontal=True, key="dota_detail_mode")

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [
            "date", "league", "opp_teamname", "result", "side", "teamkills", "oppkills",
            "totalkills", "towers", "roshans", "barracks", "time_min"
        ]
        out = df[[c for c in cols if c in df.columns]].copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "result" in out.columns:
            out["result"] = out["result"].map({1: "W", 0: "L"}).fillna("-")
        if "time_min" in out.columns:
            out["time_min"] = out["time_min"].apply(fmt_mmss_from_minutes)
        rename = {
            "date": "Data",
            "league": "Campeonato",
            "opp_teamname": "Oponente",
            "result": "R",
            "side": "Side",
            "teamkills": "K",
            "oppkills": "KA",
            "totalkills": "TK",
            "towers": "Torres",
            "roshans": "Roshans",
            "barracks": "Barracks",
            "time_min": "Tempo",
        }
        return out.rename(columns=rename)

    def prep_h2h(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out["time_min"] = out["time_min"].apply(fmt_mmss_from_minutes)
        return out[["date", "league", "kills_t1", "kills_t2", "totalkills", "towers_t1", "towers_t2", "roshans_t1", "roshans_t2", "barracks_t1", "barracks_t2", "time_min"]].rename(columns={
            "date": "Data",
            "league": "Campeonato",
            "kills_t1": f"K {team1}",
            "kills_t2": f"K {team2}",
            "totalkills": "TK",
            "towers_t1": f"T {team1}",
            "towers_t2": f"T {team2}",
            "roshans_t1": f"R {team1}",
            "roshans_t2": f"R {team2}",
            "barracks_t1": f"B {team1}",
            "barracks_t2": f"B {team2}",
            "time_min": "Tempo",
        })
    base_options = {
        f"{team1} - Ano todo (todos confrontos)": prep(details["team1"]),
        f"{team2} - Ano todo (todos confrontos)": prep(details["team2"]),
        f"H2H - Ano todo ({team1} x {team2})": prep_h2h(details["h2h"]),
    }
    base_name = st.selectbox("Base", list(base_options.keys()), index=0, key="dota_detail_base")
    df_show = base_options.get(base_name, pd.DataFrame())
    if df_show.empty:
        st.info("Base vazia para este recorte.")
        return
    st.dataframe(df_show if view == "Auditoria" else df_show.head(12), use_container_width=True, hide_index=True)


def _render_quick_selector(summary_df: pd.DataFrame) -> None:
    _render_context_bar(
        st.session_state.get("dota_year"),
        st.session_state.get("dota_league", "(Todas)"),
        st.session_state.get("dota_team1", ""),
        st.session_state.get("dota_team2", ""),
        ["Leitura: Seletor Rápido"],
    )
    st.markdown("<div class='gabi-kicker'>Operação</div><div class='gabi-section'>Seletor Rápido</div>", unsafe_allow_html=True)
    st.caption("Filtre o cenário operacional por odd, mercado, lado e sinal do ML antes de inspecionar a tabela consolidada.")
    if summary_df.empty:
        st.info("Defina linhas para gerar o Seletor Rápido.")
        return
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.5, 1.2])
    max_odd = c1.slider("Odd maxima do modelo", min_value=1.01, max_value=10.0, value=float(st.session_state["dota_summary_max_odd"]), step=0.01, key="dota_summary_max_odd")
    market_options = ["(Todos)"] + sorted(summary_df["Mercado"].dropna().astype(str).unique().tolist()) if "Mercado" in summary_df.columns else ["(Todos)"]
    side_options = ["(Todos)"] + sorted(summary_df["Lado"].dropna().astype(str).unique().tolist()) if "Lado" in summary_df.columns else ["(Todos)"]
    signal_options = ["(Todos)"] + sorted(
        [value for value in summary_df.get("Sinal ML", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if value and value != "-"]
    )
    market_sel = c2.selectbox("Mercado", market_options, index=0, key="dota_quick_market")
    side_sel = c3.selectbox("Lado", side_options, index=0, key="dota_quick_side")
    signal_sel = c4.selectbox("Sinal ML", signal_options, index=0, key="dota_quick_signal")
    quick_df = summary_df.copy()
    quick_df["Odd Modelo Num"] = pd.to_numeric(quick_df["Odd (Modelo)"], errors="coerce")
    quick_df = quick_df[quick_df["Odd Modelo Num"] <= max_odd].drop(columns=["Odd Modelo Num"])
    if market_sel != "(Todos)" and "Mercado" in quick_df.columns:
        quick_df = quick_df[quick_df["Mercado"].astype(str) == str(market_sel)].copy()
    if side_sel != "(Todos)" and "Lado" in quick_df.columns:
        quick_df = quick_df[quick_df["Lado"].astype(str) == str(side_sel)].copy()
    if signal_sel != "(Todos)" and "Sinal ML" in quick_df.columns:
        quick_df = quick_df[quick_df["Sinal ML"].astype(str) == str(signal_sel)].copy()
    cards = st.columns(4)
    cards[0].metric("Linhas", int(len(quick_df)))
    if "Mercado" in quick_df.columns:
        cards[1].metric("Mercados", int(quick_df["Mercado"].nunique()))
    if "Odd (Hist)" in quick_df.columns:
        hist_nonempty = int((quick_df["Odd (Hist)"].astype(str).str.strip() != "-").sum())
        cards[2].metric("Com hist", hist_nonempty)
    cards[3].metric("Filtro odd", f"{max_odd:.2f}")
    st.caption(f"Configuração: mercado={market_sel} | lado={side_sel} | sinal={signal_sel}")
    preferred_cols = [
        "Mercado",
        "Lado",
        "Linha",
        "Confianca (Score)",
        "Laplace (Liga)",
        "Laplace (H2H)",
        "Odd (Modelo)",
        "Odd (ML)",
        "Sinal ML",
        "Fonte",
        "Stats",
        "Odd (Hist)",
        "Odd (Mix)",
    ]
    display_cols = [col for col in preferred_cols if col in quick_df.columns]
    st.dataframe(quick_df[display_cols] if display_cols else quick_df, use_container_width=True, hide_index=True)


def _render_summary(year: Optional[int], league: str, team1: str, team2: str, summary_df: pd.DataFrame) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Resumo final"])
    st.markdown("### Resumo de Odds")
    st.caption(
        "Mostra as odds do Modelo final que passam no filtro. Em ML (Partida), a coluna ML exibe a odd bruta do "
        "classificador supervisionado, Fonte informa se houve blend com o heuristico e Sinal ML resume a qualidade operacional."
    )
    if summary_df.empty:
        st.info("Defina linhas para gerar o resumo.")
        return
    with st.container(border=True):
        max_odd = st.number_input("Odd maxima (Resumo)", min_value=1.01, max_value=10.0, step=0.01, value=float(st.session_state["dota_summary_max_odd"]), key="dota_summary_max_odd_resumo")
        summary_view = summary_df.copy()
        summary_view["Odd Modelo Num"] = pd.to_numeric(summary_view["Odd (Modelo)"], errors="coerce")
        summary_view = summary_view[summary_view["Odd Modelo Num"] <= float(max_odd)].drop(columns=["Odd Modelo Num"])
        c1, c2, c3 = st.columns(3)
        with c1:
            req_liga = st.checkbox("Exigir Liga", value=bool(st.session_state["dota_resumo_req_liga"]), key="dota_resumo_req_liga")
        with c2:
            req_h2h = st.checkbox("Exigir H2H", value=bool(st.session_state["dota_resumo_req_h2h"]), key="dota_resumo_req_h2h")
        with c3:
            req_model = st.checkbox("Exigir Modelo", value=bool(st.session_state["dota_resumo_req_model"]), key="dota_resumo_req_model")
        st.checkbox(
            "Mostrar odds do Histórico (Laplace) no Resumo",
            value=bool(st.session_state["dota_resumo_show_laplace_cols"]),
            key="dota_resumo_show_laplace_cols",
        )
        if req_liga and "Laplace (Liga)" in summary_view.columns:
            summary_view = summary_view[summary_view["Laplace (Liga)"].astype(str).str.strip() != "-"].copy()
        if req_h2h and "Laplace (H2H)" in summary_view.columns:
            summary_view = summary_view[summary_view["Laplace (H2H)"].astype(str).str.strip() != "-"].copy()
        if req_model and "Odd (Modelo)" in summary_view.columns:
            summary_view = summary_view[summary_view["Odd (Modelo)"].astype(str).str.strip() != "-"].copy()
        cards = st.columns(4)
        cards[0].metric("Linhas", int(len(summary_view)))
        if "Mercado" in summary_view.columns:
            cards[1].metric("Mercados", int(summary_view["Mercado"].nunique()))
        if "Lado" in summary_view.columns:
            cards[2].metric("Lados", int(summary_view["Lado"].nunique()))
        cards[3].metric("Odd maxima", f"{float(max_odd):.2f}")
    if summary_view.empty:
        st.info('Nada passou no filtro com as linhas atuais. Dica: ajuste a odd maxima ou desmarque "Exigir H2H"/"Exigir Liga" para diagnosticar.')
        return
    preferred_cols = [
        "Mercado",
        "Lado",
        "Linha",
        "Confianca (Score)",
        "Laplace (Liga)",
        "Laplace (H2H)",
        "Odd (Modelo)",
        "Odd (ML)",
        "Sinal ML",
        "Fonte",
        "Stats",
        "Odd (Hist)",
        "Odd (Mix)",
    ]
    if not bool(st.session_state.get("dota_resumo_show_laplace_cols", True)):
        preferred_cols = [col for col in preferred_cols if col not in {"Laplace (Liga)", "Laplace (H2H)", "Odd (Hist)"}]
    display_cols = [col for col in preferred_cols if col in summary_view.columns]
    st.dataframe(summary_view[display_cols] if display_cols else summary_view, use_container_width=True, hide_index=True)
    if "Confianca (Score)" in summary_view.columns:
        with st.expander("O que é 'Confianca (Score)' no Resumo", expanded=False):
            st.caption(
                "Score (0-100) da robustez da pick. Combina tamanho de amostra Liga, amostra H2H e acordo entre "
                "Modelo e Histórico. Quanto mais alto, mais estável tende a ser a leitura."
            )


@st.cache_data(ttl=1800, show_spinner=False)
def _od_team_summary_cached(team_id: int) -> dict:
    return get_team_summary(team_id)


@st.cache_data(ttl=1800, show_spinner=False)
def _od_team_heroes_cached(team_id: int) -> List[dict]:
    return get_team_heroes(team_id)


@st.cache_data(ttl=1800, show_spinner=False)
def _od_team_players_cached(team_id: int) -> List[dict]:
    return get_team_players(team_id)


@st.cache_data(ttl=1800, show_spinner=False)
def _od_team_matches_cached(team_id: int) -> List[dict]:
    return get_team_matches(team_id)


@st.cache_data(ttl=1800, show_spinner=False)
def _od_player_heroes_cached(account_id: int) -> List[dict]:
    return get_player_heroes(account_id)


@st.cache_data(ttl=21600, show_spinner=False)
def _od_hero_stats_cached() -> List[dict]:
    return get_hero_stats()


def _normalize_hero_token(text: str) -> str:
    token = str(text or "").strip().lower()
    token = token.replace("'", "").replace("-", "").replace(".", "").replace(" ", "")
    return token


def _hero_catalog() -> tuple[Dict[int, dict], Dict[str, dict]]:
    hero_meta_rows = _od_hero_stats_cached()
    by_id = {int(item.get("id") or 0): item for item in hero_meta_rows if item.get("id") is not None}
    by_norm: Dict[str, dict] = {}
    for item in hero_meta_rows:
        norm = _normalize_hero_token(item.get("localized_name") or "")
        if norm:
            by_norm[norm] = item
    aliases = {
        "furion": "naturesprophet",
        "wisp": "io",
        "np": "naturesprophet",
        "am": "antimage",
        "qop": "queenofpain",
        "naix": "lifestealer",
        "bara": "spiritbreaker",
        "clock": "clockwerk",
        "wr": "windranger",
        "veno": "venomancer",
    }
    for alias, target in aliases.items():
        if target in by_norm:
            by_norm[alias] = by_norm[target]
    return by_id, by_norm


def _resolve_hero_text(text: str) -> Optional[dict]:
    _, by_norm = _hero_catalog()
    return by_norm.get(_normalize_hero_token(text))


def _hero_role_tags(hero: dict) -> List[str]:
    return [str(x) for x in (hero.get("roles") or [])]


def _player_name(account_id: int, fallback: str = "-") -> str:
    if account_id <= 0:
        return fallback
    return str(fallback or f"Player {account_id}")


def _infer_recent_lineup(team_id: int, team_name: str, sample_matches: int = 8) -> pd.DataFrame:
    players_ref = _od_team_players_cached(team_id)
    player_ref: Dict[int, dict] = {}
    for row in players_ref:
        try:
            account_id = int(row.get("account_id") or 0)
        except Exception:
            account_id = 0
        if account_id > 0:
            player_ref[account_id] = row

    matches = _od_team_matches_cached(team_id)[: max(1, int(sample_matches))]
    counts: Dict[int, dict] = {}
    for summary in matches:
        match_id = int(summary.get("match_id") or 0)
        if match_id <= 0:
            continue
        team_is_radiant = bool(summary.get("radiant"))
        payload = _od_match_cached(match_id)
        start_time = int(payload.get("start_time") or 0)
        for player in payload.get("players") or []:
            is_same_side = bool(player.get("isRadiant")) == team_is_radiant
            if not is_same_side:
                continue
            account_id = int(player.get("account_id") or 0)
            if account_id <= 0:
                continue
            ref = player_ref.get(account_id, {})
            bucket = counts.setdefault(
                account_id,
                {
                    "account_id": account_id,
                    "name": str(player.get("personaname") or player.get("name") or ref.get("name") or f"Player {account_id}"),
                    "appearances": 0,
                    "last_seen": 0,
                    "current_member": bool(ref.get("is_current_team_member")),
                    "team_games": int(ref.get("games_played") or 0),
                    "team_wins": int(ref.get("wins") or 0),
                },
            )
            bucket["appearances"] += 1
            bucket["last_seen"] = max(int(bucket["last_seen"]), start_time)
            if not bucket.get("name") or str(bucket["name"]).startswith("Player "):
                bucket["name"] = str(player.get("personaname") or player.get("name") or ref.get("name") or bucket["name"])
    if not counts:
        return pd.DataFrame()
    rows = pd.DataFrame(list(counts.values()))
    rows = rows.sort_values(["appearances", "current_member", "last_seen"], ascending=[False, False, False]).reset_index(drop=True)
    if "last_seen" in rows.columns:
        rows["last_seen_label"] = pd.to_datetime(rows["last_seen"], unit="s", utc=True, errors="coerce").dt.tz_convert(None).dt.strftime("%Y-%m-%d")
    return rows


def _player_top_heroes_table(account_id: int, limit: int = 5) -> pd.DataFrame:
    rows = _od_player_heroes_cached(account_id)
    by_id, _ = _hero_catalog()
    records: List[dict] = []
    for row in rows[: max(1, int(limit))]:
        hero_id = int(row.get("hero_id") or 0)
        games = int(row.get("games") or 0)
        wins = int(row.get("win") or 0)
        if games <= 0:
            continue
        meta = by_id.get(hero_id, {})
        records.append(
            {
                "Heroi": meta.get("localized_name") or f"Hero {hero_id}",
                "Games": games,
                "Wins": wins,
                "WR": f"{(wins / games) * 100.0:.1f}%",
                "Último uso": pd.to_datetime(int(row.get("last_played") or 0), unit="s", utc=True, errors="coerce").tz_convert(None).strftime("%Y-%m-%d") if row.get("last_played") else "-",
            }
        )
    return pd.DataFrame(records)


def _team_heroes_table(team_id: int, limit: int = 12) -> pd.DataFrame:
    rows = _od_team_heroes_cached(team_id)
    by_id, _ = _hero_catalog()
    records: List[dict] = []
    for row in rows[: max(1, int(limit))]:
        hero_id = int(row.get("hero_id") or 0)
        games = int(row.get("games_played") or 0)
        wins = int(row.get("wins") or 0)
        meta = by_id.get(hero_id, {})
        wr = (wins / games * 100.0) if games > 0 else float("nan")
        records.append(
            {
                "Heroi": row.get("localized_name") or meta.get("localized_name") or f"Hero {hero_id}",
                "Games": games,
                "Wins": wins,
                "WR": f"{wr:.1f}%" if math.isfinite(wr) else "-",
                "Roles": ", ".join(meta.get("roles") or []) if meta.get("roles") else "-",
                "Pro WR": f"{(float(meta.get('pro_win') or 0) / max(float(meta.get('pro_pick') or 0), 1)) * 100.0:.1f}%" if float(meta.get("pro_pick") or 0) > 0 else "-",
            }
        )
    return pd.DataFrame(records)


def _parse_hero_list(text: str) -> List[str]:
    tokens = re.split(r"[,\n;|]+", str(text or "").strip())
    return [str(token).strip() for token in tokens if str(token).strip()]


def _draft_team_table(team_id: int, heroes: List[str]) -> pd.DataFrame:
    team_rows = _od_team_heroes_cached(team_id)
    by_team_hero = {int(row.get("hero_id") or 0): row for row in team_rows}
    by_id, _ = _hero_catalog()
    out: List[dict] = []
    for token in heroes:
        hero = _resolve_hero_text(token)
        if not hero:
            out.append({"Heroi": token, "Status": "Nao reconhecido", "Games time": "-", "WR time": "-", "Roles": "-"})
            continue
        hero_id = int(hero.get("id") or 0)
        team_row = by_team_hero.get(hero_id, {})
        games = int(team_row.get("games_played") or 0)
        wins = int(team_row.get("wins") or 0)
        out.append(
            {
                "Heroi": hero.get("localized_name"),
                "Status": "OK",
                "Games time": games,
                "WR time": f"{(wins / games) * 100.0:.1f}%" if games > 0 else "-",
                "Roles": ", ".join(_hero_role_tags(hero)) or "-",
            }
        )
    return pd.DataFrame(out)


def _draft_summary(heroes: List[str], team_id: int) -> dict:
    team_rows = _od_team_heroes_cached(team_id)
    by_team_hero = {int(row.get("hero_id") or 0): row for row in team_rows}
    _, by_norm = _hero_catalog()
    resolved = []
    for token in heroes:
        hero = by_norm.get(_normalize_hero_token(token))
        if hero:
            resolved.append(hero)
    if not resolved:
        return {
            "recognized": 0,
            "recognized_ratio": 0.0,
            "comfort": 0.0,
            "pro_wr": 0.0,
            "roles": {},
            "attack_mix": {"Melee": 0, "Ranged": 0},
            "team_games_avg": 0.0,
            "confidence": 0.0,
        }
    comfort_scores = []
    pro_wrs = []
    team_games = []
    role_counts: Dict[str, int] = {}
    attack_mix = {"Melee": 0, "Ranged": 0}
    for hero in resolved:
        hero_id = int(hero.get("id") or 0)
        team_row = by_team_hero.get(hero_id, {})
        games = float(team_row.get("games_played") or 0)
        wins = float(team_row.get("wins") or 0)
        team_games.append(games)
        comfort_scores.append((wins + 1.0) / (games + 2.0) if games >= 0 else 0.5)
        pro_pick = float(hero.get("pro_pick") or 0)
        pro_win = float(hero.get("pro_win") or 0)
        pro_wrs.append((pro_win / pro_pick) if pro_pick > 0 else 0.5)
        for role in _hero_role_tags(hero):
            role_counts[role] = role_counts.get(role, 0) + 1
        attack_type = str(hero.get("attack_type") or "").strip().title()
        if attack_type in attack_mix:
            attack_mix[attack_type] += 1
    recognized_ratio = float(len(resolved) / max(len(heroes), 1))
    team_games_avg = float(sum(team_games) / max(len(team_games), 1)) if team_games else 0.0
    confidence = (0.6 * recognized_ratio) + (0.4 * min(team_games_avg / 12.0, 1.0))
    return {
        "recognized": len(resolved),
        "recognized_ratio": recognized_ratio,
        "comfort": sum(comfort_scores) / len(comfort_scores),
        "pro_wr": sum(pro_wrs) / len(pro_wrs),
        "roles": role_counts,
        "attack_mix": attack_mix,
        "team_games_avg": team_games_avg,
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def _draft_tags(summary: dict) -> str:
    roles = dict(summary.get("roles") or {})
    tags: List[str] = []
    if roles.get("Initiator", 0) + roles.get("Disabler", 0) >= 2:
        tags.append("controle")
    if roles.get("Carry", 0) >= 2:
        tags.append("duplo core")
    if roles.get("Pusher", 0) >= 1:
        tags.append("push")
    if roles.get("Durable", 0) >= 2:
        tags.append("frontline")
    if int((summary.get("attack_mix") or {}).get("Ranged", 0)) >= 3:
        tags.append("range")
    return ", ".join(tags) if tags else "sem leitura forte"


def _hero_focus_read(team1: str, team2: str, row_a: dict, row_b: dict) -> str:
    games_a = int(row_a.get("Games") or 0)
    games_b = int(row_b.get("Games") or 0)
    wr_a = str(row_a.get("WR") or "-")
    wr_b = str(row_b.get("WR") or "-")
    if games_a <= 0 and games_b <= 0:
        return "Sem base nos dois lados"
    if games_a > 0 and games_b <= 0:
        return f"Hero mais presente em {team1}"
    if games_b > 0 and games_a <= 0:
        return f"Hero mais presente em {team2}"
    if abs(games_a - games_b) >= 4:
        return f"Uso mais forte: {team1 if games_a > games_b else team2}"
    if wr_a != "-" and wr_b != "-" and wr_a != wr_b:
        return f"WR melhor: {team1 if wr_a > wr_b else team2}"
    return "Leitura equilibrada"


def _parse_pct_token(value: object) -> float:
    text = str(value or "").strip().replace("%", "").replace(",", ".")
    if not text or text == "-":
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def _hero_pool_style(df_team: pd.DataFrame) -> str:
    if df_team is None or df_team.empty or "Roles" not in df_team.columns:
        return "Sem base"
    role_tokens: List[str] = []
    for value in df_team["Roles"].astype(str).tolist():
        role_tokens.extend([token.strip() for token in value.split(",") if token.strip() and token.strip() != "-"])
    if not role_tokens:
        return "Sem leitura forte"
    counts = pd.Series(role_tokens).value_counts()
    top_role = str(counts.index[0]) if not counts.empty else ""
    mapping = {
        "Pusher": "Pool mais voltado a push",
        "Durable": "Pool com frontline forte",
        "Initiator": "Pool com iniciacao forte",
        "Disabler": "Pool de controle",
        "Carry": "Pool com carga de core",
        "Support": "Pool mais utilitario",
    }
    return mapping.get(top_role, f"Pool puxando para {top_role.lower()}")


def _hero_pool_confidence(df_team: pd.DataFrame) -> str:
    if df_team is None or df_team.empty or "Games" not in df_team.columns:
        return "Baixa"
    games = pd.to_numeric(df_team["Games"], errors="coerce").fillna(0)
    avg_games = float(games.head(5).mean()) if not games.empty else 0.0
    if avg_games >= 18:
        return "Alta"
    if avg_games >= 9:
        return "Media"
    return "Baixa"


def _lineup_confidence(df_team: pd.DataFrame) -> str:
    if df_team is None or df_team.empty:
        return "Baixa"
    appearances = pd.to_numeric(df_team.get("appearances"), errors="coerce").fillna(0)
    current = pd.to_numeric(df_team.get("current_member"), errors="coerce").fillna(0)
    score = (0.7 * float(min(appearances.mean() / 5.0, 1.0))) + (0.3 * float(current.mean()))
    if score >= 0.75:
        return "Alta"
    if score >= 0.45:
        return "Media"
    return "Baixa"


def _lineup_compare_read(team1: str, team2: str, lineup_a: pd.DataFrame, lineup_b: pd.DataFrame) -> str:
    conf_a = _lineup_confidence(lineup_a)
    conf_b = _lineup_confidence(lineup_b)
    order = {"Baixa": 0, "Media": 1, "Alta": 2}
    if order.get(conf_a, 0) == order.get(conf_b, 0):
        return "Lineups com confianca parecida"
    return f"Lineup mais estável: {team1 if order.get(conf_a, 0) > order.get(conf_b, 0) else team2}"


def _player_focus_compare(team1: str, team2: str, df_a: pd.DataFrame, df_b: pd.DataFrame) -> str:
    if df_a is None or df_a.empty or df_b is None or df_b.empty:
        return "Sem base comparativa suficiente"
    shared = sorted(set(df_a["Heroi"].astype(str)).intersection(set(df_b["Heroi"].astype(str))))
    avg_games_a = float(pd.to_numeric(df_a["Games"], errors="coerce").fillna(0).head(5).mean())
    avg_games_b = float(pd.to_numeric(df_b["Games"], errors="coerce").fillna(0).head(5).mean())
    avg_wr_a = float(pd.Series([_parse_pct_token(v) for v in df_a["WR"].head(5)]).dropna().mean()) if not df_a.empty else float("nan")
    avg_wr_b = float(pd.Series([_parse_pct_token(v) for v in df_b["WR"].head(5)]).dropna().mean()) if not df_b.empty else float("nan")
    if shared:
        return f"Heroes em comum: {', '.join(shared[:5])}"
    if avg_games_a > avg_games_b + 2:
        return f"Pool mais testado: {team1}"
    if avg_games_b > avg_games_a + 2:
        return f"Pool mais testado: {team2}"
    if math.isfinite(avg_wr_a) and math.isfinite(avg_wr_b) and abs(avg_wr_a - avg_wr_b) >= 4.0:
        return f"WR medio melhor: {team1 if avg_wr_a > avg_wr_b else team2}"
    return "Pools de player sem leitura forte"


def _draft_edge_label(team1: str, team2: str, score_a: float, score_b: float) -> str:
    diff = float(score_a - score_b)
    if abs(diff) < 0.015:
        return "Equilibrado"
    if abs(diff) < 0.05:
        return f"Edge leve: {team1 if diff > 0 else team2}"
    return f"Edge claro: {team1 if diff > 0 else team2}"


def _draft_confidence_label(value: float) -> str:
    val = float(max(0.0, min(1.0, value)))
    if val >= 0.75:
        return "Alta"
    if val >= 0.5:
        return "Media"
    if val >= 0.25:
        return "Baixa"
    return "Muito baixa"


def _draft_signal_value(summary: dict, signal: str) -> int:
    roles = dict(summary.get("roles") or {})
    attack_mix = dict(summary.get("attack_mix") or {})
    if signal == "Controle":
        return int(roles.get("Initiator", 0)) + int(roles.get("Disabler", 0))
    if signal == "Push":
        return int(roles.get("Pusher", 0))
    if signal == "Frontline":
        return int(roles.get("Durable", 0))
    if signal == "Range":
        return int(attack_mix.get("Ranged", 0))
    if signal == "Duplo core":
        return int(roles.get("Carry", 0))
    return 0


def _draft_signal_read(team1: str, team2: str, summary_a: dict, summary_b: dict) -> pd.DataFrame:
    rows: List[dict] = []
    for signal in ["Controle", "Push", "Frontline", "Range", "Duplo core"]:
        val_a = _draft_signal_value(summary_a, signal)
        val_b = _draft_signal_value(summary_b, signal)
        if val_a == val_b:
            resolution = "Equilibrado"
        else:
            resolution = team1 if val_a > val_b else team2
        rows.append(
            {
                "Sinal": signal,
                team1: val_a,
                team2: val_b,
                "Resolucao": resolution,
            }
        )
    return pd.DataFrame(rows)


def _draft_aspect_scores(summary: dict) -> dict:
    control = float(_draft_signal_value(summary, "Controle"))
    push = float(_draft_signal_value(summary, "Push"))
    frontline = float(_draft_signal_value(summary, "Frontline"))
    scale = float(_draft_signal_value(summary, "Duplo core")) + (0.5 * float(_draft_signal_value(summary, "Range")))
    teamfight = control + frontline
    objectives = push + (0.75 * frontline)
    pace = control + push + (0.5 * frontline) - (0.5 * float(_draft_signal_value(summary, "Duplo core")))
    return {
        "Teamfight": teamfight,
        "Objetivos": objectives,
        "Escala": scale,
        "Ritmo": pace,
    }


def _draft_operational_read(team1: str, team2: str, summary_a: dict, summary_b: dict) -> pd.DataFrame:
    a = _draft_aspect_scores(summary_a)
    b = _draft_aspect_scores(summary_b)
    rows: List[dict] = []
    for aspect in ["Teamfight", "Objetivos", "Escala", "Ritmo"]:
        va = float(a.get(aspect, 0.0))
        vb = float(b.get(aspect, 0.0))
        if abs(va - vb) < 0.5:
            resolution = "Equilibrado"
        else:
            resolution = team1 if va > vb else team2
        rows.append(
            {
                "Aspecto": aspect,
                team1: f"{va:.1f}",
                team2: f"{vb:.1f}",
                "Resolucao": resolution,
            }
        )
    return pd.DataFrame(rows)


def _render_series_tab(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Previsão de Séries + EV"])
    winner = analyze_winner_market(
        dataset, year, league, team1, team2, bool(st.session_state.get("dota_laplace", True))
    )
    ml_p1 = float(winner.get("model_t1_prob", 0.5))

    # --- Probabilidade base ---
    with st.container(border=True):
        st.markdown("#### Probabilidade por Jogo")
        c_chk, c_val = st.columns([1, 2])
        with c_chk:
            use_manual = st.checkbox("Usar valor manual", key="dota_series_use_manual")
        if use_manual:
            with c_val:
                p1_pct = st.number_input(
                    f"P({team1} vence 1 jogo) %",
                    min_value=1.0,
                    max_value=99.0,
                    value=float(st.session_state.get("dota_series_manual_p1", round(ml_p1 * 100, 1))),
                    step=0.5,
                    key="dota_series_manual_p1",
                    format="%.1f",
                )
            p1 = p1_pct / 100.0
            source_label = "manual"
        else:
            p1 = ml_p1
            source_label = "modelo"
            with c_val:
                m1, m2 = st.columns(2)
                m1.metric(team1, f"{p1 * 100:.1f}%")
                m2.metric(team2, f"{(1 - p1) * 100:.1f}%")

    p2 = 1.0 - p1

    def _bo3(p: float) -> float:
        return p * p * (3.0 - 2.0 * p)

    def _bo5(p: float) -> float:
        q = 1.0 - p
        return p ** 3 * (1.0 + 3.0 * q + 6.0 * q * q)

    formats_probs = [("BO1", p1), ("BO3", _bo3(p1)), ("BO5", _bo5(p1))]

    # --- Tabela de séries ---
    with st.container(border=True):
        st.markdown("#### Previsão de Série")
        st.caption(f"Base: P({team1} por jogo) = {p1 * 100:.1f}% ({source_label})")
        cols = st.columns(3)
        for col, (fmt, ps1) in zip(cols, formats_probs):
            ps2 = 1.0 - ps1
            with col:
                st.markdown(f"**{fmt}**")
                st.metric(team1, f"{ps1 * 100:.1f}%", delta=f"odd {1/ps1:.2f}" if ps1 > 0 else "-")
                st.metric(team2, f"{ps2 * 100:.1f}%", delta=f"odd {1/ps2:.2f}" if ps2 > 0 else "-")
        with st.expander("Breakdown de scores", expanded=False):
            q = p2
            breakdown = pd.DataFrame([
                {"Formato": "BO3", "Score (T1-T2)": "2-0", "Prob.": f"{p1*p1*100:.1f}%"},
                {"Formato": "BO3", "Score (T1-T2)": "2-1", "Prob.": f"{2*p1*p1*q*100:.1f}%"},
                {"Formato": "BO3", "Score (T1-T2)": "1-2", "Prob.": f"{2*p2*p2*p1*100:.1f}%"},
                {"Formato": "BO3", "Score (T1-T2)": "0-2", "Prob.": f"{p2*p2*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "3-0", "Prob.": f"{p1**3*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "3-1", "Prob.": f"{3*p1**3*q*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "3-2", "Prob.": f"{6*p1**3*q**2*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "2-3", "Prob.": f"{6*p2**3*p1**2*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "1-3", "Prob.": f"{3*p2**3*p1*100:.1f}%"},
                {"Formato": "BO5", "Score (T1-T2)": "0-3", "Prob.": f"{p2**3*100:.1f}%"},
            ])
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

    # --- Calculadora EV ---
    def _ev(p: float, odd: float) -> float:
        return p * (odd - 1.0) - (1.0 - p)

    def _kelly(p: float, odd: float) -> float:
        b = odd - 1.0
        if b <= 0.0:
            return 0.0
        return max(0.0, (p * b - (1.0 - p)) / b)

    with st.container(border=True):
        st.markdown("#### Calculadora de EV")
        st.caption("Odd justa = 1 / probabilidade. EV positivo indica edge sobre a casa. ½ Kelly é a fração conservadora do bankroll.")
        ev_tabs = st.tabs(["BO1", "BO3", "BO5"])
        for ev_tab, (fmt, ps1) in zip(ev_tabs, formats_probs):
            ps2 = 1.0 - ps1
            with ev_tab:
                c_t1, c_t2 = st.columns(2)
                with c_t1:
                    st.markdown(f"**{team1}** — P={ps1 * 100:.1f}% | Justa: {1/ps1:.3f}")
                    odd_t1 = st.number_input(
                        "Odd da casa",
                        min_value=1.01,
                        max_value=50.0,
                        step=0.05,
                        value=max(1.01, round(1 / ps1, 2)) if ps1 > 0.02 else 1.01,
                        key=f"dota_ev_{fmt}_t1",
                        format="%.2f",
                    )
                    ev1 = _ev(ps1, odd_t1)
                    kel1 = _kelly(ps1, odd_t1)
                    if ev1 > 0:
                        st.success(f"EV: +{ev1 * 100:.2f}%  |  Valor positivo")
                    else:
                        st.error(f"EV: {ev1 * 100:.2f}%  |  Sem valor")
                    st.metric("Kelly", f"{kel1 * 100:.2f}%", delta=f"½ Kelly: {kel1 * 50:.2f}%")
                with c_t2:
                    st.markdown(f"**{team2}** — P={ps2 * 100:.1f}% | Justa: {1/ps2:.3f}")
                    odd_t2 = st.number_input(
                        "Odd da casa",
                        min_value=1.01,
                        max_value=50.0,
                        step=0.05,
                        value=max(1.01, round(1 / ps2, 2)) if ps2 > 0.02 else 1.01,
                        key=f"dota_ev_{fmt}_t2",
                        format="%.2f",
                    )
                    ev2 = _ev(ps2, odd_t2)
                    kel2 = _kelly(ps2, odd_t2)
                    if ev2 > 0:
                        st.success(f"EV: +{ev2 * 100:.2f}%  |  Valor positivo")
                    else:
                        st.error(f"EV: {ev2 * 100:.2f}%  |  Sem valor")
                    st.metric("Kelly", f"{kel2 * 100:.2f}%", delta=f"½ Kelly: {kel2 * 50:.2f}%")

    # --- Handicap de Série ---
    with st.container(border=True):
        st.markdown("#### Handicap de Série")
        st.caption("Calcula P(cobre) para handicaps de mapa (+1.5 = não pode perder por 2-0; -1.5 = precisa 2-0 sweep, etc.).")
        hc_c1, hc_c2, hc_c3 = st.columns(3)
        with hc_c1:
            hc_fmt = st.selectbox("Formato", ["BO3", "BO5"], key="dota_hc_fmt")
        with hc_c2:
            hc_team = st.selectbox("Time com handicap", [team1, team2], key="dota_hc_team")
        with hc_c3:
            if hc_fmt == "BO3":
                hc_opts = ["+1.5  (cobre se não levar 0-2)", "-1.5  (precisa 2-0 sweep)"]
            else:
                hc_opts = [
                    "+1.5  (cobre se não levar por 2+)",
                    "-1.5  (vence por 2+ mapas)",
                    "+2.5  (cobre se não levar 0-3)",
                    "-2.5  (precisa 3-0 sweep)",
                ]
            hc_type = st.selectbox("Handicap", hc_opts, key="dota_hc_type")

        pt = p1 if hc_team == team1 else p2
        po = 1.0 - pt
        hc_val = hc_type[:4].strip()

        if hc_fmt == "BO3":
            p_cover = (1.0 - po ** 2) if hc_val == "+1.5" else pt ** 2
        else:
            if hc_val == "+1.5":
                p_cover = 1.0 - po ** 3 - 3.0 * po ** 3 * pt
            elif hc_val == "-1.5":
                p_cover = pt ** 3 + 3.0 * pt ** 3 * po
            elif hc_val == "+2.5":
                p_cover = 1.0 - po ** 3
            else:
                p_cover = pt ** 3

        hc_r1, hc_r2 = st.columns(2)
        hc_r1.metric(f"P({hc_team} cobre)", f"{p_cover * 100:.1f}%", delta=f"odd justa {1/p_cover:.3f}")
        with hc_r2:
            hc_odd = st.number_input(
                "Odd da casa",
                min_value=1.01,
                max_value=50.0,
                step=0.05,
                value=max(1.01, round(1.0 / p_cover, 2)) if p_cover > 0.02 else 1.01,
                key="dota_hc_odd",
                format="%.2f",
            )
        ev_hc = _ev(p_cover, hc_odd)
        kel_hc = _kelly(p_cover, hc_odd)
        if ev_hc > 0:
            st.success(f"EV: +{ev_hc * 100:.2f}%  |  Valor positivo")
        else:
            st.error(f"EV: {ev_hc * 100:.2f}%  |  Sem valor")
        st.metric("Kelly", f"{kel_hc * 100:.2f}%", delta=f"½ Kelly: {kel_hc * 50:.2f}%")

    # --- Total de Maps ---
    with st.container(border=True):
        st.markdown("#### Total de Maps")
        st.caption("Probabilidade de over/under no número total de mapas da série.")
        tm_c1, tm_c2 = st.columns(2)
        with tm_c1:
            tm_fmt = st.selectbox("Formato", ["BO3", "BO5"], key="dota_tm_fmt")
        with tm_c2:
            if tm_fmt == "BO3":
                tm_opts = ["Over 2.5  (vai a 3 mapas)", "Under 2.5  (termina em 2 mapas)"]
            else:
                tm_opts = [
                    "Over 4.5  (vai a 5 mapas)",
                    "Under 4.5  (termina em 3 ou 4 mapas)",
                    "Over 3.5  (vai a 4 ou 5 mapas)",
                    "Under 3.5  (termina em 3 mapas)",
                ]
            tm_type = st.selectbox("Mercado", tm_opts, key="dota_tm_type")

        tm_val = tm_type[:9].strip()
        if tm_fmt == "BO3":
            p_over25 = 2.0 * p1 * p2
            p_under25 = 1.0 - p_over25
            p_tm = p_over25 if tm_val.startswith("Over 2.5") else p_under25
        else:
            p5maps = 6.0 * p1 ** 2 * p2 ** 2
            p3maps = p1 ** 3 + p2 ** 3
            p4maps = 1.0 - p3maps - p5maps
            if tm_val.startswith("Over 4.5"):
                p_tm = p5maps
            elif tm_val.startswith("Under 4.5"):
                p_tm = 1.0 - p5maps
            elif tm_val.startswith("Over 3.5"):
                p_tm = p4maps + p5maps
            else:
                p_tm = p3maps

        tm_r1, tm_r2 = st.columns(2)
        tm_r1.metric("P(cobre)", f"{p_tm * 100:.1f}%", delta=f"odd justa {1/p_tm:.3f}" if p_tm > 0 else "-")
        with tm_r2:
            tm_odd = st.number_input(
                "Odd da casa",
                min_value=1.01,
                max_value=50.0,
                step=0.05,
                value=max(1.01, round(1.0 / p_tm, 2)) if p_tm > 0.02 else 1.01,
                key="dota_tm_odd",
                format="%.2f",
            )
        ev_tm = _ev(p_tm, tm_odd)
        kel_tm = _kelly(p_tm, tm_odd)
        if ev_tm > 0:
            st.success(f"EV: +{ev_tm * 100:.2f}%  |  Valor positivo")
        else:
            st.error(f"EV: {ev_tm * 100:.2f}%  |  Sem valor")
        st.metric("Kelly", f"{kel_tm * 100:.2f}%", delta=f"½ Kelly: {kel_tm * 50:.2f}%")


def _render_heroes_tab(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Heroes"])
    _render_tab_hero_box(
        "Heroes",
        "Pool dos times, hero foco e leitura comparativa via OpenDota.",
        chips=["OpenDota", "Pool dos times", "Hero foco"],
    )
    team1_id = dataset.resolve_team_id(team1)
    team2_id = dataset.resolve_team_id(team2)
    if not team1_id or not team2_id:
        st.info("Nao foi possivel resolver team_id para os dois lados no OpenDota.")
        return
    with st.container(border=True):
        _render_tab_section_head("Filtros", "Feche o recorte e ajuste quantos heroes do pool vao entrar na leitura.")
        top_n = int(
            st.slider(
                "Top heroes por time",
                min_value=5,
                max_value=20,
                value=12,
                step=1,
                key="dota_heroes_top_n",
            )
        )

    df_a = _team_heroes_table(team1_id, top_n)
    df_b = _team_heroes_table(team2_id, top_n)

    with st.container(border=True):
        _render_tab_section_head("Pool dos times", "Top pool atual dos dois lados no OpenDota, no mesmo espirito do resumo de Campeoes do LoL.")
        cards = st.columns(6)
        cards[0].metric(f"Team ID {team1}", int(team1_id))
        cards[1].metric(f"Team ID {team2}", int(team2_id))
        cards[2].metric(f"Heroes {team1}", int(len(df_a)))
        cards[3].metric(f"Heroes {team2}", int(len(df_b)))
        cards[4].metric(f"Conf. {team1}", _hero_pool_confidence(df_a))
        cards[5].metric(f"Conf. {team2}", _hero_pool_confidence(df_b))
        compare_df = pd.DataFrame(
            [
                {"Time": team1, "Estilo": _hero_pool_style(df_a), "Confianca": _hero_pool_confidence(df_a)},
                {"Time": team2, "Estilo": _hero_pool_style(df_b), "Confianca": _hero_pool_confidence(df_b)},
            ]
        )
        st.markdown("**Leitura do pool**")
        st.dataframe(compare_df, hide_index=True, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{team1}**")
            if df_a.empty:
                st.info("Sem pool disponivel no OpenDota para este time.")
            else:
                st.dataframe(df_a, hide_index=True, use_container_width=True)
        with right:
            st.markdown(f"**{team2}**")
            if df_b.empty:
                st.info("Sem pool disponivel no OpenDota para este time.")
            else:
                st.dataframe(df_b, hide_index=True, use_container_width=True)
        if not df_a.empty and not df_b.empty and "Heroi" in df_a.columns and "Heroi" in df_b.columns:
            overlap = sorted(set(df_a["Heroi"].astype(str)).intersection(set(df_b["Heroi"].astype(str))))
            overlap_label = ", ".join(overlap[:12]) if overlap else "Sem overlap forte no top pool"
            st.caption(f"Overlap do top pool: {overlap_label}")

    with st.container(border=True):
        _render_tab_section_head("Hero foco", "Escolha um hero para comparar o uso dos dois times no mesmo recorte da API.")
        hero_options = sorted(
            {
                str(v)
                for v in pd.concat(
                    [
                        df_a["Heroi"] if "Heroi" in df_a.columns else pd.Series(dtype=str),
                        df_b["Heroi"] if "Heroi" in df_b.columns else pd.Series(dtype=str),
                    ],
                    ignore_index=True,
                ).dropna()
                if str(v).strip()
            }
        )
        if not hero_options:
            st.info("Sem heroes suficientes para abrir a comparacao.")
        else:
            hero_focus = st.selectbox("Heroi", [""] + hero_options, index=0, key="dota_hero_focus")
            if not hero_focus:
                st.info("Selecione um hero para abrir a comparacao.")
            else:
                focus_rows: List[dict] = []
                team_rows_by_name: Dict[str, dict] = {}
                for team_name, df_team in ((team1, df_a), (team2, df_b)):
                    focus = df_team[df_team["Heroi"].astype(str) == str(hero_focus)].head(1) if not df_team.empty else pd.DataFrame()
                    if focus.empty:
                        row_data = {"Time": team_name, "Heroi": hero_focus, "Games": 0, "Wins": 0, "WR": "-", "Roles": "-", "Pro WR": "-"}
                    else:
                        row = focus.iloc[0]
                        row_data = {
                            "Time": team_name,
                            "Heroi": str(row.get("Heroi") or hero_focus),
                            "Games": int(pd.to_numeric(pd.Series([row.get("Games")]), errors="coerce").fillna(0).iloc[0]),
                            "Wins": int(pd.to_numeric(pd.Series([row.get("Wins")]), errors="coerce").fillna(0).iloc[0]),
                            "WR": str(row.get("WR") or "-"),
                            "Roles": str(row.get("Roles") or "-"),
                            "Pro WR": str(row.get("Pro WR") or "-"),
                        }
                    team_rows_by_name[str(team_name)] = row_data
                    focus_rows.append(row_data)
                read = _hero_focus_read(team1, team2, team_rows_by_name.get(team1, {}), team_rows_by_name.get(team2, {}))
                st.caption(f"Leitura do hero foco: {read}.")
                st.dataframe(pd.DataFrame(focus_rows), hide_index=True, use_container_width=True)

    with st.expander("Ver pools completos", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{team1}**")
            st.dataframe(_team_heroes_table(team1_id, 20), hide_index=True, use_container_width=True)
        with right:
            st.markdown(f"**{team2}**")
            st.dataframe(_team_heroes_table(team2_id, 20), hide_index=True, use_container_width=True)


def _render_players_tab(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Players e lineup"])
    _render_tab_hero_box(
        "Players",
        "Lineup recente e hero pools por player via OpenDota.",
        chips=["Lineup recente", "Herois por player", "OpenDota"],
    )
    team1_id = dataset.resolve_team_id(team1)
    team2_id = dataset.resolve_team_id(team2)
    if not team1_id or not team2_id:
        st.info("Nao foi possivel resolver team_id para os dois lados no OpenDota.")
        return
    lineup_a = _infer_recent_lineup(team1_id, team1).head(5)
    lineup_b = _infer_recent_lineup(team2_id, team2).head(5)
    with st.container(border=True):
        _render_tab_section_head("Lineup recente", "Lineup inferido pelas partidas mais recentes do time, no mesmo papel do autofill do LoL.")
        cards = st.columns(6)
        cards[0].metric(f"Players {team1}", int(len(lineup_a)))
        cards[1].metric(f"Players {team2}", int(len(lineup_b)))
        cards[2].metric(
            f"Atuais {team1}",
            int(pd.to_numeric(lineup_a.get("current_member"), errors="coerce").fillna(0).astype(int).sum()) if not lineup_a.empty else 0,
        )
        cards[3].metric(
            f"Atuais {team2}",
            int(pd.to_numeric(lineup_b.get("current_member"), errors="coerce").fillna(0).astype(int).sum()) if not lineup_b.empty else 0,
        )
        cards[4].metric(f"Conf. {team1}", _lineup_confidence(lineup_a))
        cards[5].metric(f"Conf. {team2}", _lineup_confidence(lineup_b))
        st.caption(f"Leitura do lineup: {_lineup_compare_read(team1, team2, lineup_a, lineup_b)}.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{team1}**")
            if lineup_a.empty:
                st.info("Sem lineup recente inferido.")
            else:
                show_a = lineup_a.rename(columns={"name": "Player", "appearances": "Jogos recentes", "last_seen_label": "Último uso", "current_member": "Atual"})[
                    ["Player", "Jogos recentes", "Último uso", "Atual"]
                ]
                st.dataframe(show_a, hide_index=True, use_container_width=True)
        with c2:
            st.markdown(f"**{team2}**")
            if lineup_b.empty:
                st.info("Sem lineup recente inferido.")
            else:
                show_b = lineup_b.rename(columns={"name": "Player", "appearances": "Jogos recentes", "last_seen_label": "Último uso", "current_member": "Atual"})[
                    ["Player", "Jogos recentes", "Último uso", "Atual"]
                ]
                st.dataframe(show_b, hide_index=True, use_container_width=True)

    with st.container(border=True):
        _render_tab_section_head("Herois por player", "Escolha um player foco de cada lado para abrir o pool de heroes dele.")
        c1, c2 = st.columns(2)
        focus_table_a = pd.DataFrame()
        focus_table_b = pd.DataFrame()
        with c1:
            st.markdown(f"**{team1}**")
            if lineup_a.empty:
                st.info("Sem lineup recente inferido.")
            else:
                player_opts_a = {str(row["name"]): int(row["account_id"]) for _, row in lineup_a.iterrows()}
                pick_a = st.selectbox("Player foco", list(player_opts_a.keys()), key="dota_player_focus_a")
                focus_table_a = _player_top_heroes_table(player_opts_a[pick_a])
                st.dataframe(focus_table_a, hide_index=True, use_container_width=True)
        with c2:
            st.markdown(f"**{team2}**")
            if lineup_b.empty:
                st.info("Sem lineup recente inferido.")
            else:
                player_opts_b = {str(row["name"]): int(row["account_id"]) for _, row in lineup_b.iterrows()}
                pick_b = st.selectbox("Player foco", list(player_opts_b.keys()), key="dota_player_focus_b")
                focus_table_b = _player_top_heroes_table(player_opts_b[pick_b])
                st.dataframe(focus_table_b, hide_index=True, use_container_width=True)
        if not focus_table_a.empty or not focus_table_b.empty:
            st.caption(f"Leitura dos players foco: {_player_focus_compare(team1, team2, focus_table_a, focus_table_b)}.")

    with st.expander("Ver lineup bruto", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**{team1}**")
            st.dataframe(_infer_recent_lineup(team1_id, team1), hide_index=True, use_container_width=True)
        with right:
            st.markdown(f"**{team2}**")
            st.dataframe(_infer_recent_lineup(team2_id, team2), hide_index=True, use_container_width=True)


def _render_draft_tab(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> None:
    _render_context_bar(year, league, team1, team2, ["Leitura: Draft"])
    _render_tab_hero_box(
        "Draft",
        "Leitura de draft por conforto do time e sinais globais do hero.",
        chips=["Blue side", "Fit do time", "OpenDota"],
    )
    team1_id = dataset.resolve_team_id(team1)
    team2_id = dataset.resolve_team_id(team2)
    if not team1_id or not team2_id:
        st.info("Nao foi possivel resolver team_id para os dois lados no OpenDota.")
        return
    with st.container(border=True):
        _render_tab_section_head("Cole os heroes", "Use 5 heroes por lado, separados por virgula ou quebra de linha.")
        top = st.columns([1, 1])
        top[0].radio("Blue side", ["Time 1", "Time 2"], horizontal=True, key="dota_draft_blueside")
        draft_cols = st.columns(2)
        draft_cols[0].text_area(f"Draft {team1}", key="dota_draft_team1", height=110, placeholder="ex: Tiny, Tusk, Puck, Luna, Shadow Demon")
        draft_cols[1].text_area(f"Draft {team2}", key="dota_draft_team2", height=110, placeholder="ex: Mars, Mirana, Ember Spirit, Gyrocopter, Oracle")
    heroes_a = _parse_hero_list(st.session_state.get("dota_draft_team1", ""))
    heroes_b = _parse_hero_list(st.session_state.get("dota_draft_team2", ""))
    if not heroes_a and not heroes_b:
        st.info("Digite os herois dos dois lados para abrir a leitura de draft.")
        return
    summary_a = _draft_summary(heroes_a, team1_id)
    summary_b = _draft_summary(heroes_b, team2_id)
    score_a = summary_a["comfort"] * 0.65 + summary_a["pro_wr"] * 0.35
    score_b = summary_b["comfort"] * 0.65 + summary_b["pro_wr"] * 0.35
    unresolved_a = [token for token in heroes_a if _resolve_hero_text(token) is None]
    unresolved_b = [token for token in heroes_b if _resolve_hero_text(token) is None]
    draft_view = st.radio("Visão do draft", ["Resumo", "Heroes", "Auditoria"], horizontal=True, key="dota_draft_view")

    if draft_view == "Resumo":
        with st.container(border=True):
            _render_tab_section_head("Resumo do draft", "Mesma ideia do LoL: leitura primeiro, auditoria depois.")
            cards = st.columns(6)
            cards[0].metric(f"Fit {team1}", f"{score_a * 100:.1f}")
            cards[1].metric(f"Fit {team2}", f"{score_b * 100:.1f}")
            cards[2].metric("Reconhecidos A", int(summary_a["recognized"]))
            cards[3].metric("Reconhecidos B", int(summary_b["recognized"]))
            cards[4].metric("Confianca A", _draft_confidence_label(float(summary_a.get("confidence", 0.0))))
            cards[5].metric("Confianca B", _draft_confidence_label(float(summary_b.get("confidence", 0.0))))
            summary_df = pd.DataFrame(
                [
                    {
                        "Time": team1,
                        "Conforto": f"{summary_a['comfort'] * 100:.1f}%",
                        "Pro WR medio": f"{summary_a['pro_wr'] * 100:.1f}%",
                        "Games medio": f"{float(summary_a.get('team_games_avg', 0.0)):.1f}",
                        "Confianca": _draft_confidence_label(float(summary_a.get("confidence", 0.0))),
                        "Tags": _draft_tags(summary_a),
                    },
                    {
                        "Time": team2,
                        "Conforto": f"{summary_b['comfort'] * 100:.1f}%",
                        "Pro WR medio": f"{summary_b['pro_wr'] * 100:.1f}%",
                        "Games medio": f"{float(summary_b.get('team_games_avg', 0.0)):.1f}",
                        "Confianca": _draft_confidence_label(float(summary_b.get("confidence", 0.0))),
                        "Tags": _draft_tags(summary_b),
                    },
                ]
            )
            st.caption(f"Leitura geral: {_draft_edge_label(team1, team2, score_a, score_b)}.")
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            st.markdown("**Sinais da composicao**")
            st.dataframe(_draft_signal_read(team1, team2, summary_a, summary_b), hide_index=True, use_container_width=True)
            st.markdown("**Leitura operacional**")
            st.dataframe(_draft_operational_read(team1, team2, summary_a, summary_b), hide_index=True, use_container_width=True)
            if heroes_a and heroes_b:
                edge = team1 if score_a > score_b else team2 if score_b > score_a else "Equilibrado"
                st.caption(f"Leitura inicial do draft: {edge}. Blue side: {st.session_state.get('dota_draft_blueside', 'Time 1')}.")

    elif draft_view == "Heroes":
        with st.container(border=True):
            _render_tab_section_head("Detalhe por hero", "Conforto do time no hero, WR do time e roles conhecidas.")
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.markdown(f"**{team1}**")
                if heroes_a:
                    st.dataframe(_draft_team_table(team1_id, heroes_a), hide_index=True, use_container_width=True)
                else:
                    st.info("Sem heroes colados para este lado.")
            with detail_cols[1]:
                st.markdown(f"**{team2}**")
                if heroes_b:
                    st.dataframe(_draft_team_table(team2_id, heroes_b), hide_index=True, use_container_width=True)
                else:
                    st.info("Sem heroes colados para este lado.")

    else:
        with st.container(border=True):
            _render_tab_section_head("Auditoria do draft", "Use esta visao para inspecionar a entrada bruta e os heroes nao reconhecidos.")
            aud1, aud2 = st.columns(2)
            with aud1:
                st.markdown(f"**{team1}**")
                st.code("\n".join(heroes_a) if heroes_a else "-", language="text")
            with aud2:
                st.markdown(f"**{team2}**")
                st.code("\n".join(heroes_b) if heroes_b else "-", language="text")
            if unresolved_a or unresolved_b:
                if unresolved_a:
                    st.write({team1: unresolved_a})
                if unresolved_b:
                    st.write({team2: unresolved_b})
            else:
                st.caption("Todos os heroes colados foram reconhecidos.")


def _team_id_input(label: str, resolved_team_id: Optional[int], key: str) -> Optional[int]:
    default = "" if resolved_team_id is None else str(int(resolved_team_id))
    raw = st.text_input(label, value=default, key=key)
    token = str(raw or "").strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _hero_table(rows: List[dict], hero_meta: Dict[int, dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    records: List[dict] = []
    for row in rows[:12]:
        games = int(row.get("games_played") or 0)
        wins = int(row.get("wins") or 0)
        hero_id = int(row.get("hero_id") or 0)
        wr = (wins / games * 100.0) if games > 0 else float("nan")
        meta = hero_meta.get(hero_id) or {}
        records.append(
            {
                "Heroi": row.get("localized_name") or meta.get("localized_name") or f"Hero {hero_id}",
                "Games": games,
                "Wins": wins,
                "WR": f"{wr:.1f}%" if math.isfinite(wr) else "-",
                "Roles": ", ".join(meta.get("roles") or []) if meta.get("roles") else "-",
                "Atributo": (meta.get("primary_attr") or "-").upper(),
            }
        )
    return pd.DataFrame(records)


def _render_opendota(dataset: DotaDataset, team1: str, team2: str) -> None:
    _render_tab_hero_box(
        "OpenDota",
        "Leitura de times e hero pools direto da API, no mesmo papel da area tecnica do LoL.",
        chips=["API", "Team IDs", "Hero pools"],
    )
    status = auth_status()
    if status["enabled"]:
        st.caption(f"API key ativa via {status['source']} ({status['masked_key']}).")
    else:
        st.caption("Sem API key configurada. A leitura usa os limites publicos do OpenDota.")

    team1_resolved = dataset.resolve_team_id(team1)
    team2_resolved = dataset.resolve_team_id(team2)
    with st.container(border=True):
        _render_tab_section_head("Identificacao dos times", "Confirme ou ajuste os Team IDs antes de abrir a leitura da API.")
        left, right = st.columns(2)
        with left:
            team1_id = _team_id_input(f"OpenDota Team ID - {team1}", team1_resolved, "od_team1_id")
        with right:
            team2_id = _team_id_input(f"OpenDota Team ID - {team2}", team2_resolved, "od_team2_id")

    hero_meta_rows = _od_hero_stats_cached()
    hero_meta = {int(item.get("id") or 0): item for item in hero_meta_rows if item.get("id") is not None}

    c1, c2 = st.columns(2)
    for col, team_name, team_id in [(c1, team1, team1_id), (c2, team2, team2_id)]:
        with col:
            st.markdown(f"### {team_name}")
            if not team_id:
                st.info("Sem team_id resolvido. Informe um OpenDota Team ID manualmente.")
                continue
            try:
                summary = _od_team_summary_cached(team_id)
                heroes = _od_team_heroes_cached(team_id)
            except Exception as exc:
                st.error(f"Falha ao consultar OpenDota para {team_name}: {exc}")
                continue

            with st.container(border=True):
                _render_tab_section_head(team_name, "Resumo do time e hero pools recentes da API.")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Team ID", int(team_id))
                s2.metric("Rating", f"{float(summary.get('rating') or 0.0):.0f}")
                s3.metric("Wins", int(summary.get("wins") or 0))
                s4.metric("Losses", int(summary.get("losses") or 0))
                last_match_time = summary.get("last_match_time")
                if last_match_time:
                    try:
                        last_seen = pd.to_datetime(int(last_match_time), unit="s", utc=True).tz_convert(None)
                        st.caption(f"Último jogo no OpenDota: {last_seen:%Y-%m-%d %H:%M}")
                    except Exception:
                        pass

                hero_df = _hero_table(heroes, hero_meta)
                if hero_df.empty:
                    st.info("Sem dados de herois para esse team_id.")
                else:
                    st.markdown("**Herois mais jogados**")
                    st.dataframe(hero_df, hide_index=True, use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def _od_pro_matches_cached() -> List[dict]:
    return get_pro_matches(100)


@st.cache_data(ttl=300, show_spinner=False)
def _od_match_cached(match_id: int) -> dict:
    return get_match(match_id)


def _render_opendota_api_mode() -> None:
    _render_tab_hero_box(
        "Partidas via API",
        "Consulta de match pronta na OpenDota, com resumo, draft e players.",
        chips=["Pro matches", "Match ID manual", "Importar para CSV"],
    )
    status = auth_status()
    source_text = "publico" if not status["enabled"] else f"{status['source']} ({status['masked_key']})"
    st.caption(f"Fonte: OpenDota API | autenticacao: {source_text}.")

    with st.container(border=True):
        _render_tab_section_head("Filtros", "Filtre por campeonato ou abra uma partida direto por Match ID.")
        top_left, top_right = st.columns([2, 1])
        with top_left:
            st.text_input("Filtrar campeonato (opcional)", key="od_league_filter", placeholder="ex: DreamLeague, ESL One, PGL")
        with top_right:
            st.text_input("Match ID manual (opcional)", key="od_match_id_manual", placeholder="ex: 8727860551")

    try:
        matches = _od_pro_matches_cached()
    except Exception as exc:
        st.error(f"Falha ao carregar partidas recentes do OpenDota: {exc}")
        return

    league_filter = (st.session_state.get("od_league_filter") or "").strip().lower()
    if league_filter:
        matches = [m for m in matches if league_filter in str(m.get("league_name") or "").lower()]

    options: Dict[str, int] = {}
    for item in matches:
        match_id = int(item.get("match_id") or 0)
        if match_id <= 0:
            continue
        start_time = pd.to_datetime(int(item.get("start_time") or 0), unit="s", utc=True).tz_convert(None)
        label = (
            f"{start_time:%Y-%m-%d %H:%M} | {item.get('league_name') or '-'} | "
            f"{item.get('radiant_name') or 'Radiant'} x {item.get('dire_name') or 'Dire'} | {match_id}"
        )
        options[label] = match_id

    with st.container(border=True):
        _render_tab_section_head("Partidas recentes", "Selecione uma partida recente ou use Match ID manual.")
        selected_label = st.selectbox("Partida recente", [""] + list(options.keys()), key="od_match_label")
    manual_token = str(st.session_state.get("od_match_id_manual") or "").strip()
    match_id = None
    if manual_token:
        try:
            match_id = int(manual_token)
        except ValueError:
            match_id = None
    if match_id is None and selected_label:
        match_id = options.get(selected_label)

    if not match_id:
        st.info("Selecione uma partida recente ou informe um Match ID manualmente.")
        return

    try:
        payload = _od_match_cached(match_id)
    except Exception as exc:
        st.error(f"Falha ao carregar a partida {match_id}: {exc}")
        return

    league_name = payload.get("league", {}).get("name") or payload.get("league_name") or "-"
    radiant_name = payload.get("radiant_name") or payload.get("radiant_team", {}).get("name") or "Radiant"
    dire_name = payload.get("dire_name") or payload.get("dire_team", {}).get("name") or "Dire"
    duration = int(payload.get("duration") or 0)
    start_time = payload.get("start_time")
    start_label = "-"
    if start_time:
        try:
            ts = pd.to_datetime(int(start_time), unit="s", utc=True).tz_convert(None)
            start_label = f"{ts:%Y-%m-%d %H:%M}"
        except Exception:
            pass

    with st.container(border=True):
        _render_tab_section_head("Resumo da partida", "Contexto bruto da partida aberta pela API.")
        st.markdown(
            "".join(
                [
                    f"<span class='gabi-chip'>Match ID: {match_id}</span>",
                    f"<span class='gabi-chip'>Campeonato: {league_name}</span>",
                    f"<span class='gabi-chip'>Inicio: {start_label}</span>",
                    f"<span class='gabi-chip'>Duracao: {fmt_mmss_from_minutes(duration / 60.0) if duration else '-'}</span>",
                ]
            ),
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Radiant", radiant_name)
        m2.metric("Dire", dire_name)
        m3.metric("Score", f"{int(payload.get('radiant_score') or 0)} x {int(payload.get('dire_score') or 0)}")
        m4.metric("Vencedor", radiant_name if payload.get("radiant_win") else dire_name)

        if st.button("Importar esta partida para o CSV", use_container_width=True, key="od_import_current_match"):
            try:
                report = import_match_ids([int(match_id)])
                st.success(
                    f"Partida importada. Matches novos: {report['imported_matches']} | "
                    f"Linhas novas: {report['new_rows']} | Total CSV: {report['total_rows']}"
                )
                st.json(report)
            except Exception as exc:
                st.error(f"Falha ao importar a partida atual: {exc}")

    hero_meta_rows = _od_hero_stats_cached()
    hero_meta = {int(item.get("id") or 0): item for item in hero_meta_rows if item.get("id") is not None}

    picks = payload.get("picks_bans") or []
    pb_rows = []
    for item in picks:
        hero_id = int(item.get("hero_id") or 0)
        pb_rows.append(
            {
                "Side": "Radiant" if item.get("team") == 0 else "Dire",
                "Tipo": "Pick" if item.get("is_pick") else "Ban",
                "Ordem": int(item.get("order") or 0),
                "Heroi": hero_meta.get(hero_id, {}).get("localized_name") or f"Hero {hero_id}",
            }
        )
    if pb_rows:
        with st.container(border=True):
            _render_tab_section_head("Draft", "Picks e bans da partida aberta.")
            st.dataframe(pd.DataFrame(pb_rows).sort_values(["Tipo", "Ordem"]), hide_index=True, use_container_width=True)

    players = payload.get("players") or []
    if players:
        player_rows = []
        for row in players:
            hero_id = int(row.get("hero_id") or 0)
            player_rows.append(
                {
                    "Side": "Radiant" if row.get("isRadiant") else "Dire",
                    "Player": row.get("personaname") or row.get("name") or str(row.get("account_id") or "-"),
                    "Heroi": hero_meta.get(hero_id, {}).get("localized_name") or f"Hero {hero_id}",
                    "K": int(row.get("kills") or 0),
                    "D": int(row.get("deaths") or 0),
                    "A": int(row.get("assists") or 0),
                    "GPM": int(row.get("gold_per_min") or 0),
                    "XPM": int(row.get("xp_per_min") or 0),
                    "LH": int(row.get("last_hits") or 0),
                    "DN": int(row.get("denies") or 0),
                    "HD": int(row.get("hero_damage") or 0),
                    "TD": int(row.get("tower_damage") or 0),
                }
            )
        pdata = pd.DataFrame(player_rows)
        with st.container(border=True):
            _render_tab_section_head("Players da partida", "Placar detalhado por jogador.")
            left, right = st.columns(2)
            with left:
                st.markdown(f"**{radiant_name}**")
                st.dataframe(pdata[pdata["Side"] == "Radiant"].drop(columns=["Side"]), hide_index=True, use_container_width=True)
            with right:
                st.markdown(f"**{dire_name}**")
                st.dataframe(pdata[pdata["Side"] == "Dire"].drop(columns=["Side"]), hide_index=True, use_container_width=True)


def _parse_match_ids(text: str) -> List[int]:
    tokens = re.split(r"[,\s;]+", str(text or "").strip())
    out: List[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _render_sync_panel() -> None:
    _render_tab_hero_box(
        "Atualização CSV",
        "Atualiza o mesmo data.csv usado pelo app, com sync restrito a pro matches e ligas professional.",
        chips=["CSV local", "Professional only", "OpenDota"],
    )
    health = csv_health_report()
    with st.container(border=True):
        _render_tab_section_head("Estado do sync", "Última atualização conhecida do CSV local.")
        last_sync = load_sync_state()
        cards = st.columns(4)
        cards[0].metric("Matches no CSV", int(health.get("total_matches") or 0))
        cards[1].metric("Última partida", _format_age_hours(health.get("latest_match_age_hours")))
        cards[2].metric("Parciais", int(health.get("partial_matches") or 0))
        cards[3].metric("Legacy", int(health.get("legacy_matches") or 0))
        if last_sync:
            mode = str(last_sync.get("mode") or "-")
            saved_at = _format_sync_timestamp(last_sync.get("saved_at"))
            total_rows = last_sync.get("total_rows", health.get("total_rows"))
            fetched = last_sync.get("fetched_matches", last_sync.get("imported_matches", "-"))
            st.caption(
                f"Último sync: {saved_at} | modo={mode} | coverage={last_sync.get('coverage_mode', mode)} | "
                f"matches novos={fetched} | total_rows={total_rows}"
            )
            detail_bits = []
            if last_sync.get("requested_limit"):
                detail_bits.append(f"Janela solicitada: {last_sync.get('requested_limit')}")
            if last_sync.get("source_limit"):
                detail_bits.append(f"Overscan: {last_sync.get('source_limit')}")
            if last_sync.get("recent_cutoff_utc"):
                detail_bits.append(f"Corte recente: {_format_sync_timestamp(last_sync.get('recent_cutoff_utc'))}")
            if last_sync.get("league_sweep_scanned"):
                detail_bits.append(
                    f"League sweep: {int(last_sync.get('league_sweep_scanned') or 0)} ligas / +{int(last_sync.get('league_sweep_added') or 0)} matches"
                )
            if health.get("latest_match_date"):
                detail_bits.append(f"Última partida no CSV: {_format_sync_timestamp(health.get('latest_match_date'))}")
            if detail_bits:
                st.caption(" | ".join(detail_bits))
        else:
            st.info("Ainda não há registro de sync salvo.")
        if health.get("latest_match_age_hours") is not None and float(health.get("latest_match_age_hours") or 0.0) > 24.0:
            st.warning("O CSV parece defasado: a partida mais recente local tem mais de 24h.")
        elif int(health.get("partial_matches") or 0) > 0 or int(health.get("matches_missing_objectives") or 0) > 0:
            st.info("Há partidas parciais no CSV. Revise os contadores de objectives ao auditar mercados de torres, barracks e roshans.")
        if int(health.get("legacy_matches") or 0) > 0:
            st.caption(
                f"Base legado sem flags de qualidade nas novas colunas: {int(health.get('legacy_matches') or 0)} matches. "
                f"Sem objectives (confirmado): {int(health.get('matches_missing_objectives') or 0)}."
            )

    with st.container(border=True):
        _render_tab_section_head("Operação", "Atualize partidas recentes, importe Match IDs, rode sync por ligas professional ou reidrate o legado.")
        c1, c2, c3, c4, c5 = st.columns([1, 2, 1.2, 1.2, 1.1])
        limit = c1.selectbox("Pro matches recentes", [20, 50, 100], index=2, key="od_sync_limit")
        leagues_to_scan = c4.selectbox("Ligas professional", [10, 25, 50, 100], index=1, key="od_sync_leagues")
        legacy_batch = c5.selectbox("Backfill legacy", [50, 100, 250, 500], index=2, key="od_sync_legacy_batch")
        manual_ids_raw = c2.text_input("Match IDs manuais (opcional)", key="od_sync_match_ids", placeholder="ex: 8727860551, 8727826691")
        if c1.button("Atualizar agora", use_container_width=True):
            with st.spinner(f"Buscando as {int(limit)} partidas recentes na OpenDota..."):
                try:
                    report = sync_recent_pro_matches(limit=int(limit), professional_only=True)
                    st.success(
                        f"Atualização concluída. Matches novos: {report['fetched_matches']} | "
                        f"Linhas novas: {report['new_rows']} | Total CSV: {report['total_rows']}"
                    )
                    st.json(report)
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Falha ao atualizar via OpenDota: {exc}")
        if c3.button("Importar Match IDs", use_container_width=True):
            match_ids = _parse_match_ids(manual_ids_raw)
            if not match_ids:
                st.warning("Informe pelo menos um Match ID válido.")
            else:
                with st.spinner(f"Importando {len(match_ids)} match(es) da OpenDota..."):
                    try:
                        report = import_match_ids(match_ids)
                        st.success(
                            f"Importação concluída. Matches importados: {report['imported_matches']} | "
                            f"Linhas novas: {report['new_rows']} | Total CSV: {report['total_rows']}"
                        )
                        st.json(report)
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Falha ao importar Match IDs via OpenDota: {exc}")
        if c4.button("Sync ligas professional", use_container_width=True):
            with st.spinner(f"Sincronizando {int(leagues_to_scan)} ligas professional na OpenDota..."):
                try:
                    report = sync_professional_leagues(league_limit=int(leagues_to_scan), matches_per_league=50)
                    st.success(
                        f"Sync concluido. Ligas lidas: {report['leagues_scanned']} | "
                        f"Matches novos: {report['fetched_matches']} | Linhas novas: {report['new_rows']}"
                    )
                    st.json(report)
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Falha ao sincronizar ligas professional: {exc}")
        if c5.button("Backfill legado", use_container_width=True):
            with st.spinner(f"Reidratando até {int(legacy_batch)} partidas legado na OpenDota..."):
                try:
                    report = backfill_legacy_matches(limit=int(legacy_batch))
                    st.success(
                        f"Backfill concluído. Legacy reidratados: {report['backfilled_matches']} | "
                        f"Legacy restantes: {report['legacy_matches']} | Falhas: {len(report.get('failed_matches', []))}"
                    )
                    if bool(report.get("rate_limited")):
                        st.warning("A OpenDota respondeu 429 (rate limit). Espere um pouco e rode o próximo lote.")
                    st.json(report)
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Falha ao reidratar o legado: {exc}")


def _render_top_update_bar() -> None:
    with st.container(border=True):
        c1, c2 = st.columns([1.3, 3])
        with c1:
            if st.button("Atualizar Informações", type="primary", use_container_width=True, key="dota_top_update_info_btn"):
                with st.spinner("Buscando partidas recentes na OpenDota..."):
                    try:
                        report = sync_recent_pro_matches(limit=25, professional_only=False, league_sweep=False)
                        st.session_state["_dota_last_update_report"] = report
                        st.success(
                            f"Atualização concluída. Matches novos: {report['fetched_matches']} | "
                            f"Linhas novas: {report['new_rows']} | Total CSV: {report['total_rows']}"
                        )
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Falha ao atualizar informações: {exc}")
        with c2:
            sync_state = load_sync_state()
            health = csv_health_report()
            saved_at = str(sync_state.get("saved_at") or "").strip()
            if saved_at:
                label = _format_sync_timestamp(saved_at)
                mode = str(sync_state.get("mode") or "-")
                fetched = sync_state.get("fetched_matches", sync_state.get("imported_matches", "-"))
                latest_local_age = _format_age_hours(health.get("latest_match_age_hours"))
                partial_matches = int(health.get("partial_matches") or 0)
                legacy_matches = int(health.get("legacy_matches") or 0)
                st.caption(
                    f"Última atualização: {label} | modo={mode} | matches novos={fetched} | "
                    f"última partida local: {latest_local_age} | parciais: {partial_matches} | legacy: {legacy_matches}"
                )
            else:
                st.caption("Última atualização: ainda não há atualização registrada.")


def _render_technical_page() -> None:
    _render_tab_hero_box(
        "Técnico",
        "Sincronização do CSV e leitura direta da OpenDota API.",
        chips=["CSV", "OpenDota", "Operação técnica"],
    )
    view = st.radio("Técnico", ["Atualização CSV", "OpenDota API"], horizontal=True, key="dota_technical_view")
    if view == "Atualização CSV":
        _render_sync_panel()
        return
    _render_opendota_api_mode()


def main() -> None:
    _init_state()
    _normalize_ui_state()
    st.markdown("<div class='gabi-hero'><div class='gabi-kicker'>Gabinete</div><div class='gabi-section'>Dota</div><div class='gabi-caption'>dev: Bolin</div></div>", unsafe_allow_html=True)
    st.caption("Fluxo: escolher temporada/liga, preencher linhas, analisar confronto e ler histórico/modelo.")
    _render_top_update_bar()
    st.radio("Página", ["Plays", "Técnico"], horizontal=True, key="dota_nav_page")
    if st.session_state["dota_nav_page"] == "Técnico":
        _render_technical_page()
        st.stop()
    try:
        dataset = _load_dataset()
    except Exception as exc:
        st.error(f"Falha ao carregar o CSV: {exc}")
        st.stop()

    years = dataset.years()
    if st.session_state["dota_year"] not in years:
        st.session_state["dota_year"] = years[0] if years else None
    sel_year = st.selectbox("Temporada", years, key="dota_year")
    st.subheader("Confronto")
    league_options = dataset.leagues(sel_year)
    if st.session_state["dota_league"] not in league_options:
        st.session_state["dota_league"] = league_options[0]
    team_options = dataset.teams_for_scope(sel_year, st.session_state["dota_league"])
    team_choices = [""] + team_options
    if st.session_state["dota_team1"] not in team_choices:
        st.session_state["dota_team1"] = ""
    if st.session_state["dota_team2"] not in team_choices:
        st.session_state["dota_team2"] = ""
    cA, cB, cC, cD = st.columns([1.5, 3, 3, 1.2])
    with cA:
        st.selectbox("Liga", league_options, key="dota_league")
    with cB:
        st.selectbox("Time 1", team_choices, key="dota_team1")
    with cC:
        st.selectbox("Time 2", team_choices, key="dota_team2")
    with cD:
        st.selectbox("Formato", ["BO1", "BO2", "BO3", "BO5"], key="dota_match_format")
    with st.expander("Metadados do jogo (opcional)", expanded=False):
        st.text_input("Campeonato", key="dota_event_name", placeholder="ex: DreamLeague Season 26")
    _render_saved_games(dataset, sel_year, st.session_state["dota_league"], st.session_state["dota_team1"], st.session_state["dota_team2"])

    team1 = st.session_state["dota_team1"]
    team2 = st.session_state["dota_team2"]
    league = st.session_state["dota_league"]
    if team1 and team2 and team1 != team2:
        current_saved = load_saved_games().get(make_saved_key(sel_year, league, team1, team2))
        if current_saved:
            chips = [
                "Agenda: jogo salvo",
                f"Linhas: {_lines_status_label(LinesConfig.from_dict(current_saved.get('lines') or {}))}",
            ]
            event_name = str(current_saved.get("event_name") or "").strip()
            match_format = str(current_saved.get("match_format") or "").strip()
            if event_name:
                chips.append(f"Campeonato: {event_name}")
            if match_format:
                chips.append(f"Formato: {match_format}")
            st.markdown("".join(f"<span class='gabi-chip'>{chip}</span>" for chip in chips), unsafe_allow_html=True)

    if not team1 or not team2:
        st.info("Escolha Time 1 e Time 2 para iniciar a análise.")
        st.stop()
    if team1 == team2:
        st.warning("Escolha dois times diferentes.")
        st.stop()

    with st.expander("Linhas de análise (opcional)", expanded=True):
        st.caption("Use vírgula para separar múltiplas linhas. Decimal usa ponto. Tempo aceita MM:SS ou minutos.")
        _render_line_editor()
    lines = _lines_from_state()

    cfg1, cfg2 = st.columns([1.2, 4])
    with cfg1:
        st.checkbox("Laplace (Histórico)", key="dota_laplace")
    with cfg2:
        st.caption("Controla a suavização das odds do Histórico/Laplace em Consistência e Resumo.")
    overview = build_overview(dataset, sel_year, league, team1, team2)
    summary_df = build_summary_rows(dataset, sel_year, league, team1, team2, lines, st.session_state["dota_laplace"])

    _t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9 = st.tabs([
        "Visão Geral", "Consistência (Laplace)", "Stats", "Heroes", "Players", "Draft", "Detalhes", "Séries + EV", "Seletor Rápido", "Resumo"
    ])
    with _t0:
        _render_overview(dataset, sel_year, league, team1, team2, overview)
    with _t1:
        _render_consistency(dataset, sel_year, league, team1, team2, lines)
    with _t2:
        _render_stats(dataset, sel_year, league, team1, team2)
    with _t3:
        _render_heroes_tab(dataset, sel_year, league, team1, team2)
    with _t4:
        _render_players_tab(dataset, sel_year, league, team1, team2)
    with _t5:
        _render_draft_tab(dataset, sel_year, league, team1, team2)
    with _t6:
        _render_details(dataset, sel_year, league, team1, team2)
    with _t7:
        _render_series_tab(dataset, sel_year, league, team1, team2)
    with _t8:
        _render_quick_selector(summary_df)
    with _t9:
        _render_summary(sel_year, league, team1, team2, summary_df)


if __name__ == "__main__":
    main()
