# -*- coding: utf-8 -*-
"""Gabinete Dota — mesmo layout/uso do Gabinete LoL, acento vermelho.

Paginas: Diaria (agenda PandaScore ja analisada) | Visao Geral | Analise de
Linha | Rankings. Rodar: streamlit run dota_app.py --server.port 8502
"""
from __future__ import annotations

import difflib
import math
import os
from datetime import timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from dota_ui import (
    inject_global_ui_css, red_banner, red_section_header,
    team_badge_html, pct_cell_html,
)
from dota_engine import DOTA_MARKETS

TZ_LOCAL = ZoneInfo("America/Sao_Paulo")

st.set_page_config(page_title="Gabinete Dota", page_icon="🛡️", layout="wide")
inject_global_ui_css(st)

st.markdown(
    """
<div class="gb-hero">
  <div class="title">GABINETE <span style="color:#c94a38;">DOTA</span></div>
  <div class="sub">dev: Bolin</div>
</div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Treinando WHR (roda 1x por atualizacao do CSV)...")
def get_engine(_csv_mtime: float):
    from dota_engine import DotaEngine
    return DotaEngine.build()


def csv_mtime() -> float:
    from dota_core import csv_path
    p = csv_path()
    return os.path.getmtime(p) if p.exists() else 0.0


eng = get_engine(csv_mtime())
teams = eng.product_teams()
name_by_id = {t.team_id: t.name for t in teams}
options = [t.team_id for t in teams]


def fmt_team(tid: int) -> str:
    t = eng.teams[tid]
    return f"{t.name}  ({t.rating_elo:+.0f})"


# ---------------- sidebar ----------------
with st.sidebar:
    latest = eng.matches["dt"].max()
    age_h = (pd.Timestamp.utcnow() - latest).total_seconds() / 3600.0
    st.caption(f"CSV: {eng.matches['match_id'].nunique()} partidas | "
               f"ultima: {latest:%d/%m %H:%M} UTC ({age_h:.0f}h)")
    if age_h > 24:
        st.warning("CSV com mais de 24h — atualize.")
    if st.button("🔄 Atualizar CSV (OpenDota)"):
        with st.spinner("Sincronizando..."):
            from opendota_sync import sync_recent_pro_matches
            rep = sync_recent_pro_matches(limit=200, professional_only=False, max_seconds=240)
            st.success(f"+{rep.get('fetched_matches', 0)} partidas.")
            st.cache_resource.clear()
            st.rerun()
    off = eng.side_offset()
    p_rad = 1.0 / (1.0 + math.exp(-off))
    st.caption(f"Side (rolante 30d): Radiant {p_rad*100:.1f}%")
    if eng.league_obj.get("torres") is not None:
        st.caption("Torres/Roshan/Barracks: base parcial (download em andamento)")

nav_page = st.radio(
    "Pagina", ["Diária", "Visão Geral", "Análise de Linha", "Rankings"],
    horizontal=True, key="nav_main", label_visibility="collapsed",
)


# ---------------- helpers ----------------
def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().replace("team ", "").split())


_ENGINE_NAMES = {_norm(t.name): t.team_id for t in teams}


def match_team(name: str, acronym: str = "") -> int | None:
    key = _norm(name)
    if key in _ENGINE_NAMES:
        return _ENGINE_NAMES[key]
    if acronym:
        for t in teams:
            if _norm(t.name) == _norm(acronym):
                return t.team_id
    close = difflib.get_close_matches(key, list(_ENGINE_NAMES.keys()), n=1, cutoff=0.82)
    return _ENGINE_NAMES[close[0]] if close else None


def _odds_metrics_row(a_id: int, b_id: int, na: str, nb: str, bo: int = 3,
                      ref_ts: float | None = None) -> None:
    p = eng.p_map(a_id, b_id, ref_ts=ref_ts)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{na} (mapa)", f"{p*100:.1f}%", f"odd justa {1/max(p,1e-6):.2f}")
    c4.metric(f"{nb} (mapa)", f"{(1-p)*100:.1f}%", f"odd justa {1/max(1-p,1e-6):.2f}")
    if bo == 2:
        s = eng.series_probs(a_id, b_id, bo=2, ref_ts=ref_ts)
        c2.metric("BO2: 1-1", f"{s['1-1']*100:.1f}%", f"odd {1/max(s['1-1'],1e-6):.2f}")
        c3.metric(f"BO2: {na} 2-0", f"{s['2-0']*100:.1f}%", f"odd {1/max(s['2-0'],1e-6):.2f}")
    else:
        s = eng.series_probs(a_id, b_id, bo=bo if bo in (3, 5) else 3, ref_ts=ref_ts)
        c2.metric(f"Serie BO{bo or 3}: {na}", f"{s['A']*100:.1f}%", f"odd {1/max(s['A'],1e-6):.2f}")
        c3.metric(f"Serie BO{bo or 3}: {nb}", f"{s['B']*100:.1f}%", f"odd {1/max(s['B'],1e-6):.2f}")
    fat = eng.fatigue_info(a_id, b_id, ref_ts)
    if fat["maps_a"] or fat["maps_b"]:
        aviso = ""
        if fat["adj_nat"] != 0.0:
            lado = na if fat["d24"] > 0 else nb
            aviso = f" — ⚠️ termo de fadiga ATIVO contra {lado} (gateado: -6.6pp)"
        st.caption(f"Carga 24h: {na} jogou {fat['maps_a']} mapa(s), "
                   f"{nb} jogou {fat['maps_b']}{aviso}")


def _totais_metrics(a_id: int, b_id: int) -> None:
    t1, t2, t3, t4, t5 = st.columns(5)
    k1 = eng.kills_line(a_id, b_id, 1)
    k2 = eng.kills_line(a_id, b_id, 2)
    dur = eng.duration_line(a_id, b_id)
    t1.metric("Kills mapa 1", f"{k1:.1f}" if k1 else "—")
    t2.metric("Kills mapa 2+", f"{k2:.1f}" if k2 else "—", f"+{eng.m2_kills_delta:.1f} M2 over")
    t3.metric("Duracao", f"{dur:.1f} min" if dur else "—")
    tow = eng.objective_line(a_id, b_id, "torres")
    ros = eng.objective_line(a_id, b_id, "roshans")
    t4.metric("Torres", f"{tow[0]:.1f}" if tow else "—",
              f"n_eff {tow[1]}" if tow else "base parcial")
    t5.metric("Roshans", f"{ros[0]:.1f}" if ros else "—",
              f"n_eff {ros[1]}" if ros else "base parcial")


# ---------------- Diária ----------------
if nav_page == "Diária":
    red_section_header(st, "Diária — jogos das próximas 36h (T1/T2)")

    @st.cache_data(ttl=600, show_spinner="Buscando agenda (PandaScore)...")
    def _agenda():
        from pandascore_client import upcoming_matches
        return upcoming_matches(hours_ahead=36)

    try:
        agenda = _agenda()
    except Exception as exc:
        st.error(f"Agenda indisponivel: {exc}")
        agenda = []

    tier_rank = {"s": 0, "a": 1, "b": 2}
    fixtures = [f for f in agenda if f["tier"] in tier_rank]
    fixtures.sort(key=lambda f: (tier_rank.get(f["tier"], 9), f["scheduled_at"]))
    if not fixtures:
        st.info("Sem jogos T1/T2 nas proximas 36h.")
    for f in fixtures:
        a_id = match_team(f["team_a"], f["team_a_acronym"])
        b_id = match_team(f["team_b"], f["team_b_acronym"])
        when = pd.Timestamp(f["scheduled_at"])
        if when.tzinfo is None:
            when = when.tz_localize(timezone.utc)
        hora = when.tz_convert(TZ_LOCAL).strftime("%d/%m %H:%M")
        st.markdown(red_banner(
            f"{f['league']} — {f['tournament']} &nbsp;|&nbsp; tier {f['tier'].upper()} "
            f"&nbsp;|&nbsp; {hora} &nbsp;|&nbsp; BO{f['bo']}"), unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:14px;padding:10px 12px;"
            f"background:var(--gb-panel);border:1px solid var(--gb-panel-border);"
            f"border-radius:0 0 8px 8px;'>"
            f"{team_badge_html(f['team_a'], 40)}"
            f"<span style='font-weight:800;font-size:1.05em;'>{f['team_a']}</span>"
            f"<span style='opacity:0.6;font-weight:700;'>vs</span>"
            f"{team_badge_html(f['team_b'], 40)}"
            f"<span style='font-weight:800;font-size:1.05em;'>{f['team_b']}</span>"
            f"</div>", unsafe_allow_html=True)
        if a_id and b_id and a_id != b_id:
            _odds_metrics_row(a_id, b_id, f["team_a"], f["team_b"], bo=f["bo"],
                              ref_ts=when.timestamp())
            _totais_metrics(a_id, b_id)
        else:
            faltam = [n for n, i in ((f["team_a"], a_id), (f["team_b"], b_id)) if not i]
            st.caption(f"Sem preco: time(s) nao mapeado(s) no modelo — {', '.join(faltam)}")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------- Visão Geral ----------------
elif nav_page == "Visão Geral":
    c1, c2 = st.columns(2)
    with c1:
        a_id = st.selectbox("Time A", options, format_func=fmt_team, key="vg_team_a")
    with c2:
        b_id = st.selectbox("Time B", options, index=min(1, len(options) - 1),
                            format_func=fmt_team, key="vg_team_b")
    if a_id == b_id:
        st.info("Escolha dois times diferentes.")
        st.stop()
    na, nb = name_by_id[a_id], name_by_id[b_id]

    red_section_header(st, "Confronto")
    ta, tb = eng.teams[a_id], eng.teams[b_id]
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:14px;padding:12px;"
        f"background:var(--gb-panel);border:1px solid var(--gb-panel-border);border-radius:10px;'>"
        f"{team_badge_html(na, 56)}<span style='font-weight:800;font-size:1.2em;'>{na}</span>"
        f"<span class='gb-chip'>{ta.rating_elo:+.0f}</span>"
        f"<span style='opacity:0.6;font-weight:700;margin:0 6px;'>vs</span>"
        f"{team_badge_html(nb, 56)}<span style='font-weight:800;font-size:1.2em;'>{nb}</span>"
        f"<span class='gb-chip'>{tb.rating_elo:+.0f}</span></div>",
        unsafe_allow_html=True,
    )

    red_section_header(st, "Odds do confronto (WHR)")
    _odds_metrics_row(a_id, b_id, na, nb, bo=3)
    p_r = eng.p_map(a_id, b_id, "radiant")
    p_d = eng.p_map(a_id, b_id, "dire")
    st.caption(f"Side: se {na} = Radiant {p_r*100:.1f}% | se Dire {p_d*100:.1f}%")

    red_section_header(st, "Totais (modelo)")
    _totais_metrics(a_id, b_id)

    red_section_header(st, "Histórico")
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(red_banner(f"Ultimos jogos — {na}"), unsafe_allow_html=True)
        st.dataframe(eng.team_recent(a_id), hide_index=True, use_container_width=True)
    with h2:
        st.markdown(red_banner(f"Ultimos jogos — {nb}"), unsafe_allow_html=True)
        st.dataframe(eng.team_recent(b_id), hide_index=True, use_container_width=True)
    h2h = eng.h2h(a_id, b_id)
    if len(h2h):
        st.markdown(red_banner(f"H2H (A = {na})"), unsafe_allow_html=True)
        st.dataframe(h2h, hide_index=True, use_container_width=True)

# ---------------- Análise de Linha ----------------
elif nav_page == "Análise de Linha":
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    with c1:
        a_id = st.selectbox("Time A", options, format_func=fmt_team, key="al_team_a")
    with c2:
        b_id = st.selectbox("Time B", options, index=min(1, len(options) - 1),
                            format_func=fmt_team, key="al_team_b")
    with c3:
        market = st.selectbox("Mercado", list(DOTA_MARKETS.keys()),
                              format_func=lambda k: DOTA_MARKETS[k]["label"], key="al_market")
    defaults = {"kills": 50.5, "duracao": 38.5, "torres": 12.5, "roshans": 2.5, "barracks": 3.5}
    with c4:
        line = st.number_input("Linha", value=defaults.get(market, 10.5), step=0.5, key="al_line")

    if a_id == b_id:
        st.info("Escolha dois times diferentes.")
        st.stop()
    na, nb = name_by_id[a_id], name_by_id[b_id]

    red_section_header(st, f"{DOTA_MARKETS[market]['label']} — Over {line}")
    ser_a = eng.market_series(a_id, market)
    ser_b = eng.market_series(b_id, market)
    ser_h = eng.market_series(a_id, market, opp_id=b_id)

    def _win_row(label, sa: pd.Series, sb: pd.Series) -> str:
        def cell(s: pd.Series) -> str:
            if len(s) == 0:
                return "<td style='text-align:center;'>—</td><td style='text-align:center;'>—</td>"
            over = (s > line).mean() * 100
            return (f"<td style='text-align:center;'>{s.mean():.1f}</td>"
                    f"<td style='text-align:center;'>{pct_cell_html(over, len(s))}</td>")
        return f"<tr><td style='font-weight:700;'>{label}</td>{cell(sa)}{cell(sb)}</tr>"

    header = (f"<tr><th></th><th colspan='2' style='text-align:center;'>{na}</th>"
              f"<th colspan='2' style='text-align:center;'>{nb}</th></tr>"
              f"<tr><th>Janela</th><th style='text-align:center;'>media</th>"
              f"<th style='text-align:center;'>Over%</th><th style='text-align:center;'>media</th>"
              f"<th style='text-align:center;'>Over%</th></tr>")
    body = "".join([
        _win_row("Ano todo", ser_a, ser_b),
        _win_row("L10", ser_a.head(10), ser_b.head(10)),
        _win_row("L5", ser_a.head(5), ser_b.head(5)),
    ])
    st.markdown(
        f"<table style='width:100%;border-collapse:separate;border-spacing:0 6px;'>"
        f"{header}{body}</table>", unsafe_allow_html=True)

    if len(ser_h):
        over_h = (ser_h > line).mean() * 100
        st.markdown(red_banner("H2H"), unsafe_allow_html=True)
        st.markdown(
            f"<div style='padding:10px 12px;background:var(--gb-panel);"
            f"border:1px solid var(--gb-panel-border);border-radius:0 0 8px 8px;'>"
            f"media {ser_h.mean():.1f} | Over {line}: {pct_cell_html(over_h, len(ser_h))}</div>",
            unsafe_allow_html=True)
    else:
        st.caption("Sem H2H registrado para esse mercado.")
    if DOTA_MARKETS[market]["obj_only"]:
        st.caption("Mercado de objetivos: conta so partidas com dados completos (base em expansao).")

# ---------------- Rankings ----------------
else:
    red_section_header(st, "Rankings WHR (tier professional, ativos 90d)")
    busca = st.text_input("Buscar time", "", key="rk_busca")
    rows = [{
        "#": i + 1, "time": t.name, "rating": round(t.rating_elo),
        "liga recente": t.league[:36], "jogos prof": t.prof_games,
    } for i, t in enumerate(teams)]
    df_rk = pd.DataFrame(rows)
    if busca.strip():
        df_rk = df_rk[df_rk["time"].str.contains(busca.strip(), case=False, na=False)]
    st.dataframe(df_rk, hide_index=True, use_container_width=True, height=700)
