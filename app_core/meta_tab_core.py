"""Página META — o valor dos campeões no patch, na leitura dos próprios pros.

Fonte única do "meta" no app: o Radar consome daqui (fator meta do veredito e
badge de tier no grid de draft). Tabela por campeão: tier, presença (pick+ban),
prioridade (1ª rotação / 1ª fase de ban), WR shrinkado e a seta vs patch anterior.
"""
from __future__ import annotations

from html import escape
from typing import Any, Mapping

import pandas as pd

from app_core.champions_shared_core import (  # noqa: F401  (re-export p/ importadores)
    _champ_icon_uri,
    _get_draft_rows,
    _get_registry,
)
from app_core.meta_patch_core import (
    TIER_COLORS,
    compute_meta_table,
    patches_by_recency,
)
from app_core.tab_ui_core import render_tab_hero, render_tab_section


def _norm_key(value: Any) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _delta_html(delta: Any) -> str:
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return "<span style='opacity:0.4;'>novo</span>"
    if pd.isna(d):
        return "<span style='opacity:0.4;'>novo</span>"
    if d >= 3.0:
        return f"<span style='color:#22c55e;font-weight:700;'>▲ {d:+.0f}pp</span>"
    if d <= -3.0:
        return f"<span style='color:#ef4444;font-weight:700;'>▼ {d:+.0f}pp</span>"
    return f"<span style='opacity:0.55;'>→ {d:+.0f}pp</span>"


def _tier_badge(tier: str) -> str:
    cor = TIER_COLORS.get(str(tier), "#6b7280")
    return (f"<span style='display:inline-block;min-width:22px;text-align:center;"
            f"background:{cor}26;color:{cor};border:1px solid {cor};border-radius:6px;"
            f"font-weight:800;padding:0 6px;'>{escape(str(tier))}</span>")


def render_meta_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(
        st,
        title="Meta",
        subtitle="O valor de cada campeão no patch, medido no que os pros fazem: bans, prioridade de pick e presença.",
        chips=["Presença", "Prioridade", "Subiu/caiu no patch"],
    )
    session_state = st.session_state
    csv_sig = ctx["csv_sig"]

    df = _get_draft_rows(session_state, csv_sig)
    if df is None or df.empty:
        st.warning("Sem dados de draft no CSV.")
        return

    patches = patches_by_recency(df, min_maps=30)
    if not patches:
        st.warning("Nenhum patch com amostra suficiente.")
        return

    with st.container(border=True):
        render_tab_section(st, title="Recorte", subtitle="Patch e liga (Todas = meta global do patch).")
        c1, c2, c3 = st.columns([1.2, 1.2, 2])
        with c1:
            patch_labels = [f"{p}  ({n} mapas)" for p, n in patches]
            sel = st.selectbox("Patch", patch_labels, key="meta_patch_sel")
            patch = patches[patch_labels.index(sel)][0]
        with c2:
            leagues = ["Todas"]
            if "league" in df.columns:
                sub = df[df["patch"].astype(str) == str(patch)]
                leagues += sorted({str(x) for x in sub["league"].dropna().astype(str) if str(x).strip()})
            league = st.selectbox("Liga", leagues, key="meta_league_sel")
        with c3:
            q = st.text_input("Buscar campeão", key="meta_q", placeholder="ex: yunara")

    idx = [p for p, _ in patches].index(patch)
    prev_patch = patches[idx + 1][0] if idx + 1 < len(patches) else None

    tbl = compute_meta_table(df, patch=patch, league=league, prev_patch=prev_patch)
    if tbl.empty:
        st.info("Sem jogos nesse recorte.")
        return

    if q and q.strip():
        qn = _norm_key(q)
        tbl = tbl[tbl["champion"].map(lambda n: qn in _norm_key(n))]

    n_maps = int(tbl["n_maps"].iloc[0]) if not tbl.empty else 0
    prev_txt = f" · comparando com {prev_patch}" if prev_patch else ""
    st.caption(f"Patch {patch} · {n_maps} mapas{prev_txt} · WR com shrink (k=20) — presença manda, WR complementa.")

    # ícones (mesma pasta do grid do Radar; helpers em champions_shared_core)
    registry = _get_registry(session_state)
    name2id = {str(e.get("name") or k): str(e.get("id") or "")
               for k, e in (registry.get("champions") or {}).items()}
    aliases = registry.get("aliases") or {}
    champs_map = registry.get("champions") or {}

    def _icon(name: str) -> str:
        k = _norm_key(name)
        entry = champs_map.get(str(aliases.get(k) or k))
        cid = str(entry.get("id") or "") if isinstance(entry, Mapping) else name2id.get(name, "")
        uri = _champ_icon_uri(cid)
        if not uri:
            return ""
        return f"<img src='{uri}' style='width:26px;height:26px;border-radius:5px;vertical-align:middle;margin-right:6px;'/>"

    header = ("<tr style='opacity:0.6;font-size:0.8em;text-transform:uppercase;letter-spacing:0.04em;'>"
              "<th style='text-align:left;padding:4px 8px;'>Tier</th>"
              "<th style='text-align:left;padding:4px 8px;'>Campeão</th>"
              "<th style='text-align:right;padding:4px 8px;'>Presença</th>"
              "<th style='text-align:right;padding:4px 8px;'>Pick</th>"
              "<th style='text-align:right;padding:4px 8px;'>Ban</th>"
              "<th style='text-align:right;padding:4px 8px;'>1ª rotação</th>"
              "<th style='text-align:right;padding:4px 8px;'>WR</th>"
              "<th style='text-align:right;padding:4px 8px;'>vs anterior</th></tr>")
    body = []
    for r in tbl.head(80).itertuples(index=False):
        body.append(
            "<tr style='border-top:1px solid rgba(255,255,255,0.06);'>"
            f"<td style='padding:4px 8px;'>{_tier_badge(r.tier)}</td>"
            f"<td style='padding:4px 8px;white-space:nowrap;'>{_icon(r.champion)}<b>{escape(str(r.champion))}</b></td>"
            f"<td style='text-align:right;padding:4px 8px;font-weight:700;'>{r.presence:.0f}%</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{r.pick_pct:.0f}%</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{r.ban_pct:.0f}%</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{r.early_pick_share:.0f}%</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{r.wr_shrunk:.0f}%</td>"
            f"<td style='text-align:right;padding:4px 8px;'>{_delta_html(r.presence_delta)}</td>"
            "</tr>"
        )
    st.markdown(
        "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;font-size:0.92em;'>"
        + header + "".join(body) + "</table></div>",
        unsafe_allow_html=True,
    )
