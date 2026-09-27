"""Renderer for the Resumo odds section inside Consistencia tab."""

from __future__ import annotations

from html import escape as escape_html
from typing import Any, Mapping

import pandas as pd

from app_core.championship_stats_core import enrich_resumo_rows_with_stats_detail

from app_core.resumo_runtime_core import SC_METRIC_COL, merge_with_default_lines, select_h2h_games, v3_lines_for_metric
from app_core.resumo_state_core import (
    get_resumo_view_state,
    set_resumo_market_trace,
)
from app_core.resumo_core import odd_to_prob
from app_core.picks_tracker_core import (
    load_picks,
    new_pick,
    remove_pick,
    resolve_open_picks,
    roi_summary,
    save_picks,
    upsert_pick,
)
from app_core.picks_anti_loss_core import (
    market_historical_roi,
)
from app_core.resumo_score_core import (
    TemporalProfile,
    generate_veredito,
    rank_candidates,
    _METRIC_LABEL_PT as _SCORE_METRIC_LABELS,
)
from app_core.resumo_compare_lines_core import build_line_comparisons, sparkline


PICKS_TRACKER_PATH = "ops/picks_tracker.json"


def _parse_odd_input(text: Any) -> float | None:
    """Parse a user-typed odd accepting comma decimals (1,83 -> 1.83).

    Returns None for empty / unparseable / placeholder values.
    """
    if text is None:
        return None
    s = str(text).strip().replace(",", ".")
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _row_to_pick_payload(r: dict, team_a: str, team_b: str, league: str) -> dict:
    """Derive canonical metric_key / side_key / line for a resumo row."""
    metric = str(r.get("_metric") or "").strip()
    # Strip combo and ml prefixes
    if metric.endswith("_ml_A") or metric == "ml_A":
        metric_key = "ml"; side_key = "A"; line = None
    elif metric.endswith("_ml_B") or metric == "ml_B":
        metric_key = "ml"; side_key = "B"; line = None
    else:
        metric_key = metric
        lado = str(r.get("Lado") or "").strip().upper()
        side_key = "Over" if lado in ("ACIMA", "OVER") else ("Under" if lado in ("ABAIXO", "UNDER") else lado)
        line = r.get("_line")
    return {
        "team_a": team_a, "team_b": team_b, "league": league,
        "metric_key": metric_key, "side_key": side_key,
        "line": float(line) if isinstance(line, (int, float)) and line is not None else None,
    }


def _ctx(ctx: Mapping[str, Any], key: str) -> Any:
    return ctx[key]


def _norm_team_key(value: Any) -> str:
    """Normalize a team name for matching (tolerates mojibake/casing/spacing).

    Mirrors the matching used in Análise de Linha so the same opponent resolves
    the same way in both tabs.
    """
    s = str(value or "").strip().replace("�", "").replace("?", "")
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return s.lower().strip()


def _safe_rerun(st: Any) -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def _render_meu_historico(*, st: Any, team_games) -> None:
    """Bottom-of-tab block: open picks, settled picks, ROI summary."""
    with st.expander("📒 Meu Histórico (picks salvas)", expanded=False):
        picks = load_picks(PICKS_TRACKER_PATH)
        if not picks:
            st.caption("Nenhuma pick salva ainda. Clique em ✓ Apostei num card acima para começar.")
            return
        # Auto-resolve any open picks against the current team_games.
        picks, n_settled = resolve_open_picks(picks, team_games)
        if n_settled > 0:
            save_picks(PICKS_TRACKER_PATH, picks)
            st.info(f"{n_settled} pick(s) liquidada(s) automaticamente com base nos jogos novos.")

        s = roi_summary(picks)
        cols = st.columns(5)
        with cols[0]:
            st.metric("Abertas", s["n_open"])
        with cols[1]:
            st.metric("W-L-V", f"{s['wins']}-{s['losses']}-{s['voids']}")
        with cols[2]:
            roi = s["roi_pct"]
            st.metric("ROI", f"{roi:+.1f}%" if roi is not None else "—")
        with cols[3]:
            st.metric("Profit (u)", f"{s['profit_units']:+.2f}")
        with cols[4]:
            st.metric("Risco (u)", f"{s['risked_units']:.2f}")

        # By-metric breakdown
        if s["by_metric"]:
            st.markdown("**Por mercado**")
            import pandas as pd
            rows_bm = []
            for k, v in s["by_metric"].items():
                rows_bm.append({
                    "Mercado": k,
                    "Jogos": v["n"], "Wins": v["wins"],
                    "Win %": f"{v['winrate_pct']:.0f}%" if v["winrate_pct"] is not None else "—",
                    "Profit": f"{v['profit']:+.2f}",
                    "ROI": f"{v['roi_pct']:+.1f}%" if v["roi_pct"] is not None else "—",
                })
            st.dataframe(pd.DataFrame(rows_bm), width="stretch", hide_index=True)

        # Pick list (most recent first) with remove button
        st.markdown("**Picks salvas**")
        ordered = sorted(picks, key=lambda p: p.ts, reverse=True)
        for p in ordered[:30]:
            status_icon = {"open": "⏳", "won": "✅", "lost": "❌", "void": "➖"}.get(p.status, "?")
            line_str = f" {p.line:g}" if p.line is not None else ""
            profit_str = (f" · profit {p.profit_units:+.2f}u" if p.profit_units is not None else "")
            row_cols = st.columns([6, 1])
            with row_cols[0]:
                st.text(f"{status_icon} {p.ts[:16]} · {p.team_a} vs {p.team_b} · "
                        f"{p.metric_key} {p.side_key}{line_str} @ {p.odd_casa:.2f} · "
                        f"{p.stake_units:g}u{profit_str}")
            with row_cols[1]:
                if st.button("🗑", key=f"del_pick_{p.id}", help="Remover pick"):
                    remaining = remove_pick(picks, p.id)
                    save_picks(PICKS_TRACKER_PATH, remaining)
                    st.rerun()
        if len(ordered) > 30:
            st.caption(f"…e mais {len(ordered) - 30} picks anteriores não exibidas.")


_METRIC_LABEL_PT = {
    "kills": "Kills",
    "towers": "Torres",
    "dragons": "Dragões",
    "barons": "Barões",
    "inhibitors": "Inibidores",
    "time": "Tempo (min)",
    "ml": "Vencedor (ML)",
}

_METRIC_ORDER = ["ml", "kills", "towers", "dragons", "barons", "inhibitors", "time"]


def _metric_leaf(metric_raw: str) -> str:
    """Reduce '00_ml_A' / 'zz_mltot_A_kills' / 'kills' to a leaf key."""
    m = str(metric_raw or "")
    if m.endswith("_ml_A") or m == "ml_A":
        return "ml"
    if m.endswith("_ml_B") or m == "ml_B":
        return "ml"
    if m.startswith("zz_mltot_"):
        # combo — keep as is, will live under "Combo (ML+Total)" bucket
        return "combo"
    return m


def _side_norm(lado: str) -> str:
    s = str(lado or "").strip().upper()
    if s in ("ACIMA", "OVER"):
        return "Over"
    if s in ("ABAIXO", "UNDER"):
        return "Under"
    return str(lado or "")


def _temporal_bars_html(tp: TemporalProfile) -> str:
    """Compact HTML row: Ano → L15 → L10 → L5 with color-coded % badges."""
    parts: list[str] = []
    for w in tp.windows:
        if w.pct is None or w.n == 0:
            badge = (f"<span style='background:rgba(156,163,175,0.15);border-radius:5px;"
                     f"padding:3px 9px;font-size:0.95em;opacity:0.6;'>{w.label} —</span>")
        else:
            color = "#22c55e" if w.pct >= 65 else ("#facc15" if w.pct >= 40 else "#ef4444")
            badge = (f"<span style='background:rgba(45,170,196,0.12);border-radius:5px;"
                     f"padding:3px 9px;font-size:0.95em;'>"
                     f"{w.label} <b style='color:{color};font-size:1.05em;'>{w.pct:.0f}%</b>"
                     f"<span style='opacity:0.55;font-size:0.88em;'> ({w.hits}/{w.n})</span></span>")
        parts.append(badge)
    arrow = {"recovering": "⬆", "improving": "↑", "flat": "→", "declining": "↓", "volatile": "⟳"}.get(tp.trend, "")
    trend_color = ("#22c55e" if tp.trend in ("recovering", "improving")
                   else "#ef4444" if tp.trend == "declining" else "#9ca3af")
    trend_badge = (f"<span style='font-size:0.95em;color:{trend_color};font-weight:700;'>"
                   f"&nbsp;{arrow} {tp.trend_label}</span>")
    return " &nbsp; ".join(parts) + " &nbsp;&nbsp; " + trend_badge


def _render_candidate_cards(
    *, st: Any, ranked: list, oc_map_cur: dict[str, float], matchup_key_oc: str,
    team_a: str = "", team_b: str = "", league: str = "", sort_label: str = "",
) -> None:
    """Render each scored candidate as a card with temporal breakdown."""
    if not ranked:
        st.info("Nenhuma pick passou no filtro atual. Ajuste os filtros na seção ⚙️ Ajuste fino.")
        return

    _ordenado = f"ordenado por {escape_html(sort_label)}" if sort_label else "melhores por score"
    st.markdown(
        "<div style='margin-top:8px;background:linear-gradient(180deg,#2daac4 0%,#1f8ba1 100%);"
        "padding:8px 14px;border-radius:8px;font-weight:700;letter-spacing:0.05em;"
        "font-size:0.92em;color:white;text-transform:uppercase;'>"
        f"Picks Analisadas ({_ordenado})</div>",
        unsafe_allow_html=True,
    )

    for i, c in enumerate(ranked):
        row = c.row
        # Medals for the top 3; plain numbers after (safe for any count).
        rank_icon = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i + 1}."

        mercado = str(row.get("Mercado", ""))
        lado    = str(row.get("Lado", ""))
        linha   = str(row.get("Linha", ""))
        label   = f"{mercado} · {lado}" + (f" · {linha}" if linha and linha != "-" else "")

        score_color = "#22c55e" if c.total >= 70 else ("#facc15" if c.total >= 50 else "#9ca3af")

        with st.container(border=True):
            # Título do card (largura total).
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:10px;margin-bottom:6px;'>"
                f"<span style='font-size:1.1em;'>{rank_icon}</span>"
                f"<span style='font-weight:700;font-size:1.05em;color:#2daac4;'>{escape_html(label)}</span>"
                f"<span style='font-size:0.9em;color:{score_color};font-weight:700;'>Score {c.total}/100</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Linha de cálculo em ordem lógica:
            # Hist (referência) → Sua odd (input) → Edge → Stake → ROI hist.
            row_key = f"{row.get('Mercado','')}|{row.get('Lado','')}|{row.get('Linha','')}"
            oc_key  = f"oc_{matchup_key_oc}|{row_key}"
            cur_val = oc_map_cur.get(row_key)

            def _col_label(txt: str) -> str:
                return (f"<div style='font-size:0.7em;opacity:0.55;text-transform:uppercase;"
                        f"letter-spacing:0.04em;margin-bottom:2px;'>{txt}</div>")

            oc1, oc2, oc3, oc4, oc5 = st.columns([1.1, 1.3, 1.0, 1.0, 1.1])
            with oc1:
                hist_txt = f"{c.odd_hist:.2f}" if c.odd_hist else "—"
                st.markdown(
                    _col_label("Hist (ref.)")
                    + f"<div style='color:#2daac4;font-weight:700;font-size:1.3em;'>{hist_txt}</div>",
                    unsafe_allow_html=True,
                )
            with oc2:
                st.text_input(
                    "Sua odd (da casa)",
                    value=(f"{float(cur_val):.2f}" if cur_val is not None else ""),
                    key=oc_key,
                    placeholder="ex: 1,83",
                    help="Digite a odd da casa. Aceita vírgula (1,83 = 1.83). Vazio = não preenchida.",
                )
            with oc3:
                st.markdown(_col_label("Edge"), unsafe_allow_html=True)
                if c.odd_casa and c.edge_hist is not None:
                    e_color = "#22c55e" if c.edge_hist >= 5 else ("#facc15" if c.edge_hist >= 0 else "#ef4444")
                    st.markdown(
                        f"<div style='color:{e_color};font-weight:700;font-size:1.2em;'>{c.edge_hist:+.1f}%</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='opacity:0.4;font-size:1.2em;'>—</div>", unsafe_allow_html=True)
            with oc4:
                st.markdown(_col_label("Stake"), unsafe_allow_html=True)
                if c.stake_units:
                    st.markdown(
                        f"<div style='color:#2daac4;font-weight:700;font-size:1.2em;'>{c.stake_units:g}u</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='opacity:0.4;font-size:1.2em;'>—</div>", unsafe_allow_html=True)
            with oc5:
                roi = market_historical_roi(
                    str(row.get("_metric") or ""),
                    "Over" if lado.upper() in ("ACIMA","OVER") else "Under",
                )
                st.markdown(_col_label("ROI hist"), unsafe_allow_html=True)
                if roi is not None:
                    rc = "#22c55e" if roi > 0 else "#ef4444"
                    st.markdown(
                        f"<div style='color:{rc};font-weight:700;font-size:1.05em;'>{roi:+.1f}%</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='opacity:0.4;'>—</div>", unsafe_allow_html=True)

            # ── Recortes Liga e H2H (core do critério 4/4) ──────────────────
            _all_recortes = ["Ano", "15", "10", "5"]
            liga_passed = set(str(row.get("Liga") or "").replace(" ", "").split(","))
            h2h_passed  = set(str(row.get("H2H")  or "").replace(" ", "").split(","))

            def _recorte_badges(passed_set: set, prefix: str) -> str:
                chips: list[str] = []
                for rec in _all_recortes:
                    ok = rec in passed_set and rec != "-"
                    bg  = "rgba(34,197,94,0.18)"  if ok else "rgba(239,68,68,0.12)"
                    col = "#22c55e" if ok else "#ef4444"
                    sym = "✅" if ok else "❌"
                    chips.append(
                        f"<span style='background:{bg};border-radius:5px;"
                        f"padding:2px 8px;font-size:0.95em;color:{col};'>"
                        f"{sym} {rec}</span>"
                    )
                return f"<span style='font-size:0.9em;opacity:0.7;font-weight:600;'>{prefix}</span> " + " ".join(chips)

            lap_liga = str(row.get("Laplace (Liga)") or "—")
            lap_h2h  = str(row.get("Laplace (H2H)")  or "—")

            # H2H sem NENHUM jogo no histórico (confronto inédito) não é "ruim":
            # é ausência de dado. Em vez dos 4 ❌ (que parecem reprovação),
            # mostramos uma barra lateral neutra. Quando há jogos mas as odds
            # ficam fora do limite, os ❌ continuam (reprovação legítima).
            try:
                _n_h2h = int(row.get("Amostra H2H") or 0)
            except (TypeError, ValueError):
                _n_h2h = 0
            if _n_h2h <= 0:
                _h2h_block = (
                    "<div style='margin:2px 0 6px;padding:4px 10px;line-height:1.8;"
                    "border-left:3px solid #6b7280;background:rgba(107,114,128,0.10);"
                    "border-radius:4px;'>"
                    "<span style='font-size:0.9em;opacity:0.7;font-weight:600;'>H2H</span>"
                    "&nbsp;<span style='font-size:0.88em;opacity:0.6;'>"
                    "sem confronto direto no histórico</span>"
                    "</div>"
                )
            else:
                _h2h_block = (
                    "<div style='margin:2px 0 6px;line-height:2.0;'>"
                    f"{_recorte_badges(h2h_passed, 'H2H ')}"
                    f"&nbsp;<span style='opacity:0.5;font-size:0.88em;'>odd: {lap_h2h}</span>"
                    "</div>"
                )
            st.markdown(
                f"<div style='margin:8px 0 2px;line-height:2.0;'>"
                f"{_recorte_badges(liga_passed, 'Liga')}"
                f"&nbsp;<span style='opacity:0.5;font-size:0.88em;'>odd: {lap_liga}</span>"
                f"</div>"
                f"{_h2h_block}",
                unsafe_allow_html=True,
            )

            # Score decomposition bar.
            st.markdown(
                f"<div style='font-size:0.92em;opacity:0.75;margin:6px 0 4px;"
                f"padding:5px 10px;background:rgba(45,170,196,0.05);border-radius:6px;'>"
                f"<span style='opacity:0.7;'>Conf</span> <b>{c.conf_pts:.0f}</b>"
                f" + <span style='opacity:0.7;'>Edge</span> <b>{c.edge_pts:.0f}</b>"
                f" + <span style='opacity:0.7;'>ROI</span> <b>{c.roi_pts:.0f}</b>"
                f" + <span style='opacity:0.7;'>Trend</span> <b>{c.trend_bonus:+.0f}</b>"
                f" + <span style='opacity:0.7;'>Anti-loss</span> <b>{c.anti_loss_penalty:.0f}</b>"
                f" = <b style='color:#2daac4;font-size:1.05em;'>{c.total}</b></div>",
                unsafe_allow_html=True,
            )

            # Temporal analysis — both teams.
            for tp in (c.temporal_a, c.temporal_b):
                if tp.windows and any(w.n > 0 for w in tp.windows):
                    st.markdown(
                        f"<div style='margin:5px 0;line-height:2;'>"
                        f"<span style='opacity:0.65;font-size:0.9em;font-weight:600;'>"
                        f"{escape_html(tp.team)}:</span>&nbsp;&nbsp;"
                        f"{_temporal_bars_html(tp)}</div>",
                        unsafe_allow_html=True,
                    )



def _render_bet_button(
    *, st: Any, c: Any, row_key: str, matchup_key_oc: str,
    team_a: str, team_b: str, league: str,
) -> None:
    """Render the '✓ Apostei' action for one candidate card.

    Only enabled when 'Sua odd' is filled and a discrete stake suggestion
    exists. Persists to ops/picks_tracker.json with source='manual', so it
    feeds the live ROI overlay and the 'Meu Histórico' block.
    """
    row = c.row
    oc = c.odd_casa
    units = c.stake_units
    if not oc or oc <= 1.0:
        st.caption("Preencha **Sua odd** para registrar a aposta.")
        return
    if not units:
        st.caption("Stake abaixo de 0.5u — não recomendado apostar.")
        return

    key = f"bet_{matchup_key_oc}|{row_key}"
    if st.button(f"✓ Apostei {units:g}u", key=key, width="stretch"):
        payload = _row_to_pick_payload(row, team_a, team_b, league)
        supported = ("kills", "towers", "dragons", "barons", "inhibitors", "time", "ml")
        if payload["metric_key"] not in supported:
            st.warning(f"Mercado {payload['metric_key']} não suportado pelo tracker.")
            return
        pick = new_pick(
            **payload, odd_casa=float(oc), stake_units=float(units),
            p_hist=odd_to_prob(c.odd_hist),
            # O que o MODELO disse, separado da frequencia historica. Sem isto
            # nao ha como responder depois se o modelo ganha dinheiro -- ver a
            # explicacao em app_core/picks_tracker_core.Pick.
            p_modelo=odd_to_prob(row.get("Odd (Modelo)")),
            momento="pre",
            edge_hist_pct=c.edge_hist,
        )
        picks = load_picks(PICKS_TRACKER_PATH)
        picks = upsert_pick(picks, pick)
        save_picks(PICKS_TRACKER_PATH, picks)
        # Invalidate the live-ROI cache so the new pick is reflected next run.
        st.session_state.pop("_resumo_live_roi_loaded", None)
        st.success(f"Pick registrada: {pick.id}")


def _render_line_comparisons(*, st: Any, rows: list[dict]) -> None:
    """Sweet-spot finder: para cada métrica com 2+ linhas, mostra a probabilidade
    histórica por linha + sparkline, destacando a linha mais segura."""
    groups = build_line_comparisons(rows)
    if not groups:
        return

    st.markdown(
        "<div style='margin-top:18px;background:linear-gradient(180deg,#2daac4 0%,#1f8ba1 100%);"
        "padding:8px 14px;border-radius:8px;font-weight:700;letter-spacing:0.05em;"
        "font-size:0.92em;color:white;text-transform:uppercase;'>"
        "Comparar linhas (sweet spot)</div>",
        unsafe_allow_html=True,
    )
    st.caption("Quando você tem várias linhas da mesma métrica, veja a probabilidade "
               "histórica (1/odd Hist) de cada uma. ⭐ marca a mais provável.")

    for g in groups:
        label = _SCORE_METRIC_LABELS.get(g["metric"], g["metric"].title())
        spark = sparkline([ln.get("p_hist") for ln in g["lines"]])
        with st.container(border=True):
            st.markdown(
                f"<div style='font-weight:700;color:#2daac4;'>{escape_html(label)} {escape_html(g['side'])}"
                f" <span style='opacity:0.6;font-weight:400;font-size:0.85em;'>{spark}</span></div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(g["lines"]))
            for i, (col, ln) in enumerate(zip(cols, g["lines"])):
                with col:
                    is_best = (i == g["best_idx"])
                    star = "⭐ " if is_best else ""
                    border = "#22c55e" if is_best else "rgba(255,255,255,0.12)"
                    p = ln.get("p_hist")
                    p_txt = f"{p*100:.0f}%" if p is not None else "—"
                    odd_txt = f"{ln['odd_hist']:.2f}" if ln.get("odd_hist") else "—"
                    st.markdown(
                        f"<div style='border:1px solid {border};border-radius:8px;padding:8px;text-align:center;'>"
                        f"<div style='font-size:0.75em;opacity:0.6;'>Linha</div>"
                        f"<div style='font-weight:700;font-size:1.1em;'>{star}{ln['line']:g}</div>"
                        f"<div style='font-size:0.8em;opacity:0.7;margin-top:4px;'>Hist {odd_txt}</div>"
                        f"<div style='color:#2daac4;font-weight:700;'>{p_txt}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


def _render_veredito_block(
    *, st: Any, ranked: list, team_a: str, team_b: str,
) -> None:
    """Render the verdict block: winner announcement + pros/cons + tradeoff."""
    if not ranked:
        return

    v = generate_veredito(ranked, team_a=team_a, team_b=team_b)

    st.markdown(
        "<div style='margin-top:18px;background:linear-gradient(180deg,#2daac4 0%,#1f8ba1 100%);"
        "padding:8px 14px;border-radius:8px;font-weight:700;letter-spacing:0.05em;"
        "font-size:0.92em;color:white;text-transform:uppercase;'>"
        "🏆 Veredito</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:1.1em;font-weight:700;color:#2daac4;margin-bottom:8px;'>"
            f"Melhor aposta: {escape_html(v['winner_label'])}"
            f"<span style='font-size:0.82em;opacity:0.7;font-weight:400;margin-left:10px;'>"
            f"Score {v['winner_score']}/100</span></div>",
            unsafe_allow_html=True,
        )

        if v["stake_str"] and "preencha" not in v["stake_str"]:
            st.markdown(f"**Stake sugerido:** {v['stake_str']}")
        else:
            st.caption("Preencha **Sua odd** acima para ver stake sugerido.")

        if v["pros"]:
            st.markdown("**Por que esta pick:**")
            for p in v["pros"]:
                st.markdown(f"✅ {p}")

        if v["cons"]:
            st.markdown("**Atenção:**")
            for c_item in v["cons"]:
                st.markdown(f"⚠️ {c_item}")

        if v["tradeoff"]:
            st.markdown(
                f"<div style='background:rgba(250,204,21,0.10);border-left:3px solid #facc15;"
                f"padding:8px 12px;border-radius:6px;margin-top:8px;font-size:0.9em;'>"
                f"🔁 {escape_html(v['tradeoff'])}</div>",
                unsafe_allow_html=True,
            )


def _render_diagnostic(
    *, st: Any, n_liga: int, n_h2h: int,
    min_sample_liga: int, min_sample_h2h: int,
    req_liga: bool, req_h2h: bool, req_model: bool,
    max_odd: float,
) -> None:
    """Explain WHY no row passed so the user can adjust faster.

    Builds a checklist of the active gates and which ones are blocking,
    based on the sample sizes available."""
    blockers: list[str] = []
    if req_h2h and n_h2h < min_sample_h2h:
        blockers.append(f"Amostra **H2H** insuficiente ({n_h2h} < {min_sample_h2h}) — desmarque *Exigir H2H* ou baixe o mínimo.")
    if req_liga and n_liga < min_sample_liga:
        blockers.append(f"Amostra **Liga** insuficiente ({n_liga} < {min_sample_liga}) — abra o recorte (split, ano) ou baixe o mínimo.")
    if not blockers and req_model:
        blockers.append(f"Filtro **Odd máxima ≤ {max_odd:.2f}** pode estar cortando todas as picks do Modelo. Suba o teto ou desmarque *Exigir Modelo*.")
    if not blockers:
        blockers.append("Nenhuma linha foi gerada. Verifique se há linhas preenchidas no painel principal.")

    st.warning("Nada passou no filtro. Possíveis causas:")
    for b in blockers:
        st.markdown(f"- {b}")
    st.caption(
        f"Configuração ativa: odd_max={max_odd:.2f} · "
        f"Liga {'ON' if req_liga else 'off'} (min={min_sample_liga}, atual={n_liga}) · "
        f"H2H {'ON' if req_h2h else 'off'} (min={min_sample_h2h}, atual={n_h2h}) · "
        f"Modelo {'ON' if req_model else 'off'}"
    )


def render_resumo_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    session_state = st.session_state
    view_state = get_resumo_view_state(session_state)

    # Install live ROI overlay from the tracked picks (xlsx backfill + manual).
    # Once per session is enough; the tracker is stable within a run. Falls back
    # silently to the frozen snapshot when there's no tracker yet.
    if not session_state.get("_resumo_live_roi_loaded"):
        try:
            from app_core.picks_anti_loss_core import load_live_roi_from_tracker
            load_live_roi_from_tracker()
        except Exception:
            pass
        session_state["_resumo_live_roi_loaded"] = True
    list_presets = _ctx(ctx, "list_resumo_filter_presets_fn")
    for key, default in {
        "resumo_max_odd": float(view_state["resumo_max_odd"]),
        "resumo_profile_name_new": "",
        "resumo_filter_preset": str(list_presets(session_state)[2] or "Custom"),
        "resumo_req_liga": bool(view_state["resumo_req_liga"]),
        "resumo_req_h2h": bool(view_state["resumo_req_h2h"]),
        "resumo_req_model": bool(view_state["resumo_req_model"]),
        "resumo_show_laplace_cols": bool(view_state["resumo_show_laplace_cols"]),
        "resumo_sc_thr": int(view_state["resumo_sc_thr"]),
        "resumo_sc_use": bool(view_state["resumo_sc_use"]),
        "resumo_laplace_pick": str(view_state["resumo_laplace_pick"]),
        "resumo_mix_w_model": int(view_state["resumo_mix_w_model"]),
        "resumo_mix_w_laplace": int(view_state["resumo_mix_w_laplace"]),
        "resumo_mix_w_sc": int(view_state["resumo_mix_w_sc"]),
        "resumo_combo_preset": str(view_state["resumo_combo_preset"]),
        "resumo_combo_hist_w": float(view_state["resumo_combo_hist_w"]),
        "resumo_combo_style_w": float(view_state["resumo_combo_style_w"]),
        "resumo_combo_form_w": float(view_state["resumo_combo_form_w"]),
    }.items():
        if key not in session_state:
            session_state[key] = default
    team_a = str(_ctx(ctx, "teamA"))
    team_b = str(_ctx(ctx, "teamB"))
    parse_lines = _ctx(ctx, "parse_lines_fn")
    parse_time_lines = _ctx(ctx, "parse_time_lines_fn")
    odd_from_prob = _ctx(ctx, "odd_from_prob_fn")
    fmt_recortes_disp = _ctx(ctx, "fmt_recortes_disp_fn")
    fmt_line_disp = _ctx(ctx, "fmt_line_disp_fn")
    metric_label = _ctx(ctx, "metric_label_fn")
    side_label = _ctx(ctx, "side_label_fn")
    total_over_prob = _ctx(ctx, "total_over_prob_fn")
    persist_combined_trace_file = _ctx(ctx, "persist_combined_trace_file_fn")
    team_games = _ctx(ctx, "team_games")
    filters_cls = _ctx(ctx, "Filters")
    active_filters = _ctx(ctx, "filters")
    apply_filters = _ctx(ctx, "apply_filters_fn")
    ml_info = _ctx(ctx, "ml_info")
    map_mode_ui = ctx.get("map_mode_ui")
    totals_for_lines = _ctx(ctx, "totals_for_lines")
    df_a_used = _ctx(ctx, "dfA_used")
    df_b_used = _ctx(ctx, "dfB_used")

    ensure_combo_state = _ctx(ctx, "ensure_resumo_combo_state_fn")
    apply_combo_preset = _ctx(ctx, "apply_resumo_combo_preset_fn")

    build_statscamps_scope_frame = _ctx(ctx, "build_statscamps_scope_frame_fn")
    compute_statscamps_pct = _ctx(ctx, "compute_statscamps_pct_fn")
    format_statscamps_badge = _ctx(ctx, "format_statscamps_badge_fn")
    recency_slices = _ctx(ctx, "recency_slices_fn")
    make_pricing_adapters = _ctx(ctx, "make_pricing_adapters_fn")
    ensure_metric_lines = _ctx(ctx, "ensure_metric_lines_fn")
    infer_matchup_league = _ctx(ctx, "infer_matchup_league_fn")
    build_resumo_guardrail_cfg = _ctx(ctx, "build_resumo_guardrail_cfg_fn")
    build_ml_rows = _ctx(ctx, "build_ml_rows_fn")
    build_metric_rows = _ctx(ctx, "build_metric_rows_fn")
    build_combo_rows = _ctx(ctx, "build_combo_rows_fn")
    build_market_trace_rows = _ctx(ctx, "build_market_trace_rows_fn")
    wr_laplace = _ctx(ctx, "wr_laplace_fn")
    counts_ml_totals = _ctx(ctx, "counts_ml_totals_fn")
    tail_df = _ctx(ctx, "tail_df_fn")

    st.markdown("### Resumo de Odds")

    # ── Controles principais ────────────────────────────────────────────────
    # Fonte do Histórico é o filtro central do fluxo de decisão:
    # se a pick aparecer aqui com "Média Liga+H2H 5" → critério 4/4 cumprido.
    # ORDEM IMPORTA: o ÍNDICE 0 é "Média Liga+H2H 5" de propósito. O Streamlit, ao
    # rerenderizar o selectbox dentro de st.tabs, às vezes reseta o widget pro
    # índice 0 no início do rerun (antes do nosso código). Por isso o índice 0
    # TEM que ser o default de todo o app (app_state + _VIEW_DEFAULTS = "Média
    # Liga+H2H 5"); se divergir, a HIST "pula" no 1º render. Mantenha os três
    # alinhados em "Média Liga+H2H 5".
    _laplace_pick_opts = [
        "Média Liga+H2H 5",
        "Média Liga+H2H 10",
        "Média Liga+H2H 15",
        "Média Liga+H2H Ano",
        "Liga 5",
        "Liga 10",
        "Liga 15",
        "Liga Ano",
        "H2H 5",
        "H2H 10",
        "H2H 15",
        "H2H Ano",
    ]
    pc1, pc2 = st.columns([2, 1])
    with pc1:
        if str(session_state.get("resumo_laplace_pick", "") or "") not in _laplace_pick_opts:
            # MESMO default do resto do app (app_state + _VIEW_DEFAULTS + índice 0
            # das opções) = "Média Liga+H2H 5". Tudo alinhado → reset do widget
            # cai sempre no mesmo valor, sem a HIST pular.
            session_state["resumo_laplace_pick"] = "Média Liga+H2H 5"
        st.selectbox(
            "Fonte do Histórico",
            options=_laplace_pick_opts,
            key="resumo_laplace_pick",
            help="Escolha a janela de referência. 'Média Liga+H2H 5' = os últimos 5 jogos de cada time na liga + os últimos 5 H2H, média das duas. Se a pick aparecer com esta fonte → critério de entrada cumprido.",
        )
    with pc2:
        st.number_input(
            "Odd máxima",
            min_value=1.01, max_value=5.0, step=0.05,
            key="resumo_max_odd",
        )
    st.caption(
        "Picks que aparecem abaixo passaram no filtro com a **fonte selecionada**. "
        "Cada card mostra quais janelas (Ano / 15 / 10 / 5) passam para Liga e H2H."
    )

    # Ponte: junta as linhas que o usuário adicionou na aba Análise de Linha (v3)
    # às linhas legadas do ctx (hoje vazias, pois a tela de linhas foi desativada).
    # Sem isso, qualquer linha digitada na Análise de Linha nunca chegava ao Resumo.
    def _with_v3(metric_key: str, base_vals: Any) -> list[float]:
        base = [float(x) for x in (base_vals or []) if isinstance(x, (int, float))]
        for v in v3_lines_for_metric(session_state, metric_key, team_a, team_b):
            if not any(abs(float(v) - float(x)) < 1e-9 for x in base):
                base.append(float(v))
        return base

    lines_by_metric = {
        "kills": _with_v3("kills", _ctx(ctx, "kills")),
        "towers": _with_v3("towers", _ctx(ctx, "towers")),
        "dragons": _with_v3("dragons", _ctx(ctx, "dragons")),
        "barons": _with_v3("barons", _ctx(ctx, "barons")),
        "inhibitors": _with_v3("inhibitors", _ctx(ctx, "inhibitors")),
        "time": _with_v3("time", _ctx(ctx, "time_m")),
    }
    # Mercados BINÁRIOS do registro central (ex.: Megacreeps no Dota): linha
    # fixa 0.5 — a pick Sim/Não compete no ranking pelas MESMAS regras (teto,
    # score). No LoL o registro não tem binários → no-op. (Pedido 2026-07-14.)
    try:
        from app_core.stats_config import STATS_MARKET_CONFIG as _smc_bin
        for _cfg_bin in _smc_bin.values():
            _mk_bin = str(_cfg_bin.get("metric_key") or "")
            if _mk_bin and bool(_cfg_bin.get("binary_market")) and _mk_bin not in lines_by_metric:
                lines_by_metric[_mk_bin] = [0.5]
    except Exception:
        pass

    # Valores de cálculo lidos do estado (sem UI de "ajuste fino" — removida).
    # Sem gates "Exigir": mostramos todas as picks acima de um piso mínimo de
    # amostra; os ✅/❌ por recorte comunicam a qualidade e o score penaliza
    # amostra baixa.
    sc_use = bool(view_state["resumo_sc_use"])
    sc_thr = float(view_state["resumo_sc_thr"])
    laplace_pick = str(view_state["resumo_laplace_pick"])
    w_mix_model = float(view_state["resumo_mix_w_model"])
    w_mix_lap = float(view_state["resumo_mix_w_laplace"])
    w_mix_sc = float(view_state["resumo_mix_w_sc"])
    max_odd = float(view_state["resumo_max_odd"])
    req_liga = False
    req_h2h = False
    req_model = False

    with st.expander("Ajuste ML + Totais (favorito + estilo) — só para combos", expanded=False):
        ensure_combo_state(session_state)
        combo_preset_options = ["Conservador", "Neutro", "Agressivo"]
        combo_preset = str(view_state["resumo_combo_preset"])
        combo_idx = combo_preset_options.index(combo_preset) if combo_preset in combo_preset_options else 1
        if str(session_state.get("resumo_combo_preset", "") or "") not in combo_preset_options:
            session_state["resumo_combo_preset"] = combo_preset_options[combo_idx]
        st.selectbox(
            "Preset de ajuste",
            options=combo_preset_options,
            key="resumo_combo_preset",
            on_change=lambda: apply_combo_preset(session_state),
        )
        st.checkbox(
            "Aplicar mismatch no Modelo (favoritismo vs total, experimental)",
            key="resumo_combo_use_mismatch",
            help="Desligado (padrão): o Modelo de ML+Totais não recebe ajuste de mismatch. Histórico/Frequência permanece puro.",
        )
        st.slider("Peso do histórico condicional (win/loss) no ML+Totais", 0.0, 1.0, step=0.05, key="resumo_combo_hist_w")
        st.slider("Força do viés por estilo (over/under)", 0.0, 0.5, step=0.02, key="resumo_combo_style_w")
        st.slider("Peso da forma atual (força do time no recorte) no ML+Totais", 0.0, 0.40, step=0.02, key="resumo_combo_form_w")
        st.caption("Ajustes de mismatch afetam apenas o Modelo. Histórico/Frequência não recebe mismatch.")

    combo_hist_w = float(view_state["resumo_combo_hist_w"])
    combo_style_w = float(view_state["resumo_combo_style_w"])
    combo_form_w = float(view_state["resumo_combo_form_w"])
    combo_use_mismatch = bool(view_state["resumo_combo_use_mismatch"])
    # Piso fixo de sanidade (não exposto na UI): corta lixo estatístico mas
    # mostra tudo o que tem amostra mínima. Os ✅/❌ comunicam o resto.
    min_sample_liga = 3
    min_sample_h2h = 0

    df_sc_display = build_statscamps_scope_frame(
        enabled=True,
        session_state=session_state,
        team_games=team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters,
        ml_info=ml_info,
        map_mode_ui=map_mode_ui,
        team_a=team_a,
        team_b=team_b,
    )
    df_sc_ab = build_statscamps_scope_frame(
        enabled=bool(sc_use),
        session_state=session_state,
        team_games=team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters,
        ml_info=ml_info,
        map_mode_ui=map_mode_ui,
        team_a=team_a,
        team_b=team_b,
    )

    def sc_pct(metric_key: str, side_key: str, line_val: float) -> Any:
        return compute_statscamps_pct(
            enabled=bool(sc_use),
            df_sc_ab=df_sc_ab,
            metric_key=metric_key,
            side_key=side_key,
            line_val=line_val,
            metric_col_map=SC_METRIC_COL,
        )

    def sc_badge(metric_key: str, side_key: str, line_val: float) -> str:
        return format_statscamps_badge(sc_pct(metric_key, side_key, line_val), threshold=float(sc_thr))

    df_a_all = df_a_used.copy().sort_values("date")
    df_b_all = df_b_used.copy().sort_values("date")
    # H2H = TODOS os confrontos da temporada (qualquer split/fase), não só o
    # recorte estreito. Antes o H2H saía de dfA_used (filtrado por split/playoffs),
    # então um recorte de Playoffs escondia confrontos da fase regular — ex.:
    # LGD x TT tem 3 jogos no Split 2 regular que sumiam no recorte de playoffs.
    # Fonte = team_games filtrado só por ano, com match de nome normalizado.
    _yr = getattr(active_filters, "year", None)
    df_h2h_a, df_h2h_b = select_h2h_games(
        team_games if isinstance(team_games, pd.DataFrame) else df_a_all,
        team_a=team_a, team_b=team_b, year=_yr, norm_team_fn=_norm_team_key,
    )
    if df_h2h_a.empty and df_h2h_b.empty:
        # Fallback to the (split-scoped) per-team frames only if the full-season
        # lookup found nothing — keeps behavior safe for odd column layouts.
        df_h2h_a = df_a_all[df_a_all["opponent"] == team_b].copy()
        df_h2h_b = df_b_all[df_b_all["opponent"] == team_a].copy()

    # Escopo pela REGRA DO USUÁRIO (feedback_cross_league_e_ewc_janelas):
    # "Ano todo" = ano inteiro SEM filtro de liga; em CROSS-LEAGUE (liga
    # dominante de A ≠ de B) as janelas 15/10/5 usam o campeonato MAIS RECENTE
    # de cada time (ex.: MSI), não a liga doméstica. Estas fatias alimentam a
    # Odd (Hist), os chips de recorte E o card temporal — bug flagrado
    # 2026-07-06 (TSW×LYON no MSI: hist/janelas vinham de LCP/LCS).
    from app_core.window_scope_core import resolve_window_scopes
    _scopes = resolve_window_scopes(
        team_games=team_games if isinstance(team_games, pd.DataFrame) else df_a_all,
        df_a_used=df_a_all, df_b_used=df_b_all,
        team_a=team_a, team_b=team_b,
        year=_yr, map_mode=str(map_mode_ui or "avg"),
        filters_cls=_ctx(ctx, "Filters"),
    )
    df_a_win, df_b_win = _scopes["df_a_win"], _scopes["df_b_win"]
    df_a_full, df_b_full = _scopes["df_a_full"], _scopes["df_b_full"]

    slices = recency_slices(df_a=df_a_win, df_b=df_b_win, df_h2h=df_h2h_a,
                            id_col="gameid", date_col="date",
                            df_a_year=df_a_full, df_b_year=df_b_full)
    df_league_year = slices["league_year"]
    df_h2h_year = slices["h2h_year"]
    df_league_5 = slices["league_5"]
    df_league_10 = slices["league_10"]
    df_league_15 = slices["league_15"]
    df_h2h_5 = slices["h2h_5"]
    df_h2h_10 = slices["h2h_10"]
    df_h2h_15 = slices["h2h_15"]

    n_league = int(getattr(df_league_year, "shape", [0])[0] or 0)
    n_h2h = int(getattr(df_h2h_year, "shape", [0])[0] or 0)
    if req_h2h and n_h2h <= 0:
        st.warning('H2H: não ha jogos no histórico para esse confronto. Com "Exigir H2H" marcado, nenhuma odd vai passar.')
    if req_liga and n_league <= 0:
        st.warning("Liga: não ha jogos no histórico para a liga/recorte atual. Ajuste os filtros ou escolha outro confronto.")

    pricing = make_pricing_adapters(
        laplace_pick=str(laplace_pick or "Liga 15"),
        w_model=float(w_mix_model),
        w_hist=float(w_mix_lap),
        w_stats=float(w_mix_sc),
        floor_odd=1.10,
    )

    def ensure_lines(metric_key: str, values: Any) -> list[float]:
        parsed = ensure_metric_lines(
            metric_key,
            values,
            parse_lines_fn=parse_lines,
            parse_time_lines_fn=parse_time_lines,
        )
        # Padrão da métrica (27.5 kills, 11.5 torres, ...) + linhas digitadas
        # para o confronto, SEMPRE juntas (digitou 28.5 → mostra 27.5 e 28.5).
        return merge_with_default_lines(metric_key, parsed)

    p_ml_row = float(view_state["p_ml_scope"])
    league_from_scope = infer_matchup_league(df_league_year=df_league_year, ml_info=ml_info)
    resumo_guardrail_cfg = build_resumo_guardrail_cfg(session_state, ml_trace_last=session_state.get("_ml_trace_last") or {})
    rows: list[dict[str, Any]] = []
    rows.extend(
        build_ml_rows(
            p_ml_row=float(p_ml_row),
            team_a=team_a,
            team_b=team_b,
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            df_league_year=df_league_year,
            df_league_15=df_league_15,
            df_league_10=df_league_10,
            df_league_5=df_league_5,
            df_h2h_year=df_h2h_year,
            df_h2h_15=df_h2h_15,
            df_h2h_10=df_h2h_10,
            df_h2h_5=df_h2h_5,
            odd_from_prob_fn=odd_from_prob,
            laplace_odds_fn=pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
            odd_mix_fn=pricing["odd_mix_fn"],
            fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=fmt_recortes_disp,
            guardrail_cfg=resumo_guardrail_cfg,
            matchup_league=league_from_scope,
        )
    )
    rows.extend(
        build_metric_rows(
            lines_by_metric=lines_by_metric,
            ensure_lines_fn=ensure_lines,
            metric_col_map=dict(SC_METRIC_COL),
            totals_for_lines=totals_for_lines,
            total_over_prob_fn=total_over_prob,
            odd_from_prob_fn=odd_from_prob,
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            df_league_year=df_league_year,
            df_league_15=df_league_15,
            df_league_10=df_league_10,
            df_league_5=df_league_5,
            df_h2h_year=df_h2h_year,
            df_h2h_15=df_h2h_15,
            df_h2h_10=df_h2h_10,
            df_h2h_5=df_h2h_5,
            laplace_odds_fn=pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
            odd_mix_fn=pricing["odd_mix_fn"],
            fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=fmt_recortes_disp,
            fmt_line_disp_fn=fmt_line_disp,
            metric_label_fn=metric_label,
            side_label_fn=side_label,
            sc_badge_fn=sc_badge,
            sc_pct_fn=sc_pct,
            guardrail_cfg=resumo_guardrail_cfg,
            matchup_league=league_from_scope,
        )
    )

    combo_lines_by_metric = {
        "kills": parse_lines(str(view_state["combo_kills_text"])),
        "time": parse_time_lines(str(view_state["combo_time_text"])),
    }
    p_ml_combo = float(view_state["p_ml_scope"])

    def combo_bases(side_key: str) -> dict[str, Any]:
        if side_key == "A":
            base_l_year = df_a_all.sort_values("date")
            base_h_year = df_h2h_a.sort_values("date")
        else:
            base_l_year = df_b_all.sort_values("date")
            base_h_year = df_h2h_b.sort_values("date")
        return {
            "l_year": base_l_year,
            "l_5": tail_df(base_l_year, 5).sort_values("date"),
            "l_10": tail_df(base_l_year, 10).sort_values("date"),
            "l_15": tail_df(base_l_year, 15).sort_values("date"),
            "h_year": base_h_year,
            "h_5": tail_df(base_h_year, 5).sort_values("date"),
            "h_10": tail_df(base_h_year, 10).sort_values("date"),
            "h_15": tail_df(base_h_year, 15).sort_values("date"),
        }

    rows.extend(
        build_combo_rows(
            combo_lines_by_metric=combo_lines_by_metric,
            metric_col_map=dict(SC_METRIC_COL),
            p_ml_combo=float(p_ml_combo),
            team_a=team_a,
            team_b=team_b,
            combo_hist_w=float(combo_hist_w),
            combo_style_w=float(combo_style_w),
            combo_form_w=float(combo_form_w),
            combo_use_mismatch=bool(combo_use_mismatch),
            req_liga=bool(req_liga),
            req_h2h=bool(req_h2h),
            req_model=bool(req_model),
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            max_odd=float(max_odd),
            get_combo_bases_fn=combo_bases,
            wr_laplace_fn=wr_laplace,
            counts_ml_totals_fn=counts_ml_totals,
            totals_for_lines=totals_for_lines,
            total_over_prob_fn=total_over_prob,
            odd_from_prob_fn=odd_from_prob,
            laplace_odds_fn=pricing["laplace_odds_fn"],
            laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
            odd_mix_fn=lambda om, oh, sc, mb: pricing["odd_mix_fn"](om, oh, sc, model_boost=mb),
            fmt_line_disp_fn=fmt_line_disp,
            metric_label_fn=metric_label,
            fmt_recortes_disp_fn=fmt_recortes_disp,
            fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
            sc_badge_fn=sc_badge,
            sc_pct_fn=sc_pct,
            guardrail_cfg=resumo_guardrail_cfg,
            matchup_league=league_from_scope,
        )
    )

    rows = enrich_resumo_rows_with_stats_detail(
        rows,
        df_scope=df_sc_display,
        team_a=team_a,
        team_b=team_b,
    )
    # Inject league so the tracker can capture it when user marks a pick.
    for r in rows:
        if isinstance(r, dict):
            r["_league"] = league_from_scope
    set_resumo_market_trace(session_state, build_market_trace_rows(rows))
    persist_combined_trace_file()

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        _render_diagnostic(
            st=st,
            n_liga=n_league, n_h2h=n_h2h,
            min_sample_liga=int(min_sample_liga),
            min_sample_h2h=int(min_sample_h2h),
            req_liga=bool(req_liga), req_h2h=bool(req_h2h), req_model=bool(req_model),
            max_odd=float(max_odd),
        )
        return

    # === Novo Resumo: candidatos rankeados + veredito ===
    matchup_key_oc = f"{team_a}|{team_b}"
    oc_store_key = "resumo_odd_casa_map"
    oc_map_all = dict(session_state.get(oc_store_key) or {})
    oc_map_cur = dict(oc_map_all.get(matchup_key_oc) or {})

    # Mostra TODAS as linhas que o usuário pediu, ordenadas por score (melhores
    # no topo) — sem cortar em 5. Cortar escondia linhas digitadas de propósito
    # (ex.: você quer ver o 26.5 mesmo que o 27.5 pontue mais). Teto alto só
    # como salvaguarda contra colagens gigantes.
    _sort_opts = {
        "Score (maior→menor)": "score",
        "Odd (menor→maior)": "odd",
        "Gestão (maior→menor)": "gestao",
    }
    _sort_keys = list(_sort_opts.keys())
    if str(session_state.get("resumo_picks_sort", "") or "") not in _sort_keys:
        session_state["resumo_picks_sort"] = _sort_keys[0]
    _sort_label = st.selectbox(
        "Ordenar picks por", _sort_keys, key="resumo_picks_sort",
        help="Score = melhores notas no topo. Odd = menores odds (mais prováveis) primeiro. Gestão = maiores stakes primeiro.",
    )
    _sort_by = _sort_opts.get(_sort_label, "score")
    # Sincroniza as "Sua odd" digitadas (widgets oc_*) ANTES de computar as picks.
    # Os widgets escrevem em session_state[oc_key]; ler isso aqui (no topo do
    # rerun, já com o valor novo) garante que a EDGE use a odd atual. Antes este
    # bloco ficava DEPOIS do rank_candidates -> a edge saía com a odd anterior
    # (lag de 1 rerun: o campo mostrava 1.68 mas a edge era a do 2.11).
    oc_key_prefix = f"oc_{matchup_key_oc}|"
    for ss_key in list(session_state.keys()):
        if isinstance(ss_key, str) and ss_key.startswith(oc_key_prefix):
            row_key = ss_key[len(oc_key_prefix):]
            fv = _parse_odd_input(session_state.get(ss_key))
            if fv is not None and fv > 1.0:
                oc_map_cur[row_key] = float(fv)
            elif row_key in oc_map_cur:
                oc_map_cur.pop(row_key, None)
    oc_map_all[matchup_key_oc] = oc_map_cur
    session_state[oc_store_key] = oc_map_all

    # Mercados binários: a pick fala a língua do mercado ("Sim/Não"), não
    # "Acima/Abaixo 0.5" (pedido 2026-07-14, junto com o ✓/✗ do V3).
    try:
        from app_core.stats_config import STATS_MARKET_CONFIG as _smc_lbl
        _bin_keys = {str(c.get("metric_key")) for c in _smc_lbl.values() if c.get("binary_market")}
        if _bin_keys:
            for _r in rows:
                if _metric_leaf(str(_r.get("_metric", ""))) in _bin_keys:
                    _lado = str(_r.get("Lado", ""))
                    _r["Lado"] = "Sim" if _lado in ("Acima", "Over") else ("Não" if _lado in ("Abaixo", "Under") else _lado)
                    _r["Mercado"] = str(_r.get("Mercado", "")).replace("(0.5)", "(sim/não)")
                    _r["Linha"] = "-"
    except Exception:
        pass

    # Teto de odd em TODOS os mercados — regra ATUALIZADA pelo usuário em
    # 2026-07-10: "TODAS as apostas abaixo do teto devem aparecer (ML, combos,
    # totais)". A pick entra se a odd do MODELO **ou** a odd HIST couber no
    # teto (antes, regra 06-11, era só HIST — cortava ML com modelo 1.60 e
    # hist 1.85, exatamente a ausência que ele notou). Sem NENHUMA odd
    # verificável → fora, como antes.
    try:
        _mx_teto = float(max_odd)
    except (TypeError, ValueError):
        _mx_teto = None
    if _mx_teto is not None and _mx_teto > 0:
        def _odd_dentro_do_teto(r: dict) -> bool:
            for campo in ("Odd (Hist)", "Odd (Modelo)"):
                try:
                    if float(r.get(campo)) <= float(_mx_teto):
                        return True
                except (TypeError, ValueError):
                    continue
            return False
        rows = [r for r in rows if _odd_dentro_do_teto(r)]

    # df_a_win/df_b_win/df_a_full/df_b_full vêm do bloco de escopo lá de cima
    # (regra cross-league) — mesmos frames que alimentam as fatias da Odd Hist,
    # então card temporal e odds ficam SEMPRE coerentes entre si.
    ranked = rank_candidates(
        rows,
        oc_map=oc_map_cur,
        df_a=df_a_win,
        df_b=df_b_win,
        df_a_full=df_a_full,
        df_b_full=df_b_full,
        team_a=team_a,
        team_b=team_b,
        top_n=60,
        sort_by=_sort_by,
    )

    _render_candidate_cards(
        st=st, ranked=ranked, oc_map_cur=oc_map_cur,
        matchup_key_oc=matchup_key_oc,
        team_a=team_a, team_b=team_b, league=str(league_from_scope or ""),
        sort_label=str(_sort_label),
    )

    _render_veredito_block(st=st, ranked=ranked, team_a=team_a, team_b=team_b)

    # Compare-mode: quando há 2+ linhas da mesma métrica, mostra o sweet spot.
    _render_line_comparisons(st=st, rows=rows)
