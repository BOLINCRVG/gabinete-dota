from __future__ import annotations

from typing import Any, Callable, Sequence

import pandas as pd

from app_core.consistency_core import (
    build_overall_tables,
    fmt_metric_value,
    h2h_numeric_series,
    h2h_rows_verbose,
    last_rows_metric,
    panel_title_with_suffix,
    recent_rows_with_hc_markers,
    team_metric_stats,
)

from app_core.tab_ui_core import render_tab_hero, render_tab_section


def _playoffs_label(value: Any) -> str:
    if value is None or value == "":
        return "All"
    return "Playoffs" if bool(value) else "Regular"


def render_market_context_hero(
    st: Any,
    *,
    team_a: str,
    team_b: str,
    filters: Any,
    league_mode: str,
    map_mode: str,
    window_short: str,
    max_games: Any,
) -> None:
    league_label = "-"
    if str(league_mode) == "fixed":
        league_label = str(getattr(filters, "league", "") or "-")
    elif str(league_mode) == "auto":
        league_label = "Auto"
    else:
        league_label = "Todos confrontos"
    chips = [
        f"Confronto: {team_a} x {team_b}",
        f"Ano: {getattr(filters, 'year', '-') or '-'}",
        f"Split: {getattr(filters, 'split', '-') or '-'}",
        f"Playoffs: {_playoffs_label(getattr(filters, 'playoffs', None))}",
        f"League mode: {league_mode} ({league_label})",
        f"Map mode: {map_mode}",
        f"Janela: {window_short}",
        f"Max jogos: {max_games if max_games is not None else 'All'}",
        "Contexto histórico: todos confrontos do recorte",
    ]
    render_tab_hero(
        st,
        title="Mercados",
        subtitle="Modelo, histórico e leitura operacional no mesmo recorte, com contexto fixo no topo.",
        eyebrow="Contexto Ativo",
        chips=chips,
    )


def render_market_mode_section(st: Any, *, title: str, subtitle: str = "") -> None:
    render_tab_section(st, title=title, subtitle=subtitle, eyebrow="Leitura")


def render_summary_block(st: Any, title: str) -> None:
    st.markdown(
        "<div style='background:#2daac4;padding:8px 10px;border-radius:6px 6px 0 0;"
        "text-align:center;font-weight:700;letter-spacing:.03em;'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )


def render_market_context_panel(
    st: Any,
    *,
    panel_title: str,
    col_name: str,
    team_a: str,
    team_b: str,
    df_a_context_all: pd.DataFrame,
    df_b_context_all: pd.DataFrame,
    df_a_context_season: pd.DataFrame,
    df_b_context_season: pd.DataFrame,
    df_h2h_year: pd.DataFrame,
    df_h2h_context_year_full_a: pd.DataFrame | None,
    hc_kills_lines: Sequence[float] | None,
    hc_team_selected: str,
    min_to_mmss_fn: Callable[[float], str],
    num_formatter: Callable[[Any, int], str],
) -> None:
    panel_title_fmt = panel_title_with_suffix(str(panel_title or "").strip() or "Mercado", "(combined)")
    st.markdown(f"### {panel_title_fmt}")

    def metric(col: str, value: Any, decimals: int = 1) -> str:
        return fmt_metric_value(col, value, decimals, time_formatter=min_to_mmss_fn)

    s_a_cur = team_metric_stats(df_a_context_all, col_name)
    s_b_cur = team_metric_stats(df_b_context_all, col_name)
    s_a_all = team_metric_stats(df_a_context_season, col_name)
    s_b_all = team_metric_stats(df_b_context_season, col_name)
    df_cur, df_all = build_overall_tables(
        team_a,
        team_b,
        col_name,
        s_a_cur,
        s_b_cur,
        s_a_all,
        s_b_all,
        num_formatter=num_formatter,
        metric_formatter=metric,
    )

    c_over, c_recent, c_h2h = st.columns(3)
    with c_over:
        render_summary_block(st, "OVERALL")
        with st.container(border=True):
            st.markdown(f"**{team_a}**  |  **{team_b}**")
            st.caption("Current Split")
            st.dataframe(df_cur, width="stretch", hide_index=True)
            st.caption("All Season")
            st.dataframe(df_all, width="stretch", hide_index=True)

    with c_recent:
        render_summary_block(st, "RECENT FORM")
        a5 = team_metric_stats(df_a_context_all.tail(5), col_name)
        b5 = team_metric_stats(df_b_context_all.tail(5), col_name)
        a10 = team_metric_stats(df_a_context_all.tail(10), col_name)
        b10 = team_metric_stats(df_b_context_all.tail(10), col_name)
        with st.container(border=True):
            st.markdown(f"**{team_a}**")
            st.caption(
                f"Avg (Last 5): {metric(col_name, a5['avg'])} | "
                f"Avg (Last 10): {metric(col_name, a10['avg'])}"
            )
            a_rows: list[str]
            a_hc_lines_applied: list[float] = []
            if str(col_name) == "kills_for" and "handicap kills" in panel_title_fmt.lower():
                a_rows, a_hc_lines_applied = recent_rows_with_hc_markers(
                    df_a_context_all,
                    shown_team=team_a,
                    hc_team_selected=str(hc_team_selected or team_a),
                    hc_lines=[float(x) for x in list(hc_kills_lines or [])],
                    fallback_col_name=col_name,
                    metric_formatter=metric,
                    n=10,
                )
            else:
                a_rows = last_rows_metric(df_a_context_all, col_name, 10, metric_formatter=metric)
            if a_hc_lines_applied:
                st.caption(
                    "HC ordem: "
                    + ", ".join([f"{float(line):+g}" for line in a_hc_lines_applied])
                    + " | V=bateu X=nao bateu"
                )
            st.markdown("\n".join([f"- {x}" for x in a_rows]) if a_rows else "- sem jogos")
            st.markdown("---")
            st.markdown(f"**{team_b}**")
            st.caption(
                f"Avg (Last 5): {metric(col_name, b5['avg'])} | "
                f"Avg (Last 10): {metric(col_name, b10['avg'])}"
            )
            b_rows: list[str]
            b_hc_lines_applied: list[float] = []
            if str(col_name) == "kills_for" and "handicap kills" in panel_title_fmt.lower():
                b_rows, b_hc_lines_applied = recent_rows_with_hc_markers(
                    df_b_context_all,
                    shown_team=team_b,
                    hc_team_selected=str(hc_team_selected or team_a),
                    hc_lines=[float(x) for x in list(hc_kills_lines or [])],
                    fallback_col_name=col_name,
                    metric_formatter=metric,
                    n=10,
                )
            else:
                b_rows = last_rows_metric(df_b_context_all, col_name, 10, metric_formatter=metric)
            if b_hc_lines_applied:
                st.caption(
                    "HC ordem: "
                    + ", ".join([f"{float(line):+g}" for line in b_hc_lines_applied])
                    + " | V=bateu X=nao bateu"
                )
            st.markdown("\n".join([f"- {x}" for x in b_rows]) if b_rows else "- sem jogos")

    with c_h2h:
        render_summary_block(st, "PAST FACEOFFS")
        h2h_series = h2h_numeric_series(df_h2h_year, col_name)
        with st.container(border=True):
            st.markdown(f"**{team_a} vs {team_b}**")
            st.metric("H2H Games", int(h2h_series.shape[0]))
            avg_label = "Avg (combined)" if str(col_name).startswith("total_") else f"Avg ({team_a})"
            st.metric(avg_label, metric(col_name, (h2h_series.mean() if h2h_series.shape[0] else float("nan"))))
            if (
                df_h2h_context_year_full_a is not None
                and not df_h2h_context_year_full_a.empty
                and col_name in df_h2h_context_year_full_a.columns
            ):
                h2h_rows = h2h_rows_verbose(
                    df_h2h_context_year_full_a,
                    metric_col=col_name,
                    team_a=team_a,
                    metric_formatter=metric,
                    n=10,
                )
                st.markdown("\n".join([f"- {x}" for x in h2h_rows]) if h2h_rows else "- sem H2H")
            else:
                st.markdown("- sem H2H")


def render_stateful_tab_selector(
    st: Any,
    *,
    key: str,
    options: Sequence[str],
    default: str | None = None,
    label: str = "Abas",
) -> str:
    opts = [str(x) for x in list(options or []) if str(x or "").strip()]
    if not opts:
        return ""
    fallback = str(default or opts[0] or "")
    current = str(st.session_state.get(key, fallback) or fallback)
    if current not in opts:
        current = fallback if fallback in opts else opts[0]
        st.session_state[key] = current
    selected = st.radio(
            label,
            options=opts,
            key=key,
            horizontal=True,
            label_visibility="collapsed",
        )
    return str(selected or current)


__all__ = [
    "render_market_context_hero",
    "render_market_context_panel",
    "render_market_mode_section",
    "render_stateful_tab_selector",
    "render_summary_block",
]
