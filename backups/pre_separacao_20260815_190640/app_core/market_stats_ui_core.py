from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import pandas as pd

from app_core.championship_stats_core import (
    STATS_MARKET_CONFIG,
    apply_scope_filters,
    build_relevant_championship_options,
    build_championship_scope_frame,
    build_championship_stats_table,
    build_stats_table_view,
    build_threshold_lines_by_metric,
    compute_team_metric_pct,
    scope_label_from_meta,
)
from app_core.consistency_ui_core import render_market_mode_section
from app_core.tab_state_core import normalize_select_key
from app_core.tab_ui_core import render_tab_section


def _line_options(values: list[float]) -> list[str]:
    out = [""]
    for value in values or []:
        fv = float(value)
        out.append(f"{fv:g}")
    return out


def render_market_stats_tab(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_games: pd.DataFrame | None,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    ml_info: Mapping[str, Any] | None,
    map_mode_ui: Any,
    team_a: str,
    team_b: str,
    parse_lines_fn: Callable[[str], list[float]],
    parse_time_lines_fn: Callable[[str], list[float]],
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
    current_lines_by_metric: Mapping[str, list[float]] | None = None,
) -> None:
    render_tab_section(
        st,
        title="Stats por campeonato",
        subtitle="Tabela agregada por time no recorte atual, com linha custom e leitura reaproveitada no Resumo.",
    )

    df_scope, scope_meta = build_championship_scope_frame(
        session_state=session_state,
        team_games=team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters_fn,
        ml_info=ml_info,
        map_mode_ui=map_mode_ui,
    )
    if df_scope is None or df_scope.empty:
        st.info("Sem jogos no recorte atual para montar Stats.")
        return

    left_mode, right_mode = st.columns(2)
    with left_mode:
        render_market_mode_section(
            st,
            title="Modelo",
            subtitle="Aba de apoio operacional: use a linha ativa do confronto para comparar a leitura do mercado com o histórico agregado do campeonato.",
        )
    with right_mode:
        render_market_mode_section(
            st,
            title="Histórico",
            subtitle="Selecione o campeonato para ver a tabela agregada por time, com linha padrão ou custom reutilizada no Resumo.",
        )

    champ_options, champ_selected = build_relevant_championship_options(
        team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters_fn,
        team_a=str(team_a),
        team_b=str(team_b),
        selected_league=str(session_state.get("vg_stats_championship", "") or ""),
    )
    if str(session_state.get("vg_stats_championship", "") or "") not in champ_options:
        session_state["vg_stats_championship"] = champ_selected if champ_selected in champ_options else champ_options[0]

    def _on_championship_change() -> None:
        session_state["vg_stats_split"] = ""

    with st.container(border=True):
        st.markdown("##### Campeonato em foco")
        st.caption("Por padrão, usa o campeonato mais recente dos times no recorte atual. Você pode trocar manualmente para qualquer outro campeonato disponível.")
        c_champ, c_context = st.columns([1.5, 1.0])
        with c_champ:
            selected_championship = st.selectbox(
                "Campeonato",
                champ_options,
                key="vg_stats_championship",
                on_change=_on_championship_change,
            )
        with c_context:
            st.markdown(f"**Confronto:** {team_a} x {team_b}")
            auto_label = str(champ_options[1] if len(champ_options) > 1 else "Auto")
            current_label = str(selected_championship or "Auto")
            st.caption(f"Padrão recente: {auto_label}")
            st.caption(f"Seleção atual: {current_label}")

        if str(selected_championship) != "Auto":
            _avail_splits: list[str] = []
            if team_games is not None and not team_games.empty and "split" in team_games.columns and "league" in team_games.columns:
                try:
                    _df_sp = team_games[team_games["league"].astype(str) == str(selected_championship)]
                    _active_year = getattr(active_filters, "year", None)
                    if _active_year is not None and "year" in _df_sp.columns:
                        _df_sp = _df_sp[pd.to_numeric(_df_sp["year"], errors="coerce") == int(_active_year)]
                    _avail_splits = sorted([str(x) for x in _df_sp["split"].fillna("").astype(str).unique().tolist() if str(x).strip()])
                except Exception:
                    _avail_splits = []
            _split_opts = ["", *_avail_splits]
            normalize_select_key(session_state, "vg_stats_split", _split_opts)
            c_split, _ = st.columns([1.5, 1.0])
            with c_split:
                selected_split = st.selectbox(
                    "Split (opcional)",
                    _split_opts,
                    key="vg_stats_split",
                    format_func=lambda x: "Todos os splits (ano todo)" if x == "" else x,
                )
        else:
            selected_split = ""
            session_state["vg_stats_split"] = ""

    if str(selected_championship) != "Auto":
        try:
            split_manual = str(selected_split).strip() if selected_split else None
            scoped_filters = filters_cls(
                year=getattr(active_filters, "year", None),
                league=str(selected_championship),
                split=split_manual,
                playoffs=getattr(active_filters, "playoffs", None),
            )
            df_scope = apply_filters_fn(team_games, scoped_filters).copy()
            scope_meta = {
                "league": str(selected_championship),
                "year": str(getattr(active_filters, "year", "") or ""),
                "split": str(split_manual or ""),
                "playoffs": (
                    ""
                    if getattr(active_filters, "playoffs", None) is None
                    else ("Playoffs" if bool(active_filters.playoffs) else "Regular")
                ),
                "map_selector": str(map_mode_ui or "avg"),
                "manual": True,
            }
        except Exception:
            pass
    if df_scope is None or df_scope.empty:
        st.info("Sem jogos para o campeonato selecionado.")
        return

    st.caption(f"Escopo: {scope_label_from_meta(scope_meta)}")

    market_labels = list(STATS_MARKET_CONFIG.keys())
    normalize_select_key(session_state, "vg_stats_market", market_labels, fallback="Kills")
    market_label = st.selectbox("Mercado para linha dinâmica", market_labels, key="vg_stats_market")
    market_cfg = dict(STATS_MARKET_CONFIG.get(str(market_label), {}))
    metric_key = str(market_cfg.get("metric_key", "kills"))

    current_lines = list((current_lines_by_metric or {}).get(metric_key, []) or [])
    line_opts = _line_options(current_lines)
    normalize_select_key(session_state, "vg_stats_current_line", line_opts, fallback=line_opts[0])

    with st.container(border=True):
        st.markdown("##### Linha dinâmica e filtros")
        st.caption("A linha ativa controla a leitura reaproveitada no Resumo e a coluna dinâmica do mercado selecionado.")
        c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.0])
        with c1:
            selected_current = st.selectbox("Linha do confronto", line_opts, key="vg_stats_current_line")
        with c2:
            custom_line_text = st.text_input("Linha custom", key="vg_stats_custom_line", placeholder="ex: 35.5")
        with c3:
            side_options = ["All", "Blue", "Red"]
            normalize_select_key(session_state, "vg_stats_side_filter", side_options, fallback="All")
            side_filter = st.selectbox("Side", side_options, key="vg_stats_side_filter")
        with c4:
            wl_options = ["All", "Wins", "Losses"]
            normalize_select_key(session_state, "vg_stats_wl_filter", wl_options, fallback="All")
            wl_filter = st.selectbox("Win/loss", wl_options, key="vg_stats_wl_filter")

    parser = parse_time_lines_fn if metric_key == "time" else parse_lines_fn
    custom_values = list(parser(str(custom_line_text or "")))
    active_line = float(custom_values[0]) if custom_values else None
    active_line_source = "custom" if custom_values else "confronto"
    if active_line is None and str(selected_current).strip():
        try:
            active_line = float(selected_current)
        except Exception:
            active_line = None
    if active_line is None:
        active_line_source = "-"

    df_filtered = apply_scope_filters(df_scope, side_filter=str(side_filter), win_loss_filter=str(wl_filter))
    if df_filtered.empty:
        st.info("Sem jogos nesse filtro de Side/Win-loss.")
        return

    threshold_map = build_threshold_lines_by_metric(
        session_state=session_state,
        parse_lines_fn=parse_lines_fn,
        parse_time_lines_fn=parse_time_lines_fn,
        custom_metric_label=str(market_label),
        extra_lines=([float(active_line)] if active_line is not None else current_lines),
    )
    df_stats_raw = build_championship_stats_table(df_filtered, threshold_lines_by_metric=threshold_map)
    if df_stats_raw.empty:
        st.info("Sem stats agregadas nesse recorte.")
        return

    if active_line is not None:
        source_label = "Linha custom" if active_line_source == "custom" else "Linha do confronto"
        over_a = compute_team_metric_pct(df_filtered, team=str(team_a), metric_key=metric_key, side_key="over", line_val=float(active_line))
        over_b = compute_team_metric_pct(df_filtered, team=str(team_b), metric_key=metric_key, side_key="over", line_val=float(active_line))
        under_a = compute_team_metric_pct(df_filtered, team=str(team_a), metric_key=metric_key, side_key="under", line_val=float(active_line))
        under_b = compute_team_metric_pct(df_filtered, team=str(team_b), metric_key=metric_key, side_key="under", line_val=float(active_line))
        st.info(
            f"{source_label} ativa para {market_label}: {float(active_line):g}. "
            "Essa mesma lógica é usada para preencher a coluna Stats no Resumo."
        )
        df_matchup = pd.DataFrame(
            [
                {
                    "Team": team_a,
                    "% Over": (f"{format_num_fn(float(over_a), 1)}%" if over_a is not None else "-"),
                    "% Under": (f"{format_num_fn(float(under_a), 1)}%" if under_a is not None else "-"),
                },
                {
                    "Team": team_b,
                    "% Over": (f"{format_num_fn(float(over_b), 1)}%" if over_b is not None else "-"),
                    "% Under": (f"{format_num_fn(float(under_b), 1)}%" if under_b is not None else "-"),
                },
            ]
        )
        st.dataframe(df_matchup, width="stretch", hide_index=True)

    df_stats_view = build_stats_table_view(
        df_stats_raw,
        min_to_mmss_fn=min_to_mmss_fn,
        format_num_fn=format_num_fn,
        priority_teams=[str(team_a), str(team_b)],
    )
    st.dataframe(df_stats_view, width="stretch", height=620, hide_index=True)

