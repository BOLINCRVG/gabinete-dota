# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from app_core.render_ctx_core import validate_render_ctx


def _validate_consistencia_ctx(ctx: Mapping[str, Any]) -> None:
    """Loud-fail if a required core_* import is missing (ctx=globals contract)."""
    try:
        src = inspect.getsource(render_consistencia_tab)
    except (OSError, TypeError):
        return
    validate_render_ctx(ctx, renderer_source=src, renderer_name="render_consistencia_tab")


def render_consistencia_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    _validate_consistencia_ctx(ctx)
    Filters = ctx.get("Filters")
    MatchupTotals = ctx.get("MatchupTotals")
    _MARKET_TAB_NAMES = ctx.get("_MARKET_TAB_NAMES")
    _combine_mean = ctx.get("_combine_mean")
    _combine_sd = ctx.get("_combine_sd")
    _consistency_analysis_signature = ctx.get("_consistency_analysis_signature")
    _df_contract_view = ctx.get("_df_contract_view")
    _display_ml_totals_table = ctx.get("_display_ml_totals_table")
    _display_ou_table = ctx.get("_display_ou_table")
    _fmt_line_disp = ctx.get("_fmt_line_disp")
    _fmt_recortes_disp = ctx.get("_fmt_recortes_disp")
    _fnum = ctx.get("_fnum")
    _format_num = ctx.get("_format_num")
    _lines_table = ctx.get("_lines_table")
    _lines_table_with_p_ml = ctx.get("_lines_table_with_p_ml")
    _metric_label_pt = ctx.get("_metric_label_pt")
    _min_to_mmss = ctx.get("_min_to_mmss")
    _ml_info = ctx.get("_ml_info")
    _odd_from_p = ctx.get("_odd_from_p")
    _persist_combined_trace_file = ctx.get("_persist_combined_trace_file")
    _prev_history_slices = ctx.get("_prev_history_slices")
    _prev_reference_note = ctx.get("_prev_reference_note")
    _prev_season_label = ctx.get("_prev_season_label")
    _prev_totals_for_lines = ctx.get("_prev_totals_for_lines")
    _save_app_settings_to_disk = ctx.get("_save_app_settings_to_disk")
    _side_label_pt = ctx.get("_side_label_pt")
    apply_filters = ctx.get("apply_filters")
    barons = ctx.get("barons")
    combo_kills_text = ctx.get("combo_kills_text")
    combo_time_text = ctx.get("combo_time_text")
    core_apply_resumo_combo_preset = ctx.get("core_apply_resumo_combo_preset")
    core_apply_resumo_filter_values = ctx.get("core_apply_resumo_filter_values")
    core_build_audit_base_options = ctx.get("core_build_audit_base_options")
    core_build_audit_market_options = ctx.get("core_build_audit_market_options")
    core_build_audit_view = ctx.get("core_build_audit_view")
    core_build_combo_rows = ctx.get("core_build_combo_rows")
    core_build_market_trace_rows = ctx.get("core_build_market_trace_rows")
    core_build_metric_rows = ctx.get("core_build_metric_rows")
    core_build_ml_rows = ctx.get("core_build_ml_rows")
    core_build_ml_totals_laplace_payload = ctx.get("core_build_ml_totals_laplace_payload")
    core_build_ml_totals_model_payload = ctx.get("core_build_ml_totals_model_payload")
    core_build_quick_metric_options = ctx.get("core_build_quick_metric_options")
    core_build_quick_selector_rows = ctx.get("core_build_quick_selector_rows")
    core_build_resumo_guardrail_cfg = ctx.get("core_build_resumo_guardrail_cfg")
    core_build_simple_audit_view = ctx.get("core_build_simple_audit_view")
    core_build_statscamps_scope_frame = ctx.get("core_build_statscamps_scope_frame")
    core_compute_statscamps_pct = ctx.get("core_compute_statscamps_pct")
    core_counts_ml_totals = ctx.get("core_counts_ml_totals")
    core_delete_active_resumo_filter_profile = ctx.get("core_delete_active_resumo_filter_profile")
    core_ensure_metric_lines = ctx.get("core_ensure_metric_lines")
    core_ensure_resumo_combo_state = ctx.get("core_ensure_resumo_combo_state")
    core_event_block = ctx.get("core_event_block")
    core_finalize_resumo_dataframe = ctx.get("core_finalize_resumo_dataframe")
    core_format_statscamps_badge = ctx.get("core_format_statscamps_badge")
    core_get_ml_scope = ctx.get("core_get_ml_scope")
    core_has_metric_history = ctx.get("core_has_metric_history")
    core_infer_matchup_league = ctx.get("core_infer_matchup_league")
    core_laplace_odds = ctx.get("core_laplace_odds")
    core_list_resumo_filter_presets = ctx.get("core_list_resumo_filter_presets")
    core_load_resumo_filter_custom = ctx.get("core_load_resumo_filter_custom")
    core_make_pricing_adapters = ctx.get("core_make_pricing_adapters")
    core_ou_block = ctx.get("core_ou_block")
    core_quick_outcome_probs = ctx.get("core_quick_outcome_probs")
    core_quick_team_frame = ctx.get("core_quick_team_frame")
    core_recency_slices = ctx.get("core_recency_slices")
    core_render_market_context_hero = ctx.get("core_render_market_context_hero")
    core_render_market_context_panel = ctx.get("core_render_market_context_panel")
    core_render_market_mode_section = ctx.get("core_render_market_mode_section")
    core_render_market_stats_tab = ctx.get("core_render_market_stats_tab")
    core_render_opponent_profile_tab = ctx.get("core_render_opponent_profile_tab")
    core_render_quick_selector_controls = ctx.get("core_render_quick_selector_controls")
    core_render_resumo_tab = ctx.get("core_render_resumo_tab")
    core_render_stateful_tab_selector = ctx.get("core_render_stateful_tab_selector")
    core_render_summary_block = ctx.get("core_render_summary_block")
    core_save_resumo_filter_custom = ctx.get("core_save_resumo_filter_custom")
    core_save_resumo_filter_profile = ctx.get("core_save_resumo_filter_profile")
    core_set_consistency_analysis_signature = ctx.get("core_set_consistency_analysis_signature")
    core_tail_df = ctx.get("core_tail_df")
    core_wr_laplace = ctx.get("core_wr_laplace")
    dfA_used = ctx.get("dfA_used")
    dfB_used = ctx.get("dfB_used")
    dragons = ctx.get("dragons")
    filter_team_games = ctx.get("filter_team_games")
    filters = ctx.get("filters")
    hc_dragons_text = ctx.get("hc_dragons_text")
    hc_kills_text = ctx.get("hc_kills_text")
    hc_towers_text = ctx.get("hc_towers_text")
    inhib = ctx.get("inhib")
    kills = ctx.get("kills")
    league_mode = ctx.get("league_mode")
    map_mode = ctx.get("map_mode")
    max_games = ctx.get("max_games")
    pA = ctx.get("pA")
    pB = ctx.get("pB")
    parse_lines_text = ctx.get("parse_lines_text")
    parse_time_lines_text = ctx.get("parse_time_lines_text")
    state_get_bool = ctx.get("state_get_bool")
    state_get_float = ctx.get("state_get_float")
    state_get_int = ctx.get("state_get_int")
    state_get_str = ctx.get("state_get_str")
    teamA = ctx.get("teamA")
    teamA_kills_line_text = ctx.get("teamA_kills_line_text")
    teamB = ctx.get("teamB")
    teamB_kills_line_text = ctx.get("teamB_kills_line_text")
    team_games = ctx.get("team_games")
    time_m = ctx.get("time_m")
    total_over_prob = ctx.get("total_over_prob")
    totals_for_lines = ctx.get("totals_for_lines")
    towers = ctx.get("towers")
    window_short = ctx.get("window_short")
    # --- extracted block body ---
    st.markdown(
        """
<div class="gb-hero" style="margin-top:4px;">
  <div class="title" style="font-size:1.1rem;">Consist\u00eancia de Mercado</div>
  <div class="sub">Hist\u00f3rico ajustado (sanity-check) com compara\u00e7\u00e3o do Modelo no mesmo fluxo.</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    _cons_sig_now = _consistency_analysis_signature()
    core_set_consistency_analysis_signature(st.session_state, _cons_sig_now)
    st.caption("Consist\u00eancia em dia para os par\u00e2metros atuais.")

    st.markdown("### Resumo")

    # Bases já filtradas (mesmo filtros/league_mode/map_mode do motor)
    dfA_all = dfA_used.copy()
    dfB_all = dfB_used.copy()
    _context_filters = Filters(
        year=filters.year,
        split=filters.split,
        playoffs=filters.playoffs,
        league=None,
    )
    _context_filters_no_split = Filters(
        year=filters.year,
        split=None,
        playoffs=None,
        league=None,
    )
    dfA_context_all = filter_team_games(
        team_games,
        team=teamA,
        filters=_context_filters,
        league_mode="all",
        map_mode=map_mode,
        max_games=max_games,
    )
    dfB_context_all = filter_team_games(
        team_games,
        team=teamB,
        filters=_context_filters,
        league_mode="all",
        map_mode=map_mode,
        max_games=max_games,
    )
    dfA_context_season = filter_team_games(
        team_games,
        team=teamA,
        filters=_context_filters_no_split,
        league_mode="all",
        map_mode=map_mode,
        max_games=max_games,
    )
    dfB_context_season = filter_team_games(
        team_games,
        team=teamB,
        filters=_context_filters_no_split,
        league_mode="all",
        map_mode=map_mode,
        max_games=max_games,
    )
    dfA_context_year_full = filter_team_games(
        team_games,
        team=teamA,
        filters=_context_filters_no_split,
        league_mode="all",
        map_mode=map_mode,
        max_games=None,
    )
    dfB_context_year_full = filter_team_games(
        team_games,
        team=teamB,
        filters=_context_filters_no_split,
        league_mode="all",
        map_mode=map_mode,
        max_games=None,
    )
    if True:
    
        # H2H: perspectiva do Team A
        df_h2h_A = dfA_all[dfA_all["opponent"] == teamB].copy()
        df_h2h_B = dfB_all[dfB_all["opponent"] == teamA].copy()
        df_h2h_context_year_full_A = dfA_context_year_full[
            dfA_context_year_full["opponent"] == teamB
        ].copy()

        # Histórico/Laplace desta tela deve considerar todos os jogos do ano
        # no recorte amplo do time, incluindo campeonatos paralelos como EMEA Masters.
        _history_slices = core_recency_slices(
            df_a=dfA_context_year_full,
            df_b=dfB_context_year_full,
            df_h2h=df_h2h_context_year_full_A,
            id_col="gameid",
            date_col="date",
            dedupe_league=True,
            backfill_unique_targets=True,
        )
        df_league_year = _history_slices["league_year"]
        df_league_15 = _history_slices["league_15"]
        df_league_10 = _history_slices["league_10"]
        df_league_5 = _history_slices["league_5"]
        df_h2h_year = _history_slices["h2h_year"]
        df_h2h_15 = _history_slices["h2h_15"]
        df_h2h_10 = _history_slices["h2h_10"]
        df_h2h_5 = _history_slices["h2h_5"]
    
        # Painel extra (sem remover o que ja existe): resumo visual estilo print.
    
        def _num(v: Any, d: int = 1) -> str:
            try:
                x = float(v)
                if not math.isfinite(x):
                    return "-"
                return f"{x:.{int(d)}f}"
            except (TypeError, ValueError):
                return "-"
    
        # Diagnóstico rápido: evitar confusão quando H2H/Liga não tem amostra
        # (não depende de variáveis do Resumo; usa session_state se existir)
        _n_league = int(getattr(df_league_year, 'shape', [0])[0] or 0)
        _n_h2h = int(getattr(df_h2h_year, 'shape', [0])[0] or 0)
        req_h2h = state_get_bool(st.session_state, 'resumo_req_h2h', False)
        req_liga = state_get_bool(st.session_state, 'resumo_req_liga', False)
        min_sample_liga = state_get_int(st.session_state, 'resumo_min_sample_liga', 8)
        min_sample_h2h = state_get_int(st.session_state, 'resumo_min_sample_h2h', 3)
        if req_h2h and _n_h2h < int(min_sample_h2h):
            st.warning(f'H2H: amostra insuficiente ({_n_h2h}/{int(min_sample_h2h)}).')
        if req_liga and _n_league < int(min_sample_liga):
            st.warning(f'Liga: amostra insuficiente ({_n_league}/{int(min_sample_liga)}).')
    
        # Parse extra lines (multi-line)
        teamA_kills_lines = parse_lines_text(teamA_kills_line_text)
        teamB_kills_lines = parse_lines_text(teamB_kills_line_text)
        hc_kills_lines = parse_lines_text(hc_kills_text)
        hc_towers_lines = parse_lines_text(hc_towers_text)
        hc_dragons_lines = parse_lines_text(hc_dragons_text)
        combo_kills_lines = parse_lines_text(combo_kills_text)
        combo_time_lines = parse_time_lines_text(combo_time_text)
    
        core_render_market_context_hero(
            st,
            team_a=teamA,
            team_b=teamB,
            filters=filters,
            league_mode=str(league_mode),
            map_mode=str(map_mode),
            window_short=str(window_short),
            max_games=max_games,
        )

        def _render_full_resumo_odds_tab() -> None:
            core_render_resumo_tab(
                st=st,
                ctx={
                    "teamA": teamA,
                    "teamB": teamB,
                    "team_games": team_games,
                    "Filters": Filters,
                    "filters": filters,
                    "apply_filters_fn": apply_filters,
                    "ml_info": _ml_info,
                    "map_mode_ui": ctx.get("map_mode_ui"),
                    "kills": kills,
                    "towers": towers,
                    "dragons": dragons,
                    "barons": barons,
                    "inhibitors": inhib,
                    "time_m": time_m,
                    "totals_for_lines": totals_for_lines,
                    "dfA_used": dfA_used,
                    "dfB_used": dfB_used,
                    "state_get_bool_fn": state_get_bool,
                    "state_get_float_fn": state_get_float,
                    "state_get_int_fn": state_get_int,
                    "state_get_str_fn": state_get_str,
                    "save_app_settings_to_disk_fn": _save_app_settings_to_disk,
                    "parse_lines_fn": parse_lines_text,
                    "parse_time_lines_fn": parse_time_lines_text,
                    "odd_from_prob_fn": _odd_from_p,
                    "fmt_recortes_disp_fn": _fmt_recortes_disp,
                    "fmt_line_disp_fn": _fmt_line_disp,
                    "metric_label_fn": _metric_label_pt,
                    "side_label_fn": _side_label_pt,
                    "format_num_fn": _format_num,
                    "df_contract_view_fn": _df_contract_view,
                    "total_over_prob_fn": total_over_prob,
                    "persist_combined_trace_file_fn": _persist_combined_trace_file,
                    "list_resumo_filter_presets_fn": core_list_resumo_filter_presets,
                    "load_resumo_filter_custom_fn": core_load_resumo_filter_custom,
                    "apply_resumo_filter_values_fn": core_apply_resumo_filter_values,
                    "save_resumo_filter_custom_fn": core_save_resumo_filter_custom,
                    "save_resumo_filter_profile_fn": core_save_resumo_filter_profile,
                    "delete_active_resumo_filter_profile_fn": core_delete_active_resumo_filter_profile,
                    "ensure_resumo_combo_state_fn": core_ensure_resumo_combo_state,
                    "apply_resumo_combo_preset_fn": core_apply_resumo_combo_preset,
                    "build_statscamps_scope_frame_fn": core_build_statscamps_scope_frame,
                    "compute_statscamps_pct_fn": core_compute_statscamps_pct,
                    "format_statscamps_badge_fn": core_format_statscamps_badge,
                    "recency_slices_fn": core_recency_slices,
                    "make_pricing_adapters_fn": core_make_pricing_adapters,
                    "ensure_metric_lines_fn": core_ensure_metric_lines,
                    "infer_matchup_league_fn": core_infer_matchup_league,
                    "build_resumo_guardrail_cfg_fn": core_build_resumo_guardrail_cfg,
                    "build_ml_rows_fn": core_build_ml_rows,
                    "build_metric_rows_fn": core_build_metric_rows,
                    "build_combo_rows_fn": core_build_combo_rows,
                    "build_market_trace_rows_fn": core_build_market_trace_rows,
                    "finalize_resumo_dataframe_fn": core_finalize_resumo_dataframe,
                    "wr_laplace_fn": core_wr_laplace,
                    "counts_ml_totals_fn": core_counts_ml_totals,
                    "tail_df_fn": core_tail_df,
                },
            )
        # Tabs por mercado (igual seu app)
        _market_tab_names = list(_MARKET_TAB_NAMES)
        _consistency_market_tab = core_render_stateful_tab_selector(
            st,
            key="consistency_market_tab",
            options=_market_tab_names,
            default=("Win Rate" if "Win Rate" in _market_tab_names else _market_tab_names[0]),
            label="Mercados",
        )

        def _render_metric_tab(metric_key: str, title: str, value_col: str, lines_list: list[float]):
            core_render_market_context_panel(
                st,
                panel_title=title,
                col_name=value_col,
                team_a=teamA,
                team_b=teamB,
                df_a_context_all=dfA_context_all,
                df_b_context_all=dfB_context_all,
                df_a_context_season=dfA_context_season,
                df_b_context_season=dfB_context_season,
                df_h2h_year=df_h2h_year,
                df_h2h_context_year_full_a=df_h2h_context_year_full_A,
                hc_kills_lines=hc_kills_lines,
                hc_team_selected=str(st.session_state.get("hc_kills_team") or teamA),
                min_to_mmss_fn=_min_to_mmss,
                num_formatter=_num,
            )
            st.divider()
            _current_history_slices = {
                "league_year": df_league_year,
                "league_15": df_league_15,
                "league_10": df_league_10,
                "league_5": df_league_5,
                "h2h_year": df_h2h_year,
                "h2h_15": df_h2h_15,
                "h2h_10": df_h2h_10,
                "h2h_5": df_h2h_5,
            }

            def _render_history_tables(
                *,
                header: str,
                subtitle: str,
                history_slices: Dict[str, pd.DataFrame],
                metric_col: str,
            ) -> None:
                if not lines_list:
                    st.info("Sem linhas para este mercado.")
                    return
                st.markdown(f"**{header}**")
                st.caption(subtitle)
                for ln in lines_list:
                    ln_disp = _fmt_line_disp(metric_key, ln)
                    st.markdown(f"**{header} | Linha: {ln_disp}**")
                    tbl = core_ou_block(
                        history_slices.get("league_year", pd.DataFrame()),
                        history_slices.get("league_15", pd.DataFrame()),
                        history_slices.get("league_10", pd.DataFrame()),
                        history_slices.get("league_5", pd.DataFrame()),
                        history_slices.get("h2h_year", pd.DataFrame()),
                        history_slices.get("h2h_15", pd.DataFrame()),
                        history_slices.get("h2h_10", pd.DataFrame()),
                        history_slices.get("h2h_5", pd.DataFrame()),
                        value_col=metric_col,
                        line=float(ln),
                        floor_odd=1.10,
                    )
                    st.dataframe(tbl, width='stretch')

            _season_view_options = ["Atual"]
            _season_view_default = "Atual"
            _has_prev_reference = bool(_prev_season_label and _prev_history_slices)
            if _has_prev_reference:
                _prev_view_label = f"Ver {_prev_season_label}"
                _season_view_options.append(_prev_view_label)
                if not isinstance(df_h2h_year, pd.DataFrame) or df_h2h_year.empty:
                    _season_view_default = _prev_view_label
            else:
                _prev_view_label = ""

            _season_view = core_render_stateful_tab_selector(
                st,
                key=f"consistency_season_view_{metric_key}",
                options=_season_view_options,
                default=_season_view_default,
                label="Temporada da leitura",
            )

            _show_prev_view = bool(_prev_view_label and _season_view == _prev_view_label)
            _active_totals_for_lines = _prev_totals_for_lines if _show_prev_view else totals_for_lines
            _active_history_slices = _prev_history_slices if _show_prev_view else _current_history_slices
            _active_title = f"{title} | {_prev_season_label}" if _show_prev_view else title
            _active_history_subtitle = (
                f"Leitura histórica da {_prev_season_label} com o mesmo mercado e linhas."
                if _show_prev_view
                else "Frequência ajustada por recorte para as mesmas linhas, com foco em coerência e sanidade."
            )
            _active_model_subtitle = (
                f"Média esperada e odds justas do motor para a {_prev_season_label}."
                if _show_prev_view
                else "Média esperada e odds justas do motor para as linhas ativas."
            )

            # Modelo (média/odds)
            left, right = st.columns([1, 1])
            with left:
                core_render_market_mode_section(
                    st,
                    title=("Modelo" if not _show_prev_view else f"Modelo | {_prev_season_label}"),
                    subtitle=_active_model_subtitle,
                )
                if _show_prev_view and _prev_reference_note:
                    st.caption(_prev_reference_note)
                t = _active_totals_for_lines.get(metric_key)
                if t is not None and math.isfinite(float(getattr(t, "mean", float("nan")))):
                    mean = t.mean
                    sd = t.sd
                    if metric_key == "time":
                        st.write(f"M\u00e9dia: {_min_to_mmss(mean)}  |  SD: {_min_to_mmss(sd)}")
                    else:
                        st.write(f"M\u00e9dia: {mean:.2f}  |  SD: {sd:.2f}")
                    st.dataframe(_display_ou_table(_lines_table(metric_key if metric_key != "time" else "time", lines_list, t)), width='stretch')
                else:
                    st.info("Modelo indisponível para este mercado.")

            with right:
                core_render_market_mode_section(
                    st,
                    title=("Histórico" if not _show_prev_view else f"Histórico | {_prev_season_label}"),
                    subtitle=_active_history_subtitle,
                )
                if _show_prev_view and _prev_reference_note:
                    st.caption(_prev_reference_note)
                _render_history_tables(
                    header=_active_title,
                    subtitle=_active_history_subtitle,
                    history_slices=_active_history_slices,
                    metric_col=value_col,
                )
                if metric_key == "dragons" and core_has_metric_history(
                    _active_history_slices.get("league_year", pd.DataFrame()),
                    _active_history_slices.get("h2h_year", pd.DataFrame()),
                    metric_col="total_elementaldrakes",
                ):
                    st.divider()
                    _render_history_tables(
                        header=(
                            f"Dragões Elementais | {_prev_season_label}"
                            if _show_prev_view
                            else "Dragões Elementais"
                        ),
                        subtitle=(
                            "Só dragões elementais na temporada anterior; ancião fica fora."
                            if _show_prev_view
                            else "Conta só os dragões elementais. Ancião não entra nessa leitura."
                        ),
                        history_slices=_active_history_slices,
                        metric_col="total_elementaldrakes",
                    )
    
        if _consistency_market_tab == "Kills":
            _render_metric_tab("kills", "Kills (Total)", "total_kills", kills)
        elif _consistency_market_tab == "Torres":
            _render_metric_tab("towers", "Torres (Total)", "total_towers", towers)
        elif _consistency_market_tab == "Dragões":
            _render_metric_tab("dragons", "Drag\u00f5es (Total)", "total_dragons", dragons)
        elif _consistency_market_tab == "Dragões Elementais":
            _render_metric_tab("dragons_elemental", "Dragões Elementais", "total_elementaldrakes", dragons)
        elif _consistency_market_tab == "Barões":
            _render_metric_tab("barons", "Bar\u00f5es (Total)", "total_nashors", barons)
        elif _consistency_market_tab == "Inibidores":
            _render_metric_tab("inhibitors", "Inibidores (Total)", "total_inhibitors", inhib)
        elif _consistency_market_tab == "Tempo":
            _render_metric_tab("time", "Tempo", "game_time_min", time_m)
    
        elif _consistency_market_tab == "Kills p/ Time":
            st.markdown("**Total Kills por time (linha por time)**")
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"### {teamA}")
                if not teamA_kills_lines:
                    st.info("Digite uma linha em Extras: Total Kills do Time 1.")
                else:
                    for ln in teamA_kills_lines:
                        st.markdown(f"**Linha {ln:g}**")
                        tbl = core_ou_block(
                            dfA_all, dfA_all.tail(15), dfA_all.tail(10), dfA_all.tail(5),
                            df_h2h_A, df_h2h_A.tail(15), df_h2h_A.tail(10), df_h2h_A.tail(5),
                            value_col="kills_for", line=float(ln), floor_odd=1.10,
                        )
                        st.dataframe(tbl, width='stretch')
            with colB:
                st.markdown(f"### {teamB}")
                if not teamB_kills_lines:
                    st.info("Digite uma linha em Extras: Total Kills do Time 2.")
                else:
                    for ln in teamB_kills_lines:
                        st.markdown(f"**Linha {ln:g}**")
                        tbl = core_ou_block(
                            dfB_all, dfB_all.tail(15), dfB_all.tail(10), dfB_all.tail(5),
                            df_h2h_B, df_h2h_B.tail(15), df_h2h_B.tail(10), df_h2h_B.tail(5),
                            value_col="kills_for", line=float(ln), floor_odd=1.10,
                        )
                        st.dataframe(tbl, width='stretch')
    
        elif _consistency_market_tab == "Handicaps":
            st.markdown("**Handicaps (cobrir vs não cobrir)**")
            if (not hc_kills_lines) and (not hc_towers_lines) and (not hc_dragons_lines):
                st.info("Digite algum handicap em Extras.")
            else:
                def _render_hcap(
                    title: str,
                    diff_col_for: str,
                    diff_col_against: str,
                    line: Optional[float],
                    df_year: pd.DataFrame,
                    df_h2h_year: pd.DataFrame,
                ):
                    if line is None:
                        return
                    st.markdown(f"### {title} (linha {line:+g})")
    
                    def event_cover(df: pd.DataFrame):
                        diff = pd.to_numeric(df.get(diff_col_for), errors="coerce") - pd.to_numeric(df.get(diff_col_against), errors="coerce")
                        adj = diff + float(line)
                        return adj > 0
    
                    tbl = core_event_block(
                        df_year,
                        df_year.tail(15),
                        df_year.tail(10),
                        df_year.tail(5),
                        df_h2h_year,
                        df_h2h_year.tail(15),
                        df_h2h_year.tail(10),
                        df_h2h_year.tail(5),
                        event_fn=event_cover,
                        label_yes="Cover",
                        label_no="Fail",
                        floor_odd=1.10,
                    )
                    st.dataframe(tbl, width='stretch')
    
                # HC Kills: mostrar os dois lados automaticamente (Time escolhido + o oposto)
                if hc_kills_lines:
                    _hc_team = st.session_state.get("hc_kills_team") or teamA
                    if _hc_team not in [teamA, teamB]:
                        _hc_team = teamA
                    _other_team = teamB if _hc_team == teamA else teamA
    
                    _df_for = dfA_all if _hc_team == teamA else dfB_all
                    _df_h_for = df_h2h_A if _hc_team == teamA else df_h2h_B
                    _df_other = dfB_all if _hc_team == teamA else dfA_all
                    _df_h_other = df_h2h_B if _hc_team == teamA else df_h2h_A
    
                    for hln in hc_kills_lines:
                        st.markdown(f"### Linha HC Kills {hln:+g}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"#### {_hc_team}")
                            _render_hcap(
                                f"HC Kills ({_hc_team})",
                                "kills_for",
                                "kills_against",
                                float(hln),
                                _df_for,
                                _df_h_for,
                            )
                        with col2:
                            st.markdown(f"#### {_other_team}")
                            _render_hcap(
                                f"HC Kills ({_other_team})",
                                "kills_for",
                                "kills_against",
                                -float(hln),
                                _df_other,
                                _df_h_other,
                            )
    
                # Mantém Torres/Dragões como antes (perspectiva Time 1) para evitar confusão
                for hln in hc_towers_lines:
                    _render_hcap("HC Torres (Time 1)", "towers_for", "towers_against", float(hln), dfA_all, df_h2h_A)
                for hln in hc_dragons_lines:
                    _render_hcap("HC Dragões (Time 1)", "dragons_for", "dragons_against", float(hln), dfA_all, df_h2h_A)
    
        elif _consistency_market_tab == "Primeiros Objetivos":
            # Reaproveita bloco original do backup (primeiros objetivos)
            st.markdown("**Primeiros Objetivos (probabilidade do time conseguir)**")
            obj_cols = []
            for c in ["firstblood", "firsttower", "firstdragon", "firstherald", "firstbaron", "firstinhibitor"]:
                if c in dfA_all.columns:
                    obj_cols.append(c)
            if not obj_cols:
                st.info("CSV não possui colunas de primeiros objetivos (firstblood/firsttower/...).")
            else:
                obj_tab_names = {
                    "firstblood": "FB",
                    "firsttower": "FT",
                    "firstdragon": "FD",
                    "firstherald": "FH",
                    "firstbaron": "FBaron",
                    "firstinhibitor": "FInib",
                }
                _first_objective_tab = core_render_stateful_tab_selector(
                    st,
                    key="consistency_first_objective_tab",
                    options=[obj_tab_names.get(c, c) for c in obj_cols],
                    default=obj_tab_names.get(obj_cols[0], obj_cols[0]),
                    label="Primeiros Objetivos",
                )
                for _i, col in enumerate(obj_cols):
                    _obj_name = obj_tab_names.get(col, col)
                    if _first_objective_tab != _obj_name:
                        continue
                    st.markdown(f"### {_obj_name}")

                    def ev_yes(df: pd.DataFrame):
                        return pd.to_numeric(df.get(col), errors="coerce") > 0.5

                    def _team_row(df_year, df_15, df_10, df_5):
                        y1 = n1 = 0
                        if df_year is not None and not df_year.empty:
                            ok = ev_yes(df_year).fillna(False).astype(bool)
                            n1 = int(ok.shape[0]); y1 = int(ok.sum())
                        y15 = n15 = 0
                        if df_15 is not None and not df_15.empty:
                            ok = ev_yes(df_15).fillna(False).astype(bool)
                            n15 = int(ok.shape[0]); y15 = int(ok.sum())
                        y10 = n10 = 0
                        if df_10 is not None and not df_10.empty:
                            ok = ev_yes(df_10).fillna(False).astype(bool)
                            n10 = int(ok.shape[0]); y10 = int(ok.sum())
                        y5 = n5 = 0
                        if df_5 is not None and not df_5.empty:
                            ok = ev_yes(df_5).fillna(False).astype(bool)
                            n5 = int(ok.shape[0]); y5 = int(ok.sum())
                        def fmt(y, n):
                            _odd = core_laplace_odds(y, n, floor_odd=1.10)
                            return f"{y}/{n} ({float(_odd):.2f}x)" if (n and _odd is not None and math.isfinite(float(_odd))) else (f"{y}/{n}" if n else "-")
                        return [fmt(y1, n1), fmt(y15, n15), fmt(y10, n10), fmt(y5, n5)]

                    rowA_l = _team_row(dfA_all, dfA_all.tail(15), dfA_all.tail(10), dfA_all.tail(5))
                    rowB_l = _team_row(dfB_all, dfB_all.tail(15), dfB_all.tail(10), dfB_all.tail(5))
                    rowA_h = _team_row(df_h2h_A, df_h2h_A.tail(15), df_h2h_A.tail(10), df_h2h_A.tail(5))
                    rowB_h = _team_row(df_h2h_B, df_h2h_B.tail(15), df_h2h_B.tail(10), df_h2h_B.tail(5))

                    tbl = pd.DataFrame(
                        [rowA_l, rowB_l, rowA_h, rowB_h],
                        columns=["Ano todo", "Últimos 15", "Últimos 10", "Últimos 5"],
                        index=[f"Liga - {teamA}", f"Liga - {teamB}", f"H2H - {teamA}", f"H2H - {teamB}"],
                    )
                    st.dataframe(tbl, width='stretch')

        elif _consistency_market_tab in {"Vencedor", "Win Rate"}:
            core_render_market_mode_section(
                st,
                title="Histórico",
                subtitle="Win rate geral, forma recente e H2H com contexto completo de campeonato.",
            )
            st.markdown("**Win Rate (vencedor)**")

            _filters_all_comp = Filters(
                year=filters.year,
                league=None,
                split=filters.split,
                playoffs=filters.playoffs,
            )
            _filters_all_comp_no_split = Filters(
                year=filters.year,
                league=None,
                split=None,
                playoffs=filters.playoffs,
            )
            dfA_wr_comp = filter_team_games(
                team_games,
                team=teamA,
                filters=_filters_all_comp,
                league_mode="all",
                map_mode=map_mode,
                max_games=max_games,
            )
            dfB_wr_comp = filter_team_games(
                team_games,
                team=teamB,
                filters=_filters_all_comp,
                league_mode="all",
                map_mode=map_mode,
                max_games=max_games,
            )
            dfA_wr_season_comp = filter_team_games(
                team_games,
                team=teamA,
                filters=_filters_all_comp_no_split,
                league_mode="all",
                map_mode=map_mode,
                max_games=max_games,
            )
            dfB_wr_season_comp = filter_team_games(
                team_games,
                team=teamB,
                filters=_filters_all_comp_no_split,
                league_mode="all",
                map_mode=map_mode,
                max_games=max_games,
            )
            try:
                df_h2h_wr_comp = dfA_wr_comp[dfA_wr_comp["opponent"] == teamB].copy()
            except KeyError:
                df_h2h_wr_comp = pd.DataFrame()

            def _to_win_frame(df_in: pd.DataFrame) -> pd.DataFrame:
                if df_in is None or (not isinstance(df_in, pd.DataFrame)) or df_in.empty:
                    return pd.DataFrame(columns=["date", "opponent", "win", "side", "league"])
                d = df_in.copy()
                if "date" not in d.columns:
                    d["date"] = pd.NaT
                if "opponent" not in d.columns:
                    d["opponent"] = "-"
                d["win"] = pd.to_numeric(d.get("win"), errors="coerce")
                if "side" not in d.columns:
                    d["side"] = ""
                if "league" not in d.columns:
                    d["league"] = ""
                return d[["date", "opponent", "win", "side", "league"]].copy()

            def _wr_tuple(df_in: pd.DataFrame) -> tuple[int, int, int, float]:
                if df_in is None or df_in.empty:
                    return 0, 0, 0, float("nan")
                w = pd.to_numeric(df_in.get("win"), errors="coerce")
                m = w.notna()
                n = int(m.sum())
                if n <= 0:
                    return 0, 0, 0, float("nan")
                k = int((w[m] > 0.5).sum())
                l = int(n - k)
                return n, k, l, float(k / n)

            def _side_wr(df_in: pd.DataFrame, side_token: str) -> float:
                if df_in is None or df_in.empty or "side" not in df_in.columns:
                    return float("nan")
                side_ser = df_in["side"].astype(str).str.strip().str.lower()
                sub = df_in[side_ser == str(side_token).strip().lower()].copy()
                _, _, _, wr = _wr_tuple(sub)
                return float(wr)

            def _fmt_wr(wr: float) -> str:
                return f"{100.0 * float(wr):.1f}%" if math.isfinite(float(wr)) else "-"

            def _recent_rows(df_in: pd.DataFrame, n: int = 10) -> list[str]:
                d = _to_win_frame(df_in).sort_values("date").tail(int(n))
                out = []
                for _, r in d.iterrows():
                    opp = str(r.get("opponent", "") or "-")
                    league_name = str(r.get("league", "") or "").strip()
                    wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                    mark = "W" if pd.notna(wv) and float(wv) > 0.5 else "L"
                    if league_name:
                        out.append(f"{opp} {mark} | {league_name}")
                    else:
                        out.append(f"{opp} {mark}")
                return out

            def _vs_table(df_in: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
                d = _to_win_frame(df_in)
                if d.empty:
                    return pd.DataFrame(columns=["Vs", "Games", "W", "L", "Win Rate"])
                g = (
                    d.groupby("opponent", dropna=False)
                    .agg(Games=("win", "size"), W=("win", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.5).sum())))
                    .reset_index()
                )
                g["L"] = g["Games"] - g["W"]
                g["Win Rate"] = (100.0 * (g["W"] / g["Games"])).round(1).astype(str) + "%"
                g = g.rename(columns={"opponent": "Vs"}).sort_values(["Games", "W"], ascending=[False, False]).head(int(top_n))
                return g[["Vs", "Games", "W", "L", "Win Rate"]]

            dfA_w = _to_win_frame(dfA_wr_comp)
            dfB_w = _to_win_frame(dfB_wr_comp)
            dfA_w_season = _to_win_frame(dfA_wr_season_comp)
            dfB_w_season = _to_win_frame(dfB_wr_season_comp)
            df_h2h_w = _to_win_frame(df_h2h_wr_comp)

            c_over, c_recent, c_h2h = st.columns(3)
            with c_over:
                core_render_summary_block(st, "OVERALL")
                with st.container(border=True):
                    nA, wA, lA, wrA = _wr_tuple(dfA_w)
                    nB, wB, lB, wrB = _wr_tuple(dfB_w)
                    nA2, wA2, lA2, wrA2 = _wr_tuple(dfA_w_season)
                    nB2, wB2, lB2, wrB2 = _wr_tuple(dfB_w_season)
                    st.markdown(f"**{teamA}**  |  **{teamB}**")
                    st.caption("Current Split")
                    _tbl_current = pd.DataFrame(
                        [
                            {"Métrica": "Win Rate", teamA: _fmt_wr(wrA), teamB: _fmt_wr(wrB)},
                            {"Métrica": "# Games", teamA: nA, teamB: nB},
                            {"Métrica": "Record", teamA: f"{wA}W-{lA}L", teamB: f"{wB}W-{lB}L"},
                            {"Métrica": "Blue Side", teamA: _fmt_wr(_side_wr(dfA_w, "blue")), teamB: _fmt_wr(_side_wr(dfB_w, "blue"))},
                            {"Métrica": "Red Side", teamA: _fmt_wr(_side_wr(dfA_w, "red")), teamB: _fmt_wr(_side_wr(dfB_w, "red"))},
                        ]
                    ).astype(str)
                    st.dataframe(
                        _tbl_current,
                        width='stretch',
                        hide_index=True,
                    )
                    st.caption("All Season")
                    _tbl_all = pd.DataFrame(
                        [
                            {"Métrica": "Win Rate", teamA: _fmt_wr(wrA2), teamB: _fmt_wr(wrB2)},
                            {"Métrica": "# Games", teamA: nA2, teamB: nB2},
                            {"Métrica": "Record", teamA: f"{wA2}W-{lA2}L", teamB: f"{wB2}W-{lB2}L"},
                        ]
                    ).astype(str)
                    st.dataframe(
                        _tbl_all,
                        width='stretch',
                        hide_index=True,
                    )
                    st.markdown(f"**{teamA} - Vs (amostra)**")
                    st.dataframe(_vs_table(dfA_w_season), width='stretch', hide_index=True)
                    st.markdown(f"**{teamB} - Vs (amostra)**")
                    st.dataframe(_vs_table(dfB_w_season), width='stretch', hide_index=True)

            with c_recent:
                core_render_summary_block(st, "RECENT FORM")
                with st.container(border=True):
                    _, _, _, wrA5 = _wr_tuple(dfA_w.tail(5))
                    _, _, _, wrA10 = _wr_tuple(dfA_w.tail(10))
                    _, _, _, wrB5 = _wr_tuple(dfB_w.tail(5))
                    _, _, _, wrB10 = _wr_tuple(dfB_w.tail(10))
                    st.markdown(f"**{teamA}**")
                    st.caption(f"Last 5: {_fmt_wr(wrA5)} | Last 10: {_fmt_wr(wrA10)}")
                    _ra = _recent_rows(dfA_w, 10)
                    st.markdown("\n".join([f"- {x}" for x in _ra]) if _ra else "- sem jogos")
                    st.markdown("---")
                    st.markdown(f"**{teamB}**")
                    st.caption(f"Last 5: {_fmt_wr(wrB5)} | Last 10: {_fmt_wr(wrB10)}")
                    _rb = _recent_rows(dfB_w, 10)
                    st.markdown("\n".join([f"- {x}" for x in _rb]) if _rb else "- sem jogos")

            with c_h2h:
                core_render_summary_block(st, "PAST FACEOFFS")
                with st.container(border=True):
                    st.markdown(f"**{teamA} vs {teamB}**")
                    nH, _wH, _lH, wrH = _wr_tuple(df_h2h_w)
                    st.metric("H2H Games", int(nH))
                    st.metric(teamA, _fmt_wr(wrH))
                    st.metric(teamB, _fmt_wr(1.0 - wrH) if math.isfinite(float(wrH)) else "-")
                    if nH > 0:
                        d = df_h2h_w.sort_values("date").tail(10).copy()
                        rows = []
                        for _, r in d.iterrows():
                            dt = pd.to_datetime(r.get("date"), errors="coerce")
                            dts = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "-"
                            league_name = str(r.get("league", "") or "").strip()
                            wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                            winner = teamA if (pd.notna(wv) and float(wv) > 0.5) else teamB
                            suffix = f" | {league_name}" if league_name else ""
                            rows.append(f"- {dts} - {winner}{suffix}")
                        st.markdown("\n".join(rows) if rows else "- sem H2H")
                    else:
                        st.markdown("- sem H2H")
    
        elif _consistency_market_tab == "Stats":
            core_render_market_stats_tab(
                st=st,
                session_state=st.session_state,
                team_games=team_games,
                filters_cls=Filters,
                active_filters=filters,
                apply_filters_fn=apply_filters,
                ml_info=_ml_info,
                map_mode_ui=ctx.get("map_mode_ui"),
                team_a=teamA,
                team_b=teamB,
                parse_lines_fn=parse_lines_text,
                parse_time_lines_fn=parse_time_lines_text,
                min_to_mmss_fn=_min_to_mmss,
                format_num_fn=_format_num,
                current_lines_by_metric={
                    "kills": list(kills or []),
                    "towers": list(towers or []),
                    "dragons": list(dragons or []),
                    "barons": list(barons or []),
                    "inhibitors": list(inhib or []),
                    "time": list(time_m or []),
                },
            )

        elif _consistency_market_tab == "Perfil de Oponente":
            core_render_opponent_profile_tab(
                st=st,
                session_state=st.session_state,
                team_games=team_games,
                filters_cls=Filters,
                active_filters=filters,
                apply_filters_fn=apply_filters,
                ml_info=_ml_info,
                map_mode_ui=ctx.get("map_mode_ui"),
                team_a=teamA,
                team_b=teamB,
                parse_lines_fn=parse_lines_text,
                parse_time_lines_fn=parse_time_lines_text,
                min_to_mmss_fn=_min_to_mmss,
                format_num_fn=_format_num,
                current_lines_by_metric={
                    "kills": list(kills or []),
                    "towers": list(towers or []),
                    "dragons": list(dragons or []),
                    "barons": list(barons or []),
                    "inhibitors": list(inhib or []),
                    "time": list(time_m or []),
                },
            )

        elif _consistency_market_tab == "ML + Totais":
            st.markdown("**ML + Totais (frequência ajustada)**")
            core_render_market_mode_section(
                st,
                title="Modelo",
                subtitle="Odds justas combinando ML e total nas linhas específicas de ML + Totais.",
            )
            _ml_totals_payload = core_build_ml_totals_model_payload(
                team_a=teamA,
                team_b=teamB,
                totals_for_lines=totals_for_lines,
                combo_kills_lines=combo_kills_lines,
                combo_time_lines=combo_time_lines,
                p_ml_scope=core_get_ml_scope(st.session_state),
                lines_table_with_p_ml_fn=_lines_table_with_p_ml,
                display_ml_totals_table_fn=_display_ml_totals_table,
            )
            if not bool(_ml_totals_payload.get("has_any_lines", False)):
                st.info('Preencha em Extras: "Linha Kills (para ML + Totais)" e/ou "Linha Tempo (para ML + Totais)".')
            else:
                _consistency_ml_totals_tab = core_render_stateful_tab_selector(
                    st,
                    key="consistency_ml_totals_tab",
                    options=["Kills", "Tempo"],
                    default="Kills",
                    label="ML + Totais",
                )
                _tabs_payload = list(_ml_totals_payload.get("tabs", []) or [])
                for _i, _tab in enumerate(_tabs_payload):
                    if _consistency_ml_totals_tab != str(_tab.get("label", "") or "Kills"):
                        continue
                    if not bool(_tab.get("has_lines", False)):
                        st.info('Sem linhas para este mercado.')
                    else:
                        cA, cB = st.columns(2)
                        with cA:
                            st.markdown(f"**{_tab.get('team_a_label', f'Time 1: {teamA}')}**")
                            st.dataframe(_tab.get("table_a", pd.DataFrame()), width='stretch')
                        with cB:
                            st.markdown(f"**{_tab.get('team_b_label', f'Time 2: {teamB}')}**")
                            st.dataframe(_tab.get("table_b", pd.DataFrame()), width='stretch')
            st.divider()
            core_render_market_mode_section(
                st,
                title="Histórico",
                subtitle="Frequência ajustada por lado e por linha usando os mesmos recortes do confronto.",
            )
            _lap_payload = core_build_ml_totals_laplace_payload(
                team_a=teamA,
                team_b=teamB,
                combo_kills_lines=combo_kills_lines,
                combo_time_lines=combo_time_lines,
                df_a_all=dfA_all,
                df_b_all=dfB_all,
                df_h2h_a=df_h2h_A,
                df_h2h_b=df_h2h_B,
                event_block_fn=core_event_block,
                floor_odd=1.10,
            )
            if not bool(_lap_payload.get("has_any_lines", False)):
                st.info("Digite uma linha em Extras: Linha Kills e/ou Linha Tempo.")
            else:
                for _r in list(_lap_payload.get("rows", []) or []):
                    st.markdown(f"### {_r.get('title', 'ML + Totais')!s} (linha {float(_r.get('line', 0.0))})")
                    c_l, c_r = st.columns(2)
                    with c_l:
                        st.markdown(f"**{_r.get('team_a_label', f'Time de referência (Time 1): {teamA}')!s}**")
                        st.dataframe(_r.get("tbl_over_a", pd.DataFrame()), width='stretch')
                        st.dataframe(_r.get("tbl_under_a", pd.DataFrame()), width='stretch')
                    with c_r:
                        st.markdown(f"**{_r.get('team_b_label', f'Time de referência (Time 2): {teamB}')!s}**")
                        st.dataframe(_r.get("tbl_over_b", pd.DataFrame()), width='stretch')
                        st.dataframe(_r.get("tbl_under_b", pd.DataFrame()), width='stretch')
    
        elif _consistency_market_tab == "Detalhes dos Jogos":
            core_render_market_mode_section(
                st,
                title="Leitura",
                subtitle="Use o modo simples para decisão rápida e auditoria para inspeção completa.",
            )
            def _build_audit_market_options_ui() -> Dict[str, Tuple[str, List[float]]]:
                opts = dict(
                    core_build_audit_market_options(
                        teamA,
                        teamB,
                        kills,
                        towers,
                        dragons,
                        barons,
                        inhib,
                        time_m,
                        teamA_kills_lines,
                        teamB_kills_lines,
                    )
                    or {}
                )
                opts["Total Dragões Elementais"] = ("total_elementaldrakes", list(dragons or []))
                return opts

            options = core_build_audit_base_options(
                teamA,
                teamB,
                df_league_year,
                df_league_15,
                df_league_10,
                df_league_5,
                df_h2h_year,
                df_h2h_15,
                df_h2h_10,
                df_h2h_5,
                dfA_context_year_full,
                dfB_context_year_full,
            )
            ds_name = st.selectbox("Base", [o[0] for o in options], index=0)
            df_base = dict(options).get(ds_name)
            if str(st.session_state.get("audit_detail_mode", "") or "") not in {"Simples", "Auditoria"}:
                st.session_state["audit_detail_mode"] = "Simples"
            detail_mode = st.radio("Modo", ["Simples", "Auditoria"], horizontal=True, key="audit_detail_mode")

            market_opts = _build_audit_market_options_ui()
            market_name = st.selectbox("Mercado", list(market_opts.keys()), index=0)
            col, line_list = market_opts.get(market_name, (None, []))
            line_list = list(line_list or [])
            ln = st.selectbox("Linha para classificar (Acima/Abaixo)", options=line_list, index=0) if line_list else None
            if df_base is None or df_base.empty:
                st.info("Base vazia para este recorte.")
            else:
                if str(detail_mode) == "Simples":
                    view = core_build_simple_audit_view(df_base, col, (float(ln) if ln is not None else None), _min_to_mmss)
                else:
                    view = core_build_audit_view(df_base, col, (float(ln) if ln is not None else None), _min_to_mmss)
                st.dataframe(view, width='stretch', height=420)

        elif _consistency_market_tab == "Seletor Rápido":
            def _build_quick_metric_options_ui(
                team_a: str,
                team_b: str,
                kills_lines: list[float],
                towers_lines: list[float],
                dragons_lines: list[float],
                barons_lines: list[float],
                inhib_lines: list[float],
                time_lines: list[float],
                team_a_lines: list[float],
                team_b_lines: list[float],
                team_sel: str,
            ) -> Dict[str, Tuple[str, str, List[float]]]:
                opts = dict(
                    core_build_quick_metric_options(
                        team_a,
                        team_b,
                        kills_lines,
                        towers_lines,
                        dragons_lines,
                        barons_lines,
                        inhib_lines,
                        time_lines,
                        team_a_lines,
                        team_b_lines,
                        team_sel,
                    )
                    or {}
                )
                opts["Dragões Elementais"] = ("dragons_elemental", "total_elementaldrakes", list(dragons_lines or []))
                return opts

            quick_selector = core_render_quick_selector_controls(
                st=st,
                session_state=st.session_state,
                team_a=teamA,
                team_b=teamB,
                kills=kills,
                towers=towers,
                dragons=dragons,
                barons=barons,
                inhib=inhib,
                time_m=time_m,
                team_a_kills_lines=teamA_kills_lines,
                team_b_kills_lines=teamB_kills_lines,
                build_metric_options_fn=_build_quick_metric_options_ui,
                parse_lines_fn=parse_lines_text,
                parse_time_lines_fn=parse_time_lines_text,
                num_cast_fn=_fnum,
            )
            team_sel = str(quick_selector.get("team_sel", teamA) or teamA)
            side_sel_raw = str(quick_selector.get("side_sel_raw", "Blue") or "Blue")
            result_sel = str(quick_selector.get("result_sel", "Vit\u00f3ria") or "Vit\u00f3ria")
            metric_name = str(quick_selector.get("metric_name", "") or "")
            metric_key = str(quick_selector.get("metric_key", "kills") or "kills")
            value_col = str(quick_selector.get("value_col", "total_kills") or "total_kills")
            selected_lines = list(quick_selector.get("selected_lines", []) or [])

            if not selected_lines:
                st.info("Sem linhas para o mercado selecionado. Use linha pré-definida e/ou linha manual.")
            else:
                team_is_a = str(team_sel) == str(teamA)
                df_team, df_h2h_team = core_quick_team_frame(team_is_a, dfA_all, dfB_all, df_h2h_A, df_h2h_B)
                side_key_raw = str(side_sel_raw).strip().lower()
                side_key = None if side_key_raw.startswith("amb") else side_key_raw
                result_key_raw = str(result_sel).strip().lower()
                if result_key_raw.startswith("amb"):
                    want_win = None
                else:
                    want_win = bool(result_key_raw.startswith("vit"))
                _all_rows = []
                for line_f in selected_lines:
                    _pml_ss = core_get_ml_scope(st.session_state)
                    p_ml_base = float(_pml_ss) if _pml_ss is not None else float("nan")
                    p_outcome, p_side_ctx = core_quick_outcome_probs(
                        p_ml_scope=p_ml_base,
                        team_is_a=team_is_a,
                        want_win=want_win,
                        df_team=df_team,
                        side_key=side_key,
                    )
    
                    p_over = None
                    if metric_key in totals_for_lines:
                        try:
                            p_over = float(total_over_prob(totals_for_lines[metric_key], line_f))
                        except (TypeError, ValueError, OverflowError):
                            p_over = None
                    elif metric_key == "team_kills":
                        try:
                            if team_is_a:
                                mean_k = _combine_mean(pA.kills_for, pB.kills_against)
                                sd_k = _combine_sd(pA.kills_for_sd, pB.kills_against_sd)
                            else:
                                mean_k = _combine_mean(pB.kills_for, pA.kills_against)
                                sd_k = _combine_sd(pB.kills_for_sd, pA.kills_against_sd)
                            obj_k = MatchupTotals(mean=float(mean_k), sd=float(max(sd_k, 1.0)), dist="normal")
                            p_over = float(total_over_prob(obj_k, line_f))
                        except (TypeError, ValueError, OverflowError):
                            p_over = None

                    def _line_fmt(metric_name_key: str, line_value: float) -> str:
                        return _min_to_mmss(line_value) if str(metric_name_key) == "time" else f"{float(line_value):g}"

                    result_rows = core_build_quick_selector_rows(
                        df_team=df_team,
                        df_h2h_team=df_h2h_team,
                        value_col=value_col,
                        line_f=float(line_f),
                        side_key=side_key,
                        want_win=want_win,
                        p_outcome=float(p_outcome),
                        p_side_ctx=p_side_ctx,
                        p_over=p_over,
                        laplace_pick=str(st.session_state.get("resumo_laplace_pick", "Liga 15") or "Liga 15"),
                        mix_w_model=float(st.session_state.get("resumo_mix_w_model", 40) or 40),
                        mix_w_laplace=float(st.session_state.get("resumo_mix_w_laplace", 40) or 40),
                        mix_w_sc=float(st.session_state.get("resumo_mix_w_sc", 20) or 20),
                        metric_key=str(metric_key),
                        format_num_fn=_format_num,
                        line_formatter_fn=_line_fmt,
                    )
                    _all_rows.extend(result_rows)

                st.markdown(f"**Configuração:** {team_sel} | {side_sel_raw} | {result_sel} | {metric_name}")
                st.dataframe(pd.DataFrame(_all_rows), width='stretch', hide_index=True)
        elif _consistency_market_tab == "Resumo":
            _render_full_resumo_odds_tab()

