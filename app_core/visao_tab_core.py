# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import json
import math
from html import escape as _esc
from typing import Any, Callable, Dict, List, Mapping

import pandas as pd

from app_core.render_ctx_core import validate_render_ctx
from app_core.team_logos_core import team_logo_html


def _vg_caption(st: Any) -> Callable[[str], None]:
    """Caption fn for the ML pipeline that drops the 'Guardião marginal' line.

    That diagnostic is technical noise the user asked to hide on Visão Geral;
    other pipeline captions (e.g. drift warnings) still show.
    """
    def _cap(msg: str) -> None:
        s = str(msg or "")
        if s.startswith(("Guardião marginal", "Guardiao marginal")):
            return
        st.caption(s)
    return _cap


def _validate_visao_ctx(ctx: Mapping[str, Any]) -> None:
    """Loud-fail if a required core_* import is missing from the render ctx.

    Guards the ctx=globals() contract: a removed import would otherwise become
    a silent None and crash deep in the render. See render_ctx_core.
    """
    try:
        src = inspect.getsource(render_visao_tab)
    except (OSError, TypeError):
        return  # source unavailable (frozen/optimized) — skip rather than break
    validate_render_ctx(ctx, renderer_source=src, renderer_name="render_visao_tab")


def render_visao_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    _validate_visao_ctx(ctx)
    CoreMlPrePipelineOutput = ctx.get("CoreMlPrePipelineOutput")
    _FIT_PARAMS_PATH = ctx.get("_FIT_PARAMS_PATH")
    _LEAGUE_TRANSFER_FEATURES_PATH = ctx.get("_LEAGUE_TRANSFER_FEATURES_PATH")
    _ML_CONFIDENCE_BASELINE_PATH = ctx.get("_ML_CONFIDENCE_BASELINE_PATH")
    _POS_CANON = ctx.get("_POS_CANON")
    _POS_CANON_ML = ctx.get("_POS_CANON_ML")
    _TEAM_STATE_ARTIFACT_PATH = ctx.get("_TEAM_STATE_ARTIFACT_PATH")
    _append_feature_trace = ctx.get("_append_feature_trace")
    _apply_ml_marginal_trust_adjust = ctx.get("_apply_ml_marginal_trust_adjust")
    _apply_ml_uncertainty_adjust = ctx.get("_apply_ml_uncertainty_adjust")
    _apply_series_gamma = ctx.get("_apply_series_gamma")
    _apply_shrink_m = ctx.get("_apply_shrink_m")
    _artifact_sig = ctx.get("_artifact_sig")
    _best_match = ctx.get("_best_match")
    _calibrate_p_from_artifact = ctx.get("_calibrate_p_from_artifact")
    _delta_players_effect = ctx.get("_delta_players_effect")
    _df_contract_view = ctx.get("_df_contract_view")
    _elo_season_players_p_map = ctx.get("_elo_season_players_p_map")
    _ensure_lineup_runtime_state = ctx.get("_ensure_lineup_runtime_state")
    _feature_flag_enabled = ctx.get("_feature_flag_enabled")
    _file_sig = ctx.get("_file_sig")
    _fit_form_for_league = ctx.get("_fit_form_for_league")
    _fit_players_lane_for_league = ctx.get("_fit_players_lane_for_league")
    _fit_players_league_weight_for_league = ctx.get("_fit_players_league_weight_for_league")
    _fnum = ctx.get("_fnum")
    _format_num = ctx.get("_format_num")
    _format_pct = ctx.get("_format_pct")
    _fuse_probs_by_precision = ctx.get("_fuse_probs_by_precision")
    _get_lineup_loaded_key = ctx.get("_get_lineup_loaded_key")
    _get_lineup_runtime_settings = ctx.get("_get_lineup_runtime_settings")
    _get_lineup_selected = ctx.get("_get_lineup_selected")
    _identity_team_name_candidates = ctx.get("_identity_team_name_candidates")
    _latest_lineup_for_team = ctx.get("_latest_lineup_for_team")
    _lines_table = ctx.get("_lines_table")
    _lines_table_with_p_ml = ctx.get("_lines_table_with_p_ml")
    _lineup_key = ctx.get("_lineup_key")
    _load_artifact_cached = ctx.get("_load_artifact_cached")
    _load_confidence_baseline_runtime = ctx.get("_load_confidence_baseline_runtime")
    _load_fitted_params_cached = ctx.get("_load_fitted_params_cached")
    _load_league_transfer_features_cached = ctx.get("_load_league_transfer_features_cached")
    _load_lineup_for_matchup = ctx.get("_load_lineup_for_matchup")
    _load_ml_confidence_baseline_cached = ctx.get("_load_ml_confidence_baseline_cached")
    _load_player_rows = ctx.get("_load_player_rows")
    _load_players_artifact_cached = ctx.get("_load_players_artifact_cached")
    _load_players_impact = ctx.get("_load_players_impact")
    _load_team_state_artifact_cached = ctx.get("_load_team_state_artifact_cached")
    _ml_confidence_from_layer_validation = ctx.get("_ml_confidence_from_layer_validation")
    _ml_consistency_signature = ctx.get("_ml_consistency_signature")
    _ml_info = ctx.get("_ml_info")
    _ml_runtime_cfg = ctx.get("_ml_runtime_cfg")
    _norm_pos = ctx.get("_norm_pos")
    _norm_pos_ml = ctx.get("_norm_pos_ml")
    _odd_from_p = ctx.get("_odd_from_p")
    _p_from_diff_points = ctx.get("_p_from_diff_points")
    _patch_changed_leagues_set = ctx.get("_patch_changed_leagues_set")
    _patch_meta_guard_payload = ctx.get("_patch_meta_guard_payload")
    _persist_combined_trace_file = ctx.get("_persist_combined_trace_file")
    _render_odds_cards = ctx.get("_render_odds_cards")
    _runtime_confidence_governor_budget_for_league = ctx.get("_runtime_confidence_governor_budget_for_league")
    _runtime_series_coherence_post_cap_for_league = ctx.get("_runtime_series_coherence_post_cap_for_league")
    _safe_key = ctx.get("_safe_key")
    _safe_logit = ctx.get("_safe_logit")
    _safe_sigmoid = ctx.get("_safe_sigmoid")
    _save_lineup_for_matchup = ctx.get("_save_lineup_for_matchup")
    _series_gamma_for_league = ctx.get("_series_gamma_for_league")
    _series_totals_table = ctx.get("_series_totals_table")
    _set_lineup_loaded_key = ctx.get("_set_lineup_loaded_key")
    _team_with_league_label = ctx.get("_team_with_league_label")
    _totals_table = ctx.get("_totals_table")
    artifact_path = ctx.get("artifact_path")
    as_of_date = ctx.get("as_of_date")
    barons = ctx.get("barons")
    bo = ctx.get("bo")
    clip_prob = ctx.get("clip_prob")
    combo_kills_text = ctx.get("combo_kills_text")
    combo_time_text = ctx.get("combo_time_text")
    compute_delta_and_confidence = ctx.get("compute_delta_and_confidence")
    core_append_ml_live_feature_traces = ctx.get("core_append_ml_live_feature_traces")
    core_append_ml_shadow_feature_traces = ctx.get("core_append_ml_shadow_feature_traces")
    core_apply_confidence_governor = ctx.get("core_apply_confidence_governor")
    core_apply_runtime_tuning_to_session = ctx.get("core_apply_runtime_tuning_to_session")
    core_apply_series_bo_dplite_live = ctx.get("core_apply_series_bo_dplite_live")
    core_autofill_lineup_state = ctx.get("core_autofill_lineup_state")
    core_build_confidence_summary = ctx.get("core_build_confidence_summary")
    core_build_manual_lineup_choices = ctx.get("core_build_manual_lineup_choices")
    core_build_microseg_caption = ctx.get("core_build_microseg_caption")
    core_build_ml_lineup_choices = ctx.get("core_build_ml_lineup_choices")
    core_build_ml_trace_context = ctx.get("core_build_ml_trace_context")
    core_build_ml_trace_payload = ctx.get("core_build_ml_trace_payload")
    core_build_pipeline_explainability = ctx.get("core_build_pipeline_explainability")
    core_build_pipeline_step_rows = ctx.get("core_build_pipeline_step_rows")
    core_build_pre_engine_input = ctx.get("core_build_pre_engine_input")
    core_build_pre_live_input = ctx.get("core_build_pre_live_input")
    core_build_runtime_input_signature = ctx.get("core_build_runtime_input_signature")
    core_compute_state_shadow_alite = ctx.get("core_compute_state_shadow_alite")
    core_compute_state_shadow_areal = ctx.get("core_compute_state_shadow_areal")
    core_compute_winner_pricing_bundle = ctx.get("core_compute_winner_pricing_bundle")
    core_current_season_override = ctx.get("core_current_season_override")
    core_enforce_series_economic_coherence = ctx.get("core_enforce_series_economic_coherence")
    core_get_lv_last_report = ctx.get("core_get_lv_last_report")
    core_get_ml_fused_post_cal_info = ctx.get("core_get_ml_fused_post_cal_info")
    core_get_ml_microseg_info = ctx.get("core_get_ml_microseg_info")
    core_get_ml_trace_snapshot = ctx.get("core_get_ml_trace_snapshot")
    core_get_wf_ml_corr_info = ctx.get("core_get_wf_ml_corr_info")
    core_hybrid_momentum_blend = ctx.get("core_hybrid_momentum_blend")
    core_league_is_excluded = ctx.get("core_league_is_excluded")
    core_lineup_continuity_from_details = ctx.get("core_lineup_continuity_from_details")
    core_persist_runtime_analysis_state = ctx.get("core_persist_runtime_analysis_state")
    core_render_lineup_selector_column = ctx.get("core_render_lineup_selector_column")
    core_render_stateful_tab_selector = ctx.get("core_render_stateful_tab_selector")
    core_resolve_players_layer_config = ctx.get("core_resolve_players_layer_config")
    core_resolve_shared_league_hint = ctx.get("core_resolve_shared_league_hint")
    core_run_ml_post_layers = ctx.get("core_run_ml_post_layers")
    core_run_ml_pre_pipeline = ctx.get("core_run_ml_pre_pipeline")
    core_run_ml_shadow_layers = ctx.get("core_run_ml_shadow_layers")
    core_score_manual_lineup_team = ctx.get("core_score_manual_lineup_team")
    core_season_form_blend = ctx.get("core_season_form_blend")
    core_set_latest_ml_trace = ctx.get("core_set_latest_ml_trace")
    core_set_ml_scope = ctx.get("core_set_ml_scope")
    core_transfer_signal_from_details = ctx.get("core_transfer_signal_from_details")
    core_unpack_post_layer_output = ctx.get("core_unpack_post_layer_output")
    core_unpack_shadow_output = ctx.get("core_unpack_shadow_output")
    csv_path = ctx.get("csv_path")
    csv_sig = ctx.get("csv_sig")
    dfA_used = ctx.get("dfA_used")
    dfB_used = ctx.get("dfB_used")
    dragons = ctx.get("dragons")
    filters = ctx.get("filters")
    fixed_league = ctx.get("fixed_league")
    half_life_days = ctx.get("half_life_days")
    inhib = ctx.get("inhib")
    kills = ctx.get("kills")
    league_fixed_value = ctx.get("league_fixed_value")
    league_mode = ctx.get("league_mode")
    map_mode = ctx.get("map_mode")
    map_mode_ui = ctx.get("map_mode_ui")
    max_games = ctx.get("max_games")
    ml_mode = ctx.get("ml_mode")
    n_sims = ctx.get("n_sims")
    pA = ctx.get("pA")
    pB = ctx.get("pB")
    p_map_cal = ctx.get("p_map_cal")
    p_map_raw = ctx.get("p_map_raw")
    p_series = ctx.get("p_series")
    parse_lines_text = ctx.get("parse_lines_text")
    parse_time_lines_text = ctx.get("parse_time_lines_text")
    playoffs_opt = ctx.get("playoffs_opt")
    prob_win_series = ctx.get("prob_win_series")
    season_label = ctx.get("season_label")
    split_opt = ctx.get("split_opt")
    teamA = ctx.get("teamA")
    teamB = ctx.get("teamB")
    team_games = ctx.get("team_games")
    tier_priority = ctx.get("tier_priority")
    time_m = ctx.get("time_m")
    totals_avg = ctx.get("totals_avg")
    totals_for_lines = ctx.get("totals_for_lines")
    totals_map = ctx.get("totals_map")
    totals_mode = ctx.get("totals_mode")
    is_geral = bool(ctx.get("is_geral"))
    totals_breakdown = ctx.get("totals_breakdown") or {}
    towers = ctx.get("towers")
    weight_mode = ctx.get("weight_mode")
    year_opt = ctx.get("year_opt")
    # --- extracted block body ---
    # Invalidate ALL post-layer transient cache when the analysis context
    # changes (matchup, filters, league mode, history window, modes).
    # Without this, a value computed under one window leaks into the next
    # render of a different window, producing inconsistent odds/score
    # (e.g., toggling "Ano todo" -> "Últimos 5" -> "Ano todo" returned
    # different odds because the post-layer read stale trace/cache from
    # session state).
    _ml_trace_ctx_fp = (
        str(season_label or ""), str(teamA or ""), str(teamB or ""), int(bo or 1),
        str(year_opt or ""), str(split_opt or ""), str(playoffs_opt or ""),
        str(league_mode or ""), str(fixed_league or ""),
        int(max_games) if max_games else 0,
        str(map_mode_ui or ""), str(totals_mode or ""), str(ml_mode or ""),
        str(weight_mode or ""), int(half_life_days) if half_life_days is not None else -1,
    )
    if st.session_state.get("_ml_trace_last_ctx_fp") != _ml_trace_ctx_fp:
        # Different context: clear all session-level pipeline state so each
        # downstream layer recomputes from scratch. Keep the keys list in
        # sync with _FLOW_RUNTIME_TRANSIENT_KEYS in flow_runtime_core.py.
        for _stale_key in (
            "_ml_trace_last", "_ml_trace_combined",
            "_ml_trace_markets", "_ml_trace_players",
            "_mlcore_cache_key", "_mlcore_cache_val",
            "_ml_microseg_info", "_ml_fused_post_cal_info",
            "_pre_delivery_report",
            "_p_map_used", "_p_series_used", "_p_ml_scope",
            "_ml_diff_used", "_ml_delta_players", "_ml_delta_players_eff",
            "_ml_team_games_min", "_ml_consistency_prev",
            "wf_ml_corr_info",
        ):
            st.session_state.pop(_stale_key, None)
        st.session_state["_ml_trace_last_ctx_fp"] = _ml_trace_ctx_fp

    # Estrutura da Visão Geral (reforma 2026-06, decidida pelo usuário):
    # Métricas (Confiança/Score/Base) → Confronto → Odds, com o diagnóstico
    # técnico restante (explicabilidade, avisos de pipeline) recolhido num
    # expander único no fim. Containers desacoplam ordem de exibição da ordem
    # do código — nenhuma lógica/cálculo muda de lugar, só o destino do render.
    _box_metricas = st.container()
    _box_confronto = st.container()
    _box_contexto = st.container()
    _box_detalhes = st.expander("Detalhes do ML (explicabilidade e avisos do pipeline)", expanded=False)

    # Card de odds único: exibimos apenas o painel principal abaixo,
    # já com todas as calibrações/fusões aplicadas.
    _msi_caption = core_build_microseg_caption(core_get_ml_microseg_info(st.session_state))
    if str(_msi_caption).strip():
        _box_contexto.caption(str(_msi_caption))

    _lg_hint = core_resolve_shared_league_hint(
        ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
        league_a=locals().get("_lgA", ""),
        league_b=locals().get("_lgB", ""),
    )
    # Como o modelo se sai NESTE torneio (previsibilidade + honestidade do %).
    # No Dota isso pesa mais que no LoL: os piores torneios batem ECE 0.19/0.25,
    # ou seja, o % da tela nao e preco ali.
    try:
        from pathlib import Path as _P

        from app_core.nota_liga_core import render_nota_liga_badge as _nl_badge

        _nl_badge(st=st, root=_P(__file__).resolve().parents[1],
                  league=str(_lg_hint or ""), box=_box_contexto)
    except (ImportError, OSError, TypeError, ValueError):
        pass

    _conf_baseline = {}
    try:
        if _ML_CONFIDENCE_BASELINE_PATH.exists():
            _conf_baseline = _load_ml_confidence_baseline_cached(_file_sig(str(_ML_CONFIDENCE_BASELINE_PATH))) or {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        _conf_baseline = {}
    _trace_state = core_get_ml_trace_snapshot(st.session_state)
    _conf = _ml_confidence_from_layer_validation(
        core_get_lv_last_report(st.session_state),
        league_hint=_lg_hint,
        n_team_eff=float(st.session_state.get("_ml_team_games_min", float("nan"))),
        ml_trace=_trace_state.get("_ml_trace_last"),
        confidence_baseline=_conf_baseline,
    )
    _conf_summary = core_build_confidence_summary(_conf if isinstance(_conf, dict) else {})

    # CONFIANCA ML — refeita em 15/09/2026, depois de ele perguntar por que o
    # cartao mostrava "C (Baixa) 35/100" TODO DIA. Mostrava porque a conta era
    # 50 de base menos 15 de "nao tenho o relatorio de validacao": o relatorio
    # era procurado so na memoria da sessao, que zera a cada reinicio, e o
    # arquivo em disco era de 04/03/2026 com uma liga so.
    # Agora a fonte e a medicao por liga (33 ligas, mesma que alimenta a linha
    # abaixo do confronto), mais o ajuste deste confronto que ja era calculado
    # aqui. Sem medicao utilizavel o cartao mostra "—", nunca um numero.
    # Ver app_core/confianca_ml_core.py.
    try:
        from pathlib import Path as _P2

        from app_core.confianca_ml_core import montar_confianca as _montar_conf
        from app_core.nota_liga_core import (
            _idade_em_dias as _idade_medicao,
            load_notas as _load_notas_liga,
            nota_for_league as _nota_da_liga,
        )
        # Mesma raiz que a linha da liga usa logo acima (linha ~270), para as
        # duas lerem exatamente o mesmo arquivo de medicao.
        _raiz_notas = _P2(__file__).resolve().parents[1]
        # CONFRONTO ENTRE LIGAS (consertado em 15/09/2026, relatado por ele).
        # `_lg_hint` so vem preenchido quando as DUAS ligas sao a mesma; num
        # PRM x TCL ele vem vazio e o cartao apagava — justo onde confianca
        # mais importa, porque ligar duas escalas de rating e a conta mais
        # dificil que o modelo faz. Agora olha as duas e fica com a PIOR.
        from app_core.confianca_ml_core import escolher_entre_ligas as _entre_ligas
        if str(_lg_hint or "").strip():
            _linha_conf = _nota_da_liga(_raiz_notas, _lg_hint)
            _entre = False
            _liga_txt = str(_lg_hint)
        else:
            # NAO chamar esta variavel de `_esc`: esse nome ja e a funcao de
            # escapar HTML, importada no topo do arquivo. Eu a atropelei com um
            # dicionario em 15/09/2026 e a Visao Geral inteira caiu com
            # "'dict' object is not callable" — 300 linhas abaixo, onde o
            # `_esc` de verdade era usado.
            _escolha_liga = _entre_ligas(
                _nota_da_liga(_raiz_notas, str(locals().get("_lgA", "") or "")),
                _nota_da_liga(_raiz_notas, str(locals().get("_lgB", "") or "")))
            _linha_conf = _escolha_liga["linha"]
            _entre = bool(_escolha_liga["entre_ligas"])
            _liga_txt = "+".join([x for x in _escolha_liga["ligas"] if x]) or ""

        _conf_card = _montar_conf(
            linha_liga=_linha_conf,
            idade_dias=_idade_medicao(_load_notas_liga(_raiz_notas).get("gerado_em")),
            liga=_liga_txt,
            entre_ligas=_entre,
            ajuste_do_confronto=float((_conf or {}).get("pipeline_adj", 0.0) or 0.0),
            resumo_do_confronto=str((_conf or {}).get("pipeline_summary", "") or ""),
            n_jogos_dos_times=float(st.session_state.get("_ml_team_games_min", float("nan"))),
        )
    except Exception as _e_conf:
        _conf_card = {"disponivel": False, "grade": "—", "score_text": "—",
                      "source": "nao medida", "motivo": str(_e_conf)[:120],
                      "porque": ""}

    with _box_metricas:
        ccf1, ccf2, ccf3 = st.columns(3)
        with ccf1:
            st.metric("Confianca ML", value=str(_conf_card.get("grade", "—")))
        with ccf2:
            st.metric("Score", value=str(_conf_card.get("score_text", "—")))
        with ccf3:
            st.metric("Base", value=str(_conf_card.get("source", "nao medida")))
        # O cartao explica a si mesmo. Nota sem motivo vira superstricao.
        if _conf_card.get("disponivel"):
            _idade = _conf_card.get("idade_dias")
            _sufixo = (" · medida ha {:.0f} dia(s)".format(_idade)
                       if isinstance(_idade, (int, float)) else "")
            if str(_conf_card.get("porque") or "").strip():
                st.caption(str(_conf_card["porque"]) + _sufixo)
        else:
            st.caption("Confianca nao medida: {}. {}".format(
                _conf_card.get("motivo", ""), _conf_card.get("porque", "")))
    # Diagnósticos técnicos (Semáforo / Baseline / Camadas A/B/C / dica de
    # manutenção) removidos a pedido do usuário — ruído para o uso diário.
    # Os 3 cards acima voltaram ao topo a pedido do usuário (2026-06-10).

    # Explicabilidade curta por confronto (camadas que mais puxaram o preço)
    _trace_last = (_trace_state.get("_ml_trace_last") or {}) if isinstance(_trace_state, dict) else {}
    _mt_last = _trace_last.get("ml_trace", {}) if isinstance(_trace_last, dict) else {}
    _step_rows = core_build_pipeline_step_rows(_mt_last if isinstance(_mt_last, dict) else {})
    if _step_rows:
        _explain = core_build_pipeline_explainability(_step_rows, _conf if isinstance(_conf, dict) else {})
        _top_rows = list(_explain.get("rows") or []) if isinstance(_explain, dict) else []
        if _top_rows:
            with _box_detalhes:
                st.markdown("**Explicabilidade do ML (resumo do confronto)**")
                if str(_explain.get("moves_line", "")).strip():
                    st.caption(str(_explain.get("moves_line", "")))
                if str(_explain.get("why_line", "")).strip():
                    st.caption(str(_explain.get("why_line", "")))
                _top = pd.DataFrame(_top_rows)
                if not _top.empty:
                    st.dataframe(
                        _df_contract_view(
                            _top,
                            ["Etapa", "Aplicou", "Antes %", "Depois %", "Delta (pp)", "Fonte"],
                            context="Visão Geral > Explicabilidade ML",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

    # -----------------------------
    # Lineup (Players) — ajuste opcional do ML
    # -----------------------------
    _lk = _lineup_key(season_label, teamA, teamB)
    _lineup_rt = _ensure_lineup_runtime_state()
    if _get_lineup_loaded_key() != _lk:
        _load_lineup_for_matchup(_lk)
        _set_lineup_loaded_key(_lk)
        _lineup_rt = _get_lineup_runtime_settings()

    # defaults para p usado (sem ajuste)
    ml_engine = st.session_state.get("ml_engine", "mlcore_v2")

    p_map_used = float(p_map_cal) if math.isfinite(float(p_map_cal)) else float("nan")
    p_series_used = float(p_series) if math.isfinite(float(p_series)) else float("nan")
    diff_used = float(_ml_info.get('diff_points', float('nan'))) if _ml_info else float('nan')
    delta_players = 0.0

    # Mantém ajuste de lineup estritamente opt-in via checkbox para evitar
    # mudanças de odd em reruns (ex.: editar/salvar linhas sem alterar inputs de ML).

    with _box_detalhes:
        # UI de lineup removida da Visão Geral (2026-06-10, pedido do usuário): os
        # widgets (checkbox/sliders/CSV/selects) não eram usados — o ajuste manual era
        # opt-in e ficava desligado. O que IMPORTA continua: o autofill preenche a
        # lineup do último jogo nas chaves de sessão e o players layer do core
        # (w_lineup fitado no artefato) consome essas lineups no motor. Remover o
        # autofill derrubaria a coverage do players layer (parte da nota da fusão).
        if 'use_lineup_adjust' not in st.session_state:
            st.session_state['use_lineup_adjust'] = bool(_lineup_rt.get('use_lineup_adjust', False))
        if 'players_csv_path' not in st.session_state:
            st.session_state['players_csv_path'] = str(_lineup_rt.get('players_csv_path', 'data/oracleselixir_players.csv') or 'data/oracleselixir_players.csv')
        use_lineup = bool(st.session_state.get('use_lineup_adjust', False))

        df_pr = None
        try:
            df_pr = _load_player_rows(csv_sig)
        except (OSError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            df_pr = None

        core_autofill_lineup_state(
            session_state=st.session_state,
            lineup_key=_lk,
            do_auto=False,
            df_pr=df_pr,
            team_a=teamA,
            team_b=teamB,
            filters=filters,
            league_fixed_value=league_fixed_value,
            pos_canon=list(_POS_CANON),
            pos_canon_ml=list(_POS_CANON_ML),
            safe_key_fn=_safe_key,
            latest_lineup_for_team_fn=_latest_lineup_for_team,
            save_lineup_for_matchup_fn=_save_lineup_for_matchup,
        )

        _pl_cfg = core_resolve_players_layer_config(
            artifact_path=str(artifact_path),
            session_state=st.session_state,
            load_artifact_cached_fn=_load_artifact_cached,
            artifact_sig_fn=_artifact_sig,
            ui_weight_default=1.15,
        )
        pl_enabled = bool(_pl_cfg.get("enabled"))
        pl_players_path = str(_pl_cfg.get("players_path") or "")
        pl_w = float(_pl_cfg.get("weight", 1.15) or 1.15)
        pl_w_src = str(_pl_cfg.get("weight_source") or "ui")

        # Sem o radio: comportamento igual ao default antigo (index=0).
        use_ml_lineup = bool(pl_enabled)
        lineup_mode = "ML players layer" if use_ml_lineup else "CSV byTournament (manual)"

        df_imp = pd.DataFrame(columns=['team','pos','player','gp','score'])
        try:
            psig = _file_sig(str(_lineup_rt.get('players_csv_path','data/oracleselixir_players.csv') or 'data/oracleselixir_players.csv'))
            df_imp = _load_players_impact(psig)
            df_imp = _apply_shrink_m(df_imp, int(_lineup_rt.get('lineup_shrink_m', 10) or 10))
        except (OSError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            df_imp = pd.DataFrame(columns=['team','pos','player','gp','score'])
        teams_imp = sorted(df_imp['team'].dropna().astype(str).unique().tolist()) if df_imp is not None and not df_imp.empty else []
        teamA_key = _best_match(teamA, teams_imp) if teams_imp else ''
        teamB_key = _best_match(teamB, teams_imp) if teams_imp else ''

        # ------------------------------------------------------------------
        # AJUSTAR ESCALACAO — devolvido em 11/09/2026, a pedido do dono.
        #
        # HISTORIA, para ninguem remover de novo sem saber: esta UI foi tirada
        # da Visao Geral em 10/06/2026, tambem a pedido dele, porque o ajuste
        # manual era opt-in, ficava desligado e so ocupava espaco. O que ficou
        # foi o preenchimento automatico com a ultima escalacao conhecida.
        #
        # O QUE MUDOU PARA JUSTIFICAR A VOLTA: em 11/09 a camada de jogadores
        # passou a entrar na probabilidade servida (estagio players_pscore).
        # Antes disso, mexer na escalacao nao movia odd nenhuma — por isso a
        # UI era inutil. Agora move, e esta medido quanto:
        #
        #   escalacao da ultima partida (o automatico) ... +0,00402
        #   escalacao CORRIGIDA a mao .................... +0,00577
        #   -----------------------------------------------------------
        #   valor de corrigir ............................ +0,00175
        #                                   IC [+0,00119, +0,00231]
        #
        # Fechado por padrao, de proposito: fechado, a tela fica igual a de
        # hoje. So abre quando o dono sabe de uma troca.
        # ------------------------------------------------------------------
        # BLINDAGEM (11/09/2026, com o site fora do ar): este bloco desenha
        # widgets, e widget que levanta excecao derruba a PAGINA INTEIRA, nao
        # so o bloco. Um ajuste de escalacao nunca pode custar o app.
        #
        # Nao ficou provado que foi ele que derrubou o site — as medicoes
        # apontaram para outro lado (0,44 s e 1,4 MB por tela, nenhuma
        # excecao no fluxo local). Mas "nao consegui reproduzir" nao e o
        # mesmo que "nao pode acontecer", e a nuvem tem dados e versoes que
        # a maquina daqui nao tem. Entao ele passa a falhar sozinho, calado
        # e visivel, em vez de levar o resto junto.
        try:
          with st.expander("Ajustar escalação", expanded=False):
              st.caption(
                  "Preenchido com a última escalação conhecida de cada time. "
                  "Corrija só quando souber de uma troca — o modelo usa estes "
                  "cinco de cada lado para calcular o efeito dos jogadores."
              )
              if df_pr is None or getattr(df_pr, "empty", True):
                  st.info("Sem dados de jogadores carregados para editar a escalação.")
              else:
                  _cols_ok = {"team", "pos_ml", "player_key"}.issubset(set(df_pr.columns))
                  if not _cols_ok:
                      st.info("A base de jogadores não tem as colunas necessárias.")
                  else:
                      def _opcoes_do_time(nome_time: str, pos: str) -> list:
                          """Quem pode entrar naquela rota, do mais provavel ao menos.

                          POR QUE NAO SO O TIME (conserto de 11/09/2026, no mesmo
                          dia): a primeira versao oferecia so quem ja jogou por
                          aquele time naquela rota. Conferido dirigindo o app: dava
                          DUAS opcoes — o titular e o vazio. Ou seja, era impossivel
                          colocar um substituto, que e exatamente o caso para o
                          qual esta tela existe.

                          Agora a lista vai em tres camadas, nesta ordem:
                            1. quem ja jogou por ESTE time nesta rota (o provavel)
                            2. quem joga a MESMA rota na mesma liga (o emprestimo,
                               o promovido da academia, o reserva)
                            3. o resto da rota, para o caso raro
                          O campo e pesquisavel, entao lista longa nao atrapalha.
                          """
                          try:
                              d_rota = df_pr[df_pr["pos_ml"].astype(str) == str(pos)]
                              if "date" in d_rota.columns:
                                  d_rota = d_rota.sort_values("date", ascending=False)

                              do_time = d_rota[d_rota["team"].astype(str) == str(nome_time)]

                              da_liga = d_rota.iloc[0:0]
                              if "league" in d_rota.columns:
                                  _ligas = set(
                                      do_time["league"].astype(str).unique().tolist()
                                  ) if not do_time.empty else set()
                                  if _ligas:
                                      da_liga = d_rota[d_rota["league"].astype(str).isin(_ligas)]

                              vistos, saida = set(), []
                              for _parte in (do_time, da_liga, d_rota):
                                  for k in _parte["player_key"].astype(str).tolist():
                                      if k and k not in vistos:
                                          vistos.add(k)
                                          saida.append(k)
                              return saida
                          except (KeyError, TypeError, ValueError):
                              return []

                      _nome_por_chave = {}
                      try:
                          if "player" in df_pr.columns:
                              for _k, _n in zip(df_pr["player_key"].astype(str),
                                                df_pr["player"].astype(str)):
                                  if _k and _k not in _nome_por_chave:
                                      _nome_por_chave[_k] = _n
                      except (KeyError, TypeError, ValueError):
                          _nome_por_chave = {}

                      def _mostrar(chave: str) -> str:
                          if not chave:
                              return "— ninguém —"
                          return _nome_por_chave.get(str(chave), str(chave))

                      _mudou = False
                      for _lado, _nome_time in (("A", teamA), ("B", teamB)):
                          st.markdown(f"**{_nome_time}**")
                          _cs = st.columns(len(_POS_CANON_ML))
                          for _i, _pos in enumerate(_POS_CANON_ML):
                              _k = f"lineup_ml_{_lado}_{_safe_key(_pos)}"
                              _atual = str(st.session_state.get(_k, "") or "")
                              _opts = _opcoes_do_time(_nome_time, _pos)
                              # O valor atual TEM de estar nas opcoes, senao o
                              # widget recusa. Jogador emprestado de outro time,
                              # ou recem-chegado, cai neste caso.
                              if _atual and _atual not in _opts:
                                  _opts = [_atual] + _opts
                              _opts = [""] + _opts
                              with _cs[_i]:
                                  _novo = st.selectbox(
                                      _pos, options=_opts,
                                      index=(_opts.index(_atual) if _atual in _opts else 0),
                                      format_func=_mostrar,
                                      key=f"ui_{_k}",
                                  )
                              if str(_novo or "") != _atual:
                                  st.session_state[_k] = str(_novo or "")
                                  _mudou = True

                      if _mudou:
                          # Guarda POR CONFRONTO: corrigiu Time A x Time B, fica
                          # salvo so para esse confronto.
                          _save_lineup_for_matchup(_lk)

                      if st.button("Voltar à escalação automática", key="lineup_reset_btn"):
                          for _lado in ("A", "B"):
                              for _pos in _POS_CANON_ML:
                                  st.session_state.pop(f"ui_lineup_ml_{_lado}_{_safe_key(_pos)}", None)
                          st.session_state.pop("_lineup_autofilled_key", None)
                          _rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                          if callable(_rerun):
                              _rerun()
        except Exception as _e_esc:  # noqa: BLE001 - a tela vale mais que o bloco
            try:
                st.warning(
                    "Não consegui montar o ajuste de escalação ({}). O resto da "
                    "análise segue normal, com a última escalação conhecida."
                    .format(type(_e_esc).__name__)
                )
            except Exception:
                pass

        lineupA_sel = _get_lineup_selected('A', mode='ml' if use_ml_lineup else 'manual')
        lineupB_sel = _get_lineup_selected('B', mode='ml' if use_ml_lineup else 'manual')

        # defaults
        delta_players = 0.0
        delta_players_eff = 0.0
        diff_used = float(_ml_info.get('diff_points', float('nan'))) if _ml_info is not None else float('nan')
        p_map_used = float(p_map_cal)
        p_series_used = float(p_series)
        trace_p_map_team = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        trace_p_map_players = float("nan")
        trace_p_map_fused = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        trace_n_team_eff = float("nan")
        trace_n_players_eff = float("nan")
        trace_fusion_info: Dict[str, Any] = {}
        trace_form_info: Dict[str, Any] = {}
        trace_hybrid_info: Dict[str, Any] = {}
        trace_conf_guard_info: Dict[str, Any] = {}
        trace_patch_guard_info: Dict[str, Any] = {}
        trace_series_gamma_info: Dict[str, Any] = {}
        trace_uncertainty_info: Dict[str, Any] = {}
        trace_marginal_trust_info: Dict[str, Any] = {}
        st.session_state.pop("_ml_uncertainty_info", None)
        trace_pipeline_steps: List[Dict[str, Any]] = []
        trace_pipeline_order: List[str] = []
        _p_pipe_prev = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")

        _runtime_league_hint = core_resolve_shared_league_hint(
            ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
            fit_league_hint=locals().get("_fit_league_hint", ""),
            match_league=locals().get("match_league", ""),
            league_a=locals().get("_lgA", ""),
            league_b=locals().get("_lgB", ""),
        )
        _runtime_conf_baseline = _load_confidence_baseline_runtime()
        _runtime_tuning = core_apply_runtime_tuning_to_session(
            session_state=st.session_state,
            league_hint=_runtime_league_hint,
            bo=int(bo),
            confidence_baseline=_runtime_conf_baseline,
            confidence_governor_budget_fn=_runtime_confidence_governor_budget_for_league,
            series_coherence_post_cap_fn=_runtime_series_coherence_post_cap_for_league,
            governor_default_pp=float(st.session_state.get("ml_confidence_governor_max_total_shrink_pp", 8.0) or 8.0),
        )
        _runtime_series_post_cap = _runtime_tuning.get("series_post_cap")

        _pre_engine_callbacks = {
            "league_is_excluded_fn": core_league_is_excluded,
            "apply_ml_uncertainty_adjust_fn": _apply_ml_uncertainty_adjust,
            "prob_win_series_fn": prob_win_series,
            "load_players_artifact_cached_fn": _load_players_artifact_cached,
            "file_sig_fn": _file_sig,
            "fit_players_lane_for_league_fn": _fit_players_lane_for_league,
            "fit_players_league_weight_for_league_fn": _fit_players_league_weight_for_league,
            "compute_delta_and_confidence_fn": compute_delta_and_confidence,
            "delta_players_effect_fn": _delta_players_effect,
            "transfer_signal_from_details_fn": lambda details: core_transfer_signal_from_details(
                details,
                tier_priority_fn=tier_priority,
            ),
            "lineup_continuity_from_details_fn": lambda details, freshness_avg, cov_ratio: core_lineup_continuity_from_details(
                details,
                freshness_avg=freshness_avg,
                cov_ratio=cov_ratio,
            ),
            "calibrate_p_from_artifact_fn": _calibrate_p_from_artifact,
            "fuse_probs_by_precision_fn": _fuse_probs_by_precision,
            "score_team_fn": lambda team_key, lineup: core_score_manual_lineup_team(
                team_key=team_key,
                lineup=lineup,
                df_imp=df_imp,
                pos_canon=list(_POS_CANON),
                min_gp=int(_lineup_rt.get('lineup_min_gp', 5) or 5),
                best_match_fn=_best_match,
            ),
            "p_from_diff_points_fn": _p_from_diff_points,
            "elo_season_players_p_map_fn": _elo_season_players_p_map,
            "apply_confidence_governor_fn": core_apply_confidence_governor,
        }
        # Calibrador com grupo (05/09/2026, divida D8). Le o bloco do
        # params_fitted.json; SEM o bloco fica None e o app se comporta
        # exatamente como antes. Apagar a chave do arquivo DESLIGA.
        try:
            _grupo_coef = ((_load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {})
                           .get("postfusion_group_calibration") or None)
        except Exception:
            _grupo_coef = None
        _pre_engine_in = core_build_pre_engine_input(
            p_map_used=p_map_used,
            p_series_used=p_series_used,
            bo=int(bo),
            ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
            lgA=str(locals().get("_lgA", "") or ""),
            lineup_mode=str(locals().get("lineup_mode", "")),
            pl_enabled=bool(pl_enabled),
            pl_players_path=pl_players_path,
            lineupA_sel=(lineupA_sel if isinstance(lineupA_sel, dict) else {}),
            lineupB_sel=(lineupB_sel if isinstance(lineupB_sel, dict) else {}),
            as_of_date=as_of_date,
            ml_engine=str(ml_engine),
            use_lineup=bool(use_lineup),
            artifact_path=artifact_path,
            pl_w=pl_w,
            playoffs_opt=playoffs_opt,
            teamA=str(teamA),
            teamB=str(teamB),
            team_aliases_a=_identity_team_name_candidates(str(teamA)),
            team_aliases_b=_identity_team_name_candidates(str(teamB)),
            teamA_key=str(teamA_key),
            teamB_key=str(teamB_key),
            team_games=team_games,
            csv_path=csv_path,
            filters=filters,
            diff_used=diff_used,
            delta_players=delta_players,
            delta_players_eff=delta_players_eff,
            trace_p_map_team=trace_p_map_team,
            trace_p_map_players=trace_p_map_players,
            trace_p_map_fused=trace_p_map_fused,
            trace_n_team_eff=trace_n_team_eff,
            trace_n_players_eff=trace_n_players_eff,
            trace_fusion_info=trace_fusion_info,
            trace_uncertainty_info=trace_uncertainty_info,
            league_transfer_features=(
                _load_league_transfer_features_cached(_file_sig(str(_LEAGUE_TRANSFER_FEATURES_PATH)))
                if _LEAGUE_TRANSFER_FEATURES_PATH.exists()
                else {}
            ),
            p_pipe_prev=_p_pipe_prev,
            session_state=st.session_state,
            trace_pipeline_steps=trace_pipeline_steps,
            callbacks=_pre_engine_callbacks,
            format_num_fn=_format_num,
            format_pct_fn=_format_pct,
            caption_fn=_vg_caption(_box_detalhes),
            warning_fn=_box_detalhes.warning,
            match_league=str(locals().get("match_league", "") or ""),
            postfusion_grupo_coef=_grupo_coef,
        )
        _pre_live_callbacks = {
            "season_form_blend_fn": core_season_form_blend,
            "current_season_override_fn": core_current_season_override,
            "hybrid_momentum_blend_fn": core_hybrid_momentum_blend,
            "clip_prob_fn": clip_prob,
            "ml_consistency_signature_fn": _ml_consistency_signature,
            "feature_flag_enabled_fn": _feature_flag_enabled,
            "apply_series_bo_dplite_live_fn": core_apply_series_bo_dplite_live,
            "enforce_series_economic_coherence_fn": core_enforce_series_economic_coherence,
            "series_gamma_for_league_fn": _series_gamma_for_league,
            "apply_series_gamma_fn": _apply_series_gamma,
            "patch_meta_guard_payload_fn": _patch_meta_guard_payload,
            "patch_changed_leagues_set_fn": _patch_changed_leagues_set,
            "file_sig_fn": _file_sig,
            "prob_win_series_fn": prob_win_series,
            "fit_form_for_league_fn": _fit_form_for_league,
            "apply_confidence_governor_fn": core_apply_confidence_governor,
        }
        _pre_live_in = core_build_pre_live_input(
            p_map_used=p_map_used,
            p_series_used=p_series_used,
            bo=int(bo),
            season_label=str(season_label),
            teamA=str(teamA),
            teamB=str(teamB),
            ml_engine=str(ml_engine),
            year_opt=year_opt,
            split_opt=split_opt,
            playoffs_opt=playoffs_opt,
            league_mode=str(league_mode),
            fixed_league=fixed_league,
            as_of_date=as_of_date,
            artifact_path=str(artifact_path),
            lineup_mode=str(locals().get("lineup_mode", "")),
            lineupA_sel=(lineupA_sel if isinstance(lineupA_sel, dict) else {}),
            lineupB_sel=(lineupB_sel if isinstance(lineupB_sel, dict) else {}),
            diff_used=diff_used,
            delta_players=delta_players,
            trace_n_team_eff=trace_n_team_eff,
            trace_n_players_eff=trace_n_players_eff,
            trace_patch_guard_info=trace_patch_guard_info,
            trace_style_b_lite_live=(
                locals().get("trace_style_b_lite_live")
                if isinstance(locals().get("trace_style_b_lite_live"), dict)
                else {}
            ),
            ml_conf_score_snapshot=locals().get("_ml_conf_score_snapshot", None),
            lg_a_shadow=str(locals().get("_lg_a_shadow", "") or ""),
            lg_b_shadow=str(locals().get("_lg_b_shadow", "") or ""),
            csv_path=csv_path,
            fit_params_path=_FIT_PARAMS_PATH,
            filters=filters,
            dfA_used=dfA_used,
            dfB_used=dfB_used,
            session_state=st.session_state,
            trace_pipeline_steps=trace_pipeline_steps,
            callbacks=_pre_live_callbacks,
            caption_fn=_vg_caption(_box_detalhes),
        )
        _pre_pipeline_out = core_run_ml_pre_pipeline(
            pre_engine=_pre_engine_in,
            pre_live=_pre_live_in,
        )
        _pre_model = (
            _pre_pipeline_out
            if isinstance(_pre_pipeline_out, CoreMlPrePipelineOutput)
            else CoreMlPrePipelineOutput.from_mapping(
                _pre_pipeline_out if isinstance(_pre_pipeline_out, dict) else {}
            )
        )
        p_map_used = float(_pre_model.p_map_used)
        p_series_used = float(_pre_model.p_series_used)
        diff_used = float(_pre_model.diff_used) if math.isfinite(float(_pre_model.diff_used)) else float(diff_used)
        delta_players = float(_pre_model.delta_players)
        delta_players_eff = float(_pre_model.delta_players_eff)
        trace_p_map_team = float(_pre_model.trace_p_map_team)
        trace_p_map_players = float(_pre_model.trace_p_map_players)
        trace_p_map_fused = float(_pre_model.trace_p_map_fused)
        trace_n_team_eff = float(_pre_model.trace_n_team_eff)
        trace_n_players_eff = float(_pre_model.trace_n_players_eff)
        trace_fusion_info = dict(_pre_model.trace_fusion_info)
        trace_uncertainty_info = dict(_pre_model.trace_uncertainty_info)
        _delta_mode = str(
            _pre_model._delta_mode
            or str(st.session_state.get("lineup_delta_mode", "saturado") or "saturado")
        )
        _delta_cap = (
            float(_pre_model._delta_cap)
            if math.isfinite(float(_pre_model._delta_cap))
            else float(st.session_state.get("lineup_delta_cap", 1.6) or 1.6)
        )
        _delta_slope = (
            float(_pre_model._delta_slope)
            if math.isfinite(float(_pre_model._delta_slope))
            else float(st.session_state.get("lineup_delta_slope", 2.4) or 2.4)
        )
        lineup_mode = str(_pre_model.lineup_mode or str(locals().get("lineup_mode", "")))
        match_league = str(_pre_model.match_league or str(locals().get("match_league", "") or ""))
        _elo_info = _pre_model._elo_info
        _p_pipe_prev = (
            float(_pre_model._p_pipe_prev)
            if math.isfinite(float(_pre_model._p_pipe_prev))
            else float(_p_pipe_prev)
        )
        trace_form_info = dict(_pre_model.trace_form_info)
        trace_hybrid_info = dict(_pre_model.trace_hybrid_info)
        trace_conf_guard_info = dict(_pre_model.trace_conf_guard_info)
        trace_patch_guard_info = dict(_pre_model.trace_patch_guard_info)
        trace_series_gamma_info = dict(_pre_model.trace_series_gamma_info)
        trace_series_coherence_info = dict(_pre_model.trace_series_coherence_info)
        _guard_triggered = bool(_pre_model._guard_triggered)
        _guard_dpp = float(_pre_model._guard_dpp)
        _ff_ml_series_bo_dplite_live = bool(_pre_model._ff_ml_series_bo_dplite_live)
        trace_series_bo_dplite_live = dict(_pre_model.trace_series_bo_dplite_live)
        _lineup_mode_sig = str(_pre_model._lineup_mode_sig or str(locals().get("lineup_mode", "")))
        _lineupA_sig = dict(_pre_model._lineupA_sig)
        _lineupB_sig = dict(_pre_model._lineupB_sig)
        trace_pipeline_steps = list(_pre_model.trace_pipeline_steps)

# FORMA RECENTE (16/11/2026, aprovada por ele): o modelo reage devagar a fase boa/ruim (WE no
    # LoL; Synapse x Spirit Academy no Dota). Entra NO FIM, no p que vira preco (as misturas do meio
    # do caminho diluiam o ajuste: 67,5% virava 59,5%). Ver app_core/forma_recente_core.py.
    try:
        from pathlib import Path as _P
        from app_core.forma_recente_core import aplicar_no_confronto as _forma_aplicar
        p_map_used, p_series_used = _forma_aplicar(
            st, _P(__file__).resolve().parents[1], str(teamA or ""), str(teamB or ""),
            float(p_map_used), float(p_series_used), int(bo or 1), prob_win_series)
    except Exception as _e_forma:
        st.caption("Forma recente indisponível ({}); preço sem esse ajuste.".format(type(_e_forma).__name__))

# persistência para as outras abas
    _team_games_min_used = core_persist_runtime_analysis_state(
        session_state=st.session_state,
        p_map_used=p_map_used,
        p_series_used=p_series_used,
        diff_used=diff_used,
        delta_players=delta_players,
        delta_players_eff=delta_players_eff,
        ml_info=(_ml_info if isinstance(_ml_info, dict) and not _ml_info.get("__error__") else {}),
    )
    _input_signature_obj = core_build_runtime_input_signature(
        session_state=st.session_state,
        season_label=season_label,
        teamA=teamA,
        teamB=teamB,
        bo=int(bo),
        year_opt=year_opt,
        split_opt=split_opt,
        playoffs_opt=playoffs_opt,
        league_mode=league_mode,
        fixed_league=fixed_league,
        weight_mode=weight_mode,
        half_life_days=half_life_days,
        ml_engine=ml_engine,
        totals_mode=totals_mode,
        ml_mode=ml_mode,
        map_mode_ui=map_mode_ui,
        n_sims=n_sims,
        lineup_mode=locals().get("lineup_mode", ""),
        lineup_sig_a=locals().get("_lineupA_sig", ""),
        lineup_sig_b=locals().get("_lineupB_sig", ""),
        csv_path=csv_path,
        artifact_path=artifact_path,
        fit_params_path=_FIT_PARAMS_PATH,
        ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
        file_sig_fn=_file_sig,
    )
    _input_signature_payload = dict(_input_signature_obj.get("payload") or {})
    _input_signature_hash = str(_input_signature_obj.get("sha256") or "")
    _wf_info = core_get_wf_ml_corr_info(st.session_state)
    _shadow_out = core_run_ml_shadow_layers(
        p_map_used=p_map_used,
        p_series_used=p_series_used,
        bo=int(bo),
        ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
        trace_uncertainty_info=(trace_uncertainty_info if isinstance(trace_uncertainty_info, dict) else {}),
        trace_n_team_eff=trace_n_team_eff,
        trace_n_players_eff=trace_n_players_eff,
        team_games_min_used=_team_games_min_used,
        trace_patch_guard_info=(trace_patch_guard_info if isinstance(trace_patch_guard_info, dict) else {}),
        trace_hybrid_info=(trace_hybrid_info if isinstance(trace_hybrid_info, dict) else {}),
        trace_form_info=(trace_form_info if isinstance(trace_form_info, dict) else {}),
        trace_fusion_info=(trace_fusion_info if isinstance(trace_fusion_info, dict) else {}),
        guard_triggered=bool(_guard_triggered),
        guard_dpp=_guard_dpp,
        teamA=str(teamA),
        teamB=str(teamB),
        session_state=st.session_state,
        team_state_artifact_path=_TEAM_STATE_ARTIFACT_PATH,
        safe_logit_fn=_safe_logit,
        safe_sigmoid_fn=_safe_sigmoid,
        clip_prob_fn=clip_prob,
        prob_win_series_fn=prob_win_series,
        odd_from_prob_fn=_odd_from_p,
        feature_flag_enabled_fn=_feature_flag_enabled,
        compute_state_shadow_alite_fn=core_compute_state_shadow_alite,
        compute_state_shadow_areal_fn=core_compute_state_shadow_areal,
        load_team_state_artifact_cached_fn=_load_team_state_artifact_cached,
        file_sig_fn=_file_sig,
    )
    _shadow_state = core_unpack_shadow_output(_shadow_out)
    _ff_ml_shadow_vnext = bool(_shadow_state.get("_ff_ml_shadow_vnext", False))
    trace_shadow_ml_vnext = dict(_shadow_state.get("trace_shadow_ml_vnext") or {})
    _ff_ml_shadow_state_alite = bool(_shadow_state.get("_ff_ml_shadow_state_alite", False))
    trace_shadow_state_alite = dict(_shadow_state.get("trace_shadow_state_alite") or {})
    _ff_ml_shadow_state_areal = bool(_shadow_state.get("_ff_ml_shadow_state_areal", False))
    trace_shadow_state_areal = dict(_shadow_state.get("trace_shadow_state_areal") or {})
    _lg_a_shadow = str(_shadow_state.get("_lg_a_shadow", "") or "")
    _lg_b_shadow = str(_shadow_state.get("_lg_b_shadow", "") or "")
    _team_aliases_a = _identity_team_name_candidates(teamA)
    _team_aliases_b = _identity_team_name_candidates(teamB)
    _post_ml_out = core_run_ml_post_layers(
        p_map_used=p_map_used,
        p_series_used=p_series_used,
        bo=int(bo),
        trace_shadow_state_alite=(
            trace_shadow_state_alite if isinstance(trace_shadow_state_alite, dict) else {}
        ),
        trace_shadow_state_areal=(
            trace_shadow_state_areal if isinstance(trace_shadow_state_areal, dict) else {}
        ),
        feature_flag_enabled_fn=_feature_flag_enabled,
        prob_win_series_fn=prob_win_series,
        session_state=st.session_state,
        trace_pipeline_steps=trace_pipeline_steps,
        live_kwargs={
            "playoffs_opt": playoffs_opt,
            "ml_info": (_ml_info if isinstance(_ml_info, dict) else {}),
            "pA": (pA if "pA" in locals() else None),
            "pB": (pB if "pB" in locals() else None),
            "team_games": (team_games if "team_games" in locals() else None),
            "filters": (filters if "filters" in locals() else None),
            "league_mode": str(league_mode),
            "map_mode": str(map_mode),
            "half_life_days": half_life_days,
            "max_games": max_games,
            "teamA": str(teamA),
            "teamB": str(teamB),
            "team_aliases_a": list(_team_aliases_a),
            "team_aliases_b": list(_team_aliases_b),
            "year_opt": (year_opt if "year_opt" in locals() else None),
            "split_opt": (split_opt if "split_opt" in locals() else None),
            "trace_n_team_eff": trace_n_team_eff,
            "trace_n_players_eff": trace_n_players_eff,
            "trace_patch_guard_info": (
                trace_patch_guard_info if isinstance(trace_patch_guard_info, dict) else {}
            ),
            "trace_fusion_info": (trace_fusion_info if isinstance(trace_fusion_info, dict) else {}),
            "trace_series_bo_dplite_live": (
                trace_series_bo_dplite_live
                if isinstance(trace_series_bo_dplite_live, dict)
                else {}
            ),
            "trace_hybrid_info": (trace_hybrid_info if isinstance(trace_hybrid_info, dict) else {}),
            "trace_p_map_players": trace_p_map_players,
            "trace_p_map_team": trace_p_map_team,
            "fit_live": (
                locals().get("_fit_live", {})
                if isinstance(locals().get("_fit_live", {}), dict)
                else {}
            ),
            "conf_baseline": (
                locals().get("_conf_baseline", {})
                if isinstance(locals().get("_conf_baseline", {}), dict)
                else {}
            ),
            "league_transfer_features": (
                _load_league_transfer_features_cached(_file_sig(str(_LEAGUE_TRANSFER_FEATURES_PATH)))
                if _LEAGUE_TRANSFER_FEATURES_PATH.exists()
                else {}
            ),
            "match_league": match_league,
            "lg_a_shadow": str(_lg_a_shadow or ""),
            "lg_b_shadow": str(_lg_b_shadow or ""),
            "p_map_cal": p_map_cal,
            "guard_triggered": bool(_guard_triggered),
            "guard_dpp": (_guard_dpp if "_guard_dpp" in locals() else None),
            "fit_params_path": _FIT_PARAMS_PATH,
            "load_fitted_params_fn": lambda: (
                _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
            ),
            "apply_confidence_governor_fn": core_apply_confidence_governor,
        },
        marginal_kwargs={
            "ml_info": (_ml_info if isinstance(_ml_info, dict) else {}),
            "trace_n_team_eff": trace_n_team_eff,
            "lv_last_report": core_get_lv_last_report(st.session_state),
            "ml_trace_last": core_get_ml_trace_snapshot(st.session_state).get("_ml_trace_last"),
            "ml_confidence_fn": _ml_confidence_from_layer_validation,
            "marginal_adjust_fn": _apply_ml_marginal_trust_adjust,
            "apply_confidence_governor_fn": core_apply_confidence_governor,
        },
        confidence_baseline_path=_ML_CONFIDENCE_BASELINE_PATH,
        load_confidence_baseline_cached_fn=_load_ml_confidence_baseline_cached,
        file_sig_fn=_file_sig,
        lg_a_hint=str(locals().get("_lgA", "") or ""),
        caption_fn=_vg_caption(_box_detalhes),
        apply_confidence_governor_fn=core_apply_confidence_governor,
        enforce_series_economic_coherence_fn=core_enforce_series_economic_coherence,
        series_coherence_post_max_dev_pp=(
            float(_runtime_series_post_cap)
            if ("_runtime_series_post_cap" in locals() and _runtime_series_post_cap is not None)
            else None
        ),
    )
    _post_layer_state = core_unpack_post_layer_output(
        _post_ml_out,
        default_trace_pipeline_steps=trace_pipeline_steps,
    )
    p_map_used = float(_post_layer_state.get("p_map_used", float("nan")))
    p_series_used = float(_post_layer_state.get("p_series_used", float("nan")))
    _ff_ml_state_areal_live = bool(_post_layer_state.get("_ff_ml_state_areal_live", False))
    _ff_ml_state_alite_live = bool(_post_layer_state.get("_ff_ml_state_alite_live", False))
    _ff_ml_style_b_lite_live = bool(_post_layer_state.get("_ff_ml_style_b_lite_live", False))
    _ff_ml_league_rank_sos_live = bool(_post_layer_state.get("_ff_ml_league_rank_sos_live", False))
    _ff_ml_roster_continuity_live = bool(_post_layer_state.get("_ff_ml_roster_continuity_live", False))
    _ff_ml_hier_calibration_live = bool(_post_layer_state.get("_ff_ml_hier_calibration_live", False))
    _ff_ml_meta_c_lite_live = bool(_post_layer_state.get("_ff_ml_meta_c_lite_live", False))
    _ff_ml_layer_uncertainty_live = bool(_post_layer_state.get("_ff_ml_layer_uncertainty_live", False))
    trace_state_alite_live = dict(_post_layer_state.get("trace_state_alite_live") or {})
    trace_state_areal_live = dict(_post_layer_state.get("trace_state_areal_live") or {})
    trace_style_b_lite_live = dict(_post_layer_state.get("trace_style_b_lite_live") or {})
    trace_league_rank_sos_live = dict(_post_layer_state.get("trace_league_rank_sos_live") or {})
    trace_roster_continuity_live = dict(_post_layer_state.get("trace_roster_continuity_live") or {})
    trace_hier_calibration_live = dict(_post_layer_state.get("trace_hier_calibration_live") or {})
    trace_meta_c_lite_live = dict(_post_layer_state.get("trace_meta_c_lite_live") or {})
    trace_layer_uncertainty_live = dict(_post_layer_state.get("trace_layer_uncertainty_live") or {})
    trace_marginal_trust_info = dict(_post_layer_state.get("trace_marginal_trust_info") or {})
    trace_series_coherence_post_info = dict(_post_layer_state.get("trace_series_coherence_post_info") or {})
    trace_pipeline_steps = list(_post_layer_state.get("trace_pipeline_steps") or trace_pipeline_steps)
    trace_pipeline_order = list(_post_layer_state.get("trace_pipeline_order") or [])
    _trace_ctx_model = core_build_ml_trace_context(
        session_state=st.session_state,
        season_label=season_label,
        teamA=teamA,
        teamB=teamB,
        bo=int(bo),
        input_signature_hash=_input_signature_hash,
        input_signature_payload=_input_signature_payload,
        year_opt=year_opt,
        split_opt=split_opt,
        playoffs_opt=playoffs_opt,
        league_mode=league_mode,
        fixed_league=fixed_league,
        ml_engine=ml_engine,
        p_map_raw=p_map_raw,
        p_map_cal=p_map_cal,
        trace_p_map_team=trace_p_map_team,
        trace_p_map_players=trace_p_map_players,
        trace_p_map_fused=trace_p_map_fused,
        p_map_used=p_map_used,
        p_series_used=p_series_used,
        use_lineup=bool(use_lineup),
        lineup_mode=str(locals().get("lineup_mode", "")),
        delta_players=delta_players,
        delta_players_eff=delta_players_eff,
        delta_mode=_delta_mode,
        delta_cap=_delta_cap,
        delta_slope=_delta_slope,
        trace_n_team_eff=trace_n_team_eff,
        trace_n_players_eff=trace_n_players_eff,
        trace_fusion_info=trace_fusion_info,
        trace_form_info=trace_form_info,
        trace_hybrid_info=trace_hybrid_info,
        trace_conf_guard_info=trace_conf_guard_info,
        trace_uncertainty_info=trace_uncertainty_info,
        trace_marginal_trust_info=trace_marginal_trust_info,
        trace_patch_guard_info=trace_patch_guard_info,
        trace_series_gamma_info=trace_series_gamma_info,
        trace_series_coherence_info=trace_series_coherence_info,
        trace_series_coherence_post_info=trace_series_coherence_post_info,
        trace_pipeline_order=trace_pipeline_order,
        trace_pipeline_steps=trace_pipeline_steps,
        guard_triggered=bool(_guard_triggered),
        guard_dpp=_guard_dpp,
        wf_info=_wf_info,
        fused_post_cal_info=core_get_ml_fused_post_cal_info(st.session_state),
        shadow_output=_shadow_state,
        post_output=_post_layer_state,
        trace_series_bo_dplite_live=trace_series_bo_dplite_live,
    )
    _trace = core_build_ml_trace_payload(
        ctx=_trace_ctx_model,
        odd_from_prob_fn=_odd_from_p,
    )
    core_append_ml_shadow_feature_traces(
        append_feature_trace_fn=_append_feature_trace,
        ff_ml_shadow_vnext=bool(_ff_ml_shadow_vnext),
        trace_shadow_ml_vnext=(
            trace_shadow_ml_vnext if isinstance(trace_shadow_ml_vnext, dict) else {}
        ),
        ff_ml_shadow_state_alite=bool(_ff_ml_shadow_state_alite),
        trace_shadow_state_alite=(
            trace_shadow_state_alite if isinstance(trace_shadow_state_alite, dict) else {}
        ),
        ff_ml_shadow_state_areal=bool(_ff_ml_shadow_state_areal),
        trace_shadow_state_areal=(
            trace_shadow_state_areal if isinstance(trace_shadow_state_areal, dict) else {}
        ),
        season_label=season_label,
        teamA=teamA,
        teamB=teamB,
        bo=bo,
        ml_engine=ml_engine,
        input_signature_hash=_input_signature_hash,
    )
    core_append_ml_live_feature_traces(
        append_feature_trace_fn=_append_feature_trace,
        ff_ml_style_b_lite_live=bool(_ff_ml_style_b_lite_live),
        trace_style_b_lite_live=(
            trace_style_b_lite_live if isinstance(trace_style_b_lite_live, dict) else {}
        ),
        ff_ml_league_rank_sos_live=bool(_ff_ml_league_rank_sos_live),
        trace_league_rank_sos_live=(
            trace_league_rank_sos_live if isinstance(trace_league_rank_sos_live, dict) else {}
        ),
        ff_ml_roster_continuity_live=bool(_ff_ml_roster_continuity_live),
        trace_roster_continuity_live=(
            trace_roster_continuity_live if isinstance(trace_roster_continuity_live, dict) else {}
        ),
        ff_ml_hier_calibration_live=bool(_ff_ml_hier_calibration_live),
        trace_hier_calibration_live=(
            trace_hier_calibration_live if isinstance(trace_hier_calibration_live, dict) else {}
        ),
        ff_ml_series_bo_dplite_live=bool(_ff_ml_series_bo_dplite_live),
        trace_series_bo_dplite_live=(
            trace_series_bo_dplite_live if isinstance(trace_series_bo_dplite_live, dict) else {}
        ),
        ff_ml_layer_uncertainty_live=bool(_ff_ml_layer_uncertainty_live),
        trace_layer_uncertainty_live=(
            trace_layer_uncertainty_live if isinstance(trace_layer_uncertainty_live, dict) else {}
        ),
        ff_ml_meta_c_lite_live=bool(_ff_ml_meta_c_lite_live),
        trace_meta_c_lite_live=(
            trace_meta_c_lite_live if isinstance(trace_meta_c_lite_live, dict) else {}
        ),
        season_label=season_label,
        teamA=teamA,
        teamB=teamB,
        bo=bo,
        ml_engine=ml_engine,
        input_signature_hash=_input_signature_hash,
    )
    core_set_latest_ml_trace(st.session_state, _trace)
    _persist_combined_trace_file()

    # Painel centralizado (times + liga + odds + chances)
    # IMPORTANT: a liga exibida deve refletir o recorte/league_mode (e não "vazar" por grafia de time ou por artifact).
    _la = _lb = ""
    if league_mode == "fixed" and filters.league:
        _la = _lb = str(filters.league)
    elif league_mode == "auto":
        # Usa a liga efetivamente usada nas stats (após normalização/auto-league)
        if isinstance(dfA_used, pd.DataFrame) and (not dfA_used.empty) and "league" in dfA_used.columns:
            _la_counts = dfA_used["league"].dropna().astype(str).value_counts()
            if not _la_counts.empty:
                _la = str(_la_counts.index[0]).strip()
        if isinstance(dfB_used, pd.DataFrame) and (not dfB_used.empty) and "league" in dfB_used.columns:
            _lb_counts = dfB_used["league"].dropna().astype(str).value_counts()
            if not _lb_counts.empty:
                _lb = str(_lb_counts.index[0]).strip()

    # Fallback: se não conseguiu inferir pela base, usa o que veio do ML core (artifact)
    if (not _la or not _lb) and _ml_info is not None:
        _la = _la or str(((_ml_info.get("blue") or {}).get("league")) or "").strip()
        _lb = _lb or str(((_ml_info.get("red") or {}).get("league")) or "").strip()

    teamA_disp = _team_with_league_label(teamA, _la)
    teamB_disp = _team_with_league_label(teamB, _lb)

    _winner_pricing = core_compute_winner_pricing_bundle(
        p_team_a_map=st.session_state.get('_p_map_used', p_map_cal),
        p_team_a_series=st.session_state.get('_p_series_used', p_series),
        bo_for_series=int(bo),
        prob_win_series_fn=prob_win_series,
        odd_from_prob_fn=_odd_from_p,
    )
    pA_map = float(_winner_pricing.get("pA_map", float("nan")))
    pB_map = float(_winner_pricing.get("pB_map", float("nan")))
    pA_series = float(_winner_pricing.get("pA_series", float("nan")))
    pB_series = float(_winner_pricing.get("pB_series", float("nan")))
    pA_bo3 = float(_winner_pricing.get("pA_bo3", float("nan")))
    pB_bo3 = float(_winner_pricing.get("pB_bo3", float("nan")))
    pA_bo5 = float(_winner_pricing.get("pA_bo5", float("nan")))
    pB_bo5 = float(_winner_pricing.get("pB_bo5", float("nan")))
    oddA_map = _winner_pricing.get("oddA_map")
    oddB_map = _winner_pricing.get("oddB_map")
    oddA_series = _winner_pricing.get("oddA_series")
    oddB_series = _winner_pricing.get("oddB_series")
    oddA_bo3 = _winner_pricing.get("oddA_bo3")
    oddB_bo3 = _winner_pricing.get("oddB_bo3")
    oddA_bo5 = _winner_pricing.get("oddA_bo5")
    oddB_bo5 = _winner_pricing.get("oddB_bo5")

    with _box_confronto:
        # Visual "Cards + barra de duelo" (escolha do usuário, 2026-07-03):
        # bloco inteiro em UM html flex — dois cards de time, medalhão VS no
        # meio e barra de probabilidade dividida embaixo. Um html único elimina
        # por construção qualquer desalinhamento de colunas do Streamlit.
        try:
            _bo_hdr = int(bo)
        except (TypeError, ValueError):
            _bo_hdr = 1
        _bo_hl = "MD5" if _bo_hdr >= 5 else ("MD3" if _bo_hdr >= 3 else "Mapa")

        _COR_A = "#2daac4"   # ciano (accent do app) — Time 1
        _COR_B = "#8b5cf6"   # violeta — Time 2 (sem semântica bom/ruim)

        def _odds_rows_html(rows: list) -> str:
            # "entra ≥ X" = justa × 1.05 (margem de valor recomendada; substitui
            # a conta de cabeça do "+15% de juice" — pedido dele 2026-07-10).
            parts: list[str] = []
            for _lab, _o, _p in rows:
                try:
                    _entra = f" <span style='opacity:0.5;font-size:0.82em;'>entra ≥{float(_o) * 1.05:.2f}</span>"
                except (TypeError, ValueError):
                    _entra = ""
                _line = (
                    f"<span style='opacity:0.75;'>{_lab}</span>&nbsp; "
                    f"<b>{_format_num(_o, 2)}</b> "
                    f"<span style='opacity:0.7;font-size:0.9em;'>({_format_pct(_p)})</span>"
                    f"{_entra}"
                )
                if _lab == _bo_hl:
                    _line = (
                        "<span style='background:rgba(45,170,196,0.16);"
                        "border:1px solid rgba(45,170,196,0.5);padding:2px 10px;"
                        "border-radius:9px;display:inline-block;font-weight:700;'>"
                        + _line + "</span>"
                    )
                parts.append(f"<div style='padding:3px 0;'>{_line}</div>")
            return "".join(parts)

        def _team_card_html(team_raw: str, disp: str, rows: list, cor: str, rotulo: str) -> str:
            return (
                f"<div style='flex:1;max-width:340px;background:rgba(255,255,255,0.025);"
                f"border:1px solid rgba(255,255,255,0.09);border-top:3px solid {cor};"
                f"border-radius:14px;padding:12px 16px 10px;text-align:center;'>"
                f"<div style='opacity:0.55;font-size:0.76em;text-transform:uppercase;"
                f"letter-spacing:0.06em;'>{_esc(rotulo)}</div>"
                f"<div style='display:flex;align-items:center;justify-content:center;gap:9px;"
                f"margin:6px 0 10px;'>"
                f"{team_logo_html(str(team_raw), 44)}"
                f"<span style='display:inline-flex;align-items:center;height:44px;font-weight:800;"
                f"font-size:1.25em;color:{cor};line-height:1;'>{_esc(str(disp))}</span>"
                f"</div>"
                f"{_odds_rows_html(rows)}"
                f"</div>"
            )

        _card_a = _team_card_html(str(teamA), str(teamA_disp), [
            ("Mapa", oddA_map, pA_map),
            ("MD3", oddA_bo3, pA_bo3),
            ("MD5", oddA_bo5, pA_bo5),
        ], _COR_A, "Time 1")
        _card_b = _team_card_html(str(teamB), str(teamB_disp), [
            ("Mapa", oddB_map, pB_map),
            ("MD3", oddB_bo3, pB_bo3),
            ("MD5", oddB_bo5, pB_bo5),
        ], _COR_B, "Time 2")

        # Barra de duelo: probabilidade do formato em destaque (BO selecionado).
        _p_duel = {"Mapa": pA_map, "MD3": pA_bo3, "MD5": pA_bo5}.get(_bo_hl, pA_map)
        _duel_html = ""
        try:
            _pd = float(_p_duel)
        except (TypeError, ValueError):
            _pd = float("nan")
        if math.isfinite(_pd) and 0.0 < _pd < 1.0:
            _pa_pct = 100.0 * _pd
            _duel_html = (
                f"<div style='max-width:734px;margin:12px auto 2px;'>"
                f"<div style='display:flex;justify-content:space-between;font-size:0.85em;"
                f"margin-bottom:4px;'>"
                f"<span style='color:{_COR_A};font-weight:700;'>{_esc(str(teamA_disp))} {_pa_pct:.1f}%</span>"
                f"<span style='opacity:0.55;font-size:0.85em;'>{_bo_hl}</span>"
                f"<span style='color:{_COR_B};font-weight:700;'>{100.0 - _pa_pct:.1f}% {_esc(str(teamB_disp))}</span>"
                f"</div>"
                f"<div style='display:flex;height:13px;border-radius:7px;overflow:hidden;"
                f"box-shadow:inset 0 0 0 1px rgba(255,255,255,0.08);'>"
                f"<div style='width:{_pa_pct:.1f}%;background:linear-gradient(90deg,{_COR_A},#1f8ba1);'></div>"
                f"<div style='flex:1;background:linear-gradient(90deg,#6d46c9,{_COR_B});'></div>"
                f"</div>"
                f"</div>"
            )

        st.markdown(
            "<div style='display:flex;gap:14px;justify-content:center;align-items:stretch;"
            "flex-wrap:wrap;'>"
            + _card_a
            + "<div style='align-self:center;flex:0 0 auto;width:40px;height:40px;border-radius:50%;"
            "background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.14);"
            "display:flex;align-items:center;justify-content:center;font-weight:800;"
            "font-size:0.85em;opacity:0.75;'>VS</div>"
            + _card_b
            + "</div>"
            + _duel_html,
            unsafe_allow_html=True,
        )

    # Aviso de "Divergência no ML" (2ª opinião Elo vs core) removido da Visão
    # Geral a pedido do usuário (reforma 2026-06). O toggle ml_engine_compare
    # continua existindo nos Parâmetros; só o warning saiu daqui.

    # "Detalhar cálculo (ML core v2)" removido da Visão Geral.

    # Seção "Médias esperadas (totais)" removida da Visão Geral a pedido do
    # usuário (2026-06-10, print). Os totais continuam na Análise de Linha e
    # alimentando o Resumo — só a tabela visual saiu daqui.

    # Painel visual "Odds justas (linhas)" removido da Visão Geral (a pedido).
    # O cálculo do p_ml_scope é MANTIDO porque outras abas (Resumo, Consistência)
    # consomem _p_ml_scope via session_state — não é só visual.
    _pmap_u = float(st.session_state.get('_p_map_used', p_map_cal))
    _pser_u = float(st.session_state.get('_p_series_used', p_series))
    p_ml_scope = _pmap_u if ml_mode == "map" else _pser_u
    core_set_ml_scope(st.session_state, p_ml_scope)

