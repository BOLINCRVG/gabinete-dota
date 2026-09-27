# -*- coding: utf-8 -*-
# GERADO por build_clone.py — NAO EDITAR A MAO (edite os patches).
# Fonte: C:\Users\steam\Desktop\gabinetelolweb3.1\plays_app.py
# -*- coding: utf-8 -*-
# plays_app.py
# Streamlit app: Plays (ML + Totais)
# - Lê ml_artifact.json (ML core v2) para p_map (calibrado) e calcula odds de série (BO1/3/5)
# - Lê CSV OracleElixir (team rows) para calcular Totais (kills/towers/dragons/barons/inhibitors/time)
# - Calcula odds justas de Over/Under e combos ML+Over/Under (helper conjunto com fallback independente)

from __future__ import annotations

# Portatil (2026-07-10): o usuario manda as pastas num .rar pro amigo —
# o repo do LoL e descoberto por vizinhanca (dota_paths), nunca por
# caminho absoluto com username.
import sys as _dota_sys
_dota_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from dota_paths import ensure_lol_on_path as _dota_lol_path
_dota_lol_path()
import dota_overrides  # noqa: F401  Baroes->Roshans etc. ANTES de qualquer app_core derivar listas

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datetime import date
import inspect as _inspect
import json
import sys

import math
import re

from app_core.render_ctx_core import build_render_ctx as _build_render_ctx

import pandas as pd
import numpy as np
import streamlit as st

from mlcore.artifact import MLArtifact
from mlcore.bo import prob_win_series
from mlcore.tiers import tier_base_points, league_base_points
from mlcore.players_layer import PlayersArtifact
from mlcore.achievements import load_achievements, team_boost_points
from mlcore.utils import clip_prob
from mlcore.walkforward_report import run_walkforward
from mlcore.fit_ml_params import fit_params as _fit_ml_params_run
from mlcore.layer_validation import run_layer_validation
from mlcore.calibration_diagnostics import run_calibration_diagnostics
from mlcore.league_context import league_macro_key as shared_league_macro_key
from app_core.market import (
    parse_lines_text,
    parse_time_lines_text,
    matchup_key as build_matchup_key,
    odd_from_prob as _odd_from_p,
)
from app_core.matchup_filters_core import resolve_prev_season_with_fallback as _resolve_prev_season_with_fallback
from app_core.tab_dispatcher import (
    TAB_PARAMS,
    TAB_VISAO,
    TAB_CONS,
    TAB_LINE_ANALYSIS,
    SECONDARY_TAB_ORDER,
    dispatch_secondary_tabs,
)
from app_core.winner_pricing_core import compute_winner_pricing_bundle as core_compute_winner_pricing_bundle
from app_core.line_store import load_json_store, persist_json_store, ensure_session_store
from app_core.lineup_state import (
    POS_CANON as _POS_CANON,
    ensure_lineup_runtime_state as _ensure_lineup_runtime_state_impl,
    get_lineup_loaded_key as _get_lineup_loaded_key_impl,
    get_lineup_runtime_settings as _get_lineup_runtime_settings_impl,
    set_lineup_loaded_key as _set_lineup_loaded_key_impl,
    norm_pos as _norm_pos,
    norm_pos_ml as _norm_pos_ml,
    parse_pct as _parse_pct,
    load_lineup_for_matchup as _load_lineup_for_matchup_impl,
    save_lineup_for_matchup as _save_lineup_for_matchup_impl,
    get_lineup_selected as _get_lineup_selected_impl,
)
from app_core.lines_state import load_lines_for_matchup as _load_lines_for_matchup_impl, save_lines_for_matchup as _save_lines_for_matchup_impl
from app_core.navigation import TAB_ALL as _TAB_ALL, pick_tabs as _pick_tabs
from app_core.navigation_state_core import (
    DEFAULT_UI_PAGE_MODE as NAV_DEFAULT_UI_PAGE_MODE,
    TOP_NAV_OPTIONS as NAV_TOP_NAV_OPTIONS,
    get_ui_page_mode as core_get_ui_page_mode,
    resolve_nav_page as core_resolve_nav_page,
    set_flash_saved_profile_msg as core_set_flash_saved_profile_msg,
)
from app_core.ui_labels import (
    fmt_prob_pct as _format_pct,
    fmt_num as _format_num,
    min_to_mmss as _min_to_mmss,
    metric_label_pt as _metric_label_pt,
    side_label_pt as _side_label_pt,
    fmt_line_disp as _fmt_line_disp,
    fmt_recortes_disp as _fmt_recortes_disp,
)
from app_core.odds_core import (
    tail_df as core_tail_df,
    recency_slices as core_recency_slices,
)
from app_core.resumo_core import (
    wr_laplace as core_wr_laplace,
    counts_ml_totals as core_counts_ml_totals,
    make_pricing_adapters as core_make_pricing_adapters,
    finalize_resumo_dataframe as core_finalize_resumo_dataframe,
    build_ml_rows as core_build_ml_rows,
    build_metric_rows as core_build_metric_rows,
    build_combo_rows as core_build_combo_rows,
)
from app_core.players_core import (
    build_player_means_rows as core_build_player_means_rows,
    build_player_market_rows as core_build_player_market_rows,
    build_recent_player_stats_rows as core_build_recent_player_stats_rows,
)
from app_core.players_feature_trace_core import (
    append_players_recent_stats_panel_trace as core_append_players_recent_stats_panel_trace,
    append_players_fair_panel_trace as core_append_players_fair_panel_trace,
    append_players_markets_panel_trace as core_append_players_markets_panel_trace,
    append_players_panels_bundle_trace as core_append_players_panels_bundle_trace,
)
from app_core.players_state import (
    parse_players_text as core_parse_players_text,
    resolve_players as core_resolve_players,
    build_lane_rows as core_build_lane_rows,
    ensure_players_text_state as core_ensure_players_text_state,
    ensure_players_filter_state as core_ensure_players_filter_state,
    ensure_player_line_input_state as core_ensure_player_line_input_state,
    apply_players_submission_state as core_apply_players_submission_state,
    apply_players_autofill_state as core_apply_players_autofill_state,
    build_per_player_lines_by_idx as core_build_per_player_lines_by_idx,
)
from app_core.players_data_core import (
    load_player_rows as core_load_player_rows,
    apply_filters_players as core_apply_filters_players,
    fuzzy_match_one as core_fuzzy_match_one,
)
from app_core.ui_cache_core import cached_champion_vocab as core_champion_vocab
from app_core.players_lineup_impact_core import (
    load_players_impact as core_load_players_impact,
    apply_shrink_m as core_apply_shrink_m,
    best_match as core_best_match,
    latest_lineup_for_team as core_latest_lineup_for_team,
    latest_lineup_names as core_latest_lineup_names,
)
from app_core.players_elo_core import (
    load_player_rows_multi as core_load_player_rows_multi,
    compute_player_elo_global as core_compute_player_elo_global,
    elo_rank_table as _elo_rank_table_core,
)
from app_core.display_tables_core import (
    df_contract_view as core_df_contract_view,
    display_ou_table as core_display_ou_table,
    display_ml_totals_table as core_display_ml_totals_table,
)
from app_core.csv_bootstrap_runtime_core import (
    apply_bootstrap_selection as core_apply_bootstrap_selection,
    build_runtime_config_defaults as core_build_runtime_config_defaults,
    discover_season_csvs as _discover_season_csvs,
    list_csv_candidates as _list_csv_candidates,
    request_csv_runtime_refresh as core_request_csv_runtime_refresh,
    reset_bootstrap_selection as core_reset_bootstrap_selection,
    resolve_active_csv_and_artifact as core_resolve_active_csv_and_artifact,
    resolve_ml_paths as core_resolve_ml_paths,
    resolve_season_context as core_resolve_season_context,
)
from app_core.csv_prepare_core import (
    csv_diag_path as core_csv_diag_path,
    csv_header_check_path as core_csv_header_check_path,
)
from app_core.flow_runtime_core import (
    apply_pending_load as core_apply_pending_load,
    consume_analysis_force_once as core_consume_analysis_force_once,
    consume_consistency_refresh_pending as core_consume_consistency_refresh_pending,
    clear_matchup_selection as core_clear_matchup_selection,
    consume_pending_reset_matchup as core_consume_pending_reset_matchup,
    ensure_matchup_selection_state as core_ensure_matchup_selection_state,
    get_flow_stage as core_get_flow_stage,
    get_last_season_label as core_get_last_season_label,
    get_lines_loaded_key as core_get_lines_loaded_key,
    reset_flow_state as core_reset_flow_state,
    resolve_players_artifact_path as core_resolve_players_artifact_path,
    request_analysis_refresh as core_request_analysis_refresh,
    set_last_season_label as core_set_last_season_label,
    set_flow_stage as core_set_flow_stage,
    set_lines_loaded_key as core_set_lines_loaded_key,
    set_matchup_key as core_set_matchup_key,
    set_selected_team as core_set_selected_team,
)
from app_core.schedule_import_core import (
    load_schedule_aliases as core_load_schedule_aliases,
    normalize_schedule_team_key as core_normalize_schedule_team_key,
    parse_schedule_text as core_parse_schedule_text,
    resolve_schedule_team as core_resolve_schedule_team,
    save_schedule_aliases as core_save_schedule_aliases,
    suggest_schedule_teams as core_suggest_schedule_teams,
)
from app_core.schedule_state_core import (
    append_schedule_games as core_append_schedule_games,
    bind_schedule_store_path as core_bind_schedule_store_path,
    clear_pending_schedule_import as core_clear_pending_schedule_import,
    clear_schedule_games as core_clear_schedule_games,
    clear_schedule_text as core_clear_schedule_text,
    ensure_schedule_alias_state as core_ensure_schedule_alias_state,
    get_pending_schedule_import as core_get_pending_schedule_import,
    get_schedule_games as core_get_schedule_games,
    has_schedule_unknown_aliases as core_has_schedule_unknown_aliases,
    persist_schedule_runtime_state as core_persist_schedule_runtime_state,
    restore_schedule_runtime_state as core_restore_schedule_runtime_state,
    save_pending_schedule_import as core_save_pending_schedule_import,
    set_schedule_unknown_aliases as core_set_schedule_unknown_aliases,
)
from app_core.pre_delivery_checks_core import (
    assess_anchor_regression as core_assess_anchor_regression,
    assess_confidence_baseline_file as core_assess_confidence_baseline_file,
    assess_combined_trace_file as core_assess_combined_trace_file,
    assess_feature_trace_health as core_assess_feature_trace_health,
    assess_navigation_integrity as core_assess_navigation_integrity,
    assess_patch_calibration_freshness as core_assess_patch_calibration_freshness,
    assess_params_baseline_stability as core_assess_params_baseline_stability,
    assess_ranking_coverage as core_assess_ranking_coverage,
    assess_session_trace as core_assess_session_trace,
    assess_state_artifact_file as core_assess_state_artifact_file,
    assess_utf8_file as core_assess_utf8_file,
    run_pre_delivery_smokes as core_run_pre_delivery_smokes,
)
from app_core.pre_delivery_runtime_core import run_pre_delivery_checks as core_run_pre_delivery_checks_runtime
from app_core.runtime_session_state_core import (
    get_wf_last_report as core_get_wf_last_report,
)
from app_core.daily_pipeline_runtime_core import run_daily_pipeline as core_run_daily_pipeline
from app_core.analysis_runtime_state_core import (
    build_analysis_signature as core_build_analysis_signature,
    build_consistency_analysis_signature as core_build_consistency_analysis_signature,
    compute_sim_seed_series as core_compute_sim_seed_series,
    resolve_active_line_texts as core_resolve_active_line_texts,
    stable_seed_from_parts as _stable_seed_from_parts,
)
from app_core.maintenance_runtime_core import (
    read_artifact_summary_path as core_read_artifact_summary_path,
)
from app_core.bootstrap_maintenance_state_core import (
    read_json_file_safe as core_read_json_file_safe,
)
from app_core.app_state import (
    bootstrap_state as core_bootstrap_state,
    apply_recommended_preset as core_apply_recommended_preset,
    apply_excellent_value_preset as core_apply_excellent_value_preset,
    apply_twenty_ten_preset as core_apply_twenty_ten_preset,
    get_state_schema as core_get_state_schema,
    get_bool as state_get_bool,
    get_float as state_get_float,
    get_int as state_get_int,
    get_str as state_get_str,
)
from app_core.champions_state_core import (
    ensure_champions_filter_state as core_ensure_champions_filter_state,
    reset_champions_league_dependents as core_reset_champions_league_dependents,
    resolve_blue_side_choice as core_resolve_blue_side_choice,
    sync_blue_side_choice as core_sync_blue_side_choice,
)
from app_core.resumo_state_core import (
    apply_resumo_combo_preset as core_apply_resumo_combo_preset,
    apply_resumo_filter_values as core_apply_resumo_filter_values,
    delete_active_resumo_filter_profile as core_delete_active_resumo_filter_profile,
    ensure_resumo_combo_state as core_ensure_resumo_combo_state,
    list_resumo_filter_presets as core_list_resumo_filter_presets,
    load_resumo_filter_custom as core_load_resumo_filter_custom,
    save_resumo_filter_custom as core_save_resumo_filter_custom,
    save_resumo_filter_profile as core_save_resumo_filter_profile,
)
from app_core.resumo_guardrail_core import (
    build_resumo_guardrail_cfg as core_build_resumo_guardrail_cfg,
)
from app_core.resumo_runtime_core import (
    build_market_trace_rows as core_build_market_trace_rows,
    build_statscamps_scope_frame as core_build_statscamps_scope_frame,
    compute_statscamps_pct as core_compute_statscamps_pct,
    ensure_metric_lines as core_ensure_metric_lines,
    format_statscamps_badge as core_format_statscamps_badge,
    infer_matchup_league as core_infer_matchup_league,
)
from app_core.settings_core import (
    settings_path as core_settings_path,
    load_settings_from_disk as core_load_settings_from_disk,
    apply_settings_to_session as core_apply_settings_to_session,
    save_settings_to_disk as core_save_settings_to_disk,
    process_pending_settings_actions as core_process_pending_settings_actions,
    init_settings_once as core_init_settings_once,
)
from app_core.settings_ui import render_settings_sidebar as core_render_settings_sidebar
from app_core.rankings_tab_core import render_rankings_tab as core_render_rankings_tab
from app_core.schedule_tab_core import render_schedule_tab as core_render_schedule_tab
from app_core.series_tab_core import render_series_tab as core_render_series_tab
from app_core.statscamps_tab_core import render_statscamps_tab as core_render_statscamps_tab
from app_core.line_analysis_tab_core import render_line_analysis_tab as core_render_line_analysis_tab
from app_core.players_tab_core import render_players_tab as core_render_players_tab
from app_core.champions_tab_core import render_champions_tab as core_render_champions_tab
from app_core.resumo_tab_core import render_resumo_tab as core_render_resumo_tab
from app_core.identity_tab_core import render_identity_tab as core_render_identity_tab
from app_core.live_bet_tab_core import render_live_bet_tab as core_render_live_bet_tab
from app_core.radar_tab_core import render_radar_tab as core_render_radar_tab
from app_core.meta_tab_core import render_meta_tab as core_render_meta_tab
from app_core.identity_registry_core import (
    load_identity_registry as core_load_identity_registry,
    team_identity_candidates as core_team_identity_candidates,
    team_identity_league_candidates as core_team_identity_league_candidates,
)
from app_core.league_transfer_features_core import (
    load_league_transfer_features as core_load_league_transfer_features,
)
from app_core.matchup_selection_core import (
    render_matchup_selection as core_render_matchup_selection,
    team_with_league_label as _team_with_league_label,
    canon_team_key as _canon_team_key,
    team_canon_maps as _team_canon_maps,
)
from app_core.lines_parsing_core import (
    parse_single_float as _parse_single_float,
)
from app_core.ml_oos_gate_core import (
    wf_correct_p_map as _wf_correct_p_map_core,
)
from app_core.process_runner_core import run_cmd as _run_cmd_core
from app_core.ml_probability_adjustments_core import (
    apply_ml_uncertainty_adjust as _apply_ml_uncertainty_adjust_core,
    apply_ml_marginal_trust_adjust as _apply_ml_marginal_trust_adjust_core,
    apply_ml_microseg_calibration as _apply_ml_microseg_calibration_core,
)
from app_core.anchor_snapshot_core import compute_anchor_snapshot as _compute_anchor_snapshot
from app_core.ml_microseg_core import build_ml_microseg_table as _build_ml_microseg_table_core
from app_core.ml_prediction_core import (
    compute_ml_core_prediction as _compute_ml_core_prediction,
    compute_ml_post_layers as _compute_ml_post_layers,
)
from app_core.schedule_alias_ui_core import render_schedule_alias_mapping as core_render_schedule_alias_mapping
from app_core.plays_bootstrap_ui_core import (
    apply_season_change_reset as core_apply_season_change_reset,
    ensure_achievement_path as core_ensure_achievement_path,
    get_boot_hist_csvs as core_get_boot_hist_csvs,
    normalize_matchup_selection_teams as core_normalize_matchup_selection_teams,
)
from app_core.matchup_league_resolution_core import (
    resolve_runtime_team_leagues as core_resolve_runtime_team_leagues,
)
from app_core.csv_bootstrap_selector_ui_core import render_bootstrap_selector as core_render_bootstrap_selector
from app_core.ml_quality_alarm_ui_core import render_ml_quality_alarm as core_render_ml_quality_alarm
from app_core.technical_runtime_config_ui_core import render_technical_runtime_config_panel as core_render_technical_runtime_config_panel
from app_core.trace_core import (
    build_combined_trace as core_build_combined_trace,
    coerce_feature_flag as core_coerce_feature_flag,
)
from app_core.ml_settings_runtime_core import (
    shared_ml_defaults as core_shared_ml_defaults,
    build_runtime_param_defaults as core_build_runtime_param_defaults,
    local_runtime_defaults as core_local_runtime_defaults,
)
from app_core.ml_session_runtime_core import (
    apply_fit_runtime_state_to_session as core_apply_fit_runtime_state_to_session,
    apply_players_context_runtime_override as core_apply_players_context_runtime_override,
    clear_ml_fused_post_cal_info as core_clear_ml_fused_post_cal_info,
    clear_ml_microseg_info as core_clear_ml_microseg_info,
    clear_wf_ml_corr_info as core_clear_wf_ml_corr_info,
    get_achievement_path as core_get_achievement_path,
    get_analysis_gate_state as core_get_analysis_gate_state,
    get_feature_trace_state as core_get_feature_trace_state,
    get_fit_runtime_snapshot as core_get_fit_runtime_snapshot,
    get_ml_engine_runtime_settings as core_get_ml_engine_runtime_settings,
    get_ml_fused_post_cal_info as core_get_ml_fused_post_cal_info,
    get_ml_microseg_info as core_get_ml_microseg_info,
    get_mlcore_cache_state as core_get_mlcore_cache_state,
    get_ml_trace_snapshot as core_get_ml_trace_snapshot,
    persist_combined_trace_file as core_persist_combined_trace_file,
    set_achievement_path as core_set_achievement_path,
    set_analysis_signature as core_set_analysis_signature,
    set_consistency_analysis_signature as core_set_consistency_analysis_signature,
    set_fit_runtime_labels as core_set_fit_runtime_labels,
    set_fit_used_scales as core_set_fit_used_scales,
    set_ml_fused_post_cal_info as core_set_ml_fused_post_cal_info,
    set_ml_microseg_info as core_set_ml_microseg_info,
    set_mlcore_cache_state as core_set_mlcore_cache_state,
    set_wf_ml_corr_info as core_set_wf_ml_corr_info,
)
from app_core.results_runtime_state_core import (
    resolve_hc_kills_team as core_resolve_hc_kills_team,
    resolve_results_runtime_settings as core_resolve_results_runtime_settings,
)
from app_core.calibration_schedule import build_patch_snapshot as core_build_patch_snapshot, recommend_recalibration as core_recommend_recalibration
from app_core.fusion_core import (
    fuse_with_coherence_guard as core_fuse_with_coherence_guard,
)
from app_core.params_tuning_core import (
    enrich_params_with_areal_prob_tuning as _enrich_params_with_areal_prob_tuning,
    enrich_params_with_ml_marginal_band_coeffs as core_enrich_params_with_ml_marginal_band_coeffs,
)
from app_core.fit_runtime_core import (
    apply_fused_post_calibration as core_apply_fused_post_calibration,
    build_fit_runtime_state as core_build_fit_runtime_state,
    fit_form_for_league as core_fit_form_for_league,
    fit_players_context_for_league as core_fit_players_context_for_league,
    fit_players_lane_for_league as core_fit_players_lane_for_league,
    fit_players_league_weight_for_league as core_fit_players_league_weight_for_league,
    fit_scales_for_league as core_fit_scales_for_league,
)
from app_core.ml_league_runtime_core import (
    MAJOR_LEAGUES as CORE_ML_MAJOR_LEAGUES,
    canon_league_display_label as core_canon_league_display_label,
    is_marginal_league_for_ml as core_is_marginal_league_for_ml,
    league_is_excluded as core_league_is_excluded,
    ml_league_conf_band as core_ml_league_conf_band,
    ml_marginal_band_profile as core_ml_marginal_band_profile,
    norm_league as core_norm_league,
    norm_league_key as core_norm_league_key,
    sanitize_ml_marginal_band_coeffs as core_sanitize_ml_marginal_band_coeffs,
    tier_from_league as core_tier_from_league,
)
from app_core.ml_league_inference_core import (
    infer_match_league_from_csv as core_infer_match_league_from_csv,
    infer_team_league_from_csv as core_infer_team_league_from_csv,
)
from app_core.ml_artifact_pricing_core import (
    mlcore_p_map_v2 as core_mlcore_p_map_v2,
    p_from_diff_points as core_p_from_diff_points,
    strength_breakdown as core_strength_breakdown,
    team_meta_from_artifact as _team_meta_from_artifact,
)
from mlcore.lines_engine import (
    build_profile,
    filter_team_games,
    matchup_expected_totals,
    simulate_series_total,
    total_over_prob,
    normalize_map_mode,
    _combine_mean,
    _combine_sd,
    MatchupTotals,
)
from app_core.window_blend_core import (
    GERAL_WINDOWS as _GERAL_WINDOWS,
    blend_matchup_totals as _blend_matchup_totals,
    geral_window_weight as _geral_window_weight,
)
from mlcore.state_team_artifact import (
    load_team_state_artifact as core_load_team_state_artifact,
)

from mlcore.stats_teams import Filters, build_team_games, load_team_rows, apply_filters

from app_core.text_normalization import (
    _fix_mojibake_text,
    _normalize_text_key,
    _canonicalize_tab_labels,
    _patch_streamlit_text_rendering,
)
from app_core.params_tab_core import render_params_tab as _render_params_tab
from app_core.visao_tab_core import render_visao_tab as _render_visao_tab
from app_core.consistencia_tab_core import render_consistencia_tab as _render_consistencia_tab

# NOTE: plays_app passa ctx=globals() para render_visao_tab, que lê dezenas de
# core_* via ctx.get("core_x"). Esses imports são "usados" só por string —
# ferramentas de lint (F401) não enxergam o uso. Mantidos abaixo com noqa.
# fmt: off
from app_core.confidence_governor_core import apply_confidence_governor as core_apply_confidence_governor
from app_core.consistency_ui_core import render_stateful_tab_selector as core_render_stateful_tab_selector
from app_core.lineup_runtime_core import (
    autofill_lineup_state as core_autofill_lineup_state,
    build_manual_lineup_choices as core_build_manual_lineup_choices,
    build_ml_lineup_choices as core_build_ml_lineup_choices,
    resolve_players_layer_config as core_resolve_players_layer_config,
    score_manual_lineup_team as core_score_manual_lineup_team,
)
from app_core.lineup_ui_core import render_lineup_selector_column as core_render_lineup_selector_column
from app_core.ml_explainability_core import (
    build_confidence_summary as core_build_confidence_summary,
    build_pipeline_explainability as core_build_pipeline_explainability,
    build_pipeline_step_rows as core_build_pipeline_step_rows,
)
from app_core.ml_live_feature_trace_core import append_ml_live_feature_traces as core_append_ml_live_feature_traces
from app_core.ml_orchestrator_core import run_ml_post_layers as core_run_ml_post_layers
from app_core.ml_pre_pipeline_core import run_ml_pre_pipeline as core_run_ml_pre_pipeline
from app_core.ml_session_runtime_core import (
    get_wf_ml_corr_info as core_get_wf_ml_corr_info,
    set_latest_ml_trace as core_set_latest_ml_trace,
    set_ml_scope as core_set_ml_scope,
)
from app_core.ml_shadow_feature_trace_core import append_ml_shadow_feature_traces as core_append_ml_shadow_feature_traces
from app_core.ml_shadow_layers_core import run_ml_shadow_layers as core_run_ml_shadow_layers
from app_core.ml_trace_payload_core import build_ml_trace_payload as core_build_ml_trace_payload
from app_core.runtime_session_state_core import get_lv_last_report as core_get_lv_last_report
from app_core.state_strength_areal_core import compute_state_shadow_areal as core_compute_state_shadow_areal
from app_core.state_strength_core import compute_state_shadow_alite as core_compute_state_shadow_alite
from app_core.visao_geral_ml_pipeline_core import (
    build_ml_trace_context as core_build_ml_trace_context,
    build_pre_engine_input as core_build_pre_engine_input,
    build_pre_live_input as core_build_pre_live_input,
    build_runtime_input_signature as core_build_runtime_input_signature,
    persist_runtime_analysis_state as core_persist_runtime_analysis_state,
    unpack_post_layer_output as core_unpack_post_layer_output,
    unpack_shadow_output as core_unpack_shadow_output,
)
from app_core.visao_geral_runtime_core import (
    apply_runtime_tuning_to_session as core_apply_runtime_tuning_to_session,
    build_microseg_caption as core_build_microseg_caption,
    resolve_shared_league_hint as core_resolve_shared_league_hint,
)
from mlcore.fusion_core import (
    current_season_override as core_current_season_override,
    hybrid_momentum_blend as core_hybrid_momentum_blend,
    season_form_blend as core_season_form_blend,
)
from mlcore.roster_signal_core import (
    lineup_continuity_from_details as core_lineup_continuity_from_details,
    transfer_signal_from_details as core_transfer_signal_from_details,
)
from mlcore.series_bo_core import (
    apply_series_bo_dplite_live as core_apply_series_bo_dplite_live,
    enforce_series_economic_coherence as core_enforce_series_economic_coherence,
)
from app_core.analysis_runtime_state_core import ml_consistency_signature as _ml_consistency_signature
from app_core.consistency_core import (
    build_audit_base_options as core_build_audit_base_options,
    build_audit_market_options as core_build_audit_market_options,
    build_audit_view as core_build_audit_view,
    build_ml_totals_laplace_payload as core_build_ml_totals_laplace_payload,
    build_ml_totals_model_payload as core_build_ml_totals_model_payload,
    build_quick_metric_options as core_build_quick_metric_options,
    build_quick_selector_rows as core_build_quick_selector_rows,
    build_simple_audit_view as core_build_simple_audit_view,
    has_metric_history as core_has_metric_history,
    quick_outcome_probs as core_quick_outcome_probs,
    quick_team_frame as core_quick_team_frame,
)
from app_core.consistency_ui_core import (
    render_market_context_hero as core_render_market_context_hero,
    render_market_context_panel as core_render_market_context_panel,
    render_market_mode_section as core_render_market_mode_section,
    render_summary_block as core_render_summary_block,
)
from app_core.display_tables_core import totals_table as _totals_table
from app_core.lineup_state import (
    POS_CANON_ML as _POS_CANON_ML,
    lineup_key as _lineup_key,
    safe_key as _safe_key,
)
from app_core.market_stats_ui_core import render_market_stats_tab as core_render_market_stats_tab
from app_core.ml_confidence_scoring_core import ml_confidence_from_layer_validation as _ml_confidence_from_layer_validation
from app_core.ml_probability_adjustments_core import (
    _safe_logit,
    _safe_sigmoid,
    apply_series_gamma as _apply_series_gamma,
)
from app_core.ml_session_runtime_core import get_ml_scope as core_get_ml_scope
from app_core.odds_core import (
    event_block as core_event_block,
    laplace_odds as core_laplace_odds,
    ou_block as core_ou_block,
)
from app_core.opponent_profile_ui_core import render_opponent_profile_tab as core_render_opponent_profile_tab
from app_core.players_elo_core import delta_players_effect as _delta_players_effect
from app_core.quick_selector_ui_core import render_quick_selector_controls as core_render_quick_selector_controls
from app_core.ui_labels import MARKET_TAB_NAMES as _MARKET_TAB_NAMES
from mlcore.players_layer import compute_delta_and_confidence
from mlcore.tiers import tier_priority
from app_core.ml_output_models import MlPrePipelineOutput as CoreMlPrePipelineOutput
from mlcore.utils import safe_float_or_none as _fnum
# fmt: on
from app_core.global_css_core import inject_global_ui_css as _inject_global_ui_css
from app_core.global_banners_ui_core import (
    render_global_feature_trace_banner as _render_global_feature_trace_banner,
    render_global_maintenance_banner as _render_global_maintenance_banner,
)
from app_core.calibracoes_ui_core import render_calibracoes as _render_calibracoes
from app_core.csv_bootstrap_flow_core import (
    csv_bootstrap_gate as _csv_bootstrap_gate_flow,
    collect_maintenance_status as _collect_maintenance_status_flow,
    run_auto_maintenance_now as _run_auto_maintenance_now_flow,
)
from app_core.odds_cards_ui_core import render_odds_cards as _render_odds_cards_core
from app_core.achievements_resolve_core import ach_layer_for_artifact as _ach_layer_for_artifact_core
from app_core.rank_table_core import mlcore_rank_table_v2 as _mlcore_rank_table_v2_core
from app_core.team_state_persist_core import generate_and_persist_team_state_artifact as _generate_and_persist_team_state_artifact_core
from app_core.elo_matchup_core import elo_season_players_p_map as _elo_season_players_p_map_core
from app_core.backup_core import backup_file_if_exists as _backup_file_if_exists_core
from app_core.calibrate_p_core import calibrate_p_from_artifact as _calibrate_p_from_artifact_core
from app_core.patch_guard_core import patch_meta_guard_payload as _patch_meta_guard_payload_core, patch_changed_leagues_set as _patch_changed_leagues_set_core
from app_core.lines_table_core import lines_table as _lines_table_core, lines_table_with_p_ml as _lines_table_with_p_ml_core
from app_core.series_totals_core import series_totals_table as _series_totals_table_core
from app_core.confidence_runtime_core import (
    load_confidence_baseline_runtime as _load_confidence_baseline_runtime_core,
    series_gamma_for_league as _series_gamma_for_league_core,
    runtime_confidence_governor_budget_for_league as _runtime_confidence_governor_budget_for_league_core,
    runtime_series_coherence_post_cap_for_league as _runtime_series_coherence_post_cap_for_league_core,
)
from app_core.ml_confidence_baseline_persist_core import generate_and_persist_ml_confidence_baseline as _generate_and_persist_ml_confidence_baseline_core
from app_core.feature_trace_append_core import append_feature_trace as _append_feature_trace_core
from app_core.players_artifact_runtime_core import resolve_runtime_players_artifact_path as _resolve_runtime_players_artifact_path_core


def _league_macro_key(league: Any) -> str:
    return str(shared_league_macro_key(league, unknown="OTHER"))


# _team_with_league_label imported from app_core.matchup_selection_core


def _clear_streamlit_cache_data() -> None:
    cache_data = getattr(st, "cache_data", None)
    clear_fn = getattr(cache_data, "clear", None) if cache_data is not None else None
    if callable(clear_fn):
        clear_fn()


def _request_streamlit_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()





_patch_streamlit_text_rendering()


# -----------------------------
# App settings (persistência em disco)
# -----------------------------
_SETTINGS_FILE = ".gabinete_settings.json"

# Chaves que vale a pena persistir entre sessões (evitar salvar coisas gigantes do session_state)
_PERSIST_KEYS: List[str] = [
    "cfg_profile",
    # paths / boot
    "cfg_auto_paths",
    "cfg_csv_override",
    "cfg_artifact_override",
    "boot_daily_csv",
    "boot_hist_csvs",
    # filtros / modos comuns
    "window_short",
    "totals_mode",
    "ml_mode",
    "map_mode_ui",
    "league_mode",
    "year_opt",
    "split_opt",
    "playoffs_opt",
    "fixed_league",
    "weight_mode",
    "half_life_days",
    "n_sims",
    "resumo_max_odd",
    "resumo_combo_preset",
    "resumo_filter_preset",
    "resumo_filter_custom",
    "resumo_filter_profiles",
    "resumo_filter_profile_active",
    "resumo_combo_hist_w",
    "resumo_combo_style_w",
    "resumo_combo_form_w",
    "resumo_combo_use_mismatch",
    "resumo_guardrail_profile",
    "resumo_guardrail_strength_major",
    "resumo_guardrail_strength_mid",
    "resumo_guardrail_strength_minor",
    "resumo_guardrail_target_major",
    "resumo_guardrail_target_mid",
    "resumo_guardrail_target_minor",
    "ml_engine",
    "ml_engine_compare",
    "ml_engine_alert_pp",
    "ml_microseg_enabled",
    "ml_microseg_strength",
    "ml_microseg_min_n",
    "ml_microseg_shrink_k",
    "ml_microseg_lookback_games",
    "ml_microseg_cap_pp",
    "ml_microseg_hierarchical",
    "ml_microseg_hier_decay",
    "ml_fused_post_cal_use_fitted",
    "ml_uncertainty_enabled",
    "ml_uncertainty_strength",
    "ml_uncertainty_prior_games",
    "ml_uncertainty_max_shrink",
    "ml_uncertainty_exclude_leagues",
    "ml_confidence_governor_enabled",
    "ml_confidence_governor_live_enabled",
    "ml_confidence_governor_max_total_shrink_pp",
    "ml_confidence_governor_use_fitted_by_league",
    "ml_series_coherence_post_use_fitted_by_league",
    "ml_series_coherence_post_max_dev_pp_bo3",
    "ml_series_coherence_post_max_dev_pp_bo5",
    "ml_marginal_trust_enabled",
    "ml_marginal_trust_strength",
    "ml_marginal_trust_prior_games",
    "ml_marginal_trust_max_shrink",
    "ml_marginal_trust_cap_prob",
    "ml_marginal_trust_score_soft",
    "ml_marginal_trust_score_hard",
    "ml_marginal_trust_exclude_leagues",
    "ml_marginal_trust_band_tuning",
    "ml_marginal_trust_band_coeffs",
    "ml_show_calc_details_default",
    "ml_as_of_date",
    "ui_fast_flow",
    # lineup
    "use_lineup_adjust",
    "players_csv_path",
    "lineup_lambda_points",
    "lineup_delta_mode",
    "lineup_delta_cap",
    "lineup_delta_slope",
    "lineup_min_gp",
    "lineup_shrink_m",
    "ml_players_w_lineup",
    "ml_players_lane_w_top",
    "ml_players_lane_w_jungle",
    "ml_players_lane_w_mid",
    "ml_players_lane_w_adc",
    "ml_players_lane_w_support",
    "ml_players_lg_w_lpl",
    "ml_players_lg_w_lck",
    "ml_players_lg_w_emea",
    "ml_players_lg_w_na",
    "ml_players_lg_w_br",
    "ml_players_lg_w_apac",
    "ml_players_lg_w_other",
    "ml_players_context_use_fitted",
    "ml_players_context_enabled",
    "ml_players_context_lookback_games",
    "ml_players_context_half_life_days",
    "ml_players_context_min_games",
    "ml_players_context_strength",
    "ml_players_context_cap",
    "ml_players_context_min_reliability",
    "ml_players_context_inter_mult",
    "fusion_team_weight_early",
    "fusion_team_weight_mid",
    "fusion_team_weight_playoffs",
    "fusion_early_season_team_weight",
    "fusion_season_knee_games",
    "fusion_coverage_power",
    "fusion_transfer_boost",
    "fusion_divergence_pp_cap",
    "fusion_divergence_low_coverage",
    "fusion_divergence_shrink",
    "fusion_profile_preset",
    "ml_strategy_profile",
    "fusion_roster_cont_thresh_base",
    "fusion_roster_cont_thresh_early_boost",
    "fusion_roster_low_freshness_cut",
    "fusion_roster_low_freshness_boost",
    "resumo_use_edge_gate",
    "resumo_min_edge_pp",
    "wf_min_train_games",
    "wf_k",
    "wf_scale",
    "auto_maint_fast_mode",
    "auto_maint_run_layer_validation",
    "auto_maint_force_fast",
    "auto_maint_force_full",
    "auto_maint_fast_new_games_max",
    "auto_maint_cmd_timeout_s",
    "auto_maint_params_timeout_s",
    "auto_maint_lock_ttl_s",
    "wf_apply_ml_correction",
    "wf_ml_corr_strength",
    "lv_min_train_games",
    "lv_min_league_games",
    "resumo_min_sample_liga",
    "resumo_min_sample_h2h",
    "resumo_hist_shrink_n",
    "sim_seed_mode",
    "sim_seed_manual",
    "ml_consistency_guard",
    "ml_consistency_tol_pp",
    "ml_hybrid_blend_enabled",
    "ml_hybrid_blend_max_weight",
    "ml_hybrid_blend_knee_games",
    "ml_hybrid_momentum_games",
    "ml_hybrid_beta",
    "ml_conf_guard_enabled",
    "ml_conf_guard_min_games",
    "ml_conf_guard_max_shrink",
    "ml_patch_guard_enabled",
    "ml_patch_guard_warn_shrink",
    "ml_patch_guard_refit_shrink",
    "ml_patch_guard_focus_bonus",
    "ml_series_gamma_use_fitted",
    "cal_diag_bins",
    "cal_diag_min_league_games",
    # elo params (se usar / comparar)
    "elo_team_hl",
    "elo_player_hl",
    "elo_w_players",
    "elo_k_team",
    "elo_k_player",
    "elo_shrink_m",
]

def _settings_path() -> Path:
    return core_settings_path(Path(__file__).resolve().parent, _SETTINGS_FILE)

def _load_app_settings_from_disk() -> Dict[str, Any]:
    return core_load_settings_from_disk(_settings_path(), date_keys=("ml_as_of_date",))

def _apply_settings_to_session(settings: Dict[str, Any], force: bool = False) -> None:
    core_apply_settings_to_session(st.session_state, settings or {}, persist_keys=_PERSIST_KEYS, force=bool(force))

def _save_app_settings_to_disk() -> None:
    core_save_settings_to_disk(_settings_path(), st.session_state, persist_keys=_PERSIST_KEYS)

_STATE_SCHEMA = core_get_state_schema()
_PARAM_DEFAULTS: Dict[str, Any] = core_build_runtime_param_defaults(_STATE_SCHEMA.get("defaults", {}))
_STATE_MIGRATIONS: Dict[str, str] = dict(_STATE_SCHEMA.get("migrations", {}))
_STATE_CONSTRAINTS: Dict[str, Tuple[float, float]] = dict(_STATE_SCHEMA.get("constraints", {}))

_SHARED_ML_DEFAULTS: Dict[str, Any] = core_shared_ml_defaults()

# Bootstrap central de estado (defaults + migrações + saneamento).
core_bootstrap_state(
    st.session_state,
    defaults=_PARAM_DEFAULTS,
    migrations=_STATE_MIGRATIONS,
    constraints=_STATE_CONSTRAINTS,
)

# Defaults locais (não dependem de schema externo).
_LOCAL_DEFAULTS: Dict[str, Any] = core_local_runtime_defaults()
for _k, _v in _LOCAL_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("auto_maint_fast_mode", True)
st.session_state.setdefault("auto_maint_run_layer_validation", False)
st.session_state.setdefault("auto_maint_force_fast", True)
st.session_state.setdefault("auto_maint_force_full", False)
st.session_state.setdefault("auto_maint_fast_new_games_max", 30)
st.session_state.setdefault("auto_maint_cmd_timeout_s", 1200)
st.session_state.setdefault("auto_maint_params_timeout_s", 14000)
st.session_state.setdefault("auto_maint_lock_ttl_s", 600)


def _reset_all_tuning_to_defaults() -> None:
    core_apply_recommended_preset(st.session_state, "Padrão")
    for k, v in _PARAM_DEFAULTS.items():
        st.session_state[k] = v


def _render_settings_sidebar() -> None:
    core_render_settings_sidebar(
        save_settings_fn=_save_app_settings_to_disk,
        settings_path=_settings_path(),
    )


def _process_pending_settings_actions() -> None:
    core_process_pending_settings_actions(
        st.session_state,
        load_settings_fn=_load_app_settings_from_disk,
        apply_settings_fn=_apply_settings_to_session,
        apply_recommended_preset_fn=lambda prof: core_apply_recommended_preset(st.session_state, profile=prof),
        apply_excellent_preset_fn=lambda: core_apply_excellent_value_preset(st.session_state),
        apply_twenty_ten_preset_fn=lambda: core_apply_twenty_ten_preset(st.session_state),
        reset_all_tuning_fn=_reset_all_tuning_to_defaults,
        param_defaults=_PARAM_DEFAULTS,
    )


def _init_app_settings_once() -> None:
    core_init_settings_once(
        st.session_state,
        load_settings_fn=_load_app_settings_from_disk,
        apply_settings_fn=_apply_settings_to_session,
        process_pending_fn=_process_pending_settings_actions,
    )


# -----------------------------
# Helpers
# -----------------------------
def _file_sig(path: str) -> Tuple[str, int, int]:
    p = Path(path)
    try:
        st_ = p.stat()
        return (str(p.resolve()), int(st_.st_size), int(st_.st_mtime))
    except OSError:
        return (str(p), 0, 0)




@st.cache_data(show_spinner=False)
def _csv_header_check(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    csv_path, _size, _mtime = csv_sig
    return core_csv_header_check_path(csv_path)




def _csv_bootstrap_gate(
    *,
    csv_path: str,
    hist_csvs: Optional[List[str]] = None,
    artifact_path: str = "",
) -> None:
    """Tela separada de preparo do CSV + manuten\u00e7\u00e3o autom\u00e1tica quando o CSV muda."""
    return _csv_bootstrap_gate_flow(csv_path=csv_path, hist_csvs=hist_csvs, artifact_path=artifact_path, st=st, ctx=globals())



@st.cache_data(show_spinner=False)
def _csv_diag(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    csv_path, _size, _mtime = csv_sig
    return core_csv_diag_path(csv_path)



@st.cache_data(show_spinner=False)
def _load_team_games(csv_sig: Tuple[str, int, int]) -> pd.DataFrame:
    csv_path, _size, _mtime = csv_sig
    rows = load_team_rows(csv_path)
    df = build_team_games(rows)

    # Anotações manuais de fase (ex.: EWC qualify vs main) — define `split` por
    # liga+intervalo de datas. NO-OP enquanto competition_phases.json tiver
    # 'rules' vazio. (Cache: editar o JSON exige rerun/limpar cache.)
    try:
        from app_core.competition_phases_core import (
            apply_competition_phases,
            apply_ewc_regional_phases,
        )
        # Automático: rotula jogos da EWC por qualifier regional (intra-região →
        # "Qualify KR/NA/EU/APAC/CN") vs evento principal (cross-região → "Main").
        df = apply_ewc_regional_phases(df)
        # Manual: regras extras por liga+data (no-op enquanto o JSON estiver vazio).
        df = apply_competition_phases(df, path=Path(__file__).resolve().parent / "competition_phases.json")
    except Exception:
        pass

    # Canonicaliza nomes de times para evitar que capitalização diferente (ex.: "Kia" vs "KIA")
    # faça o mesmo time aparecer como se fosse outro e caia em liga errada no modo auto.
    maps = _team_canon_maps(csv_sig)
    key_to_best = maps.get("variant_to_best_key") if isinstance(maps, dict) else None
    if isinstance(key_to_best, dict) and len(key_to_best) > 0 and "team" in df.columns:
        def _canonize_name(x: Any) -> str:
            s = ("" if x is None else str(x)).strip()
            s = re.sub(r"\s+", " ", s)
            k = _canon_team_key(s)
            return str(key_to_best.get(k, s))
        df["team"] = df["team"].map(_canonize_name)
        if "opponent" in df.columns:
            df["opponent"] = df["opponent"].map(_canonize_name)
    # Canonicaliza ligas em aliases conhecidos (ex.: CCWS -> PRM) para não fragmentar filtros/rankings.
    if "league" in df.columns:
        df["league"] = df["league"].map(core_canon_league_display_label)
    # EWC: divide por região (KR/EU/CN/NA/...), inferida da liga nativa de cada time.
    from app_core.ewc_region_split import annotate_ewc_region_split
    df = annotate_ewc_region_split(df)
    return df
# _canon_team_key imported from app_core.matchup_selection_core


@st.cache_data(show_spinner=False)




@st.cache_data(show_spinner=False)
def _list_teams(csv_sig: Tuple[str, int, int]) -> List[str]:
    maps = _team_canon_maps(csv_sig)
    teams = maps.get("best_list") if isinstance(maps, dict) else None
    if not isinstance(teams, list) or len(teams) == 0:
        # fallback (não deveria acontecer)
        csv_path, _size, _mtime = csv_sig
        rows = load_team_rows(csv_path)
        teams = sorted(rows["teamname"].dropna().unique().tolist())
    return teams


@st.cache_data(show_spinner=False)
def _load_player_rows(csv_sig: Tuple[str, int, int]) -> pd.DataFrame:
    """Carrega rows de players (participantid 1..10) do OracleElixir."""
    csv_path, _size, _mtime = csv_sig
    return core_load_player_rows(
        csv_path=str(csv_path),
        canon_league_display_label_fn=core_canon_league_display_label,
        norm_pos_ml_fn=_norm_pos_ml,
    )



# _stable_seed_from_parts, _ml_consistency_signature imported from app_core.analysis_runtime_state_core


@st.cache_data(show_spinner=False)
def _series_sims_cached(
    bo: int,
    p_blue_map: float,
    total_obj,
    n_sims: int,
    seed: int = 1337,
) -> np.ndarray:
    """Cache Monte Carlo sims for series totals (per metric)."""
    # total_obj is a frozen dataclass (MatchupTotals), safe to cache by value.
    return simulate_series_total(
        bo=int(bo),
        p_blue_map=float(p_blue_map),
        total=total_obj,
        n_sims=int(n_sims),
        seed=int(seed),
    )


# -----------------------------
# ML core v2: carregar artifact, calcular total_strength, ML e ranking
# -----------------------------

def _resolve_relpath(base_file: str, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((Path(base_file).parent / p).resolve())


def _artifact_sig(path: str) -> Tuple[str, int, int]:
    return _file_sig(path)


@st.cache_resource(show_spinner=False)
def _load_artifact_cached(art_sig: Tuple[str, int, int]) -> MLArtifact:
    art_path, _size, _mtime = art_sig
    return MLArtifact.load_json(art_path)


@st.cache_data(show_spinner=False)
def _load_achievements_cached(ach_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Carrega achievements.json com tolerância a erros.

    Nunca deve derrubar o app (ex.: JSON corrompido). Em caso de falha,
    retorna um dict com '__load_error__' para o chamador decidir o fallback.
    """
    ach_path, _size, _mtime = ach_sig
    try:
        return load_achievements(ach_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        return {
            "__load_error__": f"{type(e).__name__}: {e}",
            "__path__": str(ach_path),
        }


@st.cache_data(show_spinner=False)
def _load_players_artifact_cached(p_sig: Tuple[str, int, int]) -> PlayersArtifact:
    p_path, _size, _mtime = p_sig
    return PlayersArtifact.load_json(p_path)

@st.cache_data(show_spinner=False)
def _load_fitted_params_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


@st.cache_data(show_spinner=False)
def _load_ml_confidence_baseline_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}


@st.cache_data(show_spinner=False)
def _load_team_state_artifact_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    d = core_load_team_state_artifact(cfg_path)
    return d if isinstance(d, dict) else {}


@st.cache_data(show_spinner=False)
def _load_league_transfer_features_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    d = core_load_league_transfer_features(cfg_path)
    return d if isinstance(d, dict) else {}



@st.cache_data(show_spinner=False)
def _patch_meta_guard_payload(
    csv_sig: Tuple[str, int, int],
    fit_sig: Tuple[str, int, int],
) -> Dict[str, Any]:
    return _patch_meta_guard_payload_core(csv_sig, fit_sig, ctx=globals())


def _patch_changed_leagues_set(changed_rows: Any) -> set:
    return _patch_changed_leagues_set_core(changed_rows)



def _ach_layer_for_artifact(art: MLArtifact, artifact_path: str, ach_override_path: Optional[str] = None) -> Dict[str, Any]:
    return _ach_layer_for_artifact_core(art, artifact_path, ach_override_path, ctx=globals())


def _p_from_diff_points(diff_points: float, scale_points: float = 400.0) -> float:
    return float(core_p_from_diff_points(diff_points, scale_points=scale_points, clip_prob_fn=clip_prob))




# -----------------------------
# Helpers: infer league/tier for ML core v2 (to avoid artifact mislabeling)
# -----------------------------
_MAJOR_LEAGUES = {
    *CORE_ML_MAJOR_LEAGUES,
}


@st.cache_data(show_spinner=False)
def _load_identity_registry_cached(_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    path = APP_ROOT / "identity_registry.json"
    return dict(core_load_identity_registry(path))


def _identity_registry_sig() -> Tuple[str, int, int]:
    path = APP_ROOT / "identity_registry.json"
    if not path.exists():
        return ("", 0, 0)
    return _file_sig(str(path))


def _identity_team_name_candidates(team: str) -> list[str]:
    registry = _load_identity_registry_cached(_identity_registry_sig())
    return list(core_team_identity_candidates(registry, team))


def _identity_team_league_candidates(team: str) -> list[str]:
    registry = _load_identity_registry_cached(_identity_registry_sig())
    out: list[str] = []
    for value in list(core_team_identity_league_candidates(registry, team)):
        text = str(value or "").strip()
        if not text:
            continue
        canon = str(core_norm_league_key(text) or text).strip()
        if canon and canon not in out:
            out.append(canon)
    return out


@st.cache_data(show_spinner=False)
def _league_inference_csv_sigs(
    _primary_sig: Tuple[str, int, int],
    year: Optional[int],
) -> list[Tuple[str, int, int]]:
    out: list[Tuple[str, int, int]] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        if not path.exists():
            return
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        out.append(_file_sig(str(path)))

    primary_raw = str((_primary_sig or ("", 0, 0))[0] or "").strip()
    if primary_raw:
        _push(Path(primary_raw))

    candidates: list[Path] = []
    for base in [APP_ROOT / "CSV_DIARIO", APP_ROOT / "CSV_HISTORICO"]:
        if not base.exists():
            continue
        candidates.extend(base.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))

    def _year_score(path: Path) -> int:
        _prefix = str(path.name).split("_", 1)[0].strip()
        return int(_prefix) if _prefix.isdigit() else 0

    try:
        target_year = int(year) if year is not None else 0
    except (TypeError, ValueError):
        target_year = 0
    candidates = sorted(
        candidates,
        key=lambda path: (abs(_year_score(path) - target_year) if target_year else 0, -_year_score(path), path.name),
    )
    for path in candidates:
        _push(path)
    return out


@st.cache_data(show_spinner=False)
def _infer_team_league_from_csv(_csv_sig: Tuple[str, int, int], team: str, year: Optional[int]) -> str:
    aliases = _identity_team_name_candidates(team)
    for _sig in _league_inference_csv_sigs(_csv_sig, year):
        out = core_infer_team_league_from_csv(
            _sig,
            team,
            year,
            norm_league_key_fn=core_norm_league_key,
            major_leagues=_MAJOR_LEAGUES,
            team_aliases=aliases,
            fallback_any_year=True,
        )
        if str(out or "").strip():
            return str(out)
    for league in _identity_team_league_candidates(team):
        if str(league or "").strip():
            return str(league)
    return ""


@st.cache_data(show_spinner=False)
def _infer_match_league_from_csv(_csv_sig: Tuple[str, int, int], teamA: str, teamB: str, year: Optional[int]) -> str:
    aliases_a = _identity_team_name_candidates(teamA)
    aliases_b = _identity_team_name_candidates(teamB)
    for _sig in _league_inference_csv_sigs(_csv_sig, year):
        out = core_infer_match_league_from_csv(
            _sig,
            teamA,
            teamB,
            year,
            norm_league_key_fn=core_norm_league_key,
            major_leagues=_MAJOR_LEAGUES,
            load_team_games_fn=_load_team_games,
            infer_team_league_fn=lambda sig, team, y: core_infer_team_league_from_csv(
                sig,
                team,
                y,
                norm_league_key_fn=core_norm_league_key,
                major_leagues=_MAJOR_LEAGUES,
                team_aliases=_identity_team_name_candidates(team),
                fallback_any_year=True,
            ),
            team_aliases_a=aliases_a,
            team_aliases_b=aliases_b,
            fallback_any_year=True,
        )
        if str(out or "").strip():
            return str(out)
    return ""



def _strength_breakdown(
    *,
    art: MLArtifact,
    team: str,
    as_of: date,
    team_meta: Dict[str, Dict[str, str]],
    ach_layer: Dict[str, Any],
    league_hint: Optional[str] = None,
    tier_hint: Optional[str] = None,
) -> Dict[str, Any]:
    return core_strength_breakdown(
        art=art,
        team=team,
        as_of=as_of,
        team_meta=team_meta,
        ach_layer=ach_layer,
        norm_league_fn=core_norm_league,
        tier_from_league_fn=core_tier_from_league,
        tier_base_points_fn=tier_base_points,
        league_base_points_fn=league_base_points,
        team_boost_points_fn=team_boost_points,
        league_hint=league_hint,
        tier_hint=tier_hint,
    )


def _mlcore_p_map_v2(
    artifact_path: str,
    team_blue: str,
    team_red: str,
    as_of: date,
    ach_override_path: Optional[str] = None,
    league_hint_blue: Optional[str] = None,
    league_hint_red: Optional[str] = None,
    tier_hint_blue: Optional[str] = None,
    tier_hint_red: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return core_mlcore_p_map_v2(
        artifact_path=artifact_path,
        team_blue=team_blue,
        team_red=team_red,
        as_of=as_of,
        ach_override_path=ach_override_path,
        league_hint_blue=league_hint_blue,
        league_hint_red=league_hint_red,
        tier_hint_blue=tier_hint_blue,
        tier_hint_red=tier_hint_red,
        clip_prob_fn=clip_prob,
        artifact_sig_fn=_artifact_sig,
        load_artifact_cached_fn=_load_artifact_cached,
        team_meta_from_artifact_fn=_team_meta_from_artifact,
        ach_layer_for_artifact_fn=_ach_layer_for_artifact,
        strength_breakdown_fn=_strength_breakdown,
        p_from_diff_points_fn=_p_from_diff_points,
    )




@st.cache_data(show_spinner=False)
def _build_ml_microseg_table(
    csv_sig: Tuple[str, int, int],
    art_sig: Tuple[str, int, int],
    lookback_games: int = 1200,
    shrink_k: float = 90.0,
) -> Dict[str, Dict[str, float]]:
    return _build_ml_microseg_table_core(
        _load_team_games(csv_sig),
        _load_artifact_cached(art_sig),
        lookback_games=lookback_games,
        shrink_k=shrink_k,
    )


def _apply_ml_microseg_calibration(
    p_map_cal: float,
    *,
    league: str,
    split: str,
    side: str,
    bo: str,
    phase: str,
    table,
    min_n: int,
    strength: float,
    cap_pp: float,
    hierarchical: bool = True,
    hierarchy_decay: float = 0.80,
):
    return _apply_ml_microseg_calibration_core(
        p_map_cal,
        clip_prob_fn=clip_prob,
        league=league,
        split=split,
        side=side,
        bo=bo,
        phase=phase,
        table=table,
        min_n=min_n,
        strength=strength,
        cap_pp=cap_pp,
        hierarchical=hierarchical,
        hierarchy_decay=hierarchy_decay,
    )




def _apply_ml_uncertainty_adjust(
    p_in: float,
    ml_info,
    *,
    enabled: bool = True,
    strength: float = 0.55,
    prior_games: float = 22.0,
    max_shrink: float = 0.65,
):
    return _apply_ml_uncertainty_adjust_core(
        p_in,
        ml_info,
        clip_prob_fn=clip_prob,
        enabled=enabled,
        strength=strength,
        prior_games=prior_games,
        max_shrink=max_shrink,
    )


def _apply_ml_marginal_trust_adjust(
    p_in: float,
    ml_info,
    *,
    league: str = "",
    enabled: bool = True,
    strength: float = 0.70,
    prior_games: float = 28.0,
    max_shrink: float = 0.72,
    cap_prob: float = 0.84,
    confidence_score=None,
    score_soft: float = 68.0,
    score_hard: float = 46.0,
    exclude_leagues=None,
    band_tuning: bool = True,
    band_coeffs=None,
):
    return _apply_ml_marginal_trust_adjust_core(
        p_in,
        ml_info,
        clip_prob_fn=clip_prob,
        norm_league_key_fn=core_norm_league_key,
        tier_from_league_fn=core_tier_from_league,
        ml_league_conf_band_fn=core_ml_league_conf_band,
        ml_marginal_band_profile_fn=core_ml_marginal_band_profile,
        league_is_excluded_fn=core_league_is_excluded,
        is_marginal_league_for_ml_fn=core_is_marginal_league_for_ml,
        sanitize_ml_marginal_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
        league=league,
        enabled=enabled,
        strength=strength,
        prior_games=prior_games,
        max_shrink=max_shrink,
        cap_prob=cap_prob,
        confidence_score=confidence_score,
        score_soft=score_soft,
        score_hard=score_hard,
        exclude_leagues=exclude_leagues if exclude_leagues is not None else "",
        band_tuning=band_tuning,
        band_coeffs=band_coeffs,
    )

# -----------------------------
# ML engine alternativo: Elo season (time) + Elo global (players)
# -----------------------------


@st.cache_data(show_spinner=False)
def _load_player_rows_multi_cached(sig_list: Tuple[Tuple[str, int, int], ...]) -> pd.DataFrame:
    return core_load_player_rows_multi(
        sig_list=sig_list,
        load_player_rows_fn=_load_player_rows,
    )



@st.cache_data(show_spinner=False)
def _compute_player_elo_global_cached(
    sig_list: Tuple[Tuple[str, int, int], ...],
    *,
    half_life_days: float,
    k_player: float,
    shrink_m: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    return core_compute_player_elo_global(
        sig_list=sig_list,
        load_player_rows_multi_fn=_load_player_rows_multi_cached,
        half_life_days=half_life_days,
        k_player=k_player,
        shrink_m=shrink_m,
        base_rating=base_rating,
        scale=scale,
    )




def _elo_season_players_p_map(
    *,
    team_games: pd.DataFrame,
    csv_path: str,
    teamA: str,
    teamB: str,
    filters: Filters,
    team_half_life_days: float,
    player_half_life_days: float,
    k_team: float,
    k_player: float,
    shrink_m: float,
    w_players: float,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, Any]:
    return _elo_season_players_p_map_core(
        team_games=team_games,
        csv_path=csv_path,
        teamA=teamA,
        teamB=teamB,
        filters=filters,
        team_half_life_days=team_half_life_days,
        player_half_life_days=player_half_life_days,
        k_team=k_team,
        k_player=k_player,
        shrink_m=shrink_m,
        w_players=w_players,
        base_rating=base_rating,
        scale=scale,
        ctx=globals(),
        st=st,
    )



def _mlcore_rank_table_v2(artifact_path: str, as_of: date, top_n: int = 200) -> pd.DataFrame:
    return _mlcore_rank_table_v2_core(artifact_path, as_of, top_n, ctx=globals())




# -----------------------------
# Linhas por confronto (persistem enquanto os times não mudam)
# -----------------------------


# Persistência das linhas (para não perder ao fechar o app)
_LINES_STORE_DEFAULTS_KEY = "__defaults__"  # padrão global (fallback quando o confronto não tem linhas salvas)
_LINES_STORE_FILE = Path(__file__).resolve().parent / "user_lines_store.json"


def _load_lines_store_disk() -> dict:
    return load_json_store(_LINES_STORE_FILE)


def _persist_lines_store_disk(store: dict) -> None:
    persist_json_store(_LINES_STORE_FILE, store)


def _lines_store() -> dict:
    # Session-only: NOT seeded from disk. Every new session starts with no saved
    # lines, so each matchup opens with the standard default lines. The store
    # still carries the lines the user types from the fill step into the Resumo
    # within the SAME analysis. (User decision 2026-06-02: "sempre padrão".)
    return ensure_session_store(st.session_state, "_lines_store", {})


def _load_lines_for_matchup(key: str) -> None:
    _load_lines_for_matchup_impl(st.session_state, _lines_store(), key, defaults_key=_LINES_STORE_DEFAULTS_KEY)


def _save_lines_for_matchup(key: str) -> None:
    # Update the session store only — no disk persistence (lines are session-only).
    store = _lines_store()
    _save_lines_for_matchup_impl(st.session_state, store, key)


# -----------------------------
# Lineup (Players) — store + scoring (ajuste do ML)
# -----------------------------


def _load_lineup_for_matchup(key: str) -> None:
    _load_lineup_for_matchup_impl(st.session_state, key)


def _save_lineup_for_matchup(key: str) -> None:
    _save_lineup_for_matchup_impl(st.session_state, key)


def _ensure_lineup_runtime_state() -> Dict[str, Any]:
    return _ensure_lineup_runtime_state_impl(st.session_state)


def _get_lineup_runtime_settings() -> Dict[str, Any]:
    return _get_lineup_runtime_settings_impl(st.session_state)


def _get_lineup_loaded_key() -> str:
    return _get_lineup_loaded_key_impl(st.session_state)


def _set_lineup_loaded_key(key: str) -> str:
    return _set_lineup_loaded_key_impl(st.session_state, key)


def _get_lineup_selected(side: str, *, mode: str = "manual") -> dict:
    return _get_lineup_selected_impl(st.session_state, side, mode=mode)


@st.cache_data(show_spinner=False)
def _load_players_impact(players_sig: tuple) -> pd.DataFrame:
    csv_path = str(players_sig[0])
    return core_load_players_impact(
        csv_path=csv_path,
        norm_pos_fn=_norm_pos,
        parse_pct_fn=_parse_pct,
    )


_apply_shrink_m = core_apply_shrink_m
_best_match = core_best_match


def _latest_lineup_for_team(
    df_player_rows: pd.DataFrame,
    team: str,
    filters: Filters,
    league_fixed: str | None = None,
    *,
    pos_col: str = "pos",
    player_col: str = "player",
    pos_list: list[str] | None = None,
) -> dict:
    return core_latest_lineup_for_team(
        df_player_rows=df_player_rows,
        team=team,
        filters=filters,
        league_fixed=league_fixed,
        pos_col=pos_col,
        player_col=player_col,
        pos_list=(pos_list or list(_POS_CANON)),
        norm_pos_fn=_norm_pos,
        norm_pos_ml_fn=_norm_pos_ml,
    )


def _calibrate_p_from_artifact(artifact_path: str, p_raw: float) -> float:
    return _calibrate_p_from_artifact_core(artifact_path, p_raw, ctx=globals())




# -----------------------------
# Ranking (força tipo ML) - Elo com recência (half-life)
# -----------------------------

def _elo_rank_table(
    team_games: pd.DataFrame,
    *,
    filters: Filters,
    half_life_days: float,
    k_factor: float = 24.0,
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> pd.DataFrame:
    return _elo_rank_table_core(
        team_games,
        filters=filters,
        half_life_days=half_life_days,
        k_factor=k_factor,
        base_rating=base_rating,
        scale=scale,
        format_pct_fn=_format_pct,
        format_num_fn=_format_num,
    )
# -----------------------------
# Helpers de exibição (tirar colunas irrelevantes)
# -----------------------------

def _df_contract_view(
    df: pd.DataFrame | None,
    required_cols: list[str],
    *,
    optional_cols: list[str] | None = None,
    context: str = "",
    fill_value: Any = "-",
    warn_on_missing: bool = True,
) -> pd.DataFrame:
    return core_df_contract_view(
        df=df,
        required_cols=required_cols,
        optional_cols=optional_cols,
        context=context,
        fill_value=fill_value,
        warn_on_missing=warn_on_missing,
        warning_fn=st.warning,
    )

def _display_ou_table(df: pd.DataFrame) -> pd.DataFrame:
    return core_display_ou_table(
        df,
        df_contract_view_fn=_df_contract_view,
        format_pct_fn=_format_pct,
        format_num_fn=_format_num,
    )


def _display_ml_totals_table(df: pd.DataFrame, team_label: Optional[str] = None) -> pd.DataFrame:
    return core_display_ml_totals_table(
        df,
        df_contract_view_fn=_df_contract_view,
        format_pct_fn=_format_pct,
        format_num_fn=_format_num,
        team_label=team_label,
    )


def _render_odds_cards(df_raw: pd.DataFrame, mode: str = 'ou', table_label: str = 'Ver tabela') -> None:
    return _render_odds_cards_core(df_raw, mode, table_label, st=st)


def _lines_table(metric: str, lines_list: List[float], total_obj) -> pd.DataFrame:
    return _lines_table_core(metric, lines_list, total_obj, st=st)


def _lines_table_with_p_ml(metric: str, lines_list: List[float], total_obj, p_ml_value: float) -> pd.DataFrame:
    return _lines_table_with_p_ml_core(metric, lines_list, total_obj, p_ml_value, st=st)



def _fuse_probs_by_precision(
    *,
    p_team: float,
    p_players: float,
    n_team: float,
    n_players: float,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    coverage: Optional[float] = None,
    transfer_signal: float = 0.0,
    lineup_continuity: Optional[float] = None,
    roster_changed: bool = False,
    season_phase: str = "auto",
) -> Dict[str, Any]:
    nt = float(n_team) if math.isfinite(float(n_team)) else 0.0
    np_ = float(n_players) if math.isfinite(float(n_players)) else 0.0
    ts = float(team_scale) if math.isfinite(float(team_scale)) else 1.0
    ps = float(players_scale) if math.isfinite(float(players_scale)) else 1.0
    return core_fuse_with_coherence_guard(
        p_team=float(p_team),
        p_players=float(p_players),
        n_team=float(max(0.0, nt)),
        n_players=float(max(0.0, np_)),
        team_scale=float(max(0.0, ts)),
        players_scale=float(max(0.0, ps)),
        coverage=coverage,
        transfer_signal=float(transfer_signal),
        lineup_continuity=lineup_continuity,
        roster_changed=bool(roster_changed),
        season_phase=str(season_phase or "auto"),
        team_weight_early=float(st.session_state.get("fusion_team_weight_early", 0.58) or 0.58),
        team_weight_mid=float(st.session_state.get("fusion_team_weight_mid", 0.75) or 0.75),
        team_weight_playoffs=float(st.session_state.get("fusion_team_weight_playoffs", 0.82) or 0.82),
        early_season_team_weight=float(st.session_state.get("fusion_early_season_team_weight", 0.55) or 0.55),
        season_knee_games=float(st.session_state.get("fusion_season_knee_games", 18.0) or 18.0),
        coverage_power=float(st.session_state.get("fusion_coverage_power", 1.5) or 1.5),
        transfer_boost=float(st.session_state.get("fusion_transfer_boost", 0.35) or 0.35),
        divergence_pp_cap=float(st.session_state.get("fusion_divergence_pp_cap", 22.0) or 22.0),
        divergence_low_coverage=float(st.session_state.get("fusion_divergence_low_coverage", 0.30) or 0.30),
        divergence_shrink=float(st.session_state.get("fusion_divergence_shrink", 0.35) or 0.35),
    )



# _format_pct (fmt_prob_pct), _format_num (fmt_num) imported from app_core.ui_labels





# _totals_table imported from app_core.display_tables_core

def _series_totals_table(tot_avg) -> pd.DataFrame:
    return _series_totals_table_core(tot_avg, st=st, ctx=globals())


# -----------------------------
# UI (fluxo: Times -> Linhas -> Resultados)
# -----------------------------
st.set_page_config(page_title="Gabinete Dota", page_icon="🛡️", layout="wide")
_inject_global_ui_css(st)
from dota_ui import inject_global_ui_css as _dota_inject_css
_dota_inject_css(st)

# Hydrate session_state from persisted preferences (silent on first run)
try:
    from app_core.ui_state_persistence_core import sync_session_with_persistence as _sync_persist
    _sync_persist(st.session_state)
except Exception:
    pass
st.markdown(
    """
<div class="gb-hero">
  <div class="title">GABINETE <span style="color:#c94a38;">DOTA</span></div>
  <div class="sub">dev: Bolin</div>
</div>
    """,
    unsafe_allow_html=True,
)
st.caption("Fluxo: escolher Season/Times -> preencher linhas -> ver resultados (modelo + histórico).")

APP_ROOT = Path(__file__).resolve().parent
_FIT_PARAMS_PATH = APP_ROOT / "params_fitted.json"
_ODDS_ANCHOR_BASELINE_PATH = APP_ROOT / "odds_anchor_baseline.json"
_TEAM_STATE_ARTIFACT_PATH = APP_ROOT / "team_state_artifact.json"
_ML_CONFIDENCE_BASELINE_PATH = APP_ROOT / "ml_confidence_baseline.json"
_LEAGUE_TRANSFER_FEATURES_PATH = APP_ROOT / "league_transfer_features.json"
_WF_LAST_REPORT_PATH = APP_ROOT / "wf_last_report.json"
_LV_LAST_REPORT_PATH = APP_ROOT / "lv_last_report.json"
_CSV_BOOTSTRAP_MAINT_CACHE_PATH = APP_ROOT / "csv_bootstrap_maintenance_cache.json"
_CSV_BOOTSTRAP_MAINT_LOCK_PATH = APP_ROOT / "csv_bootstrap_maintenance.lock.json"
_CSV_BOOTSTRAP_MAINT_STATE_PATH = APP_ROOT / "csv_bootstrap_maintenance_state.json"


# _compute_anchor_snapshot imported from app_core.anchor_snapshot_core


def _persist_combined_trace_file() -> None:
    """Persist unified trace (ml base + resumo markets + players) into ml_trace_last.json."""
    try:
        core_persist_combined_trace_file(
            st.session_state,
            app_root=APP_ROOT,
            build_combined_trace_fn=core_build_combined_trace,
            app_version="gabinete-2026.02.12",
        )
    except (OSError, TypeError, ValueError):
        pass

_fit_cfg = {}
try:
    if _FIT_PARAMS_PATH.exists():
        _fit_cfg = _load_fitted_params_cached(_file_sig(str(_FIT_PARAMS_PATH))) or {}
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    _fit_cfg = {}

_fit_form_cfg: Dict[str, Any] = {}
_fit_form_cfg = core_apply_fit_runtime_state_to_session(
    st.session_state,
    _fit_cfg,
    build_fit_runtime_state_fn=core_build_fit_runtime_state,
    sanitize_marginal_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
)

def _fit_runtime_snapshot() -> Dict[str, Any]:
    return core_get_fit_runtime_snapshot(st.session_state)

def _fit_scales_for_league(league_hint: str) -> tuple[float, float, str]:
    _fit_rt = _fit_runtime_snapshot()
    return core_fit_scales_for_league(
        _fit_cfg,
        float(_fit_rt.get("_fit_team_scale", 1.0) or 1.0),
        float(_fit_rt.get("_fit_players_scale", 1.0) or 1.0),
        league_hint,
    )


def _fit_form_for_league(league_hint: str) -> tuple[Dict[str, Any], str]:
    return core_fit_form_for_league(
        _fit_cfg,
        _fit_form_cfg,
        league_hint,
        league_macro_key_fn=_league_macro_key,
    )


def _fit_players_lane_for_league(league_hint: str) -> tuple[Dict[str, float], str]:
    _fit_rt = _fit_runtime_snapshot()
    return core_fit_players_lane_for_league(
        _fit_rt.get("_fit_lane_weights_cfg"),
        league_hint,
        league_macro_key_fn=_league_macro_key,
    )


def _fit_players_league_weight_for_league(league_hint: str) -> tuple[float, str]:
    _fit_rt = _fit_runtime_snapshot()
    return core_fit_players_league_weight_for_league(
        _fit_rt.get("_fit_players_league_weights_cfg"),
        league_hint,
        league_macro_key_fn=_league_macro_key,
    )


def _fit_players_context_for_league(league_hint: str) -> tuple[Dict[str, Any], str]:
    _fit_rt = _fit_runtime_snapshot()
    return core_fit_players_context_for_league(
        _fit_rt.get("_fit_players_context_cfg"),
        league_hint,
        league_macro_key_fn=_league_macro_key,
    )



def _apply_fused_post_calibration(
    p_map_cal: float,
    *,
    league_hint: str,
    use_fitted: bool = True,
) -> tuple[float, Dict[str, Any]]:
    _fit_rt = _fit_runtime_snapshot()
    return core_apply_fused_post_calibration(
        p_map_cal,
        _fit_rt.get("_fit_fused_post_cal_cfg"),
        league_hint=league_hint,
        use_fitted=use_fitted,
    )


def _series_gamma_for_league(league_hint: str, bo: int) -> tuple:
    return _series_gamma_for_league_core(league_hint, bo, st=st, ctx=globals())


def _load_confidence_baseline_runtime() -> Dict[str, Any]:
    return _load_confidence_baseline_runtime_core(ctx=globals())


def _runtime_confidence_governor_budget_for_league(
    league_hint: str,
    confidence_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _runtime_confidence_governor_budget_for_league_core(league_hint, confidence_baseline, st=st, ctx=globals())


def _runtime_series_coherence_post_cap_for_league(
    *,
    league_hint: str,
    bo: int,
    confidence_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _runtime_series_coherence_post_cap_for_league_core(league_hint=league_hint, bo=bo, confidence_baseline=confidence_baseline, st=st, ctx=globals())


def _boot_select_csvs(root: Path) -> None:
    core_render_bootstrap_selector(
        session_state=st.session_state,
        root=root,
        list_csv_candidates_fn=_list_csv_candidates,
        apply_bootstrap_selection_fn=core_apply_bootstrap_selection,
    )




# -----------------------------
# Boot: carregar config salva + sidebar (antes do fluxo)
# -----------------------------
_init_app_settings_once()
_render_settings_sidebar()

# Deferred save used by one-click profile actions in tuning UI.
if bool(st.session_state.pop("_pending_save_settings_after_apply", False)):
    _save_app_settings_to_disk()
    core_set_flash_saved_profile_msg(st.session_state, "Perfil aplicado e salvo como padrao.")


# Tela inicial (antes de qualquer fluxo)
_boot_select_csvs(APP_ROOT)

# Navegacao (centraliza calibracao/retreino em um lugar so)
_nav_options = [o for o in NAV_TOP_NAV_OPTIONS if o not in ("Radar", "Meta")]
_nav_main_cur = core_resolve_nav_page(st.session_state)
if str(st.session_state.get("nav_main", "") or "") not in _nav_options:
    st.session_state["nav_main"] = _nav_main_cur if _nav_main_cur in _nav_options else _nav_options[0]
nav_page = st.radio("Pagina", _nav_options, horizontal=True, key="nav_main")

def _reset_flow(keep_config: bool = True):
    core_reset_flow_state(st.session_state, keep_config=bool(keep_config))



# Aplica carregamentos pendentes ANTES de instanciar widgets (evita erro de session_state em widgets como BO)
core_apply_pending_load(st.session_state, reset_flow_fn=_reset_flow)
# -----------------------------
# Configurações (raramente muda)
# -----------------------------
seasons = _discover_season_csvs(APP_ROOT)
_runtime_cfg_defaults = core_build_runtime_config_defaults(st.session_state, app_root=APP_ROOT, seasons=seasons)

if nav_page == "Tecnico":
    core_render_technical_runtime_config_panel(
        session_state=st.session_state,
        seasons=seasons,
        runtime_cfg_defaults=_runtime_cfg_defaults,
        request_csv_runtime_refresh_fn=core_request_csv_runtime_refresh,
        reset_bootstrap_selection_fn=core_reset_bootstrap_selection,
    )




def _paths_for_ml(root: Path) -> tuple[str, list[str], str]:
    return core_resolve_ml_paths(st.session_state, root)


def _resolve_players_artifact_path(artifact_path: str, root: Path) -> str:
    return core_resolve_players_artifact_path(
        artifact_path,
        root,
        load_artifact_fn=lambda p: _load_artifact_cached(_artifact_sig(str(p))),
    )


def _feature_flag_enabled(flag_key: str, default: bool = False) -> bool:
    """Safe feature-flag read from session_state (tolerates stale/invalid values)."""
    return bool(core_coerce_feature_flag(st.session_state.get(str(flag_key), default), default=default))


def _append_feature_trace(
    *,
    feature: str,
    event: str,
    enabled=None,
    payload: Optional[Dict[str, Any]] = None,
    dedupe_signature: Optional[str] = None,
    max_items: int = 200,
    persist_files: bool = False,
) -> None:
    return _append_feature_trace_core(
        feature=feature, event=event, enabled=enabled, payload=payload,
        dedupe_signature=dedupe_signature, max_items=max_items,
        persist_files=persist_files, st=st, ctx=globals(),
    )



def _resolve_runtime_players_artifact_path(artifact_path: str) -> str:
    return _resolve_runtime_players_artifact_path_core(artifact_path, ctx=globals())


def _players_artifact_output_path(artifact_path: str) -> Path:
    resolved = Path(_resolve_runtime_players_artifact_path(str(artifact_path))).expanduser()
    try:
        if resolved.exists():
            return resolved
    except OSError:
        pass
    artifact_raw = str(artifact_path or "").strip()
    if artifact_raw:
        return Path(artifact_raw).expanduser().with_name("players_artifact.json")
    return APP_ROOT / "players_artifact.json"


def _collect_maintenance_status(csv_path_s: str, artifact_path_s: str) -> Dict[str, Any]:
    return _collect_maintenance_status_flow(csv_path_s, artifact_path_s, st=st, ctx=globals())


def _run_auto_maintenance_now(
    csv_path_s: str,
    artifact_path_s: str,
    status=None,
    *,
    force_full: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    return _run_auto_maintenance_now_flow(csv_path_s, artifact_path_s, status, force_full=force_full, progress_cb=progress_cb, st=st, ctx=globals())


def _read_artifact_summary(artifact_path: str) -> dict:
    return dict(core_read_artifact_summary_path(artifact_path))


def _run_cmd(
    cmd,
    *,
    timeout_s=None,
    heartbeat_cb=None,
    heartbeat_interval_s: float = 2.0,
):
    return _run_cmd_core(cmd, timeout_s=timeout_s, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)



def _read_json_file_safe(path_like: str | Path) -> Dict[str, Any]:
    return dict(core_read_json_file_safe(path_like))






def _enrich_params_with_ml_marginal_band_coeffs(params_obj: Any, *, root: Path) -> Tuple[Dict[str, Any], str]:
    return core_enrich_params_with_ml_marginal_band_coeffs(
        params_obj,
        root=root,
        sanitize_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
    )




def _backup_file_if_exists(path_like, *, tag: str = "manual") -> str:
    return _backup_file_if_exists_core(path_like, tag=tag, app_root=APP_ROOT)

@st.cache_data(show_spinner=False, ttl=1800)
def _run_walkforward_cached(
    csv_paths: tuple[str, ...],
    min_train_games: int,
    k: float,
    scale: float,
    artifact_path: str = "",
    players_artifact_path: str = "",
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    bins: int = 10,
    min_league_games: int = 250,
    params_fitted_path: str = "",
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_walkforward(
        clean,
        min_train_games=int(min_train_games),
        k=float(k),
        scale=float(scale),
        artifact_path=str(Path(str(artifact_path)).expanduser()) if str(artifact_path or "").strip() else "",
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()) if str(players_artifact_path or "").strip() else "",
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        bins=int(bins),
        min_league_games=int(min_league_games),
        params_fitted_path=str(Path(str(params_fitted_path)).expanduser()) if str(params_fitted_path or "").strip() else "",
    )


@st.cache_data(show_spinner=False, ttl=1800)
def _run_layer_validation_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    min_train_games: int,
    team_scale: float,
    players_scale: float,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_layer_validation(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        min_train_games=int(min_train_games),
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        min_league_games=int(min_league_games),
    )


@st.cache_data(show_spinner=False, ttl=1800)
def _run_calibration_diag_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    team_scale: float,
    players_scale: float,
    bins: int,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_calibration_diagnostics(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        bins=int(bins),
        min_league_games=int(min_league_games),
    )




def _generate_and_persist_ml_confidence_baseline(
    *,
    root: Path,
    lv_report: Optional[Dict[str, Any]],
    wf_report: Optional[Dict[str, Any]],
    params_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _generate_and_persist_ml_confidence_baseline_core(root=root, lv_report=lv_report, wf_report=wf_report, params_fit=params_fit)


def _generate_and_persist_team_state_artifact(
    *,
    root: Path,
    csv_paths: List[str],
    keep_playoffs: Optional[bool] = None,
) -> Dict[str, Any]:
    return _generate_and_persist_team_state_artifact_core(root=root, csv_paths=csv_paths, keep_playoffs=keep_playoffs, ctx=globals())


def _run_daily_pipeline(
    root: Path,
    *,
    csv_path_s: str,
    artifact_path_s: str,
    run_smoke: bool = True,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    return core_run_daily_pipeline(
        st.session_state,
        root,
        csv_path_s=csv_path_s,
        artifact_path_s=artifact_path_s,
        run_smoke=bool(run_smoke),
        fast_mode=bool(fast_mode),
        baseline_path=_ODDS_ANCHOR_BASELINE_PATH,
        paths_for_ml_fn=_paths_for_ml,
        resolve_players_artifact_path_fn=_resolve_players_artifact_path,
        compute_anchor_snapshot_fn=_compute_anchor_snapshot,
        fit_ml_params_run_fn=_fit_ml_params_run,
        enrich_params_with_areal_prob_tuning_fn=_enrich_params_with_areal_prob_tuning,
        run_layer_validation_cached_fn=_run_layer_validation_cached,
        run_calibration_diag_cached_fn=_run_calibration_diag_cached,
        run_walkforward_cached_fn=_run_walkforward_cached,
        load_fitted_params_cached_fn=_load_fitted_params_cached,
        file_sig_fn=_file_sig,
        generate_ml_confidence_baseline_fn=_generate_and_persist_ml_confidence_baseline,
        generate_team_state_artifact_fn=_generate_and_persist_team_state_artifact,
        run_pre_delivery_checks_fn=_run_pre_delivery_checks,
        clear_team_state_cache_fn=lambda: _load_team_state_artifact_cached.clear(),
    )



def _wf_correct_p_map(
    report,
    p_in: float,
    league_hint: str = "",
    strength: float = 0.60,
):
    return _wf_correct_p_map_core(report, p_in, league_hint=league_hint, strength=strength)



# _ml_confidence_from_layer_validation imported from app_core.ml_confidence_scoring_core


# Página Técnico (só aqui tem drift/retreino/update)
if nav_page == "Tecnico":
    _render_calibracoes(APP_ROOT, st=st, ctx=globals())
    st.stop()

# -----------------------------
# Etapa 1 — escolher Season + Times
# -----------------------------
_season_ctx = core_resolve_season_context(seasons, selected_label=str(st.session_state.get("season_label", "") or ""))
_season_options = list(_season_ctx.get("options") or ["S15"])
_season_default = _season_options[int(_season_ctx.get("selected_idx", 0) or 0)] if _season_options else "S15"
if str(st.session_state.get("season_label", "") or "") not in _season_options:
    st.session_state["season_label"] = _season_default
season_label = st.selectbox(
    "Temporada",
    _season_options,
    key="season_label",
)
_season_ctx = core_resolve_season_context(seasons, selected_label=str(season_label or ""))
season_info = dict(_season_ctx.get("season_info") or {})
season_year = int(_season_ctx.get("season_year", 2025) or 2025)

# Decide caminhos ativos
csv_path, artifact_path = core_resolve_active_csv_and_artifact(
    st.session_state,
    app_root=APP_ROOT,
    season_label=season_label,
    season_year=season_year,
    season_csv_path=str(season_info.get("csv_path") or ""),
    seasons_available=bool(_season_ctx.get("has_seasons", False)),
)

# Tela separada de preparo do CSV + manutenção automática quando houver troca.
_csv_bootstrap_gate(
    csv_path=csv_path,
    hist_csvs=core_get_boot_hist_csvs(st.session_state),
    artifact_path=artifact_path,
)

# Alerta global visível: drift/falta de parâmetros/calibração.
_render_global_maintenance_banner(st, csv_path, artifact_path, collect_status_fn=_collect_maintenance_status)
# Alerta global visível: observabilidade/shadow features (se houver falha de trace).
_render_global_feature_trace_banner(st)
# Alerta global visível: NOTA do ML caiu vs baseline (escrito pelo qualidade_alarme da manutenção).
core_render_ml_quality_alarm(st=st, root=APP_ROOT)

# Achievements path default (para ML core v2 e aba Vencedores)
# - Mantém em sessão para que o ML use o mesmo arquivo que você edita na aba "Vencedores"
core_ensure_achievement_path(
    session_state=st.session_state,
    artifact_path=str(artifact_path),
    app_root=APP_ROOT,
    get_achievement_path_fn=core_get_achievement_path,
    set_achievement_path_fn=core_set_achievement_path,
    load_artifact_fn=_load_artifact_cached,
    artifact_sig_fn=_artifact_sig,
    resolve_relpath_fn=_resolve_relpath,
)


# Se trocou season, reseta fluxo (pra não misturar times/linhas)
core_apply_season_change_reset(
    session_state=st.session_state,
    season_label=season_label,
    get_last_season_label_fn=core_get_last_season_label,
    set_last_season_label_fn=core_set_last_season_label,
    reset_flow_fn=_reset_flow,
    clear_matchup_selection_fn=core_clear_matchup_selection,
)

csv_sig = _file_sig(csv_path)

# Carrega times — APENAS os do CSV ativo (season atual / 2026). Times que só
# jogaram em 2024/2025 (ex.: LDL, sem dado em 2026) NÃO entram nos seletores,
# mesmo o modelo tendo rating antigo deles — a pedido: só times da season atual.
try:
    teams = _list_teams(csv_sig)
except (OSError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
    st.error(f"Falha ao ler CSV para listar times: {e}")
    st.stop()

# Normaliza grafias (ex.: "Dplus Kia" vs "Dplus KIA") para evitar cair na liga errada no modo auto.
core_normalize_matchup_selection_teams(
    session_state=st.session_state,
    csv_sig=csv_sig,
    team_canon_maps_fn=_team_canon_maps,
    canon_team_key_fn=_canon_team_key,
    ensure_matchup_selection_state_fn=core_ensure_matchup_selection_state,
    set_selected_team_fn=core_set_selected_team,
)

if nav_page == "Live-Bet":
    team_games = _load_team_games(csv_sig)
    years = sorted([int(y) for y in team_games["year"].dropna().unique().tolist()]) if team_games is not None and not team_games.empty else [int(season_year)]
    _results_runtime_live = core_resolve_results_runtime_settings(
        st.session_state,
        season_year=season_year,
        years=years,
    )
    _live_year_opt = _results_runtime_live["year_opt"]
    _live_split_opt = _results_runtime_live["split_opt"]
    _live_playoffs_opt = _results_runtime_live["playoffs_opt"]
    _live_fixed_league = _results_runtime_live["fixed_league"]
    _live_filters = Filters(
        year=None if _live_year_opt == "All" else int(_live_year_opt),
        league=None,
        split=None if _live_split_opt == "All" else str(_live_split_opt),
        playoffs=None if _live_playoffs_opt == "all" else (True if _live_playoffs_opt == "true" else False),
    )
    if _results_runtime_live["league_mode"] == "fixed" and _live_fixed_league and _live_fixed_league != "All":
        _live_filters = Filters(
            year=_live_filters.year,
            league=_live_fixed_league,
            split=_live_filters.split,
            playoffs=_live_filters.playoffs,
        )

    _team_a_live = str(st.session_state.get("teamA") or (teams[0] if teams else ""))
    _team_b_live = str(st.session_state.get("teamB") or "")
    if not _team_b_live or _team_b_live == _team_a_live:
        _team_b_live = next((t for t in teams if str(t) != _team_a_live), _team_a_live)

    core_render_live_bet_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "Filters": Filters,
            "active_filters": _live_filters,
            "apply_filters_fn": apply_filters,
            "teamA": _team_a_live,
            "teamB": _team_b_live,
            "csv_sig": csv_sig,
            "state_get_str_fn": state_get_str,
            "load_player_rows_fn": _load_player_rows,
            "apply_filters_players_fn": core_apply_filters_players,
            "champion_vocab_fn": core_champion_vocab,
            "fuzzy_match_one_fn": core_fuzzy_match_one,
            "format_num_fn": _format_num,
            "min_to_mmss_fn": _min_to_mmss,
            "artifact_path": str(artifact_path),
        },
    )
    st.stop()


if nav_page == "Radar":
    team_games = _load_team_games(csv_sig)
    core_render_radar_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "Filters": Filters,
            "apply_filters_fn": apply_filters,
            "csv_sig": csv_sig,
            "load_player_rows_fn": _load_player_rows,
            "format_num_fn": _format_num,
            "min_to_mmss_fn": _min_to_mmss,
            "artifact_path": str(artifact_path),
        },
    )
    st.stop()


if nav_page == "Meta":
    core_render_meta_tab(st=st, ctx={"csv_sig": csv_sig})
    st.stop()


# Se pediu reset manual do confronto, aplica ANTES dos widgets (evita erro de session_state)
if core_consume_pending_reset_matchup(st.session_state):
    _reset_flow(keep_config=True)
    core_clear_matchup_selection(st.session_state)

flow_stage = core_get_flow_stage(st.session_state)


# -----------------------------
# Jogos (agenda) - adicionar por colagem (na página Plays)
# -----------------------------
_schedule_store_path = APP_ROOT / "user_schedule_store.json"
core_restore_schedule_runtime_state(st.session_state, _schedule_store_path)
core_bind_schedule_store_path(st.session_state, _schedule_store_path)
core_get_schedule_games(st.session_state)
core_ensure_schedule_alias_state(st.session_state)
core_set_schedule_unknown_aliases(st.session_state, core_has_schedule_unknown_aliases(st.session_state))

import time as _dota_time
if _dota_time.time() - float(st.session_state.get("dota_diaria_auto_ts", 0.0) or 0.0) > 900.0:
    st.session_state["dota_diaria_auto_ts"] = _dota_time.time()
    try:
        from pandascore_client import upcoming_matches as _dota_up
        from zoneinfo import ZoneInfo as _dota_zi
        _fx = [f for f in _dota_up(hours_ahead=48) if f.get("tier") in ("s", "a", "b")]
        _fx.sort(key=lambda f: str(f.get("scheduled_at") or ""))
        _als = core_load_schedule_aliases(st.session_state, APP_ROOT / "user_team_aliases.json")
        from app_core.schedule_state_core import save_schedule_games as _dota_save_ag
        _games_now = list(core_get_schedule_games(st.session_state))
        _auto, _pend = [], []
        _updated = 0
        _mutated = False
        for _f in _fx:
            _w = pd.Timestamp(_f["scheduled_at"])
            _w = (_w.tz_localize("UTC") if _w.tzinfo is None else _w)
            # jogos que ja comecaram FICAM (usuario consulta info com jogo rolando)
            _wl = _w.tz_convert(_dota_zi("America/Sao_Paulo"))
            _a, _ok_a = core_resolve_schedule_team(str(_f["team_a"]), teams, _als)
            _b, _ok_b = core_resolve_schedule_team(str(_f["team_b"]), teams, _als)
            _psid = str(_f.get("match_id") or "")
            _g = {"date": _wl.strftime("%d/%m"), "time": _wl.strftime("%H:%M"),
                  "league": str(_f.get("league") or ""), "teamA": _a, "teamB": _b,
                  "bo": int(_f.get("bo") or 3) or 3, "lines": "", "ps_id": _psid}
            if not (_ok_a and _ok_b):
                _pend.append({"date": _g["date"], "time": _g["time"], "league": _g["league"],
                              "teamA_raw": str(_f["team_a"]), "teamB_raw": str(_f["team_b"])})
                continue
            # Jogo ADIADO (pedido 13/07): rastreia pelo id do PandaScore —
            # horario mudou -> ATUALIZA a entrada (a hora antiga sai), em vez
            # de duplicar. Rebatida legitima do mesmo par no dia = id novo.
            _hit = None
            if _psid:
                _hit = next((x for x in _games_now
                             if str(x.get("ps_id") or "") == _psid), None)
            if _hit is None:
                _hit = next((x for x in _games_now
                             if not str(x.get("ps_id") or "")
                             and str(x.get("date")) == _g["date"]
                             and str(x.get("teamA")) == _g["teamA"]
                             and str(x.get("teamB")) == _g["teamB"]), None)
            if _hit is not None:
                if (str(_hit.get("time")) != _g["time"] or str(_hit.get("bo")) != str(_g["bo"])
                        or str(_hit.get("ps_id") or "") != _psid):
                    if str(_hit.get("time")) != _g["time"]:
                        _updated += 1
                    _hit["time"], _hit["bo"], _hit["ps_id"] = _g["time"], _g["bo"], _psid
                    _mutated = True
                # limpa duplicatas sem id do mesmo confronto/dia (horas antigas)
                _dups = [x for x in _games_now if x is not _hit
                         and not str(x.get("ps_id") or "")
                         and str(x.get("date")) == _g["date"]
                         and str(x.get("teamA")) == _g["teamA"]
                         and str(x.get("teamB")) == _g["teamB"]]
                if _dups:
                    _games_now = [x for x in _games_now if x not in _dups]
                    _mutated = True
            else:
                _auto.append(_g)
        if _mutated:
            _dota_save_ag(st.session_state, _games_now)
        if _auto:
            core_append_schedule_games(st.session_state, _auto)
        if _pend:
            _pp = core_get_pending_schedule_import(st.session_state)
            _seen = {(str(p.get("date", "")), str(p.get("time", "")), str(p.get("teamA_raw", "")), str(p.get("teamB_raw", ""))) for p in _pp}
            _pp.extend([p for p in _pend if (str(p["date"]), str(p["time"]), str(p["teamA_raw"]), str(p["teamB_raw"])) not in _seen])
            core_save_pending_schedule_import(st.session_state, _pp)
        _parts = []
        if _auto:
            _parts.append(f"{len(_auto)} jogo(s) carregados na agenda")
        if _updated:
            _parts.append(f"{_updated} horário(s) atualizado(s)")
        if _pend:
            _parts.append(f"{len(_pend)} aguardando mapeamento de nomes (abra 'Jogos (agenda)')")
        st.session_state["dota_diaria_auto_msg"] = ("Diária automática: " + "; ".join(_parts) + ".") if _parts else ""
    except Exception as _dota_exc:
        st.session_state["dota_diaria_auto_msg"] = f"Diária automática indisponível: {_dota_exc}"
_dota_msg = str(st.session_state.get("dota_diaria_auto_msg", "") or "")
if _dota_msg:
    (st.warning if "indisponível" in _dota_msg else st.info)(_dota_msg)

# --- painel DIÁRIA: seletor de dia + jogos do dia (nada some ao começar) ---
# Botão SEMPRE visível (pedido 13/07: "imprescindível conseguir ver os
# jogos") — força a busca na hora, ignorando o intervalo do auto-fetch.
_cbt, _cinfo = st.columns([1.4, 5])
with _cbt:
    if st.button("🔄 Procurar jogos", key="dota_diaria_force"):
        st.session_state["dota_diaria_auto_ts"] = 0.0
        st.rerun()
with _cinfo:
    _ts_last = float(st.session_state.get("dota_diaria_auto_ts", 0.0) or 0.0)
    if _ts_last > 0:
        _min_atras = max(0, int((_dota_time.time() - _ts_last) / 60))
        st.caption(f"Agenda buscada há {_min_atras} min (auto a cada 15 min e ao abrir o app).")
    else:
        st.caption("Agenda ainda não buscada nesta sessão.")

_dota_games_all = core_get_schedule_games(st.session_state)
if not _dota_games_all:
    st.info("Sem jogos T1/T2 na agenda — clique em 🔄 Procurar jogos ou aguarde a "
            "próxima janela (busca cobre 48h à frente).")
if _dota_games_all:
    from app_core.schedule_state_core import set_schedule_pending_load as _dota_set_pending
    from app_core.schedule_state_core import save_schedule_games as _dota_save_games
    from zoneinfo import ZoneInfo as _dota_zi2
    _tz_now = pd.Timestamp.now(tz=_dota_zi2("America/Sao_Paulo"))
    _hoje_d = _tz_now.normalize()

    def _dota_parse_ddmm(s):
        try:
            _dd, _mm = str(s).strip().split("/")[:2]
            cand = pd.Timestamp(year=_tz_now.year, month=int(_mm), day=int(_dd), tz=str(_tz_now.tz))
            # virada de ano: dd/mm sem ano — muito no futuro = era do ano passado
            if (cand - _hoje_d).days > 200:
                cand = cand.replace(year=_tz_now.year - 1)
            elif (_hoje_d - cand).days > 200:
                cand = cand.replace(year=_tz_now.year + 1)
            return cand
        except Exception:
            return None

    # Pedido 13/07: dia que PASSOU sai da agenda; seletor mostra só hoje +48h
    # (jogos de hoje seguem visíveis mesmo já iniciados — decisão anterior).
    _keep = []
    for _g in _dota_games_all:
        _dt_g = _dota_parse_ddmm(_g.get("date"))
        if _dt_g is not None and _dt_g < _hoje_d:
            continue
        _keep.append(_g)
    if len(_keep) != len(_dota_games_all):
        _dota_save_games(st.session_state, _keep)
        _dota_games_all = _keep

    _dota_dates = []
    for _g in _dota_games_all:
        _d = str(_g.get("date", "") or "")
        _dt_g = _dota_parse_ddmm(_d)
        if _dt_g is None or _dt_g > _hoje_d + pd.Timedelta(days=2):
            continue
        if _d and _d not in _dota_dates:
            _dota_dates.append(_d)
    _dota_dates.sort(key=lambda s: _dota_parse_ddmm(s) or _hoje_d)
    _hoje = _tz_now.strftime("%d/%m")
if _dota_games_all and _dota_dates:
    _c1, _c2, _c3 = st.columns([1, 3, 1])
    with _c1:
        if str(st.session_state.get("dota_diaria_day", "") or "") not in _dota_dates:
            st.session_state["dota_diaria_day"] = _dota_dates[_dota_dates.index(_hoje) if _hoje in _dota_dates else 0]
        _day = st.selectbox("📅 Dia", _dota_dates, key="dota_diaria_day")
    _day_games = [g for g in _dota_games_all if str(g.get("date", "")) == _day]
    _labels = [f"{g.get('time','')}  {g.get('teamA','')} x {g.get('teamB','')} (BO{g.get('bo','?')})"
               for g in _day_games]
    with _c2:
        _pick = st.selectbox("Jogos do dia", ["", *_labels], key="dota_diaria_pick")
    with _c3:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        _load_clicked = st.button("Carregar", key="dota_diaria_load", width='stretch')
    if _load_clicked and _pick:
        _g = _day_games[_labels.index(_pick)]
        _payload = {"_reset_flow": True}
        if _g.get("teamA") in teams:
            _payload["teamA"] = _g.get("teamA")
        if _g.get("teamB") in teams:
            _payload["teamB"] = _g.get("teamB")
        if str(_g.get("bo")) in ("1", "2", "3", "5"):
            _payload["bo"] = int(_g.get("bo"))
        _dota_set_pending(st.session_state, _payload)
        st.rerun()

with st.expander("Jogos (agenda) - colar / adicionar", expanded=False):
    st.caption("Cole aqui um bloco de jogos (data, hora, time A, time B). O app tenta inferir a liga pelos cabeçalhos e te pede para mapear times desconhecidos.")

    if st.button("🗓️ Buscar jogos do dia (Diária/PandaScore)", key="dota_diaria_fetch"):
        try:
            from pandascore_client import upcoming_matches as _dota_up
            from zoneinfo import ZoneInfo as _dota_zi
            import pandas as _dota_pd
            _fx = [f for f in _dota_up(hours_ahead=48) if f.get("tier") in ("s", "a", "b")]
            _fx.sort(key=lambda f: str(f.get("scheduled_at") or ""))
            _ls, _prev = [], ""
            for _f in _fx:
                _w = _dota_pd.Timestamp(_f["scheduled_at"])
                _w = (_w.tz_localize("UTC") if _w.tzinfo is None else _w).tz_convert(_dota_zi("America/Sao_Paulo"))
                _d = _w.strftime("%d/%m")
                if _d != _prev:
                    _ls.append(_d); _prev = _d
                _ls.extend([_w.strftime("%H:%M"), str(_f["team_a"]), "-", str(_f["team_b"])])
            if _ls:
                st.session_state["plays_schedule_text"] = "\n".join(_ls)
                st.success(f"{len(_fx)} jogo(s) das próximas 36h (tier S/A/B). Agora clique em 'Processar texto'.")
            else:
                st.info("Sem jogos T1/T2 nas próximas 36h.")
        except Exception as _dota_exc:
            st.error(f"Diária indisponível: {_dota_exc}")

    _alias_path = APP_ROOT / "user_team_aliases.json"

    # ---- UI de importação ----
    st.session_state.setdefault("plays_schedule_default_bo", 3)
    default_bo = st.selectbox("BO padrão (para os jogos importados)", [1, 2, 3, 5], key="plays_schedule_default_bo")

    txt = st.text_area(
        "Cole os jogos aqui (formato livre)",
        height=220,
        key="plays_schedule_text",
        placeholder="Ex.:\\nLeague IconLCKFlag Icon\\n22/01\\n05:00\\nBRION\\nHanwha Life Esports\\n-\\n-\\n..."
    )
    core_persist_schedule_runtime_state(st.session_state)

    cA, cB, cC = st.columns([1,1,2])
    with cA:
        parse_clicked = st.button("Processar texto", width='stretch', key="plays_schedule_parse_btn")
    with cB:
        clear_clicked = st.button("Limpar texto", width='stretch', key="plays_schedule_clear_btn")
    with cC:
        if st.button("Limpar agenda (lista atual)", width='stretch', key="plays_schedule_clear_list"):
            core_clear_schedule_games(st.session_state)
            st.success("Agenda limpa.")

    if clear_clicked:
        core_clear_schedule_text(st.session_state)
        st.rerun()


    with st.expander("Adicionar 1 jogo manual", expanded=False):
        st.session_state.setdefault("plays_manual_date", "")
        st.session_state.setdefault("plays_manual_time", "")
        st.session_state.setdefault("plays_manual_league", "")
        st.session_state.setdefault("plays_manual_teamA", "")
        st.session_state.setdefault("plays_manual_teamB", "")
        st.session_state.setdefault("plays_manual_bo", 3)
        cM1, cM2, cM3 = st.columns([1,1,1])
        with cM1:
            m_date = st.text_input("Data (DD/MM)", key="plays_manual_date", placeholder="22/01")
        with cM2:
            m_time = st.text_input("Hora (HH:MM)", key="plays_manual_time", placeholder="05:00")
        with cM3:
            m_league = st.text_input("Liga (opcional)", key="plays_manual_league", placeholder="LCK")

        cM4, cM5, cM6 = st.columns([2,2,1])
        with cM4:
            m_teamA = st.text_input("Time A (como você tem)", key="plays_manual_teamA", placeholder="BRION")
        with cM5:
            m_teamB = st.text_input("Time B (como você tem)", key="plays_manual_teamB", placeholder="Hanwha Life Esports")
        with cM6:
            m_bo = st.selectbox("BO", [1, 2, 3, 5], key="plays_manual_bo")

        if st.button("Adicionar (vai pedir mapeamento se precisar)", width='stretch', key="plays_manual_add"):
            # valida mínimo
            if not (m_date.strip() and m_time.strip() and m_teamA.strip() and m_teamB.strip()):
                st.warning("Preencha pelo menos Data, Hora, Time A e Time B.")
            else:
                pending_now = core_get_pending_schedule_import(st.session_state)
                pending_now.append({
                    "date": m_date.strip(),
                    "time": m_time.strip(),
                    "league": m_league.strip(),
                    "teamA_raw": m_teamA.strip(),
                    "teamB_raw": m_teamB.strip(),
                })
                core_save_pending_schedule_import(st.session_state, pending_now)
                st.success("Adicionado à prévia. Agora mapeie (se necessário) e clique em **Adicionar à agenda**.")
                st.rerun()

    if parse_clicked:
        parsed = core_parse_schedule_text(txt)
        if not parsed:
            st.warning("Não consegui encontrar jogos nesse texto. Dica: precisa ter linhas com hora (HH:MM) e logo depois 2 linhas de times.")
        else:
            core_save_pending_schedule_import(st.session_state, parsed)
            st.success(f"Encontrados: {len(parsed)} jogo(s). Agora confira o mapeamento e clique em **Adicionar à agenda**.")
            st.rerun()

    pending = core_get_pending_schedule_import(st.session_state)
    core_set_schedule_unknown_aliases(st.session_state, False)
    if isinstance(pending, list) and pending:
        st.markdown("#### Prévia (antes de adicionar)")
        st.dataframe(pd.DataFrame(pending), width='stretch', hide_index=True)

        aliases = core_load_schedule_aliases(st.session_state, _alias_path)

        # detecta times não resolvidos
        unknowns: List[str] = []
        for g in pending:
            a_raw = str(g.get("teamA_raw", "") or "").strip()
            b_raw = str(g.get("teamB_raw", "") or "").strip()
            a_res, ok_a = core_resolve_schedule_team(a_raw, teams, aliases)
            b_res, ok_b = core_resolve_schedule_team(b_raw, teams, aliases)
            if not ok_a and a_raw:
                unknowns.append(a_raw)
            if not ok_b and b_raw:
                unknowns.append(b_raw)
        unknowns = sorted(list({u for u in unknowns}))
        core_set_schedule_unknown_aliases(st.session_state, bool(unknowns))

        mapping: Dict[str, str] = {}
        if unknowns:
            st.markdown("#### Mapear times desconhecidos")
            st.caption("Se o nome do site não bater com o nome do CSV, escolha o time correto aqui. Isso fica salvo (apelido -> time do CSV).")
            for raw in unknowns:
                sug = core_suggest_schedule_teams(raw, teams)
                pick = st.selectbox(
                    f"Esse time '{raw}' seria qual",
                    options=["", *sug],
                    key=f"map_{core_normalize_schedule_team_key(raw)}",
                )
                if pick:
                    mapping[raw] = pick

        c1, c2 = st.columns([1,1])
        with c1:
            add_clicked = st.button("Adicionar à agenda", type="primary", width='stretch', key="plays_schedule_add_btn")
        with c2:
            cancel_clicked = st.button("Cancelar importação", width='stretch', key="plays_schedule_cancel_btn")

        if cancel_clicked:
            core_clear_pending_schedule_import(st.session_state)
            st.info("Importação cancelada.")
            st.rerun()

        if add_clicked:
            # aplica mapping + aliases existentes
            aliases = core_load_schedule_aliases(st.session_state, _alias_path)
            for raw, canon in mapping.items():
                if canon:
                    aliases[core_normalize_schedule_team_key(raw)] = canon
            core_save_schedule_aliases(st.session_state, _alias_path, aliases)

            out_games: List[Dict[str, Any]] = []
            for g in pending:
                a_raw = str(g.get("teamA_raw", "") or "").strip()
                b_raw = str(g.get("teamB_raw", "") or "").strip()
                a_res, ok_a = core_resolve_schedule_team(a_raw, teams, aliases)
                b_res, ok_b = core_resolve_schedule_team(b_raw, teams, aliases)

                # se ainda ficou desconhecido, mantém raw (não bloqueia), mas não vai carregar automático pro confronto
                out_games.append({
                    "date": str(g.get("date", "") or "").strip(),
                    "time": str(g.get("time", "") or "").strip(),
                    "league": str(g.get("league", "") or "").strip(),
                    "teamA": a_res,
                    "teamB": b_res,
                    "bo": int(default_bo),
                    "lines": "",
                })

            core_append_schedule_games(st.session_state, out_games)
            core_clear_pending_schedule_import(st.session_state)
            st.success(f"Adicionados: {len(out_games)} jogo(s) na agenda.")
            st.rerun()

    st.caption(f"Agenda atual: {len(core_get_schedule_games(st.session_state))} jogo(s).")


teamA, teamB, bo = core_render_matchup_selection(
    st.session_state,
    flow_stage=flow_stage,
    teams=teams,
    season_label=season_label,
    matchup_key_fn=build_matchup_key,
    save_lines_for_matchup_fn=_save_lines_for_matchup,
    load_lines_for_matchup_fn=_load_lines_for_matchup,
    reset_flow_fn=_reset_flow,
)

core_render_schedule_alias_mapping(
    st.session_state,
    alias_path=(APP_ROOT / "user_team_aliases.json"),
    teams=teams,
)


def _run_pre_delivery_checks(app_root: Path, csv_path_s: str, artifact_path_s: str, run_smoke: bool = False) -> Dict[str, Any]:
    return core_run_pre_delivery_checks_runtime(
        st.session_state,
        app_root=app_root,
        csv_path_s=csv_path_s,
        artifact_path_s=artifact_path_s,
        run_smoke=bool(run_smoke),
        assess_confidence_baseline_file_fn=core_assess_confidence_baseline_file,
        assess_state_artifact_file_fn=core_assess_state_artifact_file,
        assess_patch_calibration_freshness_fn=core_assess_patch_calibration_freshness,
        assess_combined_trace_file_fn=core_assess_combined_trace_file,
        assess_feature_trace_health_fn=core_assess_feature_trace_health,
        assess_session_trace_fn=core_assess_session_trace,
        assess_navigation_integrity_fn=core_assess_navigation_integrity,
        assess_utf8_file_fn=core_assess_utf8_file,
        assess_ranking_coverage_fn=core_assess_ranking_coverage,
        assess_params_baseline_stability_fn=core_assess_params_baseline_stability,
        assess_anchor_regression_fn=core_assess_anchor_regression,
        run_pre_delivery_smokes_fn=core_run_pre_delivery_smokes,
        build_patch_snapshot_fn=core_build_patch_snapshot,
        recommend_recalibration_fn=core_recommend_recalibration,
        compute_anchor_snapshot_fn=_compute_anchor_snapshot,
        get_feature_trace_state_fn=core_get_feature_trace_state,
        get_ml_trace_snapshot_fn=core_get_ml_trace_snapshot,
        pick_tabs_fn=_pick_tabs,
        required_tabs=_TAB_ALL,
        rank_table_fn=_mlcore_rank_table_v2,
        anchor_baseline_path=_ODDS_ANCHOR_BASELINE_PATH,
        python_executable=sys.executable,
        resolve_players_artifact_path_fn=_resolve_runtime_players_artifact_path,
    )

if flow_stage == "select":
    _can_continue_preview = bool(teamA) and bool(teamB) and (str(teamA) != str(teamB))
    if _can_continue_preview:
        st.info("Clique em **Analisar Confronto** para abrir o preenchimento das linhas.")
    else:
        st.info("Escolha os times e clique **Analisar Confronto**.")
    st.stop()

# -----------------------------
# Etapa 2 — preencher linhas
# -----------------------------
if flow_stage == "lines":
    st.subheader("Linhas (preencha e confirme)")

    # garante que as linhas exibidas correspondem ao confronto atual (season + times)
    key = build_matchup_key(season_label, teamA, teamB)
    core_set_matchup_key(st.session_state, key)
    if core_get_lines_loaded_key(st.session_state) != key:
        _load_lines_for_matchup(key)
        core_set_lines_loaded_key(st.session_state, key)

    st.caption("Use vírgula para separar múltiplas linhas. Decimal usa **ponto**. Tempo: x, x.0, x.5.")

    with st.expander("Padrão global de linhas (opcional)", expanded=False):
        st.caption("Se você costuma usar as mesmas linhas em vários confrontos, salve aqui para preencher automaticamente quando não houver linhas salvas para o confronto.")
        cP1, cP2 = st.columns(2)
        with cP1:
            if st.button("Salvar como padrão global", width='stretch', key="lines_global_save"):
                _save_lines_for_matchup(_LINES_STORE_DEFAULTS_KEY)
                st.success("Padrão global salvo.")
        with cP2:
            if st.button("Limpar padrão global", width='stretch', key="lines_global_clear"):
                s = _lines_store()
                if isinstance(s, dict):
                    s.pop(_LINES_STORE_DEFAULTS_KEY, None)
                    _persist_lines_store_disk(s)
                st.success("Padrão global removido.")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.text_input("Kills (ex: 27.5, 30.5)", key="lines_kills", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Torres (ex: 11.5, 12.5)", key="lines_towers", on_change=_save_lines_for_matchup, args=(key,))
    with g2:
        st.text_input("Dragões (ex: 4.5, 5.5)", key="lines_dragons", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Barões (ex: 1.5)", key="lines_barons", on_change=_save_lines_for_matchup, args=(key,))
    with g3:
        st.text_input("Inibidores (ex: 1.5)", key="lines_inhib", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Tempo (ex: 32, 32.5)", key="lines_time", on_change=_save_lines_for_matchup, args=(key,))

    _lines_extras_ctx = st.expander("Extras (opcional)", expanded=False)
    with _lines_extras_ctx:
        st.text_input(f"Total Kills {teamA} (ex: 16.5)", key="teamA_kills_line_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input(f"Total Kills {teamB} (ex: 15.5)", key="teamB_kills_line_text", on_change=_save_lines_for_matchup, args=(key,))

        st.caption("Handicap: interpreta como (métrica_for + handicap) > métrica_against (sem push; use .5).")
        _hc_team_cur, hc_team_opts = core_resolve_hc_kills_team(st.session_state, team_a=teamA, team_b=teamB)
        st.selectbox(
            "Handicap Kills — Time",
            options=hc_team_opts,
            key="hc_kills_team",
            on_change=_save_lines_for_matchup,
            args=(key,),
        )
        _hc_team, _ = core_resolve_hc_kills_team(st.session_state, team_a=teamA, team_b=teamB, write_back=False)
        st.text_input(f"HC Kills ({_hc_team}) (ex: -2.5)", key="hc_kills_text", on_change=_save_lines_for_matchup, args=(key,))
        _hc_val_preview = _parse_single_float(st.session_state.get("hc_kills_text", ""))
        if _hc_val_preview is not None:
            _hc_other = teamB if _hc_team == teamA else teamA
            st.caption(f"Outro lado (automático): **{_hc_other} {(-_hc_val_preview):+g}**")
        st.text_input(f"HC Torres {teamA} (ex: -3.5)", key="hc_towers_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input(f"HC Dragões {teamA} (ex: -1.5)", key="hc_dragons_text", on_change=_save_lines_for_matchup, args=(key,))

        st.text_input("Linha Kills (para ML + Totais)", key="combo_kills_text", on_change=_save_lines_for_matchup, args=(key,))
        st.text_input("Linha Tempo (para ML + Totais)", key="combo_time_text", on_change=_save_lines_for_matchup, args=(key,))

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Salvar linhas e ver resultados", width='stretch'):
            # persiste linhas para esse confronto (para reaparecer ao editar, e ao voltar sem trocar times)
            _save_lines_for_matchup(build_matchup_key(season_label, teamA, teamB))
            core_set_flow_stage(st.session_state, "results")
            core_request_analysis_refresh(st.session_state)
            _request_streamlit_rerun()
    with c2:
        if st.button("Voltar", width='stretch'):
            _reset_flow(keep_config=True)
            _request_streamlit_rerun()

    st.stop()

# -----------------------------
# Etapa 3 — resultados
# -----------------------------
# Load base dataframe (team_games) uma vez (cacheado)
team_games = _load_team_games(csv_sig)
_prev_season_info = next(
    (
        dict(info)
        for info in (seasons or [])
        if int(info.get("year", 0) or 0) == int(season_year) - 1
    ),
    {},
)
_prev_season_label = str(_prev_season_info.get("season") or "")
_prev_season_year = int(_prev_season_info.get("year", 0) or 0)
_prev_season_csv_path = str(_prev_season_info.get("csv_path") or "").strip()
_prev_team_games = pd.DataFrame()
if _prev_season_csv_path and Path(_prev_season_csv_path).exists():
    try:
        _prev_team_games = _load_team_games(_file_sig(_prev_season_csv_path))
    except (OSError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        _prev_team_games = pd.DataFrame()

# Janela rápida (layout estilo print: Ano Todo / 10 / 5 / Geral)
# "Geral" = consolidado recency-ponderado das 3 janelas (só afeta os TOTAIS;
# o ML do vencedor vem do artefato Elo e independe da janela). Ver
# app_core/window_blend_core.py.
win_map = {
    "Geral": "Geral",
    "Ano todo": "All",
    "Últimos 10": "Last 10",
    "Últimos 5": "Last 5",
}
if str(st.session_state.get("window_short", "") or "") not in win_map:
    st.session_state["window_short"] = next(iter(win_map.keys()))
window_short = st.radio("Janela (Histórico)", list(win_map.keys()), horizontal=True, key="window_short")
window_opt = win_map.get(window_short, "All")

# Controles de engine/filtros foram movidos para a aba Técnico/Parâmetros.
# Aqui no Plays apenas consumimos os valores já definidos no session_state.
years = sorted([int(y) for y in team_games["year"].dropna().unique().tolist()])
_results_runtime = core_resolve_results_runtime_settings(
    st.session_state,
    season_year=season_year,
    years=years,
)
default_year = int(_results_runtime["default_year"])
year_opt = _results_runtime["year_opt"]
split_opt = _results_runtime["split_opt"]
playoffs_opt = _results_runtime["playoffs_opt"]
fixed_league = _results_runtime["fixed_league"]
totals_mode = _results_runtime["totals_mode"]
ml_mode = _results_runtime["ml_mode"]
map_mode_ui = _results_runtime["map_mode_ui"]
league_mode = _results_runtime["league_mode"]
weight_mode = _results_runtime["weight_mode"]
half_life_days = _results_runtime["half_life_days"]
n_sims = int(_results_runtime["n_sims"])

# Seed determinística para simulações (evita variação por rerun sem mudança real de inputs).
_sim_seed_series = core_compute_sim_seed_series(
    st.session_state,
    stable_seed_fn=_stable_seed_from_parts,
    season_label=season_label,
    team_a=teamA,
    team_b=teamB,
    bo=int(bo),
    year_opt=year_opt,
    split_opt=split_opt,
    playoffs_opt=playoffs_opt,
    league_mode=league_mode,
    map_mode_ui=map_mode_ui,
    weight_mode=weight_mode,
    half_life_days=half_life_days,
    n_sims=n_sims,
    totals_mode=totals_mode,
    ml_mode=ml_mode,
)

# Recorte -> max_games
# "Geral" mantém max_games=None (= ano todo) para TODO o resto do app
# (ML, outras abas, filtros); o consolidado recency-ponderado é aplicado só
# nos totais, mais abaixo, quando is_geral é True.
max_games = None
is_geral = window_opt == "Geral"
if window_opt.startswith("Last"):
    try:
        max_games = int(window_opt.split()[-1])
    except (TypeError, ValueError):
        max_games = None

# Linhas (texto): em "results", prefira o store (evita perder valores quando os widgets não estão renderizados)
_active_match_key = build_matchup_key(season_label, teamA, teamB)
_saved = (_lines_store().get(_active_match_key) or {}) if flow_stage == "results" else {}
_line_texts = core_resolve_active_line_texts(
    st.session_state,
    flow_stage=str(flow_stage),
    saved_lines=_saved,
)
kills_lines = _line_texts["lines_kills"]
towers_lines = _line_texts["lines_towers"]
dragons_lines = _line_texts["lines_dragons"]
barons_lines = _line_texts["lines_barons"]
inhib_lines = _line_texts["lines_inhib"]
time_lines = _line_texts["lines_time"]

teamA_kills_line_text = _line_texts["teamA_kills_line_text"]
teamB_kills_line_text = _line_texts["teamB_kills_line_text"]
hc_kills_text = _line_texts["hc_kills_text"]
hc_towers_text = _line_texts["hc_towers_text"]
hc_dragons_text = _line_texts["hc_dragons_text"]
combo_kills_text = _line_texts["combo_kills_text"]
combo_time_text = _line_texts["combo_time_text"]

kills = parse_lines_text(kills_lines)
towers = parse_lines_text(towers_lines)
dragons = parse_lines_text(dragons_lines)
barons = parse_lines_text(barons_lines)
inhib = parse_lines_text(inhib_lines)
time_m = parse_time_lines_text(time_lines)

# mapeia as linhas preenchidas (por mercado) para reutilizar em ML+Totais
lines_for = {
    "kills": kills,
    "towers": towers,
    "dragons": dragons,
    "barons": barons,
    "inhibitors": inhib,
    "time": time_m,
}


def _analysis_signature() -> int:
    return core_build_analysis_signature(
        st.session_state,
        stable_seed_fn=_stable_seed_from_parts,
        sanitize_marginal_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
        season_label=season_label,
        team_a=teamA,
        team_b=teamB,
        bo=int(bo),
        year_opt=year_opt,
        split_opt=split_opt,
        playoffs_opt=playoffs_opt,
        league_mode=league_mode,
        fixed_league=fixed_league,
        window_opt=window_opt,
        weight_mode=weight_mode,
        half_life_days=half_life_days,
        n_sims=n_sims,
        totals_mode=totals_mode,
        ml_mode=ml_mode,
        map_mode_ui=map_mode_ui,
        line_texts=_line_texts,
    )


def _consistency_analysis_signature() -> int:
    return core_build_consistency_analysis_signature(
        st.session_state,
        stable_seed_fn=_stable_seed_from_parts,
        sanitize_marginal_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
        season_label=season_label,
        team_a=teamA,
        team_b=teamB,
        bo=int(bo),
        year_opt=year_opt,
        split_opt=split_opt,
        playoffs_opt=playoffs_opt,
        league_mode=league_mode,
        fixed_league=fixed_league,
        window_opt=window_opt,
        weight_mode=weight_mode,
        half_life_days=half_life_days,
        n_sims=n_sims,
        totals_mode=totals_mode,
        map_mode_ui=map_mode_ui,
        line_texts=_line_texts,
    )

if core_consume_consistency_refresh_pending(st.session_state):
    core_set_consistency_analysis_signature(st.session_state, _consistency_analysis_signature())

# Build Filters object
filters = Filters(
    year=None if year_opt == "All" else int(year_opt),
    league=None,  # handled by league_mode
    split=None if split_opt == "All" else str(split_opt),
    playoffs=None if playoffs_opt == "all" else (True if playoffs_opt == "true" else False),
)

league_fixed_value: Optional[str] = None
if league_mode == "fixed" and fixed_league and fixed_league != "All":
    league_fixed_value = fixed_league
    filters = Filters(
        year=filters.year,
        league=league_fixed_value,
        split=filters.split,
        playoffs=filters.playoffs,
    )

_prev_filters_active, _prev_dfA_context_year_full, _prev_dfB_context_year_full, _prev_reference_note = (
    _resolve_prev_season_with_fallback(
        _prev_team_games,
        filters=filters,
        team_a=teamA,
        team_b=teamB,
        prev_season_year=_prev_season_year,
        league_mode=league_mode,
        map_mode=map_mode_ui,
        filter_team_games_fn=filter_team_games,
    )
)
_prev_history_slices: Dict[str, pd.DataFrame] = {}
if not _prev_dfA_context_year_full.empty and not _prev_dfB_context_year_full.empty:
    _prev_df_h2h_context_year_full_A = _prev_dfA_context_year_full[
        _prev_dfA_context_year_full["opponent"] == teamB
    ].copy()
    _prev_history_slices = core_recency_slices(
        df_a=_prev_dfA_context_year_full,
        df_b=_prev_dfB_context_year_full,
        df_h2h=_prev_df_h2h_context_year_full_A,
        id_col="gameid",
        date_col="date",
        dedupe_league=True,
        backfill_unique_targets=True,
    )

# -----------------------------
# ML odds (engine)
# -----------------------------
_ml_runtime_cfg = core_get_ml_engine_runtime_settings(st.session_state)
ml_engine = _ml_runtime_cfg["ml_engine"]
as_of_date = _ml_runtime_cfg.get("ml_as_of_date") or date.today()

p_map_raw = p_map_cal = float("nan")
_ml_info = None
_lgA = _lgB = _tierA = _tierB = ""

# Calcula ML core v2 somente quando necessário (engine=mlcore ou comparação no debug).
if ml_engine == "mlcore_v2" or bool(_ml_runtime_cfg["ml_engine_compare"]):
    core_clear_ml_fused_post_cal_info(st.session_state)
    core_clear_ml_microseg_info(st.session_state)

    _art_sig = ("", 0, 0)
    try:
        if artifact_path:
            _art_sig = _file_sig(artifact_path)
    except OSError:
        _art_sig = ("", 0, 0)
    _fit_sig_ml = ("", 0, 0)
    try:
        if _FIT_PARAMS_PATH.exists():
            _fit_sig_ml = _file_sig(str(_FIT_PARAMS_PATH))
    except OSError:
        _fit_sig_ml = ("", 0, 0)

    _mlcore_cache_key = (
        "mlcore_v2",
        csv_sig,
        _art_sig,
        _fit_sig_ml,
        str(teamA).strip().casefold(),
        str(teamB).strip().casefold(),
        str(as_of_date),
        league_mode,
        str(getattr(filters, "league", "")) if league_mode == "fixed" else "",
        getattr(filters, "year", None),
        core_get_achievement_path(st.session_state),
        bool(_ml_runtime_cfg["ml_microseg_enabled"]),
        int(_ml_runtime_cfg["ml_microseg_lookback_games"]),
        float(_ml_runtime_cfg["ml_microseg_shrink_k"]),
        int(_ml_runtime_cfg["ml_microseg_min_n"]),
        float(_ml_runtime_cfg["ml_microseg_strength"]),
        float(_ml_runtime_cfg["ml_microseg_cap_pp"]),
        bool(_ml_runtime_cfg["ml_microseg_hierarchical"]),
        float(_ml_runtime_cfg["ml_microseg_hier_decay"]),
        bool(_ml_runtime_cfg["ml_fused_post_cal_use_fitted"]),
    )
    _ml_cache_state = core_get_mlcore_cache_state(st.session_state)
    _cache = _ml_cache_state.get("value")
    if _ml_cache_state.get("key") == _mlcore_cache_key and isinstance(_cache, dict):
        _ml_info = _cache.get("ml_info")
        p_map_raw = float(_cache.get("p_raw", float("nan")))
        p_map_cal = float(_cache.get("p_cal", float("nan")))
        if isinstance(_cache.get("microseg_info"), dict):
            core_set_ml_microseg_info(st.session_state, _cache.get("microseg_info"))
        if isinstance(_cache.get("fused_post_cal_info"), dict):
            core_set_ml_fused_post_cal_info(st.session_state, _cache.get("fused_post_cal_info"))
        _lgA = str(_cache.get("league_a", _cache.get("match_league", "")) or "")
        _lgB = str(_cache.get("league_b", _cache.get("match_league", "")) or "")
        _tierA = str(_cache.get("tier_a", _cache.get("match_tier", "")) or "")
        _tierB = str(_cache.get("tier_b", _cache.get("match_tier", "")) or "")
    else:
        def _resolve_leagues_for_matchup(ta, tb, year, lmode, fixed_lg):
            return core_resolve_runtime_team_leagues(
                team_a=ta, team_b=tb, year=year, league_mode=str(lmode),
                fixed_league=fixed_lg,
                infer_team_league_fn=lambda team, yr: _infer_team_league_from_csv(csv_sig, team, yr),
                infer_match_league_fn=lambda a, b, yr: _infer_match_league_from_csv(csv_sig, a, b, yr),
                tier_from_league_fn=core_tier_from_league,
                team_aliases_fn=_identity_team_name_candidates,
            )

        _core_result = _compute_ml_core_prediction(
            team_a=teamA,
            team_b=teamB,
            artifact_path=artifact_path,
            as_of_date=as_of_date,
            achievement_path=core_get_achievement_path(st.session_state),
            ml_runtime_cfg=_ml_runtime_cfg,
            filter_year=getattr(filters, "year", None),
            filter_league=getattr(filters, "league", None),
            filter_split=getattr(filters, "split", None),
            filter_playoffs=getattr(filters, "playoffs", None),
            bo=int(bo),
            league_mode=str(league_mode),
            csv_sig=csv_sig,
            art_sig=_art_sig,
            fit_params_path=str(_FIT_PARAMS_PATH) if _FIT_PARAMS_PATH.exists() else None,
            resolve_leagues_fn=_resolve_leagues_for_matchup,
            p_map_v2_fn=_mlcore_p_map_v2,
            build_microseg_table_fn=_build_ml_microseg_table,
            apply_microseg_fn=_apply_ml_microseg_calibration,
            apply_fused_postcal_fn=_apply_fused_post_calibration,
            load_fitted_params_fn=_load_fitted_params_cached,
            file_sig_fn=_file_sig,
        )

        _ml_info = _core_result.get("ml_info")
        p_map_raw = float(_core_result.get("p_map_raw", float("nan")))
        p_map_cal = float(_core_result.get("p_map_cal", float("nan")))
        _lgA = str(_core_result.get("league_a", ""))
        _lgB = str(_core_result.get("league_b", ""))
        _tierA = str(_core_result.get("tier_a", ""))
        _tierB = str(_core_result.get("tier_b", ""))

        if isinstance(_core_result.get("microseg_info"), dict):
            core_set_ml_microseg_info(st.session_state, _core_result["microseg_info"])
        if isinstance(_core_result.get("fused_post_cal_info"), dict):
            core_set_ml_fused_post_cal_info(st.session_state, _core_result["fused_post_cal_info"])

        core_set_mlcore_cache_state(st.session_state, _mlcore_cache_key, {
            "ml_info": _ml_info,
            "p_raw": p_map_raw,
            "p_cal": p_map_cal,
            "microseg_info": core_get_ml_microseg_info(st.session_state),
            "fused_post_cal_info": core_get_ml_fused_post_cal_info(st.session_state),
            "match_league": _lgA,
            "match_tier": _tierA,
            "league_a": _lgA,
            "league_b": _lgB,
            "tier_a": _tierA,
            "tier_b": _tierB,
        })

# Post layers: WF correction, p_series, fit scales + players context.
core_clear_wf_ml_corr_info(st.session_state)
_post_result = _compute_ml_post_layers(
    ml_info=_ml_info,
    p_map_cal=p_map_cal,
    bo=int(bo),
    ml_runtime_cfg=_ml_runtime_cfg,
    wf_report=core_get_wf_last_report(st.session_state),
    wf_correct_p_map_fn=_wf_correct_p_map,
    fit_scales_for_league_fn=_fit_scales_for_league,
    fit_players_context_for_league_fn=_fit_players_context_for_league,
    prob_win_series_fn=prob_win_series,
)
p_map_cal = float(_post_result.get("p_map_cal", p_map_cal))
p_series = float(_post_result.get("p_series", float("nan")))
if isinstance(_post_result.get("wf_corr_info"), dict):
    core_set_wf_ml_corr_info(st.session_state, _post_result["wf_corr_info"])
_ts_used = float(_post_result.get("fit_team_scale", 1.0))
_ps_used = float(_post_result.get("fit_players_scale", 1.0))
_lg_used = str(_post_result.get("fit_league_used", ""))
_pcfg_used = _post_result.get("players_context_cfg") or {}
_pcfg_src = str(_post_result.get("players_context_src", ""))
core_set_fit_used_scales(st.session_state, team_scale=_ts_used, players_scale=_ps_used)
core_set_fit_runtime_labels(
    st.session_state,
    scale_league_used=_lg_used,
    players_context_source_used=_pcfg_src,
)
core_apply_players_context_runtime_override(
    st.session_state,
    _pcfg_used,
    _pcfg_src,
    use_fitted=bool(_ml_runtime_cfg["ml_players_context_use_fitted"]),
)

if ml_engine == "mlcore_v2" and not math.isfinite(p_map_cal):
    st.warning("Não consegui calcular p_map via ML core v2 com esse ml_artifact.json. (ML fica indisponível, mas Totais continuam.)")


map_mode = normalize_map_mode(map_mode_ui)
hl_txt = "Uniforme" if half_life_days is None else f"{half_life_days}d"

# -----------------------------
# Totals (map or series)
# -----------------------------
dfA_used = filter_team_games(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode=map_mode, max_games=max_games)
dfB_used = filter_team_games(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode=map_mode, max_games=max_games)

pA = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=max_games)
pB = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=max_games)

if pA.n_games == 0 or pB.n_games == 0:
    st.error("Sem jogos suficientes com esses filtros. Ajuste Split/Playoffs/League mode ou a janela.")
    st.caption("Aviso: os mercados do confronto podem ficar incompletos, mas abas independentes (ex.: Players/Campeões) continuam disponíveis.")


def _expected_totals_dc(_df: pd.DataFrame, _pA, _pB) -> Dict[str, Any]:
    """Totais V0 + blend Dixon-Coles nos mercados validados (kills/dragões/
    tempo; gate ACCEPT 2026-07-06, ver mlcore.lines_engine.apply_dc_blend)."""
    _t = matchup_expected_totals(_pA, _pB)
    try:
        from mlcore.lines_engine import apply_dc_blend
        return apply_dc_blend(_df, str(getattr(_pA, "team", "")), str(getattr(_pB, "team", "")), _t)
    except Exception:
        return _t


totals = _expected_totals_dc(team_games, pA, pB)

# AVG maps baseline (gol.gg style)
pA_avg = pA
pB_avg = pB
avg_totals = None
if map_mode != "avg":
    pA_avg = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=max_games)
    pB_avg = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=max_games)
    avg_totals = _expected_totals_dc(team_games, pA_avg, pB_avg)

totals_map = totals
totals_avg = avg_totals if avg_totals is not None else totals

# --- Modo "Geral": consolida as 3 janelas (recency-ponderado) nos TOTAIS ---
# Recalcula os totais para cada janela (Ano/L10/L5) e blendeia. O ML do
# vencedor não é tocado (vem do artefato Elo, independe da janela).
# totals_breakdown guarda o detalhe por janela p/ exibir abaixo da tabela.
totals_breakdown: Dict[str, Any] = {}
if is_geral:
    def _totals_for_window(_mg: Optional[int]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        _pA = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=_mg)
        _pB = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode=map_mode, half_life_days=half_life_days, max_games=_mg)
        _tm = _expected_totals_dc(team_games, _pA, _pB) if (_pA.n_games > 0 and _pB.n_games > 0) else {}
        if map_mode != "avg":
            _pAa = build_profile(team_games, team=teamA, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=_mg)
            _pBa = build_profile(team_games, team=teamB, filters=filters, league_mode=league_mode, map_mode="avg", half_life_days=half_life_days, max_games=_mg)
            _ta = _expected_totals_dc(team_games, _pAa, _pBa) if (_pAa.n_games > 0 and _pBa.n_games > 0) else _tm
        else:
            _ta = _tm
        return _tm, _ta

    _per_map: list = []
    _per_avg: list = []
    for _lbl, _mg in _GERAL_WINDOWS:
        _w = _geral_window_weight(_mg)
        _tm, _ta = _totals_for_window(_mg)
        if _tm:
            _per_map.append((_w, _tm))
        if _ta:
            _per_avg.append((_w, _ta))
        # breakdown segue o mesmo modo (map/avg) exibido na tabela
        totals_breakdown[_lbl] = (_tm if totals_mode == "map" else _ta)
    _blend_map = _blend_matchup_totals(_per_map, totals_cls=MatchupTotals)
    _blend_avg = _blend_matchup_totals(_per_avg, totals_cls=MatchupTotals)
    if _blend_map:
        totals_map = _blend_map
    if _blend_avg:
        totals_avg = _blend_avg

totals_for_lines = totals_map if totals_mode == "map" else totals_avg

_prev_totals_map: Dict[str, Any] = {}
_prev_totals_avg: Dict[str, Any] = {}
_prev_totals_for_lines: Dict[str, Any] = {}
if (
    isinstance(_prev_team_games, pd.DataFrame)
    and not _prev_team_games.empty
    and _prev_filters_active is not None
):
    _prev_pA = build_profile(
        _prev_team_games,
        team=teamA,
        filters=_prev_filters_active,
        league_mode=league_mode,
        map_mode=map_mode,
        half_life_days=half_life_days,
        max_games=max_games,
    )
    _prev_pB = build_profile(
        _prev_team_games,
        team=teamB,
        filters=_prev_filters_active,
        league_mode=league_mode,
        map_mode=map_mode,
        half_life_days=half_life_days,
        max_games=max_games,
    )
    if _prev_pA.n_games > 0 and _prev_pB.n_games > 0:
        _prev_totals_map = _expected_totals_dc(_prev_team_games, _prev_pA, _prev_pB)
        if map_mode != "avg":
            _prev_pA_avg = build_profile(
                _prev_team_games,
                team=teamA,
                filters=_prev_filters_active,
                league_mode=league_mode,
                map_mode="avg",
                half_life_days=half_life_days,
                max_games=max_games,
            )
            _prev_pB_avg = build_profile(
                _prev_team_games,
                team=teamB,
                filters=_prev_filters_active,
                league_mode=league_mode,
                map_mode="avg",
                half_life_days=half_life_days,
                max_games=max_games,
            )
            if _prev_pA_avg.n_games > 0 and _prev_pB_avg.n_games > 0:
                _prev_totals_avg = _expected_totals_dc(_prev_team_games, _prev_pA_avg, _prev_pB_avg)
        if not _prev_totals_avg:
            _prev_totals_avg = dict(_prev_totals_map)
        _prev_totals_for_lines = _prev_totals_map if totals_mode == "map" else _prev_totals_avg

# -----------------------------
# Main result tabs (formato dos prints)
# -----------------------------
with st.sidebar:
    st.markdown("### Navega\u00e7\u00e3o")
    _page_modes = ["Fluxo R\u00e1pido", "Mercados", "An\u00e1lise", "Completo", "Dev"]
    _page_mode_cur = _fix_mojibake_text(core_get_ui_page_mode(st.session_state, NAV_DEFAULT_UI_PAGE_MODE) or NAV_DEFAULT_UI_PAGE_MODE)
    if _page_mode_cur not in _page_modes:
        _page_mode_cur = NAV_DEFAULT_UI_PAGE_MODE
    if str(st.session_state.get("ui_page_mode", "") or "") not in _page_modes:
        st.session_state["ui_page_mode"] = _page_mode_cur
    _page_mode_selected = st.radio(
        "Modo de p\u00e1ginas",
        options=_page_modes,
        key="ui_page_mode",
        help="Reduz a quantidade de abas vis\u00edveis sem remover funcionalidades.",
    )

# Gate de análise: em "results", só recalcula após clique explícito.
if flow_stage == "results":
    _analysis_gate_state = core_get_analysis_gate_state(st.session_state)
    _sig_now = _analysis_signature()
    _force_once = core_consume_analysis_force_once(st.session_state)
    _sig_last = _analysis_gate_state.get("analysis_sig_last")
    if _sig_last is None or _force_once:
        core_set_analysis_signature(st.session_state, _sig_now)
        _sig_last = int(_sig_now)
    _needs_reanalyze = int(_sig_last if _sig_last is not None else _sig_now) != int(_sig_now)
    ga1, ga2 = st.columns([4, 1])
    with ga1:
        if _needs_reanalyze:
            st.warning("Par\u00e2metros/linhas mudaram. Clique em **Analisar confronto** para atualizar as odds.")
        else:
            st.caption("An\u00e1lise em dia para os par\u00e2metros atuais.")
    with ga2:
        if st.button("Analisar confronto", type="primary", width='stretch', key="analyze_gate_btn"):
            core_set_analysis_signature(st.session_state, _sig_now)
            core_set_consistency_analysis_signature(st.session_state, _consistency_analysis_signature())
            _request_streamlit_rerun()
    if _needs_reanalyze:
        st.stop()

_page_mode_now = _fix_mojibake_text(core_get_ui_page_mode(st.session_state, NAV_DEFAULT_UI_PAGE_MODE) or NAV_DEFAULT_UI_PAGE_MODE)
_active_tabs = _canonicalize_tab_labels(_pick_tabs(_page_mode_now), _TAB_ALL)
_tab_norm = lambda t: _normalize_text_key(t)
_TAB_VISAO = TAB_VISAO
_TAB_CONS = TAB_CONS
_TAB_PARAMS = TAB_PARAMS
if _page_mode_now in ("Fluxo R\u00e1pido", "Mercados"):
    # An\u00e1lise de Linha agora ocupa o slot onde antes ficava Consist\u00eancia das
    # linhas (que foi movida para o modo Dev junto com Par\u00e2metros/Identidade).
    from app_core.tab_dispatcher import TAB_LINE_ANALYSIS as _TAB_LA
    _active_norm = {_tab_norm(t) for t in _active_tabs}
    if _tab_norm(_TAB_LA) not in _active_norm:
        _visao_label = next((t for t in _active_tabs if _tab_norm(t) == _tab_norm(_TAB_VISAO)), None)
        if _visao_label is not None:
            _ix = _active_tabs.index(_visao_label) + 1
            _active_tabs = [*_active_tabs[:_ix], _TAB_LA, *_active_tabs[_ix:]]
        else:
            _active_tabs = [_TAB_VISAO, _TAB_LA, *list(_active_tabs)]

# Dedupe final por rótulo canônico (evita variantes quebradas/duplicadas no topo).
_active_tabs = _canonicalize_tab_labels(_active_tabs, _TAB_ALL)

tab_refs = {n: t for n, t in zip(_active_tabs, st.tabs(_active_tabs))}

if _TAB_PARAMS in tab_refs:
 with tab_refs[_TAB_PARAMS]:
    _render_params_tab(st=st, ctx=globals())


if _TAB_VISAO in tab_refs:
 with tab_refs[_TAB_VISAO]:
    # Explicit minimal ctx: only the keys render_visao_tab actually reads,
    # built from globals(). Decouples the renderer from plays_app internals
    # and validates the contract (loud-fail on missing core_* import).
    _render_visao_tab(st=st, ctx=_build_render_ctx(
        globals(), renderer_source=_inspect.getsource(_render_visao_tab),
        renderer_name="render_visao_tab",
    ))

if _TAB_CONS in tab_refs:
 with tab_refs[_TAB_CONS]:
    _render_consistencia_tab(st=st, ctx=_build_render_ctx(
        globals(), renderer_source=_inspect.getsource(_render_consistencia_tab),
        renderer_name="render_consistencia_tab",
    ))

    
# -----------------------------
# Secondary tabs extracted from plays_app
# -----------------------------
# -----------------------------
# Oportunidades: a Pinnacle inteira contra o modelo (23/09/2026)
# -----------------------------
# Ele pediu a Pinnacle "de forma total, 100%... tanto para o lol, quanto para
# o dota". O robo deste repositorio (.github/workflows/odds.yml) grava as odds
# do Dota; esta aba e a MESMA do LoL (app_core/oportunidades_*, identicos),
# ligada ao caminho de producao do Dota.
#
# DIFERENCA DE PROPOSITO: aqui o app NAO casa nome de time sozinho. A regra do
# LoL (app_core/casar_time_core.py) foi medida em 52 apelidos de LoL; no Dota
# nunca foi medida, e a identidade de time aqui e team_id + elenco, nao o
# nome. Casa so nome exato ou apelido que ele ensinar.
from mlcore.lines_engine import prob_over_from_total as core_prob_over_from_total


def _oportunidades_resolver_time(nome: str) -> str:
    try:
        achado, ok = core_resolve_schedule_team(str(nome), list(teams),
                                                _oportunidades_apelidos())
    except Exception:
        return ""
    return str(achado) if ok else ""


def _oportunidades_apelidos():
    try:
        return core_load_schedule_aliases(st.session_state, APP_ROOT / "user_team_aliases.json")
    except Exception:
        return {}


def _oportunidades_calcular_totais(casa: str, fora: str, mapa: str):
    """Os totais do confronto pelo MESMO caminho da tela de analise do Dota."""
    a_csv = _oportunidades_resolver_time(casa)
    b_csv = _oportunidades_resolver_time(fora)
    if not a_csv or not b_csv:
        return None

    def _construir(modo):
        _pA = build_profile(team_games, team=a_csv, filters=filters, league_mode=league_mode,
                            map_mode=modo, half_life_days=half_life_days, max_games=max_games)
        _pB = build_profile(team_games, team=b_csv, filters=filters, league_mode=league_mode,
                            map_mode=modo, half_life_days=half_life_days, max_games=max_games)
        if _pA.n_games == 0 or _pB.n_games == 0:
            return None, int(_pA.n_games), int(_pB.n_games)
        return _expected_totals_dc(team_games, _pA, _pB), int(_pA.n_games), int(_pB.n_games)

    # QUAL PERFIL (24/09/2026): o mapa 4 da Shopify tinha UM jogo e virou
    # 94% em torres. Regra e medicao em app_core/oportunidades_core.py.
    from app_core.oportunidades_core import totais_para_o_mapa
    return totais_para_o_mapa(_construir, mapa)


def _oportunidades_jogos_por_time(nome: str) -> int:
    alvo = _oportunidades_resolver_time(nome)
    if not alvo:
        return -1
    try:
        return int((team_games["team"].astype(str) == alvo).sum())
    except Exception:
        return -1


def _oportunidades_sugerir_times(cru: str) -> list:
    try:
        return list(core_suggest_schedule_teams(str(cru), list(teams)))
    except Exception:
        return []


def _oportunidades_ensinar_apelido(cru: str, escolhido: str) -> None:
    caminho = APP_ROOT / "user_team_aliases.json"
    apelidos = core_load_schedule_aliases(st.session_state, caminho)
    apelidos[core_normalize_schedule_team_key(str(cru))] = str(escolhido)
    core_save_schedule_aliases(st.session_state, caminho, apelidos)


def _oportunidades_times_nao_casados() -> list:
    from app_core.odds_coletadas_core import carregar_odds_coletadas, times_da_coleta
    try:
        _odds = carregar_odds_coletadas(APP_ROOT)
    except Exception:
        return []
    return [n for n in times_da_coleta(_odds) if not _oportunidades_resolver_time(n)]


def _oportunidades_medias_kills(casa: str, fora: str):
    """Kills esperados do mandante e do visitante (24/09/2026), pela mesma
    conta Dixon-Coles do total (mlcore/abates_por_time.py). No Dota so o
    HANDICAP passou na prova cega; o total por time fica sem numero."""
    a_csv = _oportunidades_resolver_time(casa)
    b_csv = _oportunidades_resolver_time(fora)
    if not a_csv or not b_csv:
        return None
    from mlcore.abates_por_time import medias_dc
    return medias_dc(team_games, a_csv, b_csv)


def _render_oportunidades_dota() -> None:
    """Protegida: o despachante do Dota nao isola falha, e uma aba com defeito
    nao pode derrubar o app inteiro."""
    try:
        from app_core.oportunidades_tab_core import render_oportunidades_tab
        render_oportunidades_tab(st=st, ctx={
            "raiz": APP_ROOT,
            "calcular_totais": _oportunidades_calcular_totais,
            "prob_over": core_prob_over_from_total,
            "medias_kills": _oportunidades_medias_kills,
            "casados_sozinhos": {},
            "jogos_por_time": _oportunidades_jogos_por_time,
            "times_nao_casados": _oportunidades_times_nao_casados(),
            "resolver_time": _oportunidades_resolver_time,
            "sugerir_times": _oportunidades_sugerir_times,
            "ensinar_apelido": _oportunidades_ensinar_apelido,
            # O MESMO numero do card do confronto (visao_tab_core le estes).
            "confronto": {
                "teamA": teamA, "teamB": teamB, "bo": int(bo),
                "p_map": st.session_state.get("_p_map_used", p_map_cal),
                "p_series": st.session_state.get("_p_series_used", p_series),
            },
        })
    except Exception as _e:  # noqa: BLE001
        st.error("A aba Oportunidades falhou ({}: {}). O resto do app segue.".format(
            type(_e).__name__, _e))


_secondary_tab_renderers = {
    "Oportunidades": _render_oportunidades_dota,
    "Rankings": lambda: core_render_rankings_tab(
        st=st,
        team_games=team_games,
        Filters=Filters,
        artifact_path=artifact_path,
        as_of_date=as_of_date,
        half_life_days=float(half_life_days),
        mlcore_rank_table_v2_fn=_mlcore_rank_table_v2,
        elo_rank_table_fn=_elo_rank_table,
        format_num_fn=_format_num,
        team_a=teamA,
        team_b=teamB,
    ),
    "Jogos": lambda: core_render_schedule_tab(st=st, bo=int(bo), teams=teams),
    "Series": lambda: core_render_series_tab(
        st=st,
        ctx={
            "ml_info": _ml_info,
            "teamA": teamA,
            "teamB": teamB,
            "bo": int(bo),
            "p_map_cal": float(p_map_cal),
            "state_get_float_fn": state_get_float,
            "state_get_int_fn": state_get_int,
            "state_get_str_fn": state_get_str,
            "winner_pricing_bundle_fn": core_compute_winner_pricing_bundle,
            "prob_win_series_fn": prob_win_series,
            "odd_from_prob_fn": _odd_from_p,
            "team_with_league_label_fn": _team_with_league_label,
            "format_num_fn": _format_num,
            "format_pct_fn": _format_pct,
            "df_contract_view_fn": _df_contract_view,
            "parse_lines_fn": parse_lines_text,
            "series_sims_cached_fn": _series_sims_cached,
            "combine_mean_fn": _combine_mean,
            "combine_sd_fn": _combine_sd,
            "matchup_totals_cls": MatchupTotals,
            "n_sims": int(n_sims),
            "totals_avg": totals_avg,
            "profile_a": pA,
            "profile_b": pB,
        },
    ),
    "StatsCamps": lambda: core_render_statscamps_tab(
        st=st,
        team_games=team_games,
        Filters=Filters,
        apply_filters_fn=apply_filters,
        parse_lines_fn=parse_lines_text,
        # Motor do LoL atualizado em 27/09/2026: a aba passou a pedir o
        # leitor de linhas de TEMPO tambem (sem isto ela quebrava).
        parse_time_lines_fn=parse_time_lines_text,
        min_to_mmss_fn=_min_to_mmss,
        format_num_fn=_format_num,
        df_contract_view_fn=_df_contract_view,
    ),
    TAB_LINE_ANALYSIS: lambda: core_render_line_analysis_tab(
        st=st,
        team_games=team_games,
        Filters=Filters,
        apply_filters_fn=apply_filters,
        min_to_mmss_fn=_min_to_mmss,
        session_state=st.session_state,
        resumo_renderer=lambda: core_render_resumo_tab(
            st=st,
            ctx={
                "teamA": teamA,
                "teamB": teamB,
                "team_games": team_games,
                "Filters": Filters,
                "filters": filters,
                "apply_filters_fn": apply_filters,
                "ml_info": _ml_info,
                "map_mode_ui": map_mode,
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
        ),
    ),
    "Campeões": lambda: core_render_champions_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "teamA": teamA,
            "teamB": teamB,
            "csv_sig": csv_sig,
            "Filters": Filters,
            "state_get_str_fn": state_get_str,
            "state_get_int_fn": state_get_int,
            "load_player_rows_fn": _load_player_rows,
            "apply_filters_players_fn": core_apply_filters_players,
            "champion_vocab_fn": core_champion_vocab,
            "fuzzy_match_one_fn": core_fuzzy_match_one,
            "df_contract_view_fn": _df_contract_view,
            "format_num_fn": _format_num,
            "ensure_champions_filter_state_fn": core_ensure_champions_filter_state,
            "reset_champions_league_dependents_fn": core_reset_champions_league_dependents,
            "resolve_blue_side_choice_fn": core_resolve_blue_side_choice,
            "sync_blue_side_choice_fn": core_sync_blue_side_choice,
        },
    ),
    "Players": lambda: core_render_players_tab(
        st=st,
        ctx={
            "team_games": team_games,
            "Filters": Filters,
            "csv_sig": csv_sig,
            "teamA": teamA,
            "teamB": teamB,
            "load_player_rows_fn": _load_player_rows,
            "apply_filters_players_fn": core_apply_filters_players,
            "latest_lineup_names_fn": core_latest_lineup_names,
            "norm_pos_fn": _norm_pos,
            "parse_players_text_fn": core_parse_players_text,
            "resolve_players_fn": core_resolve_players,
            "fuzzy_match_one_fn": core_fuzzy_match_one,
            "build_lane_rows_fn": core_build_lane_rows,
            "ensure_players_filter_state_fn": core_ensure_players_filter_state,
            "ensure_players_text_state_fn": core_ensure_players_text_state,
            "apply_players_autofill_state_fn": core_apply_players_autofill_state,
            "ensure_player_line_input_state_fn": core_ensure_player_line_input_state,
            "apply_players_submission_state_fn": core_apply_players_submission_state,
            "build_per_player_lines_by_idx_fn": core_build_per_player_lines_by_idx,
            "build_player_means_rows_fn": core_build_player_means_rows,
            "build_recent_player_stats_rows_fn": core_build_recent_player_stats_rows,
            "build_player_market_rows_fn": core_build_player_market_rows,
            "append_players_recent_stats_panel_trace_fn": core_append_players_recent_stats_panel_trace,
            "append_players_fair_panel_trace_fn": core_append_players_fair_panel_trace,
            "append_players_markets_panel_trace_fn": core_append_players_markets_panel_trace,
            "append_players_panels_bundle_trace_fn": core_append_players_panels_bundle_trace,
            "append_feature_trace_fn": _append_feature_trace,
            "feature_flag_enabled_fn": _feature_flag_enabled,
            "df_contract_view_fn": _df_contract_view,
            "format_num_fn": _format_num,
            "format_pct_fn": _format_pct,
            "format_pct1_fn": _format_pct,
            "odd_from_prob_fn": _odd_from_p,
            "state_get_float_fn": state_get_float,
            "state_get_int_fn": state_get_int,
            "parse_lines_fn": parse_lines_text,
            "fuse_probs_by_precision_fn": _fuse_probs_by_precision,
            "persist_combined_trace_file_fn": _persist_combined_trace_file,
            "playoffs_opt": playoffs_opt,
        },
    ),
    "Identidade": lambda: core_render_identity_tab(
        st=st,
        ctx={
            "registry_path": (APP_ROOT / "identity_registry.json"),
        },
    ),
}

dispatch_secondary_tabs(st, tab_refs, _secondary_tab_renderers, order=SECONDARY_TAB_ORDER)


# -----------------------------
# Vencedores (Achievements) - removido (bloco legado desativado)
# -----------------------------

                                                