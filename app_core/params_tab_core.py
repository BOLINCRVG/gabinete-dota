# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd


def _norm_gate_decision(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in {"promote", "shadow_only", "reject"} else "unknown"



def render_params_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    APP_ROOT = ctx.get("APP_ROOT")
    _FIT_PARAMS_PATH = ctx.get("_FIT_PARAMS_PATH")
    _ML_CONFIDENCE_BASELINE_PATH = ctx.get("_ML_CONFIDENCE_BASELINE_PATH")
    _ODDS_ANCHOR_BASELINE_PATH = ctx.get("_ODDS_ANCHOR_BASELINE_PATH")
    _PARAM_DEFAULTS = ctx.get("_PARAM_DEFAULTS")
    _TEAM_STATE_ARTIFACT_PATH = ctx.get("_TEAM_STATE_ARTIFACT_PATH")
    _artifact_sig = ctx.get("_artifact_sig")
    _backup_file_if_exists = ctx.get("_backup_file_if_exists")
    _build_training_csv_list = ctx.get("_build_training_csv_list")
    _compute_anchor_snapshot = ctx.get("_compute_anchor_snapshot")
    _df_contract_view = ctx.get("_df_contract_view")
    _enrich_params_with_areal_prob_tuning = ctx.get("_enrich_params_with_areal_prob_tuning")
    _enrich_params_with_ml_marginal_band_coeffs = ctx.get("_enrich_params_with_ml_marginal_band_coeffs")
    _file_sig = ctx.get("_file_sig")
    _fit_ml_params_run = ctx.get("_fit_ml_params_run")
    _format_num = ctx.get("_format_num")
    _generate_and_persist_ml_confidence_baseline = ctx.get("_generate_and_persist_ml_confidence_baseline")
    _generate_and_persist_team_state_artifact = ctx.get("_generate_and_persist_team_state_artifact")
    _load_artifact_cached = ctx.get("_load_artifact_cached")
    _load_fitted_params_cached = ctx.get("_load_fitted_params_cached")
    _load_ml_confidence_baseline_cached = ctx.get("_load_ml_confidence_baseline_cached")
    _load_team_state_artifact_cached = ctx.get("_load_team_state_artifact_cached")
    _ml_info = ctx.get("_ml_info")
    _paths_for_ml = ctx.get("_paths_for_ml")
    _persist_combined_trace_file = ctx.get("_persist_combined_trace_file")
    _read_json_file_safe = ctx.get("_read_json_file_safe")
    _request_streamlit_rerun = ctx.get("_request_streamlit_rerun")
    _resolve_players_artifact_path = ctx.get("_resolve_players_artifact_path")
    _run_daily_pipeline = ctx.get("_run_daily_pipeline")
    _run_layer_validation_cached = ctx.get("_run_layer_validation_cached")
    _run_pre_delivery_checks = ctx.get("_run_pre_delivery_checks")
    _run_walkforward_cached = ctx.get("_run_walkforward_cached")
    _save_app_settings_to_disk = ctx.get("_save_app_settings_to_disk")
    _wf_gate_multiplier = ctx.get("_wf_gate_multiplier")
    artifact_path = ctx.get("artifact_path")
    core_append_governance_log = ctx.get("core_append_governance_log")
    core_build_anchor_cases_rows = ctx.get("core_build_anchor_cases_rows")
    core_build_leakage_file_rows = ctx.get("core_build_leakage_file_rows")
    core_build_parts_rows = ctx.get("core_build_parts_rows")
    core_build_phase12_by_league_rows = ctx.get("core_build_phase12_by_league_rows")
    core_build_stop_rules_banner = ctx.get("core_build_stop_rules_banner")
    core_build_stop_rules_global_band_rows = ctx.get("core_build_stop_rules_global_band_rows")
    core_build_stop_rules_league_band_rows = ctx.get("core_build_stop_rules_league_band_rows")
    core_build_stop_rules_major_rows = ctx.get("core_build_stop_rules_major_rows")
    core_combine_governance_decisions = ctx.get("core_combine_governance_decisions")
    core_consume_flash_saved_profile_msg = ctx.get("core_consume_flash_saved_profile_msg")
    core_decision_banner_parts = ctx.get("core_decision_banner_parts")
    core_get_cal_diag_last_report = ctx.get("core_get_cal_diag_last_report")
    core_get_daily_pipeline_report = ctx.get("core_get_daily_pipeline_report")
    core_get_fit_scale_state = ctx.get("core_get_fit_scale_state")
    core_get_governance_last_log = ctx.get("core_get_governance_last_log")
    core_get_lv_last_report = ctx.get("core_get_lv_last_report")
    core_get_pre_delivery_report = ctx.get("core_get_pre_delivery_report")
    core_get_wf_last_report = ctx.get("core_get_wf_last_report")
    core_get_wf_ml_corr_info = ctx.get("core_get_wf_ml_corr_info")
    core_normalize_governance_decision = ctx.get("core_normalize_governance_decision")
    core_render_calibration_support_panels = ctx.get("core_render_calibration_support_panels")
    core_render_feature_flags_shadow_trace_panel = ctx.get("core_render_feature_flags_shadow_trace_panel")
    core_render_fit_calibration_panel = ctx.get("core_render_fit_calibration_panel")
    core_run_governance_commands = ctx.get("core_run_governance_commands")
    core_run_governance_operation = ctx.get("core_run_governance_operation")
    core_set_daily_pipeline_report = ctx.get("core_set_daily_pipeline_report")
    core_set_fit_base_scales = ctx.get("core_set_fit_base_scales")
    core_set_flash_saved_profile_msg = ctx.get("core_set_flash_saved_profile_msg")
    core_set_governance_last_log = ctx.get("core_set_governance_last_log")
    core_set_lv_last_report = ctx.get("core_set_lv_last_report")
    core_set_ml_confidence_baseline_last_report = ctx.get("core_set_ml_confidence_baseline_last_report")
    core_set_pending_apply_param_values = ctx.get("core_set_pending_apply_param_values")
    core_set_pre_delivery_report = ctx.get("core_set_pre_delivery_report")
    core_set_quick_maint_last_log = ctx.get("core_set_quick_maint_last_log")
    core_set_team_state_artifact_last_report = ctx.get("core_set_team_state_artifact_last_report")
    core_set_wf_last_report = ctx.get("core_set_wf_last_report")
    csv_path = ctx.get("csv_path")
    # --- extracted block body ---
    st.markdown("### Parâmetros")
    st.caption("Central de ajustes. Use Aplicar para enviar mudanças e Voltar ao padrão para resetar tudo.")
    with st.expander("Checklist de validação final", expanded=False):
        st.caption("Use está lista antes de fechar versão/entrega.")
        st.markdown(
            "- 1) Conferir confronto base: `WE x TES`, `LNG x TT`, `GEN.G x T1`\n"
            "- 2) Validar ML Mapa e Série (sem mudança ao salvar sem editar)\n"
            "- 3) Validar Consistência: Kills/Torres/Tempo com linha principal\n"
            "- 4) Validar ML + Totais: Modelo, Hist, Mix com linhas específicas\n"
            "- 5) Validar Players: auto-preencher, analisar, odds e médias\n"
            "- 6) Validar Campeões: colagem de draft + lane matchups\n"
            "- 7) Validar `ml_trace_last.json` com `p_map_used` e traces preenchidos\n"
            "- 8) Rodar manutenção: retreino/params/walk-forward se sinalizado\n"
            "- 9) Revisar strings visuais (sem caracteres quebrados)\n"
            "- 10) Executar suíte de testes local antes de versão final"
        )
    with st.expander("Pré-entrega (automático)", expanded=False):
        st.caption("Roda checagens automáticas e mostra Aprovado/Reprovado por item.")
        st.session_state.setdefault("pre_delivery_daily_with_smoke", True)
        st.session_state.setdefault("pre_delivery_daily_fast_mode", True)
        d1, d2 = st.columns(2)
        with d1:
            st.checkbox("Rotina diária inclui smoke", key="pre_delivery_daily_with_smoke")
        with d2:
            st.checkbox("Rotina diária modo rápido", key="pre_delivery_daily_fast_mode")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            run_pre = st.button("Rodar pré-entrega", width='stretch', key="pre_delivery_run")
        with b2:
            run_pre_smoke = st.button("Rodar pré-entrega + smoke", width='stretch', key="pre_delivery_run_smoke")
        with b3:
            if st.button("Gerar baseline âncoras", width='stretch', key="pre_delivery_make_anchor_baseline"):
                try:
                    snap = _compute_anchor_snapshot(csv_path)
                    _ODDS_ANCHOR_BASELINE_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
                    st.success(f"Baseline de âncoras salvo: {_ODDS_ANCHOR_BASELINE_PATH.name}")
                except (OSError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as _exc_anchor:
                    st.error(f"Falha ao gerar baseline: {_exc_anchor}")
        with b4:
            run_daily = st.button("Rodar rotina diária (1 clique)", width='stretch', key="pre_delivery_run_daily")

        if run_daily:
            with st.spinner("Executando rotina diária (fit + validações + pré-entrega)..."):
                rep_daily = _run_daily_pipeline(
                    APP_ROOT,
                    csv_path_s=str(csv_path),
                    artifact_path_s=str(artifact_path),
                    run_smoke=bool(st.session_state.get("pre_delivery_daily_with_smoke", True)),
                    fast_mode=bool(st.session_state.get("pre_delivery_daily_fast_mode", True)),
                )
            core_set_daily_pipeline_report(st.session_state, rep_daily)

        if run_pre or run_pre_smoke:
            with st.spinner("Executando validações de pré-entrega..."):
                rep = _run_pre_delivery_checks(APP_ROOT, csv_path, artifact_path, run_smoke=bool(run_pre_smoke))
            core_set_pre_delivery_report(st.session_state, rep)

        _rep_pre = core_get_pre_delivery_report(st.session_state)
        if isinstance(_rep_pre, dict):
            ok_n = int(_rep_pre.get("ok_count", 0) or 0)
            total_n = int(_rep_pre.get("total_count", 0) or 0)
            all_ok = bool(_rep_pre.get("all_ok", False))
            st.caption(f"Ultima execucao: {_rep_pre.get('generated_at', '-')!s}")
            if all_ok:
                st.success(f"Pré-entrega aprovada ({ok_n}/{total_n}).")
            else:
                st.warning(f"Pré-entrega com pendências ({ok_n}/{total_n}).")
                fx1, fx2 = st.columns(2)
                with fx1:
                    _auto_fix_with_smoke = st.checkbox(
                        "Auto-correção inclui smoke",
                        key="pre_delivery_auto_fix_with_smoke",
                    )
                with fx2:
                    if st.button("Auto-corrigir pendências comuns", width='stretch', key="pre_delivery_auto_fix"):
                        with st.spinner("Aplicando auto-correção (fast) e revalidando..."):
                            rep_daily = _run_daily_pipeline(
                                APP_ROOT,
                                csv_path_s=str(csv_path),
                                artifact_path_s=str(artifact_path),
                                run_smoke=bool(_auto_fix_with_smoke),
                                fast_mode=True,
                            )
                        core_set_daily_pipeline_report(st.session_state, rep_daily)
                        st.rerun()
            _df_pre = pd.DataFrame(list(_rep_pre.get("rows", []) or []))
            if not _df_pre.empty:
                _show_cols = [c for c in ["Item", "Status", "Detalhe"] if c in _df_pre.columns]
                st.dataframe(_df_pre[_show_cols], width='stretch', hide_index=True)
        _rep_daily = core_get_daily_pipeline_report(st.session_state)
        if isinstance(_rep_daily, dict):
            _ok_daily = bool(_rep_daily.get("ok", False))
            if _ok_daily:
                st.success("Rotina diária concluída com sucesso.")
            else:
                st.warning("Rotina diária concluiu com pendências. Veja etapas abaixo.")
            _df_daily = pd.DataFrame(list(_rep_daily.get("steps", []) or []))
            if not _df_daily.empty:
                st.dataframe(_df_daily, width='stretch', hide_index=True)
        if isinstance(_rep_pre, dict):
            st.download_button(
                "Baixar relatório pré-entrega (JSON)",
                data=json.dumps(_rep_pre, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="pre_delivery_report.json",
                mime="application/json",
                key="pre_delivery_download_json",
                on_click="ignore",
            )
            try:
                _bundle = {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "pre_delivery": _rep_pre,
                    "layer_validation": core_get_lv_last_report(st.session_state),
                    "calibration_diag": core_get_cal_diag_last_report(st.session_state),
                    "odds_anchor_baseline": (
                        json.loads(_ODDS_ANCHOR_BASELINE_PATH.read_text(encoding="utf-8"))
                        if _ODDS_ANCHOR_BASELINE_PATH.exists()
                        else None
                    ),
                }
                st.download_button(
                    "Baixar bundle release (JSON)",
                    data=json.dumps(_bundle, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="release_bundle.json",
                    mime="application/json",
                    key="pre_delivery_release_bundle",
                    on_click="ignore",
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

    _lap_opts = [
        "Liga 15", "Liga 10", "Liga 5", "Liga Ano",
        "H2H 15", "H2H 10", "H2H 5", "H2H Ano",
        "Média Liga+H2H 15", "Média Liga+H2H 10", "Média Liga+H2H 5", "Média Liga+H2H Ano",
    ]

    def _ui_key(k: str) -> str:
        return f"ctl_{k}"

    def _safe_ui_rerun() -> None:
        _request_streamlit_rerun()

    def _ui_param_value(param_key: str, default: Any) -> Any:
        return st.session_state.get(_ui_key(param_key), st.session_state.get(param_key, default))

    def _collect_pending_param_values(*, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        vals: Dict[str, Any] = {}
        for param_key, param_default in _PARAM_DEFAULTS.items():
            vals[param_key] = _ui_param_value(param_key, param_default)
        if isinstance(overrides, Mapping):
            vals.update(dict(overrides))
        return vals

    def _queue_param_values(
        *,
        overrides: Mapping[str, Any] | None = None,
        flash_msg: str = "",
        save_default: bool = False,
    ) -> None:
        core_set_pending_apply_param_values(
            st.session_state,
            _collect_pending_param_values(overrides=overrides),
        )
        if flash_msg:
            core_set_flash_saved_profile_msg(st.session_state, flash_msg)
        if save_default:
            st.session_state["_pending_save_settings_after_apply"] = True
        _safe_ui_rerun()

    def _set_session_float_values(values: Mapping[str, Any]) -> None:
        for key, value in dict(values).items():
            st.session_state[str(key)] = float(value)

    for _k, _v in _PARAM_DEFAULTS.items():
        _uk = _ui_key(_k)
        if _uk not in st.session_state:
            st.session_state[_uk] = st.session_state.get(_k, _v)

    st.markdown("#### Resumo")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Odd máxima (Resumo)", min_value=1.01, max_value=10.0, step=0.01, key=_ui_key("resumo_max_odd"))
        st.checkbox("Exigir Liga", key=_ui_key("resumo_req_liga"))
        st.checkbox("Exigir H2H", key=_ui_key("resumo_req_h2h"))
        st.checkbox("Exigir Modelo", key=_ui_key("resumo_req_model"))
        st.slider("Min amostra Liga", 0, 80, step=1, key=_ui_key("resumo_min_sample_liga"))
        st.slider("Min amostra H2H", 0, 30, step=1, key=_ui_key("resumo_min_sample_h2h"))
    with c2:
        st.checkbox("Mostrar colunas de frequência ajustada", key=_ui_key("resumo_show_laplace_cols"))
        st.checkbox("Usar StatsCamps", key=_ui_key("resumo_sc_use"))
        st.slider("Limiar StatsCamps", 50, 90, key=_ui_key("resumo_sc_thr"))
        st.slider("Shrink histórico (N)", 0, 80, step=1, key=_ui_key("resumo_hist_shrink_n"))
        _idx_lap = _lap_opts.index(str(st.session_state.get(_ui_key("resumo_laplace_pick"), "Liga 15"))) if str(st.session_state.get(_ui_key("resumo_laplace_pick"), "Liga 15")) in _lap_opts else 0
        if str(st.session_state.get(_ui_key("resumo_laplace_pick"), "") or "") not in _lap_opts:
            st.session_state[_ui_key("resumo_laplace_pick")] = _lap_opts[_idx_lap]
        st.selectbox("Fonte de frequência para Hist/Mix", options=_lap_opts, key=_ui_key("resumo_laplace_pick"))
    with c3:
        st.slider("Peso Mix: Modelo", 0, 100, key=_ui_key("resumo_mix_w_model"))
        st.slider("Peso Mix: Frequência", 0, 100, key=_ui_key("resumo_mix_w_laplace"))
        st.slider("Peso Mix: StatsCamps", 0, 100, key=_ui_key("resumo_mix_w_sc"))
        st.selectbox(
            "Guardrail (perfil)",
            options=["Conservador", "Equilibrado", "Agressivo"],
            key=_ui_key("resumo_guardrail_profile"),
        )
        st.caption("Ajuste fino por grupo de liga (Major/LCK+LPL, Mid, Minor).")
        st.slider("Força Major", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_strength_major"))
        st.slider("Força Mid", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_strength_mid"))
        st.slider("Força Minor", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_strength_minor"))
        st.slider("Target N Major", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_target_major"))
        st.slider("Target N Mid", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_target_mid"))
        st.slider("Target N Minor", 0.60, 1.40, step=0.05, key=_ui_key("resumo_guardrail_target_minor"))

    st.markdown("#### ML + Totais")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.selectbox("Preset ML+Totais", options=["Conservador", "Neutro", "Agressivo"], key=_ui_key("resumo_combo_preset"))
        st.checkbox("Ativar mismatch no Modelo (experimental)", key=_ui_key("resumo_combo_use_mismatch"))
    with c5:
        st.slider("Peso histórico condicional", 0.0, 1.0, step=0.05, key=_ui_key("resumo_combo_hist_w"))
        st.slider("Peso estilo over/under", 0.0, 0.5, step=0.02, key=_ui_key("resumo_combo_style_w"))
    with c6:
        st.slider("Peso forma atual", 0.0, 0.40, step=0.02, key=_ui_key("resumo_combo_form_w"))

    st.markdown("#### Perfil ML (1 clique)")
    _ml_strategy_profiles: Dict[str, Dict[str, Any]] = {
        "Conservador": {
            "values": {
                "ml_strategy_profile": "Conservador",
                "ml_microseg_enabled": True,
                "ml_microseg_strength": 0.40,
                "ml_microseg_min_n": 45,
                "ml_microseg_shrink_k": 130.0,
                "ml_microseg_lookback_games": 1600,
                "ml_microseg_cap_pp": 5.0,
                "ml_microseg_hierarchical": True,
                "ml_microseg_hier_decay": 0.88,
                "ml_uncertainty_enabled": True,
                "ml_uncertainty_strength": 0.72,
                "ml_uncertainty_prior_games": 30.0,
                "ml_uncertainty_max_shrink": 0.75,
                "ml_uncertainty_exclude_leagues": "LCK,LPL",
                "ml_marginal_trust_enabled": True,
                "ml_marginal_trust_strength": 0.80,
                "ml_marginal_trust_prior_games": 36.0,
                "ml_marginal_trust_max_shrink": 0.82,
                "ml_marginal_trust_cap_prob": 0.78,
                "ml_marginal_trust_score_soft": 72.0,
                "ml_marginal_trust_score_hard": 52.0,
                "ml_marginal_trust_exclude_leagues": "LCK,LPL,LEC,LCS,CBLOL,LCP,PCS,VCS",
                "ml_marginal_trust_band_tuning": True,
                "ml_form_blend_enabled": True,
                "ml_form_use_fitted": True,
                "ml_form_blend_max_weight": 0.60,
                "ml_form_blend_knee_games": 8.0,
                "ml_hybrid_blend_enabled": True,
                "ml_hybrid_blend_max_weight": 0.50,
                "ml_hybrid_blend_knee_games": 16.0,
                "ml_conf_guard_enabled": True,
                "ml_conf_guard_min_games": 14.0,
                "ml_conf_guard_max_shrink": 0.40,
                "ml_patch_guard_enabled": True,
                "ml_patch_guard_warn_shrink": 0.10,
                "ml_patch_guard_refit_shrink": 0.20,
                "ml_series_gamma_use_fitted": True,
                "fusion_profile_preset": "Conservador (Team-first)",
                "fusion_team_weight_early": 0.70,
                "fusion_team_weight_mid": 0.84,
                "fusion_team_weight_playoffs": 0.90,
                "fusion_early_season_team_weight": 0.68,
                "fusion_transfer_boost": 0.20,
                "fusion_divergence_pp_cap": 16.0,
                "fusion_divergence_low_coverage": 0.45,
                "fusion_divergence_shrink": 0.55,
            },
            "pros": [
                "Maior estabilidade entre reruns e menos oscilacao com amostra curta.",
                "Reduz extremos quando lineup ou recorte ainda estao ruidosos.",
                "Ideal para banca defensiva e filtros mais rigidos.",
            ],
            "cons": [
                "Pode reagir devagar a mudancas reais de roster/meta.",
                "Perde parte do upside em spots de edge recente.",
            ],
        },
        "Equilibrado (Recomendado)": {
            "values": {
                "ml_strategy_profile": "Equilibrado (Recomendado)",
                "ml_microseg_enabled": True,
                "ml_microseg_strength": 0.60,
                "ml_microseg_min_n": 30,
                "ml_microseg_shrink_k": 90.0,
                "ml_microseg_lookback_games": 1200,
                "ml_microseg_cap_pp": 8.0,
                "ml_microseg_hierarchical": True,
                "ml_microseg_hier_decay": 0.80,
                "ml_uncertainty_enabled": True,
                "ml_uncertainty_strength": 0.55,
                "ml_uncertainty_prior_games": 22.0,
                "ml_uncertainty_max_shrink": 0.65,
                "ml_uncertainty_exclude_leagues": "LCK",
                "ml_marginal_trust_enabled": True,
                "ml_marginal_trust_strength": 0.70,
                "ml_marginal_trust_prior_games": 28.0,
                "ml_marginal_trust_max_shrink": 0.72,
                "ml_marginal_trust_cap_prob": 0.84,
                "ml_marginal_trust_score_soft": 68.0,
                "ml_marginal_trust_score_hard": 46.0,
                "ml_marginal_trust_exclude_leagues": "LCK,LPL,LEC,LCS,CBLOL,LCP,PCS,VCS",
                "ml_marginal_trust_band_tuning": True,
                "ml_form_blend_enabled": True,
                "ml_form_use_fitted": True,
                "ml_form_blend_max_weight": 0.75,
                "ml_form_blend_knee_games": 6.0,
                "ml_hybrid_blend_enabled": True,
                "ml_hybrid_blend_max_weight": 0.66,
                "ml_hybrid_blend_knee_games": 14.0,
                "ml_conf_guard_enabled": True,
                "ml_conf_guard_min_games": 12.0,
                "ml_conf_guard_max_shrink": 0.28,
                "ml_patch_guard_enabled": True,
                "ml_patch_guard_warn_shrink": 0.08,
                "ml_patch_guard_refit_shrink": 0.16,
                "ml_series_gamma_use_fitted": True,
                "fusion_profile_preset": "Balanceado (Recomendado)",
                "fusion_team_weight_early": 0.58,
                "fusion_team_weight_mid": 0.75,
                "fusion_team_weight_playoffs": 0.82,
                "fusion_early_season_team_weight": 0.55,
                "fusion_transfer_boost": 0.35,
                "fusion_divergence_pp_cap": 22.0,
                "fusion_divergence_low_coverage": 0.30,
                "fusion_divergence_shrink": 0.35,
            },
            "pros": [
                "Melhor equilibrio entre robustez estatistica e sensibilidade.",
                "Boa consistencia para LPL/LCK sem ficar travado demais.",
                "Base recomendada para uso diario e evolucao gradual.",
            ],
            "cons": [
                "Nao e o mais rapido em viradas bruscas de forma.",
                "Nao e o mais agressivo para buscar odds extremas.",
            ],
        },
        "Agressivo": {
            "values": {
                "ml_strategy_profile": "Agressivo",
                "ml_microseg_enabled": True,
                "ml_microseg_strength": 0.78,
                "ml_microseg_min_n": 18,
                "ml_microseg_shrink_k": 55.0,
                "ml_microseg_lookback_games": 900,
                "ml_microseg_cap_pp": 12.0,
                "ml_microseg_hierarchical": True,
                "ml_microseg_hier_decay": 0.70,
                "ml_uncertainty_enabled": True,
                "ml_uncertainty_strength": 0.35,
                "ml_uncertainty_prior_games": 12.0,
                "ml_uncertainty_max_shrink": 0.45,
                "ml_uncertainty_exclude_leagues": "",
                "ml_marginal_trust_enabled": True,
                "ml_marginal_trust_strength": 0.48,
                "ml_marginal_trust_prior_games": 16.0,
                "ml_marginal_trust_max_shrink": 0.50,
                "ml_marginal_trust_cap_prob": 0.90,
                "ml_marginal_trust_score_soft": 62.0,
                "ml_marginal_trust_score_hard": 40.0,
                "ml_marginal_trust_exclude_leagues": "LCK,LPL,LEC,LCS,CBLOL,LCP,PCS,VCS",
                "ml_marginal_trust_band_tuning": True,
                "ml_form_blend_enabled": True,
                "ml_form_use_fitted": True,
                "ml_form_blend_max_weight": 0.88,
                "ml_form_blend_knee_games": 5.0,
                "ml_hybrid_blend_enabled": True,
                "ml_hybrid_blend_max_weight": 0.82,
                "ml_hybrid_blend_knee_games": 10.0,
                "ml_conf_guard_enabled": True,
                "ml_conf_guard_min_games": 10.0,
                "ml_conf_guard_max_shrink": 0.25,
                "ml_patch_guard_enabled": True,
                "ml_patch_guard_warn_shrink": 0.10,
                "ml_patch_guard_refit_shrink": 0.20,
                "ml_series_gamma_use_fitted": True,
                "fusion_profile_preset": "Agressivo (Players-first)",
                "fusion_team_weight_early": 0.46,
                "fusion_team_weight_mid": 0.62,
                "fusion_team_weight_playoffs": 0.72,
                "fusion_early_season_team_weight": 0.44,
                "fusion_transfer_boost": 0.50,
                "fusion_divergence_pp_cap": 30.0,
                "fusion_divergence_low_coverage": 0.22,
                "fusion_divergence_shrink": 0.20,
            },
            "pros": [
                "Reage rapido a momentum, lineup nova e meta recente.",
                "Aumenta chance de capturar edges de curto prazo.",
                "Melhor para quem aceita maior volatilidade nas odds.",
            ],
            "cons": [
                "Mais sensivel a ruido de amostra curta.",
                "Pode exagerar favoritos/zebras em ligas menos estaveis.",
            ],
        },
    }
    _ml_profile_opts = list(_ml_strategy_profiles.keys())
    _ml_profile_cur = str(
        _ui_param_value("ml_strategy_profile", "Equilibrado (Recomendado)")
        or "Equilibrado (Recomendado)"
    )
    if _ml_profile_cur not in _ml_profile_opts:
        _ml_profile_cur = "Equilibrado (Recomendado)"
    _ml_profile_idx = _ml_profile_opts.index(_ml_profile_cur) if _ml_profile_cur in _ml_profile_opts else 0
    if str(st.session_state.get(_ui_key("ml_strategy_profile"), "") or "") not in _ml_profile_opts:
        st.session_state[_ui_key("ml_strategy_profile")] = _ml_profile_opts[_ml_profile_idx]
    st.selectbox(
        "Perfil de estrategia ML",
        options=_ml_profile_opts,
        key=_ui_key("ml_strategy_profile"),
    )
    _ml_profile_sel = str(_ui_param_value("ml_strategy_profile", _ml_profile_cur) or _ml_profile_cur)
    _ml_profile_cfg = _ml_strategy_profiles.get(_ml_profile_sel, _ml_strategy_profiles["Equilibrado (Recomendado)"])
    _mpp1, _mpp2 = st.columns(2)
    with _mpp1:
        st.caption("Pros")
        for _ln in list(_ml_profile_cfg.get("pros") or []):
            st.write(f"+ {_ln}")
    with _mpp2:
        st.caption("Contras")
        for _ln in list(_ml_profile_cfg.get("cons") or []):
            st.write(f"- {_ln}")
    _mpb1, _mpb2 = st.columns(2)
    with _mpb1:
        if st.button("Aplicar perfil ML", key="ml_profile_apply_btn", width='stretch'):
            _queue_param_values(
                overrides={
                    **dict(_ml_profile_cfg.get("values") or {}),
                    "ml_strategy_profile": _ml_profile_sel,
                },
                flash_msg=f"Perfil ML aplicado: {_ml_profile_sel}.",
            )
    with _mpb2:
        if st.button("Aplicar perfil ML e salvar padrão", key="ml_profile_apply_save_btn", width='stretch'):
            _queue_param_values(
                overrides={
                    **dict(_ml_profile_cfg.get("values") or {}),
                    "ml_strategy_profile": _ml_profile_sel,
                },
                save_default=True,
            )

    st.markdown("#### Motor / Cálculo")
    c7, c8, c9 = st.columns(3)
    with c7:
        st.selectbox("ML engine", options=["mlcore_v2", "elo_season_players"], key=_ui_key("ml_engine"))
        st.checkbox("Comparar com engine alternativa", key=_ui_key("ml_engine_compare"))
        st.slider("Alerta divergencia (pp)", 5, 30, key=_ui_key("ml_engine_alert_pp"))
        st.checkbox("Calibração fina por contexto (liga/split/side)", key=_ui_key("ml_microseg_enabled"))
        st.slider("Calibração fina: força", 0.0, 1.0, step=0.05, key=_ui_key("ml_microseg_strength"))
        st.slider("Calibração fina: amostra mínima", 10, 200, step=5, key=_ui_key("ml_microseg_min_n"))
        st.checkbox("Incerteza explicita (sample-aware)", key=_ui_key("ml_uncertainty_enabled"))
        st.slider("Incerteza: forca", 0.0, 1.0, step=0.05, key=_ui_key("ml_uncertainty_strength"))
        st.checkbox("Guardiao ligas marginais (confianca)", key=_ui_key("ml_marginal_trust_enabled"))
        st.checkbox("Marginal: auto-tuning por faixa de liga", key=_ui_key("ml_marginal_trust_band_tuning"))
        st.slider("Marginal: forca", 0.0, 1.0, step=0.05, key=_ui_key("ml_marginal_trust_strength"))
        st.checkbox("Abrir 'Detalhar cálculo' por padrão", key=_ui_key("ml_show_calc_details_default"))
        st.checkbox("Guardiao de consistencia ML", key=_ui_key("ml_consistency_guard"))
        st.number_input("Tolerancia drift (pp)", min_value=0.00, max_value=1.00, step=0.01, key=_ui_key("ml_consistency_tol_pp"))
    with c8:
        st.selectbox("Pesos", options=["Recencia (half-life)", "Uniforme"], key=_ui_key("weight_mode"))
        st.slider("Half-life (dias)", 30, 365, step=10, key=_ui_key("half_life_days"))
        st.slider("Simulacoes serie", 3000, 60000, step=1000, key=_ui_key("n_sims"))
        st.slider("Calibração fina: shrink k", 10.0, 300.0, step=5.0, key=_ui_key("ml_microseg_shrink_k"))
        st.slider("Calibração fina: histórico (jogos)", 200, 5000, step=100, key=_ui_key("ml_microseg_lookback_games"))
        st.slider("Incerteza: prior games", 2.0, 120.0, step=1.0, key=_ui_key("ml_uncertainty_prior_games"))
        st.slider("Marginal: prior games", 2.0, 180.0, step=1.0, key=_ui_key("ml_marginal_trust_prior_games"))
    with c9:
        st.selectbox("Totais", options=["map", "series"], key=_ui_key("totals_mode"))
        st.selectbox("ML (combos)", options=["map", "series"], key=_ui_key("ml_mode"))
        st.selectbox("Stats map_mode", options=["avg", "map1", "map2", "map3", "map4", "map5"], key=_ui_key("map_mode_ui"))
        st.selectbox("League mode", options=["auto", "all", "fixed"], key=_ui_key("league_mode"))
        st.slider("Calibração fina: limite de ajuste (pp)", 0.0, 20.0, step=0.5, key=_ui_key("ml_microseg_cap_pp"))
        st.checkbox("Calibração fina hierárquica (mix)", key=_ui_key("ml_microseg_hierarchical"))
        st.slider("Calibração fina: decaimento da hierarquia", 0.30, 1.00, step=0.05, key=_ui_key("ml_microseg_hier_decay"))
        st.slider("Incerteza: max shrink", 0.0, 0.95, step=0.01, key=_ui_key("ml_uncertainty_max_shrink"))
        st.slider("Marginal: max shrink", 0.0, 0.95, step=0.01, key=_ui_key("ml_marginal_trust_max_shrink"))
        st.slider("Marginal: cap prob max", 0.55, 0.95, step=0.01, key=_ui_key("ml_marginal_trust_cap_prob"))
        st.slider("Marginal: score soft", 40.0, 95.0, step=1.0, key=_ui_key("ml_marginal_trust_score_soft"))
        st.slider("Marginal: score hard", 20.0, 90.0, step=1.0, key=_ui_key("ml_marginal_trust_score_hard"))
        st.text_input(
            "Incerteza: excluir ligas (csv)",
            key=_ui_key("ml_uncertainty_exclude_leagues"),
            help="Ex.: LCK, LCK CL. Nessas ligas a camada de incerteza não é aplicada.",
        )
        st.text_input(
            "Marginal: excluir ligas (csv)",
            key=_ui_key("ml_marginal_trust_exclude_leagues"),
            help="Ex.: LCK,LPL,LEC. Nessas ligas o guardiao marginal não e aplicado.",
        )
        st.selectbox("Seed simulacao", options=["Deterministico (confronto+config)", "Fixo manual"], key=_ui_key("sim_seed_mode"))
        st.number_input("Seed manual", min_value=1, max_value=2_147_483_647, step=1, key=_ui_key("sim_seed_manual"))
        st.caption(f"Seed atual (sessao): {int(st.session_state.get('_sim_seed_series', 1337) or 1337)}")

    st.markdown("#### Lineup (Players)")
    c10, c11 = st.columns(2)
    with c10:
        st.checkbox("Aplicar ajuste por lineup", key=_ui_key("use_lineup_adjust"))
        st.slider("Lineup peso (pontos)", 0, 200, step=5, key=_ui_key("lineup_lambda_points"))
        st.selectbox("Lineup delta mode", options=["saturado", "linear"], key=_ui_key("lineup_delta_mode"))
    with c11:
        st.slider("Lineup min GP", 0, 50, step=1, key=_ui_key("lineup_min_gp"))
        st.slider("Lineup shrink m", 1, 30, step=1, key=_ui_key("lineup_shrink_m"))
        st.slider("Lineup delta cap", 0.20, 3.00, step=0.05, key=_ui_key("lineup_delta_cap"))
        st.slider("Lineup delta slope", 0.20, 4.00, step=0.05, key=_ui_key("lineup_delta_slope"))

    with st.expander("Players: pesos por lane e liga", expanded=False):
        _pw1, _pw2 = st.columns(2)
        with _pw1:
            if st.button("Preset competitivo (balanceado)", key="players_weights_preset_comp", width='stretch'):
                _preset_vals = {
                    "ml_players_lane_w_top": 0.95,
                    "ml_players_lane_w_jungle": 1.10,
                    "ml_players_lane_w_mid": 1.15,
                    "ml_players_lane_w_adc": 1.10,
                    "ml_players_lane_w_support": 0.95,
                    "ml_players_lg_w_lpl": 1.10,
                    "ml_players_lg_w_lck": 1.10,
                    "ml_players_lg_w_emea": 1.00,
                    "ml_players_lg_w_na": 0.95,
                    "ml_players_lg_w_br": 0.90,
                    "ml_players_lg_w_apac": 0.92,
                    "ml_players_lg_w_other": 0.90,
                }
                _set_session_float_values(_preset_vals)
                _save_app_settings_to_disk()
                st.success("Preset competitivo aplicado.")
                _safe_ui_rerun()
        with _pw2:
            if st.button("Resetar pesos (1.00)", key="players_weights_reset", width='stretch'):
                _set_session_float_values({k: 1.0 for k in [
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
                ]})
                _save_app_settings_to_disk()
                st.success("Pesos resetados para 1.00.")
                _safe_ui_rerun()
        lw1, lw2 = st.columns(2)
        with lw1:
            st.slider("Lane Top", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_top"))
            st.slider("Lane Jungle", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_jungle"))
            st.slider("Lane Mid", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_mid"))
            st.slider("Lane ADC/Bot", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_adc"))
            st.slider("Lane Support", 0.50, 1.80, step=0.05, key=_ui_key("ml_players_lane_w_support"))
        with lw2:
            st.slider("Liga LPL (CN)", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_lpl"))
            st.slider("Liga LCK (KR)", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_lck"))
            st.slider("Liga EMEA", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_emea"))
            st.slider("Liga NA", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_na"))
            st.slider("Liga BR", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_br"))
            st.slider("Liga APAC", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_apac"))
            st.slider("Liga OTHER", 0.60, 1.60, step=0.05, key=_ui_key("ml_players_lg_w_other"))

    with st.expander("Fusão Team x Players (coerência)", expanded=False):
        _fusion_profiles: Dict[str, Dict[str, Any]] = {
            "Balanceado (Recomendado)": {
                "values": {
                    "fusion_team_weight_early": 0.58,
                    "fusion_team_weight_mid": 0.75,
                    "fusion_team_weight_playoffs": 0.82,
                    "fusion_early_season_team_weight": 0.55,
                    "fusion_season_knee_games": 18.0,
                    "fusion_coverage_power": 1.50,
                    "fusion_transfer_boost": 0.35,
                    "fusion_divergence_pp_cap": 22.0,
                    "fusion_divergence_low_coverage": 0.30,
                    "fusion_divergence_shrink": 0.35,
                    "fusion_roster_cont_thresh_base": 0.55,
                    "fusion_roster_cont_thresh_early_boost": 0.10,
                    "fusion_roster_low_freshness_cut": 0.60,
                    "fusion_roster_low_freshness_boost": 0.05,
                },
                "pros": [
                    "Bom equilibrio entre historico do time e sinal dos players.",
                    "Menos oscilações em trocas pequenas de lineup.",
                    "Ponto de partida seguro para uso diario.",
                ],
                "cons": [
                    "Pode ficar conservador demais em rosters totalmente novos.",
                    "Nem sempre captura picos de forma muito rapido.",
                ],
            },
            "Conservador (Team-first)": {
                "values": {
                    "fusion_team_weight_early": 0.72,
                    "fusion_team_weight_mid": 0.82,
                    "fusion_team_weight_playoffs": 0.88,
                    "fusion_early_season_team_weight": 0.68,
                    "fusion_season_knee_games": 24.0,
                    "fusion_coverage_power": 1.90,
                    "fusion_transfer_boost": 0.20,
                    "fusion_divergence_pp_cap": 16.0,
                    "fusion_divergence_low_coverage": 0.45,
                    "fusion_divergence_shrink": 0.55,
                    "fusion_roster_cont_thresh_base": 0.48,
                    "fusion_roster_cont_thresh_early_boost": 0.06,
                    "fusion_roster_low_freshness_cut": 0.50,
                    "fusion_roster_low_freshness_boost": 0.03,
                },
                "pros": [
                    "Mais estavel e com menor chance de overreaction.",
                    "Boa opcao quando lineup de players esta ruidoso/incompleto.",
                ],
                "cons": [
                    "Pode demorar para reagir a mudancas reais de elenco.",
                    "Reduz upside em spots de player edge claro.",
                ],
            },
            "Agressivo (Players-first)": {
                "values": {
                    "fusion_team_weight_early": 0.46,
                    "fusion_team_weight_mid": 0.62,
                    "fusion_team_weight_playoffs": 0.72,
                    "fusion_early_season_team_weight": 0.44,
                    "fusion_season_knee_games": 14.0,
                    "fusion_coverage_power": 1.10,
                    "fusion_transfer_boost": 0.50,
                    "fusion_divergence_pp_cap": 30.0,
                    "fusion_divergence_low_coverage": 0.22,
                    "fusion_divergence_shrink": 0.20,
                    "fusion_roster_cont_thresh_base": 0.60,
                    "fusion_roster_cont_thresh_early_boost": 0.14,
                    "fusion_roster_low_freshness_cut": 0.68,
                    "fusion_roster_low_freshness_boost": 0.08,
                },
                "pros": [
                    "Reage rapido a trocas de roster e momentum de players.",
                    "Maior sensibilidade para detectar mismatch recente.",
                ],
                "cons": [
                    "Mais volatil e sujeito a ruido de amostra curta.",
                    "Pode gerar odds mais extremas em ligas com poucos jogos.",
                ],
            },
        }
        _fusion_profile_opts = list(_fusion_profiles.keys())
        _fusion_cur = str(
            _ui_param_value("fusion_profile_preset", "Balanceado (Recomendado)")
            or "Balanceado (Recomendado)"
        )
        if _fusion_cur not in _fusion_profile_opts:
            _fusion_cur = "Balanceado (Recomendado)"
        _fusion_idx = _fusion_profile_opts.index(_fusion_cur) if _fusion_cur in _fusion_profile_opts else 0
        if str(st.session_state.get(_ui_key("fusion_profile_preset"), "") or "") not in _fusion_profile_opts:
            st.session_state[_ui_key("fusion_profile_preset")] = _fusion_profile_opts[_fusion_idx]
        st.selectbox(
            "Perfil de fusao",
            options=_fusion_profile_opts,
            key=_ui_key("fusion_profile_preset"),
        )

        _fusion_selected = str(_ui_param_value("fusion_profile_preset", _fusion_cur) or _fusion_cur)
        _fusion_cfg = _fusion_profiles.get(_fusion_selected, _fusion_profiles["Balanceado (Recomendado)"])

        _pf_info1, _pf_info2 = st.columns(2)
        with _pf_info1:
            st.caption("Pros")
            for _line in list(_fusion_cfg.get("pros") or []):
                st.write(f"+ {_line}")
        with _pf_info2:
            st.caption("Contras")
            for _line in list(_fusion_cfg.get("cons") or []):
                st.write(f"- {_line}")

        _pf_b1, _pf_b2, _pf_b3 = st.columns(3)
        with _pf_b1:
            if st.button("Aplicar perfil", key="fusion_profile_apply_btn", width='stretch'):
                _queue_param_values(
                    overrides={
                        **dict(_fusion_cfg.get("values") or {}),
                        "fusion_profile_preset": _fusion_selected,
                    },
                    flash_msg=f"Perfil aplicado: {_fusion_selected}.",
                )
        with _pf_b2:
            if st.button("Aplicar e salvar padrão", key="fusion_profile_apply_save_btn", width='stretch'):
                _queue_param_values(
                    overrides={
                        **dict(_fusion_cfg.get("values") or {}),
                        "fusion_profile_preset": _fusion_selected,
                    },
                    save_default=True,
                )
        with _pf_b3:
            if st.button("Resetar para balanceado", key="fusion_profile_reset_btn", width='stretch'):
                _base_cfg = _fusion_profiles.get("Balanceado (Recomendado)", {})
                _queue_param_values(
                    overrides={
                        **dict(_base_cfg.get("values") or {}),
                        "fusion_profile_preset": "Balanceado (Recomendado)",
                    },
                    save_default=True,
                )

        _flash_msg = core_consume_flash_saved_profile_msg(st.session_state).strip()
        if _flash_msg:
            st.success(_flash_msg)

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.slider("Team weight (early)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_early"))
            st.slider("Team weight (mid)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_mid"))
            st.slider("Team weight (playoffs)", 0.0, 1.0, step=0.01, key=_ui_key("fusion_team_weight_playoffs"))
        with fc2:
            st.slider("Early season team weight", 0.0, 1.0, step=0.01, key=_ui_key("fusion_early_season_team_weight"))
            st.slider("Season knee games", 1.0, 80.0, step=1.0, key=_ui_key("fusion_season_knee_games"))
            st.slider("Coverage power", 0.0, 4.0, step=0.05, key=_ui_key("fusion_coverage_power"))
        with fc3:
            st.slider("Transfer boost", 0.0, 2.0, step=0.05, key=_ui_key("fusion_transfer_boost"))
            st.slider("Cap divergência (pp)", 5.0, 80.0, step=1.0, key=_ui_key("fusion_divergence_pp_cap"))
            st.slider("Low coverage gate", 0.0, 1.0, step=0.01, key=_ui_key("fusion_divergence_low_coverage"))
            st.slider("Pull-back divergência", 0.0, 1.0, step=0.01, key=_ui_key("fusion_divergence_shrink"))
            st.slider("Roster gate base (continuidade)", 0.30, 0.85, step=0.01, key=_ui_key("fusion_roster_cont_thresh_base"))
            st.slider("Roster gate extra (early season)", 0.00, 0.25, step=0.01, key=_ui_key("fusion_roster_cont_thresh_early_boost"))
            st.slider("Cutoff baixa freshness", 0.20, 0.90, step=0.01, key=_ui_key("fusion_roster_low_freshness_cut"))
            st.slider("Roster gate extra (baixa freshness)", 0.00, 0.20, step=0.01, key=_ui_key("fusion_roster_low_freshness_boost"))
    with st.expander("Ajuste por forma da temporada (data-driven)", expanded=False):
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            st.checkbox("Ativar blend de forma no ML", key=_ui_key("ml_form_blend_enabled"))
            st.checkbox("Usar parâmetros fitted (params_fitted.json)", key=_ui_key("ml_form_use_fitted"))
            st.slider("Peso máximo da forma", 0.0, 0.95, step=0.01, key=_ui_key("ml_form_blend_max_weight"))
        with sf2:
            st.slider("Knee de jogos (amostra)", 1.0, 40.0, step=1.0, key=_ui_key("ml_form_blend_knee_games"))
            st.slider("Beta prior", 0.5, 8.0, step=0.1, key=_ui_key("ml_form_blend_beta"))
            st.slider("Potência do sinal", 0.1, 3.0, step=0.1, key=_ui_key("ml_form_blend_signal_power"))
            st.slider("Paridade elite mesma liga", 0.0, 0.8, step=0.01, key=_ui_key("ml_form_blend_parity_strength"))
            st.markdown("**Override temporada atual**")
            st.slider("Min jogos por time (override)", 3, 30, step=1, key=_ui_key("ml_form_override_min_games"))
            st.slider("Divergência mínima p.p. (override)", 0.0, 30.0, step=0.5, key=_ui_key("ml_form_override_div_pp"))
            st.slider("Peso máximo da forma (override)", 0.0, 0.95, step=0.01, key=_ui_key("ml_form_override_max_w"))
        with sf3:
            st.checkbox("Ativar blend híbrido (momentum/early/macro)", key=_ui_key("ml_hybrid_blend_enabled"))
            st.slider("Peso máx híbrido", 0.0, 0.95, step=0.01, key=_ui_key("ml_hybrid_blend_max_weight"))
            st.slider("Knee híbrido (jogos)", 1.0, 80.0, step=1.0, key=_ui_key("ml_hybrid_blend_knee_games"))
            st.slider("Janela momentum", 3, 30, step=1, key=_ui_key("ml_hybrid_momentum_games"))
            st.slider("Beta prior híbrido", 0.5, 8.0, step=0.1, key=_ui_key("ml_hybrid_beta"))
            st.checkbox("Guardrail de confiança (baixa amostra)", key=_ui_key("ml_conf_guard_enabled"))
            st.slider("Min jogos (guardrail)", 1.0, 60.0, step=1.0, key=_ui_key("ml_conf_guard_min_games"))
            st.slider("Máx shrink para 50%", 0.0, 0.9, step=0.01, key=_ui_key("ml_conf_guard_max_shrink"))
            st.checkbox("Guardrail patch/meta por liga", key=_ui_key("ml_patch_guard_enabled"))
            st.slider("Shrink em status ATENÇÃO", 0.0, 0.5, step=0.01, key=_ui_key("ml_patch_guard_warn_shrink"))
            st.slider("Shrink em status REFIT", 0.0, 0.7, step=0.01, key=_ui_key("ml_patch_guard_refit_shrink"))
            st.slider("Bônus shrink liga afetada", 0.0, 0.3, step=0.01, key=_ui_key("ml_patch_guard_focus_bonus"))
            st.checkbox("Usar calibracao série BO3/BO5 fitted", key=_ui_key("ml_series_gamma_use_fitted"))

    with st.expander("Manutenção do ML", expanded=False):
        st.markdown("#### Walk-forward + Manutenção")
        _daily_csv_m, _hist_csvs_m, _artifact_cfg_m = _paths_for_ml(APP_ROOT)
        _csv_path_m = str(_daily_csv_m or "").strip()
        _csv_sig_m = list(_file_sig(_csv_path_m)) if _csv_path_m and Path(_csv_path_m).exists() else ["", 0, 0]
        _maint_state_path = APP_ROOT / "ops" / "maintenance_state.json"
        _maint_state = _read_json_file_safe(_maint_state_path)
        _last_csv_sig = list(_maint_state.get("csv_sig") or []) if isinstance(_maint_state, dict) else []
        _csv_changed = bool(_csv_sig_m and (_csv_sig_m != _last_csv_sig))
        _last_dec = _norm_gate_decision((_maint_state or {}).get("decision"))
        _last_at = str((_maint_state or {}).get("last_run_at") or "-")
        if _csv_changed:
            st.warning("CSV novo detectado. Recomendado: Rodar manutenção completa (12 etapas).")
        else:
            st.caption(f"última manutenção: {_last_at} | decisão={_last_dec}.")

        if st.button("Rodar manutenção completa (12 etapas)", key="ctl_run_full_maint_10", type="primary", width='stretch'):
            if not _csv_path_m or (not Path(_csv_path_m).exists()):
                st.error("CSV diário inválido/ausente para manutenção completa.")
            else:
                _phase12_speed = "fast"
                _steps = [
                    {
                        "name": "Fase 1/12: Data contract",
                        "cmd": [sys.executable, "tools/run_phase1_data_contract.py", "--timeout", "600"],
                        "report": APP_ROOT / "phase1_data_contract_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 2/12: Roster continuidade (1.2)",
                        "cmd": [sys.executable, "tools/eval_roster_continuity_phase12.py", "--speed", _phase12_speed, "--fit", "--write-params"],
                        "report": APP_ROOT / "phase12_roster_continuity_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 3/12: A-real gate",
                        "cmd": [sys.executable, "tools/run_phase2_areal_gate.py"],
                        "report": APP_ROOT / "phase2_areal_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 4/12: A-real tuning (liga/patch)",
                        "cmd": [sys.executable, "tools/run_phase2b_areal_tuning_gate.py", "--speed", "fast", "--write-params"],
                        "report": APP_ROOT / "phase2b_areal_tuning_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 5/12: B-real gate",
                        "cmd": [sys.executable, "tools/run_phase3_breal_gate.py"],
                        "report": APP_ROOT / "phase3_breal_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 6/12: Rank-SOS gate",
                        "cmd": [sys.executable, "tools/run_phase4_ranksos_gate.py"],
                        "report": APP_ROOT / "phase4_ranksos_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 7/12: Marginal tuning (bandas top/core/edge)",
                        "cmd": [
                            sys.executable,
                            "tools/run_phase4b_marginal_tuning_gate.py",
                            "--speed",
                            "fast",
                            "--max-cases",
                            "14",
                            "--write-params",
                        ],
                        "report": APP_ROOT / "phase4b_marginal_tuning_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 8/12: Calibração hierárquica",
                        "cmd": [sys.executable, "tools/run_phase5_hier_calibration_gate.py"],
                        "report": APP_ROOT / "phase5_hier_calibration_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 9/12: Série BO",
                        "cmd": [sys.executable, "tools/run_phase6_series_bo_gate.py"],
                        "report": APP_ROOT / "phase6_series_bo_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 10/12: Incerteza por camada",
                        "cmd": [sys.executable, "tools/run_phase7_layer_uncertainty_gate.py"],
                        "report": APP_ROOT / "phase7_layer_uncertainty_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 11/12: Operação contínua",
                        "cmd": [sys.executable, "tools/run_phase8_operations_gate.py", "--skip-runs"],
                        "report": APP_ROOT / "phase8_operations_report.json",
                        "key": "decision",
                    },
                    {
                        "name": "Fase 12/12: Guardrail final",
                        "cmd": [sys.executable, "tools/run_lacuna_guardrails.py", "--strict-phase-coverage", "--skip-runs"],
                        "report": APP_ROOT / "lacuna_guardrail_report.json",
                        "key": "decision",
                    },
                ]

                _prog = st.progress(0)
                _live = st.empty()
                _rows: List[Dict[str, Any]] = []
                _all_promote = True
                _full_log_parts: List[str] = []

                for _i, _stp in enumerate(_steps, start=1):
                    _live.info(f"{_stp['name']} - executando...")
                    _cp = subprocess.run(_stp["cmd"], cwd=str(APP_ROOT), capture_output=True, text=True)
                    _rep = _read_json_file_safe(_stp["report"])
                    _dec = _norm_gate_decision(_rep.get(_stp["key"]))
                    _ok = (int(_cp.returncode) == 0) and (_dec == "promote")
                    if not _ok:
                        _all_promote = False
                    _rows.append(
                        {
                            "Etapa": _stp["name"],
                            "Status": ("OK" if _ok else "Falha"),
                            "Decisão": _dec,
                            "ReturnCode": int(_cp.returncode),
                        }
                    )
                    _full_log_parts.append(
                        f"[{_stp['name']}]\n"
                        + str(_cp.stdout or "")
                        + (("\n" + str(_cp.stderr)) if str(_cp.stderr or "").strip() else "")
                    )
                    _prog.progress(int((_i / len(_steps)) * 100))
                    if _ok:
                        _live.success(f"{_stp['name']} - concluída com êxito.")
                    else:
                        _live.error(f"{_stp['name']} - falhou (decisão={_dec}, rc={int(_cp.returncode)}).")

                _action = "none"
                if _all_promote:
                    _tag = datetime.now().strftime("full_maint_%Y%m%d_%H%M%S")
                    _cp_pr = subprocess.run(
                        [sys.executable, "tools/promote_baseline.py", "--tag", _tag],
                        cwd=str(APP_ROOT),
                        capture_output=True,
                        text=True,
                    )
                    if int(_cp_pr.returncode) == 0:
                        _action = "promote_baseline"
                        _full_log_parts.append("[promote_baseline]\n" + str(_cp_pr.stdout or ""))
                    else:
                        _action = "promote_failed"
                        _all_promote = False
                        _full_log_parts.append(
                            "[promote_baseline]\n"
                            + str(_cp_pr.stdout or "")
                            + (("\n" + str(_cp_pr.stderr)) if str(_cp_pr.stderr or "").strip() else "")
                        )

                _state_payload = {
                    "version": "maintenance_state_v1",
                    "last_run_at": datetime.now().isoformat(timespec="seconds"),
                    "csv_path": _csv_path_m,
                    "csv_sig": _csv_sig_m,
                    "decision": ("promote" if _all_promote else "shadow_only"),
                    "action": _action,
                    "steps": _rows,
                }
                try:
                    _maint_state_path.parent.mkdir(parents=True, exist_ok=True)
                    _maint_state_path.write_text(json.dumps(_state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                except (OSError, TypeError, ValueError):
                    pass

                core_set_quick_maint_last_log(st.session_state, "\n\n".join(_full_log_parts).strip())
                st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
                if _all_promote:
                    st.success("Manutenção completa concluída com sucesso (10/10) e baseline promovido.")
                else:
                    st.error("Manutenção completa finalizou com pendências. Veja as etapas com falha e o log.")
        core_render_calibration_support_panels(
            session_state=st.session_state,
            app_root=APP_ROOT,
            artifact_path=str(artifact_path),
            ml_info=(_ml_info if isinstance(_ml_info, dict) else {}),
            ui_key_fn=_ui_key,
            format_num_fn=_format_num,
            paths_for_ml_fn=_paths_for_ml,
            run_walkforward_cached_fn=_run_walkforward_cached,
            wf_gate_multiplier_fn=_wf_gate_multiplier,
            get_wf_last_report_fn=core_get_wf_last_report,
            set_wf_last_report_fn=core_set_wf_last_report,
            get_wf_ml_corr_info_fn=core_get_wf_ml_corr_info,
            build_training_csv_list_fn=_build_training_csv_list,
            resolve_players_artifact_path_fn=_resolve_players_artifact_path,
            get_lv_last_report_fn=core_get_lv_last_report,
            set_lv_last_report_fn=core_set_lv_last_report,
            get_fit_scale_state_fn=core_get_fit_scale_state,
            run_layer_validation_cached_fn=_run_layer_validation_cached,
            load_fitted_params_cached_fn=_load_fitted_params_cached,
            file_sig_fn=_file_sig,
            fit_params_path=_FIT_PARAMS_PATH,
            generate_and_persist_ml_confidence_baseline_fn=_generate_and_persist_ml_confidence_baseline,
            ml_confidence_baseline_path=_ML_CONFIDENCE_BASELINE_PATH,
            load_ml_confidence_baseline_cached_fn=_load_ml_confidence_baseline_cached,
            set_ml_confidence_baseline_last_report_fn=core_set_ml_confidence_baseline_last_report,
            generate_and_persist_team_state_artifact_fn=_generate_and_persist_team_state_artifact,
            team_state_artifact_path=_TEAM_STATE_ARTIFACT_PATH,
            load_team_state_artifact_cached_fn=_load_team_state_artifact_cached,
            set_team_state_artifact_last_report_fn=core_set_team_state_artifact_last_report,
        )
    
        core_render_fit_calibration_panel(
            session_state=st.session_state,
            app_root=APP_ROOT,
            artifact_path=str(artifact_path),
            fit_params_path=_FIT_PARAMS_PATH,
            ui_key_fn=_ui_key,
            format_num_fn=_format_num,
            load_fitted_params_cached_fn=_load_fitted_params_cached,
            file_sig_fn=_file_sig,
            get_fit_scale_state_fn=core_get_fit_scale_state,
            paths_for_ml_fn=_paths_for_ml,
            fit_ml_params_run_fn=_fit_ml_params_run,
            enrich_areal_tuning_fn=_enrich_params_with_areal_prob_tuning,
            enrich_marginal_tuning_fn=_enrich_params_with_ml_marginal_band_coeffs,
            backup_file_if_exists_fn=_backup_file_if_exists,
            set_fit_base_scales_fn=core_set_fit_base_scales,
            load_artifact_fn=_load_artifact_cached,
            artifact_sig_fn=_artifact_sig,
        )
    
    st.caption("Paineis legados de backtest/diagnostico de calibracao foram removidos da interface. Use manutenção automática e pre-entrega para validacao quando necessario.")


    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Aplicar alterações", type="primary", width='stretch', key="ctl_apply_all"):
            _vals = {}
            for _k in _PARAM_DEFAULTS.keys():
                _vals[_k] = st.session_state.get(_ui_key(_k), _PARAM_DEFAULTS[_k])
            core_set_pending_apply_param_values(st.session_state, _vals)
            _request_streamlit_rerun()
    with b2:
        if st.button("Preset Excelente (valor)", width='stretch', key="ctl_preset_excellent"):
            st.session_state["_pending_apply_excellent"] = True
            _request_streamlit_rerun()
        if st.button("Preset 20/10 (agressivo)", width='stretch', key="ctl_preset_twenty_ten"):
            st.session_state["_pending_apply_twenty_ten"] = True
            _request_streamlit_rerun()
    with b3:
        if st.button("Voltar ao padrão", width='stretch', key="ctl_reset_all"):
            st.session_state["_pending_reset_all_tuning"] = True
            _request_streamlit_rerun()
    with b4:
        if st.button("Salvar configuração", width='stretch', key="ctl_save_all"):
            _save_app_settings_to_disk()
            st.success("Configuração salva em disco.")

    core_render_feature_flags_shadow_trace_panel(
        session_state=st.session_state,
        persist_combined_trace_file_fn=_persist_combined_trace_file,
        df_contract_view_fn=_df_contract_view,
    )

    with st.expander("Governanca ML (backtest / promocao / calibração)", expanded=False):
        _bt_report_path = APP_ROOT / "backtest_pipeline_abc_by_league_report.json"
        _anchor_report_path = APP_ROOT / "anchor_regression_report.json"
        _leak_report_path = APP_ROOT / "data_leakage_label_report.json"
        _lacuna_report_path = APP_ROOT / "lacuna_guardrail_report.json"
        _prod_refresh_report_path = APP_ROOT / "prod_refresh_report.json"
        _phase1_report_path = APP_ROOT / "phase1_data_contract_report.json"
        _phase12_report_path = APP_ROOT / "phase12_roster_continuity_report.json"
        _phase4b_report_path = APP_ROOT / "phase4b_marginal_tuning_report.json"
        _phase6_report_path = APP_ROOT / "phase6_series_bo_report.json"
        _phase7_report_path = APP_ROOT / "phase7_layer_uncertainty_report.json"
        _phase8_report_path = APP_ROOT / "phase8_operations_report.json"
        _gov_last_run = str(st.session_state.get("_gov_last_run", "") or "").strip()
        _gov_last_log = str(core_get_governance_last_log(st.session_state) or "").strip()

        def _read_json_safe(_p: Path) -> Dict[str, Any]:
            try:
                if _p.exists():
                    _obj = json.loads(_p.read_text(encoding="utf-8"))
                    return _obj if isinstance(_obj, dict) else {}
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return {}
            return {}

        def _render_status_message(message: Any, message_type: Any = "error") -> None:
            _msg = str(message or "")
            _fn = getattr(st, str(message_type or "error"), None)
            if callable(_fn):
                _fn(_msg)
            else:
                st.caption(_msg)

        _bt_rep = _read_json_safe(_bt_report_path)
        _ar = _read_json_safe(_anchor_report_path)
        _lrp = _read_json_safe(_leak_report_path)
        _lgr = _read_json_safe(_lacuna_report_path)
        _pr = _read_json_safe(_prod_refresh_report_path)
        _p1 = _read_json_safe(_phase1_report_path)
        _p12 = _read_json_safe(_phase12_report_path)
        _p4b = _read_json_safe(_phase4b_report_path)
        _p6 = _read_json_safe(_phase6_report_path)
        _p7 = _read_json_safe(_phase7_report_path)
        _p8 = _read_json_safe(_phase8_report_path)

        _gvr1, _gvr2, _gvr3, _gvr4 = st.columns([1, 1, 1, 1])
        with _gvr1:
            _run_gov_fast = st.button("Rodar governança (rápido)", width='stretch', key="gov_run_fast")
        with _gvr2:
            _run_anchor_only = st.button("Rodar apenas âncoras", width='stretch', key="gov_run_anchor_only")
        with _gvr3:
            _run_lacuna_full = st.button("Rodar lacunas (completo)", width='stretch', key="gov_run_lacuna_full")
        with _gvr4:
            if _gov_last_run:
                st.caption(f"última execução: {_gov_last_run}")
        _gvr5, _gvr6 = st.columns([1.2, 2.8])
        with _gvr5:
            _run_prod_refresh = st.button("Rodar Prod Refresh", width='stretch', key="gov_run_prod_refresh")
        with _gvr6:
            _pr_dec = str((_pr.get("final_decision") if isinstance(_pr, dict) else "") or "").strip().lower()
            _pr_gen = str((_pr.get("generated_at") if isinstance(_pr, dict) else "") or "").strip()
            _pr_action = str((_pr.get("action") if isinstance(_pr, dict) else "") or "").strip()
            if _pr_dec:
                st.caption(f"Prod Refresh atual: {_pr_dec} | ação={_pr_action or '-'} | gerado: {_pr_gen or '-'}")
        _gvp1, _gvp2 = st.columns([1.2, 2.8])
        with _gvp1:
            _run_phase1_gate = st.button("Rodar Fase 1 (data contract)", width='stretch', key="gov_run_phase1_data_contract")
        with _gvp2:
            _p1_dec_hint = str((_p1.get("decision") if isinstance(_p1, dict) else "") or "").strip().lower()
            _p1_gen_hint = str((_p1.get("generated_at") if isinstance(_p1, dict) else "") or "").strip()
            if _p1_dec_hint:
                st.caption(f"Phase 1 atual: {_p1_dec_hint} | gerado: {_p1_gen_hint or '-'}")
        _gvp3, _gvp4 = st.columns([1.2, 2.8])
        with _gvp3:
            _run_phase12 = st.button("Rodar Fase 1.2 (roster A/B)", width='stretch', key="gov_run_phase12_roster")
        with _gvp4:
            _p12_dec = str((_p12.get("decision") if isinstance(_p12, dict) else "") or "").strip().lower()
            _p12_gen = str((_p12.get("generated_at") if isinstance(_p12, dict) else "") or "").strip()
            if _p12_dec:
                st.caption(f"Phase 1.2 atual: {_p12_dec} | gerado: {_p12_gen or '-'}")
        _gvp5, _gvp6 = st.columns([1.2, 2.8])
        with _gvp5:
            _run_phase8_ops = st.button("Rodar Fase 8 (operação)", width='stretch', key="gov_run_phase8_ops")
        with _gvp6:
            _p8_dec = str((_p8.get("decision") if isinstance(_p8, dict) else "") or "").strip().lower()
            _p8_gen = str((_p8.get("generated_at") if isinstance(_p8, dict) else "") or "").strip()
            if _p8_dec:
                st.caption(f"Phase 8 atual: {_p8_dec} | gerado: {_p8_gen or '-'}")

        if _run_gov_fast or _run_anchor_only or _run_lacuna_full or _run_phase1_gate or _run_phase12 or _run_phase8_ops or _run_prod_refresh:
            _cmds: List[List[str]] = []
            if _run_phase1_gate:
                _cmds.append([sys.executable, "tools/run_phase1_data_contract.py", "--timeout", "600"])
            elif _run_phase12:
                _cmds.append([sys.executable, "tools/eval_roster_continuity_phase12.py", "--speed", "fast"])
            elif _run_phase8_ops:
                _cmds.append([sys.executable, "tools/run_phase8_operations_gate.py", "--timeout", "900"])
            elif _run_prod_refresh:
                _cmds.append([sys.executable, "tools/prod_refresh.py", "--mode", "quick", "--phase12-speed", "fast", "--promote-on-pass"])
            else:
                if _run_lacuna_full:
                    _cmds.append([sys.executable, "tools/run_lacuna_guardrails.py"])
                else:
                    _cmds.append([sys.executable, "tools/anchor_regression_suite.py"])
                if _run_gov_fast and (not _run_lacuna_full):
                    _cmds.append([sys.executable, "tools/check_data_leakage_and_labels.py"])
            with st.spinner("Executando governança..."):
                _gov_exec = core_run_governance_commands(_cmds, cwd=APP_ROOT)
            st.session_state["_gov_last_run"] = str(_gov_exec.get("last_run") or datetime.now().isoformat(timespec="seconds"))
            core_set_governance_last_log(st.session_state, "\n\n".join([str(x) for x in (_gov_exec.get("logs") or [])]))
            _gov_last_run = str(st.session_state.get("_gov_last_run", "") or "").strip()
            _gov_last_log = str(core_get_governance_last_log(st.session_state) or "").strip()
            _bt_rep = _read_json_safe(_bt_report_path)
            _ar = _read_json_safe(_anchor_report_path)
            _lrp = _read_json_safe(_leak_report_path)
            _lgr = _read_json_safe(_lacuna_report_path)
            _pr = _read_json_safe(_prod_refresh_report_path)
            _p1 = _read_json_safe(_phase1_report_path)
            _p12 = _read_json_safe(_phase12_report_path)
            _p4b = _read_json_safe(_phase4b_report_path)
            _p6 = _read_json_safe(_phase6_report_path)
            _p7 = _read_json_safe(_phase7_report_path)
            _p8 = _read_json_safe(_phase8_report_path)

        if _gov_last_log:
            with st.expander("Log da última execução de governança", expanded=False):
                st.code(_gov_last_log, language="text")

        _parts = {
            "stop_rules": core_normalize_governance_decision(((_bt_rep.get("promotion_decision") or {}) if isinstance(_bt_rep, dict) else {}).get("decision")),
            "anchors": core_normalize_governance_decision((_ar.get("decision") if isinstance(_ar, dict) else "")),
            "leakage_labels": core_normalize_governance_decision((_lrp.get("decision") if isinstance(_lrp, dict) else "")),
            "lacunas": core_normalize_governance_decision((_lgr.get("decision") if isinstance(_lgr, dict) else "")),
            "prod_refresh": core_normalize_governance_decision((_pr.get("final_decision") if isinstance(_pr, dict) else "")),
            "phase1_data_contract": core_normalize_governance_decision((_p1.get("decision") if isinstance(_p1, dict) else "")),
            "phase12_roster_ab": core_normalize_governance_decision((_p12.get("decision") if isinstance(_p12, dict) else "")),
            "phase4b_marginal_tuning": core_normalize_governance_decision((_p4b.get("decision") if isinstance(_p4b, dict) else "")),
            "phase6_series_bo": core_normalize_governance_decision((_p6.get("decision") if isinstance(_p6, dict) else "")),
            "phase7_layer_uncertainty": core_normalize_governance_decision((_p7.get("decision") if isinstance(_p7, dict) else "")),
            "phase8_operations": core_normalize_governance_decision((_p8.get("decision") if isinstance(_p8, dict) else "")),
        }
        _gov_combo = core_combine_governance_decisions(_parts)
        _final = str(_gov_combo.get("final") or "unknown")
        _final_msg = str(_gov_combo.get("message") or "")
        if _final == "promote":
            _render_status_message(_final_msg, "success")
        elif _final == "shadow_only":
            _render_status_message(_final_msg, "warning")
        else:
            _render_status_message(_final_msg, "error")

        st.caption("Resumo operacional do último backtest full: decisão de promoção (stop-rules) e calibração por faixa de odd.")
        if not _bt_report_path.exists():
            st.caption("Relatório de backtest não encontrado. Rode: python tools\\backtest_pipeline_abc_by_league.py --speed full")
        else:
            if not _bt_rep:
                st.error(f"Falha ao ler {_bt_report_path.name!s}")
            if isinstance(_bt_rep, dict) and _bt_rep:
                _global_bt = _bt_rep.get("global") or {}
                _bt_banner = core_build_stop_rules_banner(_bt_rep)
                _render_status_message(_bt_banner.get("message"), _bt_banner.get("message_type"))
                _g1, _g2, _g3 = st.columns(3)
                with _g1:
                    st.metric("Rows", value=str(int(_bt_rep.get("rows_total") or 0)))
                with _g2:
                    st.metric("Ganho logloss", value=_format_num(float(_global_bt.get("gain_logloss_vs_base", float("nan"))), 4))
                with _g3:
                    st.metric("Ganho brier", value=_format_num(float(_global_bt.get("gain_brier_vs_base", float("nan"))), 4))

                _reasons = _bt_banner.get("reasons") or []
                if isinstance(_reasons, list) and _reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _reasons[:8]]))

                _df_major = pd.DataFrame(core_build_stop_rules_major_rows(_bt_rep))
                if not _df_major.empty:
                    if not _df_major.empty:
                        st.markdown("##### Ligas principais (stop-rules)")
                        st.dataframe(
                            _df_contract_view(
                                _df_major,
                                ["Liga", "Rows", "Ganho logloss", "Fail drawdown"],
                                context="Parâmetros > Governanca ML > Ligas principais",
                                fill_value="-",
                            ),
                            width='stretch',
                            hide_index=True,
                        )

                _mon = _bt_rep.get("monitoring") or {}
                _df_bands = pd.DataFrame(core_build_stop_rules_global_band_rows(_bt_rep))
                if not _df_bands.empty:
                    st.markdown("##### Calibração por faixa (favorito do modelo, global)")
                    st.dataframe(
                        _df_contract_view(
                            _df_bands,
                            ["Faixa odd fav", "N", "Gap calib final (pp)", "Gap calib base (pp)", "Winrate real %", "Prob média %"],
                            context="Parâmetros > Governanca ML > Faixas de odd",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                _by_league_mon = _mon.get("by_league") or {}
                if isinstance(_by_league_mon, dict) and _by_league_mon:
                    _lg_opts = ["(global)", *sorted([str(k) for k in _by_league_mon.keys() if str(k).strip()])]
                    if str(st.session_state.get("gov_ml_mon_league_sel", "") or "") not in _lg_opts:
                        st.session_state["gov_ml_mon_league_sel"] = _lg_opts[0]
                    _lg_sel = st.selectbox("Liga (monitoramento por faixa)", _lg_opts, key="gov_ml_mon_league_sel")
                    if _lg_sel != "(global)":
                        _df_lf = pd.DataFrame(core_build_stop_rules_league_band_rows(_bt_rep, _lg_sel))
                        if not _df_lf.empty:
                            st.dataframe(
                                _df_contract_view(
                                    _df_lf,
                                    ["Faixa odd fav", "N", "Gap calib final (pp)", "Winrate real %", "Prob média %"],
                                    context="Parâmetros > Governanca ML > Faixas por liga",
                                    fill_value="-",
                                ),
                                width='stretch',
                                hide_index=True,
                            )

        st.markdown("##### Casos âncora (regressão semântica)")
        st.caption("Valida faixas de odds em confrontos de referência (ex.: LNG/TT, WE/TES, Gen.G/T1).")
        if not _anchor_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\anchor_regression_suite.py")
        else:
            if not _ar:
                st.error(f"Falha ao ler {_anchor_report_path.name!s}")
            if isinstance(_ar, dict) and _ar:
                _sum_ar = _ar.get("summary") or {}
                _ar_banner = core_decision_banner_parts(_ar.get("decision"), prefix="âncoras")
                _render_status_message(_ar_banner.get("message"), _ar_banner.get("message_type"))
                _ac1, _ac2, _ac3 = st.columns(3)
                with _ac1:
                    st.metric("âncoras (total)", value=str(int(_sum_ar.get("total") or 0)))
                with _ac2:
                    st.metric("OK", value=str(int(_sum_ar.get("ok") or 0)))
                with _ac3:
                    st.metric("FAIL", value=str(int(_sum_ar.get("fail") or 0)))
                _df_ar = pd.DataFrame(core_build_anchor_cases_rows(_ar))
                if not _df_ar.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_ar,
                            ["Caso", "OK", "Mercado", "Lado", "Odd", "Min", "Max"],
                            optional_cols=["Erro"],
                            context="Parâmetros > Governanca ML > Casos âncora",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

        st.markdown("##### Qualidade de dados / leakage / labels")
        st.caption("Checks automáticos de qualidade de rótulo e risco de leakage (heurístico e auditável).")
        if not _leak_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\check_data_leakage_and_labels.py")
        else:
            if not _lrp:
                st.error(f"Falha ao ler {_leak_report_path.name!s}")
            if isinstance(_lrp, dict) and _lrp:
                _lrp_banner = core_decision_banner_parts(_lrp.get("decision"), prefix="Leakage/labels")
                _render_status_message(_lrp_banner.get("message"), _lrp_banner.get("message_type"))
                _sum_lrp = _lrp.get("summary") or {}
                _lc1, _lc2, _lc3, _lc4 = st.columns(4)
                with _lc1:
                    st.metric("CSV files", value=str(int(_sum_lrp.get("csv_files") or 0)))
                with _lc2:
                    st.metric("Rows", value=str(int(_sum_lrp.get("rows_total") or 0)))
                with _lc3:
                    st.metric("Label issues", value=str(int(_sum_lrp.get("label_issues_total") or 0)))
                with _lc4:
                    st.metric("Critical", value=str(int(_sum_lrp.get("critical_issues_total") or 0)))
                _leak_meta = (_lrp.get("leakage_risk") or {})
                _rk = _leak_meta.get("high_risk_raw_columns_present") or []
                if isinstance(_rk, list) and _rk:
                    st.caption("Colunas de outcome presentes no CSV (informativo): " + ", ".join([str(x) for x in _rk[:20]]))
                _df_lrp = pd.DataFrame(core_build_leakage_file_rows(_lrp))
                if not _df_lrp.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_lrp,
                            ["Arquivo", "OK", "Rows"],
                            optional_cols=["Errors", "Warnings"],
                            context="Parâmetros > Governanca ML > Leakage/labels",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

        st.markdown("##### Lacunas (guardrail único)")
        st.caption("Decisão consolidada: stop-rules + âncoras + leakage + cobertura + frescor de artifacts.")
        if not _lacuna_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\run_lacuna_guardrails.py")
        else:
            if not _lgr:
                st.error(f"Falha ao ler {_lacuna_report_path.name!s}")
            if isinstance(_lgr, dict) and _lgr:
                _lgr_banner = core_decision_banner_parts(_lgr.get("decision"), prefix="Lacunas")
                _render_status_message(_lgr_banner.get("message"), _lgr_banner.get("message_type"))

                _parts_lg = _lgr.get("parts") or {}
                _df_parts = pd.DataFrame(core_build_parts_rows(_parts_lg))
                if not _df_parts.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_parts,
                            ["Check", "Decision"],
                            context="Parâmetros > Governanca ML > Lacunas (parts)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                _cov = _lgr.get("coverage") or {}
                _cov_sum = _cov.get("summary") or {}
                _k1, _k2, _k3, _k4 = st.columns(4)
                with _k1:
                    st.metric("Ligas", value=str(int(_cov_sum.get("league_count") or 0)))
                with _k2:
                    st.metric("Rows total", value=str(int(_cov_sum.get("rows_total") or 0)))
                with _k3:
                    st.metric("Low sample ligas", value=str(int(_cov_sum.get("low_sample_leagues_lt30") or 0)))
                with _k4:
                    st.metric("Blank team rate", value=_format_num(float(_cov_sum.get("blank_team_rate", 0.0) or 0.0) * 100.0, 3) + "%")

        st.markdown("##### Phase 1 - Data Contract")
        st.caption("Gate de consistência de dados pré-ML (leakage/labels + cobertura).")
        if not _phase1_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\run_phase1_data_contract.py --timeout 600")
        else:
            if not _p1:
                st.error(f"Falha ao ler {_phase1_report_path.name!s}")
            if isinstance(_p1, dict) and _p1:
                _p1_banner = core_decision_banner_parts(_p1.get("decision"), prefix="Fase 1")
                _render_status_message(_p1_banner.get("message"), _p1_banner.get("message_type"))
                _p1_parts = _p1.get("parts") or {}
                _df_p1_parts = pd.DataFrame(core_build_parts_rows(_p1_parts))
                if not _df_p1_parts.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_p1_parts,
                            ["Check", "Decision"],
                            context="Parâmetros > Governanca ML > Fase 1 (parts)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )
                _p1_reasons = _p1.get("reasons") or []
                if isinstance(_p1_reasons, list) and _p1_reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _p1_reasons[:8]]))

        st.markdown("##### Phase 1.2 - Roster Continuity A/B")
        st.caption("Valida impacto objetivo da camada roster_continuity_v2 (off vs on) em logloss/brier.")
        if not _phase12_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\eval_roster_continuity_phase12.py --speed fast")
        else:
            if not _p12:
                st.error(f"Falha ao ler {_phase12_report_path.name!s}")
            if isinstance(_p12, dict) and _p12:
                _p12_banner = core_decision_banner_parts(_p12.get("decision"), prefix="Fase 1.2")
                _render_status_message(_p12_banner.get("message"), _p12_banner.get("message_type"))
                _g_p12 = (_p12.get("global") or {}) if isinstance(_p12.get("global"), dict) else {}
                _k1, _k2, _k3, _k4 = st.columns(4)
                with _k1:
                    st.metric("Rows", value=str(int(_p12.get("rows_total") or 0)))
                with _k2:
                    st.metric("Applied rate", value=f"{_format_num(float(_p12.get('applied_rate', 0.0) or 0.0)*100.0,1)}%")
                with _k3:
                    st.metric("Gain logloss", value=_format_num(float(_g_p12.get("gain_logloss", float("nan"))), 6))
                with _k4:
                    st.metric("Gain brier", value=_format_num(float(_g_p12.get("gain_brier", float("nan"))), 6))
                _df_p12_lg = core_build_phase12_by_league_rows(_p12)
                if _df_p12_lg:
                    _dfp12 = pd.DataFrame(_df_p12_lg).sort_values(["Rows"], ascending=False).head(20)
                    st.dataframe(
                        _df_contract_view(
                            _dfp12,
                            ["Liga", "Rows", "Gain logloss", "LL off", "LL on"],
                            context="Parâmetros > Governanca ML > Fase 1.2 (ligas)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )
                _p12_reasons = _p12.get("reasons") or []
                if isinstance(_p12_reasons, list) and _p12_reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _p12_reasons[:8]]))

        st.markdown("##### Promoção / Rollback")
        st.caption("Operação de produção com trilha de auditoria: promover baseline (com testes) ou rollback do último baseline.")
        _pr1, _pr2 = st.columns(2)
        with _pr1:
            _run_promote = st.button("Promover baseline agora", width='stretch', key="gov_promote_baseline")
        with _pr2:
            _run_rollback = st.button("Rollback baseline (mais recente)", width='stretch', key="gov_rollback_baseline")

        if _run_promote or _run_rollback:
            with st.spinner("Executando operação..."):
                _op_result = core_run_governance_operation(
                    promote=bool(_run_promote),
                    cwd=APP_ROOT,
                    tag="app_governance",
                    timeout_s=1200,
                )
                _merged_gov = core_append_governance_log(
                    core_get_governance_last_log(st.session_state),
                    _op_result.get("log_line"),
                    now_iso=datetime.now().isoformat(timespec="seconds"),
                )
                core_set_governance_last_log(st.session_state, str(_merged_gov.get("last_log") or ""))
                st.session_state["_gov_last_run"] = str(_merged_gov.get("last_run") or "")
                if bool(_op_result.get("ok")):
                    st.success("Operação concluída com sucesso.")
                else:
                    _err_txt = str(_op_result.get("error") or "").strip()
                    if _err_txt:
                        st.error(f"Falha ao executar operação: {_err_txt}")
                    else:
                        st.error("Operação falhou. Veja o log abaixo.")
