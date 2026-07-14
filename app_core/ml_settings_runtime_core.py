from __future__ import annotations

from typing import Any, Dict, Mapping


def shared_ml_defaults() -> Dict[str, Any]:
    """Return ML parameter defaults shared across all runtime environments (live + offline)."""
    return {
        "resumo_guardrail_profile": "Equilibrado",
        "resumo_guardrail_strength_major": 1.00,
        "resumo_guardrail_strength_mid": 1.00,
        "resumo_guardrail_strength_minor": 1.00,
        "resumo_guardrail_target_major": 1.00,
        "resumo_guardrail_target_mid": 1.00,
        "resumo_guardrail_target_minor": 1.00,
        "fusion_roster_cont_thresh_base": 0.55,
        "fusion_roster_cont_thresh_early_boost": 0.10,
        "fusion_roster_low_freshness_cut": 0.60,
        "fusion_roster_low_freshness_boost": 0.05,
        "fusion_profile_preset": "Balanceado (Recomendado)",
        "ml_strategy_profile": "Equilibrado (Recomendado)",
        "ml_microseg_hierarchical": True,
        "ml_microseg_hier_decay": 0.80,
        "ml_fused_post_cal_use_fitted": True,
        "ml_players_context_use_fitted": True,
        "ml_players_context_enabled": False,
        "ml_players_context_lookback_games": 24,
        "ml_players_context_half_life_days": 45.0,
        "ml_players_context_min_games": 6,
        "ml_players_context_strength": 0.35,
        "ml_players_context_cap": 0.22,
        "ml_players_context_min_reliability": 0.28,
        "ml_players_context_inter_mult": 0.88,
        # Uncertainty: disabled by default after ablation study (tools/run_ablation_confidence_sweep.py)
        # confirmed 0/7 configurations beat the baseline on 1313-sample holdout (worst: -0.38pp,
        # best: -0.085pp). Keep params for future re-evaluation if base model changes.
        "ml_uncertainty_enabled": False,
        "ml_uncertainty_strength": 0.55,
        "ml_uncertainty_prior_games": 22.0,
        "ml_uncertainty_max_shrink": 0.65,
        "ml_uncertainty_exclude_leagues": "LCK",
        "ml_interregional_uncertainty_strength_mult": 0.30,
        "ml_interregional_uncertainty_prior_mult": 0.60,
        "ml_interregional_uncertainty_cap_mult": 0.40,
        "ml_interregional_conf_guard_players_eff_weight": 0.12,
        "ml_interregional_conf_guard_players_eff_cap": 18.0,
        "ml_interregional_conf_guard_max_shrink_mult": 0.35,
        # Marginal Trust: disabled by default after ablation study (tools/run_ablation_confidence_sweep.py)
        # confirmed 0/21 configurations (7 confidence_scores x 3 strengths) beat the baseline on
        # 1313-sample holdout (worst: -3.34pp, best: -0.247pp). Keep params for future re-evaluation.
        "ml_marginal_trust_enabled": False,
        # Post-fusion calibration: applies Microseg + Platt to p_fused AFTER team+players fusion.
        # Validated through Phase 1 gate (tools/run_phase1_validation.py) on 2827-sample
        # holdout (2026 Apr-May, chronological split):
        #   delta_logloss = +0.218pp, CI 95% [+0.034, +0.381], p=0.008
        #   delta_ECE = +0.677pp (calibration also improves)
        # DECISION: ACCEPT — flag enabled by default.
        "ml_postfusion_cal_enabled": True,
        # Live state-to-win models (Phase 6): LightGBM trained on snapshots @10/15/20/25.
        # Validated through gate (tools/run_phase6_live_gate.py) on 5000+ samples per snapshot:
        #   @10min: delta_logloss = +9.87pp,  p=0.0000
        #   @15min: delta_logloss = +14.85pp, p=0.0000
        #   @20min: delta_logloss = +20.94pp, p=0.0000
        #   @25min: delta_logloss = +29.28pp, p=0.0000
        # Models saved under live_models/minute_{10,15,20,25}/.
        # Integrated into build_live_bet_live_assessment() — when ON, the assessment
        # includes the calibrated `live_p_win` in its output. When OFF, that field
        # is None and the app continues using existing heuristics.
        # DECISION: ON by default (huge gain on test set, all 4 gates ACCEPT).
        "ml_live_state_model_enabled": True,
        "ml_live_state_models_dir": "live_models",
        "ml_marginal_trust_strength": 0.70,
        "ml_marginal_trust_prior_games": 28.0,
        "ml_marginal_trust_max_shrink": 0.72,
        "ml_marginal_trust_cap_prob": 0.84,
        "ml_marginal_trust_score_soft": 68.0,
        "ml_marginal_trust_score_hard": 46.0,
        "ml_marginal_trust_exclude_leagues": "LCK,LPL,LEC,LCS,CBLOL,LCP,PCS,VCS",
        "ml_marginal_trust_band_tuning": True,
        "ml_marginal_trust_band_coeffs": {},
    }


def build_runtime_param_defaults(base_defaults: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge base_defaults with shared ML defaults, forcing ml_players_context_enabled=False."""
    out: Dict[str, Any] = dict(base_defaults or {})
    for key, value in shared_ml_defaults().items():
        if key == "ml_players_context_enabled":
            out[key] = value
        else:
            out.setdefault(key, value)
    out.setdefault("ml_confidence_governor_enabled", True)
    out.setdefault("ml_confidence_governor_live_enabled", True)
    out.setdefault("ml_confidence_governor_max_total_shrink_pp", 8.0)
    out.setdefault("ml_confidence_governor_use_fitted_by_league", True)
    out.setdefault("ml_series_coherence_post_use_fitted_by_league", True)
    out.setdefault("ml_series_coherence_post_max_dev_pp_bo3", 16.0)
    out.setdefault("ml_series_coherence_post_max_dev_pp_bo5", 20.0)
    out.setdefault("ff_ml_state_alite_live", False)
    out.setdefault("ff_ml_state_areal_live", True)
    # style_matchup_b + meta_blend_c desligados: backtest A/B/C (replay frozen, 3500 jogos)
    # mediu gain_logloss GLOBAL = -0.000417 e gain_brier = -0.000053 (ambos NEGATIVOS =
    # pioram a nota), com 20/34 ligas piorando (LCK/LPL/LEC/LCP entre elas) e hard-fail de
    # calibracao na faixa 1.30-1.50 do favorito (gap 5.8pp). Alem de nao agregar nota, eram
    # a UNICA fonte da assimetria de troca de ordem (Time1<->Time2 mudava a odd ~7.7pp porque
    # injetavam vies do slot azul num lado arbitrario no pre-jogo). decision=reject.
    out.setdefault("ff_ml_style_b_lite_live", False)
    out.setdefault("ff_ml_league_rank_sos_live", True)
    out.setdefault("ff_ml_meta_c_lite_live", False)
    out.setdefault("ff_ml_roster_continuity_live", True)
    out.setdefault("ff_ml_hier_calibration_live", True)
    out.setdefault("ff_ml_series_bo_dplite_live", True)
    out.setdefault("ff_ml_layer_uncertainty_live", True)
    out.setdefault("ff_players_panels_shadow_trace", True)
    out.setdefault("ff_ml_shadow_state_alite_trace", True)
    out.setdefault("ff_ml_shadow_state_areal_trace", True)
    return out


def local_runtime_defaults() -> Dict[str, Any]:
    """Return defaults specific to the local (Streamlit) runtime, extending shared_ml_defaults."""
    return {
        "ml_microseg_enabled": True,
        "ml_microseg_strength": 0.60,
        "ml_microseg_min_n": 30,
        "ml_microseg_shrink_k": 90.0,
        "ml_microseg_lookback_games": 1200,
        "ml_microseg_cap_pp": 8.0,
        **shared_ml_defaults(),
        "resumo_filter_preset": "Custom",
        "resumo_filter_custom": {},
        "resumo_filter_profile_active": "",
        "resumo_filter_profiles": {
            "Conservador": {
                "resumo_max_odd": 1.72,
                "resumo_req_liga": True,
                "resumo_req_h2h": True,
                "resumo_req_model": True,
                "resumo_min_sample_liga": 10,
                "resumo_min_sample_h2h": 4,
            },
            "Balanceado": {
                "resumo_max_odd": 2.00,
                "resumo_req_liga": True,
                "resumo_req_h2h": False,
                "resumo_req_model": True,
                "resumo_min_sample_liga": 8,
                "resumo_min_sample_h2h": 3,
            },
            "Agressivo": {
                "resumo_max_odd": 2.35,
                "resumo_req_liga": False,
                "resumo_req_h2h": False,
                "resumo_req_model": True,
                "resumo_min_sample_liga": 5,
                "resumo_min_sample_h2h": 1,
            },
        },
        "ml_players_lane_w_top": 1.00,
        "ml_players_lane_w_jungle": 1.00,
        "ml_players_lane_w_mid": 1.00,
        "ml_players_lane_w_adc": 1.00,
        "ml_players_lane_w_support": 1.00,
        "ml_players_lg_w_lpl": 1.00,
        "ml_players_lg_w_lck": 1.00,
        "ml_players_lg_w_emea": 1.00,
        "ml_players_lg_w_na": 1.00,
        "ml_players_lg_w_br": 1.00,
        "ml_players_lg_w_apac": 1.00,
        "ml_players_lg_w_other": 1.00,
    }
