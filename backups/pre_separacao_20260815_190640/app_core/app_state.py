"""Centralized session-state schema/helpers for Streamlit app."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, MutableMapping, Tuple


def _to_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on", "sim"}:
            return True
        if s in {"0", "false", "no", "off", "nao", "não"}:
            return False
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _to_float(v: Any, default: float) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(x)


def _to_int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _to_str(v: Any, default: str) -> str:
    if v is None:
        return str(default)
    return str(v)


def bootstrap_state(
    session_state: MutableMapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    migrations: Mapping[str, str] | None = None,
    constraints: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    mig = dict(migrations or {})
    for old_k, new_k in mig.items():
        if old_k in session_state and new_k not in session_state:
            session_state[new_k] = session_state.get(old_k)

    for k, v in dict(defaults).items():
        if k not in session_state:
            session_state[k] = v

    if constraints:
        for k, (lo, hi) in constraints.items():
            if k not in session_state:
                continue
            try:
                x = float(session_state.get(k))
                if not math.isfinite(x):
                    x = float(defaults.get(k, lo))
                x = max(float(lo), min(float(hi), float(x)))
                if isinstance(defaults.get(k), int):
                    session_state[k] = round(x)
                else:
                    session_state[k] = float(x)
            except Exception:
                session_state[k] = defaults.get(k)


def get_bool(session_state: Mapping[str, Any], key: str, default: bool = False) -> bool:
    return _to_bool(session_state.get(key, default), default)


def get_float(session_state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    return _to_float(session_state.get(key, default), default)


def get_int(session_state: Mapping[str, Any], key: str, default: int = 0) -> int:
    return _to_int(session_state.get(key, default), default)


def get_str(session_state: Mapping[str, Any], key: str, default: str = "") -> str:
    return _to_str(session_state.get(key, default), default)


def apply_recommended_preset(session_state: MutableMapping[str, Any], profile: str = "Padrão") -> None:
    _prof = str(profile or "Padrão")
    if _prof == "20/10":
        session_state["cfg_profile"] = "20/10"
    else:
        session_state["cfg_profile"] = "Avançado" if _prof == "Avançado" else "Padrão"
    session_state["ml_engine"] = "mlcore_v2"
    session_state["ml_engine_compare"] = False
    session_state["ml_engine_alert_pp"] = 12
    session_state["league_mode"] = "auto"
    session_state["weight_mode"] = "Recência (half-life)"
    session_state["half_life_days"] = _to_int(session_state.get("half_life_days", 180), 180)
    session_state["n_sims"] = _to_int(session_state.get("n_sims", 20000), 20000)
    session_state["resumo_max_odd"] = _to_float(session_state.get("resumo_max_odd", 1.72), 1.72)
    session_state["resumo_use_edge_gate"] = _to_bool(session_state.get("resumo_use_edge_gate", False), False)
    session_state["resumo_min_edge_pp"] = _to_float(session_state.get("resumo_min_edge_pp", 2.5), 2.5)
    session_state["resumo_min_sample_liga"] = _to_int(session_state.get("resumo_min_sample_liga", 8), 8)
    session_state["resumo_min_sample_h2h"] = _to_int(session_state.get("resumo_min_sample_h2h", 3), 3)
    session_state["resumo_hist_shrink_n"] = _to_int(session_state.get("resumo_hist_shrink_n", 20), 20)
    session_state["ml_consistency_guard"] = _to_bool(session_state.get("ml_consistency_guard", True), True)
    session_state["ml_consistency_tol_pp"] = _to_float(session_state.get("ml_consistency_tol_pp", 0.05), 0.05)
    session_state["wf_apply_ml_correction"] = _to_bool(session_state.get("wf_apply_ml_correction", False), False)
    session_state["wf_ml_corr_strength"] = _to_float(session_state.get("wf_ml_corr_strength", 0.60), 0.60)
    session_state["totals_mode"] = _to_str(session_state.get("totals_mode", "map"), "map")
    session_state["ml_mode"] = _to_str(session_state.get("ml_mode", "map"), "map")
    session_state["map_mode_ui"] = _to_str(session_state.get("map_mode_ui", "avg"), "avg")
    session_state["ui_fast_flow"] = _to_bool(session_state.get("ui_fast_flow", True), True)
    session_state["lineup_lambda_points"] = _to_int(session_state.get("lineup_lambda_points", 90), 90)
    session_state["ml_players_w_lineup"] = _to_float(session_state.get("ml_players_w_lineup", 1.15), 1.15)
    session_state["lineup_delta_mode"] = _to_str(session_state.get("lineup_delta_mode", "saturado"), "saturado")
    session_state["lineup_delta_cap"] = _to_float(session_state.get("lineup_delta_cap", 1.60), 1.60)
    session_state["lineup_delta_slope"] = _to_float(session_state.get("lineup_delta_slope", 2.40), 2.40)
    session_state["resumo_combo_use_mismatch"] = _to_bool(session_state.get("resumo_combo_use_mismatch", False), False)
    session_state["fusion_team_weight_early"] = _to_float(session_state.get("fusion_team_weight_early", 0.58), 0.58)
    session_state["fusion_team_weight_mid"] = _to_float(session_state.get("fusion_team_weight_mid", 0.75), 0.75)
    session_state["fusion_team_weight_playoffs"] = _to_float(session_state.get("fusion_team_weight_playoffs", 0.82), 0.82)
    session_state["fusion_early_season_team_weight"] = _to_float(session_state.get("fusion_early_season_team_weight", 0.55), 0.55)
    session_state["fusion_season_knee_games"] = _to_float(session_state.get("fusion_season_knee_games", 18.0), 18.0)
    session_state["fusion_coverage_power"] = _to_float(session_state.get("fusion_coverage_power", 1.5), 1.5)
    session_state["fusion_transfer_boost"] = _to_float(session_state.get("fusion_transfer_boost", 0.35), 0.35)
    session_state["fusion_divergence_pp_cap"] = _to_float(session_state.get("fusion_divergence_pp_cap", 22.0), 22.0)
    session_state["fusion_divergence_low_coverage"] = _to_float(session_state.get("fusion_divergence_low_coverage", 0.30), 0.30)
    session_state["fusion_divergence_shrink"] = _to_float(session_state.get("fusion_divergence_shrink", 0.35), 0.35)
    session_state["ml_form_blend_enabled"] = _to_bool(session_state.get("ml_form_blend_enabled", True), True)
    session_state["ml_form_use_fitted"] = _to_bool(session_state.get("ml_form_use_fitted", True), True)
    session_state["ml_form_blend_max_weight"] = _to_float(session_state.get("ml_form_blend_max_weight", 0.75), 0.75)
    session_state["ml_form_blend_knee_games"] = _to_float(session_state.get("ml_form_blend_knee_games", 6.0), 6.0)
    session_state["ml_form_blend_beta"] = _to_float(session_state.get("ml_form_blend_beta", 2.0), 2.0)
    session_state["ml_form_blend_signal_power"] = _to_float(session_state.get("ml_form_blend_signal_power", 1.0), 1.0)
    session_state["ml_form_blend_parity_strength"] = _to_float(
        session_state.get("ml_form_blend_parity_strength", 0.35), 0.35
    )
    session_state["ml_form_override_min_games"] = _to_int(session_state.get("ml_form_override_min_games", 8), 8)
    session_state["ml_form_override_div_pp"] = _to_float(session_state.get("ml_form_override_div_pp", 8.0), 8.0)
    session_state["ml_form_override_max_w"] = _to_float(session_state.get("ml_form_override_max_w", 0.88), 0.88)
    session_state["ml_interregional_form_max_weight_mult"] = _to_float(
        session_state.get("ml_interregional_form_max_weight_mult", 0.18), 0.18
    )
    session_state["ml_interregional_form_knee_mult"] = _to_float(
        session_state.get("ml_interregional_form_knee_mult", 2.0), 2.0
    )
    session_state["ml_interregional_disable_form_override"] = _to_bool(
        session_state.get("ml_interregional_disable_form_override", True), True
    )
    session_state["ml_hybrid_blend_enabled"] = _to_bool(session_state.get("ml_hybrid_blend_enabled", True), True)
    session_state["ml_hybrid_blend_max_weight"] = _to_float(session_state.get("ml_hybrid_blend_max_weight", 0.70), 0.70)
    session_state["ml_hybrid_blend_knee_games"] = _to_float(session_state.get("ml_hybrid_blend_knee_games", 12.0), 12.0)
    session_state["ml_hybrid_momentum_games"] = _to_int(session_state.get("ml_hybrid_momentum_games", 10), 10)
    session_state["ml_hybrid_beta"] = _to_float(session_state.get("ml_hybrid_beta", 2.0), 2.0)
    session_state["ml_interregional_hybrid_max_weight_mult"] = _to_float(
        session_state.get("ml_interregional_hybrid_max_weight_mult", 0.18), 0.18
    )
    session_state["ml_interregional_hybrid_knee_mult"] = _to_float(
        session_state.get("ml_interregional_hybrid_knee_mult", 1.8), 1.8
    )
    session_state["ml_conf_guard_enabled"] = _to_bool(session_state.get("ml_conf_guard_enabled", True), True)
    session_state["ml_conf_guard_min_games"] = _to_float(session_state.get("ml_conf_guard_min_games", 12.0), 12.0)
    session_state["ml_conf_guard_max_shrink"] = _to_float(session_state.get("ml_conf_guard_max_shrink", 0.35), 0.35)
    session_state["ml_interregional_uncertainty_strength_mult"] = _to_float(
        session_state.get("ml_interregional_uncertainty_strength_mult", 0.30), 0.30
    )
    session_state["ml_interregional_uncertainty_prior_mult"] = _to_float(
        session_state.get("ml_interregional_uncertainty_prior_mult", 0.60), 0.60
    )
    session_state["ml_interregional_uncertainty_cap_mult"] = _to_float(
        session_state.get("ml_interregional_uncertainty_cap_mult", 0.40), 0.40
    )
    session_state["ml_interregional_conf_guard_players_eff_weight"] = _to_float(
        session_state.get("ml_interregional_conf_guard_players_eff_weight", 0.12), 0.12
    )
    session_state["ml_interregional_conf_guard_players_eff_cap"] = _to_float(
        session_state.get("ml_interregional_conf_guard_players_eff_cap", 18.0), 18.0
    )
    session_state["ml_interregional_conf_guard_max_shrink_mult"] = _to_float(
        session_state.get("ml_interregional_conf_guard_max_shrink_mult", 0.35), 0.35
    )
    session_state["ml_patch_guard_enabled"] = _to_bool(session_state.get("ml_patch_guard_enabled", True), True)
    session_state["ml_patch_guard_warn_shrink"] = _to_float(session_state.get("ml_patch_guard_warn_shrink", 0.08), 0.08)
    session_state["ml_patch_guard_refit_shrink"] = _to_float(session_state.get("ml_patch_guard_refit_shrink", 0.16), 0.16)
    session_state["ml_patch_guard_focus_bonus"] = _to_float(session_state.get("ml_patch_guard_focus_bonus", 0.06), 0.06)
    session_state["ml_confidence_governor_enabled"] = _to_bool(
        session_state.get("ml_confidence_governor_enabled", True),
        True,
    )
    session_state["ml_confidence_governor_live_enabled"] = _to_bool(
        session_state.get("ml_confidence_governor_live_enabled", True),
        True,
    )
    session_state["ml_confidence_governor_max_total_shrink_pp"] = _to_float(
        session_state.get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        8.0,
    )
    session_state["ml_confidence_governor_use_fitted_by_league"] = _to_bool(
        session_state.get("ml_confidence_governor_use_fitted_by_league", True),
        True,
    )
    session_state["ml_series_coherence_post_use_fitted_by_league"] = _to_bool(
        session_state.get("ml_series_coherence_post_use_fitted_by_league", True),
        True,
    )
    session_state["ml_series_coherence_post_max_dev_pp_bo3"] = _to_float(
        session_state.get("ml_series_coherence_post_max_dev_pp_bo3", 16.0),
        16.0,
    )
    session_state["ml_series_coherence_post_max_dev_pp_bo5"] = _to_float(
        session_state.get("ml_series_coherence_post_max_dev_pp_bo5", 20.0),
        20.0,
    )
    session_state["ml_series_gamma_use_fitted"] = _to_bool(session_state.get("ml_series_gamma_use_fitted", True), True)


def apply_excellent_value_preset(session_state: MutableMapping[str, Any]) -> None:
    vals = {
        "ml_engine": "mlcore_v2",
        "ml_engine_compare": True,
        "ml_engine_alert_pp": 10,
        "ml_show_calc_details_default": False,
        "weight_mode": "Recência (half-life)",
        "half_life_days": 90,
        "n_sims": 30000,
        "totals_mode": "map",
        "ml_mode": "map",
        "map_mode_ui": "avg",
        "league_mode": "auto",
        "resumo_max_odd": 2.20,
        "resumo_req_liga": True,
        "resumo_req_h2h": False,
        "resumo_req_model": True,
        "resumo_show_laplace_cols": True,
        "resumo_sc_use": True,
        "resumo_sc_thr": 58,
        "resumo_laplace_pick": "Média Liga+H2H 5",
        "resumo_mix_w_model": 55,
        "resumo_mix_w_laplace": 30,
        "resumo_mix_w_sc": 15,
        "resumo_combo_preset": "Neutro",
        "resumo_combo_hist_w": 0.55,
        "resumo_combo_style_w": 0.20,
        "resumo_combo_form_w": 0.22,
        "resumo_combo_use_mismatch": False,
        "resumo_use_edge_gate": True,
        "resumo_min_edge_pp": 2.5,
        "wf_min_train_games": 1500,
        "wf_k": 26.0,
        "wf_scale": 400.0,
        "wf_apply_ml_correction": True,
        "wf_ml_corr_strength": 0.60,
        "resumo_min_sample_liga": 10,
        "resumo_min_sample_h2h": 3,
        "resumo_hist_shrink_n": 24,
        "sim_seed_mode": "Deterministico (confronto+config)",
        "sim_seed_manual": 1337,
        "ml_consistency_guard": True,
        "ml_consistency_tol_pp": 0.05,
        "use_lineup_adjust": False,
        "lineup_delta_mode": "saturado",
        "lineup_delta_cap": 1.60,
        "lineup_delta_slope": 2.40,
        "fusion_team_weight_early": 0.56,
        "fusion_team_weight_mid": 0.72,
        "fusion_team_weight_playoffs": 0.82,
        "fusion_early_season_team_weight": 0.52,
        "fusion_season_knee_games": 20.0,
        "fusion_coverage_power": 1.35,
        "fusion_transfer_boost": 0.45,
        "fusion_divergence_pp_cap": 24.0,
        "fusion_divergence_low_coverage": 0.35,
        "fusion_divergence_shrink": 0.40,
        "ml_form_blend_enabled": True,
        "ml_form_use_fitted": True,
        "ml_form_blend_max_weight": 0.72,
        "ml_form_blend_knee_games": 6.0,
        "ml_form_blend_beta": 2.0,
        "ml_form_blend_signal_power": 1.0,
        "ml_form_blend_parity_strength": 0.35,
        "ml_form_override_min_games": 7,
        "ml_form_override_div_pp": 6.0,
        "ml_form_override_max_w": 0.92,
        "ml_interregional_form_max_weight_mult": 0.18,
        "ml_interregional_form_knee_mult": 2.0,
        "ml_interregional_disable_form_override": True,
        "ml_hybrid_blend_enabled": True,
        "ml_hybrid_blend_max_weight": 0.66,
        "ml_hybrid_blend_knee_games": 14.0,
        "ml_hybrid_momentum_games": 10,
        "ml_hybrid_beta": 2.0,
        "ml_interregional_hybrid_max_weight_mult": 0.18,
        "ml_interregional_hybrid_knee_mult": 1.8,
        "ml_conf_guard_enabled": True,
        "ml_conf_guard_min_games": 12.0,
        "ml_conf_guard_max_shrink": 0.28,
        "ml_interregional_uncertainty_strength_mult": 0.30,
        "ml_interregional_uncertainty_prior_mult": 0.60,
        "ml_interregional_uncertainty_cap_mult": 0.40,
        "ml_interregional_conf_guard_players_eff_weight": 0.12,
        "ml_interregional_conf_guard_players_eff_cap": 18.0,
        "ml_interregional_conf_guard_max_shrink_mult": 0.35,
        "ml_patch_guard_enabled": True,
        "ml_patch_guard_warn_shrink": 0.08,
        "ml_patch_guard_refit_shrink": 0.16,
        "ml_patch_guard_focus_bonus": 0.06,
        "ml_series_gamma_use_fitted": True,
    }
    for k, v in vals.items():
        session_state[k] = v


def apply_twenty_ten_preset(session_state: MutableMapping[str, Any]) -> None:
    """Aggressive preset (20/10): prioritizes exploitable edges with safeguards."""
    apply_excellent_value_preset(session_state)
    session_state["cfg_profile"] = "20/10"
    vals = {
        "resumo_max_odd": 3.00,
        "resumo_req_h2h": False,
        "resumo_sc_thr": 56,
        "resumo_mix_w_model": 62,
        "resumo_mix_w_laplace": 26,
        "resumo_mix_w_sc": 12,
        "resumo_combo_hist_w": 0.52,
        "resumo_combo_style_w": 0.30,
        "resumo_combo_form_w": 0.30,
        "resumo_combo_use_mismatch": False,
        "half_life_days": 75,
        "n_sims": 35000,
        "wf_apply_ml_correction": True,
        "wf_ml_corr_strength": 0.70,
        "resumo_min_sample_liga": 8,
        "resumo_min_sample_h2h": 2,
        "fusion_team_weight_early": 0.50,
        "fusion_team_weight_mid": 0.68,
        "fusion_team_weight_playoffs": 0.80,
        "fusion_early_season_team_weight": 0.48,
        "fusion_season_knee_games": 22.0,
        "fusion_coverage_power": 1.20,
        "fusion_transfer_boost": 0.60,
        "fusion_divergence_pp_cap": 26.0,
        "fusion_divergence_low_coverage": 0.40,
        "fusion_divergence_shrink": 0.45,
        "ml_form_blend_enabled": True,
        "ml_form_use_fitted": True,
        "ml_form_blend_max_weight": 0.85,
        "ml_form_blend_knee_games": 5.0,
        "ml_form_blend_beta": 2.0,
        "ml_form_blend_signal_power": 0.90,
        "ml_form_blend_parity_strength": 0.35,
        "ml_form_override_min_games": 7,
        "ml_form_override_div_pp": 7.0,
        "ml_form_override_max_w": 0.92,
        "ml_interregional_form_max_weight_mult": 0.18,
        "ml_interregional_form_knee_mult": 2.0,
        "ml_interregional_disable_form_override": True,
        "ml_hybrid_blend_enabled": True,
        "ml_hybrid_blend_max_weight": 0.80,
        "ml_hybrid_blend_knee_games": 10.0,
        "ml_hybrid_momentum_games": 10,
        "ml_hybrid_beta": 2.0,
        "ml_interregional_hybrid_max_weight_mult": 0.18,
        "ml_interregional_hybrid_knee_mult": 1.8,
        "ml_conf_guard_enabled": True,
        "ml_conf_guard_min_games": 12.0,
        "ml_conf_guard_max_shrink": 0.30,
        "ml_interregional_uncertainty_strength_mult": 0.30,
        "ml_interregional_uncertainty_prior_mult": 0.60,
        "ml_interregional_uncertainty_cap_mult": 0.40,
        "ml_interregional_conf_guard_players_eff_weight": 0.12,
        "ml_interregional_conf_guard_players_eff_cap": 18.0,
        "ml_interregional_conf_guard_max_shrink_mult": 0.35,
        "ml_patch_guard_enabled": True,
        "ml_patch_guard_warn_shrink": 0.10,
        "ml_patch_guard_refit_shrink": 0.20,
        "ml_patch_guard_focus_bonus": 0.08,
        "ml_series_gamma_use_fitted": True,
    }
    for k, v in vals.items():
        session_state[k] = v


APP_STATE_DEFAULTS: Dict[str, Any] = {
    "resumo_max_odd": 1.72,
    "resumo_req_liga": True,
    "resumo_req_h2h": True,
    "resumo_req_model": True,
    "resumo_show_laplace_cols": False,
    "resumo_sc_use": True,
    "resumo_sc_thr": 60,
    "resumo_laplace_pick": "Média Liga+H2H 5",
    "resumo_mix_w_model": 40,
    "resumo_mix_w_laplace": 40,
    "resumo_mix_w_sc": 20,
    "resumo_combo_preset": "Neutro",
    "resumo_combo_hist_w": 0.45,
    "resumo_combo_style_w": 0.18,
    "resumo_combo_form_w": 0.16,
    "resumo_combo_use_mismatch": False,
    "ml_engine": "mlcore_v2",
    "ml_engine_compare": False,
    "ml_engine_alert_pp": 12,
    "ml_show_calc_details_default": False,
    "weight_mode": "Recência (half-life)",
    "half_life_days": 180,
    "n_sims": 20000,
    "totals_mode": "map",
    "ml_mode": "map",
    "map_mode_ui": "avg",
    "league_mode": "auto",
    "use_lineup_adjust": False,
    "lineup_lambda_points": 90,
    "ml_players_w_lineup": 1.15,
    "lineup_delta_mode": "saturado",
    "lineup_delta_cap": 1.60,
    "lineup_delta_slope": 2.40,
    "lineup_min_gp": 5,
    "lineup_shrink_m": 10,
    "fusion_team_weight_early": 0.58,
    "fusion_team_weight_mid": 0.75,
    "fusion_team_weight_playoffs": 0.82,
    "fusion_early_season_team_weight": 0.55,
    "fusion_season_knee_games": 18.0,
    "fusion_coverage_power": 1.5,
    "fusion_transfer_boost": 0.35,
    "fusion_divergence_pp_cap": 22.0,
    "fusion_divergence_low_coverage": 0.30,
    "fusion_divergence_shrink": 0.35,
    "ml_form_blend_enabled": True,
    "ml_form_use_fitted": True,
    "ml_form_blend_max_weight": 0.80,
    "ml_form_blend_knee_games": 6.0,
    "ml_form_blend_beta": 2.0,
    "ml_form_blend_signal_power": 1.0,
    "ml_form_blend_parity_strength": 0.35,
    "ml_form_override_min_games": 7,
    "ml_form_override_div_pp": 6.0,
    "ml_form_override_max_w": 0.92,
    "ml_interregional_form_max_weight_mult": 0.18,
    "ml_interregional_form_knee_mult": 2.0,
    "ml_interregional_disable_form_override": True,
    "ml_hybrid_blend_enabled": True,
    "ml_hybrid_blend_max_weight": 0.66,
    "ml_hybrid_blend_knee_games": 14.0,
    "ml_hybrid_momentum_games": 10,
    "ml_hybrid_beta": 2.0,
    "ml_interregional_hybrid_max_weight_mult": 0.18,
    "ml_interregional_hybrid_knee_mult": 1.8,
    "ml_conf_guard_enabled": True,
    "ml_conf_guard_min_games": 12.0,
    "ml_conf_guard_max_shrink": 0.28,
    "ml_interregional_uncertainty_strength_mult": 0.30,
    "ml_interregional_uncertainty_prior_mult": 0.60,
    "ml_interregional_uncertainty_cap_mult": 0.40,
    "ml_interregional_conf_guard_players_eff_weight": 0.12,
    "ml_interregional_conf_guard_players_eff_cap": 18.0,
    "ml_interregional_conf_guard_max_shrink_mult": 0.35,
    "ml_patch_guard_enabled": True,
    "ml_patch_guard_warn_shrink": 0.08,
    "ml_patch_guard_refit_shrink": 0.16,
    "ml_patch_guard_focus_bonus": 0.06,
    "ml_confidence_governor_enabled": True,
    "ml_confidence_governor_live_enabled": True,
    "ml_confidence_governor_max_total_shrink_pp": 8.0,
    "ml_confidence_governor_use_fitted_by_league": True,
    "ml_series_coherence_post_use_fitted_by_league": True,
    "ml_series_coherence_post_max_dev_pp_bo3": 16.0,
    "ml_series_coherence_post_max_dev_pp_bo5": 20.0,
    "ml_series_gamma_use_fitted": True,
    "resumo_use_edge_gate": False,
    "resumo_min_edge_pp": 2.5,
    "wf_min_train_games": 1500,
    "wf_k": 26.0,
    "wf_scale": 400.0,
    "wf_apply_ml_correction": False,
    "wf_ml_corr_strength": 0.60,
    "resumo_min_sample_liga": 8,
    "resumo_min_sample_h2h": 3,
    "resumo_hist_shrink_n": 20,
    "sim_seed_mode": "Deterministico (confronto+config)",
    "sim_seed_manual": 1337,
    "ml_consistency_guard": True,
    "ml_consistency_tol_pp": 0.05,
    "cal_diag_bins": 10,
    "cal_diag_min_league_games": 250,
    # Series
    "series_kills_txt": "",
    "series_towers_txt": "",
    "series_dragons_txt": "",
    "series_barons_txt": "",
    "series_team1_k_txt": "",
    "series_team2_k_txt": "",
    # StatsCamps
    "sc_year": "",
    "sc_league": "",
    "sc_split": "",
    "sc_playoffs": "",
    "sc_map": "All maps",
    "sc_th_kills": "23.5,25.5,27.5,29.5",
    "sc_th_towers": "10.5,11.5,12.5",
    "sc_th_dragons": "4.5,5.5",
    "sc_th_barons": "1.5",
    "sc_th_inhib": "1.5",
    "sc_th_time": "",
    "vg_stats_market": "Kills",
    "vg_stats_current_line": "",
    "vg_stats_custom_line": "",
    "vg_stats_side_filter": "All",
    "vg_stats_wl_filter": "All",
    # Campeões
    "ch_year": "",
    "ch_league": "",
    "ch_split": "",
    "ch_playoffs": "",
    "ch_team": "",
    "ch_min_games_mu": 3,
    "ch_blue_team": "",
    "ch_paste": "",
    # Players
    "pl_year": "",
    "pl_league": "",
    "pl_split": "",
    "pl_playoffs": "",
    "pl_txtA": "",
    "pl_txtB": "",
}

APP_STATE_MIGRATIONS: Dict[str, str] = {
    "resumo_combo_use_missmatch": "resumo_combo_use_mismatch",
    "resumo_mix_w_hist": "resumo_mix_w_laplace",
}

APP_STATE_CONSTRAINTS: Dict[str, Tuple[float, float]] = {
    "resumo_max_odd": (1.01, 10.0),
    "resumo_mix_w_model": (0.0, 100.0),
    "resumo_mix_w_laplace": (0.0, 100.0),
    "resumo_mix_w_sc": (0.0, 100.0),
    "resumo_sc_thr": (50.0, 90.0),
    "resumo_min_sample_liga": (0.0, 100.0),
    "resumo_min_sample_h2h": (0.0, 30.0),
    "resumo_hist_shrink_n": (0.0, 100.0),
    "n_sims": (1000.0, 100000.0),
    "half_life_days": (15.0, 365.0),
    "wf_ml_corr_strength": (0.0, 1.0),
    "fusion_team_weight_early": (0.0, 1.0),
    "fusion_team_weight_mid": (0.0, 1.0),
    "fusion_team_weight_playoffs": (0.0, 1.0),
    "fusion_early_season_team_weight": (0.0, 1.0),
    "fusion_season_knee_games": (1.0, 80.0),
    "fusion_coverage_power": (0.0, 4.0),
    "fusion_transfer_boost": (0.0, 2.0),
    "fusion_divergence_pp_cap": (5.0, 80.0),
    "fusion_divergence_low_coverage": (0.0, 1.0),
    "fusion_divergence_shrink": (0.0, 1.0),
    "ml_form_blend_max_weight": (0.0, 0.95),
    "ml_form_blend_knee_games": (1.0, 40.0),
    "ml_form_blend_beta": (0.5, 8.0),
    "ml_form_blend_signal_power": (0.1, 3.0),
    "ml_form_blend_parity_strength": (0.0, 0.8),
    "ml_form_override_min_games": (3.0, 30.0),
    "ml_form_override_div_pp": (0.0, 30.0),
    "ml_form_override_max_w": (0.0, 0.95),
    "ml_hybrid_blend_max_weight": (0.0, 0.95),
    "ml_hybrid_blend_knee_games": (1.0, 80.0),
    "ml_hybrid_momentum_games": (3.0, 30.0),
    "ml_hybrid_beta": (0.5, 8.0),
    "ml_conf_guard_min_games": (1.0, 60.0),
    "ml_conf_guard_max_shrink": (0.0, 0.9),
    "ml_interregional_uncertainty_strength_mult": (0.0, 1.0),
    "ml_interregional_uncertainty_prior_mult": (0.1, 2.0),
    "ml_interregional_uncertainty_cap_mult": (0.0, 1.0),
    "ml_interregional_conf_guard_players_eff_weight": (0.0, 1.0),
    "ml_interregional_conf_guard_players_eff_cap": (0.0, 60.0),
    "ml_interregional_conf_guard_max_shrink_mult": (0.0, 1.0),
    "ml_patch_guard_warn_shrink": (0.0, 0.5),
    "ml_patch_guard_refit_shrink": (0.0, 0.7),
    "ml_patch_guard_focus_bonus": (0.0, 0.3),
    "ml_confidence_governor_max_total_shrink_pp": (0.0, 30.0),
    "ml_series_coherence_post_max_dev_pp_bo3": (2.0, 35.0),
    "ml_series_coherence_post_max_dev_pp_bo5": (2.0, 35.0),
}


def get_state_schema() -> Dict[str, Dict[str, Any]]:
    return {
        "defaults": dict(APP_STATE_DEFAULTS),
        "migrations": dict(APP_STATE_MIGRATIONS),
        "constraints": dict(APP_STATE_CONSTRAINTS),
    }
