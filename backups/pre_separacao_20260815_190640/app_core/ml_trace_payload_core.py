"""Build ML trace payload outside UI module."""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Callable, Dict, List, Mapping

from app_core.ml_output_models import MlTraceContext
from app_core.prob_math_core import safe_float as _f


def _opt_f(v: Any) -> float | None:
    x = _f(v, float("nan"))
    return float(x) if math.isfinite(x) else None


def _is_dict(v: Any) -> bool:
    return isinstance(v, dict)


def _pipeline_order_from_steps(steps: Any) -> List[str]:
    if not isinstance(steps, list):
        return []
    out: List[str] = []
    seen = set()
    for row in steps:
        if not isinstance(row, Mapping):
            continue
        stage = str(row.get("stage", "") or "").strip()
        if (not stage) or (stage in seen):
            continue
        seen.add(stage)
        out.append(stage)
    return out


def build_ml_trace_payload(
    *,
    ctx: Mapping[str, Any] | MlTraceContext,
    odd_from_prob_fn: Callable[[float], Any],
) -> Dict[str, Any]:
    ctx_map: Mapping[str, Any]
    if isinstance(ctx, MlTraceContext):
        ctx_map = ctx.to_dict()
    elif isinstance(ctx, Mapping):
        ctx_map = ctx
    else:
        ctx_map = {}
    g = (lambda k, d=None: ctx_map.get(k, d))
    now = str(g("generated_at", _dt.datetime.now().isoformat(timespec="seconds")))

    p_map_used = _f(g("p_map_used"), float("nan"))
    p_series_used = _f(g("p_series_used"), float("nan"))
    trace_patch_guard_info = g("trace_patch_guard_info", {}) if _is_dict(g("trace_patch_guard_info")) else {}
    trace_marginal_trust_info = g("trace_marginal_trust_info", {}) if _is_dict(g("trace_marginal_trust_info")) else {}
    trace_series_coherence_post_info = (
        g("trace_series_coherence_post_info", {}) if _is_dict(g("trace_series_coherence_post_info")) else {}
    )
    pipeline_steps = list(g("trace_pipeline_steps", [])) if isinstance(g("trace_pipeline_steps"), list) else []
    pipeline_order = list(g("trace_pipeline_order", [])) if isinstance(g("trace_pipeline_order"), list) else []
    derived_order = _pipeline_order_from_steps(pipeline_steps)
    if derived_order:
        pipeline_order = derived_order
    gov_state = g("_ml_confidence_governor_state", {}) if _is_dict(g("_ml_confidence_governor_state")) else {}
    gov_runtime_info = (
        g("_ml_confidence_governor_budget_runtime_info", {})
        if _is_dict(g("_ml_confidence_governor_budget_runtime_info"))
        else {}
    )
    gov_enabled = bool(g("ml_confidence_governor_enabled", True))
    gov_live_enabled = bool(g("ml_confidence_governor_live_enabled", True))
    gov_max_cfg = _opt_f(g("ml_confidence_governor_max_total_shrink_pp"))
    gov_budget_runtime = _opt_f(g("ml_confidence_governor_budget_runtime_pp"))
    gov_max_state = _opt_f(gov_state.get("max_total_shrink_pp"))
    gov_used = _opt_f(gov_state.get("used_shrink_pp"))
    gov_max = gov_max_state if gov_max_state is not None else gov_max_cfg
    gov_remaining = (
        float(max(0.0, float(gov_max) - float(gov_used)))
        if (gov_max is not None) and (gov_used is not None)
        else None
    )
    gov_limited_stages: List[str] = []
    gov_applied_by_stage: Dict[str, float] = {}
    for row in pipeline_steps:
        if not isinstance(row, Mapping):
            continue
        stage = str(row.get("stage", "") or "").strip()
        if not stage:
            continue
        if bool(row.get("governor_limited", False)) and stage not in gov_limited_stages:
            gov_limited_stages.append(stage)
        ap = _opt_f(row.get("governor_applied_shrink_pp"))
        if ap is not None:
            gov_applied_by_stage[stage] = float(ap)
    series_post_tuning_info = (
        g("_ml_series_coherence_post_tuning_info", {})
        if _is_dict(g("_ml_series_coherence_post_tuning_info"))
        else {}
    )
    series_post_max_dev_runtime = _opt_f(g("ml_series_coherence_post_max_dev_pp_runtime"))
    trace_fused_post_cal_info = g("_fused_post_cal_info", {}) if _is_dict(g("_fused_post_cal_info")) else {}

    out: Dict[str, Any] = {
        "generated_at": now,
        "matchup": {
            "season": str(g("season_label", "")),
            "teamA": str(g("teamA", "")),
            "teamB": str(g("teamB", "")),
            "bo": int(g("bo", 1) or 1),
        },
        "input_signature": {
            "sha256": str(g("_input_signature_hash", "")),
            "payload": g("_input_signature_payload", {}),
        },
        "filters": {
            "year": str(g("year_opt", "")),
            "split": str(g("split_opt", "")),
            "playoffs": str(g("playoffs_opt", "")),
            "league_mode": str(g("league_mode", "")),
            "fixed_league": str(g("fixed_league", "") or ""),
        },
        "engine": str(g("ml_engine", "")),
        "model_mode": "Padrao",
        "probabilities": {
            "p_raw": _opt_f(g("p_map_raw")),
            "p_cal": _opt_f(g("p_map_cal")),
            "p_team": _opt_f(g("trace_p_map_team")),
            "p_players": _opt_f(g("trace_p_map_players")),
            "p_fused": _opt_f(g("trace_p_map_fused")),
            "p_map_used": _opt_f(p_map_used),
            "p_series_used": _opt_f(p_series_used),
        },
        "players_adjustment": {
            "enabled": bool(g("use_lineup", False)),
            "mode": str(g("lineup_mode", "")),
            "delta_raw": float(g("delta_players", 0.0) or 0.0),
            "delta_effect": float(g("delta_players_eff", 0.0) or 0.0),
            "delta_mode": str(g("_delta_mode", "")),
            "delta_cap": float(g("_delta_cap", 0.0) or 0.0),
            "delta_slope": float(g("_delta_slope", 0.0) or 0.0),
            "n_team_eff": _opt_f(g("trace_n_team_eff")),
            "n_players_eff": _opt_f(g("trace_n_players_eff")),
            "team_scale_used": float(g("team_scale_used", 1.0) or 1.0),
            "players_scale_used": float(g("players_scale_used", 1.0) or 1.0),
            "scale_league_used": str(g("scale_league_used", "") or ""),
            "delta_transfer_adj": _opt_f((g("trace_fusion_info", {}) or {}).get("delta_transfer_adj")) if _is_dict(g("trace_fusion_info")) else None,
            "player_portability_summary": (
                dict((g("trace_fusion_info", {}) or {}).get("player_portability_summary", {}))
                if _is_dict((g("trace_fusion_info", {}) or {}).get("player_portability_summary", {}))
                else {}
            ),
        },
        "fusion": {
            "season_phase": str((g("trace_fusion_info", {}) or {}).get("season_phase", "")) if _is_dict(g("trace_fusion_info")) else "",
            "coverage_eff": _opt_f((g("trace_fusion_info", {}) or {}).get("coverage_eff")) if _is_dict(g("trace_fusion_info")) else None,
            "w_team": _opt_f((g("trace_fusion_info", {}) or {}).get("w_team")) if _is_dict(g("trace_fusion_info")) else None,
            "w_players": _opt_f((g("trace_fusion_info", {}) or {}).get("w_players")) if _is_dict(g("trace_fusion_info")) else None,
            "reason": str((g("trace_fusion_info", {}) or {}).get("reason", "")) if _is_dict(g("trace_fusion_info")) else "",
            "coherence_guard": (g("trace_fusion_info", {}) or {}).get("coherence_guard", {}) if _is_dict(g("trace_fusion_info")) else {},
        },
        "season_form_blend": {
            "applied": bool((g("trace_form_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_form_info")) else False,
            "source": str((g("trace_form_info", {}) or {}).get("source", "")) if _is_dict(g("trace_form_info")) else "",
            "p_model": _opt_f((g("trace_form_info", {}) or {}).get("p_model")) if _is_dict(g("trace_form_info")) else None,
            "p_form": _opt_f((g("trace_form_info", {}) or {}).get("p_form")) if _is_dict(g("trace_form_info")) else None,
            "p_blended": _opt_f((g("trace_form_info", {}) or {}).get("p_blended")) if _is_dict(g("trace_form_info")) else None,
            "w_form": _opt_f((g("trace_form_info", {}) or {}).get("w_form")) if _is_dict(g("trace_form_info")) else None,
            "n_a": int((g("trace_form_info", {}) or {}).get("n_a")) if _is_dict(g("trace_form_info")) and (g("trace_form_info", {}) or {}).get("n_a") is not None else None,
            "n_b": int((g("trace_form_info", {}) or {}).get("n_b")) if _is_dict(g("trace_form_info")) and (g("trace_form_info", {}) or {}).get("n_b") is not None else None,
            "wins_a": int((g("trace_form_info", {}) or {}).get("wins_a")) if _is_dict(g("trace_form_info")) and (g("trace_form_info", {}) or {}).get("wins_a") is not None else None,
            "wins_b": int((g("trace_form_info", {}) or {}).get("wins_b")) if _is_dict(g("trace_form_info")) and (g("trace_form_info", {}) or {}).get("wins_b") is not None else None,
            "reason": str((g("trace_form_info", {}) or {}).get("reason", "")) if _is_dict(g("trace_form_info")) else "",
        },
        "hybrid_blend": {
            "applied": bool((g("trace_hybrid_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_hybrid_info")) else False,
            "enabled": bool((g("trace_hybrid_info", {}) or {}).get("enabled", False)) if _is_dict(g("trace_hybrid_info")) else False,
            "p_model": _opt_f((g("trace_hybrid_info", {}) or {}).get("p_model")) if _is_dict(g("trace_hybrid_info")) else None,
            "p_hybrid": _opt_f((g("trace_hybrid_info", {}) or {}).get("p_hybrid")) if _is_dict(g("trace_hybrid_info")) else None,
            "p_blended": _opt_f((g("trace_hybrid_info", {}) or {}).get("p_blended")) if _is_dict(g("trace_hybrid_info")) else None,
            "w_hybrid": _opt_f((g("trace_hybrid_info", {}) or {}).get("w_hybrid")) if _is_dict(g("trace_hybrid_info")) else None,
            "n_eff": _opt_f((g("trace_hybrid_info", {}) or {}).get("n_eff")) if _is_dict(g("trace_hybrid_info")) else None,
            "agree_to_model": _opt_f((g("trace_hybrid_info", {}) or {}).get("agree_to_model")) if _is_dict(g("trace_hybrid_info")) else None,
            "reason": str((g("trace_hybrid_info", {}) or {}).get("reason", "")) if _is_dict(g("trace_hybrid_info")) else "",
        },
        "confidence_guard": {
            "enabled": bool((g("trace_conf_guard_info", {}) or {}).get("enabled", False)) if _is_dict(g("trace_conf_guard_info")) else False,
            "applied": bool((g("trace_conf_guard_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_conf_guard_info")) else False,
            "n_ref": _opt_f((g("trace_conf_guard_info", {}) or {}).get("n_ref")) if _is_dict(g("trace_conf_guard_info")) else None,
            "min_games": _opt_f((g("trace_conf_guard_info", {}) or {}).get("min_games")) if _is_dict(g("trace_conf_guard_info")) else None,
            "coverage": _opt_f((g("trace_conf_guard_info", {}) or {}).get("coverage")) if _is_dict(g("trace_conf_guard_info")) else None,
            "shrink": _opt_f((g("trace_conf_guard_info", {}) or {}).get("shrink")) if _is_dict(g("trace_conf_guard_info")) else None,
            "p_in": _opt_f((g("trace_conf_guard_info", {}) or {}).get("p_in")) if _is_dict(g("trace_conf_guard_info")) else None,
            "p_out": _opt_f((g("trace_conf_guard_info", {}) or {}).get("p_out")) if _is_dict(g("trace_conf_guard_info")) else None,
            "governor": (
                dict((g("trace_conf_guard_info", {}) or {}).get("governor", {}))
                if _is_dict((g("trace_conf_guard_info", {}) or {}).get("governor", {}))
                else {}
            ),
        },
        "uncertainty_layer": {
            "enabled": bool((g("trace_uncertainty_info", {}) or {}).get("enabled", False)) if _is_dict(g("trace_uncertainty_info")) else False,
            "applied": bool((g("trace_uncertainty_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_uncertainty_info")) else False,
            "skipped_by_league": bool((g("trace_uncertainty_info", {}) or {}).get("skipped_by_league", False)) if _is_dict(g("trace_uncertainty_info")) else False,
            "league": str((g("trace_uncertainty_info", {}) or {}).get("league", "")) if _is_dict(g("trace_uncertainty_info")) else "",
            "exclude_leagues": str((g("trace_uncertainty_info", {}) or {}).get("exclude_leagues", "")) if _is_dict(g("trace_uncertainty_info")) else "",
            "n_eff": _opt_f((g("trace_uncertainty_info", {}) or {}).get("n_eff")) if _is_dict(g("trace_uncertainty_info")) else None,
            "prior_games": _opt_f((g("trace_uncertainty_info", {}) or {}).get("prior_games")) if _is_dict(g("trace_uncertainty_info")) else None,
            "raw_shrink": _opt_f((g("trace_uncertainty_info", {}) or {}).get("raw_shrink")) if _is_dict(g("trace_uncertainty_info")) else None,
            "shrink": _opt_f((g("trace_uncertainty_info", {}) or {}).get("shrink")) if _is_dict(g("trace_uncertainty_info")) else None,
            "p_in": _opt_f((g("trace_uncertainty_info", {}) or {}).get("p_in")) if _is_dict(g("trace_uncertainty_info")) else None,
            "p_out": _opt_f((g("trace_uncertainty_info", {}) or {}).get("p_out")) if _is_dict(g("trace_uncertainty_info")) else None,
            "governor": (
                dict((g("trace_uncertainty_info", {}) or {}).get("governor", {}))
                if _is_dict((g("trace_uncertainty_info", {}) or {}).get("governor", {}))
                else {}
            ),
        },
        "fused_post_calibration": {
            "enabled": bool(trace_fused_post_cal_info.get("enabled", False)),
            "applied": bool(trace_fused_post_cal_info.get("applied", False)),
            "reason": str(trace_fused_post_cal_info.get("reason", "")),
            "source": str(trace_fused_post_cal_info.get("source", "")),
            "version": str(trace_fused_post_cal_info.get("version", "")),
            "league_hint": str(trace_fused_post_cal_info.get("league_hint", "")),
            "gate_decision": str(trace_fused_post_cal_info.get("gate_decision", "")),
            "a": _opt_f(trace_fused_post_cal_info.get("a")),
            "b": _opt_f(trace_fused_post_cal_info.get("b")),
            "max_abs_logit_shift": _opt_f(trace_fused_post_cal_info.get("max_abs_logit_shift")),
            "logit_shift_raw": _opt_f(trace_fused_post_cal_info.get("logit_shift_raw")),
            "logit_shift_applied": _opt_f(trace_fused_post_cal_info.get("logit_shift_applied")),
            "p_in": _opt_f(trace_fused_post_cal_info.get("p_in")),
            "p_out": _opt_f(trace_fused_post_cal_info.get("p_out")),
            "delta_pp": _opt_f(trace_fused_post_cal_info.get("delta_pp")),
        },
        "marginal_trust_guard": {
            "enabled": bool(trace_marginal_trust_info.get("enabled", False)),
            "applied": bool(trace_marginal_trust_info.get("applied", False)),
            "reason": str(trace_marginal_trust_info.get("reason", "")),
            "league": str(trace_marginal_trust_info.get("league", "")),
            "tier": str(trace_marginal_trust_info.get("tier", "")),
            "band": str(trace_marginal_trust_info.get("band", "")),
            "band_tuning": bool(trace_marginal_trust_info.get("band_tuning", False)),
            "is_marginal": bool(trace_marginal_trust_info.get("is_marginal", False)),
            "excluded": bool(trace_marginal_trust_info.get("excluded", False)),
            "gate_score": _opt_f(trace_marginal_trust_info.get("gate_score")),
            "gate_n_eff": _opt_f(trace_marginal_trust_info.get("gate_n_eff")),
            "confidence_score": _opt_f(trace_marginal_trust_info.get("confidence_score")),
            "weak_blend": _opt_f(trace_marginal_trust_info.get("weak_blend")),
            "n_eff": _opt_f(trace_marginal_trust_info.get("n_eff")),
            "shrink": _opt_f(trace_marginal_trust_info.get("shrink")),
            "cap_hi": _opt_f(trace_marginal_trust_info.get("cap_hi")),
            "cap_applied": bool(trace_marginal_trust_info.get("cap_applied", False)),
            "p_in": _opt_f(trace_marginal_trust_info.get("p_in")),
            "p_out": _opt_f(trace_marginal_trust_info.get("p_out")),
            "governor": (dict(trace_marginal_trust_info.get("governor", {})) if _is_dict(trace_marginal_trust_info.get("governor", {})) else {}),
        },
        "patch_meta_guard": {
            "enabled": bool(trace_patch_guard_info.get("enabled", False)),
            "applied": bool(trace_patch_guard_info.get("applied", False)),
            "status": str(trace_patch_guard_info.get("status", "")),
            "league_hint": str(trace_patch_guard_info.get("league_hint", "")),
            "changed_leagues": list(trace_patch_guard_info.get("changed_leagues", [])),
            "age_days": _opt_f(trace_patch_guard_info.get("age_days")),
            "shrink": _opt_f(trace_patch_guard_info.get("shrink")),
            "p_in": _opt_f(trace_patch_guard_info.get("p_in")),
            "p_out": _opt_f(trace_patch_guard_info.get("p_out")),
            "governor": (dict(trace_patch_guard_info.get("governor", {})) if _is_dict(trace_patch_guard_info.get("governor", {})) else {}),
        },
        "series_gamma": {
            "enabled": bool((g("trace_series_gamma_info", {}) or {}).get("enabled", False)) if _is_dict(g("trace_series_gamma_info")) else False,
            "applied": bool((g("trace_series_gamma_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_series_gamma_info")) else False,
            "gamma": _opt_f((g("trace_series_gamma_info", {}) or {}).get("gamma")) if _is_dict(g("trace_series_gamma_info")) else None,
            "source": str((g("trace_series_gamma_info", {}) or {}).get("source", "")) if _is_dict(g("trace_series_gamma_info")) else "",
            "bo": int((g("trace_series_gamma_info", {}) or {}).get("bo")) if _is_dict(g("trace_series_gamma_info")) and (g("trace_series_gamma_info", {}) or {}).get("bo") is not None else None,
            "p_base": _opt_f((g("trace_series_gamma_info", {}) or {}).get("p_base")) if _is_dict(g("trace_series_gamma_info")) else None,
            "p_out": _opt_f((g("trace_series_gamma_info", {}) or {}).get("p_out")) if _is_dict(g("trace_series_gamma_info")) else None,
            "league_hint": str((g("trace_series_gamma_info", {}) or {}).get("league_hint", "")) if _is_dict(g("trace_series_gamma_info")) else "",
        },
        "series_economic_coherence": {
            "enabled": bool((g("trace_series_coherence_info", {}) or {}).get("enabled", True)) if _is_dict(g("trace_series_coherence_info")) else True,
            "applied": bool((g("trace_series_coherence_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_series_coherence_info")) else False,
            "reason": str((g("trace_series_coherence_info", {}) or {}).get("reason", "")) if _is_dict(g("trace_series_coherence_info")) else "",
            "bo": int((g("trace_series_coherence_info", {}) or {}).get("bo")) if _is_dict(g("trace_series_coherence_info")) and (g("trace_series_coherence_info", {}) or {}).get("bo") is not None else None,
            "p_map": _opt_f((g("trace_series_coherence_info", {}) or {}).get("p_map")) if _is_dict(g("trace_series_coherence_info")) else None,
            "p_series_before": _opt_f((g("trace_series_coherence_info", {}) or {}).get("p_series_before")) if _is_dict(g("trace_series_coherence_info")) else None,
            "p_series_after": _opt_f((g("trace_series_coherence_info", {}) or {}).get("p_series_after")) if _is_dict(g("trace_series_coherence_info")) else None,
            "p_target": _opt_f((g("trace_series_coherence_info", {}) or {}).get("p_target")) if _is_dict(g("trace_series_coherence_info")) else None,
            "diff_pp": _opt_f((g("trace_series_coherence_info", {}) or {}).get("diff_pp")) if _is_dict(g("trace_series_coherence_info")) else None,
            "bounds": (g("trace_series_coherence_info", {}) or {}).get("bounds", {}) if _is_dict(g("trace_series_coherence_info")) else {},
        },
        "series_economic_coherence_post": {
            "enabled": bool(trace_series_coherence_post_info.get("enabled", True)),
            "applied": bool(trace_series_coherence_post_info.get("applied", False)),
            "reason": str(trace_series_coherence_post_info.get("reason", "")),
            "bo": (
                int(trace_series_coherence_post_info.get("bo"))
                if trace_series_coherence_post_info.get("bo") is not None
                else None
            ),
            "p_map": _opt_f(trace_series_coherence_post_info.get("p_map")),
            "p_series_before": _opt_f(trace_series_coherence_post_info.get("p_series_before")),
            "p_series_after": _opt_f(trace_series_coherence_post_info.get("p_series_after")),
            "p_target": _opt_f(trace_series_coherence_post_info.get("p_target")),
            "diff_pp": _opt_f(trace_series_coherence_post_info.get("diff_pp")),
            "bounds": (
                dict(trace_series_coherence_post_info.get("bounds", {}))
                if _is_dict(trace_series_coherence_post_info.get("bounds", {}))
                else {}
            ),
            "max_dev_pp_runtime": series_post_max_dev_runtime,
            "tuning": dict(series_post_tuning_info),
        },
        "pipeline": {
            "version": "v1",
            "order": list(pipeline_order),
            "steps": list(pipeline_steps),
            "final_p_map": _opt_f(p_map_used),
            "final_p_series": _opt_f(p_series_used),
        },
        "guards": {
            "consistency_guard_on": bool(g("consistency_guard_on", True)),
            "consistency_guard_triggered": bool(g("_guard_triggered", False)),
            "consistency_guard_drift_pp": _opt_f(g("_guard_dpp")),
            "wf_correction_applied": bool((g("_wf_info", {}) or {}).get("applied", False)) if _is_dict(g("_wf_info")) else False,
            "wf_shrink": _opt_f((g("_wf_info", {}) or {}).get("shrink")) if _is_dict(g("_wf_info")) else None,
            "wf_league": str((g("_wf_info", {}) or {}).get("league") or "") if _is_dict(g("_wf_info")) else "",
            "fused_post_cal_applied": bool(trace_fused_post_cal_info.get("applied", False)),
            "confidence_governor_on": bool(gov_enabled),
            "confidence_governor_live_on": bool(gov_live_enabled),
            "patch_guard_on": bool(trace_patch_guard_info.get("enabled", False)),
            "patch_guard_applied": bool(trace_patch_guard_info.get("applied", False)),
            "patch_guard_status": str(trace_patch_guard_info.get("status", "")),
            "state_areal_live_on": bool(g("_ff_ml_state_areal_live", False)),
            "state_areal_live_applied": bool((g("trace_state_areal_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_state_areal_live")) else False,
            "state_alite_live_on": bool(g("_ff_ml_state_alite_live", False)),
            "state_alite_live_applied": bool((g("trace_state_alite_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_state_alite_live")) else False,
            "style_b_lite_live_on": bool(g("_ff_ml_style_b_lite_live", False)),
            "style_b_lite_live_applied": bool((g("trace_style_b_lite_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_style_b_lite_live")) else False,
            "league_rank_sos_live_on": bool(g("_ff_ml_league_rank_sos_live", False)),
            "league_rank_sos_live_applied": bool((g("trace_league_rank_sos_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_league_rank_sos_live")) else False,
            "roster_continuity_live_on": bool(g("_ff_ml_roster_continuity_live", False)),
            "roster_continuity_live_applied": bool((g("trace_roster_continuity_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_roster_continuity_live")) else False,
            "hier_calibration_live_on": bool(g("_ff_ml_hier_calibration_live", False)),
            "hier_calibration_live_applied": bool((g("trace_hier_calibration_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_hier_calibration_live")) else False,
            "series_bo_dplite_live_on": bool(g("_ff_ml_series_bo_dplite_live", False)),
            "series_bo_dplite_live_applied": bool((g("trace_series_bo_dplite_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_series_bo_dplite_live")) else False,
            "series_economic_coherence_applied": bool((g("trace_series_coherence_info", {}) or {}).get("applied", False)) if _is_dict(g("trace_series_coherence_info")) else False,
            "series_economic_coherence_post_applied": bool(trace_series_coherence_post_info.get("applied", False)),
            "layer_uncertainty_live_on": bool(g("_ff_ml_layer_uncertainty_live", False)),
            "layer_uncertainty_live_applied": bool((g("trace_layer_uncertainty_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_layer_uncertainty_live")) else False,
            "marginal_trust_guard_on": bool(g("marginal_trust_guard_on", True)),
            "marginal_trust_guard_applied": bool(trace_marginal_trust_info.get("applied", False)),
            "meta_c_lite_live_on": bool(g("_ff_ml_meta_c_lite_live", False)),
            "meta_c_lite_live_applied": bool((g("trace_meta_c_lite_live", {}) or {}).get("applied", False)) if _is_dict(g("trace_meta_c_lite_live")) else False,
        },
        "confidence_governor": {
            "enabled": bool(gov_enabled),
            "live_enabled": bool(gov_live_enabled),
            "max_total_shrink_pp_config": gov_max_cfg,
            "budget_runtime_pp": gov_budget_runtime,
            "max_total_shrink_pp_state": gov_max_state,
            "used_shrink_pp": gov_used,
            "remaining_shrink_pp": gov_remaining,
            "limited_stages": list(gov_limited_stages),
            "applied_shrink_pp_by_stage": dict(gov_applied_by_stage),
            "state": dict(gov_state),
            "tuning": dict(gov_runtime_info),
        },
        "shadow_vnext": (g("trace_shadow_ml_vnext", {}) if _is_dict(g("trace_shadow_ml_vnext")) else {}),
        "shadow_state_alite": (g("trace_shadow_state_alite", {}) if _is_dict(g("trace_shadow_state_alite")) else {}),
        "shadow_state_areal": (g("trace_shadow_state_areal", {}) if _is_dict(g("trace_shadow_state_areal")) else {}),
        "state_areal_live": (g("trace_state_areal_live", {}) if _is_dict(g("trace_state_areal_live")) else {}),
        "state_alite_live": (g("trace_state_alite_live", {}) if _is_dict(g("trace_state_alite_live")) else {}),
        "style_b_lite_live": (g("trace_style_b_lite_live", {}) if _is_dict(g("trace_style_b_lite_live")) else {}),
        "league_rank_sos_live": (g("trace_league_rank_sos_live", {}) if _is_dict(g("trace_league_rank_sos_live")) else {}),
        "roster_continuity_live": (g("trace_roster_continuity_live", {}) if _is_dict(g("trace_roster_continuity_live")) else {}),
        "hier_calibration_live": (g("trace_hier_calibration_live", {}) if _is_dict(g("trace_hier_calibration_live")) else {}),
        "series_bo_dplite_live": (g("trace_series_bo_dplite_live", {}) if _is_dict(g("trace_series_bo_dplite_live")) else {}),
        "series_economic_coherence_post_raw": (trace_series_coherence_post_info if _is_dict(trace_series_coherence_post_info) else {}),
        "layer_uncertainty_live": (g("trace_layer_uncertainty_live", {}) if _is_dict(g("trace_layer_uncertainty_live")) else {}),
        "meta_c_lite_live": (g("trace_meta_c_lite_live", {}) if _is_dict(g("trace_meta_c_lite_live")) else {}),
        "odds": {
            "teamA_map": float(odd_from_prob_fn(float(p_map_used))) if math.isfinite(float(p_map_used)) else None,
            "teamB_map": float(odd_from_prob_fn(float(1.0 - float(p_map_used)))) if math.isfinite(float(p_map_used)) else None,
        },
    }

    return out
