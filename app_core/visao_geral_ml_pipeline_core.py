from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

from app_core.ml_output_models import MlTraceContext
from app_core.ml_pre_pipeline_models import MlPreEngineInput, MlPreLiveInput
from app_core.visao_geral_runtime_core import (
    build_input_signature_payload,
    compute_input_signature,
)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return float(out) if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def compute_team_games_min_used(ml_info: Mapping[str, Any] | None) -> float:
    try:
        blue_games = float(((_mapping(ml_info).get("blue") or {}).get("games_played", 0) or 0))
        red_games = float(((_mapping(ml_info).get("red") or {}).get("games_played", 0) or 0))
        return float(min(blue_games, red_games))
    except Exception:
        return float("nan")


def persist_runtime_analysis_state(
    *,
    session_state: MutableMapping[str, Any],
    p_map_used: Any,
    p_series_used: Any,
    diff_used: Any,
    delta_players: Any,
    delta_players_eff: Any,
    ml_info: Mapping[str, Any] | None,
) -> float:
    team_games_min_used = compute_team_games_min_used(ml_info)
    session_state["_p_map_used"] = _float(p_map_used)
    session_state["_p_series_used"] = _float(p_series_used)
    session_state["_ml_diff_used"] = _float(diff_used)
    session_state["_ml_delta_players"] = _float(delta_players, 0.0)
    session_state["_ml_delta_players_eff"] = _float(delta_players_eff, 0.0)
    session_state["_ml_team_games_min"] = _float(team_games_min_used)
    return float(team_games_min_used)


def build_pre_engine_input(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    ml_info: Mapping[str, Any] | None,
    lgA: Any,
    lineup_mode: Any,
    pl_enabled: bool,
    pl_players_path: Any,
    lineupA_sel: Mapping[str, Any] | None,
    lineupB_sel: Mapping[str, Any] | None,
    as_of_date: Any,
    ml_engine: Any,
    use_lineup: bool,
    artifact_path: Any,
    pl_w: Any,
    playoffs_opt: Any,
    teamA: Any,
    teamB: Any,
    team_aliases_a: List[str] | None,
    team_aliases_b: List[str] | None,
    teamA_key: Any,
    teamB_key: Any,
    team_games: Any,
    csv_path: Any,
    filters: Any,
    diff_used: Any,
    delta_players: Any,
    delta_players_eff: Any,
    trace_p_map_team: Any,
    trace_p_map_players: Any,
    trace_p_map_fused: Any,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    trace_fusion_info: Mapping[str, Any] | None,
    trace_uncertainty_info: Mapping[str, Any] | None,
    league_transfer_features: Mapping[str, Any] | None,
    p_pipe_prev: Any,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    callbacks: Mapping[str, Callable[..., Any]] | None,
    format_num_fn: Callable[[Any, int], str] | None,
    format_pct_fn: Callable[[Any], str] | None,
    caption_fn: Callable[[str], None] | None,
    warning_fn: Callable[[str], None] | None,
    match_league: Any = "",
) -> MlPreEngineInput:
    return MlPreEngineInput(
        p_map_used=p_map_used,
        p_series_used=p_series_used,
        bo=int(bo),
        ml_info=_mapping(ml_info),
        lgA=str(lgA or ""),
        lineup_mode=str(lineup_mode or ""),
        pl_enabled=bool(pl_enabled),
        pl_players_path=pl_players_path,
        lineupA_sel=_mapping(lineupA_sel),
        lineupB_sel=_mapping(lineupB_sel),
        as_of_date=as_of_date,
        ml_engine=str(ml_engine),
        use_lineup=bool(use_lineup),
        artifact_path=artifact_path,
        pl_w=pl_w,
        playoffs_opt=playoffs_opt,
        teamA=str(teamA),
        teamB=str(teamB),
        team_aliases_a=[str(v) for v in list(team_aliases_a or []) if str(v or "").strip()],
        team_aliases_b=[str(v) for v in list(team_aliases_b or []) if str(v or "").strip()],
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
        trace_fusion_info=_mapping(trace_fusion_info),
        trace_uncertainty_info=_mapping(trace_uncertainty_info),
        league_transfer_features=_mapping(league_transfer_features),
        p_pipe_prev=p_pipe_prev,
        session_state=session_state,
        trace_pipeline_steps=_list_of_dicts(trace_pipeline_steps),
        callbacks=dict(callbacks or {}),
        format_num_fn=format_num_fn,
        format_pct_fn=format_pct_fn,
        caption_fn=caption_fn,
        warning_fn=warning_fn,
        match_league=str(match_league or ""),
    )


def build_pre_live_input(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    season_label: Any,
    teamA: Any,
    teamB: Any,
    ml_engine: Any,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    fixed_league: Any,
    as_of_date: Any,
    artifact_path: Any,
    lineup_mode: Any,
    lineupA_sel: Mapping[str, Any] | None,
    lineupB_sel: Mapping[str, Any] | None,
    diff_used: Any,
    delta_players: Any,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    trace_patch_guard_info: Mapping[str, Any] | None,
    trace_style_b_lite_live: Mapping[str, Any] | None,
    ml_conf_score_snapshot: Any,
    lg_a_shadow: Any,
    lg_b_shadow: Any,
    csv_path: Any,
    fit_params_path: Any,
    filters: Any,
    dfA_used: Any,
    dfB_used: Any,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    callbacks: Mapping[str, Callable[..., Any]] | None,
    caption_fn: Callable[[str], None] | None,
) -> MlPreLiveInput:
    return MlPreLiveInput(
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
        lineup_mode=str(lineup_mode or ""),
        lineupA_sel=_mapping(lineupA_sel),
        lineupB_sel=_mapping(lineupB_sel),
        diff_used=diff_used,
        delta_players=delta_players,
        trace_n_team_eff=trace_n_team_eff,
        trace_n_players_eff=trace_n_players_eff,
        trace_patch_guard_info=_mapping(trace_patch_guard_info),
        trace_style_b_lite_live=_mapping(trace_style_b_lite_live),
        ml_conf_score_snapshot=ml_conf_score_snapshot,
        lg_a_shadow=str(lg_a_shadow or ""),
        lg_b_shadow=str(lg_b_shadow or ""),
        csv_path=csv_path,
        fit_params_path=fit_params_path,
        filters=filters,
        dfA_used=dfA_used,
        dfB_used=dfB_used,
        session_state=session_state,
        trace_pipeline_steps=_list_of_dicts(trace_pipeline_steps),
        callbacks=dict(callbacks or {}),
        caption_fn=caption_fn,
    )


def _safe_file_signature(path_value: Any, file_sig_fn: Callable[[str], Any]) -> tuple[str, int, int]:
    try:
        path_str = str(path_value or "").strip()
        if path_str and Path(path_str).exists():
            sig = file_sig_fn(path_str)
            if isinstance(sig, tuple) and len(sig) >= 3:
                return (str(sig[0] or ""), int(sig[1] or 0), int(sig[2] or 0))
    except Exception:
        pass
    return ("", 0, 0)


def build_runtime_input_signature(
    *,
    session_state: Mapping[str, Any],
    season_label: Any,
    teamA: Any,
    teamB: Any,
    bo: int,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    fixed_league: Any,
    weight_mode: Any,
    half_life_days: Any,
    ml_engine: Any,
    totals_mode: Any,
    ml_mode: Any,
    map_mode_ui: Any,
    n_sims: Any,
    lineup_mode: Any,
    lineup_sig_a: Any,
    lineup_sig_b: Any,
    csv_path: Any,
    artifact_path: Any,
    fit_params_path: Any,
    ml_info: Mapping[str, Any] | None,
    file_sig_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    csv_sig = _safe_file_signature(csv_path, file_sig_fn)
    artifact_sig = _safe_file_signature(artifact_path, file_sig_fn)
    fit_sig = _safe_file_signature(fit_params_path, file_sig_fn)
    payload = build_input_signature_payload(
        matchup={
            "season": str(season_label),
            "teamA": str(teamA),
            "teamB": str(teamB),
            "bo": int(bo),
        },
        filters={
            "year": str(year_opt),
            "split": str(split_opt),
            "playoffs": str(playoffs_opt),
            "league_mode": str(league_mode),
            "fixed_league": str(fixed_league or ""),
            "weight_mode": str(weight_mode),
            "half_life_days": int(half_life_days) if half_life_days is not None else None,
        },
        config={
            "engine": str(ml_engine),
            "totals_mode": str(totals_mode),
            "ml_mode": str(ml_mode),
            "map_mode": str(map_mode_ui),
            "n_sims": int(n_sims),
            "ml_microseg_enabled": bool(session_state.get("ml_microseg_enabled", True)),
            "ml_uncertainty_enabled": bool(session_state.get("ml_uncertainty_enabled", True)),
            "ml_uncertainty_exclude_leagues": str(session_state.get("ml_uncertainty_exclude_leagues", "") or ""),
            "ml_confidence_governor_budget_runtime_pp": (
                float(session_state.get("_ml_confidence_governor_budget_runtime_pp"))
                if session_state.get("_ml_confidence_governor_budget_runtime_pp") is not None
                else None
            ),
            "ml_series_coherence_post_max_dev_pp_runtime": (
                float(session_state.get("_ml_series_coherence_post_max_dev_pp_runtime"))
                if session_state.get("_ml_series_coherence_post_max_dev_pp_runtime") is not None
                else None
            ),
            "use_lineup_adjust": bool(session_state.get("use_lineup_adjust", False)),
        },
        lineup={
            "mode": str(lineup_mode or ""),
            "sigA": str(lineup_sig_a or ""),
            "sigB": str(lineup_sig_b or ""),
        },
        versions={
            "csv": {"path": csv_sig[0], "size": csv_sig[1], "mtime": csv_sig[2]},
            "artifact": {"path": artifact_sig[0], "size": artifact_sig[1], "mtime": artifact_sig[2]},
            "params_fitted": {"path": fit_sig[0], "size": fit_sig[1], "mtime": fit_sig[2]},
            "artifact_version": str((_mapping(ml_info)).get("artifact_version", "")),
        },
    )
    return compute_input_signature(payload)


def unpack_shadow_output(shadow_out: Mapping[str, Any] | None) -> Dict[str, Any]:
    src = _mapping(shadow_out)
    return {
        "_ff_ml_shadow_vnext": bool(src.get("_ff_ml_shadow_vnext", False)),
        "trace_shadow_ml_vnext": _mapping(src.get("trace_shadow_ml_vnext")),
        "_ff_ml_shadow_state_alite": bool(src.get("_ff_ml_shadow_state_alite", False)),
        "trace_shadow_state_alite": _mapping(src.get("trace_shadow_state_alite")),
        "_ff_ml_shadow_state_areal": bool(src.get("_ff_ml_shadow_state_areal", False)),
        "trace_shadow_state_areal": _mapping(src.get("trace_shadow_state_areal")),
        "_lg_a_shadow": str(src.get("_lg_a_shadow", "") or ""),
        "_lg_b_shadow": str(src.get("_lg_b_shadow", "") or ""),
    }


def unpack_post_layer_output(
    post_ml_out: Mapping[str, Any] | None,
    *,
    default_trace_pipeline_steps: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    src = _mapping(post_ml_out)
    return {
        "p_map_used": _float(src.get("p_map_used")),
        "p_series_used": _float(src.get("p_series_used")),
        "_ff_ml_state_areal_live": bool(src.get("_ff_ml_state_areal_live", False)),
        "_ff_ml_state_alite_live": bool(src.get("_ff_ml_state_alite_live", False)),
        "_ff_ml_style_b_lite_live": bool(src.get("_ff_ml_style_b_lite_live", False)),
        "_ff_ml_league_rank_sos_live": bool(src.get("_ff_ml_league_rank_sos_live", False)),
        "_ff_ml_roster_continuity_live": bool(src.get("_ff_ml_roster_continuity_live", False)),
        "_ff_ml_hier_calibration_live": bool(src.get("_ff_ml_hier_calibration_live", False)),
        "_ff_ml_meta_c_lite_live": bool(src.get("_ff_ml_meta_c_lite_live", False)),
        "_ff_ml_layer_uncertainty_live": bool(src.get("_ff_ml_layer_uncertainty_live", False)),
        "trace_state_alite_live": _mapping(src.get("trace_state_alite_live")),
        "trace_state_areal_live": _mapping(src.get("trace_state_areal_live")),
        "trace_style_b_lite_live": _mapping(src.get("trace_style_b_lite_live")),
        "trace_league_rank_sos_live": _mapping(src.get("trace_league_rank_sos_live")),
        "trace_roster_continuity_live": _mapping(src.get("trace_roster_continuity_live")),
        "trace_hier_calibration_live": _mapping(src.get("trace_hier_calibration_live")),
        "trace_meta_c_lite_live": _mapping(src.get("trace_meta_c_lite_live")),
        "trace_layer_uncertainty_live": _mapping(src.get("trace_layer_uncertainty_live")),
        "trace_marginal_trust_info": _mapping(src.get("trace_marginal_trust_info")),
        "trace_series_coherence_post_info": _mapping(src.get("trace_series_coherence_post_info")),
        "trace_pipeline_steps": (
            _list_of_dicts(src.get("trace_pipeline_steps"))
            if isinstance(src.get("trace_pipeline_steps"), list)
            else _list_of_dicts(default_trace_pipeline_steps)
        ),
        "trace_pipeline_order": (
            list(src.get("trace_pipeline_order", []))
            if isinstance(src.get("trace_pipeline_order"), list)
            else []
        ),
    }


def build_ml_trace_context(
    *,
    session_state: Mapping[str, Any],
    season_label: Any,
    teamA: Any,
    teamB: Any,
    bo: int,
    input_signature_hash: Any,
    input_signature_payload: Mapping[str, Any] | None,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    fixed_league: Any,
    ml_engine: Any,
    p_map_raw: Any,
    p_map_cal: Any,
    trace_p_map_team: Any,
    trace_p_map_players: Any,
    trace_p_map_fused: Any,
    p_map_used: Any,
    p_series_used: Any,
    use_lineup: bool,
    lineup_mode: Any,
    delta_players: Any,
    delta_players_eff: Any,
    delta_mode: Any,
    delta_cap: Any,
    delta_slope: Any,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    trace_fusion_info: Mapping[str, Any] | None,
    trace_form_info: Mapping[str, Any] | None,
    trace_hybrid_info: Mapping[str, Any] | None,
    trace_conf_guard_info: Mapping[str, Any] | None,
    trace_uncertainty_info: Mapping[str, Any] | None,
    trace_marginal_trust_info: Mapping[str, Any] | None,
    trace_patch_guard_info: Mapping[str, Any] | None,
    trace_series_gamma_info: Mapping[str, Any] | None,
    trace_series_coherence_info: Mapping[str, Any] | None,
    trace_series_coherence_post_info: Mapping[str, Any] | None,
    trace_pipeline_order: List[str] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    guard_triggered: bool,
    guard_dpp: Any,
    wf_info: Mapping[str, Any] | None,
    fused_post_cal_info: Mapping[str, Any] | None,
    shadow_output: Mapping[str, Any] | None,
    post_output: Mapping[str, Any] | None,
    trace_series_bo_dplite_live: Mapping[str, Any] | None,
    generated_at: str | None = None,
) -> MlTraceContext:
    ss = session_state if isinstance(session_state, Mapping) else {}
    shadow = _mapping(shadow_output)
    post = _mapping(post_output)
    return MlTraceContext.from_mapping(
        {
            "generated_at": str(generated_at or datetime.now().isoformat(timespec="seconds")),
            "season_label": str(season_label),
            "teamA": str(teamA),
            "teamB": str(teamB),
            "bo": int(bo),
            "_input_signature_hash": str(input_signature_hash or ""),
            "_input_signature_payload": _mapping(input_signature_payload),
            "year_opt": year_opt,
            "split_opt": split_opt,
            "playoffs_opt": playoffs_opt,
            "league_mode": league_mode,
            "fixed_league": fixed_league,
            "ml_engine": ml_engine,
            "p_map_raw": p_map_raw,
            "p_map_cal": p_map_cal,
            "trace_p_map_team": trace_p_map_team,
            "trace_p_map_players": trace_p_map_players,
            "trace_p_map_fused": trace_p_map_fused,
            "p_map_used": p_map_used,
            "p_series_used": p_series_used,
            "use_lineup": bool(use_lineup),
            "lineup_mode": str(lineup_mode or ""),
            "delta_players": _float(delta_players, 0.0),
            "delta_players_eff": _float(delta_players_eff, 0.0),
            "_delta_mode": str(delta_mode or ""),
            "_delta_cap": _float(delta_cap),
            "_delta_slope": _float(delta_slope),
            "trace_n_team_eff": trace_n_team_eff,
            "trace_n_players_eff": trace_n_players_eff,
            "team_scale_used": float(ss.get("_fit_team_scale_used", ss.get("_fit_team_scale", 1.0)) or 1.0),
            "players_scale_used": float(ss.get("_fit_players_scale_used", ss.get("_fit_players_scale", 1.0)) or 1.0),
            "scale_league_used": str(ss.get("_fit_scale_league_used", "") or ""),
            "trace_fusion_info": _mapping(trace_fusion_info),
            "trace_form_info": _mapping(trace_form_info),
            "trace_hybrid_info": _mapping(trace_hybrid_info),
            "trace_conf_guard_info": _mapping(trace_conf_guard_info),
            "trace_uncertainty_info": _mapping(trace_uncertainty_info),
            "trace_marginal_trust_info": _mapping(trace_marginal_trust_info),
            "trace_patch_guard_info": _mapping(trace_patch_guard_info),
            "trace_series_gamma_info": _mapping(trace_series_gamma_info),
            "trace_series_coherence_info": _mapping(trace_series_coherence_info),
            "trace_series_coherence_post_info": _mapping(trace_series_coherence_post_info),
            "trace_pipeline_order": list(trace_pipeline_order or []),
            "trace_pipeline_steps": _list_of_dicts(trace_pipeline_steps),
            "consistency_guard_on": bool(ss.get("ml_consistency_guard", True)),
            "ml_confidence_governor_enabled": bool(ss.get("ml_confidence_governor_enabled", True)),
            "ml_confidence_governor_live_enabled": bool(ss.get("ml_confidence_governor_live_enabled", True)),
            "ml_confidence_governor_max_total_shrink_pp": float(
                ss.get("ml_confidence_governor_max_total_shrink_pp", 8.0) or 8.0
            ),
            "ml_confidence_governor_budget_runtime_pp": (
                float(ss.get("_ml_confidence_governor_budget_runtime_pp"))
                if ss.get("_ml_confidence_governor_budget_runtime_pp") is not None
                else None
            ),
            "_ml_confidence_governor_budget_runtime_info": _mapping(
                ss.get("_ml_confidence_governor_budget_runtime_info")
            ),
            "_ml_confidence_governor_state": _mapping(ss.get("_ml_confidence_governor")),
            "ml_series_coherence_post_max_dev_pp_runtime": (
                float(ss.get("_ml_series_coherence_post_max_dev_pp_runtime"))
                if ss.get("_ml_series_coherence_post_max_dev_pp_runtime") is not None
                else None
            ),
            "_ml_series_coherence_post_tuning_info": _mapping(
                ss.get("_ml_series_coherence_post_tuning_info")
            ),
            "_guard_triggered": bool(guard_triggered),
            "_guard_dpp": guard_dpp,
            "_wf_info": _mapping(wf_info),
            "_fused_post_cal_info": _mapping(fused_post_cal_info),
            "_ff_ml_state_areal_live": bool(post.get("_ff_ml_state_areal_live", False)),
            "_ff_ml_state_alite_live": bool(post.get("_ff_ml_state_alite_live", False)),
            "_ff_ml_style_b_lite_live": bool(post.get("_ff_ml_style_b_lite_live", False)),
            "_ff_ml_league_rank_sos_live": bool(post.get("_ff_ml_league_rank_sos_live", False)),
            "_ff_ml_roster_continuity_live": bool(post.get("_ff_ml_roster_continuity_live", False)),
            "_ff_ml_hier_calibration_live": bool(post.get("_ff_ml_hier_calibration_live", False)),
            "_ff_ml_series_bo_dplite_live": bool(ss.get("ff_ml_series_bo_dplite_live", True))
            if "_ff_ml_series_bo_dplite_live" not in post
            else bool(post.get("_ff_ml_series_bo_dplite_live", False)),
            "_ff_ml_layer_uncertainty_live": bool(post.get("_ff_ml_layer_uncertainty_live", False)),
            "_ff_ml_meta_c_lite_live": bool(post.get("_ff_ml_meta_c_lite_live", False)),
            "marginal_trust_guard_on": bool(ss.get("ml_marginal_trust_enabled", True)),
            "trace_shadow_ml_vnext": _mapping(shadow.get("trace_shadow_ml_vnext")),
            "trace_shadow_state_alite": _mapping(shadow.get("trace_shadow_state_alite")),
            "trace_shadow_state_areal": _mapping(shadow.get("trace_shadow_state_areal")),
            "trace_state_areal_live": _mapping(post.get("trace_state_areal_live")),
            "trace_state_alite_live": _mapping(post.get("trace_state_alite_live")),
            "trace_style_b_lite_live": _mapping(post.get("trace_style_b_lite_live")),
            "trace_league_rank_sos_live": _mapping(post.get("trace_league_rank_sos_live")),
            "trace_roster_continuity_live": _mapping(post.get("trace_roster_continuity_live")),
            "trace_hier_calibration_live": _mapping(post.get("trace_hier_calibration_live")),
            "trace_series_bo_dplite_live": _mapping(trace_series_bo_dplite_live),
            "trace_layer_uncertainty_live": _mapping(post.get("trace_layer_uncertainty_live")),
            "trace_meta_c_lite_live": _mapping(post.get("trace_meta_c_lite_live")),
        }
    )
