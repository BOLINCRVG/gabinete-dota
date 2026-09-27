"""Typed input models for ML pre-pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping


def _require_callable(
    callbacks: Mapping[str, Any] | None,
    key: str,
    *,
    required: bool = True,
) -> Callable[..., Any] | None:
    cb = (callbacks or {}).get(str(key))
    if callable(cb):
        return cb
    if required:
        raise ValueError(f"missing callable callback: {key}")
    return None


@dataclass(slots=True)
class MlPreEngineInput:
    p_map_used: Any
    p_series_used: Any
    bo: int
    ml_info: Dict[str, Any]
    lgA: str
    lineup_mode: str
    pl_enabled: bool
    pl_players_path: Any
    lineupA_sel: Dict[str, Any]
    lineupB_sel: Dict[str, Any]
    as_of_date: Any
    ml_engine: str
    use_lineup: bool
    artifact_path: Any
    pl_w: Any
    playoffs_opt: Any
    teamA: str
    teamB: str
    team_aliases_a: List[str]
    team_aliases_b: List[str]
    teamA_key: str
    teamB_key: str
    team_games: Any
    csv_path: Any
    filters: Any
    diff_used: Any
    delta_players: Any
    delta_players_eff: Any
    trace_p_map_team: Any
    trace_p_map_players: Any
    trace_p_map_fused: Any
    trace_n_team_eff: Any
    trace_n_players_eff: Any
    trace_fusion_info: Dict[str, Any]
    trace_uncertainty_info: Dict[str, Any]
    league_transfer_features: Dict[str, Any]
    p_pipe_prev: Any
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None
    trace_pipeline_steps: List[Dict[str, Any]]
    callbacks: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    format_num_fn: Callable[[Any, int], str] | None = None
    format_pct_fn: Callable[[Any], str] | None = None
    caption_fn: Callable[[str], None] | None = None
    warning_fn: Callable[[str], None] | None = None
    match_league: str = ""
    # Calibrador com grupo (05/09/2026). Vem de
    # params_fitted.json["postfusion_group_calibration"]. Sem o bloco fica None
    # e o comportamento e exatamente o de antes — apagar a chave DESLIGA.
    # E o UNICO estagio pos-fusao que o app passa a rodar: o microssegmento da
    # fusao e o Platt da fusao seguem desligados, porque foi medido que custam
    # (divida D8). Nao injetar os callbacks deles sem medir.
    postfusion_grupo_coef: Mapping[str, Any] | None = None

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "p_map_used": self.p_map_used,
            "p_series_used": self.p_series_used,
            "bo": int(self.bo),
            "ml_info": self.ml_info,
            "lgA": str(self.lgA),
            "lineup_mode": str(self.lineup_mode),
            "pl_enabled": bool(self.pl_enabled),
            "pl_players_path": self.pl_players_path,
            "lineupA_sel": self.lineupA_sel,
            "lineupB_sel": self.lineupB_sel,
            "as_of_date": self.as_of_date,
            "ml_engine": str(self.ml_engine),
            "use_lineup": bool(self.use_lineup),
            "artifact_path": self.artifact_path,
            "pl_w": self.pl_w,
            "playoffs_opt": self.playoffs_opt,
            "teamA": str(self.teamA),
            "teamB": str(self.teamB),
            "team_aliases_a": list(self.team_aliases_a),
            "team_aliases_b": list(self.team_aliases_b),
            "teamA_key": str(self.teamA_key),
            "teamB_key": str(self.teamB_key),
            "team_games": self.team_games,
            "csv_path": self.csv_path,
            "filters": self.filters,
            "diff_used": self.diff_used,
            "delta_players": self.delta_players,
            "delta_players_eff": self.delta_players_eff,
            "trace_p_map_team": self.trace_p_map_team,
            "trace_p_map_players": self.trace_p_map_players,
            "trace_p_map_fused": self.trace_p_map_fused,
            "trace_n_team_eff": self.trace_n_team_eff,
            "trace_n_players_eff": self.trace_n_players_eff,
            "trace_fusion_info": self.trace_fusion_info,
            "trace_uncertainty_info": self.trace_uncertainty_info,
            "league_transfer_features": self.league_transfer_features,
            "p_pipe_prev": self.p_pipe_prev,
            "session_state": self.session_state,
            "trace_pipeline_steps": self.trace_pipeline_steps,
            "league_is_excluded_fn": _require_callable(self.callbacks, "league_is_excluded_fn", required=True),
            "apply_ml_uncertainty_adjust_fn": _require_callable(
                self.callbacks, "apply_ml_uncertainty_adjust_fn", required=True
            ),
            "prob_win_series_fn": _require_callable(self.callbacks, "prob_win_series_fn", required=True),
            "load_players_artifact_cached_fn": _require_callable(
                self.callbacks, "load_players_artifact_cached_fn", required=True
            ),
            "file_sig_fn": _require_callable(self.callbacks, "file_sig_fn", required=True),
            "fit_players_lane_for_league_fn": _require_callable(
                self.callbacks, "fit_players_lane_for_league_fn", required=True
            ),
            "fit_players_league_weight_for_league_fn": _require_callable(
                self.callbacks, "fit_players_league_weight_for_league_fn", required=True
            ),
            "compute_delta_and_confidence_fn": _require_callable(
                self.callbacks, "compute_delta_and_confidence_fn", required=True
            ),
            "delta_players_effect_fn": _require_callable(
                self.callbacks, "delta_players_effect_fn", required=True
            ),
            "transfer_signal_from_details_fn": _require_callable(
                self.callbacks, "transfer_signal_from_details_fn", required=True
            ),
            "lineup_continuity_from_details_fn": _require_callable(
                self.callbacks, "lineup_continuity_from_details_fn", required=True
            ),
            "calibrate_p_from_artifact_fn": _require_callable(
                self.callbacks, "calibrate_p_from_artifact_fn", required=True
            ),
            "fuse_probs_by_precision_fn": _require_callable(
                self.callbacks, "fuse_probs_by_precision_fn", required=True
            ),
            "score_team_fn": _require_callable(self.callbacks, "score_team_fn", required=True),
            "p_from_diff_points_fn": _require_callable(
                self.callbacks, "p_from_diff_points_fn", required=True
            ),
            "elo_season_players_p_map_fn": _require_callable(
                self.callbacks, "elo_season_players_p_map_fn", required=True
            ),
            "apply_confidence_governor_fn": _require_callable(
                self.callbacks, "apply_confidence_governor_fn", required=False
            ),
            "format_num_fn": self.format_num_fn,
            "format_pct_fn": self.format_pct_fn,
            "caption_fn": self.caption_fn,
            "warning_fn": self.warning_fn,
            "match_league": str(self.match_league or ""),
            "postfusion_grupo_coef": self.postfusion_grupo_coef,
        }


@dataclass(slots=True)
class MlPreLiveInput:
    p_map_used: Any
    p_series_used: Any
    bo: int
    season_label: str
    teamA: str
    teamB: str
    ml_engine: str
    year_opt: Any
    split_opt: Any
    playoffs_opt: Any
    league_mode: str
    fixed_league: Any
    as_of_date: Any
    artifact_path: str
    lineup_mode: str
    lineupA_sel: Dict[str, Any]
    lineupB_sel: Dict[str, Any]
    diff_used: Any
    delta_players: Any
    trace_n_team_eff: Any
    trace_n_players_eff: Any
    trace_patch_guard_info: Dict[str, Any]
    trace_style_b_lite_live: Dict[str, Any]
    ml_conf_score_snapshot: Any
    lg_a_shadow: str
    lg_b_shadow: str
    csv_path: Any
    fit_params_path: Any
    filters: Any
    dfA_used: Any
    dfB_used: Any
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None
    trace_pipeline_steps: List[Dict[str, Any]]
    callbacks: Dict[str, Callable[..., Any]] = field(default_factory=dict)
    caption_fn: Callable[[str], None] | None = None

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "p_map_used": self.p_map_used,
            "p_series_used": self.p_series_used,
            "bo": int(self.bo),
            "season_label": str(self.season_label),
            "teamA": str(self.teamA),
            "teamB": str(self.teamB),
            "ml_engine": str(self.ml_engine),
            "year_opt": self.year_opt,
            "split_opt": self.split_opt,
            "playoffs_opt": self.playoffs_opt,
            "league_mode": str(self.league_mode),
            "fixed_league": self.fixed_league,
            "as_of_date": self.as_of_date,
            "artifact_path": str(self.artifact_path),
            "lineup_mode": str(self.lineup_mode),
            "lineupA_sel": self.lineupA_sel,
            "lineupB_sel": self.lineupB_sel,
            "diff_used": self.diff_used,
            "delta_players": self.delta_players,
            "trace_n_team_eff": self.trace_n_team_eff,
            "trace_n_players_eff": self.trace_n_players_eff,
            "trace_patch_guard_info": self.trace_patch_guard_info,
            "trace_style_b_lite_live": self.trace_style_b_lite_live,
            "ml_conf_score_snapshot": self.ml_conf_score_snapshot,
            "lg_a_shadow": str(self.lg_a_shadow or ""),
            "lg_b_shadow": str(self.lg_b_shadow or ""),
            "csv_path": self.csv_path,
            "fit_params_path": self.fit_params_path,
            "filters": self.filters,
            "dfA_used": self.dfA_used,
            "dfB_used": self.dfB_used,
            "session_state": self.session_state,
            "trace_pipeline_steps": self.trace_pipeline_steps,
            "season_form_blend_fn": _require_callable(self.callbacks, "season_form_blend_fn", required=True),
            "current_season_override_fn": _require_callable(
                self.callbacks, "current_season_override_fn", required=True
            ),
            "hybrid_momentum_blend_fn": _require_callable(
                self.callbacks, "hybrid_momentum_blend_fn", required=True
            ),
            "clip_prob_fn": _require_callable(self.callbacks, "clip_prob_fn", required=True),
            "ml_consistency_signature_fn": _require_callable(
                self.callbacks, "ml_consistency_signature_fn", required=True
            ),
            "feature_flag_enabled_fn": _require_callable(
                self.callbacks, "feature_flag_enabled_fn", required=True
            ),
            "apply_series_bo_dplite_live_fn": _require_callable(
                self.callbacks, "apply_series_bo_dplite_live_fn", required=True
            ),
            "enforce_series_economic_coherence_fn": _require_callable(
                self.callbacks, "enforce_series_economic_coherence_fn", required=True
            ),
            "series_gamma_for_league_fn": _require_callable(
                self.callbacks, "series_gamma_for_league_fn", required=True
            ),
            "apply_series_gamma_fn": _require_callable(
                self.callbacks, "apply_series_gamma_fn", required=True
            ),
            "patch_meta_guard_payload_fn": _require_callable(
                self.callbacks, "patch_meta_guard_payload_fn", required=True
            ),
            "patch_changed_leagues_set_fn": _require_callable(
                self.callbacks, "patch_changed_leagues_set_fn", required=True
            ),
            "file_sig_fn": _require_callable(self.callbacks, "file_sig_fn", required=True),
            "prob_win_series_fn": _require_callable(self.callbacks, "prob_win_series_fn", required=True),
            "fit_form_for_league_fn": _require_callable(
                self.callbacks, "fit_form_for_league_fn", required=False
            ),
            "apply_confidence_governor_fn": _require_callable(
                self.callbacks, "apply_confidence_governor_fn", required=False
            ),
            "caption_fn": self.caption_fn,
        }
