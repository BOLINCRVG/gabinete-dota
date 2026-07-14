"""Typed output/context models for ML pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from app_core.prob_math_core import safe_float as _f


def _b(v: Any, default: bool = False) -> bool:
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _s(v: Any, default: str = "") -> str:
    try:
        return str(v if v is not None else default)
    except Exception:
        return str(default)


def _d(v: Any) -> Dict[str, Any]:
    return dict(v) if isinstance(v, Mapping) else {}


def _l(v: Any) -> List[Any]:
    return list(v) if isinstance(v, list) else []


@dataclass(slots=True)
class MlPrePipelineOutput:
    p_map_used: float = float("nan")
    p_series_used: float = float("nan")
    diff_used: float = float("nan")
    delta_players: float = 0.0
    delta_players_eff: float = 0.0
    trace_p_map_team: float = float("nan")
    trace_p_map_players: float = float("nan")
    trace_p_map_fused: float = float("nan")
    trace_n_team_eff: float = float("nan")
    trace_n_players_eff: float = float("nan")
    trace_fusion_info: Dict[str, Any] = field(default_factory=dict)
    trace_uncertainty_info: Dict[str, Any] = field(default_factory=dict)
    _delta_mode: str = ""
    _delta_cap: float = float("nan")
    _delta_slope: float = float("nan")
    lineup_mode: str = ""
    match_league: str = ""
    _elo_info: Any = None
    _p_pipe_prev: float = float("nan")
    trace_form_info: Dict[str, Any] = field(default_factory=dict)
    trace_hybrid_info: Dict[str, Any] = field(default_factory=dict)
    trace_conf_guard_info: Dict[str, Any] = field(default_factory=dict)
    trace_patch_guard_info: Dict[str, Any] = field(default_factory=dict)
    trace_series_gamma_info: Dict[str, Any] = field(default_factory=dict)
    trace_series_coherence_info: Dict[str, Any] = field(default_factory=dict)
    _guard_triggered: bool = False
    _guard_dpp: float = float("nan")
    _ff_ml_series_bo_dplite_live: bool = False
    trace_series_bo_dplite_live: Dict[str, Any] = field(default_factory=dict)
    _lineup_mode_sig: str = ""
    _lineupA_sig: Dict[str, Any] = field(default_factory=dict)
    _lineupB_sig: Dict[str, Any] = field(default_factory=dict)
    trace_pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)
    pre_engine_out: Dict[str, Any] = field(default_factory=dict)
    pre_live_out: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MlPrePipelineOutput":
        src = data if isinstance(data, Mapping) else {}
        known = {
            "p_map_used",
            "p_series_used",
            "diff_used",
            "delta_players",
            "delta_players_eff",
            "trace_p_map_team",
            "trace_p_map_players",
            "trace_p_map_fused",
            "trace_n_team_eff",
            "trace_n_players_eff",
            "trace_fusion_info",
            "trace_uncertainty_info",
            "_delta_mode",
            "_delta_cap",
            "_delta_slope",
            "lineup_mode",
            "match_league",
            "_elo_info",
            "_p_pipe_prev",
            "trace_form_info",
            "trace_hybrid_info",
            "trace_conf_guard_info",
            "trace_patch_guard_info",
            "trace_series_gamma_info",
            "trace_series_coherence_info",
            "_guard_triggered",
            "_guard_dpp",
            "_ff_ml_series_bo_dplite_live",
            "trace_series_bo_dplite_live",
            "_lineup_mode_sig",
            "_lineupA_sig",
            "_lineupB_sig",
            "trace_pipeline_steps",
            "pre_engine_out",
            "pre_live_out",
        }
        extras = {str(k): v for k, v in src.items() if str(k) not in known}
        return cls(
            p_map_used=_f(src.get("p_map_used")),
            p_series_used=_f(src.get("p_series_used")),
            diff_used=_f(src.get("diff_used")),
            delta_players=_f(src.get("delta_players"), 0.0),
            delta_players_eff=_f(src.get("delta_players_eff"), 0.0),
            trace_p_map_team=_f(src.get("trace_p_map_team")),
            trace_p_map_players=_f(src.get("trace_p_map_players")),
            trace_p_map_fused=_f(src.get("trace_p_map_fused")),
            trace_n_team_eff=_f(src.get("trace_n_team_eff")),
            trace_n_players_eff=_f(src.get("trace_n_players_eff")),
            trace_fusion_info=_d(src.get("trace_fusion_info")),
            trace_uncertainty_info=_d(src.get("trace_uncertainty_info")),
            _delta_mode=_s(src.get("_delta_mode"), ""),
            _delta_cap=_f(src.get("_delta_cap")),
            _delta_slope=_f(src.get("_delta_slope")),
            lineup_mode=_s(src.get("lineup_mode"), ""),
            match_league=_s(src.get("match_league"), ""),
            _elo_info=src.get("_elo_info"),
            _p_pipe_prev=_f(src.get("_p_pipe_prev")),
            trace_form_info=_d(src.get("trace_form_info")),
            trace_hybrid_info=_d(src.get("trace_hybrid_info")),
            trace_conf_guard_info=_d(src.get("trace_conf_guard_info")),
            trace_patch_guard_info=_d(src.get("trace_patch_guard_info")),
            trace_series_gamma_info=_d(src.get("trace_series_gamma_info")),
            trace_series_coherence_info=_d(src.get("trace_series_coherence_info")),
            _guard_triggered=_b(src.get("_guard_triggered"), False),
            _guard_dpp=_f(src.get("_guard_dpp")),
            _ff_ml_series_bo_dplite_live=_b(src.get("_ff_ml_series_bo_dplite_live"), False),
            trace_series_bo_dplite_live=_d(src.get("trace_series_bo_dplite_live")),
            _lineup_mode_sig=_s(src.get("_lineup_mode_sig"), ""),
            _lineupA_sig=_d(src.get("_lineupA_sig")),
            _lineupB_sig=_d(src.get("_lineupB_sig")),
            trace_pipeline_steps=[
                dict(x) for x in _l(src.get("trace_pipeline_steps")) if isinstance(x, Mapping)
            ],
            pre_engine_out=_d(src.get("pre_engine_out")),
            pre_live_out=_d(src.get("pre_live_out")),
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "p_map_used": float(self.p_map_used),
            "p_series_used": float(self.p_series_used),
            "diff_used": float(self.diff_used),
            "delta_players": float(self.delta_players),
            "delta_players_eff": float(self.delta_players_eff),
            "trace_p_map_team": float(self.trace_p_map_team),
            "trace_p_map_players": float(self.trace_p_map_players),
            "trace_p_map_fused": float(self.trace_p_map_fused),
            "trace_n_team_eff": float(self.trace_n_team_eff),
            "trace_n_players_eff": float(self.trace_n_players_eff),
            "trace_fusion_info": dict(self.trace_fusion_info),
            "trace_uncertainty_info": dict(self.trace_uncertainty_info),
            "_delta_mode": str(self._delta_mode),
            "_delta_cap": float(self._delta_cap),
            "_delta_slope": float(self._delta_slope),
            "lineup_mode": str(self.lineup_mode),
            "match_league": str(self.match_league),
            "_elo_info": self._elo_info,
            "_p_pipe_prev": float(self._p_pipe_prev),
            "trace_form_info": dict(self.trace_form_info),
            "trace_hybrid_info": dict(self.trace_hybrid_info),
            "trace_conf_guard_info": dict(self.trace_conf_guard_info),
            "trace_patch_guard_info": dict(self.trace_patch_guard_info),
            "trace_series_gamma_info": dict(self.trace_series_gamma_info),
            "trace_series_coherence_info": dict(self.trace_series_coherence_info),
            "_guard_triggered": bool(self._guard_triggered),
            "_guard_dpp": float(self._guard_dpp),
            "_ff_ml_series_bo_dplite_live": bool(self._ff_ml_series_bo_dplite_live),
            "trace_series_bo_dplite_live": dict(self.trace_series_bo_dplite_live),
            "_lineup_mode_sig": str(self._lineup_mode_sig),
            "_lineupA_sig": dict(self._lineupA_sig),
            "_lineupB_sig": dict(self._lineupB_sig),
            "trace_pipeline_steps": [dict(x) for x in self.trace_pipeline_steps],
            "pre_engine_out": dict(self.pre_engine_out),
            "pre_live_out": dict(self.pre_live_out),
        }
        out.update(self.extras)
        return out


@dataclass(slots=True)
class MlLiveOutput:
    p_map_used: float = float("nan")
    p_series_used: float = float("nan")
    _ff_ml_style_b_lite_live: bool = False
    _ff_ml_league_rank_sos_live: bool = False
    _ff_ml_roster_continuity_live: bool = False
    _ff_ml_hier_calibration_live: bool = False
    _ff_ml_meta_c_lite_live: bool = False
    _ff_ml_layer_uncertainty_live: bool = False
    trace_style_b_lite_live: Dict[str, Any] = field(default_factory=dict)
    trace_league_rank_sos_live: Dict[str, Any] = field(default_factory=dict)
    trace_roster_continuity_live: Dict[str, Any] = field(default_factory=dict)
    trace_hier_calibration_live: Dict[str, Any] = field(default_factory=dict)
    trace_meta_c_lite_live: Dict[str, Any] = field(default_factory=dict)
    trace_layer_uncertainty_live: Dict[str, Any] = field(default_factory=dict)
    trace_pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MlLiveOutput":
        src = data if isinstance(data, Mapping) else {}
        known = {
            "p_map_used",
            "p_series_used",
            "_ff_ml_style_b_lite_live",
            "_ff_ml_league_rank_sos_live",
            "_ff_ml_roster_continuity_live",
            "_ff_ml_hier_calibration_live",
            "_ff_ml_meta_c_lite_live",
            "_ff_ml_layer_uncertainty_live",
            "trace_style_b_lite_live",
            "trace_league_rank_sos_live",
            "trace_roster_continuity_live",
            "trace_hier_calibration_live",
            "trace_meta_c_lite_live",
            "trace_layer_uncertainty_live",
            "trace_pipeline_steps",
        }
        extras = {str(k): v for k, v in src.items() if str(k) not in known}
        return cls(
            p_map_used=_f(src.get("p_map_used")),
            p_series_used=_f(src.get("p_series_used")),
            _ff_ml_style_b_lite_live=_b(src.get("_ff_ml_style_b_lite_live"), False),
            _ff_ml_league_rank_sos_live=_b(src.get("_ff_ml_league_rank_sos_live"), False),
            _ff_ml_roster_continuity_live=_b(src.get("_ff_ml_roster_continuity_live"), False),
            _ff_ml_hier_calibration_live=_b(src.get("_ff_ml_hier_calibration_live"), False),
            _ff_ml_meta_c_lite_live=_b(src.get("_ff_ml_meta_c_lite_live"), False),
            _ff_ml_layer_uncertainty_live=_b(src.get("_ff_ml_layer_uncertainty_live"), False),
            trace_style_b_lite_live=_d(src.get("trace_style_b_lite_live")),
            trace_league_rank_sos_live=_d(src.get("trace_league_rank_sos_live")),
            trace_roster_continuity_live=_d(src.get("trace_roster_continuity_live")),
            trace_hier_calibration_live=_d(src.get("trace_hier_calibration_live")),
            trace_meta_c_lite_live=_d(src.get("trace_meta_c_lite_live")),
            trace_layer_uncertainty_live=_d(src.get("trace_layer_uncertainty_live")),
            trace_pipeline_steps=[
                dict(x) for x in _l(src.get("trace_pipeline_steps")) if isinstance(x, Mapping)
            ],
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "p_map_used": float(self.p_map_used),
            "p_series_used": float(self.p_series_used),
            "_ff_ml_style_b_lite_live": bool(self._ff_ml_style_b_lite_live),
            "_ff_ml_league_rank_sos_live": bool(self._ff_ml_league_rank_sos_live),
            "_ff_ml_roster_continuity_live": bool(self._ff_ml_roster_continuity_live),
            "_ff_ml_hier_calibration_live": bool(self._ff_ml_hier_calibration_live),
            "_ff_ml_meta_c_lite_live": bool(self._ff_ml_meta_c_lite_live),
            "_ff_ml_layer_uncertainty_live": bool(self._ff_ml_layer_uncertainty_live),
            "trace_style_b_lite_live": dict(self.trace_style_b_lite_live),
            "trace_league_rank_sos_live": dict(self.trace_league_rank_sos_live),
            "trace_roster_continuity_live": dict(self.trace_roster_continuity_live),
            "trace_hier_calibration_live": dict(self.trace_hier_calibration_live),
            "trace_meta_c_lite_live": dict(self.trace_meta_c_lite_live),
            "trace_layer_uncertainty_live": dict(self.trace_layer_uncertainty_live),
            "trace_pipeline_steps": [dict(x) for x in self.trace_pipeline_steps],
        }
        out.update(self.extras)
        return out


@dataclass(slots=True)
class MlTraceContext:
    generated_at: str = ""
    season_label: str = ""
    teamA: str = ""
    teamB: str = ""
    bo: int = 1
    ml_engine: str = ""
    p_map_used: float = float("nan")
    p_series_used: float = float("nan")
    trace_pipeline_order: List[str] = field(default_factory=list)
    trace_pipeline_steps: List[Dict[str, Any]] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "MlTraceContext":
        src = data if isinstance(data, Mapping) else {}
        known = {
            "generated_at",
            "season_label",
            "teamA",
            "teamB",
            "bo",
            "ml_engine",
            "p_map_used",
            "p_series_used",
            "trace_pipeline_order",
            "trace_pipeline_steps",
        }
        extras = {str(k): v for k, v in src.items() if str(k) not in known}
        return cls(
            generated_at=_s(src.get("generated_at"), ""),
            season_label=_s(src.get("season_label"), ""),
            teamA=_s(src.get("teamA"), ""),
            teamB=_s(src.get("teamB"), ""),
            bo=int(src.get("bo", 1) or 1),
            ml_engine=_s(src.get("ml_engine"), ""),
            p_map_used=_f(src.get("p_map_used")),
            p_series_used=_f(src.get("p_series_used")),
            trace_pipeline_order=[str(x) for x in _l(src.get("trace_pipeline_order"))],
            trace_pipeline_steps=[
                dict(x) for x in _l(src.get("trace_pipeline_steps")) if isinstance(x, Mapping)
            ],
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "generated_at": str(self.generated_at),
            "season_label": str(self.season_label),
            "teamA": str(self.teamA),
            "teamB": str(self.teamB),
            "bo": int(self.bo),
            "ml_engine": str(self.ml_engine),
            "p_map_used": float(self.p_map_used),
            "p_series_used": float(self.p_series_used),
            "trace_pipeline_order": [str(x) for x in self.trace_pipeline_order],
            "trace_pipeline_steps": [dict(x) for x in self.trace_pipeline_steps],
        }
        out.update(self.extras)
        return out
