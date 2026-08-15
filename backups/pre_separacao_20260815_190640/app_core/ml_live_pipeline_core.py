"""Orchestrate live ML post-layers (B/C + uncertainty) outside UI code."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional

import pandas as pd

from app_core.ml_output_models import MlLiveOutput
from app_core.hier_calibration_core import apply_hier_calibration_live
from app_core.layer_uncertainty_core import apply_layer_uncertainty_live
from app_core.league_rank_sos_core import apply_league_rank_sos_live
from app_core.live_stage_core import (
    apply_live_stage_result,
    build_live_stage_error_trace,
    build_live_stage_trace,
)
from app_core.meta_blend_core import (
    apply_meta_blend_c_lite,
    apply_meta_blend_c_model,
    should_apply_meta_blend_c_model_gate,
)
from app_core.prob_math_core import safe_float as _f
from app_core.roster_continuity_core import (
    apply_roster_continuity_live,
    build_roster_live_inputs,
)
from app_core.style_matchup_core import apply_style_matchup_b_live
from mlcore.league_context import is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label
from mlcore.lines_engine import build_profile


def _is_playoffs(playoffs_opt: Any) -> bool:
    return str(playoffs_opt).strip().lower() in {"1", "true", "playoffs", "sim", "yes"}


def _is_explicit_interregional_league(v: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(v))


def _load_fit_params_once(
    *,
    fit_params_path: Path,
    load_fitted_params_fn: Optional[Callable[[], Dict[str, Any]]],
) -> Dict[str, Any]:
    if not isinstance(fit_params_path, Path) or (not fit_params_path.exists()):
        return {}
    if not callable(load_fitted_params_fn):
        return {}
    try:
        out = load_fitted_params_fn() or {}
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def run_ml_live_layers(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    playoffs_opt: Any,
    ml_info: Dict[str, Any] | None,
    pA: Any = None,
    pB: Any = None,
    team_games: Any = None,
    filters: Any = None,
    league_mode: str = "",
    map_mode: str = "",
    half_life_days: Any = None,
    max_games: Any = None,
    teamA: str = "",
    teamB: str = "",
    team_aliases_a: list[str] | None = None,
    team_aliases_b: list[str] | None = None,
    year_opt: Any = None,
    split_opt: Any = None,
    trace_n_team_eff: Any = None,
    trace_n_players_eff: Any = None,
    trace_patch_guard_info: Dict[str, Any] | None = None,
    trace_fusion_info: Dict[str, Any] | None = None,
    trace_shadow_state_areal: Dict[str, Any] | None = None,
    trace_shadow_state_alite: Dict[str, Any] | None = None,
    trace_state_areal_live: Dict[str, Any] | None = None,
    trace_state_alite_live: Dict[str, Any] | None = None,
    trace_series_bo_dplite_live: Dict[str, Any] | None = None,
    trace_hybrid_info: Dict[str, Any] | None = None,
    trace_p_map_players: Any = None,
    trace_p_map_team: Any = None,
    fit_live: Dict[str, Any] | None = None,
    conf_baseline: Dict[str, Any] | None = None,
    league_transfer_features: Dict[str, Any] | None = None,
    match_league: Any = None,
    lg_a_shadow: str = "",
    lg_b_shadow: str = "",
    p_map_cal: Any = None,
    guard_triggered: bool = False,
    guard_dpp: Any = None,
    feature_flag_enabled_fn: Callable[[str, bool], bool] | None = None,
    prob_win_series_fn: Callable[[float, int], float] | None = None,
    session_state: MutableMapping[str, Any] | None = None,
    trace_pipeline_steps: List[Dict[str, Any]] | None = None,
    fit_params_path: Path | None = None,
    load_fitted_params_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    apply_confidence_governor_fn: Callable[..., Dict[str, Any]] | None = None,
    return_model: bool = False,
) -> Dict[str, Any] | MlLiveOutput:
    trace_pipeline_steps = trace_pipeline_steps if isinstance(trace_pipeline_steps, list) else []
    state = session_state if isinstance(session_state, MutableMapping) else None
    prob_win_series = prob_win_series_fn if callable(prob_win_series_fn) else (lambda p, _bo: float(p))
    ff_enabled = feature_flag_enabled_fn if callable(feature_flag_enabled_fn) else (lambda _k, d=False: bool(d))

    p_map = _f(p_map_used, float("nan"))
    p_series = _f(p_series_used, float("nan"))
    bo_i = int(bo)
    gov_enabled = bool((state or {}).get("ml_confidence_governor_enabled", True))
    gov_budget_pp = float(
        (state or {}).get(
            "_ml_confidence_governor_budget_runtime_pp",
            (state or {}).get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        )
        or 8.0
    )
    gov_live_enabled = bool((state or {}).get("ml_confidence_governor_live_enabled", True))
    gov_apply_fn = apply_confidence_governor_fn if callable(apply_confidence_governor_fn) and bool(gov_live_enabled) else None

    fit_params = _load_fit_params_once(
        fit_params_path=(fit_params_path if isinstance(fit_params_path, Path) else Path("")),
        load_fitted_params_fn=load_fitted_params_fn,
    )

    ff_style = bool(ff_enabled("ff_ml_style_b_lite_live", True))
    trace_style_b_lite_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_style,
        p_map_used=p_map,
        p_series_used=p_series,
    )
    try:
        if ff_style:
            _lg_a_style = str(((((ml_info or {}).get("blue") or {}).get("league")) or "")).strip()
            _lg_b_style = str(((((ml_info or {}).get("red") or {}).get("league")) or "")).strip()
            _style_cfg_live = (fit_params.get("style_matchup_b_model") or {}) if isinstance(fit_params, dict) else {}

            _common_style_kwargs: Dict[str, Any] = {
                "p_map_in": (float(p_map) if math.isfinite(float(p_map)) else None),
                "pA": pA,
                "pB": pB,
                "league_a": _lg_a_style,
                "league_b": _lg_b_style,
                "patch_status": str(((trace_patch_guard_info or {}).get("status", "") or "")),
                "roster_changed": bool((trace_fusion_info or {}).get("roster_changed", False)),
                "bo": int(bo_i),
                "is_playoffs": _is_playoffs(playoffs_opt),
            }
            try:
                _style_pA_recent = None
                _style_pB_recent = None
                if isinstance(team_games, pd.DataFrame) and (filters is not None):
                    _hl = min(float(half_life_days or 180.0), 45.0)
                    _mg = min(int(max_games), 8) if isinstance(max_games, int) and int(max_games) > 0 else 8
                    _style_pA_recent = build_profile(
                        team_games,
                        team=teamA,
                        filters=filters,
                        league_mode=league_mode,
                        map_mode=map_mode,
                        half_life_days=_hl,
                        max_games=_mg,
                    )
                    _style_pB_recent = build_profile(
                        team_games,
                        team=teamB,
                        filters=filters,
                        league_mode=league_mode,
                        map_mode=map_mode,
                        half_life_days=_hl,
                        max_games=_mg,
                    )
                _common_style_kwargs["pA_recent"] = _style_pA_recent
                _common_style_kwargs["pB_recent"] = _style_pB_recent
            except Exception:
                _common_style_kwargs["pA_recent"] = None
                _common_style_kwargs["pB_recent"] = None

            _style_out = apply_style_matchup_b_live(
                model_cfg=(_style_cfg_live if isinstance(_style_cfg_live, dict) else {}),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else None),
                use_lite_when_gate_blocked=True,
                **_common_style_kwargs,
            )
            if isinstance(_style_out, dict):
                _style_out.setdefault("enabled", True)
                _style_apply = apply_live_stage_result(
                    out=_style_out,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="style_matchup_b_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="reason",
                    source_default="profile_style_matchup",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_style_apply.get("p_map_used", float("nan")))
                p_series = float(_style_apply.get("p_series_used", float("nan")))
                trace_style_b_lite_live = dict(_style_apply.get("out", {}))
    except Exception as exc:
        trace_style_b_lite_live = build_live_stage_error_trace(
            enabled=ff_style,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
        )

    ff_rank = bool(ff_enabled("ff_ml_league_rank_sos_live", False))
    trace_league_rank_sos_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_rank,
        p_map_used=p_map,
        p_series_used=p_series,
    )
    try:
        if ff_rank:
            _league_blue = str(((((ml_info or {}).get("blue") or {}).get("league")) or "")).strip()
            _league_red = str(((((ml_info or {}).get("red") or {}).get("league")) or "")).strip()
            _league_sos = _league_blue if (_league_blue and _league_blue == _league_red) else ""
            _p_sos_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            _rank_sos_out = apply_league_rank_sos_live(
                p_map_in=(_p_sos_before if math.isfinite(_p_sos_before) else None),
                team_games=(team_games if isinstance(team_games, pd.DataFrame) else None),
                team_a=str(teamA),
                team_b=str(teamB),
                team_aliases_a=list(team_aliases_a or []),
                team_aliases_b=list(team_aliases_b or []),
                league=_league_sos,
                year_opt=year_opt,
                split_opt=split_opt,
                playoffs_opt=playoffs_opt,
                bo=int(bo_i),
            )
            if isinstance(_rank_sos_out, dict):
                _rank_sos_out.setdefault("enabled", True)
                _rank_apply = apply_live_stage_result(
                    out=_rank_sos_out,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="league_rank_sos_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="method",
                    source_default="league_rank_sos",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_rank_apply.get("p_map_used", float("nan")))
                p_series = float(_rank_apply.get("p_series_used", float("nan")))
                trace_league_rank_sos_live = dict(_rank_apply.get("out", {}))
    except Exception as exc:
        trace_league_rank_sos_live = build_live_stage_error_trace(
            enabled=ff_rank,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
        )

    ff_roster = bool(ff_enabled("ff_ml_roster_continuity_live", False))
    trace_roster_continuity_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_roster,
        p_map_used=p_map,
        p_series_used=p_series,
    )
    try:
        if ff_roster:
            _p_ro_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            _roster_n_team_eff = float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else None
            _roster_n_players_eff = float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else None
            _roster_inputs = build_roster_live_inputs(
                fusion_info=(trace_fusion_info if isinstance(trace_fusion_info, dict) else {}),
                shadow_state_areal=(trace_shadow_state_areal if isinstance(trace_shadow_state_areal, dict) else {}),
                n_team_eff=_roster_n_team_eff,
                n_players_eff=_roster_n_players_eff,
                playoffs=playoffs_opt,
                roster_gate=((trace_fusion_info or {}).get("roster_gate") if isinstance(trace_fusion_info, dict) else None),
            )
            _rc_out = apply_roster_continuity_live(
                p_map_in=(_p_ro_before if math.isfinite(_p_ro_before) else None),
                p_players=(float(trace_p_map_players) if math.isfinite(_f(trace_p_map_players)) else None),
                p_team=(float(trace_p_map_team) if math.isfinite(_f(trace_p_map_team)) else None),
                lineup_continuity=(_roster_inputs.get("lineup_continuity") if isinstance(_roster_inputs, dict) else None),
                freshness_avg=(_roster_inputs.get("freshness_avg") if isinstance(_roster_inputs, dict) else None),
                coverage_eff=(_roster_inputs.get("coverage_eff") if isinstance(_roster_inputs, dict) else None),
                roster_changed=bool((_roster_inputs or {}).get("roster_changed", False)) if isinstance(_roster_inputs, dict) else False,
                season_phase=str((_roster_inputs or {}).get("season_phase", "") or "") if isinstance(_roster_inputs, dict) else "",
                bo=int(bo_i),
                n_team_eff=_roster_n_team_eff,
                n_players_eff=_roster_n_players_eff,
                transfer_signal=(_roster_inputs.get("transfer_signal") if isinstance(_roster_inputs, dict) else None),
                cfg=(((fit_live or {}).get("roster_continuity_v2") or {}) if isinstance(fit_live, dict) else {}),
            )
            if isinstance(_rc_out, dict):
                _rc_out.setdefault("enabled", True)
                _rc_out.setdefault("signal_source", str((_roster_inputs or {}).get("signal_source", "")) if isinstance(_roster_inputs, dict) else "")
                _rc_out.setdefault("fallback_used", bool((_roster_inputs or {}).get("fallback_used", False)) if isinstance(_roster_inputs, dict) else False)
                _rc_out.setdefault("source_parts", list((_roster_inputs or {}).get("source_parts", [])) if isinstance(_roster_inputs, dict) else [])
                _rc_out.setdefault(
                    "roster_changed_source",
                    str((_roster_inputs or {}).get("roster_changed_source", "")) if isinstance(_roster_inputs, dict) else "",
                )
                _roster_apply = apply_live_stage_result(
                    out=_rc_out,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="roster_continuity_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="method",
                    source_default="roster_continuity",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_roster_apply.get("p_map_used", float("nan")))
                p_series = float(_roster_apply.get("p_series_used", float("nan")))
                trace_roster_continuity_live = dict(_roster_apply.get("out", {}))
    except Exception as exc:
        trace_roster_continuity_live = build_live_stage_error_trace(
            enabled=ff_roster,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
        )

    ff_hier = bool(ff_enabled("ff_ml_hier_calibration_live", False))
    trace_hier_calibration_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_hier,
        p_map_used=p_map,
        p_series_used=p_series,
    )
    try:
        if ff_hier:
            _p_hc_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            _bl_payload = (conf_baseline if isinstance(conf_baseline, dict) else {})
            _lg_a_hc = str(lg_a_shadow or "").strip().upper()
            _lg_b_hc = str(lg_b_shadow or "").strip().upper()
            _lg_res_parts: List[str] = []
            if _lg_a_hc:
                _lg_res_parts.append("shadow.blue")
            if _lg_b_hc:
                _lg_res_parts.append("shadow.red")
            if (not _lg_a_hc) or (not _lg_b_hc):
                _ml_blue = str((((ml_info or {}).get("blue") or {}).get("league") or "").strip()).upper()
                _ml_red = str((((ml_info or {}).get("red") or {}).get("league") or "").strip()).upper()
                if _ml_blue:
                    _lg_a_hc = _lg_a_hc or _ml_blue
                    _lg_res_parts.append("ml_info.blue")
                if _ml_red:
                    _lg_b_hc = _lg_b_hc or _ml_red
                    _lg_res_parts.append("ml_info.red")
            if (not _lg_a_hc) or (not _lg_b_hc):
                _match_lg_hc = str(match_league or "").strip().upper()
                if _match_lg_hc:
                    _lg_a_hc = _lg_a_hc or _match_lg_hc
                    _lg_b_hc = _lg_b_hc or _match_lg_hc
                    _lg_res_parts.append("match_league")
            if (not _lg_a_hc) or (not _lg_b_hc):
                _lg_hint_hc = str((trace_patch_guard_info or {}).get("league_hint", "") or "").strip().upper()
                if _lg_hint_hc:
                    _lg_a_hc = _lg_a_hc or _lg_hint_hc
                    _lg_b_hc = _lg_b_hc or _lg_hint_hc
                    _lg_res_parts.append("patch_guard.league_hint")
            if _lg_a_hc and (not _lg_b_hc):
                _lg_b_hc = _lg_a_hc
                _lg_res_parts.append("paired_fill.red")
            elif _lg_b_hc and (not _lg_a_hc):
                _lg_a_hc = _lg_b_hc
                _lg_res_parts.append("paired_fill.blue")
            _explicit_inter_hc = bool(_is_explicit_interregional_league(_lg_a_hc) or _is_explicit_interregional_league(_lg_b_hc))
            _same_league_hc = bool(_lg_a_hc) and bool(_lg_b_hc) and (_lg_a_hc == _lg_b_hc) and (not _explicit_inter_hc)
            _interregional_hc = (not _same_league_hc) or bool(_explicit_inter_hc)
            _hc_out = apply_hier_calibration_live(
                p_map_in=(_p_hc_before if math.isfinite(_p_hc_before) else None),
                league=(_lg_a_hc if (bool(_same_league_hc) and (not bool(_interregional_hc))) else "INTER"),
                league_a=_lg_a_hc,
                league_b=_lg_b_hc,
                team_a=str(teamA),
                team_b=str(teamB),
                team_aliases_a=list(team_aliases_a or []),
                team_aliases_b=list(team_aliases_b or []),
                bo=int(bo_i),
                playoffs=playoffs_opt,
                interregional=bool(_interregional_hc),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else None),
                patch_status=str((trace_patch_guard_info or {}).get("status", "") or ""),
                baseline_payload=_bl_payload,
                league_transfer_features=(league_transfer_features if isinstance(league_transfer_features, dict) else {}),
            )
            if isinstance(_hc_out, dict):
                _hc_out.setdefault("enabled", True)
                _hc_out.setdefault("league_resolution_source", ("resolved" if (_lg_a_hc or _lg_b_hc) else "missing"))
                _hc_out.setdefault("league_resolution_parts", list(dict.fromkeys(_lg_res_parts)))
                _hier_apply = apply_live_stage_result(
                    out=_hc_out,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="hier_calibration_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="method",
                    source_default="hier_calibration",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_hier_apply.get("p_map_used", float("nan")))
                p_series = float(_hier_apply.get("p_series_used", float("nan")))
                trace_hier_calibration_live = dict(_hier_apply.get("out", {}))
    except Exception as exc:
        trace_hier_calibration_live = build_live_stage_error_trace(
            enabled=ff_hier,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
        )

    ff_meta = bool(ff_enabled("ff_ml_meta_c_lite_live", False))
    trace_meta_c_lite_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_meta,
        p_map_used=p_map,
        p_series_used=p_series,
    )
    try:
        if ff_meta:
            _anchor_map = None
            if isinstance(trace_state_areal_live, dict) and trace_state_areal_live.get("p_map_before") is not None:
                _anchor_map = trace_state_areal_live.get("p_map_before")
            elif isinstance(trace_state_alite_live, dict) and trace_state_alite_live.get("p_map_before") is not None:
                _anchor_map = trace_state_alite_live.get("p_map_before")
            elif math.isfinite(_f(p_map_cal)):
                _anchor_map = float(p_map_cal)
            _sa_inputs = (
                trace_shadow_state_alite.get("inputs")
                if isinstance(trace_shadow_state_alite, dict) and isinstance(trace_shadow_state_alite.get("inputs"), dict)
                else {}
            )
            _sa_w = (
                trace_shadow_state_alite.get("weights")
                if isinstance(trace_shadow_state_alite, dict) and isinstance(trace_shadow_state_alite.get("weights"), dict)
                else {}
            )
            _meta_cfg_live = (fit_params.get("meta_blend_c_model") or {}) if isinstance(fit_params, dict) else {}
            _common_meta_kwargs = {
                "p_anchor": _anchor_map,
                "p_adjusted": (float(p_map) if math.isfinite(float(p_map)) else None),
                "n_team_eff": (float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else _sa_inputs.get("n_team_eff")),
                "n_players_eff": (float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else _sa_inputs.get("n_players_eff")),
                "recent_evidence": _sa_inputs.get("recent_evidence"),
                "rd_conf": _sa_w.get("rd_conf"),
                "patch_status": str((trace_patch_guard_info or {}).get("status", "") or ""),
                "interregional": bool(_sa_inputs.get("interregional", False)),
                "unknown_macro": bool(_sa_inputs.get("unknown_macro", False)),
                "roster_changed": bool((trace_fusion_info or {}).get("roster_changed", False)),
                "consistency_guard_triggered": bool(guard_triggered),
                "consistency_guard_drift_pp": (float(guard_dpp) if math.isfinite(_f(guard_dpp)) else None),
                "hybrid_agree": (trace_hybrid_info or {}).get("agree_to_model") if isinstance(trace_hybrid_info, dict) else None,
                "bo": int(bo_i),
                "is_playoffs": _is_playoffs(playoffs_opt),
                "rank_sos_applied": bool((trace_league_rank_sos_live or {}).get("applied", False)) if isinstance(trace_league_rank_sos_live, dict) else False,
                "rank_sos_weight": (trace_league_rank_sos_live or {}).get("weight") if isinstance(trace_league_rank_sos_live, dict) else None,
                "rank_sos_diff_pp": (trace_league_rank_sos_live or {}).get("diff_pp") if isinstance(trace_league_rank_sos_live, dict) else None,
                "rank_sos_p": (trace_league_rank_sos_live or {}).get("p_rank") if isinstance(trace_league_rank_sos_live, dict) else None,
                "roster_cont_applied": bool((trace_roster_continuity_live or {}).get("applied", False)) if isinstance(trace_roster_continuity_live, dict) else False,
                "roster_cont_weight": (trace_roster_continuity_live or {}).get("weight") if isinstance(trace_roster_continuity_live, dict) else None,
                "roster_cont_diff_pp": (trace_roster_continuity_live or {}).get("diff_pp") if isinstance(trace_roster_continuity_live, dict) else None,
                "roster_cont_target_p": (trace_roster_continuity_live or {}).get("target_p") if isinstance(trace_roster_continuity_live, dict) else None,
                "league_a": str(_sa_inputs.get("league_a", "") or ""),
                "league_b": str(_sa_inputs.get("league_b", "") or ""),
            }
            _lite_meta_kwargs = {
                k: v
                for k, v in _common_meta_kwargs.items()
                if k not in {"bo", "is_playoffs", "rank_sos_applied", "rank_sos_weight", "rank_sos_diff_pp", "rank_sos_p"}
            }
            _meta_gate = should_apply_meta_blend_c_model_gate(
                bo=_common_meta_kwargs.get("bo"),
                recent_evidence=_common_meta_kwargs.get("recent_evidence"),
                n_team_eff=_common_meta_kwargs.get("n_team_eff"),
                n_players_eff=_common_meta_kwargs.get("n_players_eff"),
                is_playoffs=_common_meta_kwargs.get("is_playoffs"),
                patch_status=str(_common_meta_kwargs.get("patch_status") or ""),
                interregional=bool(_common_meta_kwargs.get("interregional", False)),
            )
            if bool((_meta_gate or {}).get("allow", False)):
                _mc = apply_meta_blend_c_model(
                    model_cfg=(_meta_cfg_live if isinstance(_meta_cfg_live, dict) else {}),
                    **_common_meta_kwargs,
                )
                if isinstance(_mc, dict):
                    _mc["c_model_gate"] = dict(_meta_gate)
                if not (isinstance(_mc, dict) and bool(_mc.get("applied", False))):
                    _mc = apply_meta_blend_c_lite(**_lite_meta_kwargs)
                    if isinstance(_mc, dict):
                        _mc["fallback_from_model"] = True
                        _mc["c_model_gate"] = dict(_meta_gate or {})
            else:
                _mc = {
                    "applied": False,
                    "reason": "c_gate_blocked_no_c_apply",
                    "method": "meta_blend_c_model_gate",
                    "c_model_gate": dict(_meta_gate or {}),
                    "fallback_from_model": False,
                }
            if isinstance(_mc, dict):
                _mc.setdefault("enabled", True)
                _mc["pipeline_source"] = "meta_blend_gate"
                _meta_apply = apply_live_stage_result(
                    out=_mc,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="meta_blend_c_lite_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="pipeline_source",
                    source_default="meta_blend_gate",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_meta_apply.get("p_map_used", float("nan")))
                p_series = float(_meta_apply.get("p_series_used", float("nan")))
                trace_meta_c_lite_live = dict(_meta_apply.get("out", {}))
    except Exception as exc:
        trace_meta_c_lite_live = build_live_stage_error_trace(
            enabled=ff_meta,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
        )

    ff_lu = bool(ff_enabled("ff_ml_layer_uncertainty_live", True))
    trace_layer_uncertainty_live: Dict[str, Any] = build_live_stage_trace(
        enabled=ff_lu,
        p_map_used=p_map,
        p_series_used=p_series,
        map_before_key="p_before",
        map_after_key="p_after",
    )
    trace_layer_uncertainty_live["diff_pp"] = None
    trace_layer_uncertainty_live["diff_pp_series"] = None
    try:
        if ff_lu:
            _p_lu_before = float(p_map) if math.isfinite(float(p_map)) else float("nan")
            _ps_lu_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
            _lu_out = apply_layer_uncertainty_live(
                p_map_in=_p_lu_before,
                p_series_in=_ps_lu_before,
                bo=int(bo_i),
                interregional=bool((trace_shadow_state_alite.get("inputs") or {}).get("interregional", False))
                if isinstance(trace_shadow_state_alite, dict)
                else False,
                playoffs=_is_playoffs(playoffs_opt),
                patch_status=str((trace_patch_guard_info or {}).get("status", "") or ""),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(_f(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(_f(trace_n_players_eff)) else None),
                state_areal_live=(trace_state_areal_live if isinstance(trace_state_areal_live, dict) else {}),
                state_alite_live=(trace_state_alite_live if isinstance(trace_state_alite_live, dict) else {}),
                style_b_live=(trace_style_b_lite_live if isinstance(trace_style_b_lite_live, dict) else {}),
                rank_sos_live=(trace_league_rank_sos_live if isinstance(trace_league_rank_sos_live, dict) else {}),
                roster_continuity_live=(trace_roster_continuity_live if isinstance(trace_roster_continuity_live, dict) else {}),
                hier_calibration_live=(trace_hier_calibration_live if isinstance(trace_hier_calibration_live, dict) else {}),
                series_bo_dplite_live=(trace_series_bo_dplite_live if isinstance(trace_series_bo_dplite_live, dict) else {}),
                meta_c_live=(trace_meta_c_lite_live if isinstance(trace_meta_c_lite_live, dict) else {}),
            )
            if isinstance(_lu_out, dict):
                _layer_apply = apply_live_stage_result(
                    out=_lu_out,
                    p_map_used=p_map,
                    p_series_used=p_series,
                    bo=int(bo_i),
                    stage_name="layer_uncertainty_live",
                    trace_pipeline_steps=trace_pipeline_steps,
                    prob_win_series_fn=prob_win_series,
                    session_state=state,
                    source_key="method",
                    source_default="layer_uncertainty",
                    map_before_key="p_before",
                    map_after_key="p_after",
                    p_after_key="p_after",
                    out_series_key="p_series_after",
                    apply_confidence_governor_fn=gov_apply_fn,
                    governor_enabled=bool(gov_enabled and gov_live_enabled),
                    governor_budget_pp=float(gov_budget_pp),
                )
                p_map = float(_layer_apply.get("p_map_used", float("nan")))
                p_series = float(_layer_apply.get("p_series_used", float("nan")))
                trace_layer_uncertainty_live = dict(_layer_apply.get("out", {}))
    except Exception as exc:
        trace_layer_uncertainty_live = build_live_stage_error_trace(
            enabled=ff_lu,
            error=exc,
            p_map_used=p_map,
            p_series_used=p_series,
            map_before_key="p_before",
            map_after_key="p_after",
        )
        trace_layer_uncertainty_live["diff_pp"] = None
        trace_layer_uncertainty_live["diff_pp_series"] = None

    out = {
        "p_map_used": float(p_map) if math.isfinite(float(p_map)) else float("nan"),
        "p_series_used": float(p_series) if math.isfinite(float(p_series)) else float("nan"),
        "_ff_ml_style_b_lite_live": bool(ff_style),
        "_ff_ml_league_rank_sos_live": bool(ff_rank),
        "_ff_ml_roster_continuity_live": bool(ff_roster),
        "_ff_ml_hier_calibration_live": bool(ff_hier),
        "_ff_ml_meta_c_lite_live": bool(ff_meta),
        "_ff_ml_layer_uncertainty_live": bool(ff_lu),
        "trace_style_b_lite_live": trace_style_b_lite_live if isinstance(trace_style_b_lite_live, dict) else {},
        "trace_league_rank_sos_live": trace_league_rank_sos_live if isinstance(trace_league_rank_sos_live, dict) else {},
        "trace_roster_continuity_live": trace_roster_continuity_live if isinstance(trace_roster_continuity_live, dict) else {},
        "trace_hier_calibration_live": trace_hier_calibration_live if isinstance(trace_hier_calibration_live, dict) else {},
        "trace_meta_c_lite_live": trace_meta_c_lite_live if isinstance(trace_meta_c_lite_live, dict) else {},
        "trace_layer_uncertainty_live": trace_layer_uncertainty_live if isinstance(trace_layer_uncertainty_live, dict) else {},
        "trace_pipeline_steps": trace_pipeline_steps,
    }
    if bool(return_model):
        return MlLiveOutput.from_mapping(out)
    return out
