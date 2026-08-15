"""ML live-layer feature-trace append helpers extracted from plays_app."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Mapping


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def _r6(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return None


def _dedupe(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def append_ml_live_feature_traces(
    *,
    append_feature_trace_fn: Callable[..., Any],
    ff_ml_style_b_lite_live: bool,
    trace_style_b_lite_live: Mapping[str, Any] | None,
    ff_ml_league_rank_sos_live: bool,
    trace_league_rank_sos_live: Mapping[str, Any] | None,
    ff_ml_roster_continuity_live: bool,
    trace_roster_continuity_live: Mapping[str, Any] | None,
    ff_ml_hier_calibration_live: bool,
    trace_hier_calibration_live: Mapping[str, Any] | None,
    ff_ml_series_bo_dplite_live: bool,
    trace_series_bo_dplite_live: Mapping[str, Any] | None,
    ff_ml_layer_uncertainty_live: bool,
    trace_layer_uncertainty_live: Mapping[str, Any] | None,
    ff_ml_meta_c_lite_live: bool,
    trace_meta_c_lite_live: Mapping[str, Any] | None,
    season_label: Any,
    teamA: str,
    teamB: str,
    bo: Any,
    ml_engine: Any,
    input_signature_hash: Any,
) -> Dict[str, bool]:
    emitted = {
        "ml_style_matchup_b_lite_live_v1": False,
        "ml_league_rank_sos_live_v1": False,
        "ml_roster_continuity_live_v1": False,
        "ml_hier_calibration_live_v1": False,
        "ml_series_bo_dplite_live_v1": False,
        "ml_layer_uncertainty_live_v1": False,
        "ml_meta_blend_c_lite_live_v1": False,
    }

    matchup_base = {
        "season": str(season_label),
        "teamA": str(teamA),
        "teamB": str(teamB),
    }
    engine = str(ml_engine)
    sig = str(input_signature_hash)

    def _matchup() -> Dict[str, Any]:
        out = dict(matchup_base)
        out["bo"] = int(bo)
        return out

    try:
        if bool(ff_ml_style_b_lite_live):
            sb = _as_dict(trace_style_b_lite_live)
            append_feature_trace_fn(
                feature="ml_style_matchup_b_lite_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(sb.get("applied", False)),
                    "reason": str(sb.get("reason", "") or ""),
                    "method": str(sb.get("method", "") or ""),
                    "fallback_from_model": bool(sb.get("fallback_from_model", False)),
                    "fallback_from_gate": bool(sb.get("fallback_from_gate", False)),
                    "fallback_reason": str(sb.get("fallback_reason", "") or ""),
                    "b_pipeline_path": str(sb.get("b_pipeline_path", "") or ""),
                    "b_model_gate": _as_dict(sb.get("b_model_gate")),
                    "b_model_attempt": _as_dict(sb.get("b_model_attempt")),
                    "p_before": sb.get("p_before") if sb.get("p_before") is not None else sb.get("p_map_before"),
                    "p_after": sb.get("p_after") if sb.get("p_after") is not None else sb.get("p_map_after"),
                    "diff_pp": sb.get("diff_pp"),
                    "weights": _as_dict(sb.get("weights")),
                    "inputs": _as_dict(sb.get("inputs")),
                    "style": _as_dict(sb.get("style")),
                    "model_meta": _as_dict(sb.get("model_meta")),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "p_before": _r6(
                            sb.get("p_before") if sb.get("p_before") is not None else sb.get("p_map_before")
                        ),
                        "p_after": _r6(
                            sb.get("p_after") if sb.get("p_after") is not None else sb.get("p_map_after")
                        ),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_style_matchup_b_lite_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_league_rank_sos_live):
            rk = _as_dict(trace_league_rank_sos_live)
            append_feature_trace_fn(
                feature="ml_league_rank_sos_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(rk.get("applied", False)),
                    "method": str(rk.get("method", "") or ""),
                    "reason": str(rk.get("reason", "") or ""),
                    "p_before": rk.get("p_before") if rk.get("p_before") is not None else rk.get("p_map_before"),
                    "p_after": rk.get("p_after") if rk.get("p_after") is not None else rk.get("p_map_after"),
                    "diff_pp": rk.get("diff_pp"),
                    "p_rank": rk.get("p_rank"),
                    "weight": rk.get("weight"),
                    "n_team_a": rk.get("n_team_a"),
                    "n_team_b": rk.get("n_team_b"),
                    "n_teams_league": rk.get("n_teams_league"),
                    "n_games_league": rk.get("n_games_league"),
                    "h2h_maps": rk.get("h2h_maps"),
                    "h2h_a_wins": rk.get("h2h_a_wins"),
                    "h2h_b_wins": rk.get("h2h_b_wins"),
                    "h2h_n_eff": rk.get("h2h_n_eff"),
                    "h2h_p_laplace": rk.get("h2h_p_laplace"),
                    "h2h_p_quality": rk.get("h2h_p_quality"),
                    "h2h_quality_weight": rk.get("h2h_quality_weight"),
                    "h2h_quality_edge": rk.get("h2h_quality_edge"),
                    "h2h_quality_surplus": rk.get("h2h_quality_surplus"),
                    "h2h_quality_conversion_edge": rk.get("h2h_quality_conversion_edge"),
                    "h2h_quality_coverage": rk.get("h2h_quality_coverage"),
                    "h2h_quality_freshness": rk.get("h2h_quality_freshness"),
                    "h2h_quality_data_reliability": rk.get("h2h_quality_data_reliability"),
                    "team_a_row": _as_dict(rk.get("team_a_row")),
                    "team_b_row": _as_dict(rk.get("team_b_row")),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "p_before": _r6(
                            rk.get("p_before") if rk.get("p_before") is not None else rk.get("p_map_before")
                        ),
                        "p_after": _r6(
                            rk.get("p_after") if rk.get("p_after") is not None else rk.get("p_map_after")
                        ),
                        "p_rank": _r6(rk.get("p_rank")),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_league_rank_sos_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_roster_continuity_live):
            rc = _as_dict(trace_roster_continuity_live)
            append_feature_trace_fn(
                feature="ml_roster_continuity_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(rc.get("applied", False)),
                    "method": str(rc.get("method", "") or ""),
                    "reason": str(rc.get("reason", "") or ""),
                    "p_before": rc.get("p_before") if rc.get("p_before") is not None else rc.get("p_map_before"),
                    "p_after": rc.get("p_after") if rc.get("p_after") is not None else rc.get("p_map_after"),
                    "diff_pp": rc.get("diff_pp"),
                    "weight": rc.get("weight"),
                    "target_p": rc.get("target_p"),
                    "lineup_continuity": rc.get("lineup_continuity"),
                    "freshness_avg": rc.get("freshness_avg"),
                    "coverage_eff": rc.get("coverage_eff"),
                    "roster_changed": rc.get("roster_changed"),
                    "season_phase": rc.get("season_phase"),
                    "evidence": rc.get("evidence"),
                    "roster_pressure": rc.get("roster_pressure"),
                    "target_source": rc.get("target_source"),
                    "signal_source": rc.get("signal_source"),
                    "fallback_used": rc.get("fallback_used"),
                    "source_parts": _as_list(rc.get("source_parts")),
                    "roster_changed_source": rc.get("roster_changed_source"),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "p_before": _r6(
                            rc.get("p_before") if rc.get("p_before") is not None else rc.get("p_map_before")
                        ),
                        "p_after": _r6(
                            rc.get("p_after") if rc.get("p_after") is not None else rc.get("p_map_after")
                        ),
                        "target_p": _r6(rc.get("target_p")),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_roster_continuity_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_hier_calibration_live):
            hc = _as_dict(trace_hier_calibration_live)
            append_feature_trace_fn(
                feature="ml_hier_calibration_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(hc.get("applied", False)),
                    "method": str(hc.get("method", "") or ""),
                    "reason": str(hc.get("reason", "") or ""),
                    "scope_used": str(hc.get("scope_used", "") or ""),
                    "p_before": hc.get("p_before") if hc.get("p_before") is not None else hc.get("p_map_before"),
                    "p_after": hc.get("p_after") if hc.get("p_after") is not None else hc.get("p_map_after"),
                    "diff_pp": hc.get("diff_pp"),
                    "weight": hc.get("weight"),
                    "bias_raw": hc.get("bias_raw"),
                    "bias_adj": hc.get("bias_adj"),
                    "league_resolution_source": hc.get("league_resolution_source"),
                    "league_resolution_parts": _as_list(hc.get("league_resolution_parts")),
                    "inputs": _as_dict(hc.get("inputs")),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "scope": str(hc.get("scope_used", "") or ""),
                        "p_before": _r6(
                            hc.get("p_before") if hc.get("p_before") is not None else hc.get("p_map_before")
                        ),
                        "p_after": _r6(
                            hc.get("p_after") if hc.get("p_after") is not None else hc.get("p_map_after")
                        ),
                        "bias_adj": _r6(hc.get("bias_adj")),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_hier_calibration_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_series_bo_dplite_live):
            sbod = _as_dict(trace_series_bo_dplite_live)
            append_feature_trace_fn(
                feature="ml_series_bo_dplite_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(sbod.get("applied", False)),
                    "method": str(sbod.get("method", "") or ""),
                    "reason": str(sbod.get("reason", "") or ""),
                    "bo": sbod.get("bo"),
                    "same_league": sbod.get("same_league"),
                    "interregional": sbod.get("interregional"),
                    "playoffs": sbod.get("playoffs"),
                    "p_map": sbod.get("p_map"),
                    "p_before": (
                        sbod.get("p_before")
                        if sbod.get("p_before") is not None
                        else sbod.get("p_series_before")
                    ),
                    "p_after": (
                        sbod.get("p_after")
                        if sbod.get("p_after") is not None
                        else sbod.get("p_series_after")
                    ),
                    "diff_pp": sbod.get("diff_pp"),
                    "weight": sbod.get("weight"),
                    "cap_pp": sbod.get("cap_pp"),
                    "bias_raw": sbod.get("bias_raw"),
                    "bias_adj": sbod.get("bias_adj"),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "bo": int(bo),
                        "p_before": _r6(
                            sbod.get("p_before")
                            if sbod.get("p_before") is not None
                            else sbod.get("p_series_before")
                        ),
                        "p_after": _r6(
                            sbod.get("p_after")
                            if sbod.get("p_after") is not None
                            else sbod.get("p_series_after")
                        ),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_series_bo_dplite_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_layer_uncertainty_live):
            lu = _as_dict(trace_layer_uncertainty_live)
            append_feature_trace_fn(
                feature="ml_layer_uncertainty_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(lu.get("applied", False)),
                    "reason": str(lu.get("reason", "") or ""),
                    "method": str(lu.get("method", "") or ""),
                    "p_before": lu.get("p_before"),
                    "p_after": lu.get("p_after"),
                    "diff_pp": lu.get("diff_pp"),
                    "p_series_before": lu.get("p_series_before"),
                    "p_series_after": lu.get("p_series_after"),
                    "diff_pp_series": lu.get("diff_pp_series"),
                    "evidence_score": lu.get("evidence_score"),
                    "uncertainty_score": lu.get("uncertainty_score"),
                    "confidence_score": lu.get("confidence_score"),
                    "shrink_map": lu.get("shrink_map"),
                    "shrink_series": lu.get("shrink_series"),
                    "cap_pp_map": lu.get("cap_pp_map"),
                    "cap_pp_series": lu.get("cap_pp_series"),
                    "components": _as_dict(lu.get("components")),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "pm_before": _r6(lu.get("p_before")),
                        "pm_after": _r6(lu.get("p_after")),
                        "applied": bool(lu.get("applied", False)),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_layer_uncertainty_live_v1"] = True
    except Exception:
        pass

    try:
        if bool(ff_ml_meta_c_lite_live):
            mc = _as_dict(trace_meta_c_lite_live)
            append_feature_trace_fn(
                feature="ml_meta_blend_c_lite_live_v1",
                event="live_apply",
                enabled=True,
                payload={
                    "matchup": _matchup(),
                    "engine": engine,
                    "input_signature_sha256": sig,
                    "applied": bool(mc.get("applied", False)),
                    "method": str(mc.get("method", "") or ""),
                    "reason": str(mc.get("reason", "") or ""),
                    "p_anchor": mc.get("p_anchor"),
                    "p_before": mc.get("p_before") if mc.get("p_before") is not None else mc.get("p_map_before"),
                    "p_after": mc.get("p_after") if mc.get("p_after") is not None else mc.get("p_map_after"),
                    "diff_pp": mc.get("diff_pp"),
                    "weights": _as_dict(mc.get("weights")),
                    "inputs": _as_dict(mc.get("inputs")),
                    "targets": _as_dict(mc.get("targets")),
                },
                dedupe_signature=_dedupe(
                    {
                        "sig": sig,
                        "eng": engine,
                        "p_anchor": _r6(mc.get("p_anchor")),
                        "p_before": _r6(
                            mc.get("p_before") if mc.get("p_before") is not None else mc.get("p_map_before")
                        ),
                        "p_after": _r6(
                            mc.get("p_after") if mc.get("p_after") is not None else mc.get("p_map_after")
                        ),
                    }
                ),
                persist_files=False,
            )
            emitted["ml_meta_blend_c_lite_live_v1"] = True
    except Exception:
        pass

    return emitted
