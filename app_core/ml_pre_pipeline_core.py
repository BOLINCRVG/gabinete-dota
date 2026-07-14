"""Orchestrate full ML pre-live pipeline (engine + pre-live stages)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from app_core.ml_output_models import MlPrePipelineOutput
from app_core.ml_pre_engine_core import run_ml_pre_engine_layers
from app_core.ml_pre_live_pipeline_core import run_ml_pre_live_pipeline


def _as_kwargs(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    to_kwargs = getattr(payload, "to_kwargs", None)
    if callable(to_kwargs):
        out = to_kwargs()
        return dict(out) if isinstance(out, Mapping) else {}
    raise TypeError(f"unsupported pre-pipeline input type: {type(payload).__name__}")


def run_ml_pre_pipeline(
    *,
    pre_engine: Any | None = None,
    pre_live: Any | None = None,
    pre_engine_kwargs: Mapping[str, Any] | None = None,
    pre_live_kwargs: Mapping[str, Any] | None = None,
    run_pre_engine_fn: Callable[..., Dict[str, Any]] | None = None,
    run_pre_live_fn: Callable[..., Dict[str, Any]] | None = None,
    return_model: bool = False,
) -> Dict[str, Any] | MlPrePipelineOutput:
    run_engine = run_pre_engine_fn if callable(run_pre_engine_fn) else run_ml_pre_engine_layers
    run_live = run_pre_live_fn if callable(run_pre_live_fn) else run_ml_pre_live_pipeline

    engine_in = _as_kwargs(pre_engine if pre_engine is not None else pre_engine_kwargs)
    live_in = _as_kwargs(pre_live if pre_live is not None else pre_live_kwargs)

    engine_out = run_engine(**engine_in)
    engine_out = engine_out if isinstance(engine_out, dict) else {}

    live_in["p_map_used"] = float(engine_out.get("p_map_used", live_in.get("p_map_used", float("nan"))))
    live_in["p_series_used"] = float(engine_out.get("p_series_used", live_in.get("p_series_used", float("nan"))))
    live_in["diff_used"] = float(engine_out.get("diff_used", live_in.get("diff_used", float("nan"))))
    live_in["delta_players"] = float(engine_out.get("delta_players", live_in.get("delta_players", 0.0)))
    live_in["trace_n_team_eff"] = float(
        engine_out.get("trace_n_team_eff", live_in.get("trace_n_team_eff", float("nan")))
    )
    live_in["trace_n_players_eff"] = float(
        engine_out.get("trace_n_players_eff", live_in.get("trace_n_players_eff", float("nan")))
    )
    live_in["lineup_mode"] = str(engine_out.get("lineup_mode", live_in.get("lineup_mode", "")))

    live_out = run_live(**live_in)
    live_out = live_out if isinstance(live_out, dict) else {}

    out = {
        "p_map_used": float(live_out.get("p_map_used", engine_out.get("p_map_used", float("nan")))),
        "p_series_used": float(live_out.get("p_series_used", engine_out.get("p_series_used", float("nan")))),
        "diff_used": float(live_out.get("diff_used", engine_out.get("diff_used", float("nan")))),
        "delta_players": float(live_out.get("delta_players", engine_out.get("delta_players", 0.0))),
        "delta_players_eff": float(engine_out.get("delta_players_eff", float("nan"))),
        "trace_p_map_team": float(engine_out.get("trace_p_map_team", float("nan"))),
        "trace_p_map_players": float(engine_out.get("trace_p_map_players", float("nan"))),
        "trace_p_map_fused": float(engine_out.get("trace_p_map_fused", float("nan"))),
        "trace_n_team_eff": float(engine_out.get("trace_n_team_eff", float("nan"))),
        "trace_n_players_eff": float(engine_out.get("trace_n_players_eff", float("nan"))),
        "trace_fusion_info": (
            dict(engine_out.get("trace_fusion_info", {}))
            if isinstance(engine_out.get("trace_fusion_info"), dict)
            else {}
        ),
        "trace_uncertainty_info": (
            dict(engine_out.get("trace_uncertainty_info", {}))
            if isinstance(engine_out.get("trace_uncertainty_info"), dict)
            else {}
        ),
        "_delta_mode": str(engine_out.get("_delta_mode", "")),
        "_delta_cap": float(engine_out.get("_delta_cap", float("nan"))),
        "_delta_slope": float(engine_out.get("_delta_slope", float("nan"))),
        "lineup_mode": str(engine_out.get("lineup_mode", "")),
        "match_league": str(engine_out.get("match_league", "")),
        "_elo_info": engine_out.get("_elo_info"),
        "_p_pipe_prev": float(engine_out.get("_p_pipe_prev", float("nan"))),
        "trace_form_info": (
            dict(live_out.get("trace_form_info", {}))
            if isinstance(live_out.get("trace_form_info"), dict)
            else {}
        ),
        "trace_hybrid_info": (
            dict(live_out.get("trace_hybrid_info", {}))
            if isinstance(live_out.get("trace_hybrid_info"), dict)
            else {}
        ),
        "trace_conf_guard_info": (
            dict(live_out.get("trace_conf_guard_info", {}))
            if isinstance(live_out.get("trace_conf_guard_info"), dict)
            else {}
        ),
        "trace_patch_guard_info": (
            dict(live_out.get("trace_patch_guard_info", {}))
            if isinstance(live_out.get("trace_patch_guard_info"), dict)
            else {}
        ),
        "trace_series_gamma_info": (
            dict(live_out.get("trace_series_gamma_info", {}))
            if isinstance(live_out.get("trace_series_gamma_info"), dict)
            else {}
        ),
        "trace_series_coherence_info": (
            dict(live_out.get("trace_series_coherence_info", {}))
            if isinstance(live_out.get("trace_series_coherence_info"), dict)
            else {}
        ),
        "_guard_triggered": bool(live_out.get("_guard_triggered", False)),
        "_guard_dpp": float(live_out.get("_guard_dpp", float("nan"))),
        "_ff_ml_series_bo_dplite_live": bool(live_out.get("_ff_ml_series_bo_dplite_live", False)),
        "trace_series_bo_dplite_live": (
            dict(live_out.get("trace_series_bo_dplite_live", {}))
            if isinstance(live_out.get("trace_series_bo_dplite_live"), dict)
            else {}
        ),
        "_lineup_mode_sig": str(live_out.get("_lineup_mode_sig", "")),
        "_lineupA_sig": (
            dict(live_out.get("_lineupA_sig", {}))
            if isinstance(live_out.get("_lineupA_sig"), dict)
            else {}
        ),
        "_lineupB_sig": (
            dict(live_out.get("_lineupB_sig", {}))
            if isinstance(live_out.get("_lineupB_sig"), dict)
            else {}
        ),
        "trace_pipeline_steps": (
            list(live_out.get("trace_pipeline_steps", []))
            if isinstance(live_out.get("trace_pipeline_steps"), list)
            else (
                list(engine_out.get("trace_pipeline_steps", []))
                if isinstance(engine_out.get("trace_pipeline_steps"), list)
                else []
            )
        ),
        "pre_engine_out": engine_out,
        "pre_live_out": live_out,
    }
    if bool(return_model):
        return MlPrePipelineOutput.from_mapping(out)
    return out
