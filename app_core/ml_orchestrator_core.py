"""Orchestrate post-pre-pipeline ML stages outside plays_app."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

from app_core.confidence_governor_core import read_governor_budget_summary
from app_core.ml_live_pipeline_core import run_ml_live_layers
from app_core.ml_marginal_guard_core import run_marginal_trust_guard
from app_core.ml_output_models import MlLiveOutput
from app_core.prob_math_core import safe_float as _f
from app_core.series_bo_core import enforce_series_economic_coherence
from app_core.state_live_override_core import apply_shadow_state_live_override


def _ff_enabled(
    fn: Callable[..., Any] | None,
    key: str,
    default: bool,
) -> bool:
    if not callable(fn):
        return bool(default)
    try:
        return bool(fn(str(key), default=bool(default)))
    except TypeError:
        try:
            return bool(fn(str(key), bool(default)))
        except Exception:
            return bool(default)
    except Exception:
        return bool(default)


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


def run_ml_post_layers(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    trace_shadow_state_alite: Dict[str, Any] | None,
    trace_shadow_state_areal: Dict[str, Any] | None,
    feature_flag_enabled_fn: Callable[..., Any] | None,
    prob_win_series_fn: Callable[[float, int], float],
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    live_kwargs: Mapping[str, Any],
    marginal_kwargs: Mapping[str, Any],
    confidence_baseline_path: Path | None = None,
    load_confidence_baseline_cached_fn: Callable[[Any], Any] | None = None,
    file_sig_fn: Callable[[str], Any] | None = None,
    lg_a_hint: str = "",
    caption_fn: Callable[[str], None] | None = None,
    run_live_layers_fn: Callable[..., Any] | None = None,
    run_marginal_trust_guard_fn: Callable[..., Dict[str, Any]] | None = None,
    apply_confidence_governor_fn: Callable[..., Dict[str, Any]] | None = None,
    enforce_series_economic_coherence_fn: Callable[..., Dict[str, Any]] | None = None,
    series_coherence_post_max_dev_pp: float | None = None,
) -> Dict[str, Any]:
    ss = session_state if isinstance(session_state, Mapping) else {}
    ssw = session_state if isinstance(session_state, MutableMapping) else None
    steps = trace_pipeline_steps if isinstance(trace_pipeline_steps, list) else []
    p_map = _f(p_map_used, float("nan"))
    p_series = _f(p_series_used, float("nan"))
    bo_i = int(bo)
    gov_enabled = bool(ss.get("ml_confidence_governor_enabled", True))
    gov_budget_pp = float(
        ss.get(
            "_ml_confidence_governor_budget_runtime_pp",
            ss.get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        )
        or 8.0
    )
    gov_live_enabled = bool(ss.get("ml_confidence_governor_live_enabled", True))
    gov_apply_fn = (
        apply_confidence_governor_fn
        if callable(apply_confidence_governor_fn) and bool(gov_live_enabled)
        else None
    )

    ff_state_areal = _ff_enabled(feature_flag_enabled_fn, "ff_ml_state_areal_live", False)
    ff_state_alite = _ff_enabled(feature_flag_enabled_fn, "ff_ml_state_alite_live", True) and (not bool(ff_state_areal))

    # State-override stages share at most 62.5 % of the total budget so that
    # the marginal_trust_guard always has at least 37.5 % (≈3 pp on the 8 pp
    # default) available regardless of how aggressively the overrides shrink.
    _state_override_cap_pp = float(gov_budget_pp) * 0.625

    state_alite_out = apply_shadow_state_live_override(
        p_map_used=p_map,
        p_series_used=p_series,
        bo=bo_i,
        enabled=bool(ff_state_alite),
        shadow_trace=(trace_shadow_state_alite if isinstance(trace_shadow_state_alite, dict) else {}),
        prob_win_series_fn=prob_win_series_fn,
        stage_name="state_alite_live_override",
        source_name="shadow_state_alite",
        promoted_reason="promoted_from_shadow_state_alite",
        not_applied_reason="shadow_state_alite_not_applied",
        session_state=ssw,
        include_qwl_trace=False,
        apply_confidence_governor_fn=gov_apply_fn,
        governor_enabled=bool(gov_enabled and gov_live_enabled),
        governor_budget_pp=float(gov_budget_pp),
        governor_stage_cap_pp=_state_override_cap_pp,
    )
    p_map = _f(state_alite_out.get("p_map_used"), p_map)
    p_series = _f(state_alite_out.get("p_series_used"), p_series)
    trace_state_alite_live = (
        dict(state_alite_out.get("trace", {}))
        if isinstance(state_alite_out.get("trace"), dict)
        else {}
    )
    state_alite_step = state_alite_out.get("pipeline_step")
    if isinstance(state_alite_step, dict):
        steps.append(dict(state_alite_step))

    state_areal_out = apply_shadow_state_live_override(
        p_map_used=p_map,
        p_series_used=p_series,
        bo=bo_i,
        enabled=bool(ff_state_areal),
        shadow_trace=(trace_shadow_state_areal if isinstance(trace_shadow_state_areal, dict) else {}),
        prob_win_series_fn=prob_win_series_fn,
        stage_name="state_areal_live_override",
        source_name="shadow_state_areal",
        promoted_reason="promoted_from_shadow_state_areal",
        not_applied_reason="shadow_state_areal_not_applied",
        session_state=ssw,
        include_qwl_trace=True,
        apply_confidence_governor_fn=gov_apply_fn,
        governor_enabled=bool(gov_enabled and gov_live_enabled),
        governor_budget_pp=float(gov_budget_pp),
        governor_stage_cap_pp=_state_override_cap_pp,
    )
    p_map = _f(state_areal_out.get("p_map_used"), p_map)
    p_series = _f(state_areal_out.get("p_series_used"), p_series)
    trace_state_areal_live = (
        dict(state_areal_out.get("trace", {}))
        if isinstance(state_areal_out.get("trace"), dict)
        else {}
    )
    state_areal_step = state_areal_out.get("pipeline_step")
    if isinstance(state_areal_step, dict):
        steps.append(dict(state_areal_step))

    run_live = run_live_layers_fn if callable(run_live_layers_fn) else run_ml_live_layers
    live_in = dict(live_kwargs or {})
    live_in["p_map_used"] = float(p_map)
    live_in["p_series_used"] = float(p_series)
    live_in["bo"] = int(bo_i)
    live_in["trace_shadow_state_alite"] = (
        dict(trace_shadow_state_alite)
        if isinstance(trace_shadow_state_alite, dict)
        else {}
    )
    live_in["trace_shadow_state_areal"] = (
        dict(trace_shadow_state_areal)
        if isinstance(trace_shadow_state_areal, dict)
        else {}
    )
    live_in["trace_state_alite_live"] = dict(trace_state_alite_live)
    live_in["trace_state_areal_live"] = dict(trace_state_areal_live)
    live_in["feature_flag_enabled_fn"] = feature_flag_enabled_fn
    live_in["prob_win_series_fn"] = prob_win_series_fn
    live_in["session_state"] = session_state
    live_in["trace_pipeline_steps"] = steps
    live_in["return_model"] = True
    live_raw = run_live(**live_in)
    live_out = live_raw if isinstance(live_raw, MlLiveOutput) else MlLiveOutput.from_mapping(
        live_raw if isinstance(live_raw, Mapping) else {}
    )

    p_map = float(live_out.p_map_used)
    p_series = float(live_out.p_series_used)

    lg_mt = str(lg_a_hint or "").strip()
    if not lg_mt:
        try:
            lg_mt = str((((marginal_kwargs or {}).get("ml_info") or {}).get("blue") or {}).get("league") or "").strip()
        except Exception:
            lg_mt = ""

    cb_live_mt: Dict[str, Any] = {}
    try:
        if (
            isinstance(confidence_baseline_path, Path)
            and confidence_baseline_path.exists()
            and callable(load_confidence_baseline_cached_fn)
            and callable(file_sig_fn)
        ):
            cb = load_confidence_baseline_cached_fn(file_sig_fn(str(confidence_baseline_path)))
            cb_live_mt = cb if isinstance(cb, dict) else {}
    except Exception:
        cb_live_mt = {}

    run_marginal = (
        run_marginal_trust_guard_fn
        if callable(run_marginal_trust_guard_fn)
        else run_marginal_trust_guard
    )
    mt_in = dict(marginal_kwargs or {})
    mt_in["p_map_used"] = float(p_map)
    mt_in["p_series_used"] = float(p_series)
    mt_in["bo"] = int(bo_i)
    mt_in["league_hint"] = str(lg_mt or "")
    mt_in["confidence_baseline"] = (cb_live_mt if isinstance(cb_live_mt, dict) else {})
    mt_in["session_state"] = session_state
    mt_in["prob_win_series_fn"] = prob_win_series_fn
    mt_out = run_marginal(**mt_in)
    mt_out = mt_out if isinstance(mt_out, dict) else {}
    p_map = _f(mt_out.get("p_map_used"), p_map)
    p_series = _f(mt_out.get("p_series_used"), p_series)
    trace_marginal_trust_info = (
        dict(mt_out.get("trace_marginal_trust_info", {}))
        if isinstance(mt_out.get("trace_marginal_trust_info"), dict)
        else {}
    )
    cap_applied = str(mt_out.get("caption_applied", "") or "")
    cap_not_applied = str(mt_out.get("caption_not_applied", "") or "")
    if callable(caption_fn):
        try:
            if cap_applied:
                caption_fn(cap_applied)
            elif cap_not_applied:
                caption_fn(cap_not_applied)
        except Exception:
            pass
    mt_step = mt_out.get("pipeline_step")
    if isinstance(mt_step, dict):
        steps.append(dict(mt_step))

    trace_series_coherence_post_info: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "bo": int(bo_i),
        "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
        "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
        "p_series_after": None,
    }
    try:
        coh_fn = (
            enforce_series_economic_coherence_fn
            if callable(enforce_series_economic_coherence_fn)
            else enforce_series_economic_coherence
        )
        pm = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        ps_before = float(p_series) if math.isfinite(float(p_series)) else float("nan")
        if int(bo_i) in (3, 5) and math.isfinite(pm) and math.isfinite(ps_before):
            coh = coh_fn(
                p_map=float(pm),
                p_series=float(ps_before),
                bo=int(bo_i),
                max_dev_pp=(
                    float(series_coherence_post_max_dev_pp)
                    if series_coherence_post_max_dev_pp is not None
                    else None
                ),
            )
            if isinstance(coh, dict):
                ps_after = _f(coh.get("p_series_after"), float("nan"))
                if bool(coh.get("applied", False)) and math.isfinite(float(ps_after)):
                    p_series = float(ps_after)
                trace_series_coherence_post_info = dict(coh)
                if bool(coh.get("applied", False)):
                    steps.append(
                        {
                            "stage": "series_economic_coherence_post",
                            "before": (float(ps_before) if math.isfinite(float(ps_before)) else None),
                            "after": (float(p_series) if math.isfinite(float(p_series)) else None),
                            "applied": True,
                            "enabled": True,
                            "source": str(coh.get("reason", "") or "post_layers"),
                        }
                    )
    except Exception as exc:
        trace_series_coherence_post_info = {
            "enabled": True,
            "applied": False,
            "reason": f"error:{type(exc).__name__}",
            "error": str(exc),
            "bo": int(bo_i),
            "p_map": (float(p_map) if math.isfinite(float(p_map)) else None),
            "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
            "p_series_after": None,
        }

    order = _pipeline_order_from_steps(steps)
    gov_budget_summary = read_governor_budget_summary(session_state)

    return {
        "p_map_used": float(p_map),
        "p_series_used": float(p_series),
        "_ff_ml_state_areal_live": bool(ff_state_areal),
        "_ff_ml_state_alite_live": bool(ff_state_alite),
        "trace_state_alite_live": trace_state_alite_live,
        "trace_state_areal_live": trace_state_areal_live,
        "_ff_ml_style_b_lite_live": bool(live_out._ff_ml_style_b_lite_live),
        "_ff_ml_league_rank_sos_live": bool(live_out._ff_ml_league_rank_sos_live),
        "_ff_ml_roster_continuity_live": bool(live_out._ff_ml_roster_continuity_live),
        "_ff_ml_hier_calibration_live": bool(live_out._ff_ml_hier_calibration_live),
        "_ff_ml_meta_c_lite_live": bool(live_out._ff_ml_meta_c_lite_live),
        "_ff_ml_layer_uncertainty_live": bool(live_out._ff_ml_layer_uncertainty_live),
        "trace_style_b_lite_live": dict(live_out.trace_style_b_lite_live),
        "trace_league_rank_sos_live": dict(live_out.trace_league_rank_sos_live),
        "trace_roster_continuity_live": dict(live_out.trace_roster_continuity_live),
        "trace_hier_calibration_live": dict(live_out.trace_hier_calibration_live),
        "trace_meta_c_lite_live": dict(live_out.trace_meta_c_lite_live),
        "trace_layer_uncertainty_live": dict(live_out.trace_layer_uncertainty_live),
        "trace_marginal_trust_info": trace_marginal_trust_info,
        "trace_series_coherence_post_info": trace_series_coherence_post_info,
        "trace_pipeline_steps": list(steps),
        "trace_pipeline_order": list(order),
        "caption_applied": cap_applied,
        "caption_not_applied": cap_not_applied,
        "live_output_model": live_out,
        "marginal_out": mt_out,
        "league_hint_used": str(lg_mt or ""),
        "confidence_baseline_used": cb_live_mt,
        "trace_governor_budget_final": gov_budget_summary,
    }
