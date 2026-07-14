from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping


def _default_fit_runtime_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    team_scale = float(session_state.get("_fit_team_scale", 1.0) or 1.0)
    players_scale = float(session_state.get("_fit_players_scale", 1.0) or 1.0)
    return {
        "_fit_team_scale": 1.0,
        "_fit_players_scale": 1.0,
        "_fit_team_scale_used": float(team_scale),
        "_fit_players_scale_used": float(players_scale),
        "_fit_scale_league_used": "",
        "_fit_form_cfg": {},
        "_fit_form_cfg_used": {},
        "_fit_form_cfg_league_used": "",
        "_fit_lane_weights_cfg": {},
        "_fit_players_league_weights_cfg": {},
        "_fit_players_context_cfg": {},
        "_fit_fused_post_cal_cfg": {},
        "_fit_series_gamma_cfg": {"global": {}, "by_league": {}, "by_macro": {}},
        "_fit_series_bo_dplite_cfg": {},
        "_fit_marginal_trust_band_coeffs_cfg": {},
    }


def get_fit_runtime_snapshot(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a dict snapshot of fit-runtime parameters (team/player scales, lane weights, etc.)."""
    defaults = _default_fit_runtime_state(session_state)
    out = dict(defaults)
    for key in defaults:
        value = session_state.get(key, defaults[key])
        if isinstance(defaults[key], dict):
            out[key] = dict(value or {}) if isinstance(value, Mapping) else dict(defaults[key])
        else:
            out[key] = value
    return out


def get_ml_trace_snapshot(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return all current trace payloads (last ML trace, markets trace, players trace, feature trace)."""
    return {
        "_ml_trace_last": dict(session_state.get("_ml_trace_last", {})) if isinstance(session_state.get("_ml_trace_last"), Mapping) else {},
        "_ml_trace_markets": list(session_state.get("_ml_trace_markets", []) or []),
        "_ml_trace_players": list(session_state.get("_ml_trace_players", []) or []),
        "_feature_trace_events": list(session_state.get("_feature_trace_events", []) or []),
        "_ml_trace_combined": dict(session_state.get("_ml_trace_combined", {})) if isinstance(session_state.get("_ml_trace_combined"), Mapping) else {},
        "_p_ml_scope": session_state.get("_p_ml_scope"),
    }


def set_latest_ml_trace(session_state: MutableMapping[str, Any], trace_payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Persist the ML trace payload to session state for later inspection and download."""
    payload = dict(trace_payload or {})
    session_state["_ml_trace_last"] = payload
    return payload


def get_pre_delivery_report(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the most recent pre-delivery checks report from session state."""
    value = session_state.get("_pre_delivery_report", {})
    return dict(value) if isinstance(value, Mapping) else {}


def set_pre_delivery_report(session_state: MutableMapping[str, Any], report: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Store the pre-delivery checks report payload in session state."""
    payload = dict(report or {})
    session_state["_pre_delivery_report"] = payload
    return payload


def get_analysis_gate_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the last analysis and consistency-analysis signature values for change detection."""
    return {
        "analysis_sig_last": session_state.get("_analysis_sig_last"),
        "consistency_analysis_sig_last": session_state.get("_consistency_analysis_sig_last"),
    }


def set_analysis_signature(session_state: MutableMapping[str, Any], signature: Any) -> int:
    """Persist the current analysis signature hash to detect when re-analysis is needed."""
    value = int(signature)
    session_state["_analysis_sig_last"] = value
    return value


def set_consistency_analysis_signature(session_state: MutableMapping[str, Any], signature: Any) -> int:
    """Persist the consistency analysis signature for the Consistência tab change detection."""
    value = int(signature)
    session_state["_consistency_analysis_sig_last"] = value
    return value


def get_ml_scope(session_state: Mapping[str, Any], default: float = float("nan")) -> float:
    """Return the current ML win probability (p_ML) used for ML+Over/Under combo pricing."""
    try:
        value = float(session_state.get("_p_ml_scope", default))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def set_ml_scope(session_state: MutableMapping[str, Any], value: Any) -> float:
    """Persist the ML win probability used for ML+Totals table computation."""
    try:
        scope = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        scope = float("nan")
    session_state["_p_ml_scope"] = scope
    return scope


def get_achievement_path(session_state: Mapping[str, Any]) -> str:
    """Return the user-configured achievement JSON override path."""
    return str(session_state.get("ach_path_ui", "") or "").strip()


def set_achievement_path(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist a custom achievement JSON path to override the default location."""
    text = str(value or "").strip()
    session_state["ach_path_ui"] = text
    return text


def get_pending_apply_param_values(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return parameter values staged for application on the next analysis run."""
    value = session_state.get("_pending_apply_param_values", {})
    return dict(value) if isinstance(value, Mapping) else {}


def set_pending_apply_param_values(session_state: MutableMapping[str, Any], values: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Stage parameter values to be applied on the next analysis cycle."""
    payload = dict(values or {})
    session_state["_pending_apply_param_values"] = payload
    return payload


def consume_pending_apply_param_values(session_state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Pop and return staged parameter values, clearing them from session state."""
    value = session_state.pop("_pending_apply_param_values", {})
    return dict(value) if isinstance(value, Mapping) else {}


def get_fit_scale_state(session_state: Mapping[str, Any]) -> Dict[str, float]:
    """Return current team and player scale factors (base and last-used values)."""
    return {
        "team_scale": float(session_state.get("_fit_team_scale", 1.0) or 1.0),
        "players_scale": float(session_state.get("_fit_players_scale", 1.0) or 1.0),
        "team_scale_used": float(session_state.get("_fit_team_scale_used", session_state.get("_fit_team_scale", 1.0)) or 1.0),
        "players_scale_used": float(session_state.get("_fit_players_scale_used", session_state.get("_fit_players_scale", 1.0)) or 1.0),
    }


def set_fit_base_scales(session_state: MutableMapping[str, Any], *, team_scale: Any, players_scale: Any) -> Dict[str, float]:
    """Persist the user-configured team and players scale factors to session state."""
    team = float(team_scale or 1.0)
    players = float(players_scale or 1.0)
    session_state["_fit_team_scale"] = team
    session_state["_fit_players_scale"] = players
    return {"team_scale": team, "players_scale": players}


def set_fit_used_scales(session_state: MutableMapping[str, Any], *, team_scale: Any, players_scale: Any) -> Dict[str, float]:
    """Record the scale values that were actually used in the last pipeline run."""
    team = float(team_scale or 1.0)
    players = float(players_scale or 1.0)
    session_state["_fit_team_scale_used"] = team
    session_state["_fit_players_scale_used"] = players
    return {"team_scale_used": team, "players_scale_used": players}


def get_feature_trace_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return current feature-trace events and error state for the shadow observability layer."""
    return {
        "events": list(session_state.get("_feature_trace_events", []) or []),
        "last_error": str(session_state.get("_feature_trace_last_error", "") or ""),
        "last_error_at": str(session_state.get("_feature_trace_last_error_at", "") or ""),
        "last_ok_at": str(session_state.get("_feature_trace_last_ok_at", "") or ""),
    }


def get_ml_engine_runtime_settings(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a dict of all ML engine flags (microseg, post-calibration, player context, etc.)."""
    return {
        "ml_engine": str(session_state.get("ml_engine", "mlcore_v2") or "mlcore_v2"),
        "ml_engine_compare": bool(session_state.get("ml_engine_compare", False)),
        "ml_engine_alert_pp": float(session_state.get("ml_engine_alert_pp", 12.0) or 12.0),
        "ml_as_of_date": session_state.get("ml_as_of_date"),
        "ml_microseg_enabled": bool(session_state.get("ml_microseg_enabled", True)),
        "ml_microseg_lookback_games": int(session_state.get("ml_microseg_lookback_games", 1200) or 1200),
        "ml_microseg_shrink_k": float(session_state.get("ml_microseg_shrink_k", 90.0) or 90.0),
        "ml_microseg_min_n": int(session_state.get("ml_microseg_min_n", 30) or 30),
        "ml_microseg_strength": float(session_state.get("ml_microseg_strength", 0.60) or 0.60),
        "ml_microseg_cap_pp": float(session_state.get("ml_microseg_cap_pp", 8.0) or 8.0),
        "ml_microseg_hierarchical": bool(session_state.get("ml_microseg_hierarchical", True)),
        "ml_microseg_hier_decay": float(session_state.get("ml_microseg_hier_decay", 0.80) or 0.80),
        "ml_fused_post_cal_use_fitted": bool(session_state.get("ml_fused_post_cal_use_fitted", True)),
        "wf_apply_ml_correction": bool(session_state.get("wf_apply_ml_correction", False)),
        "wf_ml_corr_strength": float(session_state.get("wf_ml_corr_strength", 0.60) or 0.60),
        "ml_players_context_use_fitted": bool(session_state.get("ml_players_context_use_fitted", True)),
    }


def get_mlcore_cache_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the last ML pipeline cache key and value for cache hit/miss detection."""
    value = session_state.get("_mlcore_cache_val")
    return {
        "key": session_state.get("_mlcore_cache_key"),
        "value": dict(value) if isinstance(value, Mapping) else None,
    }


def set_mlcore_cache_state(session_state: MutableMapping[str, Any], key: Any, value: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Store a new ML pipeline result under the given cache key."""
    session_state["_mlcore_cache_key"] = key
    payload = dict(value or {})
    session_state["_mlcore_cache_val"] = payload
    return payload


def get_ml_microseg_info(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the micro-segmentation calibration info dict from the last pipeline run."""
    value = session_state.get("_ml_microseg_info")
    return dict(value) if isinstance(value, Mapping) else {}


def set_ml_microseg_info(session_state: MutableMapping[str, Any], value: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Cache the microseg calibration result for display in the trace panel."""
    payload = dict(value or {})
    session_state["_ml_microseg_info"] = payload
    return payload


def clear_ml_microseg_info(session_state: MutableMapping[str, Any]) -> None:
    """Clear the cached microseg calibration result from session state."""
    session_state.pop("_ml_microseg_info", None)


def get_ml_fused_post_cal_info(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the fused post-calibration info from the last pipeline run."""
    value = session_state.get("_ml_fused_post_cal_info")
    return dict(value) if isinstance(value, Mapping) else {}


def set_ml_fused_post_cal_info(session_state: MutableMapping[str, Any], value: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Cache the fused post-calibration result for trace display."""
    payload = dict(value or {})
    session_state["_ml_fused_post_cal_info"] = payload
    return payload


def clear_ml_fused_post_cal_info(session_state: MutableMapping[str, Any]) -> None:
    """Clear the fused post-calibration result from session state."""
    session_state.pop("_ml_fused_post_cal_info", None)


def get_wf_ml_corr_info(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the walk-forward ML correction info used for WF probability adjustments."""
    value = session_state.get("wf_ml_corr_info")
    return dict(value) if isinstance(value, Mapping) else {}


def set_wf_ml_corr_info(session_state: MutableMapping[str, Any], value: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Persist the walk-forward ML correction coefficients to session state."""
    payload = dict(value or {})
    session_state["wf_ml_corr_info"] = payload
    return payload


def clear_wf_ml_corr_info(session_state: MutableMapping[str, Any]) -> None:
    """Clear the walk-forward ML correction info from session state."""
    session_state.pop("wf_ml_corr_info", None)


def set_fit_runtime_labels(
    session_state: MutableMapping[str, Any],
    *,
    scale_league_used: Any = None,
    players_context_source_used: Any = None,
) -> Dict[str, str]:
    """Persist display labels indicating which league/source was used by the fit pipeline."""
    out = {
        "scale_league_used": str(scale_league_used or ""),
        "players_context_source_used": str(players_context_source_used or ""),
    }
    if scale_league_used is not None:
        session_state["_fit_scale_league_used"] = out["scale_league_used"]
    if players_context_source_used is not None:
        session_state["_fit_players_context_source_used"] = out["players_context_source_used"]
    return out


def apply_players_context_runtime_override(
    session_state: MutableMapping[str, Any],
    cfg_used: Mapping[str, Any] | None,
    cfg_source: Any,
    *,
    use_fitted: bool,
) -> None:
    """Apply fitted players-context config overrides to session state, or disable if gate rejected."""
    source = str(cfg_source or "")
    if not use_fitted:
        return
    if isinstance(cfg_used, Mapping) and cfg_used:
        try:
            session_state["ml_players_context_enabled"] = bool(cfg_used.get("ml_players_context_enabled", False))
            session_state["ml_players_context_lookback_games"] = int(cfg_used.get("ml_players_context_lookback_games", 24) or 24)
            session_state["ml_players_context_half_life_days"] = float(cfg_used.get("ml_players_context_half_life_days", 45.0) or 45.0)
            session_state["ml_players_context_min_games"] = int(cfg_used.get("ml_players_context_min_games", 6) or 6)
            session_state["ml_players_context_strength"] = float(cfg_used.get("ml_players_context_strength", 0.35) or 0.35)
            session_state["ml_players_context_cap"] = float(cfg_used.get("ml_players_context_cap", 0.22) or 0.22)
            session_state["ml_players_context_min_reliability"] = float(cfg_used.get("ml_players_context_min_reliability", 0.28) or 0.28)
            session_state["ml_players_context_inter_mult"] = float(cfg_used.get("ml_players_context_inter_mult", 0.88) or 0.88)
        except Exception:
            pass
        return
    gate_decision = source.strip().lower().split(":", 1)[-1] if source.startswith("gate:") else ""
    if gate_decision in {"shadow_only", "reject"}:
        session_state["ml_players_context_enabled"] = False


def set_feature_trace_events(session_state: MutableMapping[str, Any], rows: list[dict[str, Any]] | None, *, ok_at: str | None = None) -> list[dict[str, Any]]:
    """Persist feature trace events and clear any pending error flag."""
    events = list(rows or [])
    session_state["_feature_trace_events"] = events
    session_state["_feature_trace_last_error"] = ""
    if ok_at is not None:
        session_state["_feature_trace_last_ok_at"] = str(ok_at)
    return events


def set_feature_trace_error(session_state: MutableMapping[str, Any], message: str, *, error_at: str | None = None) -> str:
    """Record a feature-trace error message to surface in the global error banner."""
    msg = str(message or "").strip()
    session_state["_feature_trace_last_error"] = msg
    if error_at is not None:
        session_state["_feature_trace_last_error_at"] = str(error_at)
    return msg


def clear_feature_trace_error(session_state: MutableMapping[str, Any]) -> None:
    """Clear the feature-trace error message from session state."""
    session_state["_feature_trace_last_error"] = ""


def apply_fit_runtime_state_to_session(
    session_state: MutableMapping[str, Any],
    fit_cfg: Mapping[str, Any] | None,
    *,
    build_fit_runtime_state_fn: Callable[..., Dict[str, Any]],
    sanitize_marginal_band_coeffs_fn: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply a fit config dict to session state via the runtime state builder; returns form config."""
    try:
        runtime_state = build_fit_runtime_state_fn(
            dict(fit_cfg or {}),
            sanitize_marginal_band_coeffs=sanitize_marginal_band_coeffs_fn,
            current_marginal_band_coeffs=session_state.get("ml_marginal_trust_band_coeffs", {}),
        )
        session_state.update(dict(runtime_state or {}))
        return dict(session_state.get("_fit_form_cfg") or {})
    except Exception:
        defaults = _default_fit_runtime_state(session_state)
        for key, value in defaults.items():
            session_state.setdefault(key, value)
        return {}


def build_combined_trace_payload(
    session_state: Mapping[str, Any],
    *,
    build_combined_trace_fn: Callable[..., Dict[str, Any]],
    app_version: str,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Assemble a combined trace payload from all active trace snapshots for download."""
    ts = str(generated_at or datetime.now().isoformat(timespec="seconds"))
    trace_state = get_ml_trace_snapshot(session_state)
    payload = build_combined_trace_fn(
        ml_base_trace=trace_state.get("_ml_trace_last"),
        resumo_markets_trace=trace_state.get("_ml_trace_markets", []),
        players_trace=trace_state.get("_ml_trace_players", []),
        feature_trace=trace_state.get("_feature_trace_events", []),
    )
    payload["meta"] = {
        "app_version": str(app_version or ""),
        "generated_at": ts,
        "season": str(session_state.get("season_label", "")),
        "teams": {
            "teamA": str(session_state.get("teamA", "")),
            "teamB": str(session_state.get("teamB", "")),
            "bo": int(session_state.get("bo", 1) or 1),
        },
        "filters": {
            "year": str(session_state.get("year_opt", "")),
            "split": str(session_state.get("split_opt", "")),
            "playoffs": str(session_state.get("playoffs_opt", "")),
            "map_mode": str(session_state.get("map_mode", "")),
            "league_mode": str(session_state.get("league_mode", "")),
        },
        "calibration": {
            "ml_engine": str(session_state.get("ml_engine", "")),
            "microseg_enabled": bool(session_state.get("ml_microseg_enabled", True)),
            "microseg_strength": float(session_state.get("ml_microseg_strength", 0.60) or 0.60),
            "wf_half_life_days": float(session_state.get("wf_half_life_days", 180.0) or 180.0),
            "series_gamma_on": bool(session_state.get("series_gamma_on", True)),
        },
    }
    return payload


def persist_combined_trace_file(
    session_state: MutableMapping[str, Any],
    *,
    app_root: str | Path,
    build_combined_trace_fn: Callable[..., Dict[str, Any]],
    app_version: str,
    filename: str = "ml_trace_last.json",
) -> Dict[str, Any]:
    """Write the combined trace payload to disk and cache it in session state."""
    payload = build_combined_trace_payload(
        session_state,
        build_combined_trace_fn=build_combined_trace_fn,
        app_version=app_version,
    )
    session_state["_ml_trace_combined"] = payload
    target = Path(str(app_root)).expanduser().resolve() / str(filename)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
