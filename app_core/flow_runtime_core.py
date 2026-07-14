from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional


_FLOW_LINE_KEYS = [
    "lines_kills",
    "lines_towers",
    "lines_dragons",
    "lines_barons",
    "lines_inhib",
    "lines_time",
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_text",
    "hc_kills_team",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
]

_FLOW_FILTER_KEYS = [
    "totals_mode",
    "ml_mode",
    "map_mode_ui",
    "window_opt",
    "window_short",
    "league_mode",
    "year_opt",
    "split_opt",
    "playoffs_opt",
    "fixed_league",
    "weight_mode",
    "half_life_days",
    "n_sims",
]

_FLOW_CONFIG_KEYS = ["cfg_auto_paths", "cfg_csv_override", "cfg_artifact_override"]

_FLOW_RUNTIME_TRANSIENT_KEYS = [
    "_p_map_used",
    "_p_series_used",
    "_p_ml_scope",
    "_ml_diff_used",
    "_ml_delta_players",
    "_ml_delta_players_eff",
    "_ml_team_games_min",
    "_ml_consistency_prev",
    "_mlcore_cache_key",
    "_mlcore_cache_val",
    "_ml_microseg_info",
    "_ml_fused_post_cal_info",
    "wf_ml_corr_info",
    "_ml_trace_last",
    "_ml_trace_markets",
    "_ml_trace_players",
    "_ml_trace_combined",
    "_pre_delivery_report",
    "_feature_trace_events",
    "_feature_trace_last_error",
    "_feature_trace_last_error_at",
    "_feature_trace_last_ok_at",
]


def reset_flow_state(session_state: MutableMapping[str, Any], *, keep_config: bool = True) -> None:
    """Reset the flow to 'select' stage, clearing line/filter/transient state (optionally keep config)."""
    session_state["flow_stage"] = "select"
    for key in _FLOW_LINE_KEYS:
        session_state.pop(key, None)
    for key in _FLOW_FILTER_KEYS:
        session_state.pop(key, None)
    for key in _FLOW_RUNTIME_TRANSIENT_KEYS:
        session_state.pop(key, None)
    if not keep_config:
        for key in _FLOW_CONFIG_KEYS:
            session_state.pop(key, None)


def set_pending_load(session_state: MutableMapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Queue a payload to be applied on the next render (team selection, line values, etc.)."""
    value = dict(payload or {})
    session_state["pending_load"] = value
    return value


def apply_pending_load(
    session_state: MutableMapping[str, Any],
    *,
    reset_flow_fn: Optional[Callable[..., None]] = None,
) -> bool:
    """Apply a queued pending_load payload; returns True if anything was applied."""
    if "pending_load" not in session_state:
        return False
    payload = session_state.pop("pending_load") or {}
    try:
        if payload.get("_reset_flow") and callable(reset_flow_fn):
            reset_flow_fn(keep_config=True)
    except Exception:
        pass
    flow_line_values = payload.get("_flow_line_values")
    if isinstance(flow_line_values, dict):
        apply_flow_line_values(session_state, flow_line_values)
    for key, value in dict(payload).items():
        if str(key).startswith("_"):
            continue
        session_state[key] = value
    return True


def get_flow_stage(session_state: MutableMapping[str, Any], default: str = "select") -> str:
    """Return the current flow stage ('select', 'results', etc.)."""
    return str(session_state.get("flow_stage", default) or default)


def set_flow_stage(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Set the flow stage and persist to session state."""
    stage = str(value or "select")
    session_state["flow_stage"] = stage
    return stage


def ensure_matchup_selection_state(session_state: MutableMapping[str, Any]) -> tuple[str, str]:
    """Initialize teamA/teamB keys if absent and return their current values."""
    for key in ("teamA", "teamB"):
        if key not in session_state:
            session_state[key] = ""
    return str(session_state.get("teamA") or ""), str(session_state.get("teamB") or "")


def clear_matchup_selection(session_state: MutableMapping[str, Any]) -> None:
    """Clear both teamA and teamB selections from session state."""
    session_state["teamA"] = ""
    session_state["teamB"] = ""


def set_selected_team(session_state: MutableMapping[str, Any], side: str, value: Any) -> str:
    """Set teamA or teamB based on side ('a'/'b'/'1'/'2') and return the stored value."""
    key = "teamA" if str(side).lower() in {"a", "teama", "team_a", "1"} else "teamB"
    team = str(value or "")
    session_state[key] = team
    return team


def get_selected_teams(session_state: MutableMapping[str, Any]) -> tuple[str, str]:
    """Return (teamA, teamB) initializing keys if absent."""
    ensure_matchup_selection_state(session_state)
    return str(session_state.get("teamA") or ""), str(session_state.get("teamB") or "")


def request_pending_reset_matchup(session_state: MutableMapping[str, Any]) -> None:
    """Signal that the matchup selection should be reset on the next render cycle."""
    session_state["pending_reset_matchup"] = True


def consume_pending_reset_matchup(session_state: MutableMapping[str, Any]) -> bool:
    """Pop and return the pending-reset-matchup flag."""
    return bool(session_state.pop("pending_reset_matchup", False))


def get_lines_loaded_key(session_state: MutableMapping[str, Any]) -> str:
    """Return the key of the last set of lines loaded for the current matchup."""
    return str(session_state.get("lines_loaded_key") or "")


def set_lines_loaded_key(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist the lines-loaded key so the UI knows which saved lines are active."""
    key = str(value or "")
    session_state["lines_loaded_key"] = key
    return key


def clear_lines_loaded_key(session_state: MutableMapping[str, Any]) -> None:
    """Clear the lines-loaded key when lines are reset or the matchup changes."""
    session_state.pop("lines_loaded_key", None)


def set_matchup_key(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist the current matchup key (used for lines store lookup)."""
    key = str(value or "")
    session_state["matchup_key"] = key
    return key


def get_last_season_label(session_state: MutableMapping[str, Any]) -> str:
    """Return the season label from the most recently completed analysis."""
    return str(session_state.get("_last_season_label") or "")


def set_last_season_label(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Persist the season label after analysis completes."""
    label = str(value or "")
    session_state["_last_season_label"] = label
    return label


def set_pending_auto_analyze(session_state: MutableMapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Queue a payload to trigger automatic analysis on the next render."""
    value = dict(payload or {})
    session_state["pending_auto_analyze"] = value
    return value


def consume_pending_auto_analyze(session_state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Pop and return the pending auto-analyze payload."""
    value = session_state.pop("pending_auto_analyze", {})
    return dict(value) if isinstance(value, dict) else {}


def request_analysis_refresh(session_state: MutableMapping[str, Any]) -> None:
    """Flag that the analysis should re-run on the next render, even if inputs haven't changed."""
    session_state["_analysis_force_once"] = True
    session_state["_consistency_refresh_pending"] = True


def consume_analysis_force_once(session_state: MutableMapping[str, Any]) -> bool:
    """Pop the force-once flag; returns True if a forced re-analysis was pending."""
    return bool(session_state.pop("_analysis_force_once", False))


def consume_consistency_refresh_pending(session_state: MutableMapping[str, Any]) -> bool:
    """Pop the consistency-refresh flag; returns True if a refresh was pending."""
    return bool(session_state.pop("_consistency_refresh_pending", False))


def apply_flow_line_values(
    session_state: MutableMapping[str, Any],
    values: Dict[str, Any] | None,
    *,
    fields: list[str] | tuple[str, ...] = tuple(_FLOW_LINE_KEYS),
) -> Dict[str, str]:
    """Apply a dict of line text values to session state, returning the applied mapping."""
    payload = dict(values or {})
    out: Dict[str, str] = {}
    for field in fields:
        text = str(payload.get(field, "") or "")
        session_state[str(field)] = text
        out[str(field)] = text
    return out


def build_training_csv_list(
    daily_csv: str | Path | None,
    hist_csvs: list[str] | tuple[str, ...] | None,
) -> List[str]:
    """Merge historical and daily CSV paths, deduplicating by resolved path."""
    candidates: List[str] = []
    for p in (hist_csvs or []):
        s = str(p or "").strip()
        if s:
            candidates.append(s)
    s_daily = str(daily_csv or "").strip()
    if s_daily:
        candidates.append(s_daily)

    out: List[str] = []
    seen: set[str] = set()
    for raw in candidates:
        p = Path(raw).expanduser()
        try:
            if not p.exists():
                continue
        except OSError:
            continue
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def resolve_players_artifact_path(
    artifact_path: str,
    root: Path,
    *,
    load_artifact_fn: Callable[[str], Any],
) -> str:
    """Find the players_artifact.json path by reading artifact metadata, falling back to sibling/root."""
    root_path = Path(str(root)).expanduser()
    artifact_raw = str(artifact_path or "").strip()
    artifact_file = Path(artifact_raw).expanduser() if artifact_raw else None
    artifact_parent: Path | None = None
    if artifact_file is not None:
        try:
            artifact_parent = artifact_file.resolve().parent
        except OSError:
            artifact_parent = artifact_file.parent

    p_default = str(root_path / "players_artifact.json")
    try:
        art_obj = load_artifact_fn(str(artifact_path))
        pl_meta = dict((getattr(art_obj, "meta", {}) or {}).get("players_layer") or {})
        meta_p = str(pl_meta.get("players_artifact") or "").strip()
        if meta_p:
            cand = Path(meta_p).expanduser()
            if not cand.is_absolute():
                base_dir = artifact_parent if artifact_parent is not None else root_path
                cand = (base_dir / cand).resolve()
            if cand.exists():
                return str(cand)
    except Exception:
        pass
    try:
        p0 = (root_path / "players_artifact.json").expanduser()
        if p0.exists():
            return str(p0)
    except OSError:
        pass
    try:
        if artifact_parent is not None:
            p1 = artifact_parent / "players_artifact.json"
            if p1.exists():
                return str(p1)
    except OSError:
        pass
    return p_default
