from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping




def get_boot_hist_csvs(session_state: Mapping[str, Any]) -> list[str]:
    return [str(path) for path in (session_state.get("boot_hist_csvs") or []) if str(path or "").strip()]


def resolve_default_achievement_path(
    *,
    artifact_path: str,
    app_root: Path,
    load_artifact_fn: Callable[[str], Any],
    artifact_sig_fn: Callable[[str], str],
    resolve_relpath_fn: Callable[[str, str], str],
) -> str:
    try:
        artifact_file = Path(artifact_path)
        if artifact_file.exists():
            artifact_obj = load_artifact_fn(artifact_sig_fn(str(artifact_file)))
            meta = dict(getattr(artifact_obj, "meta", {}) or {})
            achievements_meta = dict(meta.get("achievements") or {})
            achievement_path = str(achievements_meta.get("path") or "").strip()
            if achievement_path:
                return resolve_relpath_fn(artifact_path, achievement_path)
            return str(artifact_file.with_name("achievements.json"))
    except Exception:
        pass
    try:
        return str(Path(artifact_path).with_name("achievements.json"))
    except Exception:
        return str(app_root / "achievements.json")


def ensure_achievement_path(
    *,
    session_state: dict[str, Any],
    artifact_path: str,
    app_root: Path,
    get_achievement_path_fn: Callable[[dict[str, Any]], str],
    set_achievement_path_fn: Callable[[dict[str, Any], Any], str],
    load_artifact_fn: Callable[[str], Any],
    artifact_sig_fn: Callable[[str], str],
    resolve_relpath_fn: Callable[[str, str], str],
) -> str:
    current = get_achievement_path_fn(session_state)
    if str(current or "").strip():
        return str(current)
    value = resolve_default_achievement_path(
        artifact_path=artifact_path,
        app_root=app_root,
        load_artifact_fn=load_artifact_fn,
        artifact_sig_fn=artifact_sig_fn,
        resolve_relpath_fn=resolve_relpath_fn,
    )
    return set_achievement_path_fn(session_state, value)


def apply_season_change_reset(
    *,
    session_state: dict[str, Any],
    season_label: str,
    get_last_season_label_fn: Callable[[dict[str, Any]], str],
    set_last_season_label_fn: Callable[[dict[str, Any], str], Any],
    reset_flow_fn: Callable[..., Any],
    clear_matchup_selection_fn: Callable[[dict[str, Any]], Any],
) -> bool:
    if get_last_season_label_fn(session_state) == str(season_label):
        return False
    set_last_season_label_fn(session_state, str(season_label))
    reset_flow_fn(keep_config=True)
    clear_matchup_selection_fn(session_state)
    return True


def normalize_matchup_selection_teams(
    *,
    session_state: dict[str, Any],
    csv_sig: str,
    team_canon_maps_fn: Callable[[str], Mapping[str, Any]],
    canon_team_key_fn: Callable[[str], str],
    ensure_matchup_selection_state_fn: Callable[[dict[str, Any]], tuple[Any, Any]],
    set_selected_team_fn: Callable[[dict[str, Any], str, Any], Any],
) -> None:
    try:
        maps = team_canon_maps_fn(csv_sig)
        key_to_best = maps.get("variant_to_best_key") if isinstance(maps, Mapping) else {}

        def _normalize_team_value(value: Any) -> str:
            text = ("" if value is None else str(value)).strip()
            text = re.sub(r"\s+", " ", text)
            key = canon_team_key_fn(text)
            return str(key_to_best.get(key, text)) if isinstance(key_to_best, Mapping) else text

        team_a, team_b = ensure_matchup_selection_state_fn(session_state)
        if str(team_a).strip():
            set_selected_team_fn(session_state, "a", _normalize_team_value(team_a))
        if str(team_b).strip():
            set_selected_team_fn(session_state, "b", _normalize_team_value(team_b))
    except Exception:
        return
