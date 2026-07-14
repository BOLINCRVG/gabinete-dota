from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


def _artifact_sibling_players_path(root: Path, artifact_path: str) -> Path:
    artifact_raw = str(artifact_path or "").strip()
    if artifact_raw:
        return Path(artifact_raw).expanduser().with_name("players_artifact.json")
    return root / "players_artifact.json"


def resolve_quick_maintenance_players_artifact_path(
    *,
    root: Path,
    artifact_path: str,
    resolve_players_artifact_path_fn: Callable[[str, Path], str] | None = None,
) -> Path:
    try:
        if callable(resolve_players_artifact_path_fn):
            resolved = Path(str(resolve_players_artifact_path_fn(str(artifact_path), root) or "")).expanduser()
            if str(resolved).strip():
                return resolved
    except Exception:
        # Callback externo pode falhar; caímos para o path pareado.
        pass
    return _artifact_sibling_players_path(root, artifact_path)


def resolve_quick_maintenance_players_output_path(
    *,
    root: Path,
    artifact_path: str,
    resolve_players_artifact_path_fn: Callable[[str, Path], str] | None = None,
) -> Path:
    resolved = resolve_quick_maintenance_players_artifact_path(
        root=root,
        artifact_path=artifact_path,
        resolve_players_artifact_path_fn=resolve_players_artifact_path_fn,
    )
    try:
        if resolved.exists():
            return resolved
    except OSError:
        pass
    return _artifact_sibling_players_path(root, artifact_path)


def build_quick_walkforward_kwargs(
    *,
    csv_list: list[str],
    wf_cfg: Mapping[str, Any],
    artifact_path: str | Path,
    players_artifact_path: str | Path,
    params_fitted_path: str | Path,
    fit_scale_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fit_scale_state = fit_scale_state if isinstance(fit_scale_state, Mapping) else {}
    return {
        "csv_paths": tuple(str(p) for p in (csv_list or []) if str(p).strip()),
        "min_train_games": int(wf_cfg.get("min_train_games", 1500) or 1500),
        "k": float(wf_cfg.get("k", 26.0) or 26.0),
        "scale": float(wf_cfg.get("scale", 400.0) or 400.0),
        "artifact_path": str(artifact_path),
        "players_artifact_path": str(players_artifact_path),
        "team_scale": float(fit_scale_state.get("team_scale", 1.0) or 1.0),
        "players_scale": float(fit_scale_state.get("players_scale", 1.0) or 1.0),
        "params_fitted_path": str(params_fitted_path),
    }


