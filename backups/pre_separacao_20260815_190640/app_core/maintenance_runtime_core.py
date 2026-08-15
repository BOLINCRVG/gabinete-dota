from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


def read_artifact_summary_path(artifact_path: str | Path) -> Dict[str, Any]:
    p = Path(str(artifact_path or ""))
    if not p.exists():
        return {"exists": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"exists": True, "error": "Nao foi possivel ler o JSON (arquivo corrompido)"}
    meta = data.get("meta") if isinstance(data, Mapping) else {}
    meta = meta if isinstance(meta, Mapping) else {}
    inc = meta.get("incremental") if isinstance(meta.get("incremental"), Mapping) else {}
    return {
        "exists": True,
        "trained_at": meta.get("trained_at"),
        "data_max_date": meta.get("data_max_date"),
        "n_games_total": meta.get("n_games_total"),
        "added_games_last": inc.get("added_games"),
        "backfill_suspected": inc.get("backfill_suspected"),
    }


def parse_incremental_update_stdout(stdout_text: str) -> Dict[str, Any]:
    txt = str(stdout_text or "")
    out: Dict[str, Any] = {"new_games": None, "backfill_suspected": None}
    m_games = re.search(r"new games applied:\s*(\d+)", txt, flags=re.IGNORECASE)
    if m_games:
        out["new_games"] = int(m_games.group(1))
    m_backfill = re.search(
        r"backfill[_ ]suspected\s*[:=]\s*(true|false|1|0|yes|no|y|n)",
        txt,
        flags=re.IGNORECASE,
    )
    if m_backfill:
        out["backfill_suspected"] = str(m_backfill.group(1)).strip().lower() in {"true", "1", "yes", "y"}
    return out


def build_incremental_update_cmd(
    *,
    artifact_path: str | Path,
    csv_path: str | Path,
    players_path: str | Path | None = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlcore.update_incremental",
        "--artifact",
        str(artifact_path),
        "--csv",
        str(csv_path),
    ]
    # Passa o players_artifact que o APP realmente usa, para os jogos novos irem
    # ao arquivo certo (e nao a um path obsoleto guardado no meta do artifact).
    if players_path:
        cmd += ["--players", str(players_path)]
    return cmd


def build_train_offline_cmd(
    csv_paths: Sequence[str | Path],
    *,
    out_path: str | Path,
    holdout_days: int | None = None,
    k: float | None = None,
    scale: float | None = None,
    calibrate: str | None = None,
    keep_playoffs: str | None = None,
    fit_players: bool = False,
    players_out: str | Path | None = None,
    core: str | None = None,
    whr_w2: float | None = None,
) -> List[str]:
    cmd: List[str] = [sys.executable, "-m", "mlcore.train_offline"]
    for p in (csv_paths or []):
        cmd += ["--csv", str(p)]
    cmd += ["--out", str(out_path)]
    if holdout_days is not None:
        cmd += ["--holdout-days", str(int(holdout_days))]
    if k is not None:
        cmd += ["--k", str(float(k))]
    if scale is not None:
        cmd += ["--scale", str(float(scale))]
    if calibrate is not None:
        cmd += ["--calibrate", str(calibrate)]
    if keep_playoffs is not None:
        cmd += ["--keep-playoffs", str(keep_playoffs)]
    if core is not None:
        cmd += ["--core", str(core)]
    if whr_w2 is not None:
        cmd += ["--whr-w2", str(float(whr_w2))]
    if bool(fit_players):
        cmd += ["--fit-players"]
        if players_out is not None:
            cmd += ["--players-out", str(players_out)]
    else:
        cmd += ["--no-fit-players"]
    return cmd


def resolve_train_offline_reuse_params(
    artifact_path: str | Path,
    *,
    artifact_loader: Callable[[str | Path], Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"k": 26.0, "scale": 400.0, "calibrate": "auto"}
    try:
        loader = artifact_loader
        if loader is None:
            from mlcore.artifact import MLArtifact

            loader = MLArtifact.load_json
        art = loader(artifact_path)
        out["k"] = float(getattr(art.model.params, "k", 26.0) or 26.0)
        out["scale"] = float(getattr(art.model.params, "scale_points", 400.0) or 400.0)
        # Fidelidade ao deploy validado: core/w2 vêm do meta do artefato atual
        # (LoL: whr/60; Dota: whr/20). Sem isso o retrain cai nos defaults e
        # troca silenciosamente a configuração gateada.
        meta = dict(getattr(art, "meta", None) or {})
        if meta.get("core"):
            out["core"] = str(meta["core"])
        if meta.get("whr_w2") is not None:
            out["whr_w2"] = float(meta["whr_w2"])
        # players layer só se o artefato atual TEM (Dota ainda não tem)
        out["fit_players"] = bool(meta.get("players_layer"))
        if getattr(art, "calibrator", None) is None:
            # artefato deployado sem calibrador (decisão medida) -> manter
            out["calibrate"] = "no"
        else:
            cal_a = float(getattr(art.calibrator, "a", 1.0) or 1.0)
            cal_b = float(getattr(art.calibrator, "b", 0.0) or 0.0)
            if abs(cal_a - 1.0) < 0.05 and abs(cal_b) < 0.05:
                out["calibrate"] = "no"
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return out


def build_fit_params_cmd(
    csv_paths: Sequence[str | Path],
    *,
    artifact_path: str | Path,
    players_artifact_path: str | Path,
    out_path: str | Path,
    speed_profile: str | None = None,
    progress_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool | None = None,
    run_label: str | None = None,
    start_stage: str | None = None,
    base_params_path: str | Path | None = None,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "mlcore.fit_ml_params",
        "--artifact",
        str(artifact_path),
        "--players-artifact",
        str(players_artifact_path),
        "--out",
        str(out_path),
    ]
    if speed_profile is not None:
        cmd += ["--speed-profile", str(speed_profile)]
    if progress_path is not None and str(progress_path).strip():
        cmd += ["--progress-out", str(progress_path)]
    if checkpoint_dir is not None and str(checkpoint_dir).strip():
        cmd += ["--checkpoint-dir", str(checkpoint_dir)]
    if run_label is not None and str(run_label).strip():
        cmd += ["--run-label", str(run_label).strip()]
    if start_stage is not None and str(start_stage).strip():
        cmd += ["--start-stage", str(start_stage).strip()]
    if base_params_path is not None and str(base_params_path).strip():
        cmd += ["--base-params", str(base_params_path)]
    if resume is not None:
        cmd += ["--resume" if bool(resume) else "--no-resume"]
    for p in (csv_paths or []):
        cmd += ["--csv", str(p)]
    return cmd


def build_fit_params_runtime_paths(
    *,
    app_root: str | Path,
    scope: str,
) -> Dict[str, str]:
    root = Path(str(app_root or "")).expanduser().resolve()
    scope_raw = str(scope or "runtime").strip().lower()
    scope_key = re.sub(r"[^a-z0-9._-]+", "_", scope_raw).strip("._-") or "runtime"
    run_root = root / "ops" / "runtime_fit_params" / scope_key
    progress_path = run_root / "fit_ml_params_progress.json"
    checkpoint_dir = run_root / "checkpoints"
    return {
        "run_root": str(run_root),
        "progress_path": str(progress_path),
        "checkpoint_dir": str(checkpoint_dir),
        "run_label": f"runtime_{scope_key}",
    }


def build_retrain_check_cmd(
    *,
    artifact_path: str | Path,
    csv_path: str | Path,
    keep_playoffs: str,
    exit_code: bool = True,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "mlcore.retrain_check",
        "--artifact",
        str(artifact_path),
        "--csv",
        str(csv_path),
        "--keep-playoffs",
        str(keep_playoffs),
    ]
    if bool(exit_code):
        cmd.append("--exit-code")
    return cmd


def choose_params_primary_csvs(
    csv_paths: Sequence[str | Path],
    *,
    daily_csv: str | Path | None,
    fast_mode: bool,
) -> Dict[str, Any]:
    csv_list = [str(p) for p in (csv_paths or []) if str(p).strip()]
    primary = list(csv_list)
    mode = "full_csvs_fast"
    if not bool(fast_mode):
        return {"csvs": primary, "mode": "full_csvs_full"}
    try:
        daily_s = str(daily_csv or "").strip()
        if daily_s and Path(daily_s).exists():
            daily_key = str(Path(daily_s).expanduser().resolve())
            csv_keys = [str(Path(p).expanduser().resolve()) for p in csv_list]
            if daily_key in csv_keys:
                return {"csvs": [daily_key], "mode": "daily_only_fast"}
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return {"csvs": primary, "mode": mode}


def params_output_valid_path(path_like: str | Path) -> bool:
    try:
        obj = json.loads(Path(path_like).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(obj, Mapping) or bool(obj.get("error")):
        return False
    best = obj.get("best") if isinstance(obj.get("best"), Mapping) else {}
    try:
        team_scale = float(best.get("team_scale")) if best.get("team_scale") is not None else None
        players_scale = float(best.get("players_scale")) if best.get("players_scale") is not None else None
    except (TypeError, ValueError):
        return False
    return bool(
        team_scale is not None
        and players_scale is not None
        and math.isfinite(team_scale)
        and math.isfinite(players_scale)
    )
