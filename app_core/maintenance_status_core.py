from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app_core.bootstrap_maintenance_state_core import pid_is_alive, read_json_file_safe


def safe_mtime(path_like: str | Path) -> float | None:
    try:
        return float(Path(path_like).stat().st_mtime)
    except (OSError, TypeError, ValueError):
        return None


def _mtime_as_datetime(path_like: str | Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(Path(path_like).stat().st_mtime))
    except (OSError, OverflowError, TypeError, ValueError):
        return None


def detect_deep_fit(params_path: str | Path, artifact_path: str | Path) -> Dict[str, Any]:
    """Detecta se o full fit (profundo) foi rodado/promovido e se está defasado.

    'done' = params_fitted.json tem speed_profile=='full' (= um full fit foi
    promovido). 'stale' = o artefato (Elo) foi modificado DEPOIS do fit, então a
    camada de fusão não está mais alinhada → vale rodar de novo. 'at' = quando o
    fit foi gerado (ISO).
    """
    out: Dict[str, Any] = {"done": False, "at": "", "stale": False}
    d = read_json_file_safe(str(params_path))
    if not isinstance(d, dict) or not d:
        return out
    out["done"] = (str(d.get("speed_profile") or "").strip().lower() == "full")
    out["at"] = str(d.get("generated_at") or "")
    if out["done"]:
        fit_epoch = None
        try:
            fit_epoch = datetime.fromisoformat(str(d.get("generated_at"))).timestamp()
        except (TypeError, ValueError):
            fit_epoch = None
        art_mt = safe_mtime(artifact_path)
        # 60s de folga pra evitar falso "stale" por arredondamento de mtime.
        if fit_epoch is not None and art_mt is not None and art_mt > fit_epoch + 60.0:
            out["stale"] = True
    return out


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return int(default)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detect_params_fit_runtime(
    app_root: str | Path,
    *,
    now: Optional[datetime] = None,
    stale_s: int = 1800,
) -> Dict[str, Any]:
    root = Path(str(app_root or "")).expanduser()
    run_root = root / "ops" / "runtime_fit_params"
    if not run_root.exists():
        return {}
    now_dt = now or datetime.now()
    best: Dict[str, Any] = {}
    best_ts = float("-inf")
    for progress_path in run_root.glob("*/fit_ml_params_progress.json"):
        obj = read_json_file_safe(progress_path)
        if not isinstance(obj, dict) or not obj:
            continue
        detail = obj.get("detail") if isinstance(obj.get("detail"), dict) else {}
        updated_raw = str(obj.get("updated_at") or "").strip()
        updated_dt: Optional[datetime] = None
        try:
            if updated_raw:
                updated_dt = datetime.fromisoformat(updated_raw)
        except (TypeError, ValueError):
            updated_dt = None
        if updated_dt is None:
            updated_dt = _mtime_as_datetime(progress_path)
        updated_ts = float(updated_dt.timestamp()) if updated_dt is not None else float("-inf")
        if updated_ts < best_ts:
            continue
        proc = obj.get("process") if isinstance(obj.get("process"), dict) else {}
        pid = proc.get("pid")
        alive = pid_is_alive(pid) if pid is not None else False
        now_ts = float(now_dt.timestamp())
        age_s = float(max(0.0, now_ts - updated_ts)) if updated_dt is not None else float("inf")
        status_txt = str(obj.get("status", "") or "").strip().lower()
        scope = str(progress_path.parent.name or "")
        substage = str(detail.get("substage") or "").strip()
        phase = str(detail.get("phase") or "").strip()
        detail_scope = str(detail.get("scope") or "").strip()
        index = _to_int(detail.get("index"), 0)
        total = _to_int(detail.get("total"), 0)
        eta_s = _to_optional_float(detail.get("eta_s"))
        stage_elapsed_s = _to_optional_float(detail.get("stage_elapsed_s"))
        summary_parts: List[str] = []
        if stage_txt := str(obj.get("stage") or "").strip():
            summary_parts.append(stage_txt)
        if substage:
            summary_parts.append(substage)
        if phase:
            summary_parts.append(phase)
        if detail_scope:
            summary_parts.append(detail_scope)
        if total > 0:
            summary_parts.append(f"{max(0, index)}/{max(1, total)}")
        if eta_s is not None and math.isfinite(eta_s):
            summary_parts.append(f"eta~{int(max(0.0, eta_s))}s")
        best = {
            "scope": scope,
            "progress_path": str(progress_path),
            "status": status_txt,
            "stage": str(obj.get("stage") or ""),
            "substage": substage,
            "phase": phase,
            "detail_scope": detail_scope,
            "index": int(index),
            "total": int(total),
            "eta_s": eta_s,
            "stage_elapsed_s": stage_elapsed_s,
            "summary": " | ".join([p for p in summary_parts if str(p).strip()]),
            "updated_at": updated_raw,
            "pid": pid,
            "alive": bool(alive),
            "age_s": age_s,
            "running": bool(status_txt == "running" and alive and age_s <= float(max(60, int(stale_s)))),
        }
        best_ts = updated_ts
    return best


def collect_maintenance_status(
    csv_path_s: str,
    artifact_path_s: str,
    *,
    app_root: Path,
    wf_report_path: Path,
    wf_last_report_at: Any = None,
    ml_trace_last: Optional[Dict[str, Any]] = None,
    consistency_tol_pp: float = 0.05,
    pre_delivery_report: Optional[Dict[str, Any]] = None,
    file_sig_fn: Callable[[str], Tuple[str, int, int]],
    patch_meta_guard_payload_fn: Callable[[Tuple[str, int, int], Tuple[str, int, int]], Dict[str, Any]],
    resolve_players_artifact_path_fn: Optional[Callable[[str, Path], str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_dt = now or datetime.now()
    csv_p = Path(str(csv_path_s or "")).expanduser()
    art_p = Path(str(artifact_path_s or "")).expanduser()
    params_p = app_root / "params_fitted.json"
    try:
        if callable(resolve_players_artifact_path_fn):
            players_p = Path(str(resolve_players_artifact_path_fn(str(artifact_path_s), app_root) or "")).expanduser()
        else:
            players_p = app_root / "players_artifact.json"
    except (OSError, RuntimeError, TypeError, ValueError):
        players_p = app_root / "players_artifact.json"

    alerts: List[str] = []
    critical: List[str] = []
    need_retrain_model = False
    need_retrain_params = False
    need_walkforward = False
    need_patch_recalibration = False
    need_layer_validation = False
    need_calibration_diag = False
    resolved_wf_last_report_at = ""
    params_fit_runtime = detect_params_fit_runtime(app_root, now=now_dt)
    params_fit_running = bool(params_fit_runtime.get("running", False))

    def _newest_age_days(paths: List[Path]) -> Optional[float]:
        mtimes: List[float] = []
        for p in paths:
            mtime_dt = _mtime_as_datetime(p)
            if mtime_dt is not None:
                mtimes.append(float(mtime_dt.timestamp()))
        if not mtimes:
            return None
        try:
            newest = max(mtimes)
            return float((now_dt - datetime.fromtimestamp(newest)).total_seconds() / 86400.0)
        except (OverflowError, OSError, TypeError, ValueError):
            return None

    def _push(msg: str, *, is_critical: bool = False) -> None:
        m = str(msg or "").strip()
        if not m:
            return
        if m not in alerts:
            alerts.append(m)
        if is_critical and (m not in critical):
            critical.append(m)

    if not art_p.exists():
        _push("Artifact ausente (`ml_artifact.json`).", is_critical=True)
        need_retrain_model = True
    if not params_p.exists():
        _push("Parâmetros ausentes (`params_fitted.json`).", is_critical=True)
        need_retrain_params = True
    if not players_p.exists():
        # So cobra o players_artifact se o deploy atual USA a camada de players
        # (meta.players_layer). O Gabinete Dota deploya sem players ate a P3 —
        # cobrar um arquivo que nao faz parte do modelo e ruido.
        _expects_players = True
        try:
            import json as _json
            _meta = _json.loads(art_p.read_text(encoding="utf-8")).get("meta") or {}
            _expects_players = bool(_meta.get("players_layer"))
        except (OSError, ValueError, AttributeError):
            pass
        if _expects_players:
            _push("Players artifact ausente (`players_artifact.json`).")
    else:
        csv_m = _mtime_as_datetime(csv_p)
        pl_m = _mtime_as_datetime(players_p)
        if csv_m is not None and pl_m is not None and csv_m > pl_m + timedelta(minutes=2):
            _push("Drift de dados: CSV diário mais novo que o `players_artifact.json`.")
            need_retrain_model = True

    csv_m = _mtime_as_datetime(csv_p)
    art_m = _mtime_as_datetime(art_p)
    if csv_m is not None and art_m is not None:
        if csv_m > art_m + timedelta(minutes=2):
            _push("Drift de dados: CSV diário mais novo que o `ml_artifact.json`.")
            need_retrain_model = True
        elif now_dt - art_m > timedelta(days=10):
            _push("Artifact antigo (>10 dias).")
            need_retrain_model = True
    elif csv_m is not None and art_m is None:
        _push("Sem artifact para o CSV diário atual.", is_critical=True)
        need_retrain_model = True

    prm_m = _mtime_as_datetime(params_p)
    if prm_m is not None and art_m is not None:
        if prm_m + timedelta(minutes=2) < art_m:
            _push("Parâmetros desatualizados vs artifact.")
            need_retrain_params = True
        if now_dt.year != prm_m.year or now_dt.month != prm_m.month:
            _push("Auto-retune mensal pendente (params de mês anterior).")
            need_retrain_params = True

    if bool(need_retrain_params) and bool(params_fit_running):
        _detail = str(params_fit_runtime.get("summary") or "").strip()
        if not _detail:
            _scope = str(params_fit_runtime.get("scope") or "").strip()
            _stage = str(params_fit_runtime.get("stage") or "").strip()
            _detail = ", ".join([x for x in [_scope, _stage] if x])
        if _detail:
            _push(f"Retreino de parâmetros em execução ({_detail}).")
        else:
            _push("Retreino de parâmetros em execução.")

    wf_due = False
    if wf_last_report_at:
        try:
            wf_dt = datetime.fromisoformat(str(wf_last_report_at))
            if now_dt - wf_dt > timedelta(days=7):
                wf_due = True
        except (TypeError, ValueError):
            wf_due = True
    else:
        wf_dt_disk = _mtime_as_datetime(wf_report_path)
        if wf_dt_disk is not None:
            resolved_wf_last_report_at = wf_dt_disk.isoformat(timespec="seconds")
        if wf_dt_disk is not None:
            wf_due = bool((now_dt - wf_dt_disk) > timedelta(days=7))
        else:
            age_ref = _newest_age_days([params_p, art_p])
            if age_ref is None or age_ref > 7.0:
                wf_due = True
    if wf_due:
        _push("Walk-forward desatualizado (>7 dias).")
        need_walkforward = True

    try:
        if csv_p.exists() and params_p.exists():
            pg = patch_meta_guard_payload_fn(file_sig_fn(str(csv_p)), file_sig_fn(str(params_p)))
            st = str((pg or {}).get("status", "warn")).strip().lower()
            chg = list((pg or {}).get("changed_leagues") or [])
            age = (pg or {}).get("age_days")
            age_f = _to_optional_float(age)
            age_txt = f"{age_f:.1f}d" if age_f is not None else "-"
            if st == "refit":
                _push(f"Patch/Liga em REFIT (mudanças={len(chg)}, age={age_txt}).", is_critical=True)
                need_patch_recalibration = True
                need_retrain_params = True
                need_walkforward = True
            elif st == "warn":
                _push(f"Patch/Liga em ATENÇÃO (mudanças={len(chg)}, age={age_txt}).")
                need_patch_recalibration = True
            if isinstance(pg, dict) and pg.get("error"):
                _push(f"Patch/Liga indisponível: {pg.get('error')}")
    except Exception:
        pass

    try:
        d = None
        if isinstance(ml_trace_last, dict):
            d = ml_trace_last.get("consistency_guard_drift_pp")
        d_f = _to_optional_float(d)
        tol = _to_optional_float(consistency_tol_pp)
        if tol is None:
            tol = 0.05
        if d_f is not None and math.isfinite(d_f) and abs(d_f) > float(tol):
            _push(f"Drift de consistência acima da tolerância ({d_f:.3f} pp > {tol:.3f}).")
    except (TypeError, ValueError):
        pass

    need_layer_validation = False
    need_calibration_diag = False

    rep_pre = pre_delivery_report if isinstance(pre_delivery_report, dict) else None
    if isinstance(rep_pre, dict) and not bool(rep_pre.get("all_ok", False)):
        rows_fail = [
            r
            for r in (rep_pre.get("rows") or [])
            if isinstance(r, dict) and str(r.get("Status", "")).strip().lower() == "reprovado"
        ]
        if rows_fail:
            top = ", ".join([str(r.get("Item", "")) for r in rows_fail[:3] if str(r.get("Item", "")).strip()])
            _push(f"Pré-entrega com pendências ({len(rows_fail)}): {top}")

    need_map = {
        "need_retrain_model": bool(need_retrain_model),
        "need_retrain_params": bool(need_retrain_params),
        "need_walkforward": bool(need_walkforward),
        "need_patch_recalibration": bool(need_patch_recalibration),
        "need_layer_validation": bool(need_layer_validation),
        "need_calibration_diag": bool(need_calibration_diag),
    }
    flag_msgs = {
        "need_retrain_model": "Retreino de modelo pendente (flag interna).",
        "need_retrain_params": "Retreino de parâmetros pendente (flag interna).",
        "need_walkforward": "Walk-forward pendente (flag interna).",
        "need_patch_recalibration": "Recalibração patch/liga pendente (flag interna).",
        "need_layer_validation": "Validação por camadas pendente (flag interna).",
        "need_calibration_diag": "Diagnóstico de calibração pendente (flag interna).",
    }
    any_need = bool(any(need_map.values()))
    if any_need:
        for key, on in need_map.items():
            if bool(on):
                _push(str(flag_msgs.get(key) or f"Pendente: {key}"))

    deep_fit = detect_deep_fit(params_p, art_p)

    has_issues = bool(alerts) or any_need
    severity = "critical" if critical else ("warning" if has_issues else "ok")
    return {
        "has_issues": has_issues,
        "severity": severity,
        "alerts": alerts,
        "critical": critical,
        "players_artifact_path": str(players_p),
        "deep_fit_done": bool(deep_fit.get("done")),
        "deep_fit_at": str(deep_fit.get("at") or ""),
        "deep_fit_stale": bool(deep_fit.get("stale")),
        "need_retrain_model": bool(need_retrain_model),
        "need_retrain_params": bool(need_retrain_params),
        "need_walkforward": bool(need_walkforward),
        "need_patch_recalibration": bool(need_patch_recalibration),
        "need_layer_validation": bool(need_layer_validation),
        "need_calibration_diag": bool(need_calibration_diag),
        "params_fit_running": bool(params_fit_running),
        "params_fit_runtime": dict(params_fit_runtime or {}),
        "flags_any_need": bool(any_need),
        "resolved_wf_last_report_at": str(resolved_wf_last_report_at or ""),
    }
