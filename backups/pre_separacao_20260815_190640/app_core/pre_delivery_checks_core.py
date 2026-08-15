from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pandas as pd


def extract_p_map_used(trace_obj: Any) -> float | None:
    """Extrai o valor de probabilidade p_map_used de um dicionário de trace ML."""
    if not isinstance(trace_obj, dict):
        return None
    value = trace_obj.get("p_map_used")
    if value is None and isinstance(trace_obj.get("probabilities"), dict):
        value = trace_obj.get("probabilities", {}).get("p_map_used")
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return float(parsed)
    except (TypeError, ValueError):
        return None
    return None


def assess_combined_trace_file(trace_path: str | Path) -> dict[str, Any]:
    """Avalia a integridade do arquivo de trace combinado, verificando chaves e p_map_used."""
    path = Path(str(trace_path)).expanduser()
    if not path.exists():
        return {
            "ok": False,
            "detail": f"Ausente: {path}",
            "payload": {},
            "p_map_used": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ml_trace = payload.get("ml_trace") if isinstance(payload, dict) else {}
        p_trace = extract_p_map_used(ml_trace)
        ok = (
            isinstance(payload, dict)
            and isinstance(payload.get("ml_trace"), dict)
            and (p_trace is not None)
            and isinstance(payload.get("resumo_markets_trace"), list)
            and isinstance(payload.get("players_trace"), list)
        )
        return {
            "ok": bool(ok),
            "detail": f"p_map_used={p_trace} | resumo={len(payload.get('resumo_markets_trace') or [])} | players={len(payload.get('players_trace') or [])}",
            "payload": payload if isinstance(payload, dict) else {},
            "p_map_used": p_trace,
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "detail": f"Erro ao ler trace: {exc}",
            "payload": {},
            "p_map_used": None,
        }


def assess_feature_trace_health(
    *,
    trace_path: str | Path,
    feature_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Avalia a saúde do trace de features verificando erros de sessão e presença da chave feature_trace."""
    state = dict(feature_state or {})
    err = str(state.get("last_error", "") or "").strip()
    ok_session = not err
    file_key_ok = True
    file_detail = "trace combinado ausente"
    path = Path(str(trace_path)).expanduser()
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            file_key_ok = isinstance(payload, dict) and isinstance(payload.get("feature_trace"), list)
            file_detail = f"feature_trace={len((payload.get('feature_trace') or [])) if isinstance(payload, dict) else '-'}"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            file_key_ok = False
            file_detail = f"erro trace combinado: {exc}"
    return {
        "ok": bool(ok_session and file_key_ok),
        "detail": ("sem erro de sessão" if ok_session else f"erro sessão: {err}") + f" | {file_detail}",
    }


def assess_session_trace(
    *,
    trace_snapshot: Mapping[str, Any] | None,
    p_map_used_raw: Any,
) -> dict[str, Any]:
    """Avalia o trace ML da sessão verificando presença e validade do p_map_used."""
    snapshot = dict(trace_snapshot or {})
    trace_obj = snapshot.get("_ml_trace_last")
    p_value = extract_p_map_used(trace_obj)
    if p_value is None:
        try:
            parsed = float(p_map_used_raw)
            if math.isfinite(parsed):
                p_value = float(parsed)
        except (TypeError, ValueError):
            p_value = None
    ok = isinstance(trace_obj, dict) and (p_value is not None)
    return {
        "ok": bool(ok),
        "detail": f"p_map_used={p_value}",
        "p_map_used": p_value,
    }


def assess_confidence_baseline_file(path_like: str | Path) -> dict[str, Any]:
    """Avalia o arquivo de baseline de confiança ML verificando versão e estrutura de ligas."""
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        return {"ok": False, "detail": "arquivo ausente"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ok = (
            isinstance(payload, dict)
            and isinstance(payload.get("global"), dict)
            and isinstance(payload.get("by_league"), dict)
            and str(payload.get("version", "")).startswith("ml_confidence_baseline_v")
        )
        return {
            "ok": bool(ok),
            "detail": f"version={payload.get('version')} | ligas={len((payload.get('by_league') or {})) if isinstance(payload, dict) else 0}",
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_state_artifact_file(path_like: str | Path, *, optional_when_disabled: bool) -> dict[str, Any]:
    """Avalia o artefato de estado de times verificando versão, times e ligas presentes."""
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        return {
            "ok": bool(optional_when_disabled),
            "detail": ("opcional (A-real OFF)" if bool(optional_when_disabled) else "arquivo ausente"),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = (payload.get("meta") or {}) if isinstance(payload, dict) and isinstance(payload.get("meta"), dict) else {}
        ok = (
            isinstance(payload, dict)
            and str(payload.get("version", "")).startswith("team_state_artifact_v")
            and isinstance(payload.get("teams"), dict)
            and isinstance(payload.get("by_league"), dict)
            and isinstance(payload.get("meta"), dict)
        )
        return {
            "ok": bool(ok),
            "detail": f"version={payload.get('version')} | teams={int(meta.get('teams') or 0)} | games={int(meta.get('games_used') or 0)}",
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_patch_calibration_freshness(
    *,
    csv_path: str | Path,
    params_path: str | Path,
    build_patch_snapshot_fn: Any,
    recommend_recalibration_fn: Any,
) -> dict[str, Any]:
    """Avalia se a calibração de patch está atualizada comparando snapshot atual com o ajustado."""
    csv_p = Path(str(csv_path)).expanduser()
    params_p = Path(str(params_path)).expanduser()
    if not csv_p.exists() or not params_p.exists():
        return {"ok": False, "detail": "CSV ou params_fitted.json ausente"}
    try:
        dfp = pd.read_csv(
            str(csv_p),
            usecols=["gameid", "league", "patch", "date", "participantid"],
            low_memory=False,
        )
        dfp = dfp[pd.to_numeric(dfp.get("participantid"), errors="coerce").isin([100, 200])].copy()
        dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
        dfp = dfp.dropna(subset=["date"]).sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")
        cur_snap = build_patch_snapshot_fn(dfp)
        fit_payload = json.loads(params_p.read_text(encoding="utf-8"))
        fit_snap = fit_payload.get("patch_snapshot") if isinstance(fit_payload, dict) else {}
        fit_ts = fit_payload.get("generated_at") if isinstance(fit_payload, dict) else None
        rec = recommend_recalibration_fn(
            current_snapshot=cur_snap if isinstance(cur_snap, dict) else {},
            fitted_snapshot=fit_snap if isinstance(fit_snap, dict) else {},
            fitted_generated_at=fit_ts,
            max_age_days=7,
            min_games_new_patch=20,
        )
        ok = str((rec or {}).get("status", "warn")) == "ok"
        changed = rec.get("changed_leagues") if isinstance(rec, dict) else []
        return {
            "ok": bool(ok),
            "detail": f"status={rec.get('status')} | changed={len(changed or [])} | age_days={rec.get('age_days')}",
        }
    except Exception as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_navigation_integrity(
    *,
    tabs_full: list[str] | tuple[str, ...],
    required_tabs: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """Verifica se todas as abas obrigatórias estão presentes na lista de abas disponíveis."""
    try:
        ok = set(required_tabs).issubset(set(tabs_full))
        return {"ok": bool(ok), "detail": f"abas={len(list(tabs_full))}"}
    except TypeError as exc:
        return {"ok": False, "detail": str(exc)}


def assess_utf8_file(path_like: str | Path) -> dict[str, Any]:
    """Verifica se um arquivo pode ser lido como UTF-8 sem erros."""
    path = Path(str(path_like)).expanduser()
    try:
        _ = path.read_text(encoding="utf-8")
        return {"ok": True, "detail": str(path)}
    except (OSError, UnicodeDecodeError) as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_ranking_coverage(
    *,
    ranking_df: pd.DataFrame | None,
    required_leagues: set[str] | None = None,
) -> dict[str, Any]:
    """Verifica se o DataFrame de ranking cobre as ligas obrigatórias especificadas."""
    need = set(required_leagues or {"LCK", "LPL", "LEC", "LCS"})
    try:
        if ranking_df is None or ranking_df.empty:
            return {"ok": False, "detail": "ranking v2 vazio"}
        leagues = set(ranking_df["league"].astype(str).str.strip()) if "league" in ranking_df.columns else set()
        ok = need.issubset(leagues)
        return {"ok": bool(ok), "detail": f"presentes={sorted(list(leagues))[:12]}..."}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_params_baseline_stability(path_like: str | Path) -> dict[str, Any]:
    """Avalia se os parâmetros ajustados estão dentro de faixas aceitáveis e não divergem do baseline."""
    path = Path(str(path_like)).expanduser()
    if not path.exists():
        return {"ok": False, "detail": "params_fitted.json ausente"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = (payload.get("best") or {}) if isinstance(payload, dict) else {}
        base = (payload.get("baseline") or {}) if isinstance(payload, dict) else {}
        team_scale = float(best.get("team_scale") or 1.0)
        players_scale = float(best.get("players_scale") or 1.0)
        ll_b = base.get("logloss")
        ll_f = best.get("logloss")
        ok_scales = (0.05 <= team_scale <= 8.0) and (0.05 <= players_scale <= 8.0)
        ok_drift = True
        if ll_b is not None and ll_f is not None:
            ok_drift = float(ll_f) <= float(ll_b) + 0.020
        return {
            "ok": bool(ok_scales and ok_drift),
            "detail": f"team_scale={team_scale:.3f} players_scale={players_scale:.3f} logloss_base={ll_b} logloss_best={ll_f}",
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def assess_anchor_regression(
    *,
    csv_path: str | Path,
    baseline_path: str | Path,
    compute_anchor_snapshot_fn: Any,
) -> dict[str, Any]:
    """Verifica se as âncoras de probabilidade atuais não divergem mais que 8pp do baseline salvo."""
    csv_p = Path(str(csv_path)).expanduser()
    baseline_p = Path(str(baseline_path)).expanduser()
    if not csv_p.exists():
        return {"ok": False, "detail": "CSV ausente"}
    if not baseline_p.exists():
        return {"ok": False, "detail": f"baseline ausente: {baseline_p.name}"}
    try:
        now_payload = compute_anchor_snapshot_fn(str(csv_p))
        base_payload = json.loads(baseline_p.read_text(encoding="utf-8"))
        now_rows = {str(x.get("key")): x for x in (now_payload.get("anchors") or []) if isinstance(x, dict)}
        base_rows = {str(x.get("key")): x for x in (base_payload.get("anchors") or []) if isinstance(x, dict)}
        keys = sorted(set(now_rows.keys()) & set(base_rows.keys()))
        dmax = 0.0
        used = 0
        avail = 0
        for key in keys:
            row_now = now_rows.get(key) or {}
            row_base = base_rows.get(key) or {}
            if bool(row_base.get("ok")):
                avail += 1
            if not bool(row_now.get("ok")) or not bool(row_base.get("ok")):
                continue
            try:
                delta_now = abs(float(row_now.get("p_a")) - float(row_base.get("p_a")))
                dmax = max(dmax, delta_now)
                used += 1
            except (TypeError, ValueError):
                continue
        min_required = max(3, min(8, int(max(1, avail) * 0.5)))
        ok = (used >= int(min_required)) and (dmax <= 0.08)
        return {
            "ok": bool(ok),
            "detail": f"anchors_usadas={used}/{avail} | min_req={min_required} | max_delta_p={dmax:.4f} | regra<=0.08",
        }
    except Exception as exc:
        return {"ok": False, "detail": f"Erro: {exc}"}


def _clean_smoke_output(output: str) -> str:
    if not str(output or "").strip():
        return ""
    lines: list[str] = []
    for raw_line in str(output).splitlines():
        line = str(raw_line)
        if "missing ScriptRunContext" in line:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def run_unittest_smoke(
    *,
    test_targets: list[str] | tuple[str, ...],
    cwd: str | Path,
    csv_path: str | Path | None = None,
    timeout: int = 180,
    python_executable: str | None = None,
    extra_env: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa um subconjunto de testes unitários via subprocess e retorna resultado com saída."""
    env = dict(os.environ)
    if isinstance(extra_env, Mapping):
        for key, value in extra_env.items():
            if value is not None:
                env[str(key)] = str(value)
    csv_p = Path(str(csv_path)).expanduser() if str(csv_path or "").strip() else None
    if csv_p is not None and csv_p.exists():
        env["MLCORE_CSV"] = str(csv_p)
    cmd = [str(python_executable or sys.executable), "-m", "unittest", *[str(x) for x in test_targets]]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(Path(str(cwd)).expanduser()),
            capture_output=True,
            text=True,
            timeout=int(timeout),
            env=env,
        )
        output = ((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else "")).strip()
        output = _clean_smoke_output(output)
        return {
            "ok": int(cp.returncode) == 0,
            "detail": output[-800:] if output else "sem saída",
            "returncode": int(cp.returncode),
            "cmd": cmd,
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "ok": False,
            "detail": f"Erro ao executar: {exc}",
            "returncode": None,
            "cmd": cmd,
        }


def run_pre_delivery_smokes(
    *,
    cwd: str | Path,
    csv_path: str | Path | None = None,
    python_executable: str | None = None,
    timeout: int = 180,
    extra_env: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Executa a bateria de smoke tests de pré-entrega e retorna lista de resultados por suite."""
    specs = [
        ("Smoke test", ["mlcore.tests.test_smoke", "mlcore.tests.test_ui_smoke"]),
        (
            "Players/Campeões smoke",
            ["mlcore.tests.test_ui_smoke.UISmokeTest.test_e2e_tab_focus_players_and_champions"],
        ),
        (
            "Series smoke",
            ["mlcore.tests.test_ui_smoke.UISmokeTest.test_e2e_tab_focus_series"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for label, targets in specs:
        res = run_unittest_smoke(
            test_targets=targets,
            cwd=cwd,
            csv_path=csv_path,
            timeout=timeout,
            python_executable=python_executable,
            extra_env=extra_env,
        )
        rows.append(
            {
                "label": str(label),
                "ok": bool(res.get("ok")),
                "detail": str(res.get("detail", "")),
            }
        )
    return rows
