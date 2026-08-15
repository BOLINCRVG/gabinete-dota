from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

from app_core.flow_runtime_core import build_training_csv_list
from app_core.ml_session_runtime_core import get_fit_scale_state, set_fit_base_scales, set_pre_delivery_report
from app_core.runtime_session_state_core import (
    set_cal_diag_last_report,
    set_lv_last_report,
    set_ml_confidence_baseline_last_report,
    set_team_state_artifact_last_report,
    set_wf_last_report,
)


def run_daily_pipeline(
    session_state: MutableMapping[str, Any],
    root: Path,
    *,
    csv_path_s: str,
    artifact_path_s: str,
    run_smoke: bool = True,
    fast_mode: bool = True,
    baseline_path: Path,
    paths_for_ml_fn: Callable[[Path], tuple[Any, Any, Any]],
    resolve_players_artifact_path_fn: Callable[[str, Path], str],
    compute_anchor_snapshot_fn: Callable[[str], Dict[str, Any]],
    fit_ml_params_run_fn: Callable[..., Dict[str, Any]],
    enrich_params_with_areal_prob_tuning_fn: Callable[[Dict[str, Any], Any], tuple[Dict[str, Any], str]],
    run_layer_validation_cached_fn: Callable[..., Dict[str, Any]],
    run_calibration_diag_cached_fn: Callable[..., Dict[str, Any]],
    run_walkforward_cached_fn: Callable[..., Dict[str, Any]],
    load_fitted_params_cached_fn: Callable[[Any], Dict[str, Any]],
    file_sig_fn: Callable[[str], Any],
    generate_ml_confidence_baseline_fn: Callable[..., Dict[str, Any]],
    generate_team_state_artifact_fn: Callable[..., Dict[str, Any]],
    run_pre_delivery_checks_fn: Callable[..., Dict[str, Any]],
    clear_team_state_cache_fn: Callable[[], None] | None = None,
) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    rep_lv: Dict[str, Any] | None = None
    rep_wf: Dict[str, Any] | None = None

    def _add(step: str, ok: bool, detail: str) -> None:
        steps.append({"step": str(step), "ok": bool(ok), "detail": str(detail)})

    daily_csv, hist_csvs, _ = paths_for_ml_fn(root)
    csv_daily = str(csv_path_s or "").strip() or str(daily_csv or "").strip()
    artifact_path = str(artifact_path_s or "").strip()
    csv_list = build_training_csv_list(csv_daily, hist_csvs)

    if not csv_list:
        _add("CSVs de entrada", False, "nenhum CSV válido encontrado")
        return {"ok": False, "steps": steps, "generated_at": datetime.now().isoformat(timespec="seconds")}
    _add("CSVs de entrada", True, f"n_csv={len(csv_list)}")

    try:
        if (not baseline_path.exists()) and csv_daily and Path(csv_daily).exists():
            snap = compute_anchor_snapshot_fn(csv_daily)
            baseline_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
            _add("Baseline de âncoras", True, f"gerado: {baseline_path.name}")
        else:
            _add("Baseline de âncoras", True, f"ok: {baseline_path.name}")
    except Exception as exc:
        _add("Baseline de âncoras", False, f"erro: {exc}")

    players_artifact_path = resolve_players_artifact_path_fn(artifact_path, root)
    if not Path(artifact_path).exists():
        _add("ml_artifact.json", False, "ausente")
        return {"ok": False, "steps": steps, "generated_at": datetime.now().isoformat(timespec="seconds")}
    if not Path(players_artifact_path).exists():
        _add("players_artifact.json", False, f"ausente: {players_artifact_path}")
        return {"ok": False, "steps": steps, "generated_at": datetime.now().isoformat(timespec="seconds")}
    _add("Artifacts", True, f"ml={artifact_path} | players={players_artifact_path}")

    try:
        rep_fit = fit_ml_params_run_fn(
            artifact_path=str(artifact_path),
            players_artifact_path=str(players_artifact_path),
            csv_paths=csv_list,
            min_train_games=int(session_state.get("wf_min_train_games", 300) or 300),
            speed_profile=("fast" if bool(fast_mode) else "full"),
        )
        rep_fit, areal_src = enrich_params_with_areal_prob_tuning_fn(rep_fit, root=root)
        (root / "params_fitted.json").write_text(json.dumps(rep_fit, ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(rep_fit, Mapping) and rep_fit.get("best"):
            best = rep_fit.get("best") or {}
            set_fit_base_scales(
                session_state,
                team_scale=float(best.get("team_scale") or 1.0),
                players_scale=float(best.get("players_scale") or 1.0),
            )
            _add(
                "Auto-ajuste Team/Players",
                True,
                (
                    f"team_scale={float(best.get('team_scale') or 1.0):.3f} "
                    f"players_scale={float(best.get('players_scale') or 1.0):.3f}"
                    + (f" | areal_tuning={areal_src}" if str(areal_src or "").strip() else "")
                ),
            )
        else:
            _add("Auto-ajuste Team/Players", False, "retorno sem 'best'")
    except Exception as exc:
        _add("Auto-ajuste Team/Players", False, f"erro: {exc}")

    fit_scale_state = get_fit_scale_state(session_state)
    ts = float(fit_scale_state.get("team_scale", 1.0) or 1.0)
    ps = float(fit_scale_state.get("players_scale", 1.0) or 1.0)
    try:
        rep_lv = run_layer_validation_cached_fn(
            tuple(csv_list),
            str(artifact_path),
            str(players_artifact_path),
            int(session_state.get("lv_min_train_games", 1200) or 1200),
            float(ts),
            float(ps),
            int(session_state.get("lv_min_league_games", 400) or 400),
        )
        set_lv_last_report(session_state, rep_lv)
        ok_lv = isinstance(rep_lv, Mapping) and not rep_lv.get("error")
        _add("Validação por camadas", bool(ok_lv), f"n={int((rep_lv or {}).get('n_samples', 0) or 0)}")
    except Exception as exc:
        _add("Validação por camadas", False, f"erro: {exc}")

    try:
        rep_cd = run_calibration_diag_cached_fn(
            tuple(csv_list),
            str(artifact_path),
            str(players_artifact_path),
            float(ts),
            float(ps),
            int(session_state.get("cal_diag_bins", 10) or 10),
            int(session_state.get("cal_diag_min_league_games", 250) or 250),
        )
        set_cal_diag_last_report(session_state, rep_cd)
        ok_cd = isinstance(rep_cd, Mapping) and not rep_cd.get("error")
        _add("Diagnóstico calibração", bool(ok_cd), f"models={len((rep_cd or {}).get('models', {}) or {})}")
    except Exception as exc:
        _add("Diagnóstico calibração", False, f"erro: {exc}")

    try:
        rep_wf = run_walkforward_cached_fn(
            tuple(csv_list),
            int(session_state.get("wf_min_train_games", 1500) or 1500),
            float(session_state.get("wf_k", 26.0) or 26.0),
            float(session_state.get("wf_scale", 400.0) or 400.0),
        )
        set_wf_last_report(session_state, rep_wf)
        ok_wf = isinstance(rep_wf, Mapping) and not rep_wf.get("error")
        wf_overall = (rep_wf or {}).get("overall") if isinstance((rep_wf or {}).get("overall"), Mapping) else {}
        _add("Walk-forward", bool(ok_wf), f"folds={int((wf_overall or {}).get('folds', 0) or 0)}")
    except Exception as exc:
        rep_wf = None
        _add("Walk-forward", False, f"erro: {exc}")

    try:
        fit_live = load_fitted_params_cached_fn(file_sig_fn(str(root / "params_fitted.json"))) if (root / "params_fitted.json").exists() else {}
        rep_cb = generate_ml_confidence_baseline_fn(
            root=root,
            lv_report=rep_lv,
            wf_report=rep_wf,
            params_fit=fit_live,
        )
        if isinstance(rep_cb, Mapping) and not rep_cb.get("error"):
            set_ml_confidence_baseline_last_report(session_state, rep_cb)
            n_lg = len((rep_cb or {}).get("by_league") or {}) if isinstance(rep_cb, Mapping) else 0
            _add("Baseline confiança ML", True, f"ligas={int(n_lg)}")
        else:
            _add("Baseline confiança ML", False, f"erro: {(rep_cb or {}).get('error') if isinstance(rep_cb, Mapping) else 'desconhecido'}")
    except Exception as exc:
        _add("Baseline confiança ML", False, f"erro: {exc}")

    try:
        rep_sa = generate_team_state_artifact_fn(
            root=root,
            csv_paths=csv_list,
            keep_playoffs=None,
        )
        if isinstance(rep_sa, Mapping) and not rep_sa.get("error"):
            set_team_state_artifact_last_report(session_state, rep_sa)
            if clear_team_state_cache_fn is not None:
                try:
                    clear_team_state_cache_fn()
                except Exception:
                    pass
            sa_meta = (rep_sa.get("meta") or {}) if isinstance(rep_sa.get("meta"), Mapping) else {}
            _add(
                "State artifact (A real)",
                True,
                (
                    f"times={int(sa_meta.get('teams') or 0)} | jogos={int(sa_meta.get('games_used') or 0)}"
                    f" | mode={sa_meta.get('update_mode') or '-'!s}"
                ),
            )
        else:
            _add(
                "State artifact (A real)",
                False,
                f"erro: {(rep_sa or {}).get('error') if isinstance(rep_sa, Mapping) else 'desconhecido'}",
            )
    except Exception as exc:
        _add("State artifact (A real)", False, f"erro: {exc}")

    try:
        rep_pre = run_pre_delivery_checks_fn(root, csv_daily, artifact_path, run_smoke=bool(run_smoke))
        set_pre_delivery_report(session_state, rep_pre)
        ok_pre = isinstance(rep_pre, Mapping) and bool(rep_pre.get("all_ok", False))
        _add(
            "Pré-entrega",
            bool(ok_pre),
            f"ok={int((rep_pre or {}).get('ok_count', 0) or 0)}/{int((rep_pre or {}).get('total_count', 0) or 0)}",
        )
    except Exception as exc:
        _add("Pré-entrega", False, f"erro: {exc}")

    ok_all = all(bool(s.get("ok")) for s in steps)
    return {
        "ok": bool(ok_all),
        "steps": steps,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_smoke": bool(run_smoke),
        "fast_mode": bool(fast_mode),
    }
