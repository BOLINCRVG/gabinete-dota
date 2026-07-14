from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping


def run_pre_delivery_checks(
    session_state: MutableMapping[str, Any],
    *,
    app_root: Path,
    csv_path_s: str,
    artifact_path_s: str,
    run_smoke: bool,
    assess_confidence_baseline_file_fn: Callable[[str | Path], Mapping[str, Any]],
    assess_state_artifact_file_fn: Callable[..., Mapping[str, Any]],
    assess_patch_calibration_freshness_fn: Callable[..., Mapping[str, Any]],
    assess_combined_trace_file_fn: Callable[[str | Path], Mapping[str, Any]],
    assess_feature_trace_health_fn: Callable[..., Mapping[str, Any]],
    assess_session_trace_fn: Callable[..., Mapping[str, Any]],
    assess_navigation_integrity_fn: Callable[..., Mapping[str, Any]],
    assess_utf8_file_fn: Callable[[str | Path], Mapping[str, Any]],
    assess_ranking_coverage_fn: Callable[..., Mapping[str, Any]],
    assess_params_baseline_stability_fn: Callable[[str | Path], Mapping[str, Any]],
    assess_anchor_regression_fn: Callable[..., Mapping[str, Any]],
    run_pre_delivery_smokes_fn: Callable[..., Iterable[Mapping[str, Any]]],
    build_patch_snapshot_fn: Callable[..., Any],
    recommend_recalibration_fn: Callable[..., Any],
    compute_anchor_snapshot_fn: Callable[..., Any],
    get_feature_trace_state_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    get_ml_trace_snapshot_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    pick_tabs_fn: Callable[[str], Iterable[str]],
    required_tabs: Iterable[str],
    rank_table_fn: Callable[..., Any],
    anchor_baseline_path: Path,
    python_executable: str,
    resolve_players_artifact_path_fn: Callable[[str, Path], str] | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def _add(item: str, ok: bool, detail: str) -> None:
        rows.append({"Item": str(item), "Status": ("Aprovado" if ok else "Reprovado"), "Detalhe": str(detail), "_ok": bool(ok)})

    csv_p = Path(str(csv_path_s or "")).expanduser()
    art_p = Path(str(artifact_path_s or "")).expanduser()
    params_p = app_root / "params_fitted.json"
    conf_base_p = app_root / "ml_confidence_baseline.json"
    state_art_p = app_root / "team_state_artifact.json"
    try:
        if callable(resolve_players_artifact_path_fn):
            players_p = Path(str(resolve_players_artifact_path_fn(str(artifact_path_s), app_root) or "")).expanduser()
        else:
            players_p = app_root / "players_artifact.json"
    except (OSError, RuntimeError, TypeError, ValueError):
        players_p = app_root / "players_artifact.json"
    trace_p = app_root / "ml_trace_last.json"
    state_areal_live_on = bool(session_state.get("ff_ml_state_areal_live", False))

    _add("CSV ativo existe", bool(csv_p.exists()), str(csv_p))
    _add("Artifact ML existe", bool(art_p.exists()), str(art_p))
    _add("params_fitted.json existe", bool(params_p.exists()), str(params_p))
    _add("ml_confidence_baseline.json existe", bool(conf_base_p.exists()), str(conf_base_p))
    _add(
        "team_state_artifact.json existe",
        bool(state_art_p.exists()) or (not bool(state_areal_live_on)),
        (str(state_art_p) if state_art_p.exists() else ("opcional (A-real OFF)" if not bool(state_areal_live_on) else "ausente")),
    )
    _add("players_artifact.json existe", bool(players_p.exists()), str(players_p))

    conf_check = assess_confidence_baseline_file_fn(conf_base_p)
    _add("Baseline confiança ML íntegro", bool(conf_check.get("ok")), str(conf_check.get("detail", "")))

    state_art_check = assess_state_artifact_file_fn(
        state_art_p,
        optional_when_disabled=(not bool(state_areal_live_on)),
    )
    _add("State artifact A-real íntegro", bool(state_art_check.get("ok")), str(state_art_check.get("detail", "")))

    patch_check = assess_patch_calibration_freshness_fn(
        csv_path=csv_p,
        params_path=params_p,
        build_patch_snapshot_fn=build_patch_snapshot_fn,
        recommend_recalibration_fn=recommend_recalibration_fn,
    )
    _add("Patch/Liga calibração em dia", bool(patch_check.get("ok")), str(patch_check.get("detail", "")))

    trace_check = assess_combined_trace_file_fn(trace_p)
    _add("Trace combinado válido", bool(trace_check.get("ok")), str(trace_check.get("detail", "")))

    try:
        feature_state = get_feature_trace_state_fn(session_state)
        feature_health = assess_feature_trace_health_fn(
            trace_path=trace_p,
            feature_state=feature_state,
        )
        _add("Shadow feature trace saudável", bool(feature_health.get("ok")), str(feature_health.get("detail", "")))
    except Exception as exc:
        _add("Shadow feature trace saudável", False, f"Erro: {exc}")

    ml_trace_snapshot = get_ml_trace_snapshot_fn(session_state)
    session_trace_check = assess_session_trace_fn(
        trace_snapshot=ml_trace_snapshot,
        p_map_used_raw=session_state.get("_p_map_used", float("nan")),
    )
    _add("Sessão com p_map_used", bool(session_trace_check.get("ok")), str(session_trace_check.get("detail", "")))

    nav_check = assess_navigation_integrity_fn(
        tabs_full=list(pick_tabs_fn("Completo")),
        required_tabs=set(required_tabs),
    )
    _add("Navegação mínima íntegra", bool(nav_check.get("ok")), str(nav_check.get("detail", "")))

    utf8_check = assess_utf8_file_fn(app_root / "plays_app.py")
    _add("Encoding UTF-8 íntegro", bool(utf8_check.get("ok")), str(utf8_check.get("detail", "")))

    try:
        top = rank_table_fn(str(art_p), as_of=date.today(), top_n=600)
        ranking_check = assess_ranking_coverage_fn(ranking_df=top, required_leagues={"LCK", "LPL", "LEC", "LCS"})
        _add("Ranking global cobre ligas-chave", bool(ranking_check.get("ok")), str(ranking_check.get("detail", "")))
    except Exception as exc:
        _add("Ranking global cobre ligas-chave", False, f"Erro: {exc}")

    params_check = assess_params_baseline_stability_fn(params_p)
    _add("Parâmetros baseline estáveis", bool(params_check.get("ok")), str(params_check.get("detail", "")))

    anchor_check = assess_anchor_regression_fn(
        csv_path=csv_p,
        baseline_path=anchor_baseline_path,
        compute_anchor_snapshot_fn=compute_anchor_snapshot_fn,
    )
    _add("Regressão de odds âncora", bool(anchor_check.get("ok")), str(anchor_check.get("detail", "")))

    if run_smoke:
        for smoke_row in run_pre_delivery_smokes_fn(
            cwd=app_root,
            csv_path=csv_p,
            python_executable=python_executable,
            timeout=180,
        ):
            _add(str(smoke_row.get("label", "Smoke")), bool(smoke_row.get("ok")), str(smoke_row.get("detail", "")))

    ok_count = int(sum(1 for r in rows if bool(r.get("_ok"))))
    total_count = len(rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ok_count": ok_count,
        "total_count": total_count,
        "all_ok": bool(total_count > 0 and ok_count == total_count),
        "rows": rows,
    }
