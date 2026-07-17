# -*- coding: utf-8 -*-
"""Bootstrap flow: gate screen, CSV prepare runtime, and auto-maintenance."""
from __future__ import annotations

import json
import os
import sys
import time
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from mlcore.utils import safe_float_or_none as _fnum
from app_core.ui_labels import fmt_bytes_short as _fmt_bytes_short
from app_core.csv_prepare_ui_core import (
    render_csv_prepare_live_dashboard as _render_csv_prepare_live_dashboard,
    render_csv_prepare_report_summary as _render_csv_prepare_report_summary,
)
from app_core.process_runner_core import run_cmd as _run_cmd
from app_core.flow_runtime_core import build_training_csv_list as _build_training_csv_list
from app_core.maintenance_status_core import collect_maintenance_status as core_collect_maintenance_status
from app_core.maintenance_policy_core import (
    adjust_needs_after_incremental as core_adjust_needs_after_incremental,
    decide_maintenance_policy as core_decide_maintenance_policy,
    force_params_after_model_refresh as core_force_params_after_model_refresh,
    resolve_maintenance_preferences as core_resolve_maintenance_preferences,
    seed_maintenance_needs as core_seed_maintenance_needs,
)
from app_core.maintenance_summary_core import (
    build_maintenance_quality_status as core_build_maintenance_quality_status,
)
from app_core.bootstrap_maintenance_state_core import (
    acquire_bootstrap_maint_lock as core_acquire_bootstrap_maint_lock,
    bootstrap_maint_state_finish as core_bootstrap_maint_state_finish,
    bootstrap_maint_state_start as core_bootstrap_maint_state_start,
    bootstrap_maint_state_update as core_bootstrap_maint_state_update,
    clear_bootstrap_auto_maint_run_state as core_clear_bootstrap_auto_maint_run_state,
    cleanup_stale_bootstrap_maint_lock as core_cleanup_stale_bootstrap_maint_lock,
    get_bootstrap_runtime_state as core_get_bootstrap_runtime_state,
    load_bootstrap_maint_report_for_fp as core_load_bootstrap_maint_report_for_fp,
    read_bootstrap_maint_state as core_read_bootstrap_maint_state,
    read_json_file_safe as core_read_json_file_safe,
    release_bootstrap_maint_lock as core_release_bootstrap_maint_lock,
    set_bootstrap_active_run_id as core_set_bootstrap_active_run_id,
    set_bootstrap_auto_maint_last_report as core_set_bootstrap_auto_maint_last_report,
    set_bootstrap_auto_maint_run_fp as core_set_bootstrap_auto_maint_run_fp,
    set_bootstrap_last_report as core_set_bootstrap_last_report,
    set_bootstrap_lock_poll_n as core_set_bootstrap_lock_poll_n,
    set_bootstrap_status_before as core_set_bootstrap_status_before,
    write_json_file_atomic as core_write_json_file_atomic,
)
from app_core.csv_bootstrap_runtime_core import (
    consume_csv_prepare_refresh as core_consume_csv_prepare_refresh,
    consume_bootstrap_gate_state as core_consume_bootstrap_gate_state,
    csv_bootstrap_fingerprint as core_csv_bootstrap_fingerprint,
    csv_bootstrap_fingerprint_key as core_csv_bootstrap_fingerprint_key,
    mark_bootstrap_gate_completed as core_mark_bootstrap_gate_completed,
    reconcile_bootstrap_maintenance_status as core_reconcile_bootstrap_maintenance_status,
    request_csv_runtime_refresh as core_request_csv_runtime_refresh,
    should_run_bootstrap_auto_maintenance as core_should_run_bootstrap_auto_maintenance,
)
from app_core.maintenance_runtime_core import (
    build_fit_params_runtime_paths as core_build_fit_params_runtime_paths,
    build_fit_params_cmd as core_build_fit_params_cmd,
    build_incremental_update_cmd as core_build_incremental_update_cmd,
    build_train_offline_cmd as core_build_train_offline_cmd,
    choose_params_primary_csvs as core_choose_params_primary_csvs,
    params_output_valid_path as core_params_output_valid_path,
    parse_incremental_update_stdout as core_parse_incremental_update_stdout,
    resolve_train_offline_reuse_params as core_resolve_train_offline_reuse_params,
)
from app_core.runtime_session_state_core import (
    get_lv_last_report as core_get_lv_last_report,
    get_wf_last_report as core_get_wf_last_report,
    get_wf_last_report_at as core_get_wf_last_report_at,
    set_lv_last_report as core_set_lv_last_report,
    set_ml_confidence_baseline_last_report as core_set_ml_confidence_baseline_last_report,
    set_quick_maint_last_log as core_set_quick_maint_last_log,
    set_team_state_artifact_last_report as core_set_team_state_artifact_last_report,
    set_wf_last_report as core_set_wf_last_report,
    set_wf_last_report_at as core_set_wf_last_report_at,
)
from app_core.ml_session_runtime_core import (
    get_fit_scale_state as core_get_fit_scale_state,
    get_ml_trace_snapshot as core_get_ml_trace_snapshot,
    get_pre_delivery_report as core_get_pre_delivery_report,
    set_fit_base_scales as core_set_fit_base_scales,
)
from app_core.params_tuning_core import (
    postprocess_params_file_tuning as core_postprocess_params_file_tuning,
)
from app_core.ml_league_runtime_core import (
    sanitize_ml_marginal_band_coeffs as core_sanitize_ml_marginal_band_coeffs,
)


def collect_maintenance_status(
    csv_path_s: str,
    artifact_path_s: str,
    *,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    APP_ROOT = ctx.get("APP_ROOT")
    _WF_LAST_REPORT_PATH = ctx.get("_WF_LAST_REPORT_PATH")
    _file_sig = ctx.get("_file_sig")
    _patch_meta_guard_payload = ctx.get("_patch_meta_guard_payload")
    _resolve_runtime_players_artifact_path = ctx.get("_resolve_runtime_players_artifact_path")

    _ml_trace_state = core_get_ml_trace_snapshot(st.session_state)
    out = core_collect_maintenance_status(
        csv_path_s,
        artifact_path_s,
        app_root=APP_ROOT,
        wf_report_path=_WF_LAST_REPORT_PATH,
        wf_last_report_at=core_get_wf_last_report_at(st.session_state),
        ml_trace_last=(_ml_trace_state.get("_ml_trace_last") or None),
        consistency_tol_pp=float(st.session_state.get("ml_consistency_tol_pp", 0.05) or 0.05),
        pre_delivery_report=(core_get_pre_delivery_report(st.session_state) or None),
        file_sig_fn=_file_sig,
        patch_meta_guard_payload_fn=_patch_meta_guard_payload,
        resolve_players_artifact_path_fn=_resolve_runtime_players_artifact_path,
    )
    _wf_resolved = str((out or {}).get("resolved_wf_last_report_at") or "").strip()
    if _wf_resolved:
        core_set_wf_last_report_at(st.session_state, _wf_resolved)
    return dict(out or {})


def run_auto_maintenance_now(
    csv_path_s: str,
    artifact_path_s: str,
    status: Optional[Dict[str, Any]] = None,
    *,
    force_full: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    """Executa manutenção automática (modelo/params/walk-forward) em uma tacada."""
    APP_ROOT = ctx.get("APP_ROOT")
    _paths_for_ml = ctx.get("_paths_for_ml")
    _players_artifact_output_path = ctx.get("_players_artifact_output_path")
    _file_sig = ctx.get("_file_sig")
    _load_fitted_params_cached = ctx.get("_load_fitted_params_cached")
    _run_walkforward_cached = ctx.get("_run_walkforward_cached")
    _run_layer_validation_cached = ctx.get("_run_layer_validation_cached")
    _generate_and_persist_ml_confidence_baseline = ctx.get("_generate_and_persist_ml_confidence_baseline")
    _generate_and_persist_team_state_artifact = ctx.get("_generate_and_persist_team_state_artifact")
    _load_team_state_artifact_cached = ctx.get("_load_team_state_artifact_cached")
    _backup_file_if_exists = ctx.get("_backup_file_if_exists")
    _WF_LAST_REPORT_PATH = ctx.get("_WF_LAST_REPORT_PATH")
    _LV_LAST_REPORT_PATH = ctx.get("_LV_LAST_REPORT_PATH")

    stt = status if isinstance(status, dict) else collect_maintenance_status(csv_path_s, artifact_path_s, st=st, ctx=ctx)
    need_retrain_model = bool(stt.get("need_retrain_model", False))
    need_retrain_params = bool(stt.get("need_retrain_params", False))
    need_walkforward = bool(stt.get("need_walkforward", False))
    need_patch_recalibration = bool(stt.get("need_patch_recalibration", False))

    daily_csv, hist_csvs, art_cfg = _paths_for_ml(APP_ROOT)
    art_target = str(art_cfg or artifact_path_s or (APP_ROOT / "ml_artifact.json"))
    _players_artifact_target = _players_artifact_output_path(str(art_target))
    csv_list = _build_training_csv_list(daily_csv, hist_csvs)

    _prefs = core_resolve_maintenance_preferences(st.session_state, force_full=force_full)
    cmd_timeout_s = int(_prefs.get("cmd_timeout_s", 1200) or 1200)
    params_timeout_s = int(_prefs.get("params_timeout_s", max(cmd_timeout_s, 14000)) or max(cmd_timeout_s, 14000))
    fast_mode_user_pref = bool(_prefs.get("fast_mode_user_pref", True))
    fast_mode = bool(_prefs.get("fast_mode", fast_mode_user_pref))
    run_layer_validation = bool(_prefs.get("run_layer_validation", (not fast_mode_user_pref)))
    policy_force_full = bool(_prefs.get("policy_force_full", False))
    policy_force_fast = bool(_prefs.get("policy_force_fast", False))
    fast_new_games_max = int(_prefs.get("fast_new_games_max", 30) or 30)
    t_all0 = time.perf_counter()
    steps: List[Dict[str, Any]] = []
    _progress_total = 9

    def _emit_progress(payload: Dict[str, Any]) -> None:
        if not callable(progress_cb):
            return
        try:
            progress_cb(dict(payload))
        except Exception:
            pass

    def _emit_running(step_name: str) -> None:
        _emit_progress(
            {
                "event": "running",
                "step": str(step_name),
                "idx": int(min(len(steps) + 1, _progress_total)),
                "total": int(_progress_total),
                "status": "rodando",
            }
        )

    def _mark_step(
        name: str,
        *,
        ok: bool,
        t0: float,
        summary: str = "",
        critical: bool = True,
        skipped: bool = False,
    ) -> None:
        steps.append(
            {
                "step": str(name),
                "ok": bool(ok),
                "critical": bool(critical),
                "skipped": bool(skipped),
                "elapsed_s": float(max(0.0, time.perf_counter() - float(t0))),
                "summary": str(summary or "").strip(),
            }
        )
        _emit_progress(
            {
                "event": "step_done",
                "step": str(name),
                "idx": int(min(len(steps), _progress_total)),
                "total": int(_progress_total),
                "status": ("Pulada" if bool(skipped) else ("OK" if bool(ok) else "Erro")),
                "ok": bool(ok),
                "skipped": bool(skipped),
                "critical": bool(critical),
                "summary": str(summary or "").strip(),
            }
        )

    _emit_progress(
        {
            "event": "start",
            "step": "init",
            "idx": 0,
            "total": int(_progress_total),
            "status": "iniciando",
        }
    )

    if not csv_list:
        t0 = time.perf_counter()
        _mark_step(
            "resolver_csvs_treino",
            ok=False,
            t0=t0,
            summary="Sem CSV válido para manutenção automática.",
            critical=True,
        )
        _emit_progress(
            {
                "event": "done",
                "step": "resolver_csvs_treino",
                "idx": 1,
                "total": int(_progress_total),
                "status": "erro",
            }
        )
        return {
            "ok": False,
            "error": "Sem CSV válido para manutenção automática.",
            "log": "",
            "steps": steps,
            "duration_s": float(max(0.0, time.perf_counter() - t_all0)),
        }

    need_retrain_params, need_walkforward = core_seed_maintenance_needs(
        need_retrain_model,
        need_retrain_params,
        need_walkforward,
    )

    logs: List[str] = []
    all_ok = True
    rep_wf_auto: Optional[Dict[str, Any]] = None
    _policy_profile = "fast"
    _policy_reason = "default_fast"
    _model_new_games: Optional[int] = None
    _model_backfill_suspected = False
    _model_had_full_retrain = False
    _model_updated_this_cycle = False

    def _params_output_valid(path_like: str | Path) -> bool:
        return bool(core_params_output_valid_path(path_like))

    def _run_fit_params_once(csvs: List[str], *, speed_profile: str) -> Dict[str, Any]:
        _fit_runtime_paths = core_build_fit_params_runtime_paths(
            app_root=APP_ROOT,
            scope=("auto_fast" if str(speed_profile).strip().lower() == "fast" else "auto_full"),
        )
        cmd = core_build_fit_params_cmd(
            csvs,
            artifact_path=art_target,
            players_artifact_path=_players_artifact_target,
            out_path=(APP_ROOT / "params_fitted.json"),
            speed_profile=str(speed_profile),
            progress_path=_fit_runtime_paths.get("progress_path"),
            checkpoint_dir=_fit_runtime_paths.get("checkpoint_dir"),
            run_label=_fit_runtime_paths.get("run_label"),
        )

        def _hb_params(elapsed_s: float) -> None:
            _emit_progress(
                {
                    "event": "running",
                    "step": "params_fit",
                    "idx": int(min(len(steps) + 1, _progress_total)),
                    "total": int(_progress_total),
                    "status": f"rodando ({int(max(0.0, elapsed_s))}s)",
                }
            )

        cp = _run_cmd(
            cmd,
            timeout_s=params_timeout_s,
            heartbeat_cb=_hb_params,
            heartbeat_interval_s=2.0,
        )
        ok_rc = int(cp.returncode) == 0
        ok_out = _params_output_valid(APP_ROOT / "params_fitted.json")
        areal_fix = ""
        if bool(ok_rc and ok_out):
            try:
                _ok_fix, _msg_fix = core_postprocess_params_file_tuning(
                    APP_ROOT,
                    sanitize_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
                )
                if _ok_fix:
                    areal_fix = str(_msg_fix or "")
            except (OSError, TypeError, ValueError):
                areal_fix = ""
        return {
            "cp": cp,
            "ok_rc": bool(ok_rc),
            "ok_out": bool(ok_out),
            "ok": bool(ok_rc and ok_out),
            "n_csv": len(csvs or []),
            "speed": str(speed_profile),
            "areal_fix": str(areal_fix),
        }

    def _resolve_active_fit_scale_state() -> Dict[str, float]:
        fit_scale_state = core_get_fit_scale_state(st.session_state)
        params_path = APP_ROOT / "params_fitted.json"
        if not params_path.exists():
            return fit_scale_state
        try:
            fit_live = _load_fitted_params_cached(_file_sig(str(params_path)))
        except Exception:
            return fit_scale_state
        best = (fit_live.get("best") or {}) if isinstance(fit_live, dict) else {}
        team_scale = _fnum(best.get("team_scale"))
        players_scale = _fnum(best.get("players_scale"))
        if team_scale is None or players_scale is None:
            return fit_scale_state
        core_set_fit_base_scales(
            st.session_state,
            team_scale=float(team_scale),
            players_scale=float(players_scale),
        )
        return core_get_fit_scale_state(st.session_state)

    try:
        if bool(need_retrain_model):
            _model_ok = False
            _new_games_inc: Optional[int] = None
            _t_inc = time.perf_counter()
            if bool(force_full):
                # "Manutenção completa": pular a incremental e ir DIRETO pro full
                # retrain. A incremental com 0 jogos novos "tem sucesso" (_model_ok
                # True) e pulava o full retrain — que é o ÚNICO passo que reconstrói
                # o players_artifact e refaz o modelo. Sem isso, o players ficava em
                # drift e o params atrás do artifact, e os alertas nunca limpavam
                # por mais que se clicasse. Deixando _model_ok=False, o full roda.
                _mark_step(
                    "modelo_incremental",
                    ok=True,
                    t0=_t_inc,
                    summary="Pulado (Manutenção completa: full retrain forçado).",
                    critical=False,
                    skipped=True,
                )
            else:
                _emit_running("modelo_incremental")
                if Path(str(art_target)).exists() and Path(str(daily_csv or "")).exists():
                    try:
                        cmd_inc = core_build_incremental_update_cmd(
                            artifact_path=str(art_target),
                            csv_path=str(daily_csv),
                            players_path=str(_players_artifact_target),
                        )
                        cp_inc = _run_cmd(cmd_inc, timeout_s=cmd_timeout_s)
                        logs.append("[modelo-incremental]\n" + (cp_inc.stdout or "") + ("\n" + cp_inc.stderr if cp_inc.stderr else ""))
                        if int(cp_inc.returncode) == 0:
                            _model_ok = True
                            _model_updated_this_cycle = True
                            try:
                                _inc_info = core_parse_incremental_update_stdout(str(cp_inc.stdout or ""))
                            except (TypeError, ValueError):
                                _inc_info = {}
                            if not isinstance(_inc_info, dict):
                                _inc_info = {}
                            _new_games_val = _fnum(_inc_info.get("new_games"))
                            if _new_games_val is not None:
                                _new_games_inc = int(_new_games_val)
                                _model_new_games = int(_new_games_inc)
                                _model_updated_this_cycle = bool(int(_new_games_inc) > 0)
                            if _inc_info.get("backfill_suspected") is not None:
                                _model_backfill_suspected = bool(_inc_info.get("backfill_suspected"))
                        _mark_step(
                            "modelo_incremental",
                            ok=int(cp_inc.returncode) == 0,
                            t0=_t_inc,
                            summary=(
                                f"rc={int(cp_inc.returncode)}"
                                + (f" | new_games={_new_games_inc}" if _new_games_inc is not None else "")
                            ),
                            critical=False,
                        )
                    except Exception as _exc_inc:
                        logs.append(f"[modelo-incremental][erro] {type(_exc_inc).__name__}: {_exc_inc}")
                        _mark_step(
                            "modelo_incremental",
                            ok=False,
                            t0=_t_inc,
                            summary=f"{type(_exc_inc).__name__}: {_exc_inc}",
                            critical=False,
                        )
                else:
                    _mark_step(
                        "modelo_incremental",
                        ok=True,
                        t0=_t_inc,
                        summary="Pulado (artifact/csv diário ausente).",
                        critical=False,
                        skipped=True,
                    )

                _after_inc = core_adjust_needs_after_incremental(
                    model_ok=bool(_model_ok),
                    new_games_inc=_new_games_inc,
                    need_patch_recalibration=bool(need_patch_recalibration),
                    need_retrain_params=bool(need_retrain_params),
                    need_walkforward=bool(need_walkforward),
                )
                need_retrain_params = bool(_after_inc.get("need_retrain_params", need_retrain_params))
                need_walkforward = bool(_after_inc.get("need_walkforward", need_walkforward))
                if str(_after_inc.get("log_message") or "").strip():
                    logs.append(str(_after_inc.get("log_message") or ""))

            if not _model_ok:
                _t_model = time.perf_counter()
                _emit_running("modelo_full_retrain")
                _model_had_full_retrain = True
                _bk_art = _backup_file_if_exists(art_target, tag="pre_retrain_model")
                if _bk_art:
                    logs.append(f"[backup] artifact => {_bk_art}")
                _train_cfg = core_resolve_train_offline_reuse_params(art_target)
                cmd_m = core_build_train_offline_cmd(
                    csv_list,
                    out_path=art_target,
                    k=float(_train_cfg.get("k", 26.0) or 26.0),
                    scale=float(_train_cfg.get("scale", 400.0) or 400.0),
                    calibrate=str(_train_cfg.get("calibrate", "auto") or "auto"),
                    # Deploy validado usa holdout 30 (tools/atualizar_modelo.py). Sem isso,
                    # o train_offline cai no default 180 e congela os ratings 6 meses atras
                    # (cutoff sem nenhum jogo da season atual) -> nota despenca (7.74->4.97
                    # medido em 2026-06-10). NUNCA omitir aqui.
                    holdout_days=30,
                    core=(str(_train_cfg["core"]) if _train_cfg.get("core") else None),
                    whr_w2=(float(_train_cfg["whr_w2"]) if _train_cfg.get("whr_w2") is not None else None),
                    # players layer segue o artefato atual (Dota deploya sem players — P3)
                    fit_players=bool(_train_cfg.get("fit_players", True)),
                    players_out=_players_artifact_target,
                )
                cp_m = _run_cmd(cmd_m, timeout_s=cmd_timeout_s)
                logs.append("[modelo]\n" + (cp_m.stdout or "") + ("\n" + cp_m.stderr if cp_m.stderr else ""))
                _ok_model = int(cp_m.returncode) == 0
                if bool(_ok_model):
                    _model_updated_this_cycle = True
                _mark_step(
                    "modelo_full_retrain",
                    ok=_ok_model,
                    t0=_t_model,
                    summary=f"rc={int(cp_m.returncode)} | csvs={len(csv_list)}",
                    critical=True,
                )
                if int(cp_m.returncode) != 0:
                    all_ok = False
            else:
                _t_model = time.perf_counter()
                _mark_step(
                    "modelo_full_retrain",
                    ok=True,
                    t0=_t_model,
                    summary="Pulado (incremental bem-sucedido).",
                    critical=True,
                    skipped=True,
                )
        else:
            _t_inc = time.perf_counter()
            _mark_step(
                "modelo_incremental",
                ok=True,
                t0=_t_inc,
                summary="Pulado (sem gatilho de modelo).",
                critical=False,
                skipped=True,
            )
            _t_model = time.perf_counter()
            _mark_step(
                "modelo_full_retrain",
                ok=True,
                t0=_t_model,
                summary="Pulado (sem gatilho de modelo).",
                critical=True,
                skipped=True,
            )

        try:
            from app_core.maintenance_runtime_core import read_artifact_summary_path as core_read_artifact_summary_path
            _art_sum = dict(core_read_artifact_summary_path(art_target))
            _af_ng = _art_sum.get("added_games_last")
            if (_model_new_games is None) and (_af_ng is not None):
                _model_new_games = int(_af_ng)
            _af_backfill = _art_sum.get("backfill_suspected")
            if _af_backfill is not None:
                _model_backfill_suspected = bool(_af_backfill)
        except (OSError, TypeError, ValueError):
            pass

        _force_params = core_force_params_after_model_refresh(
            need_retrain_model=bool(need_retrain_model and all_ok),
            need_retrain_params=bool(need_retrain_params),
            model_updated_this_cycle=bool(_model_updated_this_cycle),
        )
        need_retrain_params = bool(_force_params.get("need_retrain_params", need_retrain_params))
        if str(_force_params.get("log_message") or "").strip():
            logs.append(str(_force_params.get("log_message") or ""))

        _policy = core_decide_maintenance_policy(
            policy_force_full=bool(policy_force_full),
            policy_force_fast=bool(policy_force_fast),
            need_retrain_model=bool(need_retrain_model),
            need_retrain_params=bool(need_retrain_params),
            need_walkforward=bool(need_walkforward),
            need_patch_recalibration=bool(need_patch_recalibration),
            model_backfill_suspected=bool(_model_backfill_suspected),
            model_had_full_retrain=bool(_model_had_full_retrain),
            model_new_games=_model_new_games,
            fast_new_games_max=int(fast_new_games_max),
            fast_mode_user_pref=bool(fast_mode_user_pref),
        )
        _policy_profile = str(_policy.get("profile") or "fast")
        _policy_reason = str(_policy.get("reason") or "default_fast")
        fast_mode = bool(_policy.get("fast_mode", (_policy_profile == "fast")))
        run_layer_validation = bool(st.session_state.get("auto_maint_run_layer_validation", (not fast_mode)))
        logs.append(
            "[policy]\n"
            + f"profile={_policy_profile} | reason={_policy_reason}"
            + f" | fast_user_pref={str(bool(fast_mode_user_pref)).lower()}"
            + f" | new_games={str(_model_new_games) if _model_new_games is not None else '-'}"
            + f" | backfill={str(bool(_model_backfill_suspected)).lower()}"
            + f" | need_patch={str(bool(need_patch_recalibration)).lower()}"
        )
        _t_pol = time.perf_counter()
        _emit_running("policy_fast_full")
        _mark_step(
            "policy_fast_full",
            ok=True,
            t0=_t_pol,
            summary=f"profile={_policy_profile} | reason={_policy_reason}",
            critical=False,
        )

        if bool(need_retrain_params) and all_ok:
            _t_params = time.perf_counter()
            _emit_running("params_fit")
            _bk_prm = _backup_file_if_exists(APP_ROOT / "params_fitted.json", tag="pre_retrain_params")
            if _bk_prm:
                logs.append(f"[backup] params => {_bk_prm}")
            _params_primary = core_choose_params_primary_csvs(
                csv_list,
                daily_csv=daily_csv,
                fast_mode=bool(fast_mode),
            )
            params_csv_primary = list(_params_primary.get("csvs") or list(csv_list))
            params_primary_mode = str(_params_primary.get("mode") or "full_csvs_fast")

            _p1_speed = "fast" if bool(fast_mode) else "full"
            _p1 = _run_fit_params_once(params_csv_primary, speed_profile=_p1_speed)
            _cp1 = _p1.get("cp")
            logs.append(
                "[params][primary]\n"
                + f"mode={params_primary_mode} | n_csv={int(_p1.get('n_csv') or 0)} | speed={_p1.get('speed')}\n"
                + (str(getattr(_cp1, "stdout", "") or ""))
                + (("\n" + str(getattr(_cp1, "stderr", ""))) if str(getattr(_cp1, "stderr", "") or "").strip() else "")
            )
            _ok_params = bool(_p1.get("ok", False))
            _used_mode = params_primary_mode
            _fallback_used = False
            _p2: Dict[str, Any] = {}
            _need_fallback = (
                (not _ok_params)
                and (
                    bool(fast_mode)
                    or (len(params_csv_primary) != len(csv_list))
                    or (str(_p1.get("speed") or "").strip().lower() != "full")
                )
            )
            if _need_fallback:
                _fallback_used = True
                _p2 = _run_fit_params_once(list(csv_list), speed_profile="full")
                _cp2 = _p2.get("cp")
                logs.append(
                    "[params][fallback]\n"
                    + f"mode=full_csvs_fallback | n_csv={int(_p2.get('n_csv') or 0)} | speed={_p2.get('speed')}\n"
                    + (str(getattr(_cp2, "stdout", "") or ""))
                    + (("\n" + str(getattr(_cp2, "stderr", ""))) if str(getattr(_cp2, "stderr", "") or "").strip() else "")
                )
                _ok_params = bool(_p2.get("ok", False))
                _used_mode = "full_csvs_fallback"

            _mark_step(
                "params_fit",
                ok=_ok_params,
                t0=_t_params,
                summary=(
                    f"mode={_used_mode} | csvs_primary={len(params_csv_primary)} | csvs_full={len(csv_list)}"
                    + (f" | fallback={str(_fallback_used).lower()}")
                    + (
                        f" | areal={(_p2 if _fallback_used else _p1).get('areal_fix') or '-'!s}"
                        if isinstance((_p2 if _fallback_used else _p1), dict)
                        else ""
                    )
                ),
                critical=True,
            )
            if not _ok_params:
                all_ok = False
        else:
            _t_params = time.perf_counter()
            _mark_step(
                "params_fit",
                ok=True,
                t0=_t_params,
                summary="Pulado (sem pendência ou bloqueado por falha anterior).",
                critical=True,
                skipped=True,
            )

        _fit_scale_state = _resolve_active_fit_scale_state()

        if bool(need_walkforward) and all_ok:
            _t_wf = time.perf_counter()
            _emit_running("walk_forward")
            rep = _run_walkforward_cached(
                tuple(csv_list),
                int(st.session_state.get("wf_min_train_games", 1500) or 1500),
                float(st.session_state.get("wf_k", 26.0) or 26.0),
                float(st.session_state.get("wf_scale", 400.0) or 400.0),
                str(art_target),
                str(_players_artifact_target),
                float(_fit_scale_state.get("team_scale", 1.0) or 1.0),
                float(_fit_scale_state.get("players_scale", 1.0) or 1.0),
                10,
                int(st.session_state.get("diag_min_games", 250) or 250),
                str(APP_ROOT / "params_fitted.json"),
            )
            core_set_wf_last_report(st.session_state, rep)
            try:
                core_write_json_file_atomic(_WF_LAST_REPORT_PATH, rep if isinstance(rep, dict) else {"error": "wf_report_invalido"})
            except (OSError, TypeError, ValueError):
                pass
            logs.append("[walk-forward]\n" + json.dumps(rep, ensure_ascii=False, indent=2))
            rep_wf_auto = rep if isinstance(rep, dict) else None
            _ok_wf = not (isinstance(rep, dict) and rep.get("error"))
            _mark_step(
                "walk_forward",
                ok=_ok_wf,
                t0=_t_wf,
                summary=(
                    f"error={rep.get('error')}" if isinstance(rep, dict) and rep.get("error") else "ok"
                ),
                critical=True,
            )
            if not _ok_wf:
                all_ok = False
        else:
            _t_wf = time.perf_counter()
            _mark_step(
                "walk_forward",
                ok=True,
                t0=_t_wf,
                summary="Pulado (sem pendência ou bloqueado por falha anterior).",
                critical=True,
                skipped=True,
            )

        if all_ok:
            _t_lv = time.perf_counter()
            _emit_running("layer_validation")
            try:
                rep_lv_auto = core_get_lv_last_report(st.session_state)
                if not (isinstance(rep_lv_auto, dict) and not rep_lv_auto.get("error")):
                    rep_lv_auto = core_read_json_file_safe(_LV_LAST_REPORT_PATH)
                    if isinstance(rep_lv_auto, dict) and (not rep_lv_auto.get("error")):
                        core_set_lv_last_report(st.session_state, rep_lv_auto)
                if run_layer_validation and not (isinstance(rep_lv_auto, dict) and not rep_lv_auto.get("error")):
                    _fit_live_auto = _load_fitted_params_cached(_file_sig(str(APP_ROOT / "params_fitted.json"))) if (APP_ROOT / "params_fitted.json").exists() else {}
                    _best_auto = (_fit_live_auto.get("best") or {}) if isinstance(_fit_live_auto, dict) else {}
                    _ts_auto = float(_best_auto.get("team_scale") or _fit_scale_state.get("team_scale", 1.0) or 1.0)
                    _ps_auto = float(_best_auto.get("players_scale") or _fit_scale_state.get("players_scale", 1.0) or 1.0)
                    rep_lv_auto = _run_layer_validation_cached(
                        tuple(csv_list),
                        str(art_target),
                        str(_players_artifact_target),
                        int(st.session_state.get("lv_min_train_games", 1200) or 1200),
                        float(_ts_auto),
                        float(_ps_auto),
                        int(st.session_state.get("lv_min_league_games", 400) or 400),
                    )
                    core_set_lv_last_report(st.session_state, rep_lv_auto)
                    try:
                        core_write_json_file_atomic(_LV_LAST_REPORT_PATH, rep_lv_auto if isinstance(rep_lv_auto, dict) else {"error": "lv_report_invalido"})
                    except (OSError, TypeError, ValueError):
                        pass
                    logs.append("[layer-validation]\n" + json.dumps(rep_lv_auto, ensure_ascii=False, indent=2)[:12000])
                    _mark_step(
                        "layer_validation",
                        ok=not (isinstance(rep_lv_auto, dict) and rep_lv_auto.get("error")),
                        t0=_t_lv,
                        summary="recalculado",
                        critical=False,
                    )
                elif isinstance(rep_lv_auto, dict) and not rep_lv_auto.get("error"):
                    _mark_step(
                        "layer_validation",
                        ok=True,
                        t0=_t_lv,
                        summary="Pulado (cache sessão/disco válido).",
                        critical=False,
                        skipped=True,
                    )
                else:
                    _mark_step(
                        "layer_validation",
                        ok=True,
                        t0=_t_lv,
                        summary=("Pulado (modo rápido ativo)." if fast_mode else "Pulado (sem insumo)."),
                        critical=False,
                        skipped=True,
                    )
                if rep_wf_auto is None:
                    rep_wf_auto = core_get_wf_last_report(st.session_state)
                _t_cb = float("nan")
                if rep_wf_auto is None:
                    _wf_disk = core_read_json_file_safe(_WF_LAST_REPORT_PATH)
                    rep_wf_auto = _wf_disk if (isinstance(_wf_disk, dict) and _wf_disk) else None
                _fit_live_auto2 = _load_fitted_params_cached(_file_sig(str(APP_ROOT / "params_fitted.json"))) if (APP_ROOT / "params_fitted.json").exists() else {}
                _t_cb = time.perf_counter()
                _emit_running("confidence_baseline")
                if isinstance(rep_lv_auto, dict) and not rep_lv_auto.get("error"):
                    rep_cb_auto = _generate_and_persist_ml_confidence_baseline(
                        root=APP_ROOT,
                        lv_report=rep_lv_auto if isinstance(rep_lv_auto, dict) else None,
                        wf_report=rep_wf_auto if isinstance(rep_wf_auto, dict) else None,
                        params_fit=_fit_live_auto2,
                    )
                    if isinstance(rep_cb_auto, dict) and not rep_cb_auto.get("error"):
                        core_set_ml_confidence_baseline_last_report(st.session_state, rep_cb_auto)
                        logs.append("[conf-baseline]\n" + json.dumps({"ok": True, "leagues": len((rep_cb_auto.get("by_league") or {}))}, ensure_ascii=False))
                        _mark_step(
                            "confidence_baseline",
                            ok=True,
                            t0=_t_cb,
                            summary=f"leagues={len((rep_cb_auto.get('by_league') or {}))}",
                            critical=True,
                        )
                    else:
                        logs.append("[conf-baseline]\n" + json.dumps(rep_cb_auto if isinstance(rep_cb_auto, dict) else {"error": "unknown"}, ensure_ascii=False, indent=2))
                        _mark_step(
                            "confidence_baseline",
                            ok=False,
                            t0=_t_cb,
                            summary=str((rep_cb_auto or {}).get("error") if isinstance(rep_cb_auto, dict) else "unknown"),
                            critical=True,
                        )
                        all_ok = False
                else:
                    _mark_step(
                        "confidence_baseline",
                        ok=True,
                        t0=_t_cb,
                        summary="Pulado (sem layer-validation válido; não bloqueante no modo rápido).",
                        critical=True,
                        skipped=True,
                    )
            except Exception as exc_cb:
                logs.append(f"[conf-baseline][erro] {type(exc_cb).__name__}: {exc_cb}")
                _mark_step(
                    "confidence_baseline",
                    ok=False,
                    t0=time.perf_counter(),
                    summary=f"{type(exc_cb).__name__}: {exc_cb}",
                    critical=True,
                )
                all_ok = False
            _t_sa = float("nan")
            try:
                _t_sa = time.perf_counter()
                _emit_running("state_artifact")
                _sa_box: Dict[str, Any] = {"rep": None, "err": None}

                def _sa_worker() -> None:
                    try:
                        _sa_box["rep"] = _generate_and_persist_team_state_artifact(
                            root=APP_ROOT,
                            csv_paths=csv_list,
                            keep_playoffs=None,
                        )
                    except Exception as _exc_sa_worker:
                        _sa_box["err"] = _exc_sa_worker

                _thr_sa = threading.Thread(target=_sa_worker, name="state_artifact_worker", daemon=True)
                _thr_sa.start()
                while _thr_sa.is_alive():
                    _emit_progress(
                        {
                            "event": "running",
                            "step": "state_artifact",
                            "idx": int(min(len(steps) + 1, _progress_total)),
                            "total": int(_progress_total),
                            "status": f"rodando ({int(max(0.0, time.perf_counter() - _t_sa))}s)",
                        }
                    )
                    _thr_sa.join(timeout=2.0)
                _thr_sa.join(timeout=0.1)
                if _sa_box.get("err") is not None:
                    raise _sa_box["err"]
                rep_sa_auto = _sa_box.get("rep")
                if isinstance(rep_sa_auto, dict) and not rep_sa_auto.get("error"):
                    core_set_team_state_artifact_last_report(st.session_state, rep_sa_auto)
                    _clear_cached_team_state = getattr(_load_team_state_artifact_cached, "clear", None)
                    if callable(_clear_cached_team_state):
                        _clear_cached_team_state()
                    _sa_meta_auto = (rep_sa_auto.get("meta") or {}) if isinstance(rep_sa_auto.get("meta"), dict) else {}
                    logs.append(
                        "[state-artifact]\n"
                        + json.dumps(
                            {
                                "ok": True,
                                "teams": int(_sa_meta_auto.get("teams") or 0),
                                "games_used": int(_sa_meta_auto.get("games_used") or 0),
                                "update_mode": str(_sa_meta_auto.get("update_mode") or ""),
                            },
                            ensure_ascii=False,
                        )
                    )
                    _mark_step(
                        "state_artifact",
                        ok=True,
                        t0=_t_sa,
                        summary=(
                            f"teams={int(_sa_meta_auto.get('teams') or 0)}"
                            f" | games_used={int(_sa_meta_auto.get('games_used') or 0)}"
                        ),
                        critical=False,
                    )
                else:
                    logs.append(
                        "[state-artifact]\n"
                        + json.dumps(
                            rep_sa_auto if isinstance(rep_sa_auto, dict) else {"error": "unknown"},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    _mark_step(
                        "state_artifact",
                        ok=False,
                        t0=_t_sa,
                        summary=str((rep_sa_auto or {}).get("error") if isinstance(rep_sa_auto, dict) else "unknown"),
                        critical=False,
                    )
                    all_ok = False
            except Exception as exc_sa:
                logs.append(f"[state-artifact][erro] {type(exc_sa).__name__}: {exc_sa}")
                _mark_step(
                    "state_artifact",
                    ok=False,
                    t0=time.perf_counter(),
                    summary=f"{type(exc_sa).__name__}: {exc_sa}",
                    critical=False,
                )
                all_ok = False

        else:
            _t_lv = time.perf_counter()
            _mark_step(
                "layer_validation",
                ok=True,
                t0=_t_lv,
                summary="Pulado (bloqueado por falha crítica anterior).",
                critical=False,
                skipped=True,
            )
            _t_cb = time.perf_counter()
            _mark_step(
                "confidence_baseline",
                ok=True,
                t0=_t_cb,
                summary="Pulado (bloqueado por falha crítica anterior).",
                critical=True,
                skipped=True,
            )
            _t_sa = time.perf_counter()
            _mark_step(
                "state_artifact",
                ok=True,
                t0=_t_sa,
                summary="Pulado (bloqueado por falha crítica anterior).",
                critical=False,
                skipped=True,
            )

        # Alarme de qualidade pós-retrain: se o modelo mudou neste ciclo, mede a nota
        # da season atual vs baseline (ops/monitoring/ml_quality_baseline.json). Sem isso,
        # uma degradação silenciosa (ex.: incidente holdout-180, nota 7.74→4.97 por 4 dias)
        # passa batida — o guard de artefato checa completude, não qualidade.
        _t_qc = time.perf_counter()
        if bool(_model_updated_this_cycle):
            _emit_running("qualidade_alarme")
            cmd_qc = [sys.executable, str(APP_ROOT / "tools" / "ml_quality_check.py"), "--artifact", str(art_target)]
            cp_qc = _run_cmd(cmd_qc, timeout_s=cmd_timeout_s)
            logs.append("[qualidade-alarme]\n" + (cp_qc.stdout or "") + ("\n" + cp_qc.stderr if cp_qc.stderr else ""))
            _ok_qc = int(cp_qc.returncode) == 0
            _qc_summary = f"rc={int(cp_qc.returncode)}"
            try:
                _qc_last = json.loads((APP_ROOT / "ops" / "monitoring" / "ml_quality_last_check.json").read_text(encoding="utf-8"))
                _qc_res = (_qc_last.get("result") or {}) if isinstance(_qc_last, dict) else {}
                if _qc_res.get("verdict"):
                    _qc_summary = f"{_qc_res.get('verdict')}: {str(_qc_res.get('reason', ''))[:140]}"
            except (OSError, ValueError):
                pass
            _mark_step("qualidade_alarme", ok=_ok_qc, t0=_t_qc, summary=_qc_summary, critical=False)
            if not _ok_qc:
                all_ok = False
        else:
            _mark_step(
                "qualidade_alarme",
                ok=True,
                t0=_t_qc,
                summary="Pulado (modelo não mudou neste ciclo).",
                critical=False,
                skipped=True,
            )
    except Exception as exc:
        all_ok = False
        logs.append(f"[erro] {type(exc).__name__}: {exc}")
        _t_guard = time.perf_counter()
        _mark_step(
            "manutencao_guard",
            ok=False,
            t0=_t_guard,
            summary=f"{type(exc).__name__}: {exc}",
            critical=True,
        )

    full_log = "\n\n".join(logs).strip()
    core_set_quick_maint_last_log(st.session_state, full_log)
    _emit_progress(
        {
            "event": "done",
            "step": "finish",
            "idx": int(min(len(steps), _progress_total)),
            "total": int(_progress_total),
            "status": ("ok" if bool(all_ok) else "erro"),
        }
    )
    return {
        "ok": bool(all_ok),
        "log": full_log,
        "error": (None if all_ok else "manutenção com falhas"),
        "steps": steps,
        "duration_s": float(max(0.0, time.perf_counter() - t_all0)),
        "command_timeout_s": int(cmd_timeout_s),
        "params_timeout_s": int(params_timeout_s),
        "fast_mode": bool(fast_mode),
        "run_layer_validation": bool(run_layer_validation),
        "policy_profile": str(_policy_profile),
        "policy_reason": str(_policy_reason),
        "model_new_games": (_model_new_games if _model_new_games is not None else None),
        "model_backfill_suspected": bool(_model_backfill_suspected),
    }


def ensure_csv_runtime_ready(
    csv_path: str,
    *,
    hist_csvs: Optional[List[str]] = None,
    artifact_path: Optional[str] = None,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    _file_sig = ctx.get("_file_sig")
    _resolve_players_artifact_path = ctx.get("_resolve_players_artifact_path")
    _FIT_PARAMS_PATH = ctx.get("_FIT_PARAMS_PATH")
    _TEAM_STATE_ARTIFACT_PATH = ctx.get("_TEAM_STATE_ARTIFACT_PATH")
    _ML_CONFIDENCE_BASELINE_PATH = ctx.get("_ML_CONFIDENCE_BASELINE_PATH")
    _csv_header_check = ctx.get("_csv_header_check")
    _csv_diag = ctx.get("_csv_diag")
    _list_teams = ctx.get("_list_teams")
    _load_team_games = ctx.get("_load_team_games")
    _load_player_rows = ctx.get("_load_player_rows")
    APP_ROOT = ctx.get("APP_ROOT")

    csv_path = str(csv_path or "").strip()
    hist_csvs = [str(p) for p in (hist_csvs or []) if str(p or "").strip()]
    artifact_path = str(artifact_path or "").strip()

    sig = _file_sig(csv_path)
    _force = core_consume_csv_prepare_refresh(st.session_state)
    _last_sig_raw = st.session_state.get("_csv_prepare_last_sig")
    _last_sig = tuple(_last_sig_raw) if isinstance(_last_sig_raw, (list, tuple)) else None
    _last_report = st.session_state.get("_csv_prepare_report")
    if (not _force) and _last_sig == tuple(sig) and isinstance(_last_report, dict):
        _render_csv_prepare_report_summary(st, _last_report)
        return _last_report

    started = datetime.now().isoformat(timespec="seconds")
    t_all0 = time.perf_counter()
    report: Dict[str, Any] = {
        "csv_path": csv_path,
        "csv_sig": list(sig),
        "hist_csvs": hist_csvs,
        "artifact_path": artifact_path,
        "started_at": started,
        "steps": [],
        "errors": [],
        "ok": False,
        "show_details_once": True,
    }

    st.markdown("### Carregando novo CSV")
    st.caption("Preparando cache e validando etapas do CSV ativo (arquivo, cabeçalho, diagnóstico, times, team_games e players).")
    st.code(csv_path if csv_path else "(sem caminho)", language="text")
    _prog = st.progress(0, text="Iniciando...")
    _step_cards = st.empty()
    _summary_box = st.empty()

    _artifacts_rows: List[Dict[str, Any]] = []
    if artifact_path:
        _artifacts_rows.append({"Arquivo": "ml_artifact.json", "Caminho": artifact_path, "Existe": bool(Path(artifact_path).exists())})
        # players_artifact so entra no checklist se o deploy atual USA players
        # (meta.players_layer) — o Gabinete Dota deploya sem players ate a P3.
        _expects_players_row = True
        try:
            import json as _json_pl
            _meta_pl = (_json_pl.loads(Path(artifact_path).read_text(encoding="utf-8")).get("meta") or {})
            _expects_players_row = bool(_meta_pl.get("players_layer"))
        except (OSError, ValueError, AttributeError):
            pass
        if _expects_players_row:
            _artifacts_rows.append(
                {
                    "Arquivo": "players_artifact.json",
                    "Caminho": str(_resolve_players_artifact_path(artifact_path, APP_ROOT)),
                    "Existe": bool(Path(str(_resolve_players_artifact_path(artifact_path, APP_ROOT))).exists()),
                }
            )
        _artifacts_rows.append(
            {
                "Arquivo": "params_fitted.json",
                "Caminho": str(_FIT_PARAMS_PATH),
                "Existe": bool(Path(str(_FIT_PARAMS_PATH)).exists()),
            }
        )
        _artifacts_rows.append(
            {
                "Arquivo": "team_state_artifact.json",
                "Caminho": str(_TEAM_STATE_ARTIFACT_PATH),
                "Existe": bool(Path(str(_TEAM_STATE_ARTIFACT_PATH)).exists()),
            }
        )
        _artifacts_rows.append(
            {
                "Arquivo": "ml_confidence_baseline.json",
                "Caminho": str(_ML_CONFIDENCE_BASELINE_PATH),
                "Existe": bool(Path(str(_ML_CONFIDENCE_BASELINE_PATH)).exists()),
            }
        )
    report["artifacts"] = _artifacts_rows
    report["_t_all0"] = float(t_all0)

    _steps_plan = [
        {"name": "Arquivo diário", "critical": True},
        {"name": "CSVs históricos selecionados", "critical": False},
        {"name": "Cabeçalho/colunas", "critical": True},
        {"name": "Diagnóstico rápido (rows/gameids)", "critical": False},
        {"name": "Times (lista/canonização)", "critical": True},
        {"name": "Team games (cache base)", "critical": True},
        {"name": "Players rows (cache Players/Campeões)", "critical": False},
        {"name": "Artifacts relacionados", "critical": False},
    ]
    _total_steps = max(1, len(_steps_plan))

    def _flush_live_cards(*, current_idx: Optional[int] = None, current_step: str = "") -> None:
        _render_csv_prepare_live_dashboard(
            st,
            _step_cards,
            report,
            steps_plan=_steps_plan,
            current_idx=current_idx,
            current_step=current_step,
        )

    def _run_step(step_idx: int, step_name: str, fn, *, critical: bool) -> Any:
        _prog.progress(int((step_idx / _total_steps) * 100), text=f"[{step_idx}/{_total_steps}] {step_name}...")
        _flush_live_cards(current_idx=step_idx, current_step=step_name)
        t0 = time.perf_counter()
        ok = False
        summary = ""
        payload: Any = None
        try:
            payload = fn()
            ok = True
            if isinstance(payload, str):
                summary = payload
            elif isinstance(payload, dict):
                summary = str(payload.get("summary") or payload.get("detail") or payload.get("msg") or "")
            elif payload is None:
                summary = "OK"
            else:
                summary = str(payload)
        except Exception as e:
            ok = False
            summary = f"{type(e).__name__}: {e}"
            if critical:
                report.setdefault("errors", []).append(f"{step_name}: {summary}")
        elapsed = time.perf_counter() - t0
        report["steps"].append(
            {
                "step": step_name,
                "ok": bool(ok),
                "critical": bool(critical),
                "elapsed_s": float(elapsed),
                "summary": summary,
            }
        )
        _flush_live_cards()
        if (not ok) and critical:
            raise RuntimeError(f"{step_name}: {summary}")
        return payload

    def _set_last_prepare_step_summary(summary: str) -> None:
        steps = report.get("steps")
        if not isinstance(steps, list) or not steps:
            return
        last_step = steps[-1]
        if not isinstance(last_step, dict):
            return
        last_step["summary"] = str(summary or "")
        _flush_live_cards()

    try:
        _run_step(
            1,
            "Arquivo diário",
            lambda: (
                (_ for _ in ()).throw(FileNotFoundError("CSV diário não encontrado"))
                if (not csv_path or not Path(csv_path).exists())
                else {
                    "summary": f"{Path(csv_path).name} | {_fmt_bytes_short(sig[1])} | mtime={datetime.fromtimestamp(int(sig[2])).isoformat(timespec='seconds') if int(sig[2] or 0) > 0 else '-'}"
                }
            ),
            critical=True,
        )

        def _hist_summary() -> str:
            if not hist_csvs:
                return "Nenhum histórico selecionado"
            ok_n = 0
            miss_n = 0
            for p in hist_csvs:
                if Path(str(p)).exists():
                    ok_n += 1
                else:
                    miss_n += 1
            if miss_n > 0:
                report.setdefault("errors", []).append(f"Históricos ausentes: {miss_n}")
            return f"{ok_n} encontrado(s), {miss_n} ausente(s)"

        _run_step(2, "CSVs históricos selecionados", _hist_summary, critical=False)

        hdr = _run_step(3, "Cabeçalho/colunas", lambda: _csv_header_check(sig), critical=True)
        if isinstance(hdr, dict):
            if bool(hdr.get("error")):
                raise RuntimeError(str(hdr.get("error")))
            miss = [str(x) for x in (hdr.get("missing_required") or [])]
            if miss:
                raise RuntimeError("Colunas obrigatórias ausentes: " + ", ".join(miss))
            _set_last_prepare_step_summary(f"{int(hdr.get('n_columns', 0))} colunas | obrigatórias OK")

        diag = _run_step(4, "Diagnóstico rápido (rows/gameids)", lambda: _csv_diag(sig), critical=False)
        if isinstance(diag, dict) and diag:
            report["csv_diag"] = diag
            _set_last_prepare_step_summary(
                f"{int(diag.get('rows',0))} linhas | {int(diag.get('unique_gameids',0))} gameids | "
                f"{int(diag.get('games_complete',0))} completos"
            )

        teams_local = _run_step(5, "Times (lista/canonização)", lambda: _list_teams(sig), critical=True)
        if isinstance(teams_local, list):
            report["teams_count"] = len(teams_local)
            _set_last_prepare_step_summary(f"{len(teams_local)} times")

        tg = _run_step(6, "Team games (cache base)", lambda: _load_team_games(sig), critical=True)
        if isinstance(tg, pd.DataFrame):
            report["team_games_shape"] = [int(tg.shape[0]), int(tg.shape[1])]
            _g = int(tg["gameid"].nunique()) if "gameid" in tg.columns else 0
            _lg = int(tg["league"].dropna().astype(str).nunique()) if "league" in tg.columns else 0
            _set_last_prepare_step_summary(f"{int(tg.shape[0])} rows | {_g} jogos | {_lg} ligas")

        dpl = _run_step(7, "Players rows (cache Players/Campeões)", lambda: _load_player_rows(sig), critical=False)
        if isinstance(dpl, pd.DataFrame) and (not dpl.empty):
            _players_n = int(dpl["player"].dropna().astype(str).nunique()) if "player" in dpl.columns else 0
            _champs_n = int(dpl["champ"].dropna().astype(str).nunique()) if "champ" in dpl.columns else 0
            _set_last_prepare_step_summary(f"{int(dpl.shape[0])} rows | {_players_n} players | {_champs_n} champs")

        _run_step(
            8,
            "Artifacts relacionados",
            lambda: f"{sum(1 for r in _artifacts_rows if bool(r.get('Existe')))} de {len(_artifacts_rows)} encontrados",
            critical=False,
        )

        report["ok"] = True
    except Exception as e:
        report["ok"] = False
        report.setdefault("errors", []).append(f"{type(e).__name__}: {e}")

    report["elapsed_s"] = float(time.perf_counter() - t_all0)
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _flush_live_cards()
    st.session_state["_csv_prepare_last_sig"] = list(sig)
    st.session_state["_csv_prepare_report"] = report

    _prog.progress(100, text="Concluído" if bool(report.get("ok", False)) else "Falhou")
    if bool(report.get("ok", False)):
        _summary_box.success(
            f"CSV preparado com sucesso em {report['elapsed_s']:.1f}s. "
            f"Times={int(report.get('teams_count', 0) or 0)} | "
            f"TeamGames={int((report.get('team_games_shape') or [0])[0] or 0)} linhas"
        )
    else:
        _summary_box.error("Falha ao preparar o CSV ativo. Veja as etapas acima.")
        for _err in (report.get("errors") or []):
            st.error(str(_err))
        st.stop()

    _render_csv_prepare_report_summary(st, report)
    return report


def csv_bootstrap_gate(
    *,
    csv_path: str,
    hist_csvs: Optional[List[str]] = None,
    artifact_path: str = "",
    st: Any,
    ctx: Mapping[str, Any],
) -> None:
    """Tela separada de preparo do CSV + manutenção automática quando o CSV muda."""
    APP_ROOT = ctx.get("APP_ROOT")
    _file_sig = ctx.get("_file_sig")
    _CSV_BOOTSTRAP_MAINT_CACHE_PATH = ctx.get("_CSV_BOOTSTRAP_MAINT_CACHE_PATH")
    _CSV_BOOTSTRAP_MAINT_LOCK_PATH = ctx.get("_CSV_BOOTSTRAP_MAINT_LOCK_PATH")
    _CSV_BOOTSTRAP_MAINT_STATE_PATH = ctx.get("_CSV_BOOTSTRAP_MAINT_STATE_PATH")
    _request_streamlit_rerun = ctx.get("_request_streamlit_rerun")

    # Local bootstrap maint shims (inline core_* calls with path constants from ctx)
    def _read_bootstrap_maint_state() -> Dict[str, Any]:
        return dict(core_read_bootstrap_maint_state(_CSV_BOOTSTRAP_MAINT_STATE_PATH))

    def _bootstrap_maint_state_start(*, fp_key: str, owner_token: str = "", total_steps: int = 8) -> str:
        return str(core_bootstrap_maint_state_start(
            _CSV_BOOTSTRAP_MAINT_STATE_PATH,
            fp_key=str(fp_key),
            owner_token=str(owner_token or ""),
            total_steps=int(total_steps or 8),
        ))

    def _bootstrap_maint_state_update(*, run_id: str, fp_key: str, owner_token: str = "", event: Optional[Dict[str, Any]] = None) -> None:
        core_bootstrap_maint_state_update(
            _CSV_BOOTSTRAP_MAINT_STATE_PATH,
            run_id=str(run_id or ""),
            fp_key=str(fp_key),
            owner_token=str(owner_token or ""),
            event=(dict(event or {})),
        )

    def _bootstrap_maint_state_finish(*, run_id: str, fp_key: str, owner_token: str = "", ok: bool, report: Optional[Dict[str, Any]] = None) -> None:
        core_bootstrap_maint_state_finish(
            _CSV_BOOTSTRAP_MAINT_STATE_PATH,
            run_id=str(run_id or ""),
            fp_key=str(fp_key),
            owner_token=str(owner_token or ""),
            ok=bool(ok),
            report=(dict(report or {})),
        )

    def _cleanup_stale_bootstrap_maint_lock(*, fp_key: str, stale_s: int = 180) -> Tuple[bool, str]:
        return core_cleanup_stale_bootstrap_maint_lock(
            lock_path=_CSV_BOOTSTRAP_MAINT_LOCK_PATH,
            state_path=_CSV_BOOTSTRAP_MAINT_STATE_PATH,
            fp_key=str(fp_key),
            stale_s=int(stale_s),
        )

    def _load_bootstrap_maint_report_for_fp(fp_key: str) -> Optional[Dict[str, Any]]:
        return core_load_bootstrap_maint_report_for_fp(APP_ROOT / "csv_bootstrap_maintenance_report.json", str(fp_key))

    def _acquire_bootstrap_maint_lock(*, fp_key: str, ttl_s: int = 1800, owner_token: str = "") -> Tuple[bool, str]:
        _bootstrap_rt = core_get_bootstrap_runtime_state(st.session_state)
        return core_acquire_bootstrap_maint_lock(
            lock_path=_CSV_BOOTSTRAP_MAINT_LOCK_PATH,
            state_path=_CSV_BOOTSTRAP_MAINT_STATE_PATH,
            fp_key=str(fp_key),
            ttl_s=int(ttl_s),
            owner_token=str(owner_token or ""),
            stale_s=int(_bootstrap_rt.get("lock_stale_s", 180) or 180),
        )

    def _release_bootstrap_maint_lock(*, fp_key: str, owner_token: str = "") -> None:
        core_release_bootstrap_maint_lock(
            lock_path=_CSV_BOOTSTRAP_MAINT_LOCK_PATH,
            state_path=_CSV_BOOTSTRAP_MAINT_STATE_PATH,
            fp_key=str(fp_key),
            owner_token=str(owner_token or ""),
        )

    csv_path_s = str(csv_path or "").strip()
    if not csv_path_s:
        return

    fp = core_csv_bootstrap_fingerprint(csv_path_s, hist_csvs, artifact_path, file_sig_fn=_file_sig)
    fp_key = core_csv_bootstrap_fingerprint_key(csv_path_s, hist_csvs, artifact_path, file_sig_fn=_file_sig)
    _gate_state = core_consume_bootstrap_gate_state(st.session_state)
    # Modo visualização (nuvem): o gate não existe — o app confia no
    # modelo/CSV que vieram do repositório; manutenção é papel do robô.
    from app_core.viewer_mode_core import is_viewer_mode as _is_viewer_mode
    if _is_viewer_mode():
        core_mark_bootstrap_gate_completed(st.session_state, fp_key)
        return
    # Test harness bypass: evita prender AppTest na pré-tela e disparar manutenção pesada.
    # GABINETE_E2E: idem, via ambiente — o E2E da suíte NUNCA roda manutenção
    # real (fase 2 da campanha, 17/07: E2E mutou ml_artifact e matou um
    # retreino dele no meio em 15-16/07; suíte tem que ser só-leitura).
    if bool(_gate_state.get("probe_skip", False)) or str(os.environ.get("GABINETE_E2E", "")).strip() == "1":
        core_mark_bootstrap_gate_completed(st.session_state, fp_key)
        return
    force_once = bool(_gate_state.get("force_once", False))
    force_maint = bool(_gate_state.get("force_maint", False))
    force_full_req = bool(_gate_state.get("force_full", False))
    last_done_key = str(_gate_state.get("last_done_key", "") or "")
    pending_open = bool(_gate_state.get("pending_open", False))

    if (not force_once) and (not pending_open) and fp_key == last_done_key:
        return

    st.markdown("## Preparando Base (CSV)")
    st.caption(
        "Tela de pré-carregamento: valida o CSV ativo, aquece caches críticos e executa manutenção automática quando detectar drift/pendências."
    )

    c0, c1, c2 = st.columns([1.6, 0.8, 0.8])
    with c0:
        st.code(csv_path_s, language="text")
    with c1:
        st.caption(f"Históricos: {len([p for p in (hist_csvs or []) if str(p).strip()])}")
    with c2:
        st.caption(f"Artifact: `{Path(str(artifact_path or '')).name or '-'}`")

    # Etapa 1: preparo/validação do CSV (também aquece caches principais).
    rep_prepare = ensure_csv_runtime_ready(csv_path_s, hist_csvs=hist_csvs or [], artifact_path=artifact_path, st=st, ctx=ctx)

    # Etapa 2: diagnóstico de manutenção e execução automática (uma vez por fingerprint).
    st.markdown("### Verificação de manutenção automática")
    status_before = collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx)

    _reconcile_notes: List[str] = []

    def _reconcile_maint_status(sta: Optional[Dict[str, Any]], *, stage: str) -> Dict[str, Any]:
        out, note = core_reconcile_bootstrap_maintenance_status(sta, stage=stage)
        if str(note or "").strip():
            _reconcile_notes.append(str(note))
        return out

    status_before = _reconcile_maint_status(status_before, stage="before")
    core_set_bootstrap_status_before(st.session_state, status_before)
    _bootstrap_rt = core_get_bootstrap_runtime_state(st.session_state)

    _flags_slot = st.empty()
    _quality_slot = st.empty()

    def _render_maint_flags_row(sta: Dict[str, Any], *, label: str) -> None:
        _need_flags = [
            ("Modelo", bool(sta.get("need_retrain_model", False))),
            ("Parâmetros", bool(sta.get("need_retrain_params", False))),
            ("Walk-forward", bool(sta.get("need_walkforward", False))),
            ("Patch/Liga", bool(sta.get("need_patch_recalibration", False))),
        ]
        with _flags_slot.container():
            st.caption(label)
            _flag_cols = st.columns(len(_need_flags))
            for col, (nm, on_) in zip(_flag_cols, _need_flags):
                with col:
                    if nm == "Parâmetros" and bool(sta.get("params_fit_running", False)):
                        st.info(f"{nm}\nrodando")
                        _rt = sta.get("params_fit_runtime") if isinstance(sta.get("params_fit_runtime"), dict) else {}
                        _summary = str(_rt.get("summary") or "").strip()
                        if _summary:
                            st.caption(_summary)
                    elif on_:
                        st.error(f"{nm}\npendente")
                    else:
                        st.success(f"{nm}\nok")

    def _render_quality_row(sta: Dict[str, Any]) -> None:
        qrep = core_build_maintenance_quality_status(sta if isinstance(sta, dict) else {})
        with _quality_slot.container():
            st.caption("Qualidade atual")
            _headline = str(qrep.get("headline") or "").strip()
            if _headline:
                st.write(_headline)
            _cols = st.columns(3)
            _nodes = [
                qrep.get("model_base") if isinstance(qrep.get("model_base"), dict) else {},
                qrep.get("operational") if isinstance(qrep.get("operational"), dict) else {},
                qrep.get("deep") if isinstance(qrep.get("deep"), dict) else {},
            ]
            for _col, _node in zip(_cols, _nodes):
                with _col:
                    _title = str(_node.get("title") or "-")
                    _text = str(_node.get("text") or "").strip()
                    _state = str(_node.get("state") or "").strip().lower()
                    if _state == "ready":
                        st.success(_title)
                    elif _state in {"running", "ready_warn"}:
                        st.info(_title)
                    elif _state == "pending":
                        st.warning(_title)
                    elif _state == "blocked":
                        st.error(_title)
                    else:
                        st.info(_title)
                    if _text:
                        st.caption(_text)

    _render_maint_flags_row(status_before, label="Status inicial")
    _render_quality_row(status_before)

    if status_before.get("alerts"):
        with st.expander("Pendências detectadas", expanded=True):
            for msg in status_before.get("alerts") or []:
                st.write(f"- {msg}")

    auto_ran_key = str(_bootstrap_rt.get("auto_maint_ran_fp", "") or "")
    rep_auto: Optional[Dict[str, Any]] = None
    status_after = status_before
    _auto_run_performed = False
    _full_retry_once = False
    _lock_owner = str(st.session_state.get("_csv_bootstrap_lock_owner", "") or "").strip()
    if not _lock_owner:
        _lock_owner = f"{os.getpid()}-{uuid.uuid4().hex[:10]}"
        st.session_state["_csv_bootstrap_lock_owner"] = _lock_owner
    _auto_prog_bar = st.empty()
    _auto_prog_table = st.empty()
    _auto_prog_rows: List[Dict[str, Any]] = []
    _auto_prog_idx: Dict[str, int] = {}
    _run_ctx: Dict[str, Any] = {"run_id": "", "fp_key": str(fp_key), "owner_token": str(_lock_owner)}

    def _render_live_maint_state_box(state_obj: Dict[str, Any]) -> None:
        if not isinstance(state_obj, dict) or not state_obj:
            return
        cur = state_obj.get("current") if isinstance(state_obj.get("current"), dict) else {}
        rid = str(state_obj.get("run_id") or "").strip()
        stt = str(state_obj.get("status") or "").strip().lower()
        step = str(cur.get("step") or "-")
        idx = int(max(0, cur.get("idx") or 0))
        total = int(max(1, cur.get("total") or 1))
        txt = str(cur.get("status") or stt or "").strip()
        hb_ts = float(state_obj.get("heartbeat_ts", 0.0) or 0.0)
        hb_age = float(max(0.0, time.time() - hb_ts)) if hb_ts > 0.0 else float("inf")
        _auto_prog_bar.progress(float(min(idx, total)) / float(max(1, total)), text=f"Manutenção automática: etapa {idx}/{total} | {step} | {txt}")
        st.caption(
            f"run_id={rid or '-'} | status={stt or '-'} | heartbeat={hb_age:.1f}s"
        )
        rows = list(state_obj.get("steps") or [])
        rows_show: List[Dict[str, Any]] = []
        for r in rows[-12:]:
            if not isinstance(r, dict):
                continue
            rows_show.append(
                {
                    "Etapa": str(r.get("step") or "-"),
                    "Status": str(r.get("status") or "-"),
                    "Resumo": str(r.get("summary") or ""),
                }
            )
        if rows_show:
            _auto_prog_table.dataframe(pd.DataFrame(rows_show), width="stretch", hide_index=True)

    def _on_auto_progress(evt: Dict[str, Any]) -> None:
        try:
            e = dict(evt or {})
            _rid = str(_run_ctx.get("run_id") or "").strip()
            if _rid:
                _bootstrap_maint_state_update(
                    run_id=_rid,
                    fp_key=str(_run_ctx.get("fp_key") or fp_key),
                    owner_token=str(_run_ctx.get("owner_token") or _lock_owner),
                    event=e,
                )
            total = max(1, int(e.get("total") or 1))
            idx = max(0, min(total, int(e.get("idx") or 0)))
            step_name = str(e.get("step") or "-")
            status_txt = str(e.get("status") or "").strip()
            _auto_prog_bar.progress(
                float(idx) / float(total),
                text=f"Manutenção automática: etapa {idx}/{total} | {step_name} | {status_txt}",
            )
            _ev = str(e.get("event") or "").strip().lower()
            if _ev in {"running", "step_done"}:
                _row = {
                    "Etapa": step_name,
                    "Status": status_txt or ("OK" if bool(e.get("ok", False)) else "-"),
                    "Resumo": str(e.get("summary") or "").strip(),
                }
                _i = _auto_prog_idx.get(step_name)
                if _i is None:
                    _auto_prog_idx[step_name] = len(_auto_prog_rows)
                    _auto_prog_rows.append(_row)
                else:
                    _auto_prog_rows[int(_i)] = _row
                # Hardening: dedupe adicional por etapa para evitar crescimento por reruns.
                _dedupe_map: Dict[str, Dict[str, Any]] = {}
                for _r in _auto_prog_rows:
                    if not isinstance(_r, dict):
                        continue
                    _k = str(_r.get("Etapa") or "-").strip()
                    _dedupe_map[_k] = dict(_r)
                _auto_prog_rows[:] = list(_dedupe_map.values())
                _auto_prog_idx.clear()
                for _ix, _r in enumerate(_auto_prog_rows):
                    _auto_prog_idx[str(_r.get("Etapa") or "-")] = int(_ix)
                _trim = _auto_prog_rows[-12:]
                _auto_prog_table.dataframe(pd.DataFrame(_trim), width="stretch", hide_index=True)
        except (TypeError, ValueError):
            pass

    _need_patch = bool(status_before.get("need_patch_recalibration"))
    _need_params = bool(status_before.get("need_retrain_params"))
    _need_wf = bool(status_before.get("need_walkforward"))
    _is_critical = str(status_before.get("severity", "")).strip().lower() == "critical"
    _must_run_auto = core_should_run_bootstrap_auto_maintenance(status_before, force_maint=force_maint)
    _can_auto_now = (auto_ran_key != fp_key) or force_once
    _cache_payload = core_read_json_file_safe(_CSV_BOOTSTRAP_MAINT_CACHE_PATH)
    _cache_fp = str(_cache_payload.get("fp_key", "") or "")
    _cache_ok = bool(_cache_payload.get("ok", False))
    _cache_rep = _cache_payload.get("auto_report") if isinstance(_cache_payload.get("auto_report"), dict) else None
    _cache_status_after = _cache_payload.get("status_after") if isinstance(_cache_payload.get("status_after"), dict) else None

    if _must_run_auto and _can_auto_now:
        _reuse_cache = (not force_maint) and (_cache_fp == fp_key) and _cache_ok and isinstance(_cache_rep, dict)
        if _reuse_cache:
            rep_auto = dict(_cache_rep)
            status_after = dict(_cache_status_after) if isinstance(_cache_status_after, dict) else collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx)
            status_after = _reconcile_maint_status(status_after, stage="after_cache")
            _auto_run_performed = True
            _run_ctx["run_id"] = _bootstrap_maint_state_start(
                fp_key=str(fp_key),
                owner_token=str(_lock_owner),
                total_steps=8,
            )
            _bootstrap_maint_state_finish(
                run_id=str(_run_ctx.get("run_id") or ""),
                fp_key=str(fp_key),
                owner_token=str(_lock_owner),
                ok=bool((rep_auto or {}).get("ok", False)),
                report=(rep_auto if isinstance(rep_auto, dict) else {}),
            )
            st.info("Manutenção automática reaproveitada do cache local para este CSV (sem reexecução).")
        else:
            _lock_ok, _lock_msg = _acquire_bootstrap_maint_lock(
                fp_key=fp_key,
                ttl_s=int(st.session_state.get("auto_maint_lock_ttl_s", 600) or 600),
                owner_token=_lock_owner,
            )
            if not _lock_ok:
                st.warning(
                    "Manutenção automática já está em execução em outra sessão/processo. "
                    f"Motivo: {_lock_msg}."
                )
                rep_auto = _bootstrap_rt.get("auto_maint_last_report") if isinstance(_bootstrap_rt.get("auto_maint_last_report"), dict) else rep_auto
                status_after = _reconcile_maint_status(collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx), stage="after_lock_busy")
                _live_state = _read_bootstrap_maint_state()
                _live_fp = str(_live_state.get("fp_key", "") or "") if isinstance(_live_state, dict) else ""
                _live_status = str(_live_state.get("status", "") or "").strip().lower() if isinstance(_live_state, dict) else ""
                _live_hb_ts = float(_live_state.get("heartbeat_ts", 0.0) or 0.0) if isinstance(_live_state, dict) else 0.0
                _live_hb_age = float(max(0.0, time.time() - _live_hb_ts)) if _live_hb_ts > 0.0 else float("inf")
                if _live_fp == str(fp_key):
                    _render_live_maint_state_box(_live_state if isinstance(_live_state, dict) else {})
                    if _live_status in {"done", "error"}:
                        _rep_disk = _load_bootstrap_maint_report_for_fp(fp_key)
                        if isinstance(_rep_disk, dict):
                            rep_auto = _rep_disk.get("auto_report") if isinstance(_rep_disk.get("auto_report"), dict) else rep_auto
                            _st_after_disk = _rep_disk.get("status_after") if isinstance(_rep_disk.get("status_after"), dict) else None
                            status_after = _reconcile_maint_status(
                                _st_after_disk if isinstance(_st_after_disk, dict) else collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx),
                                stage="after_lock_done_other_session",
                            )
                            st.success("Manutenção concluída em outra sessão/processo. Status final carregado automaticamente.")
                            _auto_run_performed = True
                            core_set_bootstrap_lock_poll_n(st.session_state, 0)
                    elif _live_hb_age <= float(max(10, int(_bootstrap_rt.get("lock_stale_s", 180) or 180))):
                        st.info(f"Execução ativa detectada (heartbeat {int(_live_hb_age)}s). Aguardando conclusão automática...")

                _cleaned, _clean_msg = _cleanup_stale_bootstrap_maint_lock(
                    fp_key=fp_key,
                    stale_s=int(_bootstrap_rt.get("lock_stale_s", 180) or 180),
                )
                if _cleaned:
                    st.info(f"Lock órfão detectado e limpo automaticamente ({_clean_msg}). Retentando agora...")
                    core_set_bootstrap_lock_poll_n(st.session_state, 0)
                    core_request_csv_runtime_refresh(
                        st.session_state,
                        bootstrap=True,
                        hold_screen=True,
                        force_maint=True,
                    )
                    _request_streamlit_rerun()
                _bootstrap_rt = core_get_bootstrap_runtime_state(st.session_state)
                _poll_n = int(_bootstrap_rt.get("lock_poll_n", 0) or 0)
                _poll_max = int(_bootstrap_rt.get("lock_poll_max", 240) or 240)
                _poll_sleep_s = float(_bootstrap_rt.get("lock_poll_sleep_s", 3.0) or 3.0)
                if (_poll_n < _poll_max) and (not _auto_run_performed):
                    st.info(
                        f"Aguardando liberação automática da manutenção ({_poll_n + 1}/{_poll_max}) "
                        f"em {_poll_sleep_s:.1f}s..."
                    )
                    core_set_bootstrap_lock_poll_n(st.session_state, int(_poll_n + 1))
                    time.sleep(max(0.2, min(10.0, _poll_sleep_s)))
                    core_request_csv_runtime_refresh(
                        st.session_state,
                        bootstrap=True,
                        hold_screen=True,
                        force_maint=True,
                    )
                    _request_streamlit_rerun()
                elif not _auto_run_performed:
                    core_set_bootstrap_lock_poll_n(st.session_state, 0)
                    st.info("Não consegui obter lock automaticamente. Clique em 'Rodar manutenção novamente'.")
            else:
                core_set_bootstrap_lock_poll_n(st.session_state, 0)
                _run_ctx["run_id"] = _bootstrap_maint_state_start(
                    fp_key=str(fp_key),
                    owner_token=str(_lock_owner),
                    total_steps=8,
                )
                core_set_bootstrap_active_run_id(st.session_state, _run_ctx.get("run_id") or "")
                try:
                    with st.spinner("Executando manutenção automática (modelo/params/walk-forward/baseline/state artifact)..."):
                        rep_auto = run_auto_maintenance_now(
                            csv_path_s,
                            artifact_path,
                            status_before,
                            force_full=bool(force_full_req),
                            progress_cb=_on_auto_progress,
                            st=st,
                            ctx=ctx,
                        )
                    _auto_run_performed = True
                    status_after = _reconcile_maint_status(collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx), stage="after_run")
                    # Hardening: one-shot fallback when auto run says OK but model still pending.
                    if (
                        isinstance(rep_auto, dict)
                        and bool(rep_auto.get("ok", False))
                        and bool(status_after.get("need_retrain_model", False))
                        and (rep_auto.get("model_new_games") is not None)
                        and Path(str(artifact_path or "")).exists()
                    ):
                        _full_retry_once = True
                        with st.spinner("Ajuste final: retreino completo de modelo (fallback único)..."):
                            rep_retry = run_auto_maintenance_now(
                                csv_path_s,
                                artifact_path,
                                status_after,
                                force_full=True,
                                progress_cb=_on_auto_progress,
                                st=st,
                                ctx=ctx,
                            )
                        status_after = _reconcile_maint_status(collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx), stage="after_retry_full")
                        _steps1 = rep_auto.get("steps") if isinstance(rep_auto.get("steps"), list) else []
                        _steps2 = rep_retry.get("steps") if isinstance(rep_retry, dict) and isinstance(rep_retry.get("steps"), list) else []
                        _steps2_pref: List[Dict[str, Any]] = []
                        for _s in _steps2:
                            if not isinstance(_s, dict):
                                continue
                            _sx = dict(_s)
                            _sx["step"] = f"retry::{_sx.get('step') or '-'!s}"
                            _steps2_pref.append(_sx)
                        rep_auto = {
                            **dict(rep_auto),
                            "ok": bool(rep_auto.get("ok", False)) and bool(isinstance(rep_retry, dict) and rep_retry.get("ok", False)),
                            "duration_s": float(rep_auto.get("duration_s") or 0.0) + float((rep_retry or {}).get("duration_s") or 0.0),
                            "steps": list(_steps1) + list(_steps2_pref),
                            "log": (
                                str(rep_auto.get("log", "") or "")
                                + "\n\n[retry_full_once]\n"
                                + str((rep_retry or {}).get("log", "") or "")
                            ).strip(),
                            "retry_full_once": {
                                "triggered": True,
                                "ok": bool((rep_retry or {}).get("ok", False)),
                                "reason": "model_pending_after_auto_ok",
                            },
                        }
                    _fp_key_after = json.dumps(
                        core_csv_bootstrap_fingerprint(csv_path_s, hist_csvs, artifact_path, file_sig_fn=_file_sig),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    core_write_json_file_atomic(
                        _CSV_BOOTSTRAP_MAINT_CACHE_PATH,
                        {
                            "generated_at": datetime.now().isoformat(timespec="seconds"),
                            "fp_key": _fp_key_after,
                            "ok": bool((rep_auto or {}).get("ok", False)) and (str((status_after or {}).get("severity", "")).lower() != "critical"),
                            "status_after": status_after,
                            "auto_report": rep_auto if isinstance(rep_auto, dict) else {},
                        },
                    )
                    _bootstrap_maint_state_finish(
                        run_id=str(_run_ctx.get("run_id") or ""),
                        fp_key=str(fp_key),
                        owner_token=str(_lock_owner),
                        ok=bool((rep_auto or {}).get("ok", False)),
                        report=(rep_auto if isinstance(rep_auto, dict) else {}),
                    )
                finally:
                    _rid_fin = str(_run_ctx.get("run_id") or "").strip()
                    if _rid_fin:
                        _st_fin = _read_bootstrap_maint_state()
                        if str(_st_fin.get("run_id", "") or "").strip() == _rid_fin and str(_st_fin.get("status", "") or "").strip().lower() == "running":
                            _bootstrap_maint_state_finish(
                                run_id=_rid_fin,
                                fp_key=str(fp_key),
                                owner_token=str(_lock_owner),
                                ok=bool(isinstance(rep_auto, dict) and rep_auto.get("ok", False)),
                                report=(rep_auto if isinstance(rep_auto, dict) else {"error": "maintenance_aborted"}),
                            )
                    _release_bootstrap_maint_lock(fp_key=fp_key, owner_token=_lock_owner)
                    core_set_bootstrap_lock_poll_n(st.session_state, 0)
        if isinstance(rep_auto, dict):
            _fp_key_now = json.dumps(
                core_csv_bootstrap_fingerprint(csv_path_s, hist_csvs, artifact_path, file_sig_fn=_file_sig),
                ensure_ascii=False,
                sort_keys=True,
            )
            core_set_bootstrap_auto_maint_run_fp(st.session_state, _fp_key_now)
            core_set_bootstrap_auto_maint_last_report(st.session_state, rep_auto)
            try:
                core_write_json_file_atomic(
                    APP_ROOT / "csv_bootstrap_maintenance_report.json",
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "csv_fingerprint": fp,
                        "run_id": str(_run_ctx.get("run_id") or ""),
                        "status_before": status_before,
                        "status_after": status_after,
                        "auto_report": rep_auto,
                    },
                )
            except (OSError, TypeError, ValueError):
                pass
    elif _must_run_auto and (not _can_auto_now):
        st.info("Manutenção automática já foi executada para este CSV nesta sessão. Mostrando status final atual.")
        rep_auto = _bootstrap_rt.get("auto_maint_last_report") if isinstance(_bootstrap_rt.get("auto_maint_last_report"), dict) else None
        _rep_disk_done = _load_bootstrap_maint_report_for_fp(fp_key)
        if isinstance(_rep_disk_done, dict):
            _auto_disk = _rep_disk_done.get("auto_report") if isinstance(_rep_disk_done.get("auto_report"), dict) else None
            _st_disk = _rep_disk_done.get("status_after") if isinstance(_rep_disk_done.get("status_after"), dict) else None
            if isinstance(_auto_disk, dict):
                rep_auto = dict(_auto_disk)
            status_after = _reconcile_maint_status(
                _st_disk if isinstance(_st_disk, dict) else collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx),
                stage="after_already_ran_disk",
            )
        else:
            status_after = _reconcile_maint_status(collect_maintenance_status(csv_path_s, artifact_path, st=st, ctx=ctx), stage="after_already_ran")
    else:
        _deferred = []
        if _need_params:
            _deferred.append("parâmetros")
        if _need_wf:
            _deferred.append("walk-forward")
        if _need_patch and not _is_critical:
            _deferred.append("patch/liga (warn)")
        if _deferred:
            st.info(
                "Pendências não críticas detectadas: "
                + ", ".join(_deferred)
                + ". Você pode entrar no app e rodar manutenção completa depois."
            )
        status_after = _reconcile_maint_status(status_after, stage="after_no_auto")

    _render_maint_flags_row(status_after, label="Status final")
    _render_quality_row(status_after)

    if isinstance(rep_auto, dict):
        if bool(rep_auto.get("ok", False)):
            st.success("Manutenção automática concluída.")
        else:
            st.error("Manutenção automática finalizou com falhas.")
        _dur = _fnum(rep_auto.get("duration_s")) or 0.0
        _tout = int(_fnum(rep_auto.get("command_timeout_s")) or 0)
        _tout_params = int(_fnum(rep_auto.get("params_timeout_s")) or _tout)
        _fm = "ON" if bool(rep_auto.get("fast_mode", True)) else "OFF"
        _lv = "ON" if bool(rep_auto.get("run_layer_validation", False)) else "OFF"
        _pp = str(rep_auto.get("policy_profile", "-") or "-")
        _pr = str(rep_auto.get("policy_reason", "-") or "-")
        st.caption(
            f"Duração: {_dur:.1f}s | Timeout modelo: {_tout}s | Timeout params: {_tout_params}s | Fast mode: {_fm} | "
            f"Layer validation: {_lv} | Policy: {_pp} ({_pr})"
        )
        with st.expander("Log da manutenção automática", expanded=False):
            _steps = rep_auto.get("steps") if isinstance(rep_auto.get("steps"), list) else []
            if _steps:
                _rows_steps = []
                for _s in _steps:
                    if not isinstance(_s, dict):
                        continue
                    _rows_steps.append(
                        {
                            "Etapa": str(_s.get("step") or ""),
                            "Status": ("Pulada" if bool(_s.get("skipped", False)) else ("OK" if bool(_s.get("ok", False)) else "Erro")),
                            "Crítica": bool(_s.get("critical", False)),
                            "Tempo (s)": round(_fnum(_s.get("elapsed_s")) or 0.0, 3),
                            "Resumo": str(_s.get("summary") or ""),
                        }
                    )
                if _rows_steps:
                    st.dataframe(pd.DataFrame(_rows_steps), width="stretch", hide_index=True)
            _log = str(rep_auto.get("log", "") or "").strip()
            if _log:
                st.text(_log[:50000])
            else:
                st.caption("Sem log.")
        if bool(_full_retry_once):
            st.info("Fallback aplicado: retreino completo único executado após pendência de modelo pós-auto.")

    if _reconcile_notes:
        with st.expander("Reconciliação de status (hardening)", expanded=False):
            for _rn in _reconcile_notes[-8:]:
                st.write(f"- {_rn}")

    st.markdown("### Status final desta troca de CSV")
    if bool(status_after.get("has_issues", False)):
        if str(status_after.get("severity", "")) == "critical":
            st.error("Ainda existem pendências críticas após o preparo/manutenção. Revise os itens abaixo antes de seguir.")
        else:
            st.warning("Ainda existem alertas pendentes (não críticos). Você pode seguir, mas vale revisar.")
        for msg in (status_after.get("alerts") or []):
            st.write(f"- {msg}")
    else:
        st.success("Base pronta: CSV validado, caches aquecidos e sem pendências de manutenção.")

    _go_decision = "GO"
    _go_reason = "sem_pendencias"
    _go_can_enter = True
    if bool(status_after.get("has_issues", False)):
        if str(status_after.get("severity", "")).strip().lower() == "critical":
            _go_decision = "NO-GO"
            _go_reason = "pendencias_criticas"
            _go_can_enter = False
        else:
            _go_decision = "GO_COM_ALERTAS"
            _go_reason = "alertas_nao_criticos"
            _go_can_enter = True

    st.markdown("### Decisão operacional (GO/NO-GO)")
    if _go_decision == "NO-GO":
        st.error("NO-GO: existem pendências críticas. Corrija antes de operar no app.")
    elif _go_decision == "GO_COM_ALERTAS":
        st.warning("GO com alertas: operação liberada, mas com pendências não críticas registradas.")
    else:
        st.success("GO: base pronta para operação.")

    final_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "csv_fingerprint": fp,
        "prepare_report": rep_prepare,
        "maintenance_status_before": status_before,
        "maintenance_status_after": status_after,
        "maintenance_auto_report": rep_auto,
        "maintenance_reconciliation": {
            "notes": list(_reconcile_notes),
            "auto_run_performed": bool(_auto_run_performed),
            "full_retry_once": bool(_full_retry_once),
        },
        "go_no_go": {
            "decision": _go_decision,
            "reason": _go_reason,
            "can_enter": bool(_go_can_enter),
            "severity": str(status_after.get("severity", "") or ""),
        },
    }
    core_set_bootstrap_last_report(st.session_state, final_report)
    core_request_csv_runtime_refresh(st.session_state, bootstrap=False, hold_screen=True)
    try:
        (APP_ROOT / "csv_bootstrap_last_report.json").write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns([1.2, 1.0, 1.0, 1.1])
    with c_btn1:
        _enter_disabled = bool(_go_decision == "NO-GO")
        if _enter_disabled:
            st.caption("Entrada bloqueada: corrija pendências críticas (NO-GO).")
        if st.button("Entrar no app", type="primary", width="stretch", key="csv_bootstrap_continue", disabled=_enter_disabled):
            _fp_key_now = json.dumps(
                core_csv_bootstrap_fingerprint(csv_path_s, hist_csvs, artifact_path, file_sig_fn=_file_sig),
                ensure_ascii=False,
                sort_keys=True,
            )
            core_mark_bootstrap_gate_completed(st.session_state, _fp_key_now)
            _request_streamlit_rerun()
    with c_btn2:
        if st.button("Rodar manutenção novamente", width="stretch", key="csv_bootstrap_rerun_maint"):
            core_clear_bootstrap_auto_maint_run_state(st.session_state)
            _cleanup_stale_bootstrap_maint_lock(
                fp_key=fp_key,
                stale_s=int(core_get_bootstrap_runtime_state(st.session_state).get("lock_stale_s", 180) or 180),
            )
            core_request_csv_runtime_refresh(
                st.session_state,
                bootstrap=True,
                hold_screen=True,
                force_maint=True,
            )
            _request_streamlit_rerun()
    with c_btn3:
        if st.button("Revalidar CSV", width="stretch", key="csv_bootstrap_recheck_csv"):
            core_request_csv_runtime_refresh(
                st.session_state,
                prepare=True,
                bootstrap=True,
                hold_screen=True,
            )
            _request_streamlit_rerun()
    with c_btn4:
        # Manutenção COMPLETA: full retrain (modelo + players) + refit de params.
        # Necessária quando a incremental rápida não limpa os alertas (ex.:
        # players_artifact em drift, ou params com mtime atrás do artifact).
        if st.button("Manutenção completa", width="stretch", key="csv_bootstrap_rerun_maint_full"):
            core_clear_bootstrap_auto_maint_run_state(st.session_state)
            _cleanup_stale_bootstrap_maint_lock(
                fp_key=fp_key,
                stale_s=int(core_get_bootstrap_runtime_state(st.session_state).get("lock_stale_s", 180) or 180),
            )
            core_request_csv_runtime_refresh(
                st.session_state,
                bootstrap=True,
                hold_screen=True,
                force_maint=True,
                force_full=True,
            )
            _request_streamlit_rerun()
        st.caption("Reconstrói modelo + players e refaz params. Mais lento; use p/ sincronizar de vez.")

    _non_critical_items: List[str] = []
    _prep_steps = (rep_prepare or {}).get("steps") if isinstance(rep_prepare, dict) else []
    if isinstance(_prep_steps, list):
        for _s in _prep_steps:
            if not isinstance(_s, dict):
                continue
            if bool(_s.get("ok", False)):
                continue
            if bool(_s.get("critical", False)):
                continue
            _where = str(_s.get("step") or "Etapa opcional")
            _what = str(_s.get("summary") or _s.get("error") or "falha não crítica")
            _non_critical_items.append(f"[Preparação > {_where}] {_what}")
    if isinstance(status_after, dict):
        _sev = str(status_after.get("severity", "") or "")
        if _sev != "critical":
            _alerts = status_after.get("alerts") or []
            if isinstance(_alerts, tuple):
                _alerts = list(_alerts)
            if not isinstance(_alerts, list):
                _alerts = []
            for _msg in _alerts:
                _m = str(_msg or "").strip()
                if _m:
                    _non_critical_items.append(f"[Manutenção] {_m}")
    _seen_warn: set[str] = set()
    _non_critical_items = [x for x in _non_critical_items if not (x in _seen_warn or _seen_warn.add(x))]
    if _non_critical_items:
        st.markdown("##### Avisos não críticos (você pode entrar no app)")
        for _it in _non_critical_items:
            st.write(f"- {_it}")

    st.stop()
