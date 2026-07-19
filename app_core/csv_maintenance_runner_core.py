# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.bootstrap_maintenance_state_core import (
    read_json_file_safe as core_read_json_file_safe,
    write_json_file_atomic as core_write_json_file_atomic,
)
from app_core.flow_runtime_core import build_training_csv_list as _build_training_csv_list
from app_core.maintenance_policy_core import (
    adjust_needs_after_incremental as core_adjust_needs_after_incremental,
    decide_maintenance_policy as core_decide_maintenance_policy,
    force_params_after_model_refresh as core_force_params_after_model_refresh,
    resolve_maintenance_preferences as core_resolve_maintenance_preferences,
    seed_maintenance_needs as core_seed_maintenance_needs,
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
from app_core.maintenance_status_core import collect_maintenance_status as core_collect_maintenance_status
from app_core.ml_league_runtime_core import (
    sanitize_ml_marginal_band_coeffs as core_sanitize_ml_marginal_band_coeffs,
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
from app_core.process_runner_core import run_cmd as _run_cmd
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
from mlcore.utils import safe_float_or_none as _fnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
import json
import sys
import threading
import time


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

    def _params_deep_fit_info(path_like: str | Path) -> Dict[str, Any]:
        """O params atual veio de um fit PROFUNDO? (settings_by_league populado).

        Blindagem 2026-07-19 (pedido dele): a manutenção automática rodava um
        fit RÁPIDO por cima do resultado de um fit profundo de ~11h, apagando a
        segmentação por liga do players_context (17 ligas -> 0). Aqui só
        DETECTAMOS; quem decide o que fazer é _guard_deep_params_before_fit.
        """
        try:
            import json as _json
            d = _json.loads(Path(path_like).read_text(encoding="utf-8-sig"))
            pc = d.get("players_context_fit") or {}
            ligas = pc.get("settings_by_league") or {}
            return {"deep": bool(ligas), "n_ligas": len(ligas)}
        except (OSError, ValueError, TypeError):
            return {"deep": False, "n_ligas": 0}

    def _guard_deep_params_before_fit(speed_profile: str) -> str:
        """Backup + aviso ANTES de um fit rápido pisar num fit profundo.

        Não bloqueia (o refit pode ser necessário quando o artifact muda), mas
        o trabalho profundo nunca se perde: fica em ops/backups e o log diz
        como restaurar."""
        alvo = APP_ROOT / "params_fitted.json"
        if str(speed_profile).strip().lower() != "fast":
            return ""
        info = _params_deep_fit_info(alvo)
        if not info.get("deep"):
            return ""
        try:
            import shutil as _shutil
            import time as _time
            bdir = APP_ROOT / "ops" / "backups"
            bdir.mkdir(parents=True, exist_ok=True)
            dest = bdir / f"params_fitted.deep_before_auto_fast_{_time.strftime('%Y%m%d_%H%M%S')}.json"
            _shutil.copy2(alvo, dest)
            return (f"AVISO: fit RAPIDO vai sobrescrever um fit PROFUNDO "
                    f"({info['n_ligas']} ligas no players_context). Backup salvo em "
                    f"ops/backups/{dest.name} — para restaurar, copie por cima de "
                    f"params_fitted.json.")
        except (OSError, TypeError, ValueError) as exc:
            return f"AVISO: fit rapido sobre fit profundo e o backup FALHOU ({type(exc).__name__})."

    def _run_fit_params_once(csvs: List[str], *, speed_profile: str) -> Dict[str, Any]:
        _guard_msg = _guard_deep_params_before_fit(speed_profile)
        if _guard_msg:
            logs.append(_guard_msg)
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

