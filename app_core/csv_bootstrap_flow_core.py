# -*- coding: utf-8 -*-
"""Bootstrap flow: gate screen, CSV prepare runtime, and auto-maintenance."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from mlcore.utils import safe_float_or_none as _fnum
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
    consume_bootstrap_gate_state as core_consume_bootstrap_gate_state,
    csv_bootstrap_fingerprint as core_csv_bootstrap_fingerprint,
    csv_bootstrap_fingerprint_key as core_csv_bootstrap_fingerprint_key,
    mark_bootstrap_gate_completed as core_mark_bootstrap_gate_completed,
    reconcile_bootstrap_maintenance_status as core_reconcile_bootstrap_maintenance_status,
    request_csv_runtime_refresh as core_request_csv_runtime_refresh,
    should_run_bootstrap_auto_maintenance as core_should_run_bootstrap_auto_maintenance,
)


# Helpers movidos p/ csv_maintenance_runner_core (fatiamento 18/07); re-importados
# aqui para o namespace/ctx do app seguir idêntico.
from app_core.csv_maintenance_runner_core import (  # noqa: E402,F401 - re-export
    collect_maintenance_status,
    run_auto_maintenance_now,
)




# Helpers movidos p/ csv_runtime_ready_core (fatiamento 18/07); re-importados
# aqui para o namespace/ctx do app seguir idêntico.
from app_core.csv_runtime_ready_core import (  # noqa: E402,F401 - re-export
    ensure_csv_runtime_ready,
)


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
    # NUVEM NAO FAZ MANUTENCAO NA ABERTURA (23/10/2026). Ao recriar o app na
    # Streamlit Cloud, a primeira tela virou "Executando manutencao automatica"
    # (params + walk-forward) em vez da escolha dos times: o clone novo traz os
    # relatorios do git, de julho, e tudo parece "pendente". A nuvem tem ~1GB e
    # so CARREGA o que os robos treinam (mesma regra do auto-retreino ao vivo,
    # live_models_status_core.is_training_capable_env). Forcar a mao continua valendo.
    if _must_run_auto and not force_maint:
        from app_core.live_models_status_core import is_streamlit_cloud
        if is_streamlit_cloud():
            _must_run_auto = False
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
