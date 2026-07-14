# -*- coding: utf-8 -*-
"""Página Técnico — calibração, drift, retreino e parâmetros."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app_core.maintenance_runtime_core import (
    build_incremental_update_cmd as _build_incremental_update_cmd,
    build_retrain_check_cmd as _build_retrain_check_cmd,
    build_train_offline_cmd as _build_train_offline_cmd,
)
from app_core.navigation_state_core import set_nav_pending as _set_nav_pending
from app_core.plays_engine_controls_ui_core import (
    handle_auto_run_incremental as _handle_auto_run_incremental,
    render_active_params_summary as _render_active_params_summary,
    render_engine_filters_panel as _render_engine_filters_panel,
)
from app_core.csv_bootstrap_runtime_core import (
    consume_auto_run_maintenance as _consume_auto_run_maintenance,
    consume_auto_run_incremental as _consume_auto_run_incremental,
)
from app_core.runtime_session_state_core import (
    get_ml_last_log as _get_ml_last_log,
    set_ml_last_log as _set_ml_last_log,
)


def render_calibracoes(root: Path, *, st: Any, ctx: Mapping[str, Any]) -> None:
    _paths_for_ml = ctx.get("_paths_for_ml")
    _run_auto_maintenance_now = ctx.get("_run_auto_maintenance_now")
    _clear_streamlit_cache_data = ctx.get("_clear_streamlit_cache_data")
    _request_streamlit_rerun = ctx.get("_request_streamlit_rerun")
    _csv_diag = ctx.get("_csv_diag")
    _file_sig = ctx.get("_file_sig")
    _PARAM_DEFAULTS = ctx.get("_PARAM_DEFAULTS")
    _load_team_games = ctx.get("_load_team_games")
    _save_app_settings_to_disk = ctx.get("_save_app_settings_to_disk")
    _run_cmd = ctx.get("_run_cmd")
    _read_artifact_summary = ctx.get("_read_artifact_summary")
    _players_artifact_output_path = ctx.get("_players_artifact_output_path")

    st.header("Técnico")
    st.caption("Aqui fica tudo de CSV/diagnóstico, calibração, drift, retreino e parâmetros.")

    daily_csv, hist_csvs, artifact_path = _paths_for_ml(root)

    # Auto-run disparado pelo banner global do Plays.
    _auto_maint = _consume_auto_run_maintenance(st.session_state)
    if bool(_auto_maint.get("enabled", False)):
        _st_auto = _auto_maint.get("status")
        with st.spinner("Recalibrando automaticamente (modelo/params/walk-forward)..."):
            _rep_auto = _run_auto_maintenance_now(daily_csv, artifact_path, _st_auto)
        if bool(_rep_auto.get("ok", False)):
            st.success("Recalibração automática concluída.")
            _clear_streamlit_cache_data()
            _request_streamlit_rerun()
        else:
            st.error("Recalibração automática falhou. Veja o log abaixo.")

    _csv_diag_info = None
    try:
        if str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
            _csv_diag_info = _csv_diag(_file_sig(str(daily_csv)))
    except (OSError, TypeError, ValueError):
        _csv_diag_info = None

    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        st.markdown("### Fontes")
        st.text_input("CSV diário (usado no app)", value=daily_csv, disabled=True)
        st.text_area(
            "CSVs históricos (treino completo)",
            value="\n".join([str(p) for p in hist_csvs]) if hist_csvs else "(nenhum selecionado)",
            height=120,
            disabled=True,
        )
        st.text_input("ml_artifact.json", value=artifact_path, disabled=True)

    with c2:
        st.markdown("### Status do artifact")
        summ = _read_artifact_summary(artifact_path)
        if not summ.get("exists"):
            st.warning("Ainda não existe `ml_artifact.json`. Faça **Treino completo** abaixo para gerar.")
        else:
            if summ.get("error"):
                st.error(str(summ["error"]))
            st.write({
                "trained_at": summ.get("trained_at"),
                "data_max_date": summ.get("data_max_date"),
                "n_games_total": summ.get("n_games_total"),
                "added_games_last": summ.get("added_games_last"),
                "backfill_suspected": summ.get("backfill_suspected"),
            })

    if isinstance(_csv_diag_info, dict):
        st.caption(
            f"Diagnóstico CSV: {_csv_diag_info.get('rows',0)} linhas | {_csv_diag_info.get('unique_gameids',0)} gameids | "
            f"{_csv_diag_info.get('games_complete',0)} jogos completos | {_csv_diag_info.get('games_incomplete',0)} incompletos | "
            f"{_csv_diag_info.get('rows_missing_gameid',0)} linhas sem gameid"
        )

    _render_active_params_summary(
        st.session_state,
        param_defaults=_PARAM_DEFAULTS,
    )
    _render_engine_filters_panel(
        st.session_state,
        daily_csv=daily_csv,
        load_team_games_fn=_load_team_games,
        file_sig_fn=_file_sig,
    )

    st.divider()

    st.markdown("### Preferências da manutenção automática")
    st.caption("Controla a execução automática de modelo, parâmetros e walk-forward quando entra CSV novo ou quando você dispara recalibração.")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.checkbox("Fast mode", key="auto_maint_fast_mode")
        st.checkbox("Rodar layer validation", key="auto_maint_run_layer_validation")
        st.checkbox("Forçar policy fast", key="auto_maint_force_fast")
        st.checkbox("Forçar policy full (prioridade)", key="auto_maint_force_full")
    with m2:
        st.number_input(
            "Limite fast: jogos novos",
            min_value=0,
            max_value=1000,
            step=1,
            key="auto_maint_fast_new_games_max",
        )
        st.number_input(
            "Timeout modelo (s)",
            min_value=60,
            max_value=21600,
            step=60,
            key="auto_maint_cmd_timeout_s",
        )
    with m3:
        st.number_input(
            "Timeout params (s)",
            min_value=300,
            max_value=43200,
            step=300,
            key="auto_maint_params_timeout_s",
        )
        st.number_input(
            "TTL lock (s)",
            min_value=60,
            max_value=3600,
            step=30,
            key="auto_maint_lock_ttl_s",
        )
    if st.button("Salvar preferências da manutenção", key="save_auto_maint_prefs", width='stretch'):
        _save_app_settings_to_disk()
        st.success("Preferências da manutenção salvas.")

    st.divider()

    _handle_auto_run_incremental(
        st.session_state,
        daily_csv=daily_csv,
        artifact_path=artifact_path,
        consume_auto_run_incremental_fn=_consume_auto_run_incremental,
        build_incremental_update_cmd_fn=_build_incremental_update_cmd,
        run_cmd_fn=_run_cmd,
        set_ml_last_log_fn=_set_ml_last_log,
    )

    st.markdown("### Ações")

    colA, colB, colC = st.columns([1, 1, 1])
    # 30 = deploy validado (tools/atualizar_modelo.py); 180 congela ratings fora da season.
    st.session_state.setdefault("cal_holdout_days", 30)
    st.session_state.setdefault("cal_do_calib", "auto")
    st.session_state.setdefault("cal_keep_playoffs", "auto")
    with colA:
        holdout_days = st.number_input("Holdout (dias)", min_value=0, max_value=3650, step=30, key="cal_holdout_days")
    with colB:
        do_calib = st.selectbox("Calibração (Platt)", ["auto", "yes", "no"], key="cal_do_calib")
    with colC:
        keep_playoffs = st.selectbox("Playoffs no treino/check", ["auto", "yes", "no"], key="cal_keep_playoffs")

    keep_playoffs_flag = "auto"
    if keep_playoffs == "yes":
        keep_playoffs_flag = "yes"
    elif keep_playoffs == "no":
        keep_playoffs_flag = "no"

    btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
    with btn1:
        train_clicked = st.button("Treino completo", width='stretch')
    with btn2:
        incr_clicked = st.button("Atualizar rápido (incremental)", width='stretch')
    with btn3:
        check_clicked = st.button("Checar drift / retreino", width='stretch')
    with btn4:
        back_clicked = st.button("Voltar para Plays", width='stretch')

    if back_clicked:
        _set_nav_pending(st.session_state, "Plays")

    if train_clicked:
        csvs = [p for p in (hist_csvs or []) if str(p).strip()]
        if daily_csv:
            csvs.append(daily_csv)
        if not csvs:
            st.error("Selecione pelo menos um CSV (diário ou histórico).")
        else:
            cmd = _build_train_offline_cmd(
                csvs,
                out_path=artifact_path,
                holdout_days=int(holdout_days),
                calibrate=str(do_calib),
                keep_playoffs=keep_playoffs_flag,
                fit_players=True,
                players_out=_players_artifact_output_path(str(artifact_path)),
            )
            with st.spinner("Treinando (completo)… (primeira vez pode demorar)"):
                res = _run_cmd(cmd)
            _set_ml_last_log(st.session_state, (res.stdout or "") + ("\n" + res.stderr if res.stderr else ""))
            if res.returncode == 0:
                st.success("Treino completo concluído. Artifact atualizado.")
                _clear_streamlit_cache_data()
                _request_streamlit_rerun()
            else:
                st.error("Falha no treino completo. Veja o log abaixo.")

    if incr_clicked:
        if not daily_csv or not Path(daily_csv).exists():
            st.error("CSV diário não encontrado/selecionado.")
        elif not Path(artifact_path).exists():
            st.error("ml_artifact.json não encontrado. Rode o **Treino completo** primeiro.")
        else:
            cmd = _build_incremental_update_cmd(artifact_path=artifact_path, csv_path=daily_csv)
            with st.spinner("Atualizando artifact (incremental)…"):
                res = _run_cmd(cmd)
            _set_ml_last_log(st.session_state, (res.stdout or "") + ("\n" + res.stderr if res.stderr else ""))
            if res.returncode == 0:
                st.success("Atualização incremental concluída.")
                _clear_streamlit_cache_data()
                _request_streamlit_rerun()
            else:
                st.error("Falha na atualização incremental. Veja o log abaixo.")

    if check_clicked:
        if not daily_csv or not Path(daily_csv).exists():
            st.error("CSV diário não encontrado/selecionado.")
        elif not Path(artifact_path).exists():
            st.error("ml_artifact.json não encontrado. Rode o **Treino completo** primeiro.")
        else:
            cmd = _build_retrain_check_cmd(
                artifact_path=artifact_path,
                csv_path=daily_csv,
                keep_playoffs=keep_playoffs_flag,
                exit_code=True,
            )
            with st.spinner("Rodando retrain_check…"):
                res = _run_cmd(cmd)
            out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            _set_ml_last_log(st.session_state, out)
            if res.returncode == 2:
                st.warning("O check recomendou **RETREINO COMPLETO**. (Veja o log abaixo.)")
            elif res.returncode == 0:
                st.success("Check OK: sem necessidade de retreino agora. (Veja o log abaixo.)")
            else:
                st.error("retrain_check falhou. Veja o log abaixo.")

    st.markdown("### Log")
    log = _get_ml_last_log(st.session_state)
    if log:
        st.code(log)
    else:
        st.caption("Nada rodou ainda nesta sessão.")
