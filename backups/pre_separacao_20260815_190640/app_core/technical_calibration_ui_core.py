from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import streamlit as st


def infer_same_league_hint(ml_info: Mapping[str, Any] | None) -> str:
    try:
        blue = str((((ml_info or {}).get("blue") or {}).get("league")) or "").strip()
        red = str((((ml_info or {}).get("red") or {}).get("league")) or "").strip()
        if blue and blue == red:
            return blue
    except Exception:
        return ""
    return ""


def build_wf_correction_caption(corr_info: Mapping[str, Any] | None) -> str:
    if not isinstance(corr_info, Mapping) or not corr_info.get("applied"):
        return ""
    try:
        pin = float(corr_info.get("p_in"))
        pout = float(corr_info.get("p_out"))
        shrink = float(corr_info.get("shrink"))
        return f"Correcao WF no ML ativa: p_map {pin:.4f} -> {pout:.4f} (shrink={shrink:.3f})."
    except Exception:
        return ""


def build_confidence_baseline_caption(
    baseline_live: Mapping[str, Any] | None,
    *,
    baseline_name: str,
    format_num_fn: Callable[[float, int], str],
) -> str:
    if not isinstance(baseline_live, Mapping) or baseline_live.get("error"):
        return "Baseline de confianca por liga ainda nao gerado nesta pasta."
    generated_at = str(baseline_live.get("generated_at") or "-")
    by_league = baseline_live.get("by_league")
    league_count = len(by_league) if isinstance(by_league, Mapping) else 0
    global_node = baseline_live.get("global")
    score_bias = (global_node.get("score_bias") if isinstance(global_node, Mapping) else None)
    try:
        score_bias_text = format_num_fn(float(score_bias), 1)
    except Exception:
        score_bias_text = "-"
    return (
        f"Baseline de confianca atual: {baseline_name} | gerado: {generated_at} | "
        f"ligas={league_count} | bias_global={score_bias_text}."
    )


def build_state_artifact_caption(state_artifact_live: Mapping[str, Any] | None, *, artifact_name: str) -> str:
    if not isinstance(state_artifact_live, Mapping) or state_artifact_live.get("error"):
        return "State artifact (A real) ainda nao gerado nesta pasta."
    meta = state_artifact_live.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    return (
        f"State artifact atual: {artifact_name} | gerado: {state_artifact_live.get('generated_at') or '-'!s} | "
        f"times={int(meta.get('teams') or 0)} | jogos={int(meta.get('games_used') or 0)} | "
        f"mode={meta.get('update_mode') or '-'!s}. "
    )


def render_calibration_support_panels(
    *,
    session_state: dict[str, Any],
    app_root: Path,
    artifact_path: str,
    ml_info: Mapping[str, Any] | None,
    ui_key_fn: Callable[[str], str],
    format_num_fn: Callable[[float, int], str],
    paths_for_ml_fn: Callable[[Path], tuple[str, list[str], str]],
    run_walkforward_cached_fn: Callable[..., dict[str, Any]],
    wf_gate_multiplier_fn: Callable[[Mapping[str, Any], str], float],
    get_wf_last_report_fn: Callable[[dict[str, Any]], dict[str, Any]],
    set_wf_last_report_fn: Callable[[dict[str, Any], Mapping[str, Any] | None], dict[str, Any]],
    get_wf_ml_corr_info_fn: Callable[[dict[str, Any]], dict[str, Any]],
    build_training_csv_list_fn: Callable[[str, list[str]], list[str]],
    resolve_players_artifact_path_fn: Callable[[str, Path], str],
    get_lv_last_report_fn: Callable[[dict[str, Any]], dict[str, Any]],
    set_lv_last_report_fn: Callable[[dict[str, Any], Mapping[str, Any] | None], dict[str, Any]],
    get_fit_scale_state_fn: Callable[[dict[str, Any]], dict[str, float]],
    run_layer_validation_cached_fn: Callable[..., dict[str, Any]],
    load_fitted_params_cached_fn: Callable[[str], dict[str, Any]],
    file_sig_fn: Callable[[str], str],
    fit_params_path: Path,
    generate_and_persist_ml_confidence_baseline_fn: Callable[..., dict[str, Any]],
    ml_confidence_baseline_path: Path,
    load_ml_confidence_baseline_cached_fn: Callable[[str], dict[str, Any]],
    set_ml_confidence_baseline_last_report_fn: Callable[[dict[str, Any], Mapping[str, Any] | None], dict[str, Any]],
    generate_and_persist_team_state_artifact_fn: Callable[..., dict[str, Any]],
    team_state_artifact_path: Path,
    load_team_state_artifact_cached_fn: Callable[[str], dict[str, Any]],
    set_team_state_artifact_last_report_fn: Callable[[dict[str, Any], Mapping[str, Any] | None], dict[str, Any]],
) -> None:
    w1, w2, w3 = st.columns(3)
    with w1:
        st.checkbox("Aplicar correcao WF no ML", key=ui_key_fn("wf_apply_ml_correction"))
        st.caption("Edge gate foi removido da interface para simplificar o fluxo.")
    with w2:
        st.number_input("WF min train games", min_value=200, max_value=20000, step=100, key=ui_key_fn("wf_min_train_games"))
        st.number_input("WF k", min_value=4.0, max_value=80.0, step=1.0, key=ui_key_fn("wf_k"))
        st.slider("Forca correcao WF (ML)", 0.0, 1.0, step=0.05, key=ui_key_fn("wf_ml_corr_strength"))
    with w3:
        st.number_input("WF scale", min_value=100.0, max_value=800.0, step=10.0, key=ui_key_fn("wf_scale"))
        if st.button("Rodar walk-forward", width="stretch", key="ctl_run_walkforward"):
            daily_csv, hist_csvs, _ = paths_for_ml_fn(app_root)
            csv_paths = [str(path) for path in (hist_csvs or []) if str(path).strip() and Path(str(path)).exists()]
            if (not csv_paths) and str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_paths = [str(daily_csv)]
            if not csv_paths:
                st.warning("Sem CSV valido para walk-forward. Configure os CSVs no bootstrap/calibracoes.")
            else:
                with st.spinner("Rodando walk-forward..."):
                    report = run_walkforward_cached_fn(
                        tuple(csv_paths),
                        int(session_state.get(ui_key_fn("wf_min_train_games"), 1500) or 1500),
                        float(session_state.get(ui_key_fn("wf_k"), 26.0) or 26.0),
                        float(session_state.get(ui_key_fn("wf_scale"), 400.0) or 400.0),
                    )
                set_wf_last_report_fn(session_state, report)
                if isinstance(report, Mapping) and not report.get("error"):
                    st.success("Walk-forward concluido.")
                else:
                    error = report.get("error") if isinstance(report, Mapping) else "desconhecido"
                    st.warning(f"Walk-forward retornou erro: {error}")

    wf_report = get_wf_last_report_fn(session_state)
    if isinstance(wf_report, Mapping) and not wf_report.get("error"):
        overall = wf_report.get("overall") if isinstance(wf_report.get("overall"), Mapping) else {}
        roi = overall.get("roi_proxy_mean")
        logloss = overall.get("logloss_mean")
        folds = overall.get("folds")
        league_hint = infer_same_league_hint(ml_info)
        wf_mult = wf_gate_multiplier_fn(wf_report, league_hint)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("WF folds", value=f"{int(folds or 0)}")
        with s2:
            st.metric("WF logloss", value=format_num_fn(float(logloss), 4) if logloss is not None else "-")
        with s3:
            st.metric("WF roi_proxy", value=(format_num_fn(100.0 * float(roi), 2) + "%") if roi is not None else "-")
        with s4:
            st.metric("Edge factor", value=format_num_fn(float(wf_mult), 2))
        if league_hint:
            st.caption(f"Liga do confronto para ajuste do edge gate: {league_hint}")
        corr_info = get_wf_ml_corr_info_fn(session_state)
        apply_corr = bool(session_state.get(ui_key_fn("wf_apply_ml_correction"), session_state.get("wf_apply_ml_correction", False)))
        if apply_corr:
            corr_caption = build_wf_correction_caption(corr_info)
            if corr_caption:
                st.caption(corr_caption)

    cb_cols = st.columns([1.3, 2.7])
    with cb_cols[0]:
        if st.button("Gerar baseline de confianca (liga)", width="stretch", key="ctl_gen_conf_baseline"):
            daily_csv_cb, hist_csvs_cb, _ = paths_for_ml_fn(app_root)
            csv_fit = build_training_csv_list_fn(daily_csv_cb, hist_csvs_cb)
            players_artifact_path = resolve_players_artifact_path_fn(str(artifact_path), app_root)
            if not csv_fit:
                st.warning("Sem CSV valido para baseline de confianca.")
            elif not Path(str(artifact_path)).exists():
                st.warning("ml_artifact.json não encontrado para baseline de confianca.")
            elif not Path(str(players_artifact_path)).exists():
                st.warning("players_artifact.json não encontrado para baseline de confianca.")
            else:
                lv_report_local = get_lv_last_report_fn(session_state)
                wf_report_local = get_wf_last_report_fn(session_state)
                fit_scale_state = get_fit_scale_state_fn(session_state)
                team_scale = float(fit_scale_state.get("team_scale", 1.0) or 1.0)
                players_scale = float(fit_scale_state.get("players_scale", 1.0) or 1.0)
                with st.spinner("Gerando baseline de confianca por liga..."):
                    if not (isinstance(lv_report_local, Mapping) and not lv_report_local.get("error")):
                        lv_report_local = run_layer_validation_cached_fn(
                            tuple(csv_fit),
                            str(artifact_path),
                            str(players_artifact_path),
                            int(session_state.get("lv_min_train_games", 1200) or 1200),
                            float(team_scale),
                            float(players_scale),
                            int(session_state.get("lv_min_league_games", 400) or 400),
                        )
                        set_lv_last_report_fn(session_state, lv_report_local)
                    if not (isinstance(wf_report_local, Mapping) and not wf_report_local.get("error")):
                        wf_report_local = run_walkforward_cached_fn(
                            tuple(csv_fit),
                            int(session_state.get(ui_key_fn("wf_min_train_games"), 1500) or 1500),
                            float(session_state.get(ui_key_fn("wf_k"), 26.0) or 26.0),
                            float(session_state.get(ui_key_fn("wf_scale"), 400.0) or 400.0),
                        )
                        set_wf_last_report_fn(session_state, wf_report_local)
                    fit_live = load_fitted_params_cached_fn(file_sig_fn(str(fit_params_path))) if fit_params_path.exists() else {}
                    report = generate_and_persist_ml_confidence_baseline_fn(
                        root=app_root,
                        lv_report=lv_report_local,
                        wf_report=wf_report_local,
                        params_fit=fit_live,
                    )
                if isinstance(report, Mapping) and not report.get("error"):
                    set_ml_confidence_baseline_last_report_fn(session_state, report)
                    st.success(f"Baseline de confianca salvo ({len((report or {}).get('by_league') or {})} ligas).")
                else:
                    error = report.get("error") if isinstance(report, Mapping) else "desconhecido"
                    st.warning(f"Baseline de confianca com erro: {error}")
    with cb_cols[1]:
        try:
            baseline_live = load_ml_confidence_baseline_cached_fn(file_sig_fn(str(ml_confidence_baseline_path))) if ml_confidence_baseline_path.exists() else {}
        except Exception:
            baseline_live = {}
        st.caption(
            build_confidence_baseline_caption(
                baseline_live,
                baseline_name=ml_confidence_baseline_path.name,
                format_num_fn=format_num_fn,
            )
        )

    sa_cols = st.columns([1.3, 2.7])
    with sa_cols[0]:
        if st.button("Gerar state artifact (A real)", width="stretch", key="ctl_gen_team_state_artifact"):
            daily_csv_sa, hist_csvs_sa, _ = paths_for_ml_fn(app_root)
            csv_state = build_training_csv_list_fn(daily_csv_sa, hist_csvs_sa)
            if not csv_state:
                st.warning("Sem CSV valido para gerar state artifact.")
            else:
                with st.spinner("Gerando state artifact (A real: mu/sigma por time)..."):
                    report = generate_and_persist_team_state_artifact_fn(
                        root=app_root,
                        csv_paths=csv_state,
                        keep_playoffs=None,
                    )
                if isinstance(report, Mapping) and not report.get("error"):
                    set_team_state_artifact_last_report_fn(session_state, report)
                    try:
                        load_team_state_artifact_cached_fn.clear()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    st.success(
                        f"State artifact salvo ({int((((report.get('meta') or {}).get('teams') or 0)))} times, "
                        f"{int((((report.get('meta') or {}).get('games_used') or 0)))} jogos)."
                    )
                else:
                    error = report.get("error") if isinstance(report, Mapping) else "desconhecido"
                    st.warning(f"State artifact com erro: {error}")
    with sa_cols[1]:
        try:
            state_artifact_live = load_team_state_artifact_cached_fn(file_sig_fn(str(team_state_artifact_path))) if team_state_artifact_path.exists() else {}
        except Exception:
            state_artifact_live = {}
        st.caption(build_state_artifact_caption(state_artifact_live, artifact_name=team_state_artifact_path.name))
