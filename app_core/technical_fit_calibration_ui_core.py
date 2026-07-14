from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import streamlit as st

from app_core.flow_runtime_core import resolve_players_artifact_path as core_resolve_players_artifact_path


def resolve_refit_players_artifact_path(
    *,
    app_root: Path,
    artifact_path: str,
    load_artifact_fn: Callable[[str], Any],
    artifact_sig_fn: Callable[[str], str],
) -> str:
    return core_resolve_players_artifact_path(
        str(artifact_path),
        app_root,
        load_artifact_fn=lambda path: load_artifact_fn(artifact_sig_fn(str(path))),
    )


def build_fit_summary_snapshot(
    *,
    fit_live: Mapping[str, Any] | None,
    fit_scale_state: Mapping[str, Any] | None,
    scale_league_used: str,
    form_league_used: str,
) -> dict[str, Any]:
    fit_live = fit_live if isinstance(fit_live, Mapping) else {}
    fit_best = fit_live.get("best") if isinstance(fit_live.get("best"), Mapping) else {}
    fit_best_comb = fit_live.get("best_combined") if isinstance(fit_live.get("best_combined"), Mapping) else {}
    fit_best_form = fit_live.get("best_form") if isinstance(fit_live.get("best_form"), Mapping) else {}
    fit_base = fit_live.get("baseline") if isinstance(fit_live.get("baseline"), Mapping) else {}
    fit_scale_state = fit_scale_state if isinstance(fit_scale_state, Mapping) else {}
    return {
        "fit_best": dict(fit_best),
        "fit_best_comb": dict(fit_best_comb),
        "fit_best_form": dict(fit_best_form),
        "fit_base": dict(fit_base),
        "team_scale_used": float(fit_scale_state.get("team_scale_used", 1.0) or 1.0),
        "players_scale_used": float(fit_scale_state.get("players_scale_used", 1.0) or 1.0),
        "scale_league_used": str(scale_league_used or "").strip(),
        "form_league_used": str(form_league_used or "").strip(),
        "generated_at": str(fit_live.get("generated_at", "-")) if isinstance(fit_live, Mapping) else "-",
    }


def apply_best_form_settings_to_session(session_state: dict[str, Any], settings: Mapping[str, Any] | None) -> None:
    if not isinstance(settings, Mapping) or not settings:
        return
    session_state["ml_form_blend_enabled"] = bool(settings.get("ml_form_blend_enabled", True))
    session_state["ml_form_blend_max_weight"] = float(settings.get("ml_form_blend_max_weight", 0.75) or 0.75)
    session_state["ml_form_blend_knee_games"] = float(settings.get("ml_form_blend_knee_games", 6.0) or 6.0)
    session_state["ml_form_blend_beta"] = float(settings.get("ml_form_blend_beta", 2.0) or 2.0)
    session_state["ml_form_blend_signal_power"] = float(settings.get("ml_form_blend_signal_power", 1.0) or 1.0)
    session_state["ml_form_blend_parity_strength"] = float(settings.get("ml_form_blend_parity_strength", 0.35) or 0.35)
    session_state["ml_form_use_fitted"] = bool(settings.get("ml_form_use_fitted", True))


def render_fit_calibration_panel(
    *,
    session_state: dict[str, Any],
    app_root: Path,
    artifact_path: str,
    fit_params_path: Path,
    ui_key_fn: Callable[[str], str],
    format_num_fn: Callable[[float, int], str],
    load_fitted_params_cached_fn: Callable[[str], dict[str, Any]],
    file_sig_fn: Callable[[str], str],
    get_fit_scale_state_fn: Callable[[dict[str, Any]], dict[str, float]],
    paths_for_ml_fn: Callable[[Path], tuple[str, list[str], str]],
    fit_ml_params_run_fn: Callable[..., dict[str, Any]],
    enrich_areal_tuning_fn: Callable[..., tuple[dict[str, Any], str]],
    enrich_marginal_tuning_fn: Callable[..., tuple[dict[str, Any], str]],
    backup_file_if_exists_fn: Callable[..., str],
    set_fit_base_scales_fn: Callable[..., Any],
    load_artifact_fn: Callable[[str], Any],
    artifact_sig_fn: Callable[[str], str],
) -> None:
    st.markdown("#### Calibracao ML (Team vs Players)")
    fit_live = {}
    try:
        if fit_params_path.exists():
            fit_live = load_fitted_params_cached_fn(file_sig_fn(str(fit_params_path))) or {}
    except Exception:
        fit_live = {}

    summary = build_fit_summary_snapshot(
        fit_live=fit_live,
        fit_scale_state=get_fit_scale_state_fn(session_state),
        scale_league_used=str(session_state.get("_fit_scale_league_used", "") or ""),
        form_league_used=str(session_state.get("_fit_form_cfg_league_used", "") or ""),
    )

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("team_scale (ativo)", value=format_num_fn(float(summary["team_scale_used"]), 3))
    with f2:
        st.metric("players_scale (ativo)", value=format_num_fn(float(summary["players_scale_used"]), 3))
    with f3:
        ll_base = summary["fit_base"].get("logloss")
        st.metric("Logloss baseline", value=format_num_fn(float(ll_base), 5) if ll_base is not None else "-")
    with f4:
        ll_fit = summary["fit_best_comb"].get("logloss", summary["fit_best"].get("logloss"))
        improvement = summary["fit_best_comb"].get(
            "improvement_logloss_vs_scale_only",
            summary["fit_best"].get("improvement_logloss_vs_baseline"),
        )
        st.metric(
            "Logloss fitted (final)",
            value=format_num_fn(float(ll_fit), 5) if ll_fit is not None else "-",
            delta=format_num_fn(float(improvement), 5) if improvement is not None else None,
        )

    ff1, ff2, ff3 = st.columns(3)
    fit_best_form = summary["fit_best_form"]
    with ff1:
        st.metric("Form fitted", value="ON" if bool(fit_best_form.get("enabled", False)) else "OFF")
    with ff2:
        max_weight = fit_best_form.get("max_weight")
        st.metric("Form max_weight", value=format_num_fn(float(max_weight), 3) if max_weight is not None else "-")
    with ff3:
        gain_no_form = fit_best_form.get("improvement_logloss_vs_no_form")
        st.metric("Ganho vs sem forma", value=format_num_fn(float(gain_no_form), 5) if gain_no_form is not None else "-")

    st.caption(f"Arquivo: {fit_params_path.name} | atualizado: {summary['generated_at']}")
    if summary["scale_league_used"]:
        st.caption(f"Escala por liga aplicada neste confronto: {summary['scale_league_used']}")
    else:
        st.caption("Escala global aplicada (sem override por liga).")
    if summary["form_league_used"]:
        st.caption(f"Forma da temporada fitted por liga aplicada: {summary['form_league_used']}")
    else:
        st.caption("Forma da temporada fitted global aplicada (sem override por liga).")

    rf1, rf2 = st.columns([1.2, 2.8])
    with rf1:
        if "ctl_refit_fast_mode" not in session_state:
            session_state["ctl_refit_fast_mode"] = True
        fit_fast_mode = st.checkbox(
            "Modo rápido (recomendado)",
            key="ctl_refit_fast_mode",
        )
        if st.button("Auto-ajustar ML completo", width="stretch", key="ctl_refit_ml_params"):
            daily_csv, hist_csvs, _ = paths_for_ml_fn(app_root)
            csv_fit = [str(path) for path in (hist_csvs or []) if str(path).strip() and Path(str(path)).exists()]
            if str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                csv_fit.append(str(daily_csv))
            csv_fit = list(dict.fromkeys(csv_fit))

            players_artifact_path = resolve_refit_players_artifact_path(
                app_root=app_root,
                artifact_path=str(artifact_path),
                load_artifact_fn=load_artifact_fn,
                artifact_sig_fn=artifact_sig_fn,
            )

            if not csv_fit:
                st.warning("Sem CSV valido para recalibrar.")
            elif not Path(str(artifact_path)).exists():
                st.warning("ml_artifact.json não encontrado para recalibracao.")
            elif not Path(players_artifact_path).exists():
                st.warning("players_artifact.json não encontrado para recalibracao.")
            else:
                backup_path = backup_file_if_exists_fn(fit_params_path, tag="pre_auto_fit")
                with st.spinner("Recalibrando parametros Team/Players via walk-forward..."):
                    report_fit = fit_ml_params_run_fn(
                        artifact_path=str(artifact_path),
                        players_artifact_path=str(players_artifact_path),
                        csv_paths=csv_fit,
                        min_train_games=int(session_state.get(ui_key_fn("wf_min_train_games"), 300) or 300),
                        speed_profile=("fast" if bool(fit_fast_mode) else "full"),
                    )
                    report_fit, areal_src = enrich_areal_tuning_fn(report_fit, root=app_root)
                    report_fit, mtbc_src = enrich_marginal_tuning_fn(report_fit, root=app_root)
                    Path(fit_params_path).write_text(json.dumps(report_fit, ensure_ascii=False, indent=2), encoding="utf-8")

                if isinstance(report_fit, Mapping) and report_fit.get("best"):
                    best = report_fit.get("best") or {}
                    set_fit_base_scales_fn(
                        session_state,
                        team_scale=float(best.get("team_scale") or 1.0),
                        players_scale=float(best.get("players_scale") or 1.0),
                    )
                    best_form_settings = ((report_fit.get("best_form") or {}).get("settings") or {}) if isinstance(report_fit.get("best_form"), Mapping) else {}
                    apply_best_form_settings_to_session(session_state, best_form_settings)
                    if backup_path:
                        st.caption(f"Backup criado antes do auto-ajuste: `{backup_path}`")
                    speed_profile = str((report_fit or {}).get("speed_profile", "full")) if isinstance(report_fit, Mapping) else "full"
                    source_parts: list[str] = []
                    if str(areal_src or "").strip():
                        source_parts.append(f"A-real tuning: {areal_src}")
                    if str(mtbc_src or "").strip():
                        source_parts.append(f"Marginal tuning: {mtbc_src}")
                    suffix = (" | " + " | ".join(source_parts)) if source_parts else ""
                    st.success(f"Auto-ajuste concluido (Team/Players + Forma) [{speed_profile}] e params_fitted.json atualizado.{suffix}")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                else:
                    error = report_fit.get("error") if isinstance(report_fit, Mapping) else "desconhecido"
                    st.warning(f"Recalibracao retornou erro: {error}")

        if st.button("Voltar baseline (1.0 / 1.0)", width="stretch", key="ctl_refit_reset_baseline"):
            current = {}
            try:
                if fit_params_path.exists():
                    current = load_fitted_params_cached_fn(file_sig_fn(str(fit_params_path))) or {}
            except Exception:
                current = {}
            if not isinstance(current, dict):
                current = {}
            baseline = current.get("baseline") if isinstance(current.get("baseline"), Mapping) else {}
            best = current.get("best") if isinstance(current.get("best"), Mapping) else {}
            best["team_scale"] = 1.0
            best["players_scale"] = 1.0
            if "logloss" not in best and "logloss" in baseline:
                best["logloss"] = baseline.get("logloss")
            if "brier" not in best and "brier" in baseline:
                best["brier"] = baseline.get("brier")
            current["best"] = best
            current["best_by_league"] = {}
            current["best_form_by_league"] = {}
            current["best_form_by_macro"] = {}
            current["players_league_weights"] = {"global": 1.0, "by_league": {}, "by_macro": {}, "meta": {"min_n": 0}}
            current["best_form"] = {
                "enabled": False,
                "settings": {
                    "ml_form_blend_enabled": False,
                    "ml_form_blend_max_weight": 0.75,
                    "ml_form_blend_knee_games": 6.0,
                    "ml_form_blend_beta": 2.0,
                    "ml_form_blend_signal_power": 1.0,
                    "ml_form_blend_parity_strength": 0.35,
                    "ml_form_use_fitted": False,
                },
            }
            current["generated_at"] = datetime.now().isoformat(timespec="seconds")
            backup_path = backup_file_if_exists_fn(fit_params_path, tag="pre_reset_baseline")
            Path(fit_params_path).write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            set_fit_base_scales_fn(session_state, team_scale=1.0, players_scale=1.0)
            session_state["ml_form_use_fitted"] = False
            if backup_path:
                st.caption(f"Backup criado antes do reset: `{backup_path}`")
            st.success("Baseline restaurado: team_scale=1.0, players_scale=1.0.")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    with rf2:
        st.caption("Usa walk-forward em CSVs histórico+diario para ajustar Team/Players e também os parametros de forma da temporada.")
        try:
            blind_backtest = (fit_live.get("blind_backtest_by_league") if isinstance(fit_live, Mapping) else None) or {}
            blind_rows = list(blind_backtest.get("rows") or []) if isinstance(blind_backtest, Mapping) else []
            if blind_rows:
                st.markdown("##### Blind Backtest por Liga (holdout mais recente)")
                bb_df = pd.DataFrame(blind_rows)
                cols = [
                    "league",
                    "holdout_month",
                    "n_total",
                    "logloss_team_only",
                    "logloss_players_only",
                    "logloss_fused",
                    "gain_fused_vs_team",
                    "gain_fused_vs_players",
                ]
                cols = [col for col in cols if col in bb_df.columns]
                if cols:
                    st.dataframe(bb_df[cols], width="stretch", hide_index=True)
        except Exception:
            pass
