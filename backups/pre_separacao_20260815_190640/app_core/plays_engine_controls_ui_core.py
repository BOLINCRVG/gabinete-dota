from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import pandas as pd
import streamlit as st


def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def render_active_params_summary(
    session_state: MutableMapping[str, Any],
    *,
    param_defaults: Mapping[str, Any],
) -> None:
    st.markdown("### Parametros ativos (resumo)")
    try:
        rows = [{"Parâmetro": str(key), "Valor ativo": str(session_state.get(key, param_defaults[key]))} for key in param_defaults.keys()]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    except Exception:
        pass


def render_engine_filters_panel(
    session_state: MutableMapping[str, Any],
    *,
    daily_csv: str,
    load_team_games_fn: Callable[[Any], Any],
    file_sig_fn: Callable[[str], Any],
) -> None:
    with st.expander("Engine/Filtros do Plays", expanded=False):
        cfg_profile = str(session_state.get("cfg_profile", "Padrao") or "Padrao")
        is_adv_profile = cfg_profile in {"Avancado", "20/10"}
        if not is_adv_profile:
            session_state["ml_engine"] = "mlcore_v2"
        if "ml_engine_compare" not in session_state:
            session_state["ml_engine_compare"] = False
        if "ml_engine_alert_pp" not in session_state:
            session_state["ml_engine_alert_pp"] = 12
        if str(session_state.get("ml_engine", "mlcore_v2") or "mlcore_v2") not in {"mlcore_v2", "elo_season_players"}:
            session_state["ml_engine"] = "mlcore_v2"
        st.selectbox(
            "ML engine",
            ["mlcore_v2", "elo_season_players"],
            key="ml_engine",
            disabled=(not is_adv_profile),
            format_func=lambda x: "ML core v2 (ml_artifact.json)" if x == "mlcore_v2" else "Elo season + players",
        )
        st.checkbox(
            "Comparar/alerta: calcular também Elo season + players",
            key="ml_engine_compare",
        )
        st.slider(
            "Alerta divergencia (pp) [ML Mapa]",
            5,
            30,
            step=1,
            key="ml_engine_alert_pp",
        )

        years: Sequence[Any] = ["All"]
        splits: Sequence[Any] = ["All"]
        try:
            if str(daily_csv or "").strip() and Path(str(daily_csv)).exists():
                team_games = load_team_games_fn(file_sig_fn(str(daily_csv)))
                years = ["All", *sorted([int(y) for y in team_games["year"].dropna().unique().tolist()])]
                splits = ["All", *sorted([str(x) for x in team_games["split"].dropna().unique().tolist()])]
        except Exception:
            pass

        cur_year = session_state.get("year_opt", "All")
        if cur_year not in years:
            cur_year = years[0]
        session_state["year_opt"] = cur_year
        st.selectbox("Year", list(years), key="year_opt")

        cur_split = session_state.get("split_opt", "All")
        if cur_split not in splits:
            cur_split = splits[0]
        session_state["split_opt"] = cur_split
        st.selectbox("Split", list(splits), key="split_opt")

        playoffs_opts = ["all", "true", "false"]
        cur_playoffs = str(session_state.get("playoffs_opt", "all") or "all")
        if cur_playoffs not in playoffs_opts:
            cur_playoffs = "all"
        session_state["playoffs_opt"] = cur_playoffs
        st.selectbox("Playoffs", playoffs_opts, key="playoffs_opt")


def handle_auto_run_incremental(
    session_state: MutableMapping[str, Any],
    *,
    daily_csv: str,
    artifact_path: str,
    consume_auto_run_incremental_fn: Callable[[MutableMapping[str, Any]], bool],
    build_incremental_update_cmd_fn: Callable[..., list[str]],
    run_cmd_fn: Callable[[list[str]], Any],
    set_ml_last_log_fn: Callable[[MutableMapping[str, Any], Any], str],
) -> None:
    if not consume_auto_run_incremental_fn(session_state):
        return
    if daily_csv and Path(daily_csv).exists() and Path(artifact_path).exists():
        cmd = build_incremental_update_cmd_fn(artifact_path=artifact_path, csv_path=daily_csv)
        with st.spinner("Atualizando (incremental) automaticamente..."):
            res = run_cmd_fn(cmd)
        set_ml_last_log_fn(session_state, (res.stdout or "") + ("\n" + res.stderr if res.stderr else ""))
        if res.returncode == 0:
            st.success("Atualizacao incremental concluida.")
            try:
                st.cache_data.clear()
            except Exception:
                pass
        else:
            st.error("Falha na atualizacao incremental automática.")
        _safe_rerun()
