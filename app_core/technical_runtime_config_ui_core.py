from __future__ import annotations

from typing import Any, Mapping, Sequence

import streamlit as st


def render_technical_runtime_config_panel(
    *,
    session_state: dict[str, Any],
    seasons: Sequence[Any],
    runtime_cfg_defaults: Mapping[str, Any],
    request_csv_runtime_refresh_fn,
    reset_bootstrap_selection_fn,
) -> None:
    with st.expander("Configurações (raramente muda)", expanded=False):
        if "cfg_auto_paths" not in session_state:
            session_state["cfg_auto_paths"] = bool(runtime_cfg_defaults.get("auto_paths", True))
        if "cfg_csv_override" not in session_state:
            session_state["cfg_csv_override"] = str(runtime_cfg_defaults.get("csv_override") or "")
        if "cfg_artifact_override" not in session_state:
            session_state["cfg_artifact_override"] = str(runtime_cfg_defaults.get("artifact_override") or "")
        auto_paths = st.checkbox("Auto (por Season)", key="cfg_auto_paths")
        if not seasons:
            st.warning("Não encontrei CSVs no padrão 'YYYY_LoL_esports_match_data_from_OraclesElixir.csv'. Use o override abaixo.")
            auto_paths = False
            session_state["cfg_auto_paths"] = False

        if not auto_paths:
            st.text_input(
                "CSV OracleElixir (team rows)",
                key="cfg_csv_override",
            )
        st.text_input(
            "ml_artifact.json (opcional)",
            key="cfg_artifact_override",
        )
        st.caption("Se não existir p_map para o confronto, o app continua (só perde ML).")
        st.markdown("#### ML: Técnico / Retreino")
        st.info("Para evitar treinar toda hora e para voce ter certeza quando treinou/retreinou, tudo de drift/retreino/update fica na aba Técnico (no topo).")
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Limpar cache (CSV)", width="stretch"):
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                request_csv_runtime_refresh_fn(
                    session_state,
                    prepare=True,
                    bootstrap=True,
                    hold_screen=True,
                )
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with c2:
            st.caption("Use isso se voce trocou/atualizou o CSV mantendo o mesmo nome.")

        if st.button("Trocar CSVs (voltar para seleção inicial)"):
            reset_bootstrap_selection_fn(session_state, show_selector=True)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
