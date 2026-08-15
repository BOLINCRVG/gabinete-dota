"""UI helpers for app settings sidebar."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from app_core.settings_state_core import get_settings_sidebar_state, queue_settings_sidebar_action


def render_settings_sidebar(
    *,
    save_settings_fn: Callable[[], None],
    settings_path: Path,
) -> None:
    st.sidebar.markdown("### Configuração")
    with st.sidebar.expander("Salvar/Carregar (automático)", expanded=True):
        profile_options = ["Padrão", "Avançado", "20/10"]
        sidebar_state = get_settings_sidebar_state(st.session_state)
        current_profile = str(sidebar_state["cfg_profile"])
        if str(st.session_state.get("cfg_profile", "") or "") not in profile_options:
            st.session_state["cfg_profile"] = current_profile if current_profile in profile_options else profile_options[0]
        prof = st.selectbox("Perfil", profile_options, key="cfg_profile")

        st.caption("No Padrão, o app trava o engine principal em ML core v2 para reduzir ajustes manuais.")
        st.checkbox(
            "Fluxo rápido (menos expanders)",
            key="ui_fast_flow",
            help="Mostra blocos principais já abertos para reduzir cliques no fluxo de análise.",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Aplicar preset recomendado", width='stretch'):
                queue_settings_sidebar_action(st.session_state, "_pending_apply_preset", prof)
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
        with c2:
            if st.button("Salvar configuração", width='stretch'):
                save_settings_fn()
                st.success(f"Salvo em `{Path(settings_path).name}`")

        if st.button("Aplicar preset 20/10", width='stretch'):
            queue_settings_sidebar_action(st.session_state, "_pending_apply_twenty_ten")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        if st.button("Recarregar do disco", width='stretch'):
            queue_settings_sidebar_action(st.session_state, "_pending_reload_settings")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        st.caption(f"Arquivo: `{settings_path}`")


