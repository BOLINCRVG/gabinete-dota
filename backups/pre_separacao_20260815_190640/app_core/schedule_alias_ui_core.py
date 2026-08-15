from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from app_core.flow_runtime_core import set_selected_team
from app_core.schedule_import_core import (
    load_schedule_aliases,
    suggest_schedule_teams,
    upsert_schedule_alias,
)
from app_core.schedule_state_core import has_schedule_unknown_aliases


def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


def render_schedule_alias_mapping(
    session_state: dict[str, Any],
    *,
    alias_path: str | Path,
    teams: Sequence[str],
) -> None:
    if not has_schedule_unknown_aliases(session_state):
        return

    alias_file = Path(alias_path)
    with st.expander("Time não aparece na lista Mapear apelido", expanded=True):
        st.caption("Aparece quando algum nome da agenda/colagem não bateu com o CSV. Mapeie aqui e salve o alias.")
        load_schedule_aliases(session_state, alias_file)
        if "alias_rawA" not in session_state:
            session_state["alias_rawA"] = ""
        if "alias_rawB" not in session_state:
            session_state["alias_rawB"] = ""

        col_a, col_b = st.columns(2)
        with col_a:
            raw_a = st.text_input("Apelido / nome do Time 1", key="alias_rawA")
            pick_a = st.selectbox("Mapear Time 1 para...", options=["", *suggest_schedule_teams(raw_a, teams)], key="alias_pickA")
            if st.button("Aplicar no Time 1", width="stretch", key="alias_applyA"):
                if raw_a.strip() and pick_a:
                    upsert_schedule_alias(session_state, alias_file, raw_a, pick_a)
                    set_selected_team(session_state, "a", pick_a)
                    st.success(f"Time 1 definido como: {pick_a}")
                    _safe_rerun()

        with col_b:
            raw_b = st.text_input("Apelido / nome do Time 2", key="alias_rawB")
            pick_b = st.selectbox("Mapear Time 2 para...", options=["", *suggest_schedule_teams(raw_b, teams)], key="alias_pickB")
            if st.button("Aplicar no Time 2", width="stretch", key="alias_applyB"):
                if raw_b.strip() and pick_b:
                    upsert_schedule_alias(session_state, alias_file, raw_b, pick_b)
                    set_selected_team(session_state, "b", pick_b)
                    st.success(f"Time 2 definido como: {pick_b}")
                    _safe_rerun()
