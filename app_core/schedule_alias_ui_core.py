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
from app_core.frescor_do_time_core import (
    avisar_se_velho,
    legenda as legenda_frescor,
    opcoes as opcoes_frescor,
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

        st.caption(legenda_frescor())
        col_a, col_b = st.columns(2)
        with col_a:
            raw_a = st.text_input("Apelido / nome do Time 1", key="alias_rawA")
            _rot_a, _volta_a = opcoes_frescor(
                suggest_schedule_teams(raw_a, teams),
                session_state.get("_ultimo_jogo_por_time") or {})
            pick_a = _volta_a.get(st.selectbox(
                "Mapear Time 1 para...", options=["", *_rot_a],
                key="alias_pickA"), "")
            if pick_a:
                _av = avisar_se_velho(pick_a,
                                      session_state.get("_ultimo_jogo_por_time") or {})
                if _av:
                    st.warning(_av)
            if st.button("Aplicar no Time 1", width="stretch", key="alias_applyA"):
                if raw_a.strip() and pick_a:
                    upsert_schedule_alias(session_state, alias_file, raw_a, pick_a)
                    set_selected_team(session_state, "a", pick_a)
                    st.success(f"Time 1 definido como: {pick_a}")
                    _safe_rerun()

        with col_b:
            raw_b = st.text_input("Apelido / nome do Time 2", key="alias_rawB")
            _rot_b, _volta_b = opcoes_frescor(
                suggest_schedule_teams(raw_b, teams),
                session_state.get("_ultimo_jogo_por_time") or {})
            pick_b = _volta_b.get(st.selectbox(
                "Mapear Time 2 para...", options=["", *_rot_b],
                key="alias_pickB"), "")
            if pick_b:
                _av = avisar_se_velho(pick_b,
                                      session_state.get("_ultimo_jogo_por_time") or {})
                if _av:
                    st.warning(_av)
            if st.button("Aplicar no Time 2", width="stretch", key="alias_applyB"):
                if raw_b.strip() and pick_b:
                    upsert_schedule_alias(session_state, alias_file, raw_b, pick_b)
                    set_selected_team(session_state, "b", pick_b)
                    st.success(f"Time 2 definido como: {pick_b}")
                    _safe_rerun()
