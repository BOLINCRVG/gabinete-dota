# -*- coding: utf-8 -*-
"""Etapa "Linhas (preencha e confirme)" do fluxo principal.

Fatia 2 do fatiamento do plays_app (F3, 2026-07-18): bloco movido VERBATIM —
mesmas keys/labels/comportamento. Bônus da passagem: os 6 inputs core agora
DERIVAM de STATS_MARKET_CONFIG (line_key), zerando as refs hardcoded do
plays_app (inventário da fase 3); labels históricos preservados via mapa
local com fallback pro label do registro (mercado novo aparece sozinho).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, MutableMapping

import streamlit as st

from app_core.flow_runtime_core import (
    get_lines_loaded_key,
    request_analysis_refresh,
    set_flow_stage,
    set_lines_loaded_key,
    set_matchup_key,
)
from app_core.lines_parsing_core import parse_single_float
from app_core.results_runtime_state_core import resolve_hc_kills_team
from app_core.stats_config import line_key_by_label

# Labels históricos da tela principal (diferem dos exemplos da Agenda).
_LINE_INPUT_LABELS: Dict[str, str] = {
    "lines_kills": "Kills (ex: 27.5, 30.5)",
    "lines_towers": "Torres (ex: 11.5, 12.5)",
    "lines_dragons": "Dragões (ex: 4.5, 5.5)",
    "lines_barons": "Barões (ex: 1.5)",
    "lines_inhib": "Inibidores (ex: 1.5)",
    "lines_time": "Tempo (ex: 32, 32.5)",
}


def render_lines_stage(
    session_state: MutableMapping[str, Any],
    *,
    season_label: Any,
    team_a: str,
    team_b: str,
    matchup_key_fn: Callable[..., str],
    load_lines_for_matchup_fn: Callable[[str], None],
    save_lines_for_matchup_fn: Callable[[str], None],
    lines_store_fn: Callable[[], dict],
    persist_lines_store_fn: Callable[[dict], None],
    lines_store_defaults_key: str,
    reset_flow_fn: Callable[..., Any],
    request_rerun_fn: Callable[[], None],
) -> None:
    st.subheader("Linhas (preencha e confirme)")

    # garante que as linhas exibidas correspondem ao confronto atual (season + times)
    key = matchup_key_fn(season_label, team_a, team_b)
    set_matchup_key(session_state, key)
    if get_lines_loaded_key(session_state) != key:
        load_lines_for_matchup_fn(key)
        set_lines_loaded_key(session_state, key)

    st.caption("Use vírgula para separar múltiplas linhas. Decimal usa **ponto**. Tempo: x, x.0, x.5.")

    with st.expander("Padrão global de linhas (opcional)", expanded=False):
        st.caption("Se você costuma usar as mesmas linhas em vários confrontos, salve aqui para preencher automaticamente quando não houver linhas salvas para o confronto.")
        cP1, cP2 = st.columns(2)
        with cP1:
            if st.button("Salvar como padrão global", width='stretch', key="lines_global_save"):
                save_lines_for_matchup_fn(lines_store_defaults_key)
                st.success("Padrão global salvo.")
        with cP2:
            if st.button("Limpar padrão global", width='stretch', key="lines_global_clear"):
                s = lines_store_fn()
                if isinstance(s, dict):
                    s.pop(lines_store_defaults_key, None)
                    persist_lines_store_fn(s)
                st.success("Padrão global removido.")

    line_cols = st.columns(3)
    for idx, (market_label, line_key) in enumerate(line_key_by_label().items()):
        with line_cols[(idx // 2) % 3]:
            st.text_input(
                _LINE_INPUT_LABELS.get(line_key, market_label),
                key=line_key,
                on_change=save_lines_for_matchup_fn,
                args=(key,),
            )

    _lines_extras_ctx = st.expander("Extras (opcional)", expanded=False)
    with _lines_extras_ctx:
        st.text_input(f"Total Kills {team_a} (ex: 16.5)", key="teamA_kills_line_text", on_change=save_lines_for_matchup_fn, args=(key,))
        st.text_input(f"Total Kills {team_b} (ex: 15.5)", key="teamB_kills_line_text", on_change=save_lines_for_matchup_fn, args=(key,))

        st.caption("Handicap: interpreta como (métrica_for + handicap) > métrica_against (sem push; use .5).")
        _hc_team_cur, hc_team_opts = resolve_hc_kills_team(session_state, team_a=team_a, team_b=team_b)
        st.selectbox(
            "Handicap Kills — Time",
            options=hc_team_opts,
            key="hc_kills_team",
            on_change=save_lines_for_matchup_fn,
            args=(key,),
        )
        _hc_team, _ = resolve_hc_kills_team(session_state, team_a=team_a, team_b=team_b, write_back=False)
        st.text_input(f"HC Kills ({_hc_team}) (ex: -2.5)", key="hc_kills_text", on_change=save_lines_for_matchup_fn, args=(key,))
        _hc_val_preview = parse_single_float(session_state.get("hc_kills_text", ""))
        if _hc_val_preview is not None:
            _hc_other = team_b if _hc_team == team_a else team_a
            st.caption(f"Outro lado (automático): **{_hc_other} {(-_hc_val_preview):+g}**")
        st.text_input(f"HC Torres {team_a} (ex: -3.5)", key="hc_towers_text", on_change=save_lines_for_matchup_fn, args=(key,))
        st.text_input(f"HC Dragões {team_a} (ex: -1.5)", key="hc_dragons_text", on_change=save_lines_for_matchup_fn, args=(key,))

        st.text_input("Linha Kills (para ML + Totais)", key="combo_kills_text", on_change=save_lines_for_matchup_fn, args=(key,))
        st.text_input("Linha Tempo (para ML + Totais)", key="combo_time_text", on_change=save_lines_for_matchup_fn, args=(key,))

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("Salvar linhas e ver resultados", width='stretch'):
            # persiste linhas para esse confronto (para reaparecer ao editar, e ao voltar sem trocar times)
            save_lines_for_matchup_fn(matchup_key_fn(season_label, team_a, team_b))
            set_flow_stage(session_state, "results")
            request_analysis_refresh(session_state)
            request_rerun_fn()
    with c2:
        if st.button("Voltar", width='stretch'):
            reset_flow_fn(keep_config=True)
            request_rerun_fn()

    st.stop()
