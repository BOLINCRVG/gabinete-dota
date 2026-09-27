# -*- coding: utf-8 -*-
"""Expander "Jogos (agenda) — colar / adicionar" da página Plays.

Fatia 1 do fatiamento do plays_app (F3, 2026-07-18): bloco movido VERBATIM —
mesmo markup, mesmas keys de session_state, mesmo comportamento. A fiação de
estado da agenda (restore/bind do store) fica no plays_app: ela serve também
à seleção de confronto, não só a este expander.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence

import pandas as pd
import streamlit as st

from app_core.schedule_import_core import (
    load_schedule_aliases,
    normalize_schedule_team_key,
    parse_schedule_text,
    resolve_schedule_team,
    save_schedule_aliases,
    suggest_schedule_teams,
)
from app_core.agenda_coletada_core import (
    bo_do_jogo as _bo_do_jogo,
    carregar_agenda_coletada,
)
from app_core.schedule_state_core import (
    append_schedule_games,
    clear_pending_schedule_import,
    clear_schedule_games,
    clear_schedule_text,
    get_pending_schedule_import,
    get_schedule_games,
    persist_schedule_runtime_state,
    save_pending_schedule_import,
    set_schedule_unknown_aliases,
)


def render_schedule_agenda_expander(
    session_state: MutableMapping[str, Any],
    *,
    app_root: Path,
    teams: Sequence[str],
) -> None:
    with st.expander("Agenda - jogo manual e limpeza", expanded=False):
        # O QUE SAIU DAQUI, E POR QUE (16/09/2026).
        #
        # Ele olhou este painel e disse: "isso aqui nao precisa mais eu acho,
        # ate porque voce ja pega". Tinha razao na maior parte. Sairam:
        #   - a caixa de COLAR jogos e o "Processar texto";
        #   - o "BO padrao (para os jogos importados)";
        #   - o botao "Trazer os jogos coletados".
        #
        # O robo baixa a agenda de 3 em 3 horas e os jogos entram DIRETO no
        # seletor de confronto, marcados "do calendario". Importar virou um
        # passo que nao leva a lugar nenhum.
        #
        # O QUE FICOU, porque nao tem substituto em lugar nenhum do app:
        #   - LIMPAR A AGENDA: unico jeito de apagar jogos salvos antigos;
        #   - ADICIONAR 1 JOGO MANUAL: para o que o calendario da Riot nao
        #     tiver (chave que so definiu tarde, liga fora da cobertura).
        st.caption("Os jogos do calendario entram sozinhos no seletor de "
                   "confronto, marcados **do calendario** — nao e preciso "
                   "importar nada aqui. Este painel serve para o que o "
                   "calendario nao traz, e para limpar a lista.")

        _alias_path = app_root / "user_team_aliases.json"
        persist_schedule_runtime_state(session_state)
        txt = ""
        parse_clicked = False

        if st.button("Limpar agenda (lista atual)", width='stretch',
                     key="plays_schedule_clear_list"):
            clear_schedule_games(session_state)
            st.success("Agenda limpa.")

        with st.expander("Adicionar 1 jogo manual", expanded=False):
            session_state.setdefault("plays_manual_date", "")
            session_state.setdefault("plays_manual_time", "")
            session_state.setdefault("plays_manual_league", "")
            session_state.setdefault("plays_manual_teamA", "")
            session_state.setdefault("plays_manual_teamB", "")
            session_state.setdefault("plays_manual_bo", 3)
            cM1, cM2, cM3 = st.columns([1, 1, 1])
            with cM1:
                m_date = st.text_input("Data (DD/MM)", key="plays_manual_date", placeholder="22/01")
            with cM2:
                m_time = st.text_input("Hora (HH:MM)", key="plays_manual_time", placeholder="05:00")
            with cM3:
                m_league = st.text_input("Liga (opcional)", key="plays_manual_league", placeholder="LCK")

            cM4, cM5, cM6 = st.columns([2, 2, 1])
            with cM4:
                m_teamA = st.text_input("Time A (como você tem)", key="plays_manual_teamA", placeholder="BRION")
            with cM5:
                m_teamB = st.text_input("Time B (como você tem)", key="plays_manual_teamB", placeholder="Hanwha Life Esports")
            with cM6:
                # ESTE VALOR ERA DESENHADO E NUNCA LIDO. O comentario antigo
                # dizia isso em voz alta: o BO vinha do "BO padrao" la de
                # cima, igual para todos. Era o mesmo defeito que ele relatou
                # no seletor de confronto -- um BO5 tratado como BO3 muda a
                # odd da serie inteira. Com o "BO padrao" removido, agora o
                # numero escolhido aqui e o que vale.
                m_bo = st.selectbox("BO", [1, 3, 5], key="plays_manual_bo")

            if st.button("Adicionar (vai pedir mapeamento se precisar)", width='stretch', key="plays_manual_add"):
                # valida mínimo
                if not (m_date.strip() and m_time.strip() and m_teamA.strip() and m_teamB.strip()):
                    st.warning("Preencha pelo menos Data, Hora, Time A e Time B.")
                else:
                    pending_now = get_pending_schedule_import(session_state)
                    pending_now.append({
                        "date": m_date.strip(),
                        "time": m_time.strip(),
                        "league": m_league.strip(),
                        "teamA_raw": m_teamA.strip(),
                        "teamB_raw": m_teamB.strip(),
                        "bo": int(m_bo),
                    })
                    save_pending_schedule_import(session_state, pending_now)
                    st.success("Adicionado à prévia. Agora mapeie (se necessário) e clique em **Adicionar à agenda**.")
                    st.rerun()

        if parse_clicked:
            parsed = parse_schedule_text(txt)
            if not parsed:
                st.warning("Não consegui encontrar jogos nesse texto. Dica: precisa ter linhas com hora (HH:MM) e logo depois 2 linhas de times.")
            else:
                save_pending_schedule_import(session_state, parsed)
                st.success(f"Encontrados: {len(parsed)} jogo(s). Agora confira o mapeamento e clique em **Adicionar à agenda**.")
                st.rerun()

        pending = get_pending_schedule_import(session_state)
        set_schedule_unknown_aliases(session_state, False)
        if isinstance(pending, list) and pending:
            st.markdown("#### Prévia (antes de adicionar)")
            st.dataframe(pd.DataFrame(pending), width='stretch', hide_index=True)

            aliases = load_schedule_aliases(session_state, _alias_path)

            # detecta times não resolvidos
            unknowns: List[str] = []
            for g in pending:
                a_raw = str(g.get("teamA_raw", "") or "").strip()
                b_raw = str(g.get("teamB_raw", "") or "").strip()
                a_res, ok_a = resolve_schedule_team(a_raw, teams, aliases)
                b_res, ok_b = resolve_schedule_team(b_raw, teams, aliases)
                if not ok_a and a_raw:
                    unknowns.append(a_raw)
                if not ok_b and b_raw:
                    unknowns.append(b_raw)
            unknowns = sorted(list({u for u in unknowns}))
            set_schedule_unknown_aliases(session_state, bool(unknowns))

            mapping: Dict[str, str] = {}
            if unknowns:
                st.markdown("#### Mapear times desconhecidos")
                st.caption("Se o nome do site não bater com o nome do CSV, escolha o time correto aqui. Isso fica salvo (apelido -> time do CSV).")
                from app_core.frescor_do_time_core import (
                    avisar_se_velho as _avisar_velho,
                    legenda as _legenda_frescor,
                    opcoes as _opcoes_frescor,
                )
                _ultimos = session_state.get("_ultimo_jogo_por_time") or {}
                st.caption(_legenda_frescor())
                for raw in unknowns:
                    sug = suggest_schedule_teams(raw, teams)
                    _rotulos, _de_volta = _opcoes_frescor(sug, _ultimos)
                    _escolhido = st.selectbox(
                        f"Esse time '{raw}' seria qual",
                        options=["", *_rotulos],
                        key=f"map_{normalize_schedule_team_key(raw)}",
                    )
                    # O rotulo carrega a data; o apelido tem de guardar o NOME.
                    pick = _de_volta.get(_escolhido, "")
                    if pick:
                        mapping[raw] = pick
                        _av = _avisar_velho(pick, _ultimos)
                        if _av:
                            st.warning(_av)

            c1, c2 = st.columns([1, 1])
            with c1:
                add_clicked = st.button("Adicionar à agenda", type="primary", width='stretch', key="plays_schedule_add_btn")
            with c2:
                cancel_clicked = st.button("Cancelar importação", width='stretch', key="plays_schedule_cancel_btn")

            if cancel_clicked:
                clear_pending_schedule_import(session_state)
                st.info("Importação cancelada.")
                st.rerun()

            if add_clicked:
                # aplica mapping + aliases existentes
                aliases = load_schedule_aliases(session_state, _alias_path)
                for raw, canon in mapping.items():
                    if canon:
                        aliases[normalize_schedule_team_key(raw)] = canon
                save_schedule_aliases(session_state, _alias_path, aliases)

                out_games: List[Dict[str, Any]] = []
                for g in pending:
                    a_raw = str(g.get("teamA_raw", "") or "").strip()
                    b_raw = str(g.get("teamB_raw", "") or "").strip()
                    a_res, ok_a = resolve_schedule_team(a_raw, teams, aliases)
                    b_res, ok_b = resolve_schedule_team(b_raw, teams, aliases)

                    # se ainda ficou desconhecido, mantém raw (não bloqueia), mas não vai carregar automático pro confronto
                    out_games.append({
                        "date": str(g.get("date", "") or "").strip(),
                        "time": str(g.get("time", "") or "").strip(),
                        "league": str(g.get("league", "") or "").strip(),
                        "teamA": a_res,
                        "teamB": b_res,
                        # O BO REAL do jogo, quando a coleta trouxe. O
                        # seletor de BO padrao vira so o plano B — antes ele
                        # carimbava o mesmo formato em todos os jogos, e um
                        # BO5 lido como BO3 muda a odd da serie inteira.
                        # O BO do proprio jogo. Cai em 3 so quando ninguem
                        # disse qual e -- e nao mais num seletor global que
                        # carimbava o mesmo formato em todos.
                        "bo": _bo_do_jogo(g, 3),
                        "lines": "",
                    })

                append_schedule_games(session_state, out_games)
                clear_pending_schedule_import(session_state)
                st.success(f"Adicionados: {len(out_games)} jogo(s) na agenda.")
                st.rerun()

        st.caption(f"Agenda atual: {len(get_schedule_games(session_state))} jogo(s).")
