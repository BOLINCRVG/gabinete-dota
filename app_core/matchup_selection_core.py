from __future__ import annotations

import functools

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import streamlit as st

# Opções do seletor de BO. Constante de módulo para permitir override por
# outros hosts (o Gabinete Dota inclui BO2; o LoL segue 1/3/5).
BO_OPTIONS: List[int] = [1, 3, 5]

from app_core.agenda_coletada_core import (
    ORIGEM_COLETADO as _ORIGEM_COLETADO,
    ja_passou as _ja_passou,
    rotulo_do_jogo as _rotulo_do_jogo,
)
from app_core.flow_runtime_core import (
    apply_flow_line_values,
    consume_pending_auto_analyze,
    ensure_matchup_selection_state,
    request_analysis_refresh,
    request_pending_reset_matchup,
    set_flow_stage,
    set_lines_loaded_key,
    set_matchup_key,
    set_pending_auto_analyze,
    set_pending_load,
)
from app_core.schedule_state_core import (
    get_schedule_games,
    get_schedule_line_fields,
    get_schedule_line_inputs,
    init_schedule_line_inputs,
    resolve_schedule_line_hc_team,
    save_schedule_games,
)
from app_core.stats_config import line_input_hints, line_key_by_label

# Exemplos dos inputs de linha da Agenda (variante "agenda"; diverge da tela
# principal de propósito). Fonte única: STATS_MARKET_CONFIG.input_hint_agenda —
# mercado novo herda o placeholder sozinho.
_LINE_INPUT_LABELS: Dict[str, str] = line_input_hints("agenda")


def _safe_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


_CHAVE_NAO_RESOLVIDOS = "_agenda_times_nao_resolvidos"


def eh_evento_internacional(liga_agenda: str, liga_por_time: dict) -> bool:
    """A liga da agenda nao e liga de casa de time nenhum -> jogo entre ligas.

    Sem liga na agenda, ou sem mapa de ligas, nao afirma nada (False)."""
    liga = str(liga_agenda or "").strip().lower()
    if not liga or not liga_por_time:
        return False
    casas = {str(v or "").strip().lower() for v in liga_por_time.values()}
    return liga not in casas


def set_pending_unresolved(session_state: Any, faltou: list) -> None:
    """Guarda o que nao deu para casar, para a tela dizer em vez de calar."""
    try:
        session_state[_CHAVE_NAO_RESOLVIDOS] = list(faltou or [])
    except (TypeError, KeyError, AttributeError):
        pass


def _resolver_time_salvo(session_state: Any, cru: str, times: Sequence[str]) -> str:
    """Nome guardado na agenda -> nome do CSV, ou "" quando nao da.

    Usa o MESMO resolvedor da aba Jogos, com os apelidos que ele ja ensinou.
    Antes daqui a comparacao era `nome in times` — exata, letra por letra —, e
    qualquer diferenca de sufixo deixava o campo vazio sem avisar.
    """
    alvo = str(cru or "").strip()
    if not alvo:
        return ""
    if alvo in times:
        return alvo
    try:
        from pathlib import Path

        from app_core.schedule_import_core import (
            load_schedule_aliases,
            resolve_schedule_team,
        )
        # Mesma raiz e mesmo arquivo de apelidos que a aba Jogos usa.
        raiz = Path(__file__).resolve().parents[1]
        apelidos = load_schedule_aliases(session_state, raiz / "user_team_aliases.json")
        achado, ok = resolve_schedule_team(alvo, list(times), apelidos)
        return str(achado) if ok else ""
    except Exception:
        return ""


def _jogos_do_seletor(session_state: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                              dict[str, Any]]:
    """Os jogos salvos DELE mais os que o robo coletou, numa lista so.

    ELE RELATOU com print (16/09/2026): abriu o seletor e so viu tres jogos de
    15/05 -- os antigos dele. Os 15 do calendario nao estavam la, porque eu
    tinha ligado a coleta a uma IMPORTACAO em outra aba: ir na aba Jogos,
    "Trazer os jogos coletados", conferir, "Adicionar a agenda". Tres cliques.

    Mas o pedido era: "esse jogos (agenda) e onde eu quero que fiquem os jogos
    que voce pegar do calendario, PARA QUE EU NAO PRECISE FICAR INDO ATRAS DE
    JOGOS". Coleta que exige ir atras dela nao resolve isso.

    Se a leitura do arquivo falhar, a lista dele continua igual: este seletor
    nunca fica pior do que ja era.
    """
    salvos = get_schedule_games(session_state)
    salvos = salvos if isinstance(salvos, list) else []
    try:
        from pathlib import Path

        from app_core.agenda_coletada_core import (
            carregar_agenda_coletada,
            juntar_para_escolher,
        )
        raiz = Path(__file__).resolve().parents[1]
        coleta = carregar_agenda_coletada(raiz)
        return juntar_para_escolher(salvos, coleta.get("jogos") or []), coleta
    except Exception:
        return list(salvos), {}


def render_matchup_selection(
    session_state: dict[str, Any],
    *,
    flow_stage: str,
    teams: Sequence[str],
    season_label: str,
    matchup_key_fn: Callable[[Any, Any, Any], str],
    save_lines_for_matchup_fn: Callable[[str], Any],
    load_lines_for_matchup_fn: Callable[[str], Any],
    reset_flow_fn: Callable[..., Any],
) -> tuple[str, str, int]:
    st.subheader("Time 1 - Time 2")
    st.subheader("Confronto")

    if flow_stage == "select":
        games, _coleta = _jogos_do_seletor(session_state)
        if isinstance(games, list) and games:
            with st.expander("Selecionar jogo salvo (Agenda)", expanded=False):
                # JOGO QUE JA ACONTECEU SAI DA LISTA (16/09/2026).
                #
                # Ele reclamou duas vezes. Na primeira eu ORDENEI e marquei os
                # antigos com "JA PASSOU"; ele olhou e disse "aqui, ainda esta
                # o mesmo". Estava certo: marcar nao resolve. Nao existe nada
                # a fazer com um jogo de quatro meses atras numa tela de
                # escolher confronto para apostar -- ele so ocupa espaco na
                # frente dos jogos de amanha.
                #
                # ESCONDIDO, NUNCA APAGADO: sao jogos DELE, e podem ter linha
                # digitada junto. O interruptor traz de volta, e a contagem
                # diz que existem -- sumir em silencio seria trocar um
                # incomodo por uma duvida.
                _passados = [g for g in games if _ja_passou(g)]
                _ver_passados = False
                if _passados:
                    _ver_passados = bool(st.checkbox(
                        "Mostrar tambem os {} jogo(s) que ja aconteceram".format(
                            len(_passados)),
                        value=False,
                        key="main_saved_game_ver_passados",
                    ))
                if not _ver_passados:
                    games = [g for g in games if not _ja_passou(g)]

                labels: list[str] = [_rotulo_do_jogo(g) for g in games]
                _n_coletados = sum(
                    1 for g in games
                    if str(g.get("origem") or "") == _ORIGEM_COLETADO)
                # A IDADE DA COLETA, aqui, porque e AQUI que os jogos
                # aparecem agora. Ela vivia no painel de importacao; quando
                # esse painel encolheu (16/09/2026, a pedido dele), o aviso
                # sumiu junto -- e sem ele uma coleta travada oferece os jogos
                # de ontem com toda a confianca do mundo. O sintoma apareceria
                # como "esse jogo nao existe", que nao aponta para a causa.
                _idade = _coleta.get("idade_horas")
                if _n_coletados:
                    _selo = ("coletados ha {:.0f}h".format(_idade)
                             if isinstance(_idade, (int, float))
                             else "sem data de coleta")
                    st.caption(
                        "{} jogo(s) vieram do calendario sozinhos ({}), "
                        "marcados com **do calendario**. Nao precisa importar "
                        "nada.".format(_n_coletados, _selo))
                if isinstance(_idade, (int, float)) and _idade > 24.0:
                    st.warning(
                        "A coleta do calendario tem {:.0f} horas. O robo roda "
                        "de 3 em 3 -- se esta velha, alguma corrida falhou, e "
                        "pode faltar jogo novo aqui.".format(_idade))
                elif _coleta.get("motivo"):
                    st.caption("Calendario indisponivel: {}".format(_coleta["motivo"]))
                if not labels:
                    st.caption("Nenhum jogo por vir na lista.")

                # A lista encolhe quando os antigos somem. Um indice guardado
                # de antes apontaria para fora dela -- e o Streamlit estoura
                # quando o valor salvo nao esta mais entre as opcoes.
                _guardado = session_state.get("main_saved_game_idx")
                try:
                    if int(_guardado) >= len(labels):
                        session_state["main_saved_game_idx"] = -1
                except (TypeError, ValueError):
                    pass

                pick = st.selectbox(
                    "Jogo salvo",
                    options=[-1, *list(range(len(labels)))],
                    key="main_saved_game_idx",
                    format_func=lambda x: "Selecione ..." if int(x) == -1 else labels[int(x)],
                )
                bcols = st.columns([1, 2])
                with bcols[0]:
                    do_pick = st.button(
                        "Usar este jogo",
                        type="primary",
                        width="stretch",
                        disabled=(int(pick) == -1),
                        key="main_saved_game_use",
                    )
                with bcols[1]:
                    st.caption("Dica: se o nome do time não bater com o CSV, use a aba **Jogos** para resolver/editar o texto.")

                if int(pick) >= 0:
                    try:
                        selected_game = games[int(pick)]
                    except Exception:
                        selected_game = None

                    if isinstance(selected_game, dict):
                        team_a_sel = str(selected_game.get("teamA", "") or "").strip()
                        team_b_sel = str(selected_game.get("teamB", "") or "").strip()
                        bo_sel = int(selected_game.get("bo", 3) or 3) if str(selected_game.get("bo", "")).strip() else 3

                        lines_saved = selected_game.get("lines_saved")
                        if not isinstance(lines_saved, dict):
                            lines_saved = {}

                        with st.expander("Linhas deste jogo (opcional) — salvar e/ou analisar", expanded=False):
                            prefix = f"ag_lines_{int(pick)}_"
                            fields = list(get_schedule_line_fields())
                            init_schedule_line_inputs(session_state, prefix, lines_saved, fields=tuple(fields))

                            st.markdown("**Mercados principais**")
                            line_cols = st.columns(3)
                            for idx, (market_label, line_key) in enumerate(line_key_by_label().items()):
                                with line_cols[(idx // 2) % 3]:
                                    st.text_input(
                                        _LINE_INPUT_LABELS.get(line_key, market_label),
                                        key=prefix + line_key,
                                    )

                            with st.expander("Extras (opcional)", expanded=False):
                                st.text_input(f"Total Kills {team_a_sel or 'Time 1'} (ex: 16.5)", key=prefix + "teamA_kills_line_text")
                                st.text_input(f"Total Kills {team_b_sel or 'Time 2'} (ex: 15.5)", key=prefix + "teamB_kills_line_text")

                                st.caption("Handicap: interpreta como (métrica_for + handicap) > métrica_against (sem push; use .5).")
                                if team_a_sel and team_b_sel:
                                    _cur, hc_opts = resolve_schedule_line_hc_team(
                                        session_state,
                                        prefix,
                                        team_a=team_a_sel,
                                        team_b=team_b_sel,
                                    )
                                    st.selectbox(
                                        "Handicap Kills — Time",
                                        options=hc_opts,
                                        key=prefix + "hc_kills_team",
                                    )
                                    hc_team, _ = resolve_schedule_line_hc_team(
                                        session_state,
                                        prefix,
                                        team_a=team_a_sel,
                                        team_b=team_b_sel,
                                        write_back=False,
                                    )
                                else:
                                    hc_team = team_a_sel or team_b_sel

                                st.text_input(f"HC Kills ({hc_team or 'Time'}) (ex: -2.5)", key=prefix + "hc_kills_text")
                                st.text_input(f"HC Torres ({team_a_sel or 'Time 1'}) (ex: -3.5)", key=prefix + "hc_towers_text")
                                st.text_input(f"HC Dragões ({team_a_sel or 'Time 1'}) (ex: -1.5)", key=prefix + "hc_dragons_text")
                                st.text_input("Linha Kills (para ML + Totais)", key=prefix + "combo_kills_text")
                                st.text_input("Linha Tempo (para ML + Totais)", key=prefix + "combo_time_text")

                            s1, s2 = st.columns(2)
                            with s1:
                                save_lines = st.button("Salvar linhas", width="stretch", key=f"ag_save_lines_{int(pick)}")
                            with s2:
                                analyze = st.button(
                                    "Analisar confronto",
                                    type="primary",
                                    width="stretch",
                                    key=f"ag_analyze_{int(pick)}",
                                )

                            if save_lines or analyze:
                                line_values = get_schedule_line_inputs(session_state, prefix, fields=tuple(fields))
                                try:
                                    games[int(pick)]["lines_saved"] = line_values
                                    save_schedule_games(session_state, games)
                                except Exception:
                                    pass

                                if save_lines:
                                    st.success("Linhas salvas para este jogo.")
                                    _safe_rerun()

                                if analyze:
                                    set_pending_load(
                                        session_state,
                                        {
                                            "_reset_flow": True,
                                            "teamA": team_a_sel if team_a_sel in teams else "",
                                            "teamB": team_b_sel if team_b_sel in teams else "",
                                            "bo": bo_sel,
                                        },
                                    )
                                    set_pending_auto_analyze(session_state, {"lines": line_values})
                                    _safe_rerun()

                if do_pick and int(pick) >= 0:
                    try:
                        game = games[int(pick)]
                        prefix = f"ag_lines_{int(pick)}_"
                        current_line_values = get_schedule_line_inputs(session_state, prefix, fields=tuple(get_schedule_line_fields()))
                        if isinstance(game, dict):
                            game["lines_saved"] = dict(current_line_values)
                            save_schedule_games(session_state, games)
                        payload = {"_reset_flow": True}
                        # NOME EXATO NAO BASTA (defeito relatado por ele em
                        # 15/09/2026). A agenda dele tinha "Barczaca Esports" e
                        # o CSV tem "Barczaca Esports by Yumisu" — o campo
                        # Time 1 ficava vazio, EM SILENCIO, e ele so via que
                        # nada aconteceu.
                        #
                        # O resolvedor com apelidos aprendidos ja existia no
                        # projeto (e a mesma coisa que a aba Jogos usa ao
                        # importar); aqui ele nunca tinha sido chamado.
                        _res_a = _resolver_time_salvo(session_state, str(game.get("teamA", "")), teams)
                        _res_b = _resolver_time_salvo(session_state, str(game.get("teamB", "")), teams)
                        # UM LADO CASOU E O OUTRO NAO (23/09/2026): tenta o
                        # outro pela liga do que casou. Regra medida em 52
                        # apelidos dele -- 36 certos, 16 deixados para ele,
                        # nenhum errado. O que ela decidir aparece na tela
                        # como decidido por ela; so vira apelido se ele
                        # confirmar. Ver app_core/casar_time_core.py.
                        _automaticos = []
                        try:
                            from app_core.casar_time_core import completar_par
                            _res_a, _res_b, _automaticos = completar_par(
                                game.get("teamA", ""), _res_a, game.get("teamB", ""), _res_b,
                                list(teams), session_state.get("_ligas_por_time") or {},
                                session_state.get("_ultimo_jogo_por_time") or {})
                        except Exception:
                            _automaticos = []
                        session_state["_casados_sozinhos"] = list(_automaticos)
                        if _res_a:
                            payload["teamA"] = _res_a
                        if _res_b:
                            payload["teamB"] = _res_b
                        # O que NAO resolveu fica dito, com o palpite. Campo
                        # vazio sem explicacao e o pior desfecho: parece que o
                        # botao nao funcionou.
                        _faltou = [(rot, str(game.get(ch, "") or ""))
                                   for rot, ch, r in (("Time 1", "teamA", _res_a),
                                                      ("Time 2", "teamB", _res_b))
                                   if not r and str(game.get(ch, "") or "").strip()]
                        if _faltou:
                            set_pending_unresolved(session_state, _faltou)
                        session_state["_agenda_liga_do_jogo"] = str(game.get("league", "") or "").strip()
                        if str(game.get("bo")) in {str(x) for x in BO_OPTIONS}:
                            payload["bo"] = int(game.get("bo"))
                        if isinstance(game, dict) and isinstance(game.get("lines_saved"), dict):
                            payload["_flow_line_values"] = dict(game.get("lines_saved") or {})
                        set_pending_load(session_state, payload)
                        _safe_rerun()
                    except Exception:
                        pass

    locked = flow_stage in ("lines", "results")
    ensure_matchup_selection_state(session_state)
    team_opts = ["", *list(teams)]
    if str(session_state.get("teamA", "") or "") not in team_opts:
        session_state["teamA"] = ""
    if str(session_state.get("teamB", "") or "") not in team_opts:
        session_state["teamB"] = ""
    if str(session_state.get("bo", "1") or "1") not in {str(x) for x in BO_OPTIONS}:
        session_state["bo"] = int(BO_OPTIONS[0])

    # TIME DA AGENDA QUE NAO CASOU COM O CSV (15/09/2026, relatado por ele).
    # Antes o campo simplesmente ficava vazio e ninguem dizia nada — parecia
    # que o botao "Usar este jogo" nao tinha funcionado. Agora diz QUAL time
    # falhou, ja traz o palpite, e deixa ensinar ali mesmo: o apelido fica
    # salvo e o problema nao volta.
    # O QUE FOI CASADO SOZINHO aparece, com o motivo e o botao de guardar.
    # Esconder seria o erro do UP2U de volta, so que calado.
    _auto = session_state.get("_casados_sozinhos") or []
    if _auto:
        from app_core.casar_time_core import motivo as _motivo_auto
        from app_core.schedule_import_core import (
            load_schedule_aliases as _carregar_ap_auto,
            normalize_schedule_team_key as _chave_ap_auto,
            save_schedule_aliases as _salvar_ap_auto,
        )
        for _lado, _cru_a, _escolhido, _adv in _auto:
            st.info("%s -- %s" % (_lado, _motivo_auto(_cru_a, _escolhido, _adv)))
            if st.button("Esta certo, guardar \"%s\" = %s" % (_cru_a, _escolhido),
                         key="auto_ok_%s" % _chave_ap_auto(_cru_a)):
                from pathlib import Path as _P_auto
                _arq_auto = _P_auto(__file__).resolve().parents[1] / "user_team_aliases.json"
                _ap = _carregar_ap_auto(session_state, _arq_auto)
                _ap[_chave_ap_auto(_cru_a)] = str(_escolhido)
                _salvar_ap_auto(session_state, _arq_auto, _ap)
                session_state["_casados_sozinhos"] = [
                    x for x in _auto if x[1] != _cru_a]
                st.success("Guardado: \"%s\" = %s." % (_cru_a, _escolhido))
        st.caption("Se algum estiver errado, e so escolher o time certo nas listas "
                   "de Time 1 / Time 2 -- nada foi gravado sem voce confirmar.")

    _envio = session_state.pop("_apelido_envio", None) if hasattr(session_state, "pop") else None
    if isinstance(_envio, dict):
        if _envio.get("ok"):
            st.success("Apelido salvo e enviado ao GitHub: sobrevive a reinicio do app.")
        else:
            st.error("Apelido salvo SO nesta sessao -- o envio ao GitHub falhou: {}. "
                     "Confira os Secrets do app (GABINETE_GH_TOKEN, GABINETE_GH_REPO).".format(_envio.get("msg")))
    _nao_casou = session_state.get(_CHAVE_NAO_RESOLVIDOS) or []
    if _nao_casou:
        from pathlib import Path as _Path

        from app_core.schedule_import_core import (
            load_schedule_aliases as _carregar_apelidos,
            normalize_schedule_team_key as _chave_apelido,
            save_schedule_aliases as _salvar_apelidos,
            suggest_schedule_teams as _sugerir,
        )
        _arq_apelidos = _Path(__file__).resolve().parents[1] / "user_team_aliases.json"
        for _rot, _cru in _nao_casou:
            st.warning("**{}** ficou vazio: a agenda tem *{}*, que nao existe "
                       "com esse nome no CSV.".format(_rot, _cru))

            # A LIGA DESEMPATA (16/09/2026). Ele estranhou uma analise e o
            # motivo era este: num jogo do Hitpoint Masters, "UP2U Meavedron"
            # recebeu como primeiro palpite "UP2UMEDIA Cebulaki" -- time
            # POLONES, da Rift Legends, com 4 jogos de historico. O certo,
            # "Meavedron", com 63, ficou embaixo. Nome parecido entre ligas
            # nao e acaso: e o mesmo patrocinador em paises diferentes.
            #
            # A liga vem do time que JA casou neste confronto, e nao de uma
            # tabela de nomes de liga (que nao existe e envelheceria).
            _ligas = session_state.get("_liga_por_time") or {}
            _companheiro = str(session_state.get(
                "teamB" if _rot == "Time 1" else "teamA", "") or "")
            _liga_do_jogo = str(_ligas.get(_companheiro, "") or "")
            # EVENTO INTERNACIONAL (23/10/2026). No Mundial, LOS (CBLOL) contra
            # um time da LCK recebia "este confronto e da LCK, confira o time":
            # a liga vinha do adversario. Se a liga que a AGENDA da ao jogo nao e
            # a liga de casa de nenhum time do CSV ("Mundial", "MSI"...), o jogo
            # e entre ligas e a liga do adversario nao desempata nada.
            _liga_agenda = str(session_state.get("_agenda_liga_do_jogo", "") or "")
            if eh_evento_internacional(_liga_agenda, _ligas):
                _liga_do_jogo = ""
            _mesma_liga = [t for t in teams
                           if _liga_do_jogo and str(_ligas.get(t, "")) == _liga_do_jogo]
            try:
                _sug = _sugerir(_cru, list(teams), prioridade=_mesma_liga)
            except TypeError:
                # Modulo antigo no servidor: a ordem volta a ser a de antes,
                # mas a tela continua de pe.
                _sug = _sugerir(_cru, list(teams))
            if _liga_do_jogo:
                st.caption("Os times de **{}** (a liga de {}) vem primeiro na "
                           "lista.".format(_liga_do_jogo, _companheiro))

            # QUAO VELHO E O DADO DE CADA CANDIDATO, ao lado do nome.
            # Pedido em 21/09/2026: apelido certo com dado de um ano atras e
            # indistinguivel, na lista, de apelido certo com dado de ontem.
            from app_core.frescor_do_time_core import (
                avisar_se_velho as _avisar_velho,
                legenda as _legenda_frescor,
                opcoes as _opcoes_frescor,
            )
            _ultimos = session_state.get("_ultimo_jogo_por_time") or {}
            _rotulos, _de_volta = _opcoes_frescor(list(_sug), _ultimos)
            st.caption(_legenda_frescor())
            _escolhido = st.selectbox(
                "Esse time e qual? (fica salvo)", options=["", *_rotulos],
                key="agenda_fix_{}".format(_chave_apelido(_cru)),
            )
            # O rotulo NAO e o nome: gravar ele como apelido salvaria
            # "Team Liquid - dado mais recente: ..." e nada casaria depois.
            _pick = _de_volta.get(_escolhido, "")
            if _pick:
                _aviso = _avisar_velho(_pick, _ultimos)
                if _aviso:
                    st.warning(_aviso)
            if _pick and _liga_do_jogo and str(_ligas.get(str(_pick), "")) != _liga_do_jogo:
                st.error(
                    "**{}** joga na liga **{}**, e este confronto e da **{}**. "
                    "Confira se e mesmo o time certo: nomes parecidos entre "
                    "ligas costumam ser o mesmo patrocinador em paises "
                    "diferentes, nao o mesmo time.".format(
                        _pick, _ligas.get(str(_pick), "?"), _liga_do_jogo))
            if _pick and st.button("Salvar e usar",
                                   key="agenda_fix_btn_{}".format(_chave_apelido(_cru))):
                _ap = _carregar_apelidos(session_state, _arq_apelidos)
                _ap[_chave_apelido(_cru)] = str(_pick)
                _salvar_apelidos(session_state, _arq_apelidos, _ap)

                # O BO SUMIA AQUI (relatado por ele em 16/09/2026, com print:
                # escolheu "eSuba x UP2U Meavedron | BO5" e a tela abriu BO 1).
                #
                # CAUSA: este botao vive ACIMA da caixa de BO na tela, e o
                # rerun para o script na hora -- antes de a caixa ser criada.
                # O Streamlit descarta o estado de widget que nao chegou a ser
                # desenhado, entao o BO voltava ao padrao. Escrever direto em
                # session_state nao adianta: o valor morre junto.
                #
                # O botao "Usar este jogo" nunca sofreu disso porque passa os
                # valores pela FILA, que e aplicada antes dos widgets. Aqui
                # passa a ser igual -- um caminho so, e nao dois.
                _campo = "teamA" if _rot == "Time 1" else "teamB"
                _outro = "teamB" if _rot == "Time 1" else "teamA"
                _pendente = {_campo: str(_pick)}
                _ja_escolhido = str(session_state.get(_outro, "") or "")
                if _ja_escolhido:
                    _pendente[_outro] = _ja_escolhido
                _bo_atual = session_state.get("bo")
                if str(_bo_atual) in {str(x) for x in BO_OPTIONS}:
                    _pendente["bo"] = int(_bo_atual)
                set_pending_load(session_state, _pendente)

                session_state[_CHAVE_NAO_RESOLVIDOS] = [
                    x for x in _nao_casou if x[0] != _rot]
                _safe_rerun()

    def _fmt_team_placeholder(x: str) -> str:
        return "" if str(x) == "" else str(x)

    cA, cB, cC, cD = st.columns([3, 3, 1, 3])
    with cA:
        team_a = st.selectbox("Time 1", options=team_opts, key="teamA", disabled=locked, format_func=_fmt_team_placeholder)
    with cB:
        team_b = st.selectbox("Time 2", options=team_opts, key="teamB", disabled=locked, format_func=_fmt_team_placeholder)
    with cC:
        bo = st.selectbox("BO", list(BO_OPTIONS), key="bo", disabled=locked)
    with cD:
        if flow_stage == "select":
            st.write("Ação")
            can_analyze = bool(team_a) and bool(team_b) and (str(team_a) != str(team_b))
            pending_auto = consume_pending_auto_analyze(session_state) if can_analyze else {}
            if can_analyze and pending_auto:
                lines_payload = pending_auto.get("lines") if isinstance(pending_auto, dict) else None
                if isinstance(lines_payload, dict):
                    key = matchup_key_fn(season_label, team_a, team_b)
                    set_matchup_key(session_state, key)
                    apply_flow_line_values(session_state, lines_payload, fields=tuple(get_schedule_line_fields()))
                    save_lines_for_matchup_fn(key)
                    set_flow_stage(session_state, "results")
                    request_analysis_refresh(session_state)
                    _safe_rerun()

            if st.button("Analisar Confronto", type="primary", width="stretch", disabled=not can_analyze, key="btn_go_lines_quick"):
                key = matchup_key_fn(season_label, team_a, team_b)
                set_matchup_key(session_state, key)
                load_lines_for_matchup_fn(key)
                set_lines_loaded_key(session_state, key)
                # Skip the legacy "lines" stage — users now input lines inside
                # the per-market sub-tabs (Análise de Linha v3). Go straight
                # to results so the tabs render immediately.
                set_flow_stage(session_state, "results")
                request_analysis_refresh(session_state)
                _safe_rerun()
        elif flow_stage == "lines":
            if st.button("Trocar confronto", width="stretch"):
                request_pending_reset_matchup(session_state)
                _safe_rerun()
        else:
            if st.button("Trocar confronto", width="stretch", key="btn_swap_matchup_results"):
                request_pending_reset_matchup(session_state)
                _safe_rerun()

    return str(team_a or ""), str(team_b or ""), int(bo)


def team_with_league_label(name: str, league: str) -> str:
    """Format team name with optional league suffix: 'TeamName (LCK)'."""
    team = str(name or "").strip()
    lg = str(league or "").strip()
    return f"{team} ({lg})" if lg else team


def canon_team_key(name: str) -> str:
    """Casefolded, whitespace-normalized team key for deduplication and comparison."""
    s = ("" if name is None else str(name)).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def norm_team_token(s: Any) -> str:
    """Normalize a team name token to lowercase alphanumeric words for fuzzy matching."""
    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def team_match_score(name: str, patterns: List[str]) -> int:
    """Score a team name against search patterns: 100=exact, 70=substring, 0=no match."""
    nm = norm_team_token(name)
    score = 0
    for p in patterns:
        pp = norm_team_token(p)
        if not pp:
            continue
        if nm == pp:
            score = max(score, 100)
        elif pp in nm:
            score = max(score, 70)
    return score


def pick_team_name(candidates: List[str], patterns: List[str]) -> Optional[str]:
    """Return the best-matching team name from candidates using pattern scoring, or None."""
    best_name = None
    best_score = -1
    for c in candidates:
        sc = team_match_score(c, patterns)
        if sc > best_score:
            best_score = sc
            best_name = c
    return best_name if best_score >= 60 else None


def team_canon_maps(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Mapas de nome de time, com cache pela ASSINATURA do arquivo.

    POR QUE (23/09/2026): o perfil de um clique mostrou esta funcao RELENDO o
    CSV do disco a cada clique -- 1,4 s. A assinatura (caminho, tamanho, data
    de modificacao) ja vinha no argumento, justamente para isto: ela muda
    sozinha quando ele atualiza o CSV, entao o cache nunca serve mapa velho.

    Devolve COPIA: quem chama pode mexer nas listas, e o cache e compartilhado.
    """
    import copy
    return copy.deepcopy(_team_canon_maps_em_cache(tuple(csv_sig)))


@functools.lru_cache(maxsize=4)
def _team_canon_maps_em_cache(csv_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Build team name canonicalization maps from CSV (deduplication by frequency).

    Returns:
      - variant_to_best_key: canon_key -> best_display_name
      - best_to_variants: best_display_name -> [variants...]
      - best_list: [best_display_name...]
    """
    from mlcore.stats_teams import load_team_rows

    csv_path, _size, _mtime = csv_sig
    rows = load_team_rows(csv_path)
    s = rows.get("teamname")
    if s is None:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    names = s.dropna().astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip()))
    if names.empty:
        return {"variant_to_best_key": {}, "best_to_variants": {}, "best_list": []}

    vc = names.value_counts()
    key_to_variants: Dict[str, List[str]] = {}
    for v in vc.index.tolist():
        k = canon_team_key(v)
        key_to_variants.setdefault(k, []).append(v)

    key_to_best: Dict[str, str] = {}
    best_to_variants: Dict[str, List[str]] = {}
    for k, variants in key_to_variants.items():
        best = max(variants, key=lambda v: int(vc.get(v, 0)))
        key_to_best[k] = best
        best_to_variants[best] = sorted(list(set(variants)))

    best_list = sorted(list(best_to_variants.keys()))
    return {"variant_to_best_key": key_to_best, "best_to_variants": best_to_variants, "best_list": best_list}
