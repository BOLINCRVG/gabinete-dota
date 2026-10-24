# -*- coding: utf-8 -*-
"""A GAVETA DE JOGOS da aba "Modelo x Mercado" (24/10/2026).

Pedido dele: "ao inves de eu selecionar o jogo e ir para ele, de forma
unitaria, ser uma especie de gaveta, como se fosse uma aba do google, onde eu
clico no jogo que foi achado e ele me mostra as estatisticas tanto das casas
(poly + pinnacle) quanto as analises de linha ... e uma para eu escolher os
dois times, sendo separado por dia".

O DESENHO:
  - DIAS como abas (Hoje, Amanha, ...), pelo horario de Brasilia;
  - dentro do dia, um botao por JOGO achado (Pinnacle e/ou Polymarket), em
    ordem de horario -- as "abas do navegador";
  - um botao fixo "Escolher times", para montar o confronto na mao.
Clicar num jogo ABRE o confronto (o mesmo caminho do "Analisar confronto" da
agenda): o modelo completo so e calculado para o confronto aberto, entao a
gaveta troca o confronto e o app recalcula. Leva alguns segundos por clique.

Este modulo tem as contas puras (agrupar, rotular) e o desenho do seletor.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

ESCOLHER = "__escolher__"
_DIAS_SEMANA = ("seg", "ter", "qua", "qui", "sex", "sab", "dom")


def _br(iso: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00").replace(" ", "T"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)


def rotulo_do_dia(d: date, hoje: date) -> str:
    if d == hoje:
        return "Hoje {:%d/%m}".format(d)
    if d == hoje + timedelta(days=1):
        return "Amanha {:%d/%m}".format(d)
    return "{} {:%d/%m}".format(_DIAS_SEMANA[d.weekday()], d)


def _chave_nome(n: str) -> str:
    from app_core.schedule_import_core import normalize_schedule_team_key
    return normalize_schedule_team_key(str(n or ""))


def _parecido(a: Any, b: Any) -> bool:
    ka, kb = _chave_nome(a), _chave_nome(b)
    return bool(ka and kb) and (ka == kb or ka in kb or kb in ka)


def _mesmo_par(a1: Any, b1: Any, a2: Any, b2: Any) -> bool:
    return (_parecido(a1, a2) and _parecido(b1, b2)) or (_parecido(a1, b2) and _parecido(b1, a2))


def jogos_da_gaveta(odds: Mapping[str, Any], poly: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Os jogos da Pinnacle e os da Polymarket que a Pinnacle nao tem, com
    data/hora de Brasilia. Pura."""
    fora: List[Dict[str, Any]] = []
    for j in odds.get("jogos") or []:
        quando = _br(j.get("inicio_utc"))
        if quando is None:
            continue
        fora.append({"id": j.get("id"), "casa": str(j.get("casa") or ""), "fora": str(j.get("fora") or ""),
                     "liga": str(j.get("liga") or ""), "bo": j.get("bo"), "quando": quando,
                     "fontes": ["Pinnacle"]})
    for p in poly.get("partidas") or []:
        quando = _br(p.get("inicio_utc"))
        if quando is None:
            continue
        # O MESMO JOGO COM OUTRO NOME (24/10/2026): a Polymarket escreve
        # "Invictus Gaming x Team Secret Whales", a Pinnacle "Invictus x Secret
        # Whales". Mesmo horario (ate 30 min) e os dois nomes parecidos = mesmo jogo.
        igual = next((g for g in fora if "Pinnacle" in g["fontes"]
                      and abs((g["quando"] - quando).total_seconds()) <= 1800
                      and _mesmo_par(g["casa"], g["fora"], p.get("time_a"), p.get("time_b"))), None)
        if igual is not None:
            if "Polymarket" not in igual["fontes"]:
                igual["fontes"].append("Polymarket")
            continue
        fora.append({"id": "poly:" + str(p.get("evento") or ""), "casa": str(p.get("time_a") or ""),
                     "fora": str(p.get("time_b") or ""), "liga": str(p.get("torneio") or ""),
                     "bo": p.get("bo"), "quando": quando, "fontes": ["Polymarket"]})
    return sorted(fora, key=lambda g: g["quando"])


def por_dia(jogos: List[Mapping[str, Any]], hoje: Optional[date] = None) -> Dict[str, List[Mapping[str, Any]]]:
    hoje = hoje or (datetime.now(timezone.utc) - timedelta(hours=3)).date()
    dias: Dict[str, List[Mapping[str, Any]]] = {}
    for g in jogos:
        dias.setdefault(rotulo_do_dia(g["quando"].date(), hoje), []).append(g)
    return dias


def rotulo_do_jogo(g: Mapping[str, Any]) -> str:
    return "{:%H:%M}  {} x {}".format(g["quando"], g["casa"], g["fora"])


def e_o_confronto_aberto(g: Mapping[str, Any], confronto: Mapping[str, Any],
                         resolver: Optional[Callable[[str], str]]) -> bool:
    def _r(n):
        try:
            return (resolver(n) if resolver else n) or n
        except Exception:
            return n
    a, b = str(confronto.get("teamA") or ""), str(confronto.get("teamB") or "")
    if not (a and b):
        return False
    par = {_chave_nome(_r(g["casa"])), _chave_nome(_r(g["fora"]))}
    return par == {_chave_nome(a), _chave_nome(b)}


def abrir_confronto(session_state: Any, time_a: str, time_b: str, bo: Any) -> None:
    """O mesmo caminho do botao "Analisar confronto" da agenda."""
    from app_core.flow_runtime_core import set_pending_auto_analyze, set_pending_load
    try:
        bo_i = int(bo or 1)
    except (TypeError, ValueError):
        bo_i = 1
    set_pending_load(session_state, {"_reset_flow": True, "teamA": time_a, "teamB": time_b,
                                     "bo": bo_i if bo_i in (1, 3, 5) else 1})
    set_pending_auto_analyze(session_state, {"lines": {}})


def render_gaveta(st: Any, *, odds: Mapping[str, Any], poly: Mapping[str, Any],
                  confronto: Mapping[str, Any], resolver: Optional[Callable[[str], str]],
                  times: List[str]) -> Optional[Mapping[str, Any]]:
    """Desenha dias + jogos + "Escolher times". Devolve o jogo da gaveta que
    esta aberto (ou None)."""
    ss = st.session_state
    dias = por_dia(jogos_da_gaveta(odds, poly))
    if not dias:
        st.caption("Nenhum jogo nas coletas da Pinnacle nem da Polymarket.")
    rotulos_dias = list(dias)
    aberto_dia, aberto_jogo = None, None
    for d, gs in dias.items():
        for g in gs:
            if e_o_confronto_aberto(g, confronto, resolver):
                aberto_dia, aberto_jogo = d, g
    if rotulos_dias:
        # radio horizontal, nao segmented_control/pills: o testador do Streamlit
        # (AppTest 1.52) nao reroda tela com pills de valor unico -- e sem
        # teste o "clicar abre o jogo" ficaria sem prova.
        _i_dia = rotulos_dias.index(aberto_dia) if aberto_dia in rotulos_dias else 0
        dia = st.radio("Dia", rotulos_dias, index=_i_dia, horizontal=True,
                       key="gaveta_dia_" + "|".join(rotulos_dias), label_visibility="collapsed")
        dia = dia or rotulos_dias[_i_dia]
        jogos = dias.get(dia) or []
    else:
        jogos = []

    opcoes = [str(g["id"]) for g in jogos] + [ESCOLHER]
    nomes = {str(g["id"]): rotulo_do_jogo(g) for g in jogos}
    nomes[ESCOLHER] = "+ Escolher times"
    por_id = {str(g["id"]): g for g in jogos}
    atual = str(aberto_jogo["id"]) if aberto_jogo and str(aberto_jogo["id"]) in opcoes else None
    chave = "gaveta_jogo_" + str(dia if rotulos_dias else "")

    def _ao_trocar():
        esc = ss.get(chave)
        g = por_id.get(str(esc))
        if not g:
            return
        a = resolver(g["casa"]) if resolver else g["casa"]
        b = resolver(g["fora"]) if resolver else g["fora"]
        if a and b and a in times and b in times:
            abrir_confronto(ss, a, b, g.get("bo"))
        else:
            ss["_gaveta_aviso"] = ("Nao consegui casar {} x {} com o CSV. Abra pelo \"+ Escolher "
                                   "times\" ou ensine o apelido na agenda.").format(g["casa"], g["fora"])

    _sem = "__nenhum__"
    _opcoes_r = ([] if atual else [_sem]) + opcoes
    nomes[_sem] = "(nenhum aberto)"
    escolha = st.radio("Jogos", _opcoes_r, index=_opcoes_r.index(atual) if atual else 0,
                       format_func=lambda k: nomes.get(k, k), key=chave, horizontal=True,
                       on_change=_ao_trocar, label_visibility="collapsed")
    aviso = ss.pop("_gaveta_aviso", None)
    if aviso:
        st.warning(aviso)

    if escolha == ESCOLHER:
        c1, c2, c3, c4 = st.columns([3, 3, 1, 2])
        ta = c1.selectbox("Time 1", [""] + list(times), key="gaveta_time_a")
        tb = c2.selectbox("Time 2", [""] + list(times), key="gaveta_time_b")
        bo = c3.selectbox("BO", [1, 3, 5], key="gaveta_bo")
        c4.write("")
        if c4.button("Abrir confronto", type="primary", disabled=not (ta and tb and ta != tb),
                     key="gaveta_abrir"):
            abrir_confronto(ss, ta, tb, bo)
            st.rerun()
        return None

    escolhido = por_id.get(str(escolha))
    if escolhido and not e_o_confronto_aberto(escolhido, confronto, resolver):
        st.info("Abrindo {} x {}...".format(escolhido["casa"], escolhido["fora"]))
    return aberto_jogo


def resolver_padrao(times: List[str], raiz: Any = None) -> Callable[[str], str]:
    """Nome da casa -> time do CSV, pelos apelidos salvos e pelo nome
    normalizado. Para a tela inicial, que roda antes do resolver do app."""
    import json
    from pathlib import Path
    from app_core.schedule_import_core import resolve_schedule_team
    base = Path(raiz) if raiz else Path(__file__).resolve().parents[1]
    try:
        apelidos = json.loads((base / "user_team_aliases.json").read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        apelidos = {}
    lista = [str(x) for x in times or []]

    def _r(nome: str) -> str:
        time, ok = resolve_schedule_team(str(nome or ""), lista, apelidos)
        return time if ok else ""
    return _r


def render_gaveta_inicial(st: Any, *, raiz: Any, times: List[str]) -> None:
    """A gaveta na tela inicial (antes de qualquer confronto): clicar num jogo
    ja abre o confronto analisado e leva para o resto do app (24/10/2026)."""
    from pathlib import Path
    from app_core.mercado_vencedor_core import carregar_polymarket
    from app_core.odds_coletadas_core import carregar_odds_coletadas
    odds = carregar_odds_coletadas(Path(raiz))
    poly = carregar_polymarket(Path(raiz))
    if odds.get("motivo") and not (poly.get("partidas") or []):
        return
    st.markdown("#### Jogos encontrados")
    st.caption("Clique num jogo para abrir o confronto, ou escolha os times acima.")
    render_gaveta(st, odds={} if odds.get("motivo") else odds, poly=poly, confronto={},
                  resolver=resolver_padrao(times, raiz), times=list(times or []))
