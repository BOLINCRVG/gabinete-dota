# -*- coding: utf-8 -*-
"""A aba que mostra onde a casa paga mais do que o modelo acha que vale.

O PEDIDO DELE, com as palavras dele (15/09/2026):
    "dai eu nao vou precisar mais ficar colocando linha, tu vai pegar a linha
     desses sites e ja procurar a melhor linha para mim"
e, sobre o formato: SO ONDE HA VANTAGEM, ordenado pela vantagem.

O QUE ESTA TELA NAO FAZ, E POR QUE ISSO APARECE NELA
Ela precifica TOTAIS por mapa (kills e, quando a casa abrir, torre, dragao,
barao, inibidor). Ela NAO precifica vencedor, placar exato, total de mapas nem
handicap -- isso vem do modelo de vencedor, que hoje so e chamavel de dentro
do fluxo de um confronto, com dezenas de regulagens da barra lateral. Ligar
isso aqui por um atalho criaria um SEGUNDO numero para a mesma pergunta, e
este projeto ja tem historia com isso.

Entao a tela DIZ quantas linhas ficaram de fora e por que, mercado por
mercado. Uma tela que mostra tres apostas e cala duzentas parece completa e
nao e -- foi exatamente esse tipo de silencio que segurou kills por horas.

NADA AQUI VAI A REDE. Le-se o arquivo que o robo gravou.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from app_core.odds_coletadas_core import (
    carregar_diario,
    carregar_odds_coletadas,
    houve_bloqueio,
    resumo_da_corrida,
)
from app_core.oportunidades_core import (
    bo_bate,
    mesmo_confronto,
    montar_linhas_justas,
    montar_oportunidades,
    nome_que_falta_casar,
    prob_do_vencedor,
)


def _fmt(v: Any, casas: int = 2, sufixo: str = "") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:
        return "—"
    return "{:.{}f}{}".format(f, int(casas), sufixo)


def render_oportunidades_tab(*, st: Any, ctx: Mapping[str, Any], parte: str = "tudo") -> None:
    """Desenha a aba "Modelo x Mercado". `ctx` traz o caminho de PRODUCAO.

    REORGANIZADA EM 23/10/2026 (pedido dele: manter so a Linha justa como
    estava e melhorar o resto). Ordem e motivo em app_core/modelo_mercado_core.py.
    """
    from app_core.mercado_vencedor_core import (
        carregar_polymarket, render_polymarket_lista, render_vencedor_tres_fontes)
    from app_core.modelo_mercado_core import painel_do_dia

    raiz: Path = Path(ctx.get("raiz") or ".")
    calcular_totais: Callable[..., Any] = ctx["calcular_totais"]
    prob_over: Callable[..., Any] = ctx["prob_over"]
    jogos_por_time = ctx.get("jogos_por_time")
    nao_casados: List[str] = list(ctx.get("times_nao_casados") or [])
    confronto: Mapping[str, Any] = ctx.get("confronto") or {}
    resolver: Optional[Callable[[str], str]] = ctx.get("resolver_time")

    odds = carregar_odds_coletadas(raiz)
    poly = carregar_polymarket(raiz)
    sem_odds = bool(odds.get("motivo"))

    # ---- 1. PAINEL DO DIA ------------------------------------------------
    r: Mapping[str, Any] = {}
    justas: List[Mapping[str, Any]] = []
    if not sem_odds:
        _vencedor = _montar_prob_de_vencedor(st, confronto, resolver)
        r = montar_oportunidades(odds, calcular_totais, prob_over=prob_over,
                                 minimo_pct=0.0, jogos_por_time=jogos_por_time,
                                 prob_de_vencedor=_vencedor,
                                 medias_kills=ctx.get("medias_kills"),
                                 p_mapa_da_casa=_p_mapa_da_casa(confronto, resolver))
        justas = montar_linhas_justas(odds, calcular_totais, prob_over=prob_over,
                                      medias_kills=ctx.get("medias_kills"),
                                      p_mapa_da_casa=_p_mapa_da_casa(confronto, resolver))
    completo = [x for x in (r.get("mercado_completo") or []) if isinstance(x, Mapping)]
    painel = painel_do_dia(completo)

    if sem_odds:
        st.info("Sem odds da Pinnacle para comparar: {}.".format(odds["motivo"]))
    elif odds.get("velha"):
        st.warning("As odds da Pinnacle tem {}. Preco muda: confira na casa antes de apostar."
                   .format(_fmt(odds.get("idade_horas"), 1, " horas")))
    _avisar_bloqueio(st, raiz)

    # ---- 2. AS MELHORES APOSTAS DO DIA (no tema da Linha justa) --------------
    # Desde 25/10/2026 isto e a aba propria "Melhores apostas", no fim (pedido
    # dele); a Modelo x Mercado fica so com o jogo aberto.
    if parte in ("tudo", "apostas"):
        _secao_melhores_apostas(st, ctx=ctx, odds=odds, poly=poly, completo=completo,
                                painel=painel, sem_odds=sem_odds, resolver=resolver)
    if parte == "apostas":
        return

    # ---- 3. A GAVETA DE JOGOS (24/10/2026) -----------------------------------
    # Dias como abas, um botao por jogo, e "+ Escolher times". Clicar abre o
    # confronto. Ver app_core/gaveta_jogos_core.py.
    from app_core.gaveta_jogos_core import jogo_aberto
    from app_core.modelo_mercado_core import linhas_do_jogo
    # A gaveta (escolher jogo/dia) fica no TOPO da tela desde 25/10/2026; aqui
    # so o jogo aberto, para as linhas dele e a Linha justa.
    st.markdown("### O jogo aberto")
    _na_gaveta = jogo_aberto(odds if not sem_odds else {}, poly, confronto, resolver)
    if _na_gaveta is None and confronto.get("teamA"):
        st.caption("Este confronto nao esta nas coletas da Pinnacle nem da Polymarket.")
    _mostrar_casados_sozinhos(st, ctx.get("casados_sozinhos") or {},
                              ctx.get("ensinar_apelido"))
    _jogos_que_dao = False
    if not sem_odds and _na_gaveta is None and confronto.get("teamA") and confronto.get("teamB"):
        _jogos_que_dao = _diagnostico_do_confronto(st, odds, confronto, resolver,
                                                   ctx.get("sugerir_times"), ctx.get("ensinar_apelido"))
    # A ABA LIMPA (29/10/2026, ele: "esta muito poluida, varias coisas falando a mesma coisa"):
    #   1. Suas casas x Pinnacle -- vencedor da serie e +-1,5 de mapas;
    #   2. Linha justa -- modelo x a casa que ele escolher (Pinnacle ou qualquer casa da OddsPapi);
    #   3. As melhores apostas do jogo (casa x modelo x estatistica), embaixo;
    #   4. Detalhes, recolhidos.
    from app_core.suas_casas_core import MARCAS as _MARCAS, linha_justa_da_casa, render_suas_casas
    _ctx_casas = {"calcular_totais": calcular_totais, "prob_over": prob_over,
                  "medias_kills": ctx.get("medias_kills"), "team_games": ctx.get("team_games")}
    if confronto.get("teamA") and confronto.get("teamB"):
        render_suas_casas(st, raiz=raiz, confronto=confronto, resolver=resolver, **_ctx_casas)

    todas = list(r.get("todas") or []) if not sem_odds else []
    por_jogo_cmp: Dict[Any, List[Mapping[str, Any]]] = {}
    for a in todas:
        por_jogo_cmp.setdefault(a.get("jogo"), []).append(a)
    for x in justas:
        por_jogo_cmp.setdefault(x.get("jogo"), []).append({
            "jogo": x.get("jogo"), "liga": x.get("liga"), "casa": x.get("casa"),
            "fora": x.get("fora"), "com_vitoria": x.get("conta") == "com vitoria",
            "vantagem_pct": max([v for v in (x.get("vantagem_a"), x.get("vantagem_b"))
                                 if v is not None] or [-999.0]),
            "_so_resumo": True})
    jogo_cmp = _na_gaveta.get("id") if _na_gaveta else None

    # ---- 2. LINHA JUSTA: modelo x a casa escolhida -----------------------------------
    if confronto.get("teamA") and confronto.get("teamB"):
        st.markdown("#### Linha justa: modelo x casa")
        _nomes_casas = ["Pinnacle"] + [n for _, n in _MARCAS if n != "Pinnacle"]
        _casa_lj = st.selectbox("Casa da coluna da direita", _nomes_casas, key="lj_casa_escolhida",
                                help="Pinnacle pelo robô dela; as outras pela OddsPapi (de 3 em 3 h).")
        if _casa_lj == "Pinnacle":
            if not sem_odds and jogo_cmp in por_jogo_cmp:
                _do_jogo = [a for a in todas if a.get("jogo") == jogo_cmp]
                _jg = next((j for j in (odds.get("jogos") or []) if j.get("id") == jogo_cmp), {})
                _aberto = any(a.get("metrica") == "vencedor" or a.get("com_vitoria") for a in _do_jogo)
                from app_core.tabela_casa_core import vencedores_da_tela
                _venc = vencedores_da_tela(
                    [l for l in (odds.get("linhas") or []) if isinstance(l, dict) and l.get("jogo") == jogo_cmp],
                    _do_jogo)
                _tabelas_da_casa(st, [x for x in justas if x.get("jogo") == jogo_cmp],
                                 _rotulo_do_jogo(por_jogo_cmp[jogo_cmp]).replace("ABERTO: ", ""),
                                 vencedores=_venc,
                                 bo=(confronto.get("bo") if _aberto else None) or _jg.get("bo"),
                                 casa=str(_jg.get("casa") or ""), fora=str(_jg.get("fora") or ""),
                                 titulo_fixo=True)
            else:
                st.caption("A Pinnacle não tem este jogo na leitura atual. Escolha outra casa acima.")
        else:
            _jl, _vl, _mot = linha_justa_da_casa(
                raiz=raiz, confronto=confronto, resolver=resolver, nome_casa=_casa_lj,
                calcular_totais=calcular_totais, prob_over=prob_over, medias_kills=ctx.get("medias_kills"),
                p_mapa_da_casa=_p_mapa_da_casa(confronto, resolver))
            if _mot:
                st.caption(_mot)
            else:
                _tabelas_da_casa(st, _jl, f"{confronto.get('teamA')} x {confronto.get('teamB')} · {_casa_lj}",
                                 vencedores=_vl, bo=confronto.get("bo"), casa=str(confronto.get("teamA")),
                                 fora=str(confronto.get("teamB")), nome_casa=_casa_lj, titulo_fixo=True)

    # ---- 3. AS MELHORES APOSTAS DO JOGO (embaixo) ------------------------------------
    if not sem_odds and jogo_cmp in por_jogo_cmp:
        from app_core.cruzamento_estatistica_core import cruzar as _cruzar
        from app_core.modelo_mercado_core import aposta as _aposta_j
        from app_core.modelo_mercado_html_core import html_destaques as _html_j
        from app_core.modelo_mercado_html_core import pagina as _pagina_j
        _lj = _cruzar(linhas_do_jogo(completo, jogo_cmp), ctx.get("team_games"), resolver)
        if _lj:
            from app_core.modelo_mercado_core import ODD_MODELO_MAX, separar_melhores
            _fortes, _resto = separar_melhores(_lj)

            def _desenhar(linhas):
                try:
                    import streamlit.components.v1 as _cmp
                    _cmp.html(_pagina_j(_html_j(linhas, _aposta_j, limite=40, com_jogo=False)),
                              height=34 + 47 * min(len(linhas), 40) + 60, scrolling=False)
                except Exception:
                    pass
            st.markdown("#### Melhores apostas deste jogo (Pinnacle x modelo x estatística)")
            if _fortes:
                st.caption("Odd justa do modelo abaixo de {:.2f}, da maior vantagem para a menor."
                           .format(ODD_MODELO_MAX).replace(".", ","))
                _desenhar(_fortes)
            else:
                st.caption("Nenhuma aposta deste jogo com odd justa do modelo abaixo de {:.2f}."
                           .format(ODD_MODELO_MAX).replace(".", ","))
            if _resto:
                with st.expander("Outras linhas deste jogo ({})".format(len(_resto))):
                    _desenhar(_resto)
    elif _jogos_que_dao:
        st.info("Este painel nao conseguiu precificar nenhuma linha deste confronto.")

    # ---- 4. DETALHES (recolhidos) ---------------------------------------------
    st.markdown("### Detalhes")
    st.caption("Mercado completo da Pinnacle, réguas, Polymarket, diário do robô e o que ficou de fora.")
    if st.toggle("Mostrar detalhes", value=False, key="mm_mostrar_detalhes"):
        if not sem_odds:
            _mostrar_mercado_completo(st, r)
        _mostrar_reguas_pinnacle(st, raiz)
        render_polymarket_lista(st, raiz=raiz)
        _mostrar_diario(st, raiz)
        if not sem_odds:
            _o_que_ficou_de_fora(st, r, nao_casados)


def _idade_curta(horas: Any) -> str:
    try:
        h = float(horas)
    except (TypeError, ValueError):
        return "?"
    if h < 1:
        return "agora ha pouco"
    if h < 48:
        return "ha {:.0f} h".format(h)
    return "ha {:.0f} dias".format(h / 24)


def _avisar_bloqueio(st: Any, raiz: Path) -> None:
    """So o aviso, no topo; o diario completo fica nos detalhes."""
    try:
        if houve_bloqueio(carregar_diario(raiz, ultimas=6)):
            st.warning("A casa barrou uma das ultimas leituras do robo (Pinnacle). Veja "
                       "'Ultimas corridas do robo' nos detalhes.")
    except Exception:
        pass



def _secao_melhores_apostas(st: Any, *, ctx: Mapping[str, Any], odds: Mapping[str, Any],
                            poly: Mapping[str, Any], completo: List[Mapping[str, Any]],
                            painel: Mapping[str, Any], sem_odds: bool,
                            resolver: Optional[Callable[[str], str]]) -> None:
    """As melhores apostas do dia, cruzadas com a estatistica (a aba propria)."""
    from app_core.modelo_mercado_core import linhas_com_valor, tabela_de_valor

    from app_core.modelo_mercado_core import aposta as _aposta
    from app_core.modelo_mercado_html_core import (
        LIMITE_DESTAQUES, altura_destaques, html_destaques, html_status, pagina)
    top = linhas_com_valor(completo, incluir_suspeitas=False)
    # O CRUZAMENTO (24/10/2026): o modelo ve valor -- a estatistica do historico
    # dos dois times concorda? Ver app_core/cruzamento_estatistica_core.py.
    from app_core.cruzamento_estatistica_core import cruzar as _cruzar
    top = _cruzar(top, ctx.get("team_games"), resolver)
    _contra = sum(1 for x in top[:LIMITE_DESTAQUES] if x.get("estat_veredito") == "discorda")
    st.markdown("### Melhores apostas do dia")
    corpo = html_status(jogos=painel["jogos"], com_valor=painel["linhas_com_valor"],
                        idade_pin="-" if sem_odds else _idade_curta(odds.get("idade_horas")),
                        idade_poly="-" if poly.get("motivo") else _idade_curta(poly.get("idade_horas")),
                        pin_velha=bool(odds.get("velha")))
    if _contra:
        st.warning("Em {} das apostas abaixo a estatistica dos ultimos jogos discorda do "
                   "modelo (coluna Estatistica em vermelho). Confira antes de apostar."
                   .format(_contra))
    corpo += html_destaques(top, _aposta)
    try:
        import streamlit.components.v1 as _componentes
        _componentes.html(pagina(corpo), height=altura_destaques(len(top)), scrolling=False)
    except Exception:
        st.markdown(corpo, unsafe_allow_html=True)
    if len(top) > LIMITE_DESTAQUES:
        with st.expander("Ver todas as {} apostas com valor".format(len(top))):
            st.dataframe(tabela_de_valor(top, com_jogo=True), hide_index=True, width="stretch")

def ordem_dos_jogos(por_jogo: Mapping[Any, List[Mapping[str, Any]]]) -> List[Any]:
    """O confronto aberto primeiro (o que tem conta com a chance de vitoria
    ou vencedor), depois os outros pela maior vantagem."""
    def _chave(j):
        linhas = por_jogo[j]
        aberto = any(x.get("com_vitoria") or x.get("metrica") == "vencedor" for x in linhas)
        melhor = max((x.get("vantagem_pct") or -999.0) for x in linhas)
        return (0 if aberto else 1, -melhor)
    return sorted(por_jogo, key=_chave)


def _rotulo_do_jogo(linhas: List[Mapping[str, Any]]) -> str:
    x = linhas[0] if linhas else {}
    aberto = any(y.get("com_vitoria") or y.get("metrica") == "vencedor" for y in linhas)
    return "{}{} — {} x {}  ({} linhas)".format(
        "ABERTO: " if aberto else "", x.get("liga") or "?", x.get("casa") or "?",
        x.get("fora") or "?", len(linhas))


def _aposta_no_quadro(x: Mapping[str, Any]) -> str:
    """Nome curto do lado, para o quadro do mercado inteiro."""
    lado = str(x.get("lado") or "").strip()
    baixo = lado.lower()
    time = str(x.get("time") or "").lower()
    quem = x.get("casa") if time == "home" else (x.get("fora") if time == "away" else None)
    ponto = x.get("pontos")
    if baixo in ("home", "away"):
        nome = str(x.get("casa") if baixo == "home" else x.get("fora"))
        if ponto is not None and str(x.get("tipo") or "") == "spread":
            nome += " {:+.1f}".format(float(ponto)).replace(".", ",")
        return nome
    if baixo in ("over", "under"):
        txt = "{} {}".format("acima de" if baixo == "over" else "abaixo de",
                             _fmt(ponto, 1).replace(".", ","))
        return "{}: {}".format(quem, txt) if quem else txt
    return lado or "?"


def _ordem_no_quadro(x: Mapping[str, Any]):
    v = x.get("vantagem_pct")
    try:
        p = float(x.get("pontos") or 0)
    except (TypeError, ValueError):
        p = 0.0
    return (-(v if v is not None else -1e9),
            str(x.get("nome_pt") or x.get("mercado") or ""), p)


def _mostrar_mercado_completo(st: Any, r: Mapping[str, Any]) -> None:
    """A Pinnacle INTEIRA, jogo por jogo (pedido dele em 23/09/2026: "de
    forma total, 100%").

    Toda linha que a casa oferece, com o preco justo dela. Onde o modelo sabe
    precificar, o dele ao lado e a vantagem; onde nao sabe, o motivo -- nunca
    um numero inventado.
    """
    completo = [x for x in (r.get("mercado_completo") or []) if isinstance(x, Mapping)]
    if not completo:
        return
    por_jogo: Dict[Any, List[Mapping[str, Any]]] = {}
    for x in completo:
        por_jogo.setdefault(x.get("jogo"), []).append(x)
    ordem = sorted(por_jogo, key=lambda j: str(por_jogo[j][0].get("inicio_utc") or ""))
    with st.expander("O mercado inteiro da Pinnacle, jogo por jogo ({} jogos, {} linhas)".format(
            len(ordem), len(completo)), expanded=False):
        jogo = st.selectbox(
            "Jogo", ordem,
            format_func=lambda j: "{} — {} x {}  ({})".format(
                por_jogo[j][0].get("liga") or "?", por_jogo[j][0].get("casa") or "?",
                por_jogo[j][0].get("fora") or "?",
                str(por_jogo[j][0].get("inicio_utc") or "")[5:16].replace("T", " ")),
            key="oportunidades_jogo_completo")
        so_modelo = st.checkbox("So as linhas que o modelo sabe precificar",
                                value=False, key="oportunidades_so_modelo")
        linhas = [x for x in por_jogo.get(jogo) or []
                  if not so_modelo or x.get("p_modelo") is not None]
        linhas.sort(key=_ordem_no_quadro)
        st.dataframe(tabela_do_mercado_completo(linhas), hide_index=True,
                     width="stretch")
        st.caption("Ordenado pela vantagem. **Pinnacle sem margem** e a odd "
                   "dela com a margem tirada pelo metodo da potencia (o que "
                   "melhor acertou nos estudos -- fontes em "
                   "app_core/preco_justo_casa_core.py). Linhas sem modelo "
                   "mostram so o preco da casa, com o motivo.")


def tabela_do_mercado_completo(linhas: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """As colunas do quadro. Separado da tela para dar para testar."""
    def _r(v: Any, casas: int) -> Any:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return round(f, casas) if f == f else None
    return [{
        "mercado": x.get("nome_pt") or x.get("mercado"),
        "aposta": _aposta_no_quadro(x),
        "casa paga": _r(x.get("odd_casa"), 2),
        "pinnacle sem margem": _r(x.get("odd_casa_justa"), 2),
        "modelo": _r(x.get("odd_modelo"), 2),
        "vantagem %": _r(x.get("vantagem_pct"), 1),
        "modelo x pinnacle (pts)": _r(x.get("diferenca_pp"), 1),
        "margem %": _r(x.get("margem_pct"), 1),
        # Handicap e total por time: com a chance de vitoria do cartao (so o
        # confronto aberto) ou so com os kills medios.
        "conta": ("com vitoria" if x.get("com_vitoria") else
                  ("so kills" if x.get("com_vitoria") is False else "")),
        "por que sem modelo": x.get("motivo") or "",
    } for x in linhas]


def _mostrar_diario(st: Any, raiz: Path) -> None:
    """As ultimas corridas do robo, na tela.

    ELE PERGUNTOU (16/09/2026) se tudo tinha funcionado durante a noite, e eu
    nao soube responder: o coletor parava em SILENCIO quando a casa dizia
    403/429, entao uma janela bloqueada ficava identica a uma que nem rodou.
    Agora toda corrida deixa linha -- e ela aparece AQUI, porque diario que
    exige abrir o GitHub para ler devolve o problema pela metade.
    """
    corridas = carregar_diario(raiz, ultimas=6)
    if not corridas:
        return
    with st.expander("Ultimas corridas do robo ({})".format(len(corridas))):
        for c in corridas:
            st.caption("· {}".format(resumo_da_corrida(c)))


def reguas_pinnacle(raiz: Path) -> List[Dict[str, Any]]:
    """As linhas do placar "modelo x Pinnacle", lidas de evidence/.

    Quem grava sao as reguas (tools/regua_pinnacle_comum.py); o app so le --
    nada aqui roda conta nem vai a rede.
    """
    import json as _json
    fora: List[Dict[str, Any]] = []
    for arq in sorted((Path(raiz) / "evidence").glob("regua_pinnacle_*.json")):
        try:
            d = _json.loads(arq.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for chave, conta in (("so_kills", ""), ("com_vitoria", " (com a chance de vitoria)"),
                             ("polymarket_x_pinnacle", "")):
            r = d.get(chave)
            if not isinstance(r, dict):
                continue
            dif = float(r.get("diferenca") or 0.0)
            quem = "Polymarket" if chave == "polymarket_x_pinnacle" else "modelo"
            if r.get("conclusivo"):
                veredito = (quem + " acerta mais") if dif < 0 else "Pinnacle acerta mais"
            else:
                veredito = "ainda nao da para concluir"
            fora.append({
                "mercado": str(d.get("mercado") or arq.stem) + conta,
                "partidas": "{} de {}".format(r.get("partidas"), r.get("minimo")),
                "nota do modelo": round(float(r.get("nota_modelo") or 0), 1),
                "nota da Pinnacle": round(float(r.get("nota_casa") or 0), 1),
                "quem acertou mais ate agora": quem if dif < 0 else "Pinnacle",
                "partidas em que o modelo ganhou": r.get("bateu"),
                "conclusao": veredito,
                "medido em": str(d.get("gerado_em") or "")[:10],
            })
    return fora


def _mostrar_reguas_pinnacle(st: Any, raiz: Path) -> None:
    """O placar "quem acerta mais: o modelo ou a Pinnacle" (25/09/2026).

    Ele perguntou a nota do modelo "em relacao as notas da pinnacle". As
    reguas ja respondiam, mas so no terminal. Mesma escala da nota do
    projeto: 5,0 = cara ou coroa, 9,5 = elite.
    """
    linhas = reguas_pinnacle(raiz)
    if not linhas:
        return
    with st.expander("Modelo x Pinnacle: quem acerta mais (as reguas)", expanded=False):
        st.dataframe(linhas, hide_index=True, width="stretch")
        st.caption(
            "Mesmas apostas, contra o que aconteceu, com a margem da Pinnacle tirada. "
            "Nota: 5,0 = cara ou coroa, 9,5 = elite. \"Ainda nao da para concluir\" "
            "quer dizer que a diferenca cabe na margem de erro ou que faltam partidas "
            "(o minimo esta ao lado).")


def _ensinar_apelido(st: Any, sugerir: Optional[Callable[[str], List[str]]],
                     ensinar: Optional[Callable[[str, str], None]],
                     cru: str) -> None:
    """O conserto de um clique, no lugar onde o problema aparece.

    Mesma maquina que a aba Jogos usa: o apelido fica salvo no arquivo dele e
    serve os dois lugares. NAO SE ADIVINHA -- "Liquid" poderia ser outro time,
    e analisar o confronto errado com toda a confianca e o pior desfecho
    possivel (PRINCIPIO IV). Ele escolhe.
    """
    if sugerir is None or ensinar is None:
        st.caption("Para casar o nome, use o mapeamento de apelidos na aba Jogos.")
        return
    try:
        opcoes = list(sugerir(cru))[:12]
    except Exception:
        opcoes = []
    if not opcoes:
        st.caption("Nao achei palpite parecido para **{}** no CSV.".format(cru))
        return
    # QUAO VELHO E O DADO de cada palpite, ao lado do nome. Um apelido certo
    # com dado de um ano atras e um apelido certo com dado de ontem eram
    # indistinguiveis aqui ate 21/09/2026.
    from app_core.frescor_do_time_core import (
        avisar_se_velho as _avisar_velho,
        legenda as _legenda_frescor,
        opcoes as _opcoes_frescor,
    )
    _ultimos = {}
    try:
        _ultimos = getattr(st, "session_state", {}).get("_ultimo_jogo_por_time") or {}
    except (AttributeError, TypeError):
        _ultimos = {}
    _rotulos, _de_volta = _opcoes_frescor(opcoes, _ultimos)
    if _ultimos:
        st.caption(_legenda_frescor())
    # O rotulo carrega a data; o apelido tem de guardar o NOME do time.
    escolha = _de_volta.get(st.selectbox(
        "Quem e **{}** no CSV?".format(cru), options=["", *_rotulos],
        key="oport_apelido_{}".format(cru.lower().replace(" ", "_")),
    ), "")
    if escolha:
        _av = _avisar_velho(escolha, _ultimos)
        if _av:
            st.warning(_av)
    if escolha and st.button("Salvar e comparar",
                             key="oport_apelido_btn_{}".format(cru.lower().replace(" ", "_"))):
        try:
            ensinar(cru, str(escolha))
        except Exception:
            st.error("Nao consegui salvar o apelido.")
            return
        st.success("Salvo: **{}** = **{}**.".format(cru, escolha))
        _recarregar(st)


def _recarregar(st: Any) -> None:
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(fn):
        fn()


def _diagnostico_do_confronto(st: Any, odds: Mapping[str, Any],
                              confronto: Mapping[str, Any],
                              resolver: Optional[Callable[[str], str]],
                              ctx_sugerir: Optional[Callable[[str], List[str]]] = None,
                              ctx_ensinar: Optional[Callable[[str, str], None]] = None) -> bool:
    """Diz POR QUE a tela pode estar vazia, em vez de deixar ele adivinhar.

    ELE PERGUNTOU DUAS VEZES (16/09/2026): "nao entendi, ate agora porque nao
    esta mostrando nada, voce nao disse que ja estava conseguindo pegar pelo
    menos as odds dos ml's?".

    A resposta era simples e a tela nao dava: o confronto que ele tinha aberto
    era eSuba x Meavedron, do Hitpoint Masters -- e a Pinnacle NAO OFERECE
    Hitpoint Masters. Nao havia o que comparar.

    So que a tela dizia "nenhuma aposta com vantagem", que soa como "procurei
    e nao achei nada bom". Sao coisas MUITO diferentes para quem vai apostar,
    e viraram a mesma frase. Agora ela diz qual e o confronto aberto, se a
    casa tem esse jogo, e QUAIS jogos ela tem -- que e a informacao que
    resolve: basta abrir um deles.

    Devolve True quando havia como precificar alguma coisa.
    """
    jogos = [j for j in (odds.get("jogos") or []) if isinstance(j, dict)]
    time_a = str((confronto or {}).get("teamA") or "")
    time_b = str((confronto or {}).get("teamB") or "")

    if not (time_a and time_b):
        st.info("Escolha um confronto e clique em **Analisar Confronto**. "
                "O mercado de vencedor e comparado para o confronto aberto.")
        _listar_o_que_a_casa_tem(st, jogos)
        return False

    achou = None
    if resolver is not None:
        for j in jogos:
            if mesmo_confronto(j, time_a, time_b, resolver) is not None:
                achou = j
                break

    if achou is None:
        # "A CASA NAO TEM ESSE JOGO" e "A CASA TEM, MAS O NOME NAO CASOU" sao
        # coisas diferentes e pedem acoes diferentes. Ele mandou print
        # (16/09/2026) com a tela dizendo "nao esta na Pinnacle" e listando,
        # tres linhas abaixo, o MESMO jogo: a casa escreve "Liquid" e o CSV
        # tem "Team Liquid". A mensagem contradizia a lista embaixo dela.
        quase = None
        cru = ""
        if resolver is not None:
            for j in jogos:
                faltando = nome_que_falta_casar(j, time_a, time_b, resolver)
                if faltando:
                    quase, cru = j, faltando
                    break
        if quase is not None:
            st.warning(
                "A Pinnacle TEM este confronto (**{} x {}**), mas o nome "
                "**{}** que ela usa nao casa com nenhum time do CSV. Ensine "
                "uma vez e nao volta mais.".format(
                    quase.get("casa"), quase.get("fora"), cru))
            _ensinar_apelido(st, ctx_sugerir, ctx_ensinar, cru)
            return False

        st.warning(
            "O confronto aberto (**{} x {}**) nao esta na Pinnacle, entao nao "
            "ha odd dela para comparar. Ela cobre poucas ligas.".format(
                time_a, time_b))
        _listar_o_que_a_casa_tem(st, jogos)
        return False

    st.caption("Comparando o confronto aberto: **{} x {}** (BO{} na casa).".format(
        achou.get("casa"), achou.get("fora"), achou.get("bo")))
    return True


def _listar_o_que_a_casa_tem(st: Any, jogos: List[Mapping[str, Any]]) -> None:
    """Os jogos que a casa oferece. Abrir um deles e o que resolve."""
    if not jogos:
        st.caption("A ultima coleta nao trouxe jogo nenhum da Pinnacle.")
        return
    st.caption("Jogos que a Pinnacle oferece nesta leitura -- abra um deles "
               "para o painel ter o que comparar:")
    for j in jogos:
        st.caption("· {} — {} x {} (BO{})".format(
            j.get("liga"), j.get("casa"), j.get("fora"), j.get("bo")))


def _montar_prob_de_vencedor(st: Any, confronto: Mapping[str, Any],
                             resolver: Optional[Callable[[str], str]]):
    """A odd justa do VENCEDOR, vinda do confronto aberto na tela.

    ELE ESTRANHOU a aba mostrando zero: "nao faz mt sentido". Tinha razao --
    o numero ja existe. O card do confronto mostra, por exemplo, MD5 1,66 x
    2,51. Faltava traze-lo ate aqui.

    NAO SE RECALCULA NADA. O `p_map`/`p_series` sao os MESMOS que ele ve no
    card; o projeto ja teve tres implementacoes do pre-jogo com contagens
    diferentes, e nao vai ganhar uma quarta.

    Por isso so vale para o confronto ABERTO -- e o unico para o qual o app
    calculou o numero. Se nenhum jogo da casa for esse confronto, a tela diz
    isso em vez de ficar calada.
    """
    if not confronto or resolver is None:
        return None
    time_a = str(confronto.get("teamA") or "")
    time_b = str(confronto.get("teamB") or "")
    p_map = confronto.get("p_map")
    p_series = confronto.get("p_series")
    bo_tela = confronto.get("bo")
    if not (time_a and time_b):
        return None

    avisado = {"bo": False}

    def _prob(jogo, linha):
        lado_certo = mesmo_confronto(jogo, time_a, time_b, resolver)
        if lado_certo is None:
            return None
        if not bo_bate(jogo, bo_tela):
            # UM BO5 PRECIFICADO COMO BO3 muda a probabilidade da serie
            # inteira. Fica de fora, e dito uma vez.
            if not avisado["bo"]:
                avisado["bo"] = True
                st.warning(
                    "A casa oferece este confronto em BO{} e a tela esta em "
                    "BO{}. O mercado de vencedor da serie ficou de fora: o "
                    "formato muda a probabilidade.".format(
                        jogo.get("bo"), bo_tela))
            try:
                if int(linha.get("periodo", 0) or 0) == 0:
                    return None
            except (TypeError, ValueError):
                return None
        return prob_do_vencedor(linha, p_map=p_map, p_series=p_series,
                                casa_e_time_a=bool(lado_certo), bo=bo_tela)

    return _prob


def _p_mapa_da_casa(confronto: Mapping[str, Any],
                    resolver: Optional[Callable[[str], str]]):
    """Chance de o MANDANTE da casa vencer um mapa -- o numero do cartao.

    So para o confronto aberto, como o vencedor: e o unico para o qual o app
    calculou a chance. Entra no handicap de kills e no total por time
    (mlcore/abates_por_time.py). Nao depende do BO: e por mapa.
    """
    if not confronto or resolver is None:
        return None
    time_a = str(confronto.get("teamA") or "")
    time_b = str(confronto.get("teamB") or "")
    try:
        p_map = float(confronto.get("p_map"))
    except (TypeError, ValueError):
        return None
    if not (time_a and time_b) or not 0.0 < p_map < 1.0:
        return None

    def _p(jogo):
        lado_certo = mesmo_confronto(jogo, time_a, time_b, resolver)
        if lado_certo is None:
            return None
        return p_map if lado_certo else 1.0 - p_map

    return _p


def _mostrar_casados_sozinhos(st: Any, casados: Mapping[str, Any],
                              ensinar: Optional[Callable[[str, str], None]]) -> None:
    """O que a regra de casar nome decidiu sozinha, com o botao de guardar.

    Esconder isto seria o erro do UP2U de volta, so que calado: ele veria o
    preco e nao saberia que o time foi escolhido por uma regra.
    """
    if not casados:
        return
    from app_core.casar_time_core import motivo
    with st.expander("Times casados sozinhos ({}) -- confira".format(len(casados)),
                     expanded=False):
        for cru, par in sorted(casados.items()):
            escolhido, adv = (list(par) + ["", ""])[:2]
            st.write(motivo(cru, escolhido, adv))
            if ensinar is not None and st.button(
                    "Esta certo, guardar \"{}\" = {}".format(cru, escolhido),
                    key="oport_auto_ok_{}".format(str(cru).lower().replace(" ", "_"))):
                try:
                    ensinar(str(cru), str(escolhido))
                    st.success("Guardado: \"{}\" = {}.".format(cru, escolhido))
                except Exception:
                    st.error("Nao consegui guardar o apelido.")
        st.caption("Nada disto foi gravado sem voce: e usado so para mostrar o "
                   "preco. Se algum estiver errado, ensine o certo na aba Jogos.")


def _o_que_ficou_de_fora(st: Any, r: Mapping[str, Any], nao_casados: List[str]) -> None:
    """A parte que impede a tela de mentir por omissao."""
    fora = r.get("nao_precificados") or []
    sem_modelo = r.get("sem_modelo") or []
    total_fora = sum(int(g.get("linhas") or 0) for g in fora)
    if not (total_fora or sem_modelo or nao_casados):
        return

    with st.expander("O que este painel NAO olhou ({} linha(s))".format(total_fora)):
        if nao_casados:
            st.markdown("**Times que nao casaram com o CSV**")
            st.caption("Sem casar o nome nao ha historico, e sem historico nao ha "
                       "odd justa. Ensine o apelido uma vez na aba Jogos e nao volta.")
            st.write(", ".join(escape(n) for n in nao_casados))
        if sem_modelo:
            st.markdown("**Confrontos sem numero do modelo**")
            for s in sem_modelo:
                st.caption("· {} x {} (mapa {})".format(
                    s.get("casa"), s.get("fora"), str(s.get("mapa") or "")[1:]))
        if fora:
            st.markdown("**Mercados que este painel ainda nao precifica**")
            for g in fora:
                st.caption("· {} — {} linha(s) — {}".format(
                    g.get("mercado"), g.get("linhas"), g.get("motivo")))


# ---------------------------------------------------------------------------
# LINHA JUSTA NO FORMATO DA BET365 (25/09/2026)
#
# Ele mandou o print da bet365 e pediu o MESMO desenho: tres tabelas (Fair,
# Modelo, Pinnacle), clique DENTRO da linha, a Fair fixa e Modelo/Pinnacle
# andando juntas. O desenho e o clique vivem em app_core/tabela_casa_core.py.
# ---------------------------------------------------------------------------


def _tabelas_da_casa(st: Any, justas: List[Mapping[str, Any]], titulo: str, *,
                     vencedores: List[Mapping[str, Any]] = (), bo: Any = None,
                     casa: str = "", fora: str = "", nome_casa: str = "Pinnacle",
                     titulo_fixo: bool = False) -> None:
    from app_core.tabela_casa_core import altura, html_das_tabelas, linhas_da_tela
    linhas = linhas_da_tela(justas)
    if not linhas and not vencedores:
        return
    if not titulo_fixo:
        st.markdown(f"#### Linha justa: modelo x {nome_casa}")
    try:
        import streamlit.components.v1 as _componentes
    except Exception:
        st.info("Esta versao do Streamlit nao desenha a tabela no formato da casa.")
        return
    _componentes.html(html_das_tabelas(linhas, titulo, vencedores=vencedores, bo=bo,
                                       casa=casa, fora=fora, nome_casa=nome_casa),
                      height=altura(linhas), scrolling=True)
