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
    carregar_movimento,
    carregar_odds_coletadas,
    houve_bloqueio,
    movimento_da_linha,
    resumo_da_corrida,
)
from app_core.oportunidades_core import (
    bo_bate,
    mesmo_confronto,
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


def _linha_do_jogo(a: Mapping[str, Any]) -> str:
    return "{} x {}".format(a.get("casa") or "?", a.get("fora") or "?")


def _descricao_da_aposta(a: Mapping[str, Any]) -> str:
    """"kills do mapa 1 acima de 31,5" -- em vez do jargao da casa."""
    if str(a.get("metrica") or "") == "vencedor":
        lado = str(a.get("lado") or "").lower()
        quem = a.get("casa") if lado == "home" else a.get("fora")
        return "{} vencer ({})".format(quem, a.get("mercado") or "vencedor")
    lado = str(a.get("lado") or "").lower()
    palavra = "acima de" if lado == "over" else "abaixo de"
    metrica = {"kills": "kills", "towers": "torres", "dragons": "dragoes",
               "barons": "baroes", "inhibitors": "inibidores"}.get(
                   str(a.get("metrica") or ""), str(a.get("metrica") or ""))
    mapa = str(a.get("mapa") or "")
    nmapa = mapa[1:] if mapa.startswith("m") else mapa
    ponto = _fmt(a.get("pontos"), 1).replace(".", ",")
    return "{} do mapa {} {} {}".format(metrica, nmapa, palavra, ponto)


def render_oportunidades_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    """Desenha a aba. `ctx` traz o caminho de PRODUCAO, nunca uma copia dele."""
    raiz: Path = Path(ctx.get("raiz") or ".")
    calcular_totais: Callable[..., Any] = ctx["calcular_totais"]
    prob_over: Callable[..., Any] = ctx["prob_over"]
    jogos_por_time = ctx.get("jogos_por_time")
    nao_casados: List[str] = list(ctx.get("times_nao_casados") or [])
    confronto: Mapping[str, Any] = ctx.get("confronto") or {}
    resolver: Optional[Callable[[str], str]] = ctx.get("resolver_time")

    st.markdown("### Onde a Pinnacle paga mais do que o modelo acha que vale")

    odds = carregar_odds_coletadas(raiz)
    if odds.get("motivo"):
        st.info("Sem odds para comparar: {}.".format(odds["motivo"]))
        st.caption("Quem coleta e o robo do GitHub, de 3 em 3 horas. "
                   "O app nunca vai ao site da casa.")
        return

    idade = odds.get("idade_horas")
    carimbo = "lido ha {} ({})".format(
        "menos de 1 hora" if (idade is not None and idade < 1) else _fmt(idade, 1, " horas"),
        odds.get("lido_em") or "sem carimbo")
    if odds.get("velha"):
        st.warning("Estas odds tem {}. Preco de aposta muda: confira na casa "
                   "antes de apostar.".format(_fmt(idade, 1, " horas")))
    else:
        st.caption(carimbo)

    _mostrar_diario(st, raiz)
    _mostrar_casados_sozinhos(st, ctx.get("casados_sozinhos") or {},
                              ctx.get("ensinar_apelido"))
    _jogos_que_dao = _diagnostico_do_confronto(
        st, odds, confronto, resolver,
        ctx.get("sugerir_times"), ctx.get("ensinar_apelido"))

    _vencedor = _montar_prob_de_vencedor(st, confronto, resolver)
    r = montar_oportunidades(odds, calcular_totais, prob_over=prob_over,
                             minimo_pct=0.0, jogos_por_time=jogos_por_time,
                             prob_de_vencedor=_vencedor,
                             medias_kills=ctx.get("medias_kills"))

    todas = list(r.get("todas") or [])
    # DE ONDE CADA PRECO VEIO. Nao e a abertura -- e o primeiro que NOS vimos.
    _mov = carregar_movimento(raiz)
    _por_chave = {str(l.get("chave") or ""): l for l in (odds.get("linhas") or [])
                  if isinstance(l, dict)}
    for _a in todas:
        _crua = _por_chave.get(_chave_do_achado(_a, _por_chave))
        _a["_movimento"] = movimento_da_linha(_crua or {}, _mov)
    if todas:
        # O MERCADO PRIMEIRO, a aposta depois. Foi o que ele pediu: "eu
        # selecionaria por exemplo, Vencedor e voce me mostraria a tela com a
        # odd do modelo e ao lado a odd da casa".
        por_mercado: Dict[str, List[Mapping[str, Any]]] = {}
        for a in todas:
            por_mercado.setdefault(str(a.get("mercado") or "?"), []).append(a)
        nomes = sorted(por_mercado, key=lambda k: (-max(
            (x.get("vantagem_pct") or -999) for x in por_mercado[k]), k))

        escolhido = st.selectbox(
            "Qual mercado voce quer comparar",
            nomes,
            format_func=lambda k: "{}  (melhor: {:+.1f}%)".format(
                k, max((x.get("vantagem_pct") or 0.0) for x in por_mercado[k])),
            key="oportunidades_mercado",
        )
        linhas = sorted(por_mercado.get(escolhido) or [], key=_ordem_da_linha)
        _tabela_comparada(st, escolhido, linhas)
        st.caption(
            "**Modelo** e a odd que o confronto aberto calcula; **casa** e a "
            "que a Pinnacle esta pagando; **Pinnacle sem margem** e a odd dela "
            "com a margem tirada -- o preco que ela acha justo. Embaixo dele, "
            "quantos pontos de chance o modelo da a mais (+) ou a menos (-) "
            "que ela. Vantagem positiva quer dizer que a casa paga mais do que "
            "o modelo acha que vale."
            "  ·  entra quando a casa paga acima da odd do modelo.")
    elif _jogos_que_dao:
        st.info("Este painel nao conseguiu precificar nenhuma linha deste "
                "confronto.")

    _mostrar_mercado_completo(st, r)
    _o_que_ficou_de_fora(st, r, nao_casados)


def _texto_da_diferenca(a: Mapping[str, Any]) -> str:
    """"modelo +8 pts": quanto de chance o modelo da a mais que a Pinnacle."""
    try:
        d = float(a.get("diferenca_pp"))
    except (TypeError, ValueError):
        return ""
    if d != d:
        return ""
    return "modelo {:+.0f} pts".format(d)


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
        "por que sem modelo": x.get("motivo") or "",
    } for x in linhas]


# ---------------------------------------------------------------------------
# A TELA DE COMPARACAO (16/09/2026)
#
# Ele pediu, olhando a lista de cartoes com aviso: "acho que e melhor fazer
# uma tela comparada, a linha da casa de uma cor, com a linha do modelo de
# outra, talvez ao lado... eu selecionaria por exemplo, Vencedor e voce me
# mostraria a 'tela' da visao geral, com a odd do modelo e ao lado a odd da
# casa junto com a vantagem".
#
# A lista antiga tinha um problema alem do visual: so mostrava o lado COM
# vantagem. O lado sem vantagem e metade da informacao -- e olhando o PAR que
# se ve se a discordancia e do mercado ou do modelo. Se o modelo da 56% e a
# casa da 33%, o buraco aparece nos dois lados ao mesmo tempo.
#
# Um unico bloco HTML, como o card da Visao Geral: elimina por construcao o
# desalinhamento de colunas do Streamlit.
# ---------------------------------------------------------------------------

_COR_MODELO = "#2daac4"   # ciano — o mesmo accent do app
_COR_CASA = "#d9a441"     # ambar — a casa; sem semantica de bom/ruim
_COR_BOA = "#3fb950"
_COR_RUIM = "#c9736a"


def _nome_do_lado(a: Mapping[str, Any]) -> str:
    """Como chamar este lado da aposta, em portugues."""
    lado = str(a.get("lado") or "").strip()
    baixo = lado.lower()
    if baixo in ("home", "away"):
        nome = str(a.get("casa") or "Casa") if baixo == "home" else str(a.get("fora") or "Fora")
        # Handicap: sem o numero, "Liquid" nao diz qual aposta e.
        if str(a.get("tipo") or "") == "spread" and a.get("pontos") is not None:
            nome += " {:+.1f}".format(float(a["pontos"])).replace(".", ",")
        return nome
    ponto = _fmt(a.get("pontos"), 1).replace(".", ",")
    # Total por time: DE QUEM e o total.
    time = str(a.get("time") or "").lower()
    dono = a.get("casa") if time == "home" else (a.get("fora") if time == "away" else None)
    prefixo = "{}: ".format(dono) if dono else ""
    if baixo == "over":
        return "{}acima de {}".format(prefixo, ponto)
    if baixo == "under":
        return "{}abaixo de {}".format(prefixo, ponto)
    return lado or "?"


def _texto_do_movimento(a: Mapping[str, Any]) -> str:
    """De onde este preco veio, sem chamar de "abertura".

    ELE OBSERVOU (16/09/2026): "voce pode estar pegando algo que ja mudou de
    odd ja tem um tempo... ate porque voce nao pega a abertura". Certissimo: o
    robo le de 3 em 3 horas, entao o primeiro preco que vemos e o que estava
    la quando ele passou.

    Entao a tela diz "visto desde", nunca "abriu em". O nome errado
    transformaria uma limitacao conhecida numa afirmacao falsa. E uma leitura
    so NAO vira "0% de movimento": foto nao e serie.
    """
    m = a.get("_movimento") or {}
    if not m.get("sabido"):
        if m.get("leituras"):
            return "visto 1x"
        return ""
    v = float(m.get("variacao_pct") or 0.0)
    seta = "sobe" if v > 0 else "cai"
    return "{} {:+.1f}% desde {}".format(seta, v, str(m.get("quando") or "")[5:16].replace("T", " "))


def _linha_comparada(a: Mapping[str, Any]) -> str:
    v = a.get("vantagem_pct")
    try:
        vf = float(v)
    except (TypeError, ValueError):
        vf = float("nan")
    cor_v = _COR_BOA if vf > 0 else _COR_RUIM
    marcas = ""
    if a.get("suspeito"):
        marcas += ("<span style='background:rgba(217,164,65,0.18);border:1px solid "
                   "rgba(217,164,65,0.55);border-radius:8px;padding:1px 7px;"
                   "font-size:0.72em;margin-left:8px;'>grande demais</span>")
    if a.get("dados_finos"):
        marcas += ("<span style='background:rgba(201,115,106,0.16);border:1px solid "
                   "rgba(201,115,106,0.5);border-radius:8px;padding:1px 7px;"
                   "font-size:0.72em;margin-left:6px;'>poucos jogos</span>")
    limite = a.get("limite")
    return (
        "<div style='display:grid;grid-template-columns:minmax(110px,1.4fr) "
        "repeat(4,minmax(58px,0.55fr));gap:10px;max-width:700px;"
        "align-items:center;padding:7px 4px;"
        "border-top:1px solid rgba(255,255,255,0.07);'>"
        "<div><b>{lado}</b>{marcas}<div style='opacity:0.45;font-size:0.74em;'>"
        "limite {lim}</div></div>"
        "<div style='text-align:right;color:{cm};font-weight:700;'>{justa}</div>"
        "<div style='text-align:right;opacity:0.75;'>{justa_casa}"
        "<div style='opacity:0.6;font-size:0.7em;'>{dif}</div></div>"
        "<div style='text-align:right;color:{cc};font-weight:700;'>{casa}"
        "<div style='opacity:0.45;font-size:0.7em;font-weight:400;'>{mov}</div></div>"
        "<div style='text-align:right;color:{cv};font-weight:800;'>{vant}</div>"
        "</div>"
    ).format(
        lado=escape(_nome_do_lado(a)), marcas=marcas,
        lim=(limite if limite is not None else "—"),
        cm=_COR_MODELO, justa=_fmt(a.get("odd_justa")),
        justa_casa=_fmt(a.get("odd_casa_justa")), dif=escape(_texto_da_diferenca(a)),
        cc=_COR_CASA, casa=_fmt(a.get("odd_casa")),
        mov=escape(_texto_do_movimento(a)),
        cv=cor_v, vant=("{:+.2f}%".format(vf) if vf == vf else "—"),
    )


def _tabela_comparada(st: Any, titulo: str, linhas: List[Mapping[str, Any]]) -> None:
    cabeca = (
        "<div style='display:grid;grid-template-columns:minmax(110px,1.4fr) "
        "repeat(4,minmax(58px,0.55fr));gap:10px;max-width:700px;"
        "padding:2px 4px;opacity:0.55;"
        "font-size:0.76em;text-transform:uppercase;letter-spacing:0.05em;'>"
        "<div>aposta</div>"
        "<div style='text-align:right;color:{cm};'>modelo</div>"
        "<div style='text-align:right;'>pinnacle sem margem</div>"
        "<div style='text-align:right;color:{cc};'>casa</div>"
        "<div style='text-align:right;'>vantagem</div></div>"
    ).format(cm=_COR_MODELO, cc=_COR_CASA)
    corpo = "".join(_linha_comparada(a) for a in linhas)
    st.markdown(
        "<div style='background:rgba(255,255,255,0.025);border:1px solid "
        "rgba(255,255,255,0.09);border-top:3px solid {cm};border-radius:14px;"
        "padding:12px 16px 10px;max-width:760px;'>"
        "<div style='opacity:0.6;font-size:0.8em;text-transform:uppercase;"
        "letter-spacing:0.06em;margin-bottom:4px;'>{t}</div>"
        "{cab}{corpo}</div>".format(cm=_COR_MODELO, t=escape(titulo),
                                    cab=cabeca, corpo=corpo),
        unsafe_allow_html=True)


def _ordem_da_linha(a: Mapping[str, Any]):
    """Os dois lados da mesma linha ficam juntos, e em ordem de numero."""
    try:
        ponto = float(a.get("pontos"))
    except (TypeError, ValueError):
        ponto = -1e9
    lado = str(a.get("lado") or "").lower()
    return (ponto, 0 if lado in ("over", "home") else 1)


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
    if houve_bloqueio(corridas):
        st.warning(
            "A casa barrou uma das ultimas leituras. Isso e diferente de "
            "'a janela nao rodou' -- e grave, e pede espacar a coleta. "
            "Veja o detalhe abaixo."
        )
    with st.expander("Ultimas corridas do robo ({})".format(len(corridas))):
        for c in corridas:
            st.caption("· {}".format(resumo_da_corrida(c)))


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


def _chave_do_achado(achado: Mapping[str, Any], por_chave: Mapping[str, Any]) -> str:
    """Acha a linha CRUA que deu origem a este achado.

    O achado perde a chave no caminho (ele e montado pelo nucleo, que nao
    precisa dela). Em vez de remontar a chave aqui -- o que seria uma segunda
    implementacao da identidade da aposta, e elas divergem -- procura-se a
    linha crua que bate em jogo, mercado, lado e pontos.
    """
    alvo = (achado.get("jogo"), str(achado.get("mercado") or ""),
            str(achado.get("lado") or ""), achado.get("pontos"))
    for chave, crua in (por_chave or {}).items():
        if (crua.get("jogo"), str(crua.get("mercado") or ""),
                str(crua.get("lado") or ""), crua.get("pontos")) == alvo:
            return chave
    return ""


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
                                casa_e_time_a=bool(lado_certo))

    return _prob


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
