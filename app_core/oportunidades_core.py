# -*- coding: utf-8 -*-
"""Onde a Pinnacle paga mais do que o modelo acha que vale.

POR QUE ISTO EXISTE
Ele pediu, com estas palavras: "dai eu nao vou precisar mais ficar colocando
linha, tu vai pegar a linha desses sites e ja procurar a melhor linha para
mim". O robo ja traz as linhas de 3 em 3 horas. Faltava a parte que olha para
elas e diz onde vale a pena -- porque linha guardada em arquivo continua
custando o mesmo trabalho de antes.

O QUE ELE PEDIU PARA VER (15/09/2026): so onde ha vantagem, ordenado pela
vantagem. Nao o quadro inteiro.

A CONTA, e o que ela quer dizer
Para cada linha coletada o modelo diz a probabilidade `p`. A odd JUSTA e 1/p.
A vantagem e quanto sobra em cima disso:

    vantagem = p * odd_da_casa - 1

Em dinheiro: +5% quer dizer que, apostando R$ 100 nessa linha muitas e muitas
vezes, a expectativa e terminar com R$ 105. NAO quer dizer que esta aposta vai
ganhar.

O QUE ESTE MODULO NAO FAZ, DE PROPOSITO
Nao precifica o que o motor nao precifica. O motor deste projeto calcula
TOTAIS por mapa (kills, torres, dragoes, barao, inibidor, tempo). Ele nao
calcula:
  - handicap de kills (precisa da distribuicao da DIFERENCA, que nao existe);
  - total de kills POR TIME (o motor da o total do confronto);
  - quem faz mais kills no mapa (mesma diferenca que falta);
  - placar exato e vencedor (outro caminho, o do ML -- fica para depois).

Essas linhas NAO somem: entram na contagem de "nao sei precificar" e aparecem
na tela com esse nome. PRINCIPIO IV -- ausencia nao vira numero, e ausencia
escondida e pior que ausencia dita.

E SOBRE VANTAGEM GRANDE DEMAIS
A Pinnacle trabalha com ~5% de margem. Ela nao deixa 40% na mesa. Quando a
conta der isso, o mais provavel e que o modelo esteja errado naquele
confronto -- dados de menos, time novo, liga que ele mal conhece. Entao um
achado assim vem MARCADO, e nao no alto da lista fingindo ser o melhor.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

# Acima disto, desconfie do modelo antes de desconfiar da casa. A Pinnacle
# abre com margem de ~5%: uma vantagem de 20% significaria que ela errou
# quatro vezes a propria margem, o que praticamente nao acontece.
VANTAGEM_SUSPEITA_PCT = 20.0

# Abaixo disto nao compensa: a diferenca cabe dentro do erro do proprio modelo.
VANTAGEM_MINIMA_PCT = 1.0

# Menos que isto de jogos e chute com cara de conta.
JOGOS_MINIMOS = 5

# O que o motor sabe precificar hoje, por unidade da casa.
# A UNIDADE COMO A CASA ESCREVE -> a metrica do motor.
#
# ELE PERGUNTOU (22/09/2026): "onde eu consigo ver as outras linhas, kill,
# torre etc". Resposta medida: abates, dragoes e baroes ja apareciam; TORRES
# nao, e sumiam em silencio.
#
# A causa e so o nome. A Pinnacle manda a unidade assim:
#
#     Kills    -> casa com "kills"    -> 3.674 linhas precificadas
#     Dragons  -> casa com "dragons"  ->   242 linhas
#     Barons   -> casa com "barons"   ->   239 linhas
#     Turrets  -> NAO casava          ->   225 linhas PERDIDAS
#
# "towers" e o nome do motor; "turrets" e o nome da casa. Os dois precisam
# estar aqui, e por isso a chave e a grafia DELA, nao a nossa.
#
# `inhibitors` fica: a casa hoje so oferece inibidor como "1st Inhibitor"
# (moneyline), que nao e total e nao passa por aqui. Se um dia ela abrir o
# total, ja esta pronto.
UNIDADE_PARA_METRICA = {
    "kills": "kills",
    "towers": "towers",
    "turrets": "towers",      # como a Pinnacle escreve
    "dragons": "dragons",
    "barons": "barons",
    "inhibitors": "inhibitors",
}


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metrica_da_linha(linha: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """A linha vira (metrica do motor, qual mapa) -- ou None.

    None NAO e erro: e a resposta honesta para "esta eu nao sei precificar".
    Quem chama tem de contar e mostrar, nunca engolir.
    """
    if str(linha.get("tipo") or "") != "total":
        return None                      # handicap e vencedor pedem a DIFERENCA
    lado = str(linha.get("lado") or "").strip().lower()
    if lado not in ("over", "under"):
        return None
    if _num(linha.get("pontos")) is None:
        return None
    metrica = UNIDADE_PARA_METRICA.get(str(linha.get("unidade") or "").strip().lower())
    if metrica is None:
        return None                      # "Maps" e a serie; vazio e o principal
    try:
        periodo = int(linha.get("periodo", 0) or 0)
    except (TypeError, ValueError):
        return None
    if periodo < 1 or periodo > 5:
        return None                      # o motor fala de MAPA, nao de serie
    return metrica, "m{}".format(periodo)


def vantagem_pct(p: Any, odd_casa: Any) -> Optional[float]:
    """Quanto sobra, em porcento, apostando nesta linha.

    Sem probabilidade nao ha vantagem -- nem vantagem zero. Devolve None, e
    None tem de chegar ate a tela como "nao sei".
    """
    pp = _num(p)
    oo = _num(odd_casa)
    if pp is None or oo is None:
        return None
    if not (0.0 < pp < 1.0) or oo <= 1.0:
        return None
    return (pp * oo - 1.0) * 100.0


def odd_justa(p: Any) -> Optional[float]:
    pp = _num(p)
    if pp is None or not (0.0 < pp < 1.0):
        return None
    return 1.0 / pp


def avaliar_linha(linha: Dict[str, Any], totais: Dict[str, Any],
                  *, prob_over: Callable[[Any, float], float]) -> Optional[Dict[str, Any]]:
    """Uma linha coletada + os totais do confronto -> o veredito dela.

    `prob_over` e a funcao de PRODUCAO (`mlcore.lines_engine.
    prob_over_from_total`), recebida de fora. PRINCIPIO V: nao se reimplementa
    o motor para medi-lo -- se a conta daqui divergisse da conta da tela de
    analise, o mesmo confronto daria dois numeros e nenhum dos dois seria
    confiavel.
    """
    alvo = metrica_da_linha(linha)
    if alvo is None:
        return None
    metrica, _mapa = alvo
    total = (totais or {}).get(metrica)
    if total is None:
        return None
    ponto = _num(linha.get("pontos"))
    if ponto is None:
        return None
    try:
        p_over = _num(prob_over(total, ponto))
    except Exception:
        return None
    if p_over is None:
        return None
    lado = str(linha.get("lado") or "").strip().lower()
    p = p_over if lado == "over" else (1.0 - p_over)
    v = vantagem_pct(p, linha.get("odd"))
    if v is None:
        return None
    return {
        "jogo": linha.get("jogo"),
        "mercado": linha.get("mercado"),
        "lado": linha.get("lado"),
        "pontos": ponto,
        "odd_casa": _num(linha.get("odd")),
        "odd_justa": odd_justa(p),
        "probabilidade": p,
        "vantagem_pct": v,
        "limite": linha.get("limite"),
        "margem_pct": linha.get("margem_pct"),
        "fecha_em": linha.get("fecha_em"),
        "metrica": metrica,
        "mapa": _mapa,
    }


# O motor deste projeto calcula TOTAIS por mapa. Tudo que pede a DIFERENCA
# entre os times (handicap, quem faz mais) ou o resultado da serie (vencedor,
# placar exato, total de mapas) vem de outro caminho -- o do ML --, que ainda
# nao foi ligado aqui. Cada motivo abaixo e escrito para ele entender por que
# aquela linha nao apareceu, sem precisar saber o que e "motor".
_FALTA_A_DIFERENCA = ("o motor calcula o TOTAL dos dois times somados, "
                      "nao a diferenca entre eles")
_FALTA_O_ML = "vem do modelo de vencedor, que ainda nao esta ligado nesta tela"
_FALTA_POR_TIME = "o motor calcula o total do confronto, nao o de cada time"


def motivo_de_nao_precificar(linha: Dict[str, Any]) -> str:
    """Por que esta linha ficou de fora, em palavras dele.

    Motivo em branco seria o mesmo silencio que segurou kills por horas: a
    tela pareceria completa mostrando cinco apostas e escondendo duzentas.
    """
    tipo = str(linha.get("tipo") or "")
    unidade = str(linha.get("unidade") or "").strip().lower()
    conhecida = unidade in UNIDADE_PARA_METRICA
    try:
        periodo = int(linha.get("periodo", 0) or 0)
    except (TypeError, ValueError):
        periodo = 0

    if tipo == "team_total":
        return _FALTA_POR_TIME
    if tipo in ("spread", "moneyline") and conhecida:
        return _FALTA_A_DIFERENCA
    if tipo in ("moneyline", "spread", "total") and (unidade == "maps" or periodo == 0
                                                     or not unidade):
        return _FALTA_O_ML
    return "o motor nao precifica este mercado"


def _guardar(achados: List[Dict[str, Any]], achado: Dict[str, Any],
             jogo: Dict[str, Any],
             jogos_por_time: Optional[Callable[[str], int]]) -> None:
    """Junta ao achado o que a tela precisa mostrar junto do numero."""
    achado["liga"] = jogo.get("liga")
    achado["casa"] = jogo.get("casa")
    achado["fora"] = jogo.get("fora")
    achado["inicio_utc"] = jogo.get("inicio_utc")
    if jogos_por_time is not None:
        try:
            n = min(int(jogos_por_time(str(jogo.get("casa") or ""))),
                    int(jogos_por_time(str(jogo.get("fora") or ""))))
        except Exception:
            n = -1
        achado["jogos_no_historico"] = n
        achado["dados_finos"] = bool(0 <= n < JOGOS_MINIMOS)
    achado["suspeito"] = bool(achado["vantagem_pct"] > VANTAGEM_SUSPEITA_PCT)
    achados.append(achado)


def _somar_de_fora(acc: Dict[str, Dict[str, Any]], linha: Dict[str, Any],
                   motivo: str) -> None:
    """Agrupa pelo nome que a CASA da ao mercado.

    De proposito: e esse o nome que ele ve na tela da Pinnacle. Agrupar por um
    nome nosso obrigaria a traduzir de cabeca para conferir.
    """
    nome = str(linha.get("mercado") or "(sem nome)")
    reg = acc.setdefault(nome, {"mercado": nome, "linhas": 0, "motivo": motivo})
    reg["linhas"] += 1


def montar_oportunidades(
    odds: Dict[str, Any],
    calcular_totais: Callable[[str, str, str], Optional[Dict[str, Any]]],
    *,
    prob_over: Callable[[Any, float], float],
    minimo_pct: float = VANTAGEM_MINIMA_PCT,
    jogos_por_time: Optional[Callable[[str], int]] = None,
    prob_de_vencedor: Optional[Callable[[Dict[str, Any], Dict[str, Any]],
                                        Optional[float]]] = None,
) -> Dict[str, Any]:
    """A lista do que vale a pena, e a lista honesta do que ficou de fora.

    `calcular_totais(casa, fora, mapa)` e o caminho de PRODUCAO do app
    (build_profile + _expected_totals_dc, com as MESMAS regulagens que ele
    tem na tela). Devolve None quando nao consegue -- time que nao existe no
    CSV, dados de menos --, e nesse caso o confronto inteiro entra em
    "sem_modelo" com o nome dos times. Nada de silencio.
    """
    jogos = {int(j.get("id", -1)): j for j in (odds.get("jogos") or [])}
    # O PRECO JUSTO DA CASA em toda linha (23/09/2026): a Pinnacle e a
    # referencia, e a comparacao que ele quer e modelo x justo dela, nao
    # modelo x odd com margem. Ver app_core/preco_justo_casa_core.py.
    from app_core.preco_justo_casa_core import anexar_preco_justo
    linhas = anexar_preco_justo(odds.get("linhas") or [])

    achados: List[Dict[str, Any]] = []
    nao_precificados: Dict[str, Dict[str, Any]] = {}
    sem_modelo: List[Dict[str, Any]] = []
    cache: Dict[Tuple[Any, str], Any] = {}
    falhou: set = set()
    # Achado de cada linha, pela posicao: o "mercado completo" junta o que o
    # modelo disse a TODAS as linhas da casa, inclusive as que ele nao sabe.
    achado_da_linha: Dict[int, Dict[str, Any]] = {}

    for _i, l in enumerate(linhas):
        if not isinstance(l, dict):
            continue
        jogo = jogos.get(int(l.get("jogo", -1) or -1))
        if jogo is None:
            continue
        alvo = metrica_da_linha(l)
        if alvo is None:
            # O MERCADO DE VENCEDOR vem de fora, do fluxo do confronto aberto
            # -- e o mesmo numero do card, nao uma segunda conta.
            _p = None
            if prob_de_vencedor is not None:
                try:
                    _p = prob_de_vencedor(jogo, l)
                except Exception:
                    _p = None
            if _p is None:
                _somar_de_fora(nao_precificados, l, motivo_de_nao_precificar(l))
                continue
            _v = vantagem_pct(_p, l.get("odd"))
            if _v is None:
                _somar_de_fora(nao_precificados, l,
                               "preco ilegivel para esta linha")
                continue
            achado = {
                "jogo": l.get("jogo"),
                "mercado": l.get("mercado"),
                "lado": l.get("lado"),
                "pontos": None,
                "odd_casa": _num(l.get("odd")),
                "odd_justa": odd_justa(_p),
                "probabilidade": _p,
                "vantagem_pct": _v,
                "limite": l.get("limite"),
                "margem_pct": l.get("margem_pct"),
                "fecha_em": l.get("fecha_em"),
                "metrica": "vencedor",
                "mapa": "",
            }
            _com_a_casa(achado, l)
            _guardar(achados, achado, jogo, jogos_por_time)
            achado_da_linha[_i] = achado
            continue
        _metrica, mapa = alvo
        chave = (jogo.get("id"), mapa)
        if chave in falhou:
            continue
        if chave not in cache:
            try:
                cache[chave] = calcular_totais(str(jogo.get("casa") or ""),
                                               str(jogo.get("fora") or ""), mapa)
            except Exception:
                cache[chave] = None
            if cache[chave] is None:
                falhou.add(chave)
                sem_modelo.append({"jogo": jogo.get("id"), "casa": jogo.get("casa"),
                                   "fora": jogo.get("fora"), "mapa": mapa})
                continue
        achado = avaliar_linha(l, cache[chave], prob_over=prob_over)
        if achado is None:
            _somar_de_fora(nao_precificados, l,
                           "o motor nao deu numero para este confronto")
            continue
        _com_a_casa(achado, l)
        _guardar(achados, achado, jogo, jogos_por_time)
        achado_da_linha[_i] = achado

    vale = [a for a in achados if a["vantagem_pct"] >= float(minimo_pct)]
    # Os suspeitos vao para o fim: a lista e para decidir, e um numero grande
    # demais no topo empurraria justamente para a aposta mais provavel de ser
    # erro do modelo.
    vale.sort(key=lambda a: (a["suspeito"], -a["vantagem_pct"]))
    return {
        "achados": vale,
        # TODAS as avaliadas, e nao so as que passaram do corte. A tela de
        # comparacao (pedido dele em 16/09/2026) mostra os DOIS lados de cada
        # linha -- e o lado sem vantagem e metade da informacao: e olhando o
        # par que se ve se a discordancia e do mercado ou do modelo.
        "todas": sorted(achados, key=lambda a: -a["vantagem_pct"]),
        "avaliadas": len(achados),
        "nao_precificados": sorted(nao_precificados.values(),
                                   key=lambda d: (-d["linhas"], d["mercado"])),
        "sem_modelo": sem_modelo,
        "mercado_completo": _mercado_completo(linhas, jogos, achado_da_linha),
        "lido_em": odds.get("lido_em"),
    }


def _com_a_casa(achado: Dict[str, Any], linha: Dict[str, Any]) -> None:
    """O justo da casa ao lado do justo do modelo, e a distancia entre eles.

    `diferenca_pp` em pontos percentuais: +8 quer dizer que o modelo da 8
    pontos a MAIS de chance que a Pinnacle sem margem. E a medida de
    discordancia; a vantagem (`vantagem_pct`) continua sendo contra a odd
    que ele de fato consegue apostar.
    """
    pc = _num(linha.get("p_casa_justa"))
    achado["p_casa_justa"] = pc
    achado["odd_casa_justa"] = _num(linha.get("odd_casa_justa"))
    pm = _num(achado.get("probabilidade"))
    achado["diferenca_pp"] = ((pm - pc) * 100.0) if (pm is not None and pc is not None) else None
    achado["time"] = linha.get("time")


_MOTIVO_POR_FORMA = {
    "par_impar": "par ou impar: o modelo nao calcula",
    "primeiro": "quem faz primeiro: o modelo nao calcula",
    "corrida": "quem chega primeiro a N kills: o modelo nao calcula",
    "ambos": "os dois times fazem?: o modelo nao calcula",
    "algum": "algum time faz?: o modelo nao calcula",
}


def _motivo_no_quadro(l: Dict[str, Any]) -> str:
    """Um total que o motor SABE fazer, mas nao fez, nao e "nao precifica":
    e o confronto que nao deu (time sem casar, jogos de menos). Dizer o
    motivo errado mandaria ele esperar uma coisa que nunca vem."""
    if metrica_da_linha(l) is not None:
        return "o modelo nao conseguiu este confronto (time sem casar ou jogos de menos)"
    # Pela FORMA da aposta, que a tabela de nomes da casa entende. Sem isto,
    # "par ou impar de kills" caia em "o motor calcula o total, nao a
    # diferenca" -- motivo de outro mercado (visto na tela do Dota, 23/09).
    forma = l.get("forma")
    if not forma:
        try:
            from app_core.mercado_pinnacle_core import entender_mercado
            forma = entender_mercado(l.get("mercado")).get("forma")
        except Exception:
            forma = None
    if forma in _MOTIVO_POR_FORMA:
        return _MOTIVO_POR_FORMA[forma]
    if str(l.get("tipo") or "") == "moneyline" and not str(l.get("unidade") or "").strip()             and str(l.get("lado") or "").strip().lower() in ("home", "away"):
        return "vencedor: so para o confronto aberto na tela"
    return motivo_de_nao_precificar(l)


def _mercado_completo(linhas: List[Dict[str, Any]], jogos: Dict[int, Dict[str, Any]],
                      achado_da_linha: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """TODA linha da casa, com o justo dela e -- quando existe -- o do modelo.

    Ele pediu a Pinnacle "de forma total, 100%". Este e o quadro inteiro:
    linha que o modelo nao sabe precificar aparece com o preco justo da casa
    e o motivo, nunca some (PRINCIPIO IV).
    """
    fora: List[Dict[str, Any]] = []
    for i, l in enumerate(linhas):
        if not isinstance(l, dict):
            continue
        jogo = jogos.get(int(l.get("jogo", -1) or -1))
        if jogo is None:
            continue
        a = achado_da_linha.get(i)
        fora.append({
            "jogo": l.get("jogo"), "liga": jogo.get("liga"),
            "casa": jogo.get("casa"), "fora": jogo.get("fora"),
            "inicio_utc": jogo.get("inicio_utc"),
            "mercado": l.get("mercado"), "nome_pt": l.get("nome_pt"),
            "tipo": l.get("tipo"), "periodo": l.get("periodo"),
            "lado": l.get("lado"), "time": l.get("time"),
            "pontos": l.get("pontos"), "alternativa": l.get("alternativa"),
            "odd_casa": _num(l.get("odd")), "margem_pct": l.get("margem_pct"),
            "limite": l.get("limite"),
            "p_casa_justa": _num(l.get("p_casa_justa")),
            "odd_casa_justa": _num(l.get("odd_casa_justa")),
            "p_modelo": (a or {}).get("probabilidade"),
            "odd_modelo": (a or {}).get("odd_justa"),
            "vantagem_pct": (a or {}).get("vantagem_pct"),
            "diferenca_pp": (a or {}).get("diferenca_pp"),
            "suspeito": (a or {}).get("suspeito"),
            "motivo": None if a else _motivo_no_quadro(l),
        })
    return fora


# ---------------------------------------------------------------------------
# MERCADO DE VENCEDOR (16/09/2026)
#
# Ele olhou a aba mostrando ZERO e disse: "nao faz mt sentido". Tinha razao --
# o numero JA EXISTE. O card do confronto mostra, por exemplo, MD5 1,66 x 2,51
# (60,1% x 39,9%). E exatamente a odd justa que faltava aqui.
#
# O que NAO se faz: recalcular essa probabilidade dentro desta aba. O projeto
# ja teve TRES implementacoes do pre-jogo com contagens diferentes, e existe um
# modulo inteiro (mlcore/replay_pre_jogo.py) escrito por causa disso. Aqui o
# numero e RECEBIDO do fluxo do confronto, o mesmo que ele ve na tela.
#
# Por isso este mercado so vale para o CONFRONTO ABERTO: e o unico para o qual
# o app calculou o numero. Os outros jogos da lista aparecem como nao olhados,
# com esse motivo dito.
# ---------------------------------------------------------------------------


def prob_do_vencedor(linha: Dict[str, Any], *, p_map: Any, p_series: Any,
                     casa_e_time_a: bool) -> Optional[float]:
    """A probabilidade do LADO desta linha de vencedor.

    `p_map` e `p_series` sao do TIME 1 da tela (mesma orientacao do card).
    `casa_e_time_a` diz se o mandante da casa de apostas e esse Time 1 --
    sem isso, um confronto listado ao contrario faria a tela recomendar o
    lado oposto do que o modelo quis dizer.
    """
    if str(linha.get("tipo") or "") != "moneyline":
        return None
    # "Vencedor - Kills" tambem e moneyline com lados casa/fora, mas pergunta
    # quem faz mais KILLS -- outra coisa. A unidade separa os dois.
    if str(linha.get("unidade") or "").strip():
        return None
    lado = str(linha.get("lado") or "").strip().lower()
    if lado not in ("home", "away"):
        return None          # placar exato tem lados com nome, nao casa/fora
    try:
        periodo = int(linha.get("periodo", 0) or 0)
    except (TypeError, ValueError):
        return None
    base = _num(p_series) if periodo == 0 else _num(p_map)
    if base is None or not (0.0 < base < 1.0):
        return None
    p_casa = base if casa_e_time_a else (1.0 - base)
    return p_casa if lado == "home" else (1.0 - p_casa)


def mesmo_confronto(jogo: Dict[str, Any], time_a: str, time_b: str,
                    resolver: Callable[[str], str]) -> Optional[bool]:
    """Este jogo da casa e o confronto aberto na tela?

    Devolve True quando o mandante e o Time 1, False quando esta invertido, e
    None quando nao e o mesmo confronto. Inverter silenciosamente seria o pior
    desfecho: a tela recomendaria apostar no outro time.
    """
    try:
        casa = resolver(str(jogo.get("casa") or ""))
        fora = resolver(str(jogo.get("fora") or ""))
    except Exception:
        return None
    if not casa or not fora:
        return None
    a, b = str(time_a or ""), str(time_b or "")
    if casa == a and fora == b:
        return True
    if casa == b and fora == a:
        return False
    return None


def bo_bate(jogo: Dict[str, Any], bo_da_tela: Any) -> bool:
    """O BO da casa e o mesmo que o modelo usou?

    Um BO5 precificado como BO3 muda a probabilidade da serie inteira. Quando
    nao bater, a linha fica de fora COM o motivo, em vez de sair um numero
    que parece certo.
    """
    try:
        return int(jogo.get("bo") or 0) == int(bo_da_tela or 0)
    except (TypeError, ValueError):
        return False


def nome_que_falta_casar(jogo: Dict[str, Any], time_a: str, time_b: str,
                         resolver: Callable[[str], str]) -> Optional[str]:
    """O jogo da casa E o confronto aberto, mas um nome nao casou. Qual?

    DEFEITO RELATADO POR ELE (16/09/2026, com print). Ele abriu FlyQuest x
    Team Liquid e a tela disse "o confronto aberto nao esta na Pinnacle" --
    e listou, tres linhas abaixo, "LCS — Liquid x FlyQuest (BO5)". O MESMO
    jogo. A mensagem contradizia a lista logo embaixo dela.

    A casa escreve "Liquid"; o CSV tem "Team Liquid". Um lado casou, o outro
    nao, e eu tratei isso como "jogo inexistente".

    "A casa nao tem esse jogo" e "a casa tem, mas o nome nao casou" pedem
    ACOES DIFERENTES: a primeira e esperar; a segunda e ensinar um apelido,
    uma vez, e nao volta mais. Devolver o nome cru e o que permite oferecer
    o conserto em vez de so dar a noticia ruim.

    None quer dizer "este jogo nao tem nada a ver com o confronto aberto".
    """
    alvos = {str(time_a or "").strip(), str(time_b or "").strip()}
    alvos.discard("")
    if len(alvos) < 2:
        return None
    try:
        lados = [(str(jogo.get("casa") or ""), resolver(str(jogo.get("casa") or ""))),
                 (str(jogo.get("fora") or ""), resolver(str(jogo.get("fora") or "")))]
    except Exception:
        return None

    casaram = [cru for cru, res in lados if res and res in alvos]
    faltaram = [cru for cru, res in lados if not res]
    # Exatamente um lado reconhecido e exatamente um sem nome casado: e o
    # padrao de "mesmo jogo, apelido faltando". Dois sem casar seria chute --
    # qualquer jogo da lista "poderia" ser, e adivinhar aqui analisaria o
    # confronto errado com toda a confianca (PRINCIPIO IV).
    if len(casaram) == 1 and len(faltaram) == 1:
        return faltaram[0]
    return None
