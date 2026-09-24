# -*- coding: utf-8 -*-
"""O preco JUSTO da Pinnacle: a odd dela sem a margem.

POR QUE (23/09/2026). Ele disse como usa a casa: a Pinnacle e a referencia do
preco justo, e a pergunta e onde o MODELO discorda dela. Para isso a odd da
casa nao serve crua -- ela tem a margem dentro. Uma odd de 1,87 dos dois lados
nao quer dizer 53,5% para cada um; quer dizer 50% e 50%, com 7% de margem.

QUANTO DE MARGEM, MEDIDO no historico dele: LPL 4,8%, LCS 5,3%, WSCI 9,4%,
vencedor de kills 14%, placar exato 16%. Nao e o 1-3% das ligas grandes de
futebol -- e ai o METODO de tirar a margem faz diferenca.

O METODO: POTENCIA. Acha o expoente k tal que a soma de (1/odd)^k da 1, e a
probabilidade justa de cada lado e (1/odd)^k.

FONTES (pesquisado em 23/09/2026)
  - Clarke, Kovalchik & Ingram (2017), "Adjusting Bookmaker's Odds to Allow
    for Overround", American Journal of Sports Science 5(6):45-49. Tres bases
    grandes, tres esportes: a potencia "universally outperforms the
    multiplicative method and outperforms or is comparable to the Shin
    method". PDF: outlier.bet/wp-content/uploads/2023/08/2017-clarke-adjusting_bookmakers_odds.pdf
  - Strumbelj (2014), "On determining probability forecasts from betting
    odds", International Journal of Forecasting 30(4):934-943: a divisao
    simples ("basic normalization") da probabilidade TORTA; Shin foi melhor
    em todos os pares casa/esporte. sciencedirect.com/science/article/abs/pii/S0169207014000533
  - Por que nao Shin, entao: a potencia empata ou ganha dele (Clarke 2017),
    nunca sai de 0-1, e nao precisa de iteracao com caso especial.
  - Contra: uma analise de 2025 (algorithmicsportsbetting.substack.com/p/
    its-time-to-retire-shins-method) achou a divisao simples melhor em
    CORRIDA DE CAVALO com margem de ~1%. Nao e o nosso caso: margem baixa e
    exatamente onde os metodos empatam.

O QUE AINDA NAO DA PARA MEDIR NOS DADOS DELE: qual metodo acerta mais nos
esports. Sao 55 series com preco guardado -- pouco demais para separar
metodos que diferem em 1 ou 2 pontos. Quando houver algumas centenas, a
conta e: Brier de cada metodo contra o resultado.

O QUE ESTE MODULO NAO FAZ: inventar o outro lado. Mercado com um lado so (o
outro sumiu da leitura) fica SEM preco justo -- nunca 1/odd, que traria a
margem inteira para dentro do "justo" (PRINCIPIO IV).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sem_margem(odds: Sequence[Any]) -> Optional[List[float]]:
    """Probabilidades justas pelo metodo da potencia, na ordem das odds.

    None quando nao da: menos de dois lados, odd ilegivel ou <= 1, ou a soma
    ja abaixo de 1 (a casa pagando mais que o justo -- ai nao ha margem para
    tirar, e dividir de novo inventaria uma).
    """
    ps: List[float] = []
    for o in odds:
        v = _num(o)
        if v is None or v <= 1.0:
            return None
        ps.append(1.0 / v)
    if len(ps) < 2:
        return None
    soma = sum(ps)
    if soma < 1.0 - 1e-9:
        return None
    if abs(soma - 1.0) <= 1e-12:
        return list(ps)
    # f(k) = soma(p^k) - 1 cai com k (p < 1). f(1) > 0; acha um k alto com
    # f < 0 e corta ao meio. 100 passos dao precisao de maquina.
    lo, hi = 1.0, 2.0
    while sum(p ** hi for p in ps) > 1.0:
        hi *= 2.0
        if hi > 1e6:
            return None
    for _ in range(100):
        k = 0.5 * (lo + hi)
        if sum(p ** k for p in ps) > 1.0:
            lo = k
        else:
            hi = k
    k = 0.5 * (lo + hi)
    justas = [p ** k for p in ps]
    total = sum(justas)
    return [p / total for p in justas]


def chave_do_mercado(l: Dict[str, Any]) -> Tuple[Any, ...]:
    """Quais linhas sao os lados de UMA mesma aposta.

    Total: mesma linha de pontos ("over 27,5" com "under 27,5"). Handicap: o
    mandante -1,5 e o visitante +1,5 sao o mesmo mercado -- entao a linha e
    contada do lado do MANDANTE. Valor absoluto NAO serve: a casa tambem
    oferece mandante +1,5 / visitante -1,5, e as quatro linhas cairiam num
    grupo so. Total por time: tambem o time (mandante e visitante podem ter a
    mesma linha). Vencedor, placar e "primeiro X": todos os lados juntos.
    """
    tipo = str(l.get("tipo") or "")
    pts = _num(l.get("pontos"))
    if tipo == "spread":
        lado = str(l.get("lado") or "").strip().lower()
        if pts is None or lado not in ("home", "away"):
            linha = None
        else:
            linha = pts if lado == "home" else -pts
    elif tipo in ("total", "team_total"):
        linha = pts
    else:
        linha = None
    return (l.get("jogo"), str(l.get("mercado") or ""), tipo, l.get("periodo"),
            str(l.get("time") or ""), linha)


def anexar_preco_justo(linhas: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copia das linhas, cada uma com `p_casa_justa` e `odd_casa_justa`.

    Linha sem par (ou com par ilegivel) sai com os dois campos None, e
    `sem_par=True` -- para a tela dizer por que nao tem, em vez de sumir.
    """
    grupos: Dict[Tuple[Any, ...], List[int]] = {}
    for i, l in enumerate(linhas):
        if isinstance(l, dict):
            grupos.setdefault(chave_do_mercado(l), []).append(i)
    fora: List[Dict[str, Any]] = [dict(l) if isinstance(l, dict) else l for l in linhas]
    for idx in grupos.values():
        lados = [str(fora[i].get("lado") or "").strip().lower() for i in idx]
        # Dois registros do MESMO lado nao sao um mercado: e leitura repetida
        # ou chave curta demais. Melhor sem preco justo que com um errado.
        ps = None if len(set(lados)) != len(lados) else sem_margem(
            [fora[i].get("odd") for i in idx])
        for pos, i in enumerate(idx):
            p = ps[pos] if ps else None
            fora[i]["p_casa_justa"] = p
            fora[i]["odd_casa_justa"] = (1.0 / p) if p else None
            fora[i]["sem_par"] = ps is None
    return fora
