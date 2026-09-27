# -*- coding: utf-8 -*-
"""Nota de habilidade por CONFRONTO DE TODOS CONTRA TODOS (Plackett-Luce).

╔══ O QUE ISTO SUBSTITUI ════════════════════════════════════════════════════╗
║ A nota de jogador que esta em producao hoje e a MEDIA ACUMULADA das notas  ║
║ de desempenho anteriores dele:  (soma - jogo atual) / (n + BETA).          ║
║                                                                            ║
║ Media acumulada tem tres defeitos conhecidos:                              ║
║   1. nao sabe CONTRA QUEM ele jogou — 8 de nota contra os melhores do      ║
║      mundo vale o mesmo que 8 contra os piores;                            ║
║   2. nao carrega INCERTEZA — quem tem 3 jogos e quem tem 300 saem com      ║
║      notas que parecem igualmente confiaveis (o BETA so encolhe, nao       ║
║      informa);                                                             ║
║   3. converge devagar para quem acabou de chegar.                          ║
╚════════════════════════════════════════════════════════════════════════════╝

╔══ FONTES ══════════════════════════════════════════════════════════════════╗
║ O DESENHO vem do PandaSkill, estagio 2, textualmente:                      ║
║   "We base the evaluation of the skill rating of the players on the        ║
║    OpenSkill framework."                                                   ║
║   "In this setting, all players compete against each other. After a game,  ║
║    they are ranked based on their individual PScore and their skill        ║
║    ratings are updated based on this ranking."                             ║
║   Inicializacao: "mu_i = 25 and sigma_i = 25/3".                           ║
║   https://arxiv.org/html/2501.10049  (aberta em 11/09/2026)                ║
║                                                                            ║
║ A MATEMATICA e o modelo Plackett-Luce de Weng & Lin, como implementado no  ║
║ openskill.py. As equacoes abaixo foram lidas DO CODIGO-FONTE, nao de       ║
║ memoria nem do resumo da busca — a documentacao do projeto nao traz a      ║
║ matematica, so o uso:                                                      ║
║   https://raw.githubusercontent.com/vivekjoshy/openskill.py/main/openskill/models/weng_lin/plackett_luce.py
║   (aberta em 11/09/2026)                                                   ║
║                                                                            ║
║   c        = sqrt( SOMA_s (sigma_s^2 + beta^2) )                           ║
║   sum_q[q] = SOMA_{s : posicao(s) >= posicao(q)} exp(mu_s / c)             ║
║   A[q]     = quantos empatam na posicao de q                               ║
║                                                                            ║
║   para cada jogador i, percorrendo q com posicao(q) <= posicao(i):         ║
║     quo    = exp(mu_i/c) / sum_q[q]                                        ║
║     delta += quo * (1 - quo) / A[q]                                        ║
║     omega += (1 - quo)/A[q]  se POSICAO(q) == POSICAO(i),                  ║
║               -quo/A[q]    caso contrario                                  ║
║                                                                            ║
║   CUIDADO: a condicao e MESMA POSICAO, nao 'mesmo jogador'. Escrevi        ║
║   `q == i` na primeira versao e os numeros batiam — porque sem empate as   ║
║   duas coisas sao iguais. So os empates denunciavam, com ate 2,1 de erro   ║
║   em mu. Quem pegou foi a conferencia contra a biblioteca, nao a leitura.  ║
║                                                                            ║
║   omega *= sigma_i^2 / c                                                   ║
║   delta *= sigma_i^2 / c^2  *  gamma,   com gamma = sigma_i / c            ║
║                                                                            ║
║   mu_i    += omega                                                         ║
║   sigma_i *= sqrt( max(1 - delta, kappa) )                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

POR QUE ESCRITO AQUI E NAO IMPORTADO DA BIBLIOTECA
O `requirements.txt` deste projeto tem regra explicita de espelhar o ambiente
que roda o app, porque divergencia de versao ja causou diferenca de
comportamento. Acrescentar uma dependencia ao caminho SERVIDO por causa de uma
conta de TREINO seria pagar risco a toa — o app so le um numero pronto.

Para isso nao virar "reimplementei de memoria e torci", ha um teste que
confere esta conta CONTRA a biblioteca de verdade, quando ela esta disponivel
(mlcore/tests/test_players_skill_ffa.py). Ele e pulado onde a biblioteca nao
existe, mas foi rodado, e o resultado esta registrado no commit.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# Os mesmos de PandaSkill e do openskill.
MU0 = 25.0
SIGMA0 = 25.0 / 3.0
# beta = sigma0/2 e o padrao do openskill: o quanto o acaso mexe num confronto.
BETA = SIGMA0 / 2.0
# piso da contracao da incerteza, para sigma nunca virar zero (e o modelo
# nunca ficar "certo demais" e parar de aprender).
KAPPA = 0.0001


def atualizar_ffa(
    ratings: Sequence[Tuple[float, float]],
    posicoes: Sequence[int],
    *,
    beta: float = BETA,
    kappa: float = KAPPA,
) -> List[Tuple[float, float]]:
    """Uma rodada de todos-contra-todos.

    `ratings` sao pares (mu, sigma). `posicoes` e a colocacao de cada um —
    MENOR E MELHOR, e empate e permitido (mesma posicao).

    Devolve a lista nova de (mu, sigma), na mesma ordem da entrada.
    """
    n = len(ratings)
    if n != len(posicoes):
        raise ValueError("ratings e posicoes com tamanhos diferentes")
    if n < 2:
        return [(float(m), float(s)) for m, s in ratings]

    mus = [float(m) for m, _ in ratings]
    sig2 = [float(s) * float(s) for _, s in ratings]
    b2 = float(beta) * float(beta)

    c = math.sqrt(sum(s2 + b2 for s2 in sig2))
    expo = [math.exp(m / c) for m in mus]

    # sum_q[q] = soma dos exp de quem esta na posicao de q OU PIOR
    sum_q = [0.0] * n
    for q in range(n):
        for s in range(n):
            if posicoes[s] >= posicoes[q]:
                sum_q[q] += expo[s]

    # A[q] = quantos empatam com q
    a = [sum(1 for s in range(n) if posicoes[s] == posicoes[q]) for q in range(n)]

    saida: List[Tuple[float, float]] = []
    for i in range(n):
        omega = 0.0
        delta = 0.0
        for q in range(n):
            if posicoes[q] > posicoes[i]:
                continue
            quo = expo[i] / sum_q[q]
            delta += quo * (1.0 - quo) / a[q]
            # A condicao e MESMA POSICAO, nao "mesmo jogador".
            #
            # Escrevi `q == i` na primeira versao, e o resumo da busca que eu
            # tinha lido dizia isso mesmo. Estava errado, e so apareceu porque
            # a conta foi conferida contra a biblioteca de verdade: sem empate
            # as duas condicoes coincidem (posicao e unica), e os numeros
            # batiam digito a digito. COM empate, divergiam em ate 2,1 de mu.
            #
            # Li o codigo-fonte instalado para confirmar:
            #   if team_q.rank == team_i.rank:  omega += (1 - quo)/a[q]
            #   else:                           omega -= quo/a[q]
            omega += ((1.0 - quo) if posicoes[q] == posicoes[i] else (-quo)) / a[q]

        omega *= sig2[i] / c
        gamma = math.sqrt(sig2[i]) / c
        delta *= (sig2[i] / (c * c)) * gamma

        mu_novo = mus[i] + omega
        sig_novo = math.sqrt(sig2[i]) * math.sqrt(max(1.0 - delta, kappa))
        saida.append((float(mu_novo), float(sig_novo)))
    return saida


def _posicoes_por_nota(notas: Sequence[float]) -> List[int]:
    """Colocacao a partir da nota de desempenho: MAIOR nota = MELHOR posicao.

    Empate vira a MESMA posicao, que e o que o Plackett-Luce espera — dois
    desempenhos identicos nao podem render atualizacoes diferentes so porque
    um apareceu antes na lista.
    """
    ordem = sorted(range(len(notas)), key=lambda i: -float(notas[i]))
    pos = [0] * len(notas)
    lugar = 0
    for k, i in enumerate(ordem):
        if k > 0 and float(notas[i]) < float(notas[ordem[k - 1]]):
            lugar = k
        pos[i] = lugar
    return pos


def construir_notas(
    df: pd.DataFrame,
    pscore: pd.Series,
    *,
    beta: float = BETA,
) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    """Percorre os jogos em ordem e devolve (nota ANTES de cada jogo, nota final).

    PONTO NO TEMPO por construcao: a nota de um jogador num jogo e anotada
    ANTES de aquele jogo atualizar o rating dele. Nenhum jogo enxerga a si
    proprio, e nenhum enxerga o futuro.

    A nota devolvida e `mu - 3*sigma`, e nao `mu` puro: e a leitura
    conservadora usada em sistemas deste tipo, que desconta a incerteza em vez
    de tratar quem tem 3 jogos como se fosse igual a quem tem 300. E o
    equivalente honesto do encolhimento que a media acumulada fazia com BETA.
    """
    import os as _os
    # DESCONTO DA INCERTEZA, e por que ele e uma escolha e nao um detalhe.
    #
    # `mu - 3*sigma` desconta a incerteza, como manda a pratica desses
    # sistemas. Mas sigma encolhe com o NUMERO DE JOGOS, entao esse desconto
    # embute um premio por experiencia: um time de veteranos pontua mais que
    # um de novatos mesmo que joguem igual.
    #
    # Medido em 11/09/2026 com um placebo (ordem dos 10 sorteada, so o efeito
    # de experiencia sobrando): dos +0,00921 que a camada rende, +0,00663 vem
    # dai e so +0,00258 e habilidade de fato.
    #
    # Com FFA_SEM_DESCONTO=1 a nota vira `mu` puro, para medir a habilidade
    # limpa do premio por experiencia.
    _k_sigma = 0.0 if _os.environ.get("FFA_SEM_DESCONTO") == "1" else 3.0

    d = df[["gameid", "player_key", "pos_can", "date"]].copy()
    d["pscore"] = np.asarray(pscore, dtype=float)
    d = d.sort_values(["date", "gameid"], kind="stable")

    estado: Dict[str, Tuple[float, float]] = {}
    antes = np.zeros(len(d), dtype=float)
    linhas = d.to_dict("records")
    idx_orig = list(d.index)

    inicio = 0
    n = len(linhas)
    while inicio < n:
        gid = linhas[inicio]["gameid"]
        fim = inicio
        while fim < n and linhas[fim]["gameid"] == gid:
            fim += 1

        grupo = list(range(inicio, fim))
        chaves = ["{}|||{}".format(linhas[k]["player_key"], linhas[k]["pos_can"])
                  for k in grupo]
        atuais = [estado.get(ch, (MU0, SIGMA0)) for ch in chaves]

        # a nota ANTES do jogo e a que vale para prever este jogo
        for pos_local, k in enumerate(grupo):
            mu, sg = atuais[pos_local]
            antes[k] = mu - _k_sigma * sg

        if len(grupo) >= 2:
            posicoes = _posicoes_por_nota([linhas[k]["pscore"] for k in grupo])
            novos = atualizar_ffa(atuais, posicoes, beta=beta)
            for ch, novo in zip(chaves, novos):
                estado[ch] = novo

        inicio = fim

    serie = pd.Series(antes, index=idx_orig).reindex(df.index)
    finais = {ch: {"n": float(mu - _k_sigma * sg), "mu": float(mu), "sigma": float(sg)}
              for ch, (mu, sg) in estado.items()}
    return serie, finais


def deltas_por_jogo(df: pd.DataFrame, notas_antes: pd.Series) -> Dict[str, float]:
    """gameid -> soma das notas do lado azul menos a do vermelho."""
    d = df[["gameid", "lado"]].copy()
    d["nota"] = np.asarray(notas_antes, dtype=float)
    ag = d.groupby(["gameid", "lado"], sort=False)["nota"].sum().unstack("lado")
    if "blue" not in ag.columns or "red" not in ag.columns:
        return {}
    ag = ag.dropna()
    return {str(k): float(v) for k, v in (ag["blue"] - ag["red"]).items()}
