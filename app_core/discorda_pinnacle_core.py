# -*- coding: utf-8 -*-
"""AVISO DE DISCORDANCIA COM A PINNACLE (25/11/2026). Aprovado por ele ("podemos sim, fazer dessa forma").

O diagnostico por recorte (_medicoes/diagnostico_pinnacle.py, LoL 226 / Dota 492 / CS 297 mapas, preco honesto do
modelo x ultima leitura da Pinnacle sem margem) mostrou que a distancia do modelo para a Pinnacle mora TODA nas
discordancias grandes:
  - concorda (< 5 pontos no mapa): LoL -0,006, Dota -0,006, CS -0,001 de logloss -> EMPATA com a Pinnacle;
  - discorda 10+ pontos: LoL +0,086, Dota +0,077 / +0,097, CS +0,170 -> a Pinnacle acerta bem mais.
O aviso NAO mexe no preco: so diz quando desconfiar do lado do modelo.

A medicao foi POR MAPA. Numa serie a mesma diferenca de forca vira uma diferenca maior de chance, entao a serie
e convertida de volta para a chance de mapa equivalente (mapas independentes) antes de comparar.
Arquivo IDENTICO no LoL, no Dota e no CS.
"""
from __future__ import annotations

import math
from typing import Optional

LIMITE = 0.10        # 10 pontos de chance no MAPA


def p_serie(p: float, bo: int) -> float:
    """Pura. Chance de vencer a serie MD`bo` com chance `p` em cada mapa (mapas independentes)."""
    if bo <= 1:
        return p
    alvo = bo // 2 + 1
    q = 1 - p
    return sum(math.comb(alvo - 1 + k, k) * p ** alvo * q ** k for k in range(alvo))


def mapa_equivalente(ps: float, bo: int) -> float:
    """Pura. A chance de mapa que da a chance de serie `ps` (inversa de p_serie, por bissecao)."""
    if bo <= 1:
        return ps
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        if p_serie(m, bo) < ps:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def sem_margem(odd_a: Optional[float], odd_b: Optional[float]) -> Optional[float]:
    """Pura. Chance do lado A na Pinnacle, tirada a margem (proporcional)."""
    try:
        a, b = 1 / float(odd_a), 1 / float(odd_b)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return a / (a + b) if a > 0 and b > 0 else None


def aviso(p_modelo_a: Optional[float], p_pinnacle_a: Optional[float], *, time_a: str, time_b: str,
          bo: int = 1) -> str:
    """Pura. '' quando concordam; senao o texto do aviso. As chances sao do time A, no formato do mercado
    (serie MD`bo` ou mapa com bo=1)."""
    if p_modelo_a is None or p_pinnacle_a is None:
        return ""
    ma, pa = mapa_equivalente(float(p_modelo_a), bo), mapa_equivalente(float(p_pinnacle_a), bo)
    dif = ma - pa
    if abs(dif) < LIMITE:
        return ""
    quem = time_a if dif > 0 else time_b
    pin_fav_a = pa >= 0.5
    lado = "o azarão" if (dif > 0) != pin_fav_a else "o favorito"
    return ("Discorda muito da Pinnacle: o modelo dá {:.0f} pontos a mais para {} ({}) por mapa. Medido em 25/11/2026: "
            "quando a discordância passa de 10 pontos, a Pinnacle acerta mais que o modelo (LoL, Dota e CS) — "
            "ela costuma saber algo que o modelo não sabe. Desconfie do lado do modelo aqui."
            ).format(100 * abs(dif), quem, lado)
