# -*- coding: utf-8 -*-
"""Torneios de EXIBIÇÃO — onde a previsão vale menos (achado 24/07, com ele).

Medido no artifact atual (2026): o modelo acerta **38,9%** no KeSPA Cup
(logloss 1,009) contra 66,7% na LCK e 62,4% na LPL. É torneio de pré-temporada:
times testam jogadores, misturam academy, não levam a sério — o modelo aprende
com jogos sérios e não tem como prever isso.

A tela avisa em vez de fingir confiança. Puro e testável.
"""
from __future__ import annotations

# substrings (case-insensitive) que marcam torneio de exibição/pré-temporada
_EXHIBITION_KEYS: tuple[str, ...] = (
    "kespa",        # KeSPA Cup (Coreia)
    "demacia",      # Demacia Cup (China)
    "all-star",
    "all star",
    "showmatch",
)


def is_exhibition_league(league: str | None) -> bool:
    """True para torneio de exibição/pré-temporada (previsão pouco confiável)."""
    if not league:
        return False
    s = str(league).strip().casefold()
    if not s:
        return False
    return any(k in s for k in _EXHIBITION_KEYS)


def exhibition_warning(league: str | None) -> str:
    """Aviso curto pra UI (string vazia quando não é exibição)."""
    if not is_exhibition_league(league):
        return ""
    return (f"⚠ **{str(league).strip()}** é torneio de exibição/pré-temporada — "
            "times testam roster e não jogam sério. Medido: o modelo acerta ~39% "
            "aqui (vs ~67% na LCK). Trate os números com desconfiança.")


__all__ = ["is_exhibition_league", "exhibition_warning"]
