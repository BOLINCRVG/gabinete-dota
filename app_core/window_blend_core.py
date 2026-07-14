"""Blend recency-ponderado das janelas de histórico (modo "Geral").

Hoje a Janela (Histórico) é um corte único e excludente: Ano todo / Últimos 10 /
Últimos 5. Cada um sozinho tem um defeito — Ano todo é estável mas defasado, L5
é atual mas barulhento (5 jogos, DP enorme). O modo "Geral" combina os três num
único consolidado, dando mais peso à forma recente sem jogar fora a amostra da
temporada (um estimador de shrinkage: recente puxado em direção à base do ano).

IMPORTANTE: a janela só afeta os TOTAIS/médias (build_profile). O ML do vencedor
(chip Mapa/MD3/MD5) vem do artefato Elo (histórico completo) e independe da
janela — por isso o "Geral" atua apenas no painel de Médias esperadas / Odds
justas (totais).
"""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

# Janelas que compõem o "Geral" (label, max_games). None = ano todo (sem corte).
GERAL_WINDOWS: tuple[tuple[str, Optional[int]], ...] = (
    ("Ano todo", None),
    ("L10", 10),
    ("L5", 5),
)

# Pesos recency-tilted: forma recente pesa mais, ano todo ancora. Soma 1.0.
# Chave = max_games (None p/ ano todo). Tunáveis (futuro: fitar nos dados).
GERAL_WEIGHTS: dict[Optional[int], float] = {None: 0.2, 10: 0.3, 5: 0.5}


def geral_window_weight(max_games: Optional[int]) -> float:
    """Peso da janela no blend Geral (0.0 se não faz parte do Geral)."""
    return float(GERAL_WEIGHTS.get(max_games, 0.0))


def _finite_mean(t: Any) -> bool:
    try:
        return math.isfinite(float(t.mean))
    except (TypeError, ValueError, AttributeError):
        return False


def blend_matchup_totals(
    per_window: Sequence[tuple[float, dict]],
    *,
    totals_cls: Any = None,
) -> dict:
    """Blendeia dicts de totais (Metric -> MatchupTotals) com pesos.

    per_window: lista de (peso, totals_dict). Pesos são renormalizados por
    métrica (uma métrica ausente numa janela só pesa nas janelas onde existe).

    Médias: média ponderada. DP: para poisson recalcula sqrt(média) (mantém a
    relação da distribuição); para as demais, sqrt da variância ponderada. A
    distribuição (dist) é preservada da primeira janela que tem a métrica.

    totals_cls: classe p/ construir o resultado (MatchupTotals). Se None,
    importa de mlcore.lines_engine (mantém o módulo desacoplado em testes).
    """
    if totals_cls is None:
        from mlcore.lines_engine import MatchupTotals as totals_cls

    items = [(float(w), d) for (w, d) in (per_window or []) if d and float(w) > 0.0]
    if not items:
        return {}

    # Ordem estável das métricas (primeira aparição).
    metrics: list[str] = []
    for _w, d in items:
        for k in d.keys():
            if k not in metrics:
                metrics.append(k)

    out: dict = {}
    for k in metrics:
        present = [(w, d[k]) for (w, d) in items if k in d and _finite_mean(d[k])]
        wsum = sum(w for w, _ in present)
        if not present or wsum <= 0.0:
            continue
        mean = sum(w * float(t.mean) for w, t in present) / wsum
        dist = getattr(present[0][1], "dist", "normal")
        if str(dist) == "poisson":
            sd = math.sqrt(max(mean, 1e-9))
        else:
            var = sum(w * (float(getattr(t, "sd", 0.0)) ** 2) for w, t in present) / wsum
            sd = math.sqrt(max(var, 1e-9))
        out[k] = totals_cls(mean=float(mean), sd=float(sd), dist=dist)
    return out


__all__ = [
    "GERAL_WEIGHTS",
    "GERAL_WINDOWS",
    "blend_matchup_totals",
    "geral_window_weight",
]
