# -*- coding: utf-8 -*-
"""Camada de DADOS/config da Análise de Linha (sem UI).

Fatiamento seguro do line_analysis_tab_core (F3, 2026-07-18): funções e
dados PUROS (pandas/config, sem streamlit, sem closures) movidos VERBATIM.
line_analysis_tab_core re-exporta os nomes, então usos internos e o import
externo do radar_tab_core continuam funcionando sem alteração.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app_core.championship_stats_core import STATS_MARKET_CONFIG

# Mapas derivados do registro central (add/remover mercado = editar só o
# STATS_MARKET_CONFIG). Usados pelos renderers da Análise de Linha.
_MARKET_KEYS: dict[str, str] = {
    label: str(cfg["metric_key"]) for label, cfg in STATS_MARKET_CONFIG.items()
}
_METRIC_COL: dict[str, str] = {
    str(cfg["metric_key"]): str(cfg["combined_col"]) for cfg in STATS_MARKET_CONFIG.values()
}

_MARKETS_V2_DEFAULT = [
    {"label": "Total de kills",   "key": "kills",   "metric_col": "mv2_total_kills",   "direction": "Over", "default_line": 27.5},
    {"label": "Total de torres",  "key": "towers",  "metric_col": "mv2_total_towers",  "direction": "Over", "default_line": 11.5},
    {"label": "Total de drakes",  "key": "dragons", "metric_col": "mv2_total_dragons", "direction": "Over", "default_line": 4.5},
    {"label": "Duração (min)",    "key": "time",    "metric_col": "game_time_min",     "direction": "Over", "default_line": 32.5},
]


def _safe_pct(series: "pd.Series", line: float, direction: str) -> float:
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return float("nan")
        if direction == "Under":
            return 100.0 * float((s <= line).mean())
        return 100.0 * float((s > line).mean())
    except (TypeError, ValueError, KeyError):
        return float("nan")


def _get_metric_col(market_label: str) -> Optional[str]:
    key = _MARKET_KEYS.get(market_label)
    if not key:
        return None
    return _METRIC_COL.get(key)


# Passo 2 do registro central (2026-07-06): os 6 mercados CORE derivam de
# STATS_MARKET_CONFIG (v3_core_markets) — add/remover mercado de stats = editar
# SÓ o stats_config. Os extras abaixo são de UI da V3 (não são mercados de
# stats). Equivalência com os valores históricos travada em
# mlcore/tests/test_v3_markets_registry.py.
from app_core.stats_config import v3_core_markets as _v3_core_markets


def _augment_with_total_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed mv2_total_* columns (sum of *_for and *_against) so the
    'Total de X' markets use the combined per-game total instead of one team's
    perspective only. Returns a copy of df with the new columns."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for total_col, for_col, against_col in (
        ("mv2_total_kills",      "kills_for",      "kills_against"),
        ("mv2_total_towers",     "towers_for",     "towers_against"),
        ("mv2_total_dragons",    "dragons_for",    "dragons_against"),
        ("mv2_total_barons",     "barons_for",     "barons_against"),
        ("mv2_total_inhibitors", "inhibitors_for", "inhibitors_against"),
        # Dota: megacreeps (0/1 por jogo; linha 0.5 = "vai ter megas?").
        # LoL nao tem as colunas -> skip pelo guard abaixo.
        ("mv2_total_megacreeps", "megacreeps_for", "megacreeps_against"),
    ):
        if for_col in out.columns and against_col in out.columns:
            # SEM fillna(0): jogo sem o dado (NaN — ex.: Dota com objetivos
            # ainda não baixados) precisa continuar NaN para ficar FORA das
            # janelas — fillna criava "0" falso que entrava nos % (2026-07-11).
            # No LoL não há NaN nesses pares em jogos normais → sem efeito.
            out[total_col] = (
                pd.to_numeric(out[for_col], errors="coerce")
                + pd.to_numeric(out[against_col], errors="coerce")
            )
    # Spread columns (positive = team won that metric by N)
    for spread_col, for_col, against_col in (
        ("mv2_spread_kills",   "kills_for",   "kills_against"),
        ("mv2_spread_towers",  "towers_for",  "towers_against"),
        ("mv2_spread_dragons", "dragons_for", "dragons_against"),
    ):
        if for_col in out.columns and against_col in out.columns:
            out[spread_col] = (
                pd.to_numeric(out[for_col], errors="coerce").fillna(0.0)
                - pd.to_numeric(out[against_col], errors="coerce").fillna(0.0)
            )
    return out


_V3_MARKETS: list[dict[str, Any]] = [
    {"label": "Vencedor",     "key": "winner",       "metric_col": "",                       "default_line": 0,
     "status": "todo"},
    *_v3_core_markets(),
    {"label": "Handicaps",    "key": "handicaps",    "metric_col": "mv2_spread_kills",       "default_line": 0.0,
     "status": "ready", "metric_label": "Spread", "for_col": None, "against_col": None,
     "metric_decimals": 1, "is_time_metric": False, "handicap": True},
    {"label": "First X",      "key": "first_x",      "metric_col": "firstblood",             "default_line": 0.5,
     "status": "ready", "metric_label": "FirstX", "for_col": None, "against_col": None,
     "metric_decimals": 0, "is_time_metric": False, "first_x_selector": True},
    {"label": "Kills p/Time", "key": "kills_team",   "metric_col": "kills_for",              "default_line": 13.5,
     "status": "ready", "metric_label": "Kills (lado)", "for_col": None, "against_col": None,
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "ML + Totais",  "key": "ml_totals",    "metric_col": "mv2_total_kills",        "default_line": 27.5,
     "status": "ready", "metric_label": "Kills", "for_col": "kills_for", "against_col": "kills_against",
     "metric_decimals": 1, "is_time_metric": False},
]
