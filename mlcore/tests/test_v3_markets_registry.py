# -*- coding: utf-8 -*-
"""Equivalência do passo 2 do registro central de mercados (2026-07-06).

Os 6 mercados core do _V3_MARKETS (Análise de Linha) passaram a DERIVAR de
STATS_MARKET_CONFIG (v3_core_markets). Este teste trava os valores HISTÓRICOS
exatos — se a derivação mudar qualquer campo por acidente, quebra aqui, não
na UI. (Mesma metodologia do passo 1: test_stats_config_registry.py.)
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Snapshot LITERAL do _V3_MARKETS core como era antes da derivação (2026-07-06).
_EXPECTED_CORE = [
    {"label": "Kills", "key": "kills", "metric_col": "mv2_total_kills", "default_line": 27.5,
     "status": "ready", "metric_label": "Kills", "for_col": "kills_for", "against_col": "kills_against",
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "Torres", "key": "towers", "metric_col": "mv2_total_towers", "default_line": 11.5,
     "status": "ready", "metric_label": "Torres", "for_col": "towers_for", "against_col": "towers_against",
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "Dragões", "key": "dragons", "metric_col": "mv2_total_dragons", "default_line": 4.5,
     "status": "ready", "metric_label": "Dragões", "for_col": "dragons_for", "against_col": "dragons_against",
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "Barões", "key": "barons", "metric_col": "mv2_total_barons", "default_line": 1.5,
     "status": "ready", "metric_label": "Barões", "for_col": "barons_for", "against_col": "barons_against",
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "Inibidores", "key": "inibs", "metric_col": "mv2_total_inhibitors", "default_line": 1.5,
     "status": "ready", "metric_label": "Inibidores", "for_col": "inhibitors_for", "against_col": "inhibitors_against",
     "metric_decimals": 1, "is_time_metric": False},
    {"label": "Tempo", "key": "time", "metric_col": "game_time_min", "default_line": 32.5,
     "status": "ready", "metric_label": "Tempo", "for_col": None, "against_col": None,
     "metric_decimals": 1, "is_time_metric": True},
]


class TestV3MarketsRegistry(unittest.TestCase):
    @so_no_lol("linhas-padrao do LoL (kills 27,5; no Dota e 52,5)")
    def test_derived_core_matches_historical_values(self) -> None:
        from app_core.stats_config import v3_core_markets
        derived = v3_core_markets()
        self.assertEqual(len(derived), len(_EXPECTED_CORE))
        for got, exp in zip(derived, _EXPECTED_CORE):
            self.assertEqual(got, exp, f"mercado {exp['key']} divergiu do histórico")

    def test_v3_markets_full_list_shape(self) -> None:
        """Ordem e chaves completas do _V3_MARKETS (core derivado + extras UI)."""
        from app_core.line_analysis_tab_core import _V3_MARKETS
        keys = [m["key"] for m in _V3_MARKETS]
        self.assertEqual(keys, ["winner", "kills", "towers", "dragons", "barons",
                                "inibs", "time", "handicaps", "first_x",
                                "kills_team", "ml_totals"])
        # Handicap continua flagado (modo por-time da consistência V3)
        hc = next(m for m in _V3_MARKETS if m["key"] == "handicaps")
        self.assertTrue(hc.get("handicap"))


if __name__ == "__main__":
    unittest.main()
