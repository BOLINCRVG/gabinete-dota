# -*- coding: utf-8 -*-
"""player_lines_opp_split: média K/D/A por força do adversário (pedido dele 24/07)."""
import unittest

import pandas as pd

from app_core.players_tab_core import player_lines_opp_split


def _row(gid, team, player, k, res):
    return {"gameid": gid, "team": team, "player": player,
            "kills": k, "deaths": 1, "assists": 1, "result": res}


class PlayerOppSplitTests(unittest.TestCase):
    def _dfb(self):
        rows = []
        # win-rates espalhadas p/ terciles: FORTE~1, FRACO~0, +médios
        for i in range(6):
            rows += [_row(f"s{i}", "FORTE", "x", 0, 1), _row(f"s{i}", f"perde{i}", "y", 0, 0)]
        for i in range(6):
            rows += [_row(f"w{i}", "FRACO", "z", 0, 0), _row(f"w{i}", f"ganha{i}", "w", 0, 1)]
        for i in range(4):
            rows += [_row(f"m{i}", "MED", "m", 0, i % 2), _row(f"m{i}", f"om{i}", "n", 0, 1 - i % 2)]
        # player P (time T): 2 kills vs FORTE, 8 vs FRACO
        rows += [_row("g1", "T", "P", 2, 0), _row("g1", "FORTE", "e1", 0, 1)]
        rows += [_row("g2", "T", "P", 8, 1), _row("g2", "FRACO", "e2", 0, 0)]
        return pd.DataFrame(rows)

    def test_hoje_cai_contra_forte(self):
        df = self._dfb()
        res = player_lines_opp_split(df, ["P"], "T", "FORTE")
        self.assertEqual(len(res), 1)
        r = res[0]
        self.assertEqual(r["today_bucket"], "forte")
        self.assertAlmostEqual(r["hoje"]["k"], 2.0)      # só o jogo vs FORTE
        self.assertAlmostEqual(r["geral"]["k"], 5.0)     # média (2+8)/2
        self.assertEqual(r["geral"]["n"], 2)

    def test_vazio_sem_players(self):
        self.assertEqual(player_lines_opp_split(self._dfb(), [], "T", "FORTE"), [])

    def test_df_sem_colunas(self):
        self.assertEqual(player_lines_opp_split(pd.DataFrame({"x": [1]}), ["P"], "T", "F"), [])
