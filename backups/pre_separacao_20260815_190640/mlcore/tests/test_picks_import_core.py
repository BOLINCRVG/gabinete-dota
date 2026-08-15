"""Tests for app_core.picks_import_core — xlsx backfill parsing (synthetic)."""
from __future__ import annotations

import unittest

from app_core.picks_import_core import (
    derive_stake_units,
    parse_aposta,
    parse_line,
    parse_outcome,
    parse_pick_row,
)


class TestParseAposta(unittest.TestCase):

    def test_under_kill(self):
        self.assertEqual(parse_aposta("Under Kill"), ("kills", "Under"))

    def test_over_kill(self):
        self.assertEqual(parse_aposta("Over Kill"), ("kills", "Over"))

    def test_under_torre(self):
        self.assertEqual(parse_aposta("Under Torre"), ("towers", "Under"))

    def test_dragao_with_mojibake(self):
        # 'Under Drag�o' (replacement char) must still map to dragons.
        self.assertEqual(parse_aposta("Under Drag�o"), ("dragons", "Under"))

    def test_dragao_accented(self):
        self.assertEqual(parse_aposta("Under Dragão"), ("dragons", "Under"))

    def test_under_baron(self):
        self.assertEqual(parse_aposta("Under Baron"), ("barons", "Under"))

    def test_under_inib(self):
        self.assertEqual(parse_aposta("Under Inib"), ("inhibitors", "Under"))
        self.assertEqual(parse_aposta("Under Inibidores"), ("inhibitors", "Under"))

    def test_under_tempo(self):
        self.assertEqual(parse_aposta("Under Tempo"), ("time", "Under"))

    def test_plain_ml(self):
        self.assertEqual(parse_aposta("ML"), ("ml", "A"))

    def test_combo_ml_rejected(self):
        self.assertIsNone(parse_aposta("ML / Under Kill"))
        self.assertIsNone(parse_aposta("ML - Over Tempo"))

    def test_handicap_rejected(self):
        self.assertIsNone(parse_aposta("Handicap Kill"))
        self.assertIsNone(parse_aposta("Handicap Mapa"))

    def test_player_props_rejected(self):
        self.assertIsNone(parse_aposta("Jogador Under Kill"))
        self.assertIsNone(parse_aposta("Duel players"))
        self.assertIsNone(parse_aposta("Pick Jogador"))

    def test_team_specific_rejected(self):
        self.assertIsNone(parse_aposta("Time Under Kill"))

    def test_first_x_rejected(self):
        self.assertIsNone(parse_aposta("First Blood"))
        self.assertIsNone(parse_aposta("First Atakhan"))

    def test_misc_rejected(self):
        for s in ["Outright", "Proxima Kill", "F5", "30 - 39", "Over Gols 1T", "Multipla"]:
            self.assertIsNone(parse_aposta(s), f"{s} should be rejected")

    def test_empty_returns_none(self):
        self.assertIsNone(parse_aposta(""))
        self.assertIsNone(parse_aposta(None))


class TestParseLine(unittest.TestCase):

    def test_comma_decimal(self):
        self.assertEqual(parse_line("27,5"), 27.5)

    def test_dot_decimal(self):
        self.assertEqual(parse_line("11.5"), 11.5)

    def test_negative(self):
        self.assertEqual(parse_line("-16.5"), -16.5)

    def test_time_mmss(self):
        self.assertAlmostEqual(parse_line("31:30", is_time=True), 31.5)

    def test_blank_none(self):
        self.assertIsNone(parse_line(""))
        self.assertIsNone(parse_line(None))
        self.assertIsNone(parse_line("nan"))


class TestParseOutcome(unittest.TestCase):

    def test_win(self):
        self.assertEqual(parse_outcome("W"), "won")
        self.assertEqual(parse_outcome("w"), "won")

    def test_loss(self):
        self.assertEqual(parse_outcome("L"), "lost")

    def test_cashout_void(self):
        self.assertEqual(parse_outcome("co"), "void")
        self.assertEqual(parse_outcome("CO"), "void")

    def test_unknown_none(self):
        self.assertIsNone(parse_outcome("?"))
        self.assertIsNone(parse_outcome(""))


class TestDeriveStakeUnits(unittest.TestCase):

    def test_won_recovers_stake(self):
        # profit = stake*(odd-1) → 0.85 with odd 1.85 → stake 1.0
        s = derive_stake_units(status="won", profit_units=0.85, odd=1.85)
        self.assertAlmostEqual(s, 1.0, places=4)

    def test_lost_recovers_stake(self):
        s = derive_stake_units(status="lost", profit_units=-1.0, odd=1.85)
        self.assertEqual(s, 1.0)

    def test_void_zero(self):
        self.assertEqual(derive_stake_units(status="void", profit_units=0.0, odd=1.85), 0.0)

    def test_won_missing_odd_none(self):
        self.assertIsNone(derive_stake_units(status="won", profit_units=0.85, odd=None))

    def test_missing_profit_none(self):
        self.assertIsNone(derive_stake_units(status="lost", profit_units=None, odd=1.85))


class TestParsePickRow(unittest.TestCase):

    def _row(self, **over):
        base = {
            "Data": "2026-05-02", "Esporte": "LOL", "Liga": "LCK",
            "Blue/Casa": "T1", "Red/Fora": "GenG", "Mapa": "Mapa 1",
            "Aposta": "Under Kill", "Linha": "27,5", "Equipe": "",
            "Stake (R$)": 60.0, "Odd": 1.85, "W/L": "W", "P/L (U)": 0.85,
        }
        base.update(over)
        return base

    def test_basic_under_kill_won(self):
        out = parse_pick_row(self._row())
        self.assertEqual(out["metric_key"], "kills")
        self.assertEqual(out["side_key"], "Under")
        self.assertEqual(out["line"], 27.5)
        self.assertEqual(out["status"], "won")
        self.assertAlmostEqual(out["stake_units"], 1.0, places=3)
        self.assertEqual(out["source"], "xlsx")
        self.assertEqual(out["league"], "LCK")

    def test_ml_side_a_when_equipe_is_blue(self):
        out = parse_pick_row(self._row(Aposta="ML", Linha="", Equipe="T1", **{"P/L (U)": 1.0, "Odd": 2.0, "W/L": "W"}))
        self.assertEqual(out["metric_key"], "ml")
        self.assertEqual(out["side_key"], "A")
        self.assertIsNone(out["line"])

    def test_ml_side_b_when_equipe_is_red(self):
        out = parse_pick_row(self._row(Aposta="ML", Linha="", Equipe="GenG", **{"P/L (U)": 1.0, "Odd": 2.0, "W/L": "W"}))
        self.assertEqual(out["side_key"], "B")

    def test_non_lol_skipped(self):
        self.assertIsNone(parse_pick_row(self._row(Esporte="DOTA2")))

    def test_unsupported_market_skipped(self):
        self.assertIsNone(parse_pick_row(self._row(Aposta="Handicap Kill")))

    def test_totals_without_line_skipped(self):
        self.assertIsNone(parse_pick_row(self._row(Aposta="Under Kill", Linha="")))

    def test_lost_row(self):
        out = parse_pick_row(self._row(**{"W/L": "L", "P/L (U)": -1.0}))
        self.assertEqual(out["status"], "lost")
        self.assertEqual(out["stake_units"], 1.0)
        self.assertEqual(out["profit_units"], -1.0)

    def test_void_row(self):
        out = parse_pick_row(self._row(**{"W/L": "co", "P/L (U)": 0.0}))
        self.assertEqual(out["status"], "void")
        self.assertEqual(out["stake_units"], 0.0)

    def test_time_market_mmss_line(self):
        out = parse_pick_row(self._row(Aposta="Under Tempo", Linha="31:30"))
        self.assertEqual(out["metric_key"], "time")
        self.assertAlmostEqual(out["line"], 31.5, places=3)


if __name__ == "__main__":
    unittest.main()
