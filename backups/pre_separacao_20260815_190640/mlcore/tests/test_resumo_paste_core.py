"""Tests for app_core.resumo_paste_core."""
from __future__ import annotations

import unittest

from app_core.resumo_paste_core import (
    ALLOWED_UNITS,
    apply_paste_to_oc_map,
    classify_pick,
    discrete_stake_units,
    kelly_stake_fraction,
    parse_odds_paste,
    row_signature,
)


class TestParseOddsPaste(unittest.TestCase):

    def test_basic_kills_and_ml(self):
        out = parse_odds_paste("K27.5/1.85 ML-A/1.45")
        self.assertEqual(out[("kills", "Over", "27.5")], 1.85)
        self.assertEqual(out[("ml", "A", "")], 1.45)

    def test_under_suffix(self):
        out = parse_odds_paste("K27.5U/2.10 K28.5O/1.95")
        self.assertEqual(out[("kills", "Under", "27.5")], 2.10)
        self.assertEqual(out[("kills", "Over", "28.5")], 1.95)

    def test_time_mmss_normalized(self):
        out = parse_odds_paste("TM32:30/1.85")
        self.assertIn(("time", "Over", "32.5"), out)

    def test_multiple_separators(self):
        out = parse_odds_paste("K27.5/1.85,T11.5/1.65;D4.5/1.90")
        self.assertEqual(len(out), 3)
        self.assertEqual(out[("towers", "Over", "11.5")], 1.65)
        self.assertEqual(out[("dragons", "Over", "4.5")], 1.90)

    def test_alternative_separators(self):
        # /, =, @ are all valid
        out = parse_odds_paste("K27.5=1.85 K28.5@2.10")
        self.assertEqual(out[("kills", "Over", "27.5")], 1.85)
        self.assertEqual(out[("kills", "Over", "28.5")], 2.10)

    def test_invalid_tokens_dropped_silently(self):
        out = parse_odds_paste("foo K27.5/1.85 bar123 ML-A/notanumber")
        self.assertEqual(len(out), 1)
        self.assertIn(("kills", "Over", "27.5"), out)

    def test_odd_must_be_above_one(self):
        out = parse_odds_paste("K27.5/0.95 K28.5/1.85")
        self.assertNotIn(("kills", "Over", "27.5"), out)
        self.assertIn(("kills", "Over", "28.5"), out)

    def test_empty_string(self):
        self.assertEqual(parse_odds_paste(""), {})
        self.assertEqual(parse_odds_paste("   "), {})

    def test_long_alias_preferred_over_short(self):
        # "TORRES" must not be matched as "T" + "ORRES" garbage
        out = parse_odds_paste("TORRES11.5/1.65")
        self.assertIn(("towers", "Over", "11.5"), out)

    def test_ml_team_aliases(self):
        out = parse_odds_paste("MLA/1.45 MLB/2.85")
        self.assertEqual(out[("ml", "A", "")], 1.45)
        self.assertEqual(out[("ml", "B", "")], 2.85)


class TestRowSignature(unittest.TestCase):

    def test_kills_over_row(self):
        row = {"_metric": "kills", "_line": 27.5, "Lado": "Acima"}
        self.assertEqual(row_signature(row, team_a="A", team_b="B"),
                         ("kills", "Over", "27.5"))

    def test_kills_under_row(self):
        row = {"_metric": "kills", "_line": 28.5, "Lado": "Abaixo"}
        self.assertEqual(row_signature(row, team_a="A", team_b="B"),
                         ("kills", "Under", "28.5"))

    def test_ml_a_row(self):
        row = {"_metric": "00_ml_A", "_line": 0.0, "Lado": "Gen.G"}
        self.assertEqual(row_signature(row, team_a="Gen.G", team_b="T1"),
                         ("ml", "A", ""))

    def test_ml_b_row(self):
        row = {"_metric": "00_ml_B", "_line": 0.0, "Lado": "T1"}
        self.assertEqual(row_signature(row, team_a="Gen.G", team_b="T1"),
                         ("ml", "B", ""))

    def test_combo_row_returns_none(self):
        row = {"_metric": "zz_mltot_A_kills", "_line": 27.5, "Lado": "ML & Acima"}
        self.assertIsNone(row_signature(row, team_a="A", team_b="B"))


class TestApplyPasteToOcMap(unittest.TestCase):

    def test_crosses_paste_with_rows(self):
        parsed = parse_odds_paste("K27.5/1.85 ML-A/1.45")
        rows = [
            {"_metric": "kills", "_line": 27.5, "Lado": "Acima",
             "Mercado": "Kills (27.5)", "Linha": "27.5"},
            {"_metric": "00_ml_A", "_line": 0.0, "Lado": "Gen.G",
             "Mercado": "ML (Mapa)", "Linha": "-"},
            {"_metric": "towers", "_line": 11.5, "Lado": "Acima",
             "Mercado": "Torres (11.5)", "Linha": "11.5"},  # not in paste
        ]
        out = apply_paste_to_oc_map(parsed, rows, team_a="Gen.G", team_b="T1")
        self.assertEqual(out["Kills (27.5)|Acima|27.5"], 1.85)
        self.assertEqual(out["ML (Mapa)|Gen.G|-"], 1.45)
        self.assertNotIn("Torres (11.5)|Acima|11.5", out)


class TestKellyStake(unittest.TestCase):

    def test_positive_edge_returns_quarter_kelly(self):
        # p=0.6, odd=2.0 -> full kelly = (0.6*1 - 0.4)/1 = 0.2 -> 0.05 quarter
        k = kelly_stake_fraction(0.6, 2.0, fraction=0.25)
        self.assertIsNotNone(k)
        self.assertAlmostEqual(k, 0.05, places=4)

    def test_negative_edge_returns_zero(self):
        # p=0.4, odd=2.0 -> kelly negative
        self.assertEqual(kelly_stake_fraction(0.4, 2.0), 0.0)

    def test_missing_inputs_return_none(self):
        self.assertIsNone(kelly_stake_fraction(None, 2.0))
        self.assertIsNone(kelly_stake_fraction(0.6, None))

    def test_invalid_odd_returns_none(self):
        self.assertIsNone(kelly_stake_fraction(0.6, 1.0))
        self.assertIsNone(kelly_stake_fraction(0.6, 0.5))

    def test_capped_at_one(self):
        # Extreme edge case: p=0.99, odd=10 -> kelly ~ very high, capped at 1
        k = kelly_stake_fraction(0.99, 10.0, fraction=1.0)
        self.assertIsNotNone(k)
        self.assertLessEqual(k, 1.0)


class TestDiscreteStakeUnits(unittest.TestCase):

    def test_ladder_boundaries(self):
        # Each boundary maps to the lower bound's unit.
        self.assertEqual(discrete_stake_units(0.040), 2.0)    # exactly 4%
        self.assertEqual(discrete_stake_units(0.050), 2.0)    # above 4%
        self.assertEqual(discrete_stake_units(0.030), 1.5)    # 3%
        self.assertEqual(discrete_stake_units(0.0225), 1.25)  # 2.25%
        self.assertEqual(discrete_stake_units(0.015), 1.0)    # 1.5%
        self.assertEqual(discrete_stake_units(0.010), 0.75)   # 1%
        self.assertEqual(discrete_stake_units(0.005), 0.5)    # 0.5%

    def test_below_threshold_returns_none(self):
        self.assertIsNone(discrete_stake_units(0.004))  # 0.4%
        self.assertIsNone(discrete_stake_units(0.001))

    def test_zero_or_negative_returns_none(self):
        self.assertIsNone(discrete_stake_units(0.0))
        self.assertIsNone(discrete_stake_units(-0.05))

    def test_invalid_returns_none(self):
        self.assertIsNone(discrete_stake_units(None))
        self.assertIsNone(discrete_stake_units(float("nan")))
        self.assertIsNone(discrete_stake_units(float("inf")))

    def test_all_results_in_allowed_ladder(self):
        # Sanity: every non-None result must be one of the user's allowed units.
        for kf in [0.001, 0.005, 0.010, 0.015, 0.0225, 0.030, 0.040, 0.10]:
            u = discrete_stake_units(kf)
            if u is not None:
                self.assertIn(u, ALLOWED_UNITS, f"kf={kf} returned {u} which is not in ladder")

    def test_huge_kelly_caps_at_2u(self):
        # No matter how strong, the ladder maxes at 2u.
        self.assertEqual(discrete_stake_units(0.5), 2.0)


class TestClassifyPick(unittest.TestCase):

    def test_recommend_strong_edge_and_conf(self):
        self.assertEqual(classify_pick(edge_hist_pct=8.0, conf_score=70), "recommend")

    def test_watch_marginal(self):
        self.assertEqual(classify_pick(edge_hist_pct=2.0, conf_score=55), "watch")

    def test_avoid_negative_edge(self):
        self.assertEqual(classify_pick(edge_hist_pct=-5.0, conf_score=80), "avoid")

    def test_avoid_low_conf(self):
        self.assertEqual(classify_pick(edge_hist_pct=10.0, conf_score=35), "avoid")

    def test_fallback_when_edge_none(self):
        self.assertEqual(classify_pick(edge_hist_pct=None, conf_score=75), "recommend")
        self.assertEqual(classify_pick(edge_hist_pct=None, conf_score=55), "watch")
        self.assertEqual(classify_pick(edge_hist_pct=None, conf_score=30), "avoid")


if __name__ == "__main__":
    unittest.main()
