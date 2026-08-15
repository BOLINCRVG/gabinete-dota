import unittest

from mlcore.tiers import (
    canonical_league,
    enforce_monotone_caps,
    is_international_league,
    league_base_points,
    league_strength_band,
    league_strength_score,
    tier_base_points,
    tier_for_league,
    tier_priority,
)


class TiersCoreTest(unittest.TestCase):
    def test_canonical_league_aliases(self):
        self.assertEqual(canonical_league("LTA North"), "LCS")
        self.assertEqual(canonical_league("LTA South"), "CBLOL")
        self.assertEqual(canonical_league("Americas"), "AMERICAS")
        self.assertEqual(canonical_league("Cross Conference"), "AMERICAS")
        self.assertEqual(canonical_league("World Championship"), "WORLDS")
        self.assertEqual(canonical_league("WLDS"), "WORLDS")
        self.assertEqual(canonical_league("LCK Challengers"), "LCKC")
        self.assertEqual(canonical_league("Prime League"), "PRM")

    def test_tier_for_league_handles_case_and_aliases(self):
        self.assertEqual(tier_for_league("lcs"), "T2")
        self.assertEqual(tier_for_league("CBLOL"), "T2")
        self.assertEqual(tier_for_league("AMERICAS"), "T2")
        self.assertEqual(tier_for_league("World Championship"), "T1")
        self.assertEqual(tier_for_league("LCK CL"), "MINOR")
        self.assertEqual(tier_for_league("LAS"), "MINOR")
        self.assertEqual(tier_for_league("NLC AURORA OPEN"), "ERL")

    def test_league_base_points_uses_canonical_keys(self):
        self.assertEqual(float(league_base_points("world championship")), 0.0)
        self.assertEqual(float(league_base_points("LTA North")), float(league_base_points("LCS")))

    def test_league_strength_score_orders_structural_leagues(self):
        self.assertGreater(league_strength_score("LCK"), league_strength_score("LEC"))
        self.assertGreater(league_strength_score("LEC"), league_strength_score("LCS"))
        self.assertGreater(league_strength_score("LCS"), league_strength_score("CBLOL"))
        self.assertGreater(league_strength_score("CBLOL"), league_strength_score("LFL"))
        self.assertGreater(league_strength_score("LFL"), league_strength_score("LCK CL"))
        self.assertGreater(league_strength_score("LCK CL"), league_strength_score("LAS"))

    def test_league_strength_band_is_granular(self):
        self.assertEqual(league_strength_band("LCK"), "elite")
        self.assertEqual(league_strength_band("LEC"), "major")
        self.assertEqual(league_strength_band("LCS"), "upper")
        self.assertEqual(league_strength_band("LFL"), "regional")
        self.assertEqual(league_strength_band("LCK CL"), "development")
        self.assertEqual(league_strength_band("LAS"), "local")

    def test_international_helper_covers_fst_and_americas_cup(self):
        self.assertTrue(is_international_league("FST"))
        self.assertTrue(is_international_league("First Stand"))
        self.assertTrue(is_international_league("Americas Cup"))
        self.assertTrue(is_international_league("Cross Conference"))
        self.assertFalse(is_international_league("CBLOL"))


class TestTierPriorityAndBasePoints(unittest.TestCase):

    def test_t1_lower_priority_number_than_t2(self):
        # Lower number = higher rank (T1=0 is best)
        self.assertLess(tier_priority("T1"), tier_priority("T2"))
        self.assertLess(tier_priority("T2"), tier_priority("MINOR"))

    def test_unknown_tier_handled(self):
        result = tier_priority("UNKNOWN")
        self.assertIsInstance(result, int)

    def test_tier_base_points_t1_highest(self):
        self.assertGreater(tier_base_points("T1"), tier_base_points("T2"))
        self.assertGreater(tier_base_points("T2"), tier_base_points("MINOR"))

    def test_tier_base_points_returns_float(self):
        self.assertIsInstance(tier_base_points("T1"), float)

    def test_league_base_points_lck_positive(self):
        self.assertGreater(league_base_points("LCK"), 0.0)

    def test_league_base_points_unknown_zero(self):
        self.assertAlmostEqual(league_base_points("UNKNOWN_LEAGUE_XYZ"), 0.0)


class TestEnforceMonotoneCaps(unittest.TestCase):

    def test_already_monotone_unchanged(self):
        caps = {"T1": 0.90, "T2": 0.80, "MINOR": 0.70}
        result = enforce_monotone_caps(caps)
        self.assertAlmostEqual(result["T1"], 0.90)
        self.assertAlmostEqual(result["T2"], 0.80)
        self.assertAlmostEqual(result["MINOR"], 0.70)

    def test_violating_t2_clamped_to_t1(self):
        caps = {"T1": 0.80, "T2": 0.90}
        result = enforce_monotone_caps(caps)
        self.assertLessEqual(result["T2"], result["T1"])

    def test_empty_dict_returns_empty(self):
        self.assertEqual(enforce_monotone_caps({}), {})

    def test_returns_dict(self):
        self.assertIsInstance(enforce_monotone_caps({"T1": 0.8}), dict)

    def test_does_not_mutate_input(self):
        caps = {"T1": 0.80, "T2": 0.90}
        original = dict(caps)
        enforce_monotone_caps(caps)
        self.assertEqual(caps, original)


if __name__ == "__main__":
    unittest.main()
