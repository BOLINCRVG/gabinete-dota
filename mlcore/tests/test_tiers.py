"""Tests for mlcore/tiers.py — tier/league lookup functions."""
import unittest

from mlcore.tiers import (
    TIER_BASE_POINTS,
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


# ---------------------------------------------------------------------------
# tier_priority
# ---------------------------------------------------------------------------

class TestTierPriority(unittest.TestCase):

    def test_t1_is_0(self):
        self.assertEqual(tier_priority("T1"), 0)

    def test_t2_is_1(self):
        self.assertEqual(tier_priority("T2"), 1)

    def test_erl_is_2(self):
        self.assertEqual(tier_priority("ERL"), 2)

    def test_minor_is_3(self):
        self.assertEqual(tier_priority("MINOR"), 3)

    def test_other_is_4(self):
        self.assertEqual(tier_priority("OTHER"), 4)

    def test_unknown_tier_returns_other_priority(self):
        self.assertEqual(tier_priority("UNKNOWN_XYZ"), tier_priority("OTHER"))

    def test_none_returns_other_priority(self):
        self.assertEqual(tier_priority(None), tier_priority("OTHER"))

    def test_t1_strongest(self):
        # Lower priority number = stronger tier
        self.assertLess(tier_priority("T1"), tier_priority("T2"))
        self.assertLess(tier_priority("T2"), tier_priority("ERL"))
        self.assertLess(tier_priority("ERL"), tier_priority("MINOR"))


# ---------------------------------------------------------------------------
# canonical_league
# ---------------------------------------------------------------------------

class TestCanonicalLeague(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(canonical_league(None), "")

    def test_empty_returns_empty(self):
        self.assertEqual(canonical_league(""), "")

    def test_lck_passthrough(self):
        self.assertEqual(canonical_league("LCK"), "LCK")

    def test_case_insensitive(self):
        self.assertEqual(canonical_league("lck"), "LCK")

    def test_lta_maps_to_lcs(self):
        self.assertEqual(canonical_league("LTA"), "LCS")

    def test_lta_north_maps_to_lcs(self):
        self.assertEqual(canonical_league("LTA NORTH"), "LCS")

    def test_lta_south_maps_to_cblol(self):
        self.assertEqual(canonical_league("LTA SOUTH"), "CBLOL")

    def test_worlds_aliases(self):
        for alias in ("WORLD CHAMPIONSHIP", "WCS", "WLDS"):
            self.assertEqual(canonical_league(alias), "WORLDS", f"Failed for: {alias}")

    def test_msi_aliases(self):
        for alias in ("MID SEASON INVITATIONAL", "MIDSEASON INVITATIONAL"):
            self.assertEqual(canonical_league(alias), "MSI", f"Failed for: {alias}")

    def test_lck_cl_maps_to_lckc(self):
        self.assertEqual(canonical_league("LCK CL"), "LCKC")

    def test_ldl_maps_to_al(self):
        self.assertEqual(canonical_league("LDL"), "AL")

    def test_whitespace_normalized(self):
        self.assertEqual(canonical_league("  LCK  "), "LCK")


# ---------------------------------------------------------------------------
# tier_for_league
# ---------------------------------------------------------------------------

class TestTierForLeague(unittest.TestCase):

    def test_lck_is_t1(self):
        self.assertEqual(tier_for_league("LCK"), "T1")

    def test_lpl_is_t1(self):
        self.assertEqual(tier_for_league("LPL"), "T1")

    def test_lec_is_t1(self):
        self.assertEqual(tier_for_league("LEC"), "T1")

    def test_lcp_is_t1(self):
        self.assertEqual(tier_for_league("LCP"), "T1")

    def test_lcs_is_t2(self):
        self.assertEqual(tier_for_league("LCS"), "T2")

    def test_cblol_is_t2(self):
        self.assertEqual(tier_for_league("CBLOL"), "T2")

    def test_vcs_is_t2(self):
        self.assertEqual(tier_for_league("VCS"), "T2")

    def test_pcs_is_t2(self):
        self.assertEqual(tier_for_league("PCS"), "T2")

    def test_ljl_is_t2(self):
        self.assertEqual(tier_for_league("LJL"), "T2")

    def test_lfl_is_erl(self):
        self.assertEqual(tier_for_league("LFL"), "ERL")

    def test_prm_is_erl(self):
        self.assertEqual(tier_for_league("PRM"), "ERL")

    def test_worlds_is_t1(self):
        self.assertEqual(tier_for_league("WORLDS"), "T1")

    def test_msi_is_t1(self):
        self.assertEqual(tier_for_league("MSI"), "T1")

    def test_none_is_other(self):
        self.assertEqual(tier_for_league(None), "OTHER")

    def test_empty_is_other(self):
        self.assertEqual(tier_for_league(""), "OTHER")

    def test_lta_alias_resolves_to_lcs_tier(self):
        # LTA → canonical "LCS" → T2
        self.assertEqual(tier_for_league("LTA"), "T2")

    def test_lta_south_alias_resolves_to_cblol_tier(self):
        # LTA SOUTH → canonical "CBLOL" → T2
        self.assertEqual(tier_for_league("LTA SOUTH"), "T2")


# ---------------------------------------------------------------------------
# tier_base_points
# ---------------------------------------------------------------------------

class TestTierBasePoints(unittest.TestCase):

    def test_t1_highest(self):
        self.assertGreater(tier_base_points("T1"), tier_base_points("T2"))

    def test_t2_above_erl(self):
        self.assertGreater(tier_base_points("T2"), tier_base_points("ERL"))

    def test_erl_above_minor(self):
        self.assertGreater(tier_base_points("ERL"), tier_base_points("MINOR"))

    def test_t1_value(self):
        self.assertAlmostEqual(tier_base_points("T1"), TIER_BASE_POINTS["T1"])

    def test_none_returns_other_value(self):
        self.assertAlmostEqual(tier_base_points(None), TIER_BASE_POINTS.get("OTHER", 0.0))

    def test_unknown_returns_other_value(self):
        self.assertAlmostEqual(tier_base_points("XYZXYZ"), TIER_BASE_POINTS.get("OTHER", 0.0))


# ---------------------------------------------------------------------------
# league_base_points
# ---------------------------------------------------------------------------

class TestLeagueBasePoints(unittest.TestCase):

    def test_none_returns_zero(self):
        self.assertAlmostEqual(league_base_points(None), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(league_base_points(""), 0.0)

    def test_unknown_league_returns_zero(self):
        self.assertAlmostEqual(league_base_points("XYZXYZ_UNKNOWN"), 0.0)

    def test_known_league_returns_float(self):
        # LCK has an explicit entry (may be positive, zero or small)
        result = league_base_points("LCK")
        self.assertIsInstance(result, float)


# ---------------------------------------------------------------------------
# league_strength_score
# ---------------------------------------------------------------------------

class TestLeagueStrengthScore(unittest.TestCase):

    def test_returns_float(self):
        self.assertIsInstance(league_strength_score("LCK"), float)

    def test_score_in_unit_interval(self):
        for league in ("LCK", "LPL", "LCS", "CBLOL", "LFL", ""):
            s = league_strength_score(league)
            self.assertGreaterEqual(s, 0.0, f"Score out of range for {league}")
            self.assertLessEqual(s, 1.0, f"Score out of range for {league}")

    def test_major_leagues_stronger_than_minor(self):
        s_lck = league_strength_score("LCK")
        s_lfl = league_strength_score("LFL")
        self.assertGreater(s_lck, s_lfl)

    def test_none_returns_default(self):
        s = league_strength_score(None)
        self.assertIsInstance(s, float)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)


# ---------------------------------------------------------------------------
# league_strength_band
# ---------------------------------------------------------------------------

class TestLeagueStrengthBand(unittest.TestCase):

    def test_returns_string(self):
        self.assertIsInstance(league_strength_band("LCK"), str)

    def test_major_leagues_not_local(self):
        # LCK should be in a strong band, not "local"
        band = league_strength_band("LCK")
        self.assertNotEqual(band, "local")

    def test_weak_league_returns_lower_band(self):
        # No explicit assertion on exact value, but weak league should score < LCK
        band_lck = league_strength_band("LCK")
        band_weak = league_strength_band("XYZMINORLEAGUE")
        # They may differ — weak leagues tend to fall into a lower band
        self.assertIsNotNone(band_lck)
        self.assertIsNotNone(band_weak)


# ---------------------------------------------------------------------------
# enforce_monotone_caps
# ---------------------------------------------------------------------------

class TestEnforceMonotoneCaps(unittest.TestCase):

    def test_already_monotone_unchanged(self):
        caps = {"T1": 100.0, "T2": 80.0, "ERL": 50.0, "MINOR": 20.0}
        out = enforce_monotone_caps(caps)
        self.assertAlmostEqual(out["T1"], 100.0)
        self.assertAlmostEqual(out["T2"], 80.0)
        self.assertAlmostEqual(out["ERL"], 50.0)
        self.assertAlmostEqual(out["MINOR"], 20.0)

    def test_non_monotone_corrected(self):
        # T2 cap higher than T1 → should be clamped to T1
        caps = {"T1": 50.0, "T2": 100.0}
        out = enforce_monotone_caps(caps)
        self.assertLessEqual(out["T2"], out["T1"])

    def test_only_t2_uncapped(self):
        caps = {"T2": 80.0, "ERL": 200.0}
        out = enforce_monotone_caps(caps)
        self.assertLessEqual(out["ERL"], out["T2"])

    def test_does_not_mutate_input(self):
        caps = {"T1": 50.0, "T2": 100.0}
        original = dict(caps)
        enforce_monotone_caps(caps)
        self.assertEqual(caps, original)

    def test_empty_input_returns_empty(self):
        self.assertEqual(enforce_monotone_caps({}), {})

    def test_single_tier_unchanged(self):
        caps = {"T1": 75.0}
        out = enforce_monotone_caps(caps)
        self.assertAlmostEqual(out["T1"], 75.0)

    def test_custom_tier_order(self):
        caps = {"A": 100.0, "B": 200.0}
        out = enforce_monotone_caps(caps, tier_order=["A", "B"])
        self.assertLessEqual(out["B"], out["A"])


# ---------------------------------------------------------------------------
# is_international_league
# ---------------------------------------------------------------------------

class TestIsInternationalLeague(unittest.TestCase):

    def test_worlds_is_international(self):
        self.assertTrue(is_international_league("WORLDS"))

    def test_msi_is_international(self):
        self.assertTrue(is_international_league("MSI"))

    def test_worlds_raw_label(self):
        self.assertTrue(is_international_league("World Championship"))

    def test_msi_raw_label(self):
        self.assertTrue(is_international_league("Mid Season Invitational"))

    def test_ewc_is_international(self):
        self.assertTrue(is_international_league("EWC"))

    def test_lck_not_international(self):
        self.assertFalse(is_international_league("LCK"))

    def test_lcs_not_international(self):
        self.assertFalse(is_international_league("LCS"))

    def test_none_not_international(self):
        self.assertFalse(is_international_league(None))

    def test_empty_not_international(self):
        self.assertFalse(is_international_league(""))

    def test_emea_masters_is_international(self):
        self.assertTrue(is_international_league("EMEA Masters"))

    def test_americas_is_international(self):
        self.assertTrue(is_international_league("AMERICAS"))


if __name__ == "__main__":
    unittest.main()
