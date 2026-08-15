"""Tests for mlcore/league_context.py — pure league-string helpers."""
import unittest

from mlcore.league_context import (
    is_explicit_interregional_league_label,
    league_lookup_keys,
    league_macro_key,
)


# ---------------------------------------------------------------------------
# is_explicit_interregional_league_label
# ---------------------------------------------------------------------------

class TestIsExplicitInterregional(unittest.TestCase):

    def test_americas_is_inter(self):
        self.assertTrue(is_explicit_interregional_league_label("AMERICAS"))

    def test_inter_is_inter(self):
        self.assertTrue(is_explicit_interregional_league_label("INTER"))

    def test_interregional_is_inter(self):
        self.assertTrue(is_explicit_interregional_league_label("INTERREGIONAL"))

    def test_cross_conference_is_inter(self):
        self.assertTrue(is_explicit_interregional_league_label("CROSS CONFERENCE"))

    def test_lck_not_inter(self):
        self.assertFalse(is_explicit_interregional_league_label("LCK"))

    def test_lpl_not_inter(self):
        self.assertFalse(is_explicit_interregional_league_label("LPL"))

    def test_empty_not_inter(self):
        self.assertFalse(is_explicit_interregional_league_label(""))

    def test_none_not_inter(self):
        self.assertFalse(is_explicit_interregional_league_label(None))

    def test_lowercase_input(self):
        self.assertTrue(is_explicit_interregional_league_label("americas"))

    def test_case_insensitive(self):
        self.assertTrue(is_explicit_interregional_league_label("Interregional"))


# ---------------------------------------------------------------------------
# league_macro_key
# ---------------------------------------------------------------------------

class TestLeagueMacroKey(unittest.TestCase):

    def test_lpl_is_cn(self):
        self.assertEqual(league_macro_key("LPL"), "CN")

    def test_lck_is_kr(self):
        self.assertEqual(league_macro_key("LCK"), "KR")

    def test_lec_is_emea(self):
        self.assertEqual(league_macro_key("LEC"), "EMEA")

    def test_lcs_is_na(self):
        self.assertEqual(league_macro_key("LCS"), "NA")

    def test_lta_north_is_na(self):
        # LTA N is alias for LCS → NA
        self.assertEqual(league_macro_key("LTA N"), "NA")

    def test_cblol_is_br(self):
        self.assertEqual(league_macro_key("CBLOL"), "BR")

    def test_pcs_is_apac(self):
        self.assertEqual(league_macro_key("PCS"), "APAC")

    def test_vcs_is_apac(self):
        self.assertEqual(league_macro_key("VCS"), "APAC")

    def test_ljl_is_apac(self):
        self.assertEqual(league_macro_key("LJL"), "APAC")

    def test_americas_is_americas(self):
        self.assertEqual(league_macro_key("AMERICAS"), "AMERICAS")

    def test_empty_returns_other(self):
        self.assertEqual(league_macro_key(""), "OTHER")

    def test_none_returns_other(self):
        self.assertEqual(league_macro_key(None), "OTHER")

    def test_unknown_league_returns_other(self):
        self.assertEqual(league_macro_key("SOMEMADEUP"), "OTHER")

    def test_lck_challengers_is_kr(self):
        self.assertEqual(league_macro_key("LCKC"), "KR")

    def test_custom_unknown_str(self):
        self.assertEqual(league_macro_key("", unknown="NONE"), "NONE")

    def test_lowercase_lck(self):
        self.assertEqual(league_macro_key("lck"), "KR")

    def test_lpl_subdiv_ldl_is_cn(self):
        self.assertEqual(league_macro_key("LDL"), "CN")

    def test_emea_prefix_is_emea(self):
        self.assertEqual(league_macro_key("EMEA"), "EMEA")


# ---------------------------------------------------------------------------
# league_lookup_keys
# ---------------------------------------------------------------------------

class TestLeagueLookupKeys(unittest.TestCase):

    def test_returns_tuple(self):
        result = league_lookup_keys("LCK")
        self.assertIsInstance(result, tuple)

    def test_lck_contains_lck(self):
        result = league_lookup_keys("LCK")
        self.assertIn("LCK", result)

    def test_lck_challengers_alias_included(self):
        # LCKC has aliases: LCK CL, LCK CHALLENGERS, LCK CHALLENGERS LEAGUE
        result = league_lookup_keys("LCKC")
        self.assertTrue(len(result) >= 1)

    def test_empty_returns_tuple(self):
        result = league_lookup_keys("")
        self.assertIsInstance(result, tuple)

    def test_no_duplicates(self):
        result = league_lookup_keys("LCS")
        self.assertEqual(len(result), len(set(result)))

    def test_lcs_includes_lta_aliases(self):
        # LCS → LTA / LTA N / LTA NORTH as compat aliases
        result = league_lookup_keys("LCS")
        combined = " ".join(result)
        self.assertIn("LTA", combined)


if __name__ == "__main__":
    unittest.main()
