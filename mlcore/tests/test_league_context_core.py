import unittest

from mlcore.league_context import is_explicit_interregional_league_label, league_lookup_keys, league_macro_key


class TestIsExplicitInterregionalLeagueLabel(unittest.TestCase):

    def test_cross_conference_is_interregional(self):
        self.assertTrue(is_explicit_interregional_league_label("Cross Conference"))

    def test_americas_is_interregional(self):
        self.assertTrue(is_explicit_interregional_league_label("Americas"))

    def test_interleague_is_interregional(self):
        self.assertTrue(is_explicit_interregional_league_label("INTERLEAGUE"))

    def test_lck_is_not_interregional(self):
        self.assertFalse(is_explicit_interregional_league_label("LCK"))

    def test_lpl_is_not_interregional(self):
        self.assertFalse(is_explicit_interregional_league_label("LPL"))

    def test_empty_string_returns_false(self):
        self.assertFalse(is_explicit_interregional_league_label(""))

    def test_none_returns_false(self):
        self.assertFalse(is_explicit_interregional_league_label(None))

    def test_returns_bool(self):
        self.assertIsInstance(is_explicit_interregional_league_label("LCK"), bool)


class TestLeagueMacroKey(unittest.TestCase):

    def test_lck_returns_kr(self):
        self.assertEqual(league_macro_key("LCK"), "KR")

    def test_lpl_returns_cn(self):
        self.assertEqual(league_macro_key("LPL"), "CN")

    def test_lec_returns_emea(self):
        self.assertEqual(league_macro_key("LEC"), "EMEA")

    def test_lcs_returns_na(self):
        self.assertEqual(league_macro_key("LCS"), "NA")

    def test_lta_north_alias_returns_na(self):
        self.assertEqual(league_macro_key("LTA North"), "NA")

    def test_lta_south_alias_returns_br(self):
        self.assertEqual(league_macro_key("LTA South"), "BR")

    def test_lcp_returns_apac(self):
        self.assertEqual(league_macro_key("LCP"), "APAC")

    def test_americas_returns_americas(self):
        self.assertEqual(league_macro_key("Americas"), "AMERICAS")

    def test_ldl_returns_cn(self):
        self.assertEqual(league_macro_key("LDL"), "CN")

    def test_unknown_returns_custom_default(self):
        self.assertEqual(league_macro_key("unknown_x", unknown="OTHER"), "OTHER")

    def test_empty_returns_unknown(self):
        self.assertEqual(league_macro_key("", unknown="OTHER"), "OTHER")

    def test_returns_string(self):
        self.assertIsInstance(league_macro_key("LCK"), str)


class TestLeagueLookupKeys(unittest.TestCase):

    def test_lpl_cl_first_key_is_al(self):
        self.assertEqual(league_lookup_keys("LPL CL")[0], "AL")

    def test_lpl_cl_includes_ldl(self):
        self.assertIn("LDL", league_lookup_keys("LPL CL"))

    def test_lta_north_includes_lta_north_key(self):
        self.assertIn("LTA NORTH", league_lookup_keys("LTA North"))

    def test_returns_tuple(self):
        self.assertIsInstance(league_lookup_keys("LCK"), tuple)

    def test_non_empty_for_known_league(self):
        self.assertGreater(len(league_lookup_keys("LCK")), 0)


if __name__ == "__main__":
    unittest.main()
