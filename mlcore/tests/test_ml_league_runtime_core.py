import unittest

from app_core.ml_league_runtime_core import (
    canon_league_display_label,
    is_marginal_league_for_ml,
    league_is_excluded,
    ml_league_conf_band,
    ml_marginal_band_profile,
    norm_league_key,
    parse_league_csv,
    sanitize_ml_marginal_band_coeffs,
    tier_from_league,
)


class TestNormLeagueKey(unittest.TestCase):

    def test_ccws_alias(self):
        self.assertEqual(norm_league_key("CCWS"), "PRM")

    def test_cross_conference_alias(self):
        self.assertEqual(norm_league_key("CrossConference"), "AMERICAS")

    def test_lck_challenger_alias(self):
        self.assertEqual(norm_league_key("LCK Challenger"), "LCKC")

    def test_lpl_academy_alias(self):
        self.assertEqual(norm_league_key("LPL Academy"), "AL")

    def test_empty_returns_empty(self):
        self.assertEqual(norm_league_key(""), "")

    def test_returns_string(self):
        self.assertIsInstance(norm_league_key("LCK"), str)


class TestCanonLeagueDisplayLabel(unittest.TestCase):

    def test_ccws_becomes_prm(self):
        self.assertEqual(canon_league_display_label("CCWS"), "PRM")

    def test_lta_north_becomes_lcs(self):
        self.assertEqual(canon_league_display_label("LTA North"), "LCS")

    def test_lck_cl_unchanged(self):
        self.assertEqual(canon_league_display_label("LCK CL"), "LCK CL")

    def test_empty_returns_empty(self):
        self.assertEqual(canon_league_display_label(""), "")

    def test_returns_string(self):
        self.assertIsInstance(canon_league_display_label("LCK"), str)


class TestParseLeagueCsvAndIsExcluded(unittest.TestCase):

    def test_parse_multiple_leagues(self):
        parsed = parse_league_csv("LCK, CrossConference | CCWS")
        self.assertEqual(parsed, {"LCK", "AMERICAS", "PRM"})

    def test_is_excluded_true(self):
        self.assertTrue(league_is_excluded("Cross Conference", "AMERICAS, LCK"))
        self.assertTrue(league_is_excluded("CCWS", "PRM"))

    def test_is_excluded_false(self):
        self.assertFalse(league_is_excluded("LPL", "LCK,PRM"))

    def test_empty_csv_nothing_excluded(self):
        self.assertFalse(league_is_excluded("LCK", ""))

    def test_returns_set(self):
        self.assertIsInstance(parse_league_csv("LCK"), set)


class TestMlLeagueConfBandAndTier(unittest.TestCase):

    def test_lck_is_top(self):
        self.assertEqual(ml_league_conf_band("LCK", ""), "top")

    def test_cblol_is_core(self):
        self.assertEqual(ml_league_conf_band("CBLOL", ""), "core")

    def test_minor_tier_is_edge(self):
        self.assertEqual(ml_league_conf_band("Some Minor", "MINOR"), "edge")

    def test_tier_from_lck(self):
        self.assertEqual(tier_from_league("LCK"), "T1")

    def test_tier_from_lck_challenger(self):
        self.assertEqual(tier_from_league("LCK Challenger"), "T2")

    def test_tier_from_unknown(self):
        self.assertEqual(tier_from_league("Unknown League"), "MINOR")


class TestIsMarginalLeague(unittest.TestCase):

    def test_lck_not_marginal(self):
        self.assertFalse(is_marginal_league_for_ml("LCK"))

    def test_lpl_not_marginal(self):
        self.assertFalse(is_marginal_league_for_ml("LPL"))

    def test_unknown_is_marginal(self):
        self.assertTrue(is_marginal_league_for_ml("SomeObscureLeague"))

    def test_empty_is_marginal(self):
        self.assertTrue(is_marginal_league_for_ml(""))


class TestMarginalBandProfile(unittest.TestCase):

    def test_custom_strength_scale(self):
        raw = {"top": {"strength_scale": "0.5", "gate_score": 77, "ignored": 1}}
        prof = ml_marginal_band_profile("top", band_coeffs=raw)
        self.assertEqual(float(prof["strength_scale"]), 0.5)

    def test_ignored_keys_removed(self):
        raw = {"top": {"strength_scale": "0.5", "ignored": 1}}
        sanitized = sanitize_ml_marginal_band_coeffs(raw)
        self.assertNotIn("ignored", sanitized["top"])

    def test_unknown_band_returns_defaults(self):
        self.assertEqual(float(ml_marginal_band_profile("unknown")["gate_n_eff"]), 55.0)

    def test_gate_score_applied(self):
        raw = {"top": {"gate_score": 77}}
        prof = ml_marginal_band_profile("top", band_coeffs=raw)
        self.assertEqual(float(prof["gate_score"]), 77.0)


if __name__ == "__main__":
    unittest.main()
