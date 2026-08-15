"""Tests for pure helpers in app_core/matchup_selection_core.py."""
import unittest

from app_core.matchup_selection_core import (
    canon_team_key,
    norm_team_token,
    pick_team_name,
    team_match_score,
    team_with_league_label,
)


class TestTeamWithLeagueLabel(unittest.TestCase):

    def test_with_league(self):
        self.assertEqual(team_with_league_label("T1", "LCK"), "T1 (LCK)")

    def test_without_league(self):
        self.assertEqual(team_with_league_label("T1", ""), "T1")

    def test_none_league(self):
        self.assertEqual(team_with_league_label("T1", None), "T1")

    def test_empty_team(self):
        self.assertEqual(team_with_league_label("", "LCK"), " (LCK)")

    def test_returns_string(self):
        self.assertIsInstance(team_with_league_label("T1", "LCK"), str)


class TestCanonTeamKey(unittest.TestCase):

    def test_casefolded(self):
        self.assertEqual(canon_team_key("T1"), "t1")

    def test_strips_whitespace(self):
        self.assertEqual(canon_team_key("  Gen.G  "), "Gen.G".casefold())

    def test_normalizes_spaces(self):
        self.assertEqual(canon_team_key("Team  Liquid"), "team liquid")

    def test_none_returns_empty(self):
        self.assertEqual(canon_team_key(None), "")

    def test_empty_returns_empty(self):
        self.assertEqual(canon_team_key(""), "")


class TestNormTeamToken(unittest.TestCase):

    def test_lowercases(self):
        self.assertEqual(norm_team_token("T1"), "t1")

    def test_removes_special_chars(self):
        self.assertEqual(norm_team_token("Gen.G"), "gen g")

    def test_strips_and_collapses_spaces(self):
        self.assertEqual(norm_team_token("  Team  Liquid  "), "team liquid")

    def test_none_returns_empty(self):
        self.assertEqual(norm_team_token(None), "")

    def test_empty_returns_empty(self):
        self.assertEqual(norm_team_token(""), "")

    def test_numbers_preserved(self):
        result = norm_team_token("100 Thieves")
        self.assertIn("100", result)

    def test_returns_string(self):
        self.assertIsInstance(norm_team_token("T1"), str)


class TestTeamMatchScore(unittest.TestCase):

    def test_exact_match_100(self):
        score = team_match_score("T1", ["T1"])
        self.assertEqual(score, 100)

    def test_substring_match_70(self):
        score = team_match_score("T1 Esports", ["T1"])
        self.assertEqual(score, 70)

    def test_no_match_0(self):
        score = team_match_score("GEN", ["T1"])
        self.assertEqual(score, 0)

    def test_empty_patterns_zero(self):
        score = team_match_score("T1", [])
        self.assertEqual(score, 0)

    def test_case_insensitive(self):
        score = team_match_score("t1", ["T1"])
        self.assertEqual(score, 100)

    def test_best_score_among_patterns(self):
        score = team_match_score("T1", ["T1", "GEN"])
        self.assertEqual(score, 100)


class TestPickTeamName(unittest.TestCase):

    def test_exact_match(self):
        result = pick_team_name(["T1", "GEN", "DRX"], ["T1"])
        self.assertEqual(result, "T1")

    def test_no_match_returns_none(self):
        result = pick_team_name(["GEN", "DRX"], ["T1"])
        self.assertIsNone(result)

    def test_empty_candidates_returns_none(self):
        result = pick_team_name([], ["T1"])
        self.assertIsNone(result)

    def test_partial_match(self):
        result = pick_team_name(["T1 Esports", "GEN"], ["T1"])
        self.assertEqual(result, "T1 Esports")

    def test_prefers_exact_over_partial(self):
        result = pick_team_name(["T1 Esports", "T1"], ["T1"])
        self.assertEqual(result, "T1")


if __name__ == "__main__":
    unittest.main()
