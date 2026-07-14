"""Tests for app_core/matchup_selection_core.py — pure team name helpers."""
import unittest

from app_core.matchup_selection_core import (
    canon_team_key,
    norm_team_token,
    pick_team_name,
    team_match_score,
    team_with_league_label,
)


# ---------------------------------------------------------------------------
# team_with_league_label
# ---------------------------------------------------------------------------

class TestTeamWithLeagueLabel(unittest.TestCase):

    def test_with_league(self):
        result = team_with_league_label("T1", "LCK")
        self.assertEqual(result, "T1 (LCK)")

    def test_empty_league_returns_name(self):
        self.assertEqual(team_with_league_label("T1", ""), "T1")

    def test_none_league_returns_name(self):
        self.assertEqual(team_with_league_label("T1", None), "T1")

    def test_strips_whitespace(self):
        self.assertEqual(team_with_league_label(" T1 ", "LCK"), "T1 (LCK)")

    def test_both_empty(self):
        self.assertEqual(team_with_league_label("", ""), "")


# ---------------------------------------------------------------------------
# canon_team_key
# ---------------------------------------------------------------------------

class TestCanonTeamKey(unittest.TestCase):

    def test_casefold(self):
        self.assertEqual(canon_team_key("T1"), "t1")

    def test_unicode_casefold(self):
        # casefold handles more than lower()
        self.assertEqual(canon_team_key("NaVi"), "navi")

    def test_normalizes_multiple_spaces(self):
        result = canon_team_key("Team  Liquid")
        self.assertEqual(result, "team liquid")

    def test_none_returns_empty(self):
        self.assertEqual(canon_team_key(None), "")

    def test_strips_leading_trailing(self):
        self.assertEqual(canon_team_key("  G2  "), "g2")

    def test_same_case_same_key(self):
        self.assertEqual(canon_team_key("Cloud9"), canon_team_key("CLOUD9"))


# ---------------------------------------------------------------------------
# norm_team_token
# ---------------------------------------------------------------------------

class TestNormTeamToken(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(norm_team_token("T1"), "t1")

    def test_removes_special_chars(self):
        # Non-alphanumeric → space, then collapsed
        self.assertNotIn("-", norm_team_token("Cloud-9"))

    def test_collapses_spaces(self):
        result = norm_team_token("Team   Liquid")
        self.assertNotIn("  ", result)

    def test_none_returns_empty(self):
        self.assertEqual(norm_team_token(None), "")

    def test_pure_alpha(self):
        self.assertEqual(norm_team_token("fnatic"), "fnatic")

    def test_numbers_kept(self):
        self.assertIn("9", norm_team_token("Cloud9"))


# ---------------------------------------------------------------------------
# team_match_score
# ---------------------------------------------------------------------------

class TestTeamMatchScore(unittest.TestCase):

    def test_exact_match_gives_100(self):
        score = team_match_score("T1", ["T1"])
        self.assertEqual(score, 100)

    def test_substring_match_gives_70(self):
        # "team liquid" contains "liquid"
        score = team_match_score("Team Liquid", ["liquid"])
        self.assertEqual(score, 70)

    def test_no_match_gives_0(self):
        score = team_match_score("T1", ["G2", "fnatic"])
        self.assertEqual(score, 0)

    def test_case_insensitive(self):
        self.assertEqual(team_match_score("t1", ["T1"]), 100)

    def test_best_of_multiple_patterns(self):
        # One exact, one no-match → score=100
        score = team_match_score("T1", ["T1", "XXXXXXX"])
        self.assertEqual(score, 100)

    def test_empty_patterns(self):
        self.assertEqual(team_match_score("T1", []), 0)

    def test_empty_pattern_skipped(self):
        self.assertEqual(team_match_score("T1", [""]), 0)


# ---------------------------------------------------------------------------
# pick_team_name
# ---------------------------------------------------------------------------

class TestPickTeamName(unittest.TestCase):

    def test_exact_match(self):
        result = pick_team_name(["T1", "G2", "fnatic"], ["T1"])
        self.assertEqual(result, "T1")

    def test_no_match_above_threshold_returns_none(self):
        result = pick_team_name(["G2", "fnatic"], ["Cloud9"])
        self.assertIsNone(result)

    def test_empty_candidates_returns_none(self):
        result = pick_team_name([], ["T1"])
        self.assertIsNone(result)

    def test_best_of_multiple_candidates(self):
        # "T1" exact, "T1 Challengers" substring → T1 wins with score 100
        result = pick_team_name(["T1", "T1 Challengers"], ["T1"])
        self.assertEqual(result, "T1")

    def test_substring_match_returned(self):
        result = pick_team_name(["Team Liquid"], ["liquid"])
        self.assertEqual(result, "Team Liquid")

    def test_threshold_60_enforced(self):
        # Score 70 (substring) → above threshold → should return
        result = pick_team_name(["Team Liquid"], ["liquid"])
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
