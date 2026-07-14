import unittest

from mlcore.team_artifact_normalize import _norm_team_key, _norm_upper_key, _patch_key


class TestNormTeamKey(unittest.TestCase):
    def test_strips_whitespace(self):
        self.assertEqual(_norm_team_key("  T1  "), "t1")

    def test_removes_accents(self):
        self.assertEqual(_norm_team_key("Équipe"), "equipe")

    def test_casefold(self):
        self.assertEqual(_norm_team_key("TeamLiquid"), "teamliquid")

    def test_removes_non_alphanumeric(self):
        self.assertEqual(_norm_team_key("Team Liquid!"), "teamliquid")

    def test_empty_returns_empty(self):
        self.assertEqual(_norm_team_key(""), "")
        self.assertEqual(_norm_team_key(None), "")
        self.assertEqual(_norm_team_key(0), "")

    def test_numbers_preserved(self):
        self.assertEqual(_norm_team_key("T1"), "t1")
        self.assertEqual(_norm_team_key("100 Thieves"), "100thieves")


class TestPatchKey(unittest.TestCase):
    def test_extracts_major_minor(self):
        self.assertEqual(_patch_key("14.10"), "14.10")
        self.assertEqual(_patch_key("patch_14.10_final"), "14.10")

    def test_empty_returns_empty(self):
        self.assertEqual(_patch_key(""), "")
        self.assertEqual(_patch_key(None), "")

    def test_no_version_returns_stripped_string(self):
        self.assertEqual(_patch_key("nopatch"), "nopatch")

    def test_picks_first_match(self):
        self.assertEqual(_patch_key("14.10 vs 15.1"), "14.10")


class TestNormUpperKey(unittest.TestCase):
    def test_uppercases(self):
        self.assertEqual(_norm_upper_key("abc"), "ABC")

    def test_strips_whitespace(self):
        self.assertEqual(_norm_upper_key("  hello  "), "HELLO")

    def test_empty_and_none(self):
        self.assertEqual(_norm_upper_key(""), "")
        self.assertEqual(_norm_upper_key(None), "")

    def test_preserves_non_alpha(self):
        self.assertEqual(_norm_upper_key("win_rate"), "WIN_RATE")


if __name__ == "__main__":
    unittest.main()
