import unittest

from app_core.schedule_tab_core import dedup_repeat_token, parse_schedule_text


class TestDedupRepeatToken(unittest.TestCase):

    def test_repeated_token_deduplicated(self):
        self.assertEqual(dedup_repeat_token("T1T1"), "T1")

    def test_normal_token_unchanged(self):
        self.assertEqual(dedup_repeat_token("Gen.G"), "Gen.G")

    def test_strips_whitespace(self):
        self.assertEqual(dedup_repeat_token(" Gen.G "), "Gen.G")

    def test_empty_string_returns_empty(self):
        self.assertEqual(dedup_repeat_token(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(dedup_repeat_token(None), "")

    def test_odd_length_unchanged(self):
        result = dedup_repeat_token("T1T")
        self.assertIn("T", result)

    def test_two_char_repeat(self):
        self.assertEqual(dedup_repeat_token("FNKFNK"), "FNK")

    def test_returns_string(self):
        self.assertIsInstance(dedup_repeat_token("T1"), str)


class TestParseScheduleText(unittest.TestCase):

    def _text(self):
        return "\n".join([
            "06/03",
            "15:00",
            "T1",
            "GEN.G",
            "16:30",
            "G2",
            "FNC",
        ])

    def test_parses_two_games(self):
        rows = parse_schedule_text(self._text(), default_bo=5)
        self.assertEqual(len(rows), 2)

    def test_first_game_date(self):
        rows = parse_schedule_text(self._text())
        self.assertEqual(rows[0]["date"], "06/03")

    def test_first_game_time(self):
        rows = parse_schedule_text(self._text())
        self.assertEqual(rows[0]["time"], "15:00")

    def test_first_game_team_a(self):
        rows = parse_schedule_text(self._text())
        self.assertEqual(rows[0]["teamA"], "T1")

    def test_first_game_team_b(self):
        rows = parse_schedule_text(self._text())
        self.assertEqual(rows[0]["teamB"], "GEN.G")

    def test_default_bo_applied(self):
        rows = parse_schedule_text(self._text(), default_bo=5)
        self.assertEqual(rows[0]["bo"], 5)

    def test_empty_text_returns_empty(self):
        self.assertEqual(parse_schedule_text(""), [])

    def test_returns_list(self):
        self.assertIsInstance(parse_schedule_text(""), list)

    def test_game_has_required_keys(self):
        rows = parse_schedule_text(self._text())
        for key in ("date", "time", "teamA", "teamB", "bo", "league", "lines"):
            self.assertIn(key, rows[0])


if __name__ == "__main__":
    unittest.main()
