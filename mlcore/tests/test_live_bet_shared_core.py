"""Tests for app_core/live_bet_shared_core.py — pure utility functions."""
import unittest

from app_core.live_bet_shared_core import (
    _parse_line_value,
    _parse_names,
    _normalize_compact_key,
    _build_compact_vocab,
    parse_compact_draft_tokens,
    _profile_group,
    _market_resolution,
    _sample_band,
    _confidence_label,
    _chip_html,
    _profile_tone,
)


# ---------------------------------------------------------------------------
# _parse_line_value
# ---------------------------------------------------------------------------

class TestParseLineValue(unittest.TestCase):

    def test_simple_float(self):
        self.assertAlmostEqual(_parse_line_value("25.5"), 25.5)

    def test_comma_decimal(self):
        self.assertAlmostEqual(_parse_line_value("25,5"), 25.5)

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_line_value(""))

    def test_none_returns_none(self):
        self.assertIsNone(_parse_line_value(None))

    def test_time_mm_ss(self):
        result = _parse_line_value("33:00", is_time=True)
        self.assertAlmostEqual(result, 33.0, places=3)

    def test_time_30_30(self):
        result = _parse_line_value("30:30", is_time=True)
        self.assertAlmostEqual(result, 30.5, places=3)

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_line_value("abc"))

    def test_returns_float_type(self):
        result = _parse_line_value("10.5")
        self.assertIsInstance(result, float)


# ---------------------------------------------------------------------------
# _parse_names
# ---------------------------------------------------------------------------

class TestParseNames(unittest.TestCase):

    def test_newline_separated(self):
        result = _parse_names("Jinx\nAhri\nLux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_comma_separated(self):
        result = _parse_names("Jinx, Ahri, Lux")
        self.assertEqual(result, ["Jinx", "Ahri", "Lux"])

    def test_empty_returns_empty(self):
        self.assertEqual(_parse_names(""), [])

    def test_strips_whitespace(self):
        result = _parse_names("  Jinx  ,  Ahri  ")
        self.assertEqual(result, ["Jinx", "Ahri"])

    def test_semicolon_separator(self):
        result = _parse_names("Jinx;Ahri")
        self.assertEqual(result, ["Jinx", "Ahri"])


# ---------------------------------------------------------------------------
# _normalize_compact_key
# ---------------------------------------------------------------------------

class TestNormalizeCompactKey(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(_normalize_compact_key("Jinx"), "jinx")

    def test_spaces_removed(self):
        self.assertEqual(_normalize_compact_key("Twisted Fate"), "twistedfate")

    def test_special_chars_removed(self):
        self.assertEqual(_normalize_compact_key("Kai'Sa"), "kaisa")

    def test_empty_returns_empty(self):
        self.assertEqual(_normalize_compact_key(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(_normalize_compact_key(None), "")


# ---------------------------------------------------------------------------
# _build_compact_vocab
# ---------------------------------------------------------------------------

class TestBuildCompactVocab(unittest.TestCase):

    def test_basic_vocab(self):
        mapping, _max_parts = _build_compact_vocab(["Jinx", "Ahri", "Twisted Fate"])
        self.assertIn("jinx", mapping)
        self.assertIn("ahri", mapping)
        self.assertIn("twistedfate", mapping)

    def test_max_parts_two_word(self):
        _, max_parts = _build_compact_vocab(["Twisted Fate"])
        self.assertEqual(max_parts, 2)

    def test_empty_vocab(self):
        mapping, max_parts = _build_compact_vocab([])
        self.assertEqual(mapping, {})
        self.assertEqual(max_parts, 1)


# ---------------------------------------------------------------------------
# parse_compact_draft_tokens
# ---------------------------------------------------------------------------

class TestParseCompactDraftTokens(unittest.TestCase):

    def test_matches_champions(self):
        vocab = ["Jinx", "Ahri", "Lux"]
        result = parse_compact_draft_tokens("Jinx Ahri", vocab)
        self.assertEqual(result, ["Jinx", "Ahri"])

    def test_case_insensitive_match(self):
        vocab = ["Twisted Fate"]
        result = parse_compact_draft_tokens("twistedfate", vocab)
        self.assertEqual(result, ["Twisted Fate"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(parse_compact_draft_tokens("", ["Jinx"]), [])

    def test_max_picks_respected(self):
        vocab = ["Jinx", "Ahri", "Lux", "Thresh", "Blitzcrank"]
        result = parse_compact_draft_tokens("Jinx Ahri Lux Thresh Blitzcrank", vocab, max_picks=3)
        self.assertEqual(len(result), 3)

    def test_unknown_token_kept_as_is(self):
        result = parse_compact_draft_tokens("UnknownChamp", [])
        self.assertEqual(result, ["UnknownChamp"])


# ---------------------------------------------------------------------------
# _profile_group
# ---------------------------------------------------------------------------

class TestProfileGroup(unittest.TestCase):

    def test_over(self):
        self.assertEqual(_profile_group("Over"), "over")

    def test_muito_over(self):
        self.assertEqual(_profile_group("Muito Over"), "over")

    def test_under(self):
        self.assertEqual(_profile_group("Under"), "under")

    def test_muito_under(self):
        self.assertEqual(_profile_group("Muito Under"), "under")

    def test_neutro(self):
        self.assertEqual(_profile_group("Neutro"), "neutral")

    def test_empty_neutral(self):
        self.assertEqual(_profile_group(""), "neutral")


# ---------------------------------------------------------------------------
# _market_resolution
# ---------------------------------------------------------------------------

class TestMarketResolution(unittest.TestCase):

    def test_both_over(self):
        self.assertEqual(_market_resolution("Over", "Over"), "Over")

    def test_both_under(self):
        self.assertEqual(_market_resolution("Under", "Under"), "Under")

    def test_both_neutral(self):
        self.assertEqual(_market_resolution("Neutro", "Neutro"), "Neutro")

    def test_over_neutral(self):
        self.assertEqual(_market_resolution("Over", "Neutro"), "Vies Over")

    def test_under_neutral(self):
        self.assertEqual(_market_resolution("Neutro", "Under"), "Vies Under")

    def test_over_under_clash(self):
        self.assertEqual(_market_resolution("Over", "Under"), "Choque")


# ---------------------------------------------------------------------------
# _sample_band
# ---------------------------------------------------------------------------

class TestSampleBand(unittest.TestCase):

    def test_zero_games(self):
        self.assertEqual(_sample_band(0), "sem base")

    def test_one_game(self):
        self.assertEqual(_sample_band(1), "muito curta")

    def test_three_games(self):
        self.assertEqual(_sample_band(3), "curta")

    def test_seven_games(self):
        self.assertEqual(_sample_band(7), "media")

    def test_ten_games(self):
        self.assertEqual(_sample_band(10), "forte")


# ---------------------------------------------------------------------------
# _confidence_label
# ---------------------------------------------------------------------------

class TestConfidenceLabel(unittest.TestCase):

    def test_no_games(self):
        self.assertEqual(_confidence_label(2.0, 0), "Sem base")

    def test_high_confidence(self):
        result = _confidence_label(1.5, 12)
        self.assertEqual(result, "Alta")

    def test_low_confidence_few_games(self):
        result = _confidence_label(0.1, 3)
        self.assertEqual(result, "Baixa")

    def test_medium_confidence(self):
        result = _confidence_label(0.5, 7)
        self.assertEqual(result, "Media")


# ---------------------------------------------------------------------------
# _chip_html
# ---------------------------------------------------------------------------

class TestChipHtml(unittest.TestCase):

    def test_returns_string(self):
        result = _chip_html("Over", "over")
        self.assertIsInstance(result, str)

    def test_contains_text(self):
        result = _chip_html("Over 25.5", "over")
        self.assertIn("Over 25.5", result)

    def test_html_escaped(self):
        result = _chip_html("<script>", "neutral")
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;", result)

    def test_unknown_tone_uses_neutral(self):
        result = _chip_html("Test", "nonexistent_tone")
        self.assertIn("Test", result)


# ---------------------------------------------------------------------------
# _profile_tone
# ---------------------------------------------------------------------------

class TestProfileTone(unittest.TestCase):

    def test_over_tone(self):
        self.assertEqual(_profile_tone("Over moderado"), "over")

    def test_under_tone(self):
        self.assertEqual(_profile_tone("Under forte"), "under")

    def test_neutral_tone(self):
        self.assertEqual(_profile_tone("Neutro"), "neutral")

    def test_empty_neutral(self):
        self.assertEqual(_profile_tone(""), "neutral")


if __name__ == "__main__":
    unittest.main()
