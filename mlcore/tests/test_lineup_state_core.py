import math
import unittest

from app_core.lineup_state import (
    LINEUP_SETTINGS_DEFAULTS,
    ensure_lineup_runtime_state,
    get_lineup_loaded_key,
    get_lineup_runtime_settings,
    lineup_key,
    norm_pos,
    norm_pos_ml,
    parse_pct,
    safe_key,
    set_lineup_loaded_key,
)


class TestSafeKey(unittest.TestCase):

    def test_alphanumeric_unchanged(self):
        self.assertEqual(safe_key("abc123"), "abc123")

    def test_spaces_become_underscore(self):
        self.assertEqual(safe_key("hello world"), "hello_world")

    def test_multiple_specials_collapsed(self):
        self.assertEqual(safe_key("a!@#b"), "a_b")

    def test_empty_returns_x(self):
        self.assertEqual(safe_key(""), "x")

    def test_none_returns_x(self):
        self.assertEqual(safe_key(None), "x")


class TestNormPos(unittest.TestCase):

    def test_top(self):
        self.assertEqual(norm_pos("top"), "Top")

    def test_jungle_alias(self):
        self.assertEqual(norm_pos("jng"), "Jungle")

    def test_mid(self):
        self.assertEqual(norm_pos("mid"), "Mid")

    def test_adc(self):
        self.assertEqual(norm_pos("adc"), "ADC")

    def test_support(self):
        self.assertEqual(norm_pos("sup"), "Support")

    def test_empty_returns_empty(self):
        self.assertEqual(norm_pos(""), "")


class TestNormPosML(unittest.TestCase):

    def test_bot_vs_adc(self):
        self.assertEqual(norm_pos_ml("adc"), "Bot")
        self.assertEqual(norm_pos_ml("bot"), "Bot")

    def test_top(self):
        self.assertEqual(norm_pos_ml("top"), "Top")


class TestParsePct(unittest.TestCase):

    def test_percentage_string(self):
        self.assertAlmostEqual(parse_pct("55%"), 0.55)

    def test_decimal_string(self):
        self.assertAlmostEqual(parse_pct("0.6"), 0.6)

    def test_empty_returns_nan(self):
        self.assertTrue(math.isnan(parse_pct("")))

    def test_invalid_returns_nan(self):
        self.assertTrue(math.isnan(parse_pct("abc")))


class TestLineupKey(unittest.TestCase):

    def test_teams_sorted(self):
        key1 = lineup_key("2026 Spring", "T1", "GEN")
        key2 = lineup_key("2026 Spring", "GEN", "T1")
        self.assertEqual(key1, key2)

    def test_includes_season(self):
        key = lineup_key("2026 Spring", "T1", "GEN")
        self.assertIn("2026 Spring", key)

    def test_no_season_no_prefix(self):
        key = lineup_key("", "T1", "GEN")
        self.assertNotIn("::", key)


class TestEnsureLineupRuntimeState(unittest.TestCase):

    def test_defaults_on_empty_state(self):
        snapshot = ensure_lineup_runtime_state({})
        self.assertEqual(snapshot["lineup_lambda_points"], 90)
        self.assertAlmostEqual(snapshot["ml_players_w_lineup"], 1.15)

    def test_reads_from_state(self):
        snapshot = ensure_lineup_runtime_state({"lineup_lambda_points": 110, "lineup_min_gp": 7})
        self.assertEqual(snapshot["lineup_lambda_points"], 110)
        self.assertEqual(snapshot["lineup_min_gp"], 7)

    def test_returns_dict(self):
        self.assertIsInstance(ensure_lineup_runtime_state({}), dict)


class TestLoadedKeyRoundtrip(unittest.TestCase):

    def test_empty_on_start(self):
        self.assertEqual(get_lineup_loaded_key({}), "")

    def test_set_and_get(self):
        state = {}
        set_lineup_loaded_key(state, "lk")
        self.assertEqual(get_lineup_loaded_key(state), "lk")

    def test_returns_string_from_set(self):
        self.assertEqual(set_lineup_loaded_key({}, "abc"), "abc")
