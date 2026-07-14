"""Tests for app_core/players_tab_core.py — pure trace row builder."""
import unittest

from app_core.players_tab_core import _build_trace_rows


class TestBuildTraceRows(unittest.TestCase):

    def test_empty_input_returns_empty_list(self):
        result = _build_trace_rows([])
        self.assertEqual(result, [])

    def test_returns_list(self):
        result = _build_trace_rows([{"Lane": "Top", "Team": "A"}])
        self.assertIsInstance(result, list)

    def test_length_matches_input(self):
        rows = [{"Lane": "Top"}, {"Lane": "Mid"}]
        result = _build_trace_rows(rows)
        self.assertEqual(len(result), 2)

    def test_lane_extracted(self):
        result = _build_trace_rows([{"Lane": "Jungle"}])
        self.assertEqual(result[0]["lane"], "Jungle")

    def test_team_extracted(self):
        result = _build_trace_rows([{"Team": "T1"}])
        self.assertEqual(result[0]["team"], "T1")

    def test_player_extracted(self):
        result = _build_trace_rows([{"Player": "Faker"}])
        self.assertEqual(result[0]["player"], "Faker")

    def test_missing_keys_default_to_empty_string(self):
        result = _build_trace_rows([{}])
        self.assertEqual(result[0]["lane"], "")
        self.assertEqual(result[0]["team"], "")
        self.assertEqual(result[0]["player"], "")

    def test_odd_model_raw_preserved(self):
        result = _build_trace_rows([{"_odd_model_raw": 1.85}])
        self.assertEqual(result[0]["odd_model"], 1.85)

    def test_none_odd_values_preserved(self):
        result = _build_trace_rows([{}])
        self.assertIsNone(result[0]["odd_model"])
        self.assertIsNone(result[0]["odd_hist"])

    def test_h2h_n_preserved(self):
        result = _build_trace_rows([{"_h2h_n": 5}])
        self.assertEqual(result[0]["h2h_n"], 5)

    def test_calib_reason_as_string(self):
        result = _build_trace_rows([{"_calib_reason": "no_data"}])
        self.assertEqual(result[0]["calib_reason"], "no_data")

    def test_obs_odds_as_string(self):
        result = _build_trace_rows([{"Obs Odds": "1.90"}])
        self.assertEqual(result[0]["obs_odds"], "1.90")

    def test_required_output_keys_present(self):
        result = _build_trace_rows([{"Lane": "Top"}])
        row = result[0]
        for key in ("lane", "team", "player", "mercado", "lado", "linha",
                    "odd_model", "odd_hist", "odd_h2h", "odd_mix",
                    "calib_reason", "calib_source", "h2h_n", "h2h_used",
                    "obs_odds", "obs_mix"):
            self.assertIn(key, row)

    def test_does_not_mutate_input(self):
        original = [{"Lane": "Bot", "Team": "T1"}]
        _build_trace_rows(original)
        self.assertEqual(original[0]["Lane"], "Bot")


if __name__ == "__main__":
    unittest.main()
