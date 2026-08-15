import unittest

from app_core.champions_state_core import (
    ensure_champions_filter_state,
    reset_champions_league_dependents,
    resolve_blue_side_choice,
    sync_blue_side_choice,
)


class TestEnsureChampionsFilterState(unittest.TestCase):

    def test_normalizes_all_to_empty(self):
        state = {"ch_year": "All"}
        ensure_champions_filter_state(state)
        self.assertEqual(state["ch_year"], "")

    def test_normalizes_none_string_to_empty(self):
        state = {"ch_split": "None"}
        ensure_champions_filter_state(state)
        self.assertEqual(state["ch_split"], "")

    def test_normalizes_none_value_to_empty(self):
        state = {"ch_league": None}
        ensure_champions_filter_state(state)
        self.assertEqual(state["ch_league"], "")

    def test_creates_missing_keys(self):
        state = {}
        ensure_champions_filter_state(state)
        for key in ("ch_year", "ch_league", "ch_split", "ch_playoffs", "ch_team"):
            self.assertIn(key, state)

    def test_preserves_valid_values(self):
        state = {"ch_year": "2026", "ch_league": "LCK"}
        ensure_champions_filter_state(state)
        self.assertEqual(state["ch_year"], "2026")
        self.assertEqual(state["ch_league"], "LCK")

    def test_all_keys_initialized_to_empty_on_blank_state(self):
        state = {}
        ensure_champions_filter_state(state)
        self.assertEqual(state["ch_playoffs"], "")
        self.assertEqual(state["ch_team"], "")


class TestResetChampionsLeagueDependents(unittest.TestCase):

    def test_clears_ch_split(self):
        state = {"ch_split": "Spring"}
        reset_champions_league_dependents(state)
        self.assertEqual(state["ch_split"], "")

    def test_clears_ch_team(self):
        state = {"ch_team": "T1"}
        reset_champions_league_dependents(state)
        self.assertEqual(state["ch_team"], "")

    def test_sets_missing_keys_to_empty(self):
        state = {}
        reset_champions_league_dependents(state)
        self.assertEqual(state["ch_split"], "")
        self.assertEqual(state["ch_team"], "")


class TestResolveAndSyncBlueSideChoice(unittest.TestCase):

    def test_options_contains_both_teams(self):
        options, _ = resolve_blue_side_choice({}, team_a="A", team_b="B")
        self.assertEqual(options, ["A", "B"])

    def test_returns_correct_index_for_team_b(self):
        state = {"ch_blue_team": "B"}
        _, index = resolve_blue_side_choice(state, team_a="A", team_b="B")
        self.assertEqual(index, 1)

    def test_defaults_to_team_a_when_state_empty(self):
        _, index = resolve_blue_side_choice({}, team_a="A", team_b="B")
        self.assertEqual(index, 0)

    def test_invalid_state_key_defaults_to_first(self):
        state = {"ch_blue_team": "C"}
        _, index = resolve_blue_side_choice(state, team_a="A", team_b="B")
        self.assertEqual(index, 0)

    def test_sync_updates_state_key(self):
        state = {}
        sync_blue_side_choice(state, selected_value="Time A")
        self.assertEqual(state["ch_blue_team"], "Time A")

    def test_sync_with_none_stores_empty(self):
        state = {}
        sync_blue_side_choice(state, selected_value=None)
        self.assertEqual(state["ch_blue_team"], "")


if __name__ == "__main__":
    unittest.main()
