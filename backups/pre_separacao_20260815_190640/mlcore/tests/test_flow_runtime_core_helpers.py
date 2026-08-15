"""Tests for pure/state helpers in app_core/flow_runtime_core.py."""
import unittest

from app_core.flow_runtime_core import (
    apply_flow_line_values,
    clear_lines_loaded_key,
    clear_matchup_selection,
    consume_analysis_force_once,
    consume_pending_auto_analyze,
    consume_pending_reset_matchup,
    ensure_matchup_selection_state,
    get_flow_stage,
    get_lines_loaded_key,
    get_selected_teams,
    request_analysis_refresh,
    request_pending_reset_matchup,
    reset_flow_state,
    set_flow_stage,
    set_lines_loaded_key,
    set_matchup_key,
    set_pending_auto_analyze,
    set_selected_team,
)


class TestFlowStage(unittest.TestCase):

    def test_default_is_select(self):
        self.assertEqual(get_flow_stage({}), "select")

    def test_set_and_get(self):
        state = {}
        set_flow_stage(state, "results")
        self.assertEqual(get_flow_stage(state), "results")

    def test_set_returns_value(self):
        self.assertEqual(set_flow_stage({}, "select"), "select")


class TestMatchupSelection(unittest.TestCase):

    def test_ensure_initializes_missing(self):
        state = {}
        a, b = ensure_matchup_selection_state(state)
        self.assertEqual(a, "")
        self.assertEqual(b, "")

    def test_set_team_a(self):
        state = {}
        ensure_matchup_selection_state(state)
        set_selected_team(state, "a", "T1")
        a, _b = get_selected_teams(state)
        self.assertEqual(a, "T1")

    def test_set_team_b(self):
        state = {}
        ensure_matchup_selection_state(state)
        set_selected_team(state, "b", "GEN")
        _a, b = get_selected_teams(state)
        self.assertEqual(b, "GEN")

    def test_clear_resets_both(self):
        state = {"teamA": "T1", "teamB": "GEN"}
        clear_matchup_selection(state)
        self.assertEqual(state["teamA"], "")
        self.assertEqual(state["teamB"], "")


class TestLinesLoadedKey(unittest.TestCase):

    def test_default_empty(self):
        self.assertEqual(get_lines_loaded_key({}), "")

    def test_set_and_get(self):
        state = {}
        set_lines_loaded_key(state, "matchup::T1|GEN")
        self.assertEqual(get_lines_loaded_key(state), "matchup::T1|GEN")

    def test_clear(self):
        state = {}
        set_lines_loaded_key(state, "key")
        clear_lines_loaded_key(state)
        self.assertEqual(get_lines_loaded_key(state), "")


class TestAnalysisFlags(unittest.TestCase):

    def test_request_and_consume_refresh(self):
        state = {}
        self.assertFalse(consume_analysis_force_once(state))
        request_analysis_refresh(state)
        self.assertTrue(consume_analysis_force_once(state))
        self.assertFalse(consume_analysis_force_once(state))

    def test_request_and_consume_reset_matchup(self):
        state = {}
        self.assertFalse(consume_pending_reset_matchup(state))
        request_pending_reset_matchup(state)
        self.assertTrue(consume_pending_reset_matchup(state))
        self.assertFalse(consume_pending_reset_matchup(state))


class TestPendingAutoAnalyze(unittest.TestCase):

    def test_set_and_consume(self):
        state = {}
        set_pending_auto_analyze(state, {"team": "T1"})
        result = consume_pending_auto_analyze(state)
        self.assertEqual(result.get("team"), "T1")
        empty = consume_pending_auto_analyze(state)
        self.assertEqual(empty, {})


class TestApplyFlowLineValues(unittest.TestCase):

    def test_applies_values(self):
        state = {}
        result = apply_flow_line_values(state, {"lines_kills": "25.5"}, fields=("lines_kills",))
        self.assertEqual(result["lines_kills"], "25.5")
        self.assertEqual(state["lines_kills"], "25.5")

    def test_empty_values_clears_fields(self):
        state = {"lines_kills": "25.5"}
        apply_flow_line_values(state, {}, fields=("lines_kills",))
        self.assertEqual(state["lines_kills"], "")


class TestResetFlowState(unittest.TestCase):

    def test_resets_to_select(self):
        state = {"flow_stage": "results", "teamA": "T1"}
        reset_flow_state(state)
        self.assertEqual(state["flow_stage"], "select")

    def test_keeps_config_by_default(self):
        state = {"flow_stage": "results", "teamA": "T1"}
        reset_flow_state(state, keep_config=True)
        self.assertEqual(state["teamA"], "T1")


class TestSetMatchupKey(unittest.TestCase):

    def test_set_and_read(self):
        state = {}
        result = set_matchup_key(state, "2026::T1|GEN")
        self.assertEqual(result, "2026::T1|GEN")
        self.assertEqual(state["matchup_key"], "2026::T1|GEN")


if __name__ == "__main__":
    unittest.main()
