import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app_core.schedule_state_core import (
    append_schedule_games,
    bind_schedule_store_path,
    clear_pending_schedule_import,
    clear_schedule_games,
    clear_schedule_text,
    ensure_schedule_alias_state,
    ensure_schedule_state,
    get_schedule_default_league,
    get_schedule_line_fields,
    get_schedule_line_inputs,
    get_schedule_aliases,
    get_schedule_games,
    get_schedule_text,
    get_pending_schedule_import,
    has_schedule_unknown_aliases,
    init_schedule_line_inputs,
    persist_schedule_runtime_state,
    resolve_schedule_line_hc_team,
    restore_schedule_runtime_state,
    save_schedule_games,
    save_schedule_aliases,
    save_pending_schedule_import,
    set_schedule_text,
    set_schedule_unknown_aliases,
    set_schedule_pending_load,
)


class ScheduleStateCoreTests(unittest.TestCase):
    def test_schedule_state_roundtrip(self) -> None:
        state = {}
        self.assertEqual(ensure_schedule_state(state), [])
        self.assertEqual(get_schedule_games(state), [])
        append_schedule_games(state, [{"teamA": "A", "teamB": "B"}])
        self.assertEqual(len(get_schedule_games(state)), 1)
        save_schedule_games(state, [{"teamA": "C", "teamB": "D"}])
        self.assertEqual(get_schedule_games(state)[0]["teamA"], "C")

    def test_schedule_default_and_pending_load(self) -> None:
        state = {"schedule_default_league": "LCK"}
        self.assertEqual(get_schedule_default_league(state), "LCK")
        set_schedule_pending_load(state, {"teamA": "A"})
        self.assertEqual(state["pending_load"]["teamA"], "A")
        self.assertEqual(get_schedule_text(state), "")
        self.assertEqual(set_schedule_text(state, "foo"), "foo")
        self.assertEqual(get_schedule_text(state), "foo")
        self.assertEqual(clear_schedule_text(state), "")

    def test_schedule_alias_state_roundtrip(self) -> None:
        state = {}
        self.assertEqual(ensure_schedule_alias_state(state), {})
        save_schedule_aliases(state, {"foo": "Bar"})
        self.assertEqual(get_schedule_aliases(state), {"foo": "Bar"})
        self.assertFalse(has_schedule_unknown_aliases(state))
        set_schedule_unknown_aliases(state, True)
        self.assertTrue(has_schedule_unknown_aliases(state))

    def test_pending_schedule_import_roundtrip(self) -> None:
        state = {}
        self.assertEqual(get_pending_schedule_import(state), [])
        save_pending_schedule_import(state, [{"teamA_raw": "A", "teamB_raw": "B"}])
        self.assertEqual(len(get_pending_schedule_import(state)), 1)
        clear_pending_schedule_import(state)
        self.assertEqual(get_pending_schedule_import(state), [])
        append_schedule_games(state, [{"teamA": "A", "teamB": "B"}])
        clear_schedule_games(state)
        self.assertEqual(get_schedule_games(state), [])

    def test_schedule_runtime_state_persists_and_restores(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "user_schedule_store.json"
            state = {}
            bind_schedule_store_path(state, store_path)
            set_schedule_text(state, "EMEA Masters 2026 Winter")
            save_pending_schedule_import(state, [{"teamA_raw": "A", "teamB_raw": "B"}])
            save_schedule_games(state, [{"teamA": "Team A", "teamB": "Team B", "league": "EM"}])
            persist_schedule_runtime_state(state)

            restored = {}
            payload = restore_schedule_runtime_state(restored, store_path)
            self.assertEqual(payload["plays_schedule_text"], "EMEA Masters 2026 Winter")
            self.assertEqual(len(payload["plays_pending_games_import"]), 1)
            self.assertEqual(len(payload["games_list"]), 1)
            self.assertEqual(get_schedule_text(restored), "EMEA Masters 2026 Winter")
            self.assertEqual(get_pending_schedule_import(restored)[0]["teamA_raw"], "A")
            self.assertEqual(get_schedule_games(restored)[0]["teamA"], "Team A")

    def test_schedule_line_inputs_helpers(self) -> None:
        state = {}
        prefix = "ag_lines_0_"
        init = init_schedule_line_inputs(state, prefix, {"lines_kills": "25.5", "hc_kills_team": "T1"})
        self.assertEqual(init["lines_kills"], "25.5")
        self.assertEqual(get_schedule_line_inputs(state, prefix)["lines_kills"], "25.5")
        current, options = resolve_schedule_line_hc_team(state, prefix, team_a="T1", team_b="GEN")
        self.assertEqual(current, "T1")
        self.assertEqual(options, ["T1", "GEN"])
        self.assertIn("combo_time_text", get_schedule_line_fields())

    def test_schedule_line_hc_team_can_read_without_writeback(self) -> None:
        state = {"ag_lines_0_hc_kills_team": "X"}
        current, options = resolve_schedule_line_hc_team(
            state,
            "ag_lines_0_",
            team_a="T1",
            team_b="GEN",
            write_back=False,
        )
        self.assertEqual(current, "T1")
        self.assertEqual(options, ["T1", "GEN"])
        self.assertEqual(state["ag_lines_0_hc_kills_team"], "X")


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
