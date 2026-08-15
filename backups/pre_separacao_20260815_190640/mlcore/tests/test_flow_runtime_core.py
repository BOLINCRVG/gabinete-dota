import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_core.flow_runtime_core import (
    apply_flow_line_values,
    apply_pending_load,
    build_training_csv_list,
    clear_lines_loaded_key,
    clear_matchup_selection,
    consume_analysis_force_once,
    consume_consistency_refresh_pending,
    consume_pending_auto_analyze,
    consume_pending_reset_matchup,
    ensure_matchup_selection_state,
    get_flow_stage,
    get_lines_loaded_key,
    get_last_season_label,
    get_selected_teams,
    request_analysis_refresh,
    request_pending_reset_matchup,
    reset_flow_state,
    resolve_players_artifact_path,
    set_flow_stage,
    set_last_season_label,
    set_lines_loaded_key,
    set_matchup_key,
    set_pending_auto_analyze,
    set_pending_load,
    set_selected_team,
)


class TestFlowRuntimeCore(unittest.TestCase):
    def test_reset_flow_state_keeps_or_clears_config(self):
        ss = {
            "lines_kills": "25.5",
            "year_opt": "2026",
            "cfg_auto_paths": True,
            "cfg_csv_override": "daily.csv",
            "cfg_artifact_override": "ml_artifact.json",
            "_p_map_used": 0.61,
            "_p_series_used": 0.68,
            "_mlcore_cache_key": ("x",),
            "_ml_trace_last": {"p_map_used": 0.61},
            "_ml_consistency_prev": {"sig": 1},
        }
        reset_flow_state(ss, keep_config=True)
        self.assertEqual(ss["flow_stage"], "select")
        self.assertNotIn("lines_kills", ss)
        self.assertNotIn("year_opt", ss)
        self.assertIn("cfg_csv_override", ss)
        self.assertNotIn("_p_map_used", ss)
        self.assertNotIn("_p_series_used", ss)
        self.assertNotIn("_mlcore_cache_key", ss)
        self.assertNotIn("_ml_trace_last", ss)
        self.assertNotIn("_ml_consistency_prev", ss)

        reset_flow_state(ss, keep_config=False)
        self.assertNotIn("cfg_auto_paths", ss)
        self.assertNotIn("cfg_csv_override", ss)
        self.assertNotIn("cfg_artifact_override", ss)

    def test_apply_pending_load(self):
        calls = {"reset": 0}

        def _reset_flow(*, keep_config: bool = True):
            calls["reset"] += 1

        ss = {"pending_load": {"_reset_flow": True, "teamA": "A", "teamB": "B", "_ignore": 1}}
        changed = apply_pending_load(ss, reset_flow_fn=_reset_flow)
        self.assertTrue(changed)
        self.assertEqual(calls["reset"], 1)
        self.assertEqual(ss["teamA"], "A")
        self.assertEqual(ss["teamB"], "B")
        self.assertNotIn("pending_load", ss)
        self.assertNotIn("_ignore", ss)

    def test_apply_pending_load_applies_flow_line_values(self):
        ss = {
            "pending_load": {
                "_flow_line_values": {
                    "lines_kills": "32.5",
                    "combo_time_text": "32:30",
                },
                "teamA": "A",
            }
        }
        changed = apply_pending_load(ss)
        self.assertTrue(changed)
        self.assertEqual(ss["teamA"], "A")
        self.assertEqual(ss["lines_kills"], "32.5")
        self.assertEqual(ss["combo_time_text"], "32:30")

    def test_build_training_csv_list(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            h1 = root / "2014.csv"
            h2 = root / "2015.csv"
            daily = root / "daily.csv"
            for p in [h1, h2, daily]:
                p.write_text("", encoding="utf-8")
            out = build_training_csv_list(str(daily), [str(h1), "", str(h2), str(h1), "missing.csv"])
            self.assertEqual(len(out), 3)
            self.assertEqual(Path(out[-1]).name, "daily.csv")

    def test_resolve_players_artifact_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            art = root / "ml_artifact.json"
            art.write_text("{}", encoding="utf-8")
            meta_players = root / "artifacts" / "players_meta.json"
            meta_players.parent.mkdir(parents=True, exist_ok=True)
            meta_players.write_text("{}", encoding="utf-8")

            def _load(_path: str):
                return SimpleNamespace(meta={"players_layer": {"players_artifact": "artifacts/players_meta.json"}})

            out = resolve_players_artifact_path(str(art), root, load_artifact_fn=_load)
            self.assertEqual(Path(out).name, "players_meta.json")

            root_fallback = root / "players_artifact.json"
            root_fallback.write_text("{}", encoding="utf-8")

            def _load_empty(_path: str):
                return SimpleNamespace(meta={})

            out2 = resolve_players_artifact_path(str(art), root, load_artifact_fn=_load_empty)
            self.assertEqual(Path(out2).name, "players_artifact.json")

            root_fallback.unlink()
            sibling_fallback = art.parent / "players_artifact.json"
            sibling_fallback.write_text("{}", encoding="utf-8")
            out3 = resolve_players_artifact_path(str(art), root, load_artifact_fn=_load_empty)
            self.assertEqual(Path(out3), sibling_fallback)

    def test_matchup_state_helpers(self):
        ss = {}
        self.assertEqual(ensure_matchup_selection_state(ss), ("", ""))
        self.assertEqual(get_selected_teams(ss), ("", ""))
        set_selected_team(ss, "a", "T1")
        set_selected_team(ss, "b", "Gen.G")
        self.assertEqual(get_selected_teams(ss), ("T1", "Gen.G"))
        clear_matchup_selection(ss)
        self.assertEqual(get_selected_teams(ss), ("", ""))
        self.assertEqual(get_flow_stage(ss), "select")
        set_flow_stage(ss, "lines")
        self.assertEqual(get_flow_stage(ss), "lines")
        request_pending_reset_matchup(ss)
        self.assertTrue(consume_pending_reset_matchup(ss))
        self.assertFalse(consume_pending_reset_matchup(ss))
        set_lines_loaded_key(ss, "abc")
        self.assertEqual(get_lines_loaded_key(ss), "abc")
        clear_lines_loaded_key(ss)
        self.assertEqual(get_lines_loaded_key(ss), "")
        self.assertEqual(set_matchup_key(ss, "match-1"), "match-1")
        self.assertEqual(ss["matchup_key"], "match-1")

    def test_flow_transient_state_helpers(self):
        ss = {}
        self.assertEqual(get_last_season_label(ss), "")
        self.assertEqual(set_last_season_label(ss, "2026"), "2026")
        self.assertEqual(get_last_season_label(ss), "2026")
        self.assertEqual(set_pending_load(ss, {"teamA": "T1"})["teamA"], "T1")
        self.assertEqual(ss["pending_load"]["teamA"], "T1")
        self.assertEqual(set_pending_auto_analyze(ss, {"lines": {"lines_kills": "25.5"}})["lines"]["lines_kills"], "25.5")
        self.assertEqual(consume_pending_auto_analyze(ss)["lines"]["lines_kills"], "25.5")
        self.assertEqual(consume_pending_auto_analyze(ss), {})
        request_analysis_refresh(ss)
        self.assertTrue(consume_analysis_force_once(ss))
        self.assertFalse(consume_analysis_force_once(ss))
        self.assertTrue(consume_consistency_refresh_pending(ss))
        self.assertFalse(consume_consistency_refresh_pending(ss))
        applied = apply_flow_line_values(ss, {"lines_kills": "26.5", "combo_time_text": "32.5"})
        self.assertEqual(applied["lines_kills"], "26.5")
        self.assertEqual(ss["combo_time_text"], "32.5")


if __name__ == "__main__":
    unittest.main()
