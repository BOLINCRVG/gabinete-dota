import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile

from app_core.plays_bootstrap_ui_core import (
    apply_season_change_reset,
    get_boot_hist_csvs,
    normalize_matchup_selection_teams,
    resolve_default_achievement_path,
)


class TestGetBootHistCsvs(unittest.TestCase):

    def test_filters_blanks(self):
        state = {"boot_hist_csvs": ["a.csv", "", None, "b.csv"]}
        self.assertEqual(get_boot_hist_csvs(state), ["a.csv", "b.csv"])

    def test_empty_list_returns_empty(self):
        self.assertEqual(get_boot_hist_csvs({"boot_hist_csvs": []}), [])

    def test_missing_key_returns_empty(self):
        self.assertEqual(get_boot_hist_csvs({}), [])

    def test_valid_paths_all_kept(self):
        state = {"boot_hist_csvs": ["a.csv", "b.csv", "c.csv"]}
        self.assertEqual(get_boot_hist_csvs(state), ["a.csv", "b.csv", "c.csv"])

    def test_returns_list(self):
        self.assertIsInstance(get_boot_hist_csvs({}), list)


class TestResolveDefaultAchievementPath(unittest.TestCase):

    def test_prefers_artifact_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "ml_artifact.json"
            artifact_path.write_text("{}", encoding="utf-8")
            path = resolve_default_achievement_path(
                artifact_path=str(artifact_path),
                app_root=Path(tmp),
                load_artifact_fn=lambda _sig: SimpleNamespace(meta={"achievements": {"path": "meta/achievements.json"}}),
                artifact_sig_fn=lambda path: path,
                resolve_relpath_fn=lambda base, rel: f"{base}|{rel}",
            )
        self.assertEqual(path, f"{artifact_path}|meta/achievements.json")

    def test_fallback_when_no_meta_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "ml_artifact.json"
            artifact_path.write_text("{}", encoding="utf-8")
            path = resolve_default_achievement_path(
                artifact_path=str(artifact_path),
                app_root=Path(tmp),
                load_artifact_fn=lambda _sig: SimpleNamespace(meta={}),
                artifact_sig_fn=lambda path: path,
                resolve_relpath_fn=lambda base, rel: f"{base}|{rel}",
            )
        self.assertIn("achievements.json", path)

    def test_returns_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "ml_artifact.json"
            artifact_path.write_text("{}", encoding="utf-8")
            path = resolve_default_achievement_path(
                artifact_path=str(artifact_path),
                app_root=Path(tmp),
                load_artifact_fn=lambda _sig: SimpleNamespace(meta={}),
                artifact_sig_fn=lambda path: path,
                resolve_relpath_fn=lambda base, rel: f"{base}|{rel}",
            )
        self.assertIsInstance(path, str)


class TestApplySeasonChangeReset(unittest.TestCase):

    def test_no_change_when_same_season(self):
        state = {}
        result = apply_season_change_reset(
            session_state=state,
            season_label="2026 Spring",
            get_last_season_label_fn=lambda _s: "2026 Spring",
            set_last_season_label_fn=lambda _s, _v: None,
            reset_flow_fn=lambda **_kw: None,
            clear_matchup_selection_fn=lambda _s: None,
        )
        self.assertFalse(result)

    def test_resets_when_season_changes(self):
        state = {}
        reset_called = {"v": False}
        clear_called = {"v": False}

        result = apply_season_change_reset(
            session_state=state,
            season_label="2026 Summer",
            get_last_season_label_fn=lambda _s: "2026 Spring",
            set_last_season_label_fn=lambda _s, _v: None,
            reset_flow_fn=lambda **_kw: reset_called.__setitem__("v", True),
            clear_matchup_selection_fn=lambda _s: clear_called.__setitem__("v", True),
        )
        self.assertTrue(result)
        self.assertTrue(reset_called["v"])
        self.assertTrue(clear_called["v"])


class TestNormalizeMatchupSelectionTeams(unittest.TestCase):

    def test_normalizes_team_names(self):
        state = {"teamA": "Dplus Kia", "teamB": "T1"}
        set_calls = []

        normalize_matchup_selection_teams(
            session_state=state,
            csv_sig="sig",
            team_canon_maps_fn=lambda _sig: {"variant_to_best_key": {"DPLUSKIA": "Dplus KIA", "T1": "T1"}},
            canon_team_key_fn=lambda text: text.replace(" ", "").upper(),
            ensure_matchup_selection_state_fn=lambda _state: ("Dplus Kia", "T1"),
            set_selected_team_fn=lambda _state, side, value: set_calls.append((side, value)),
        )
        self.assertEqual(set_calls, [("a", "Dplus KIA"), ("b", "T1")])


if __name__ == "__main__":
    unittest.main()
