import unittest

from app_core.results_runtime_state_core import (
    get_wf_runtime_settings,
    resolve_hc_kills_team,
    resolve_results_runtime_settings,
)


class ResultsRuntimeStateCoreTests(unittest.TestCase):
    def test_resolve_results_runtime_settings_sets_defaults(self) -> None:
        state = {}
        out = resolve_results_runtime_settings(state, season_year=2026, years=[2024, 2025])
        self.assertEqual(out["default_year"], 2025)
        self.assertEqual(out["year_opt"], 2025)
        self.assertEqual(out["totals_mode"], "map")
        self.assertEqual(out["league_mode"], "auto")
        self.assertEqual(out["n_sims"], 20000)

    def test_resolve_results_runtime_settings_handles_uniform_weight(self) -> None:
        state = {"weight_mode": "Uniforme", "half_life_days": 120, "year_opt": 2024}
        out = resolve_results_runtime_settings(state, season_year=2026, years=[2024, 2025, 2026])
        self.assertIsNone(out["half_life_days"])
        self.assertEqual(out["year_opt"], 2024)

    def test_resolve_results_runtime_settings_resets_stale_year_from_other_season(self) -> None:
        state = {"year_opt": 2026}
        out = resolve_results_runtime_settings(state, season_year=2025, years=[2025])
        self.assertEqual(out["default_year"], 2025)
        self.assertEqual(out["year_opt"], 2025)
        self.assertEqual(state["year_opt"], 2025)

    def test_resolve_hc_kills_team_normalizes_invalid_value(self) -> None:
        state = {"hc_kills_team": "X"}
        current, options = resolve_hc_kills_team(state, team_a="T1", team_b="GEN")
        self.assertEqual(current, "T1")
        self.assertEqual(options, ["T1", "GEN"])
        self.assertEqual(state["hc_kills_team"], "T1")

    def test_resolve_hc_kills_team_can_read_without_writeback(self) -> None:
        state = {"hc_kills_team": "X"}
        current, options = resolve_hc_kills_team(state, team_a="T1", team_b="GEN", write_back=False)
        self.assertEqual(current, "T1")
        self.assertEqual(options, ["T1", "GEN"])
        self.assertEqual(state["hc_kills_team"], "X")

    def test_get_wf_runtime_settings(self) -> None:
        state = {"wf_min_train_games": 1700, "wf_k": 28.0, "wf_scale": 420.0}
        out = get_wf_runtime_settings(state)
        self.assertEqual(out["min_train_games"], 1700)
        self.assertEqual(out["k"], 28.0)
        self.assertEqual(out["scale"], 420.0)


if __name__ == "__main__":
    unittest.main()
