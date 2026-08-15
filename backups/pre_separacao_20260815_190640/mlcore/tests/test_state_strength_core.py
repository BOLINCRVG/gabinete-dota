import unittest

from app_core.state_strength_core import compute_state_shadow_alite


def _mk_ml_info(*, p_cal=0.62, league_a="LPL", league_b="LPL", games_a=18, games_b=17, shrink_a=0.82, shrink_b=0.80):
    return {
        "p_cal": p_cal,
        "blue": {
            "league": league_a,
            "games_played": games_a,
            "shrink_factor": shrink_a,
            "total_strength": 300.0,
        },
        "red": {
            "league": league_b,
            "games_played": games_b,
            "shrink_factor": shrink_b,
            "total_strength": 240.0,
        },
    }


class TestStateStrengthCore(unittest.TestCase):
    def test_alite_returns_trace_and_probabilities(self):
        out = compute_state_shadow_alite(
            p_map_old=0.60,
            p_series_old=0.68,
            bo=5,
            ml_info=_mk_ml_info(p_cal=0.61),
            trace_form_info={"applied": True, "p_form": 0.70, "p_model": 0.60, "w_form": 0.35, "n_a": 12, "n_b": 12},
            trace_fusion_info={"roster_changed": False},
            trace_patch_guard_info={"status": "ok", "shrink": 0.0},
            trace_uncertainty_info={"n_eff": 20.0, "prior_games": 22.0},
            n_team_eff=14.0,
            n_players_eff=4.0,
            consistency_guard_triggered=False,
            consistency_guard_drift_pp=0.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertIn("weights", out)
        self.assertIn("uncertainty", out)
        self.assertIsNotNone(out.get("p_map_new"))
        self.assertIsNotNone(out.get("p_series_new"))
        self.assertGreater(float(out["p_map_new"]), 0.0)
        self.assertLess(float(out["p_map_new"]), 1.0)

    def test_low_evidence_damps_large_recent_signal(self):
        strong_recent = {"applied": True, "p_form": 0.88, "p_model": 0.60, "w_form": 0.80, "n_a": 3, "n_b": 3}
        base_kwargs = dict(
            p_map_old=0.60,
            p_series_old=0.68,
            bo=3,
            ml_info=_mk_ml_info(p_cal=0.60, games_a=4, games_b=4, shrink_a=0.30, shrink_b=0.30),
            trace_form_info=strong_recent,
            trace_fusion_info={"roster_changed": True},
            trace_patch_guard_info={"status": "changed", "shrink": 0.8},
            consistency_guard_triggered=True,
            consistency_guard_drift_pp=18.0,
        )
        out_low = compute_state_shadow_alite(
            **base_kwargs,
            trace_uncertainty_info={"n_eff": 2.0, "prior_games": 22.0},
            n_team_eff=2.0,
            n_players_eff=1.0,
        )
        out_high = compute_state_shadow_alite(
            **base_kwargs,
            trace_uncertainty_info={"n_eff": 25.0, "prior_games": 22.0},
            n_team_eff=15.0,
            n_players_eff=8.0,
        )
        self.assertTrue(out_low.get("applied"))
        self.assertTrue(out_high.get("applied"))
        d_low = abs(float(out_low["diff_pp_map"]))
        d_high = abs(float(out_high["diff_pp_map"]))
        self.assertLess(d_low, d_high)

    def test_interregional_damps_vs_same_region(self):
        kwargs = dict(
            p_map_old=0.63,
            p_series_old=0.73,
            bo=5,
            trace_form_info={"applied": True, "p_form": 0.40, "p_model": 0.63, "w_form": 0.55, "n_a": 10, "n_b": 10},
            trace_fusion_info={"roster_changed": False},
            trace_patch_guard_info={"status": "ok", "shrink": 0.0},
            trace_uncertainty_info={"n_eff": 18.0, "prior_games": 22.0},
            n_team_eff=12.0,
            n_players_eff=4.0,
            consistency_guard_triggered=False,
            consistency_guard_drift_pp=0.0,
        )
        out_same = compute_state_shadow_alite(
            **kwargs,
            ml_info=_mk_ml_info(p_cal=0.63, league_a="LPL", league_b="LPL"),
        )
        out_inter = compute_state_shadow_alite(
            **kwargs,
            ml_info=_mk_ml_info(p_cal=0.63, league_a="LPL", league_b="LEC"),
        )
        self.assertTrue(out_same.get("applied"))
        self.assertTrue(out_inter.get("applied"))
        self.assertGreater(abs(float(out_same["diff_pp_map"])), abs(float(out_inter["diff_pp_map"])))

    def test_interregional_relief_increases_with_stronger_evidence(self):
        base_kwargs = dict(
            p_map_old=0.63,
            p_series_old=0.73,
            bo=5,
            ml_info=_mk_ml_info(p_cal=0.63, league_a="LPL", league_b="LEC", games_a=20, games_b=19, shrink_a=0.85, shrink_b=0.84),
            trace_form_info={"applied": True, "p_form": 0.41, "p_model": 0.63, "w_form": 0.60, "n_a": 12, "n_b": 12},
            trace_fusion_info={"roster_changed": False},
            trace_patch_guard_info={"status": "ok", "shrink": 0.0},
            consistency_guard_triggered=False,
            consistency_guard_drift_pp=0.0,
        )
        out_low = compute_state_shadow_alite(
            **base_kwargs,
            trace_uncertainty_info={"n_eff": 8.0, "prior_games": 22.0},
            n_team_eff=5.0,
            n_players_eff=1.0,
        )
        out_high = compute_state_shadow_alite(
            **base_kwargs,
            trace_uncertainty_info={"n_eff": 22.0, "prior_games": 22.0},
            n_team_eff=18.0,
            n_players_eff=7.0,
        )
        w_low = out_low.get("weights") if isinstance(out_low.get("weights"), dict) else {}
        w_high = out_high.get("weights") if isinstance(out_high.get("weights"), dict) else {}
        self.assertGreater(float(w_high.get("interregional_relief", 0.0)), float(w_low.get("interregional_relief", 0.0)))
        self.assertGreater(abs(float(out_high.get("diff_pp_map", 0.0))), abs(float(out_low.get("diff_pp_map", 0.0))))


if __name__ == "__main__":
    unittest.main()
