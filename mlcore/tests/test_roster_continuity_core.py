import unittest

from app_core.roster_continuity_core import (
    apply_roster_continuity_live,
    build_roster_live_inputs,
)


class TestRosterContinuityCore(unittest.TestCase):
    def test_build_inputs_prefers_fusion_signals(self):
        out = build_roster_live_inputs(
            fusion_info={
                "lineup_continuity": 0.62,
                "freshness_avg": 0.71,
                "coverage_eff": 0.44,
                "transfer_signal": 0.28,
                "roster_changed": True,
                "season_phase": "auto",
            },
            shadow_state_areal={
                "state": {
                    "blue_roster_cont_avg": 0.95,
                    "red_roster_cont_avg": 0.96,
                    "blue_roster_cont_n": 20,
                    "red_roster_cont_n": 20,
                    "stale_days_max": 3.0,
                }
            },
            n_team_eff=18.0,
            n_players_eff=7.0,
        )
        self.assertEqual(str(out.get("signal_source")), "fusion")
        self.assertAlmostEqual(float(out.get("lineup_continuity")), 0.62, places=6)
        self.assertAlmostEqual(float(out.get("freshness_avg")), 0.71, places=6)
        self.assertAlmostEqual(float(out.get("coverage_eff")), 0.44, places=6)
        self.assertAlmostEqual(float(out.get("transfer_signal")), 0.28, places=6)
        self.assertTrue(bool(out.get("roster_changed", False)))
        self.assertEqual(str(out.get("roster_changed_source")), "fusion")

    def test_build_inputs_falls_back_to_areal(self):
        out = build_roster_live_inputs(
            fusion_info={},
            shadow_state_areal={
                "state": {
                    "blue_roster_cont_avg": 0.80,
                    "red_roster_cont_avg": 0.60,
                    "blue_roster_cont_n": 10,
                    "red_roster_cont_n": 14,
                    "stale_days_max": 5.0,
                }
            },
            n_team_eff=20.0,
            n_players_eff=0.0,
            roster_gate=0.75,
        )
        self.assertEqual(str(out.get("signal_source")), "areal_fallback")
        self.assertTrue(bool(out.get("fallback_used", False)))
        self.assertAlmostEqual(float(out.get("lineup_continuity")), 0.70, places=6)
        self.assertGreater(float(out.get("coverage_eff")), 0.40)
        self.assertLessEqual(float(out.get("freshness_avg")), 1.0)
        self.assertGreaterEqual(float(out.get("freshness_avg")), 0.25)
        self.assertAlmostEqual(float(out.get("transfer_signal")), 0.30, places=6)
        self.assertTrue(bool(out.get("roster_changed", False)))
        self.assertEqual(str(out.get("roster_changed_source")), "lineup_vs_gate")

    def test_apply_with_built_inputs_avoids_missing_signal_reason(self):
        sig = build_roster_live_inputs(
            fusion_info={},
            shadow_state_areal={
                "state": {
                    "blue_roster_cont_avg": 0.92,
                    "red_roster_cont_avg": 0.91,
                    "blue_roster_cont_n": 20,
                    "red_roster_cont_n": 20,
                    "stale_days_max": 2.0,
                }
            },
            n_team_eff=22.0,
            n_players_eff=0.0,
            roster_gate=0.55,
        )
        out = apply_roster_continuity_live(
            p_map_in=0.57,
            p_team=0.60,
            p_players=None,
            lineup_continuity=sig.get("lineup_continuity"),
            freshness_avg=sig.get("freshness_avg"),
            coverage_eff=sig.get("coverage_eff"),
            roster_changed=bool(sig.get("roster_changed", False)),
            season_phase=str(sig.get("season_phase", "") or ""),
            bo=3,
            n_team_eff=22.0,
            n_players_eff=0.0,
            transfer_signal=sig.get("transfer_signal"),
        )
        self.assertNotEqual(str(out.get("reason")), "missing_roster_signals")


if __name__ == "__main__":
    unittest.main()
