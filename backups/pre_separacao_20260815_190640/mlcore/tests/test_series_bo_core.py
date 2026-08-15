import unittest

from app_core.series_bo_core import apply_series_bo_dplite_live, enforce_series_economic_coherence
from mlcore.bo import p_series_from_pmap


class TestSeriesBoCore(unittest.TestCase):
    def test_economic_coherence_lifts_low_series_for_favorite(self):
        out = enforce_series_economic_coherence(
            p_map=0.58,
            p_series=0.53,
            bo=5,
        )
        self.assertTrue(bool(out.get("applied")))
        p_after = float(out.get("p_series_after"))
        self.assertGreaterEqual(p_after, 0.58)
        self.assertGreaterEqual(p_after, float(p_series_from_pmap(0.58, 3)))

    def test_economic_coherence_caps_large_deviation_from_target(self):
        out = enforce_series_economic_coherence(
            p_map=0.52,
            p_series=0.90,
            bo=3,
            max_dev_pp=8.0,
        )
        self.assertTrue(bool(out.get("applied")))
        p_after = float(out.get("p_series_after"))
        p_target = float(out.get("p_target"))
        self.assertLessEqual(abs(p_after - p_target), 0.0800001)

    def test_bo1_noop(self):
        out = apply_series_bo_dplite_live(
            p_map=0.58,
            p_series_in=0.58,
            bo=1,
            same_league=True,
        )
        self.assertFalse(bool(out.get("applied")))
        self.assertEqual(str(out.get("reason")), "bo1_noop")

    def test_applies_with_dp_anchor_bo5(self):
        out = apply_series_bo_dplite_live(
            p_map=0.58,
            p_series_in=0.64,
            bo=5,
            same_league=True,
            playoffs=True,
            side_bias_pp=2.5,
            style_volatility=0.2,
            layer_confidence_score=70.0,
            n_team_eff=50,
            n_players_eff=30,
        )
        self.assertIn(str(out.get("method")), ("series_bo_v5_live",))
        self.assertIn(str(out.get("anchor_mode")), ("dp_sideaware_v5",))
        self.assertTrue(isinstance(out.get("dp"), dict))
        self.assertIn("g1_blue_shift", (out.get("dp") or {}))
        self.assertIn("momentum_shift", (out.get("dp") or {}))
        self.assertIn("loser_pick_decay", (out.get("dp") or {}))
        self.assertIn("decider_variance", (out.get("dp") or {}))
        self.assertIn("decider_side_revert", (out.get("dp") or {}))
        self.assertIn("series_corr_sigma", (out.get("dp") or {}))
        self.assertIn("expected_maps", (out.get("dp") or {}))
        self.assertIn("decider_reach_prob", (out.get("dp") or {}))
        self.assertIn("map_play_prob", (out.get("dp") or {}))
        self.assertIn("map_win_if_played", (out.get("dp") or {}))
        self.assertIn("scoreline_probs", (out.get("dp") or {}))
        self.assertTrue(float(out.get("anchor_weight", 0.0)) >= 0.45)
        self.assertIn("coherence_guard", out)
        self.assertIsNotNone(out.get("p_after"))
        self.assertTrue(0.0 < float(out.get("p_after")) < 1.0)
        self.assertEqual(str((out.get("coherence_guard") or {}).get("reason", "")), "deferred_to_pipeline")

    def test_v4_dynamic_anchor_is_primary_driver(self):
        out = apply_series_bo_dplite_live(
            p_map=0.62,
            p_series_in=0.52,
            bo=5,
            same_league=True,
            interregional=False,
            playoffs=True,
            side_bias_pp=3.5,
            style_volatility=0.10,
            layer_confidence_score=80.0,
            n_team_eff=60,
            n_players_eff=35,
        )
        self.assertTrue(bool(out.get("applied")))
        self.assertGreater(float(out.get("anchor_weight", 0.0)), 0.60)
        self.assertGreater(float(out.get("cap_pp", 0.0)), 8.0)
        self.assertGreater(float(out.get("p_after", 0.0)), 0.56)

    def test_map_by_map_diagnostics_consistent(self):
        out = apply_series_bo_dplite_live(
            p_map=0.59,
            p_series_in=0.64,
            bo=5,
            same_league=True,
            playoffs=True,
            side_bias_pp=2.0,
            style_volatility=0.12,
            layer_confidence_score=76.0,
            n_team_eff=52,
            n_players_eff=28,
        )
        dp = dict(out.get("dp") or {})
        self.assertTrue(3.0 <= float(dp.get("expected_maps", 0.0)) <= 5.0)
        dec = float(dp.get("decider_reach_prob", 0.0))
        self.assertTrue(0.0 <= dec <= 1.0)
        mplay = dict(dp.get("map_play_prob") or {})
        self.assertAlmostEqual(float(mplay.get("m1", 0.0)), 1.0, delta=1e-9)
        self.assertTrue(float(mplay.get("m1", 0.0)) >= float(mplay.get("m2", 0.0)) >= float(mplay.get("m3", 0.0)))
        self.assertTrue(float(mplay.get("m3", 0.0)) >= float(mplay.get("m4", 0.0)) >= float(mplay.get("m5", 0.0)))
        self.assertAlmostEqual(float(mplay.get("m5", 0.0)), dec, delta=1e-9)

    def test_interregional_damps_move(self):
        base_kwargs = dict(
            p_map=0.62,
            p_series_in=0.72,
            bo=5,
            playoffs=False,
            side_bias_pp=3.0,
            style_volatility=0.10,
            layer_confidence_score=80.0,
            n_team_eff=60,
            n_players_eff=35,
        )
        same = apply_series_bo_dplite_live(same_league=True, interregional=False, **base_kwargs)
        inter = apply_series_bo_dplite_live(same_league=False, interregional=True, **base_kwargs)
        self.assertTrue(float(inter.get("weight", 0.0)) <= float(same.get("weight", 0.0)))
        self.assertTrue(float(inter.get("cap_pp", 0.0)) <= float(same.get("cap_pp", 0.0)))

    def test_side_bias_changes_anchor_state(self):
        a = apply_series_bo_dplite_live(
            p_map=0.55,
            p_series_in=0.61,
            bo=3,
            same_league=True,
            side_bias_pp=0.0,
            n_team_eff=40,
            n_players_eff=20,
        )
        b = apply_series_bo_dplite_live(
            p_map=0.55,
            p_series_in=0.61,
            bo=3,
            same_league=True,
            side_bias_pp=5.0,
            n_team_eff=40,
            n_players_eff=20,
        )
        self.assertNotEqual(
            float((a.get("dp") or {}).get("p_blue", 0.0)),
            float((b.get("dp") or {}).get("p_blue", 0.0)),
        )

    def test_decider_variance_reacts_to_confidence(self):
        low_conf = apply_series_bo_dplite_live(
            p_map=0.57,
            p_series_in=0.60,
            bo=5,
            same_league=True,
            side_bias_pp=2.0,
            style_volatility=0.15,
            layer_confidence_score=30.0,
            n_team_eff=45,
            n_players_eff=20,
        )
        high_conf = apply_series_bo_dplite_live(
            p_map=0.57,
            p_series_in=0.60,
            bo=5,
            same_league=True,
            side_bias_pp=2.0,
            style_volatility=0.15,
            layer_confidence_score=90.0,
            n_team_eff=45,
            n_players_eff=20,
        )
        dv_low = float((low_conf.get("dp") or {}).get("decider_variance", 0.0))
        dv_high = float((high_conf.get("dp") or {}).get("decider_variance", 0.0))
        self.assertGreater(dv_low, dv_high)

    def test_tuning_global_overrides_parameters(self):
        tuning_cfg = {
            "global": {
                "bo5": {
                    "pick_shift": 0.031,
                    "g1_blue_shift": 0.012,
                    "momentum_shift": 0.017,
                    "loser_pick_decay": 0.55,
                    "decider_variance": 0.19,
                    "decider_side_revert": 0.35,
                    "series_corr_sigma": 0.12,
                    "anchor_weight": 0.88,
                    "cap_pp": 14.0,
                }
            }
        }
        out = apply_series_bo_dplite_live(
            p_map=0.58,
            p_series_in=0.66,
            bo=5,
            same_league=True,
            interregional=False,
            playoffs=True,
            patch_status="ok",
            league_hint="LCK",
            tuning_cfg=tuning_cfg,
            n_team_eff=48,
            n_players_eff=22,
        )
        dp = dict(out.get("dp") or {})
        self.assertEqual(str((out.get("tuning") or {}).get("source")), "global")
        self.assertAlmostEqual(float(dp.get("pick_shift", 0.0)), 0.031, places=9)
        self.assertAlmostEqual(float(dp.get("g1_blue_shift", 0.0)), 0.012, places=9)
        self.assertAlmostEqual(float(dp.get("momentum_shift", 0.0)), 0.017, places=9)
        self.assertAlmostEqual(float(dp.get("loser_pick_decay", 0.0)), 0.55, places=9)
        self.assertAlmostEqual(float(dp.get("decider_variance", 0.0)), 0.19, places=9)
        self.assertAlmostEqual(float(dp.get("decider_side_revert", 0.0)), 0.35, places=9)
        self.assertAlmostEqual(float(dp.get("series_corr_sigma", 0.0)), 0.12, places=9)
        self.assertAlmostEqual(float(out.get("anchor_weight", 0.0)), 0.88, places=9)
        self.assertAlmostEqual(float(out.get("cap_pp", 0.0)), 14.0, places=9)

    def test_tuning_precedence_league_over_global(self):
        tuning_cfg = {
            "global": {"bo5": {"pick_shift": 0.020}},
            "by_league": {"TCL": {"bo5": {"pick_shift": 0.034}}},
        }
        out = apply_series_bo_dplite_live(
            p_map=0.56,
            p_series_in=0.62,
            bo=5,
            same_league=True,
            interregional=False,
            playoffs=False,
            patch_status="ok",
            league_hint="TCL",
            tuning_cfg=tuning_cfg,
            n_team_eff=40,
            n_players_eff=18,
        )
        self.assertEqual(str((out.get("tuning") or {}).get("source")), "league:TCL")
        self.assertAlmostEqual(float((out.get("dp") or {}).get("pick_shift", 0.0)), 0.034, places=9)

    def test_tuning_precedence_league_patch_over_league(self):
        tuning_cfg = {
            "by_league": {"TCL": {"bo5": {"decider_variance": 0.12}}},
            "by_league_patch": {"TCL|REFIT": {"bo5": {"decider_variance": 0.21}}},
        }
        out = apply_series_bo_dplite_live(
            p_map=0.56,
            p_series_in=0.62,
            bo=5,
            same_league=True,
            interregional=False,
            playoffs=False,
            patch_status="refit_needed",
            league_hint="TCL",
            tuning_cfg=tuning_cfg,
            n_team_eff=40,
            n_players_eff=18,
        )
        self.assertEqual(str((out.get("tuning") or {}).get("source")), "league_patch:TCL|REFIT")
        self.assertAlmostEqual(float((out.get("dp") or {}).get("decider_variance", 0.0)), 0.21, places=9)

    def test_dplite_defers_full_economic_coherence_to_pipeline(self):
        out = apply_series_bo_dplite_live(
            p_map=0.62,
            p_series_in=0.40,
            bo=5,
            same_league=True,
            interregional=False,
            playoffs=True,
            n_team_eff=60,
            n_players_eff=35,
        )
        guard = dict(out.get("coherence_guard") or {})
        self.assertEqual(str(guard.get("reason", "")), "deferred_to_pipeline")
        self.assertFalse(bool(guard.get("applied", False)))
        self.assertTrue(0.0 < float(out.get("p_after", 0.0)) < 1.0)


if __name__ == "__main__":
    unittest.main()
