import unittest
from pathlib import Path

from app_core.ml_shadow_layers_core import run_ml_shadow_layers


class TestMlShadowLayersCore(unittest.TestCase):
    def test_returns_disabled_shadow_vnext_and_respects_feature_flags(self):
        session = {}
        out = run_ml_shadow_layers(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"}},
            trace_uncertainty_info={},
            trace_n_team_eff=10.0,
            trace_n_players_eff=5.0,
            team_games_min_used=8.0,
            trace_patch_guard_info={},
            trace_hybrid_info={},
            trace_form_info={},
            trace_fusion_info={},
            guard_triggered=False,
            guard_dpp=0.0,
            teamA="A",
            teamB="B",
            session_state=session,
            team_state_artifact_path=Path("state_artifact.__missing__.json"),
            safe_logit_fn=lambda p: p,
            safe_sigmoid_fn=lambda z: z,
            clip_prob_fn=lambda p: p,
            prob_win_series_fn=lambda p, _bo: p,
            odd_from_prob_fn=lambda p: 1.0 / p,
            feature_flag_enabled_fn=lambda _k, default=False: False,
            compute_state_shadow_alite_fn=lambda **_k: {"applied": True},
            compute_state_shadow_areal_fn=lambda **_k: {"applied": True},
            load_team_state_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
        )
        self.assertFalse(bool(out.get("_ff_ml_shadow_vnext", True)))
        self.assertEqual(str((out.get("trace_shadow_ml_vnext") or {}).get("reason", "")), "disabled_by_flag")
        self.assertFalse(bool(out.get("_ff_ml_shadow_state_alite", True)))
        self.assertFalse(bool(out.get("_ff_ml_shadow_state_areal", True)))
        self.assertEqual(bool(session.get("ff_ml_shadow_vnext_trace", True)), False)

    def test_runs_alite_and_areal_shadow_callbacks(self):
        out = run_ml_shadow_layers(
            p_map_used=0.60,
            p_series_used=0.65,
            bo=3,
            ml_info={"blue": {"league": "LCK", "team": "A"}, "red": {"league": "LCK", "team": "B"}},
            trace_uncertainty_info={},
            trace_n_team_eff=10.0,
            trace_n_players_eff=5.0,
            team_games_min_used=8.0,
            trace_patch_guard_info={},
            trace_hybrid_info={},
            trace_form_info={},
            trace_fusion_info={},
            guard_triggered=False,
            guard_dpp=0.0,
            teamA="A",
            teamB="B",
            session_state={},
            team_state_artifact_path=Path("state_artifact.__missing__.json"),
            safe_logit_fn=lambda p: p,
            safe_sigmoid_fn=lambda z: z,
            clip_prob_fn=lambda p: p,
            prob_win_series_fn=lambda p, _bo: p,
            odd_from_prob_fn=lambda p: 1.0 / p,
            feature_flag_enabled_fn=lambda _k, default=False: True,
            compute_state_shadow_alite_fn=lambda **_k: {"applied": True, "p_map_new": 0.58},
            compute_state_shadow_areal_fn=lambda **_k: {"applied": True, "p_map_new": 0.57},
            load_team_state_artifact_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
        )
        self.assertTrue(bool(out.get("_ff_ml_shadow_state_alite", False)))
        self.assertTrue(bool(out.get("_ff_ml_shadow_state_areal", False)))
        self.assertTrue(bool((out.get("trace_shadow_state_alite") or {}).get("applied", False)))
        self.assertTrue(bool((out.get("trace_shadow_state_areal") or {}).get("applied", False)))
        self.assertTrue(bool((out.get("trace_shadow_state_alite") or {}).get("enabled", False)))
        self.assertTrue(bool((out.get("trace_shadow_state_areal") or {}).get("enabled", False)))


if __name__ == "__main__":
    unittest.main()
