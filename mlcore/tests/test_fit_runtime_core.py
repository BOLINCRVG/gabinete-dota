import unittest

from app_core.fit_runtime_core import (
    apply_fused_post_calibration,
    build_fit_runtime_state,
    fit_form_for_league,
    fit_fused_post_calibration_for_league,
    fit_players_context_for_league,
    fit_players_lane_for_league,
    fit_players_league_weight_for_league,
    fit_scales_for_league,
)


def _macro_key(league: str) -> str:
    return {"LCK": "major", "LPL": "major", "CBLOL": "amer"}.get(str(league or "").strip(), "")


def _sanitize_mtbc(raw):
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            continue
        out[str(key)] = {
            "a": float(row.get("a", 1.0)),
            "b": float(row.get("b", 0.0)),
        }
    return out


class FitRuntimeCoreTest(unittest.TestCase):
    def test_build_fit_runtime_state_hydrates_defaults_and_promoted_band(self):
        cfg = {
            "best": {"team_scale": 0.9, "players_scale": 1.1},
            "best_form": {"settings": {"winrate": 0.6}},
            "players_lane_weights": {"global": {"Top": 1.2}},
            "players_league_weights": {"global": 0.95},
            "players_context_fit": {"gate": {"decision": "promote"}},
            "fused_post_calibration": {"gate": {"decision": "promote"}},
            "series_gamma_global": {"bo3": 1.05},
            "series_gamma_by_league": {"LCK": {"bo3": 1.08}},
            "series_gamma_by_macro": {"major": {"bo3": 1.02}},
            "series_bo_dplite_tuning": {"enabled": True},
            "ml_marginal_trust_band_coeffs": {"base": {"a": 1.3, "b": 0.1}},
        }
        state = build_fit_runtime_state(
            cfg,
            sanitize_marginal_band_coeffs=_sanitize_mtbc,
            current_marginal_band_coeffs={},
        )
        self.assertAlmostEqual(state["_fit_team_scale"], 0.9)
        self.assertAlmostEqual(state["_fit_players_scale"], 1.1)
        self.assertEqual(state["_fit_form_cfg"]["winrate"], 0.6)
        self.assertIn("ml_marginal_trust_band_coeffs", state)
        self.assertEqual(state["ml_marginal_trust_band_coeffs"]["base"]["a"], 1.3)

    def test_fit_scales_and_form_choose_league_then_macro(self):
        cfg = {
            "best_by_league": {"LCK": {"team_scale": 0.8, "players_scale": 1.2}},
            "best_form_by_macro": {"major": {"settings": {"tempo": 0.55}}},
        }
        team_scale, players_scale, src = fit_scales_for_league(cfg, 1.0, 1.0, "LCK")
        self.assertEqual((team_scale, players_scale, src), (0.8, 1.2, "LCK"))

        form_cfg, form_src = fit_form_for_league(
            cfg,
            {"tempo": 0.5},
            "LPL",
            league_macro_key_fn=_macro_key,
        )
        self.assertEqual(form_cfg["tempo"], 0.55)
        self.assertEqual(form_src, "macro:major")

    def test_fit_players_helpers_select_league_macro_and_global(self):
        lane_cfg = {
            "global": {"Top": 0.9},
            "by_macro": {"major": {"Top": 1.1, "Mid": 1.3}},
        }
        lane_row, lane_src = fit_players_lane_for_league(
            lane_cfg,
            "LCK",
            league_macro_key_fn=_macro_key,
        )
        self.assertEqual(lane_src, "macro:major")
        self.assertEqual(lane_row["Top"], 1.1)
        self.assertEqual(lane_row["Support"], 1.0)

        lg_weight, lg_src = fit_players_league_weight_for_league(
            {"global": 0.9, "by_league": {"LCK": 1.15}},
            "LCK",
            league_macro_key_fn=_macro_key,
        )
        self.assertEqual((lg_weight, lg_src), (1.15, "league:LCK"))

    def test_fit_players_context_respects_gate_and_fallback(self):
        cfg = {
            "gate": {"decision": "promote"},
            "promoted_settings_by_macro": {
                "major": {"ml_players_context_enabled": True, "ml_players_context_strength": 0.42}
            },
            "promoted_settings": {"ml_players_context_enabled": False},
        }
        row, src = fit_players_context_for_league(
            cfg,
            "LPL",
            league_macro_key_fn=_macro_key,
        )
        self.assertTrue(row["ml_players_context_enabled"])
        self.assertEqual(src, "macro:major")

        blocked, blocked_src = fit_players_context_for_league(
            {"gate": {"decision": "reject"}},
            "LPL",
            league_macro_key_fn=_macro_key,
        )
        self.assertEqual(blocked, {})
        self.assertEqual(blocked_src, "gate:reject")

    def test_fused_post_calibration_selection_and_runtime_apply(self):
        cfg = {
            "version": "v1",
            "gate": {"decision": "promote"},
            "runtime": {
                "use_league_override": True,
                "max_abs_logit_shift": 0.15,
            },
            "promoted_model_global": {"a": 1.0, "b": 0.0},
            "promoted_model_by_league": {"LCK": {"a": 1.0, "b": 0.4}},
        }
        model, src = fit_fused_post_calibration_for_league(cfg, "LCK")
        self.assertEqual(src, "league:LCK")
        self.assertAlmostEqual(model["b"], 0.4)

        p_out, info = apply_fused_post_calibration(0.5, cfg, league_hint="LCK", use_fitted=True)
        self.assertTrue(info["applied"])
        self.assertEqual(info["source"], "league:LCK")
        self.assertAlmostEqual(info["logit_shift_applied"], 0.15)
        self.assertGreater(p_out, 0.5)

    def test_fused_post_calibration_respects_disabled_and_gate(self):
        p_out, info = apply_fused_post_calibration(0.5, {"gate": {"decision": "promote"}}, league_hint="", use_fitted=False)
        self.assertEqual(p_out, 0.5)
        self.assertEqual(info["reason"], "disabled_by_flag")

        p_out2, info2 = apply_fused_post_calibration(
            0.5,
            {"gate": {"decision": "reject"}},
            league_hint="LCK",
            use_fitted=True,
        )
        self.assertEqual(p_out2, 0.5)
        self.assertEqual(info2["gate_decision"], "reject")
        self.assertEqual(info2["reason"], "gate_not_promoted")


if __name__ == "__main__":
    unittest.main()
