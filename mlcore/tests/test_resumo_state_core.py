import unittest

from app_core.resumo_state_core import (
    apply_resumo_combo_preset,
    apply_resumo_filter_values,
    delete_active_resumo_filter_profile,
    ensure_resumo_combo_state,
    get_resumo_market_trace,
    get_resumo_view_state,
    list_resumo_filter_presets,
    load_resumo_filter_custom,
    save_resumo_filter_custom,
    save_resumo_filter_profile,
    set_resumo_market_trace,
)


class ResumoStateCoreTest(unittest.TestCase):
    def test_list_presets_and_apply_values(self):
        state = {
            "resumo_filter_profiles": {
                "valor_lpl": {"resumo_max_odd": 1.8},
            },
            "resumo_filter_preset": "valor_lpl",
        }
        profiles, options, current, preset_idx = list_resumo_filter_presets(state)
        self.assertIn("valor_lpl", profiles)
        self.assertEqual(options, ["Custom", "valor_lpl"])
        self.assertEqual(current, "valor_lpl")
        self.assertEqual(preset_idx, 1)

        applied = apply_resumo_filter_values(state, {"resumo_req_h2h": False}, preset_name="valor_lpl")
        self.assertTrue(applied)
        self.assertFalse(state["resumo_req_h2h"])
        self.assertEqual(state["resumo_filter_profile_active"], "valor_lpl")

    def test_custom_snapshot_and_profile_lifecycle(self):
        state = {
            "resumo_max_odd": 1.75,
            "resumo_req_liga": False,
            "resumo_req_h2h": True,
            "resumo_req_model": False,
            "resumo_min_sample_liga": 12,
            "resumo_min_sample_h2h": 4,
        }
        payload = save_resumo_filter_custom(state)
        self.assertEqual(payload["resumo_max_odd"], 1.75)
        self.assertEqual(load_resumo_filter_custom(state), payload)
        self.assertEqual(state["resumo_filter_preset"], "Custom")

        saved_name = save_resumo_filter_profile(state, " valor_lck ")
        self.assertEqual(saved_name, "valor_lck")
        self.assertIn("valor_lck", state["resumo_filter_profiles"])

        deleted = delete_active_resumo_filter_profile(state)
        self.assertEqual(deleted, "valor_lck")
        self.assertNotIn("valor_lck", state["resumo_filter_profiles"])
        self.assertEqual(state["resumo_filter_preset"], "Custom")

    def test_combo_preset_defaults_and_apply(self):
        state = {}
        ensure_resumo_combo_state(state)
        self.assertEqual(state["resumo_combo_preset"], "Neutro")
        self.assertFalse(state["resumo_combo_use_mismatch"])

        state["resumo_combo_preset"] = "Agressivo"
        hist_w, style_w = apply_resumo_combo_preset(state)
        self.assertEqual((hist_w, style_w), (0.65, 0.30))
        self.assertEqual(state["resumo_combo_hist_w"], 0.65)
        self.assertEqual(state["resumo_combo_style_w"], 0.30)

    def test_view_state_snapshot_and_trace_helpers(self):
        state = {
            "resumo_max_odd": 1.88,
            "resumo_req_liga": False,
            "resumo_mix_w_model": 55,
            "combo_kills_text": "26.5,27.5",
            "_p_ml_scope": 0.61,
            "_ml_trace_markets": [{"mercado": "ML"}],
        }
        snap = get_resumo_view_state(state)
        self.assertEqual(snap["resumo_max_odd"], 1.88)
        self.assertFalse(snap["resumo_req_liga"])
        self.assertEqual(snap["resumo_mix_w_model"], 55.0)
        self.assertEqual(snap["combo_kills_text"], "26.5,27.5")
        self.assertEqual(snap["p_ml_scope"], 0.61)
        self.assertEqual(snap["ml_trace_markets"], [{"mercado": "ML"}])

        rows = set_resumo_market_trace(state, [{"mercado": "Kills"}])
        self.assertEqual(rows, [{"mercado": "Kills"}])
        self.assertEqual(get_resumo_market_trace(state), [{"mercado": "Kills"}])


if __name__ == "__main__":
    unittest.main()
