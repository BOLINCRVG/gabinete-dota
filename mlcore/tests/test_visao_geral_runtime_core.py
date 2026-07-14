import unittest

from app_core.visao_geral_runtime_core import (
    apply_runtime_tuning_to_session,
    build_input_signature_payload,
    build_microseg_caption,
    compute_input_signature,
    resolve_shared_league_hint,
)


class VisaoGeralRuntimeCoreTests(unittest.TestCase):
    def test_build_microseg_caption(self):
        text = build_microseg_caption({"applied": True, "key": "x", "source": "live", "bo": 3, "phase": "late", "n": 10, "bias": 0.03})
        self.assertIn("Microseg calibracao", text)
        self.assertIn("+3.0 pp", text)

    def test_resolve_shared_league_hint(self):
        self.assertEqual(resolve_shared_league_hint(ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"}}), "LCK")
        self.assertEqual(resolve_shared_league_hint(ml_info={}, fit_league_hint="LPL"), "LPL")
        self.assertEqual(
            resolve_shared_league_hint(
                ml_info={"blue": {"league": "LCK"}, "red": {"league": "LPL"}},
                match_league="INTER",
                league_a="LCK",
                league_b="LPL",
            ),
            "",
        )

    def test_build_and_hash_input_signature(self):
        payload = build_input_signature_payload(
            matchup={"bo": 3},
            filters={"year": "2026"},
            config={"engine": "mlcore_v2"},
            lineup={"mode": "ml"},
            versions={"csv": {"path": "x"}},
        )
        sig = compute_input_signature(payload)
        self.assertIn("matchup", sig["payload"])
        self.assertEqual(len(sig["sha256"]), 64)

    def test_apply_runtime_tuning_to_session(self):
        state = {}
        out = apply_runtime_tuning_to_session(
            session_state=state,
            league_hint="LCK",
            bo=3,
            confidence_baseline={},
            confidence_governor_budget_fn=lambda league_hint, confidence_baseline=None: {"budget_pp": 7.5},
            series_coherence_post_cap_fn=lambda league_hint, bo, confidence_baseline=None: {"max_dev_pp": 14.0},
            governor_default_pp=8.0,
        )
        self.assertEqual(state["_ml_confidence_governor_budget_runtime_pp"], 7.5)
        self.assertEqual(state["_ml_series_coherence_post_max_dev_pp_runtime"], 14.0)
        self.assertEqual(out["gov_budget_pp"], 7.5)


if __name__ == "__main__":
    unittest.main()
