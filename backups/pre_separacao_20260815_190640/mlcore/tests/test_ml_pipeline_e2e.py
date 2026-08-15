"""End-to-end test for the ML post-layers pipeline.

Exercises run_ml_post_layers with real sub-functions (no mocks for the
governor, state overrides, or series coherence). Only external callables
that require live data (confidence scoring, marginal adjust) are stubbed
with minimal implementations that produce well-formed output.
"""

import math
import unittest
from pathlib import Path

from app_core.ml_orchestrator_core import run_ml_post_layers


def _ff_all_off(key, default=False):
    return False


def _prob_win_series(p: float, bo: int) -> float:
    """Combinatorial series win probability (no side effects)."""
    from mlcore.bo import p_series_from_pmap
    try:
        return float(p_series_from_pmap(float(p), int(bo)))
    except Exception:
        return float(p)


def _ml_confidence_fn(lv_last_report, *, league_hint="", n_team_eff=float("nan"), ml_trace=None, confidence_baseline=None):
    return {"score": 72.0, "league": str(league_hint or "")}


def _marginal_adjust_fn(p_in, ml_info, *, league="", enabled=True, **kwargs):
    return float(p_in), {"applied": False, "league": str(league or "")}


_BASE_MARGINAL_KWARGS = {
    "ml_info": {"blue": {"league": "LCK", "games_played": 45, "shrink_factor": 0.9}},
    "trace_n_team_eff": 28.0,
    "lv_last_report": {},
    "ml_trace_last": {},
    "ml_confidence_fn": _ml_confidence_fn,
    "marginal_adjust_fn": _marginal_adjust_fn,
}

_BASE_LIVE_KWARGS = {
    "playoffs_opt": False,
    "ml_info": {"blue": {"league": "LCK"}, "red": {"league": "LCK"}},
    "fit_params_path": Path("params_fitted.__missing__.json"),
}


class TestMlPipelineE2E(unittest.TestCase):
    """Full pipeline output sanity checks — no mocks for core logic."""

    def _run(self, p_map=0.60, p_series=None, bo=3, shadow_alite=None,
             shadow_areal=None, session_state=None, feature_flag_fn=None,
             extra_marginal=None):
        if p_series is None:
            p_series = _prob_win_series(p_map, bo)
        return run_ml_post_layers(
            p_map_used=p_map,
            p_series_used=p_series,
            bo=bo,
            trace_shadow_state_alite=shadow_alite or {},
            trace_shadow_state_areal=shadow_areal or {},
            feature_flag_enabled_fn=feature_flag_fn or _ff_all_off,
            prob_win_series_fn=_prob_win_series,
            session_state=session_state or {},
            trace_pipeline_steps=[],
            live_kwargs=dict(_BASE_LIVE_KWARGS),
            marginal_kwargs={**_BASE_MARGINAL_KWARGS, **(extra_marginal or {})},
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
        )

    # ------------------------------------------------------------------
    # Output structure
    # ------------------------------------------------------------------

    def test_output_has_required_keys(self):
        out = self._run()
        required = [
            "p_map_used", "p_series_used",
            "trace_pipeline_steps", "trace_pipeline_order",
            "trace_state_alite_live", "trace_state_areal_live",
            "trace_marginal_trust_info", "trace_series_coherence_post_info",
            "trace_governor_budget_final",
            "_ff_ml_state_alite_live", "_ff_ml_state_areal_live",
        ]
        for key in required:
            self.assertIn(key, out, f"Missing key: {key}")

    def test_p_map_is_finite_and_in_range(self):
        out = self._run(p_map=0.60)
        p = float(out["p_map_used"])
        self.assertTrue(math.isfinite(p), "p_map_used must be finite")
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_p_series_is_finite_and_in_range(self):
        out = self._run(p_map=0.60, bo=3)
        ps = float(out["p_series_used"])
        self.assertTrue(math.isfinite(ps), "p_series_used must be finite")
        self.assertGreater(ps, 0.0)
        self.assertLess(ps, 1.0)

    def test_p_series_bo5_more_extreme_than_bo3(self):
        """For a strong favorite, BO5 series probability must exceed BO3."""
        out3 = self._run(p_map=0.65, bo=3)
        out5 = self._run(p_map=0.65, bo=5)
        self.assertGreater(float(out5["p_series_used"]), float(out3["p_series_used"]))

    def test_pipeline_does_not_flip_favourite(self):
        """If blue is favoured on maps, blue must stay favoured after the pipeline."""
        out = self._run(p_map=0.62)
        self.assertGreater(float(out["p_map_used"]), 0.5)

    def test_even_matchup_stays_near_50(self):
        out = self._run(p_map=0.50)
        p = float(out["p_map_used"])
        self.assertAlmostEqual(p, 0.50, delta=0.08)

    # ------------------------------------------------------------------
    # State override
    # ------------------------------------------------------------------

    def test_shadow_alite_applied_moves_p_map(self):
        out = self._run(
            p_map=0.60,
            shadow_alite={"applied": True, "p_map_new": 0.55},
            feature_flag_fn=lambda k, d=False: str(k) == "ff_ml_state_alite_live",
        )
        tr = out.get("trace_state_alite_live", {})
        self.assertTrue(bool(tr.get("applied", False)), "state_alite should be applied")
        self.assertAlmostEqual(float(out["p_map_used"]), 0.55, delta=0.05)

    def test_disabled_shadow_alite_keeps_p_map(self):
        out = self._run(
            p_map=0.60,
            shadow_alite={"applied": True, "p_map_new": 0.50},
            feature_flag_fn=_ff_all_off,
        )
        tr = out.get("trace_state_alite_live", {})
        self.assertFalse(bool(tr.get("applied", True)))
        self.assertAlmostEqual(float(out["p_map_used"]), 0.60, delta=0.02)

    # ------------------------------------------------------------------
    # Governor budget
    # ------------------------------------------------------------------

    def test_governor_budget_summary_is_available(self):
        ss = {
            "ml_confidence_governor_enabled": True,
            "ml_confidence_governor_live_enabled": True,
            "ml_confidence_governor_max_total_shrink_pp": 8.0,
        }
        out = self._run(session_state=ss)
        budget = out.get("trace_governor_budget_final", {})
        self.assertIn("available", budget)
        self.assertIn("budget_exhausted", budget)
        self.assertIsInstance(budget.get("budget_exhausted"), bool)

    def test_governor_limits_large_state_override(self):
        """A large state override with a 2pp budget must be capped."""
        from app_core.confidence_governor_core import apply_confidence_governor

        def _gov_fn(**kwargs):
            return apply_confidence_governor(
                stage=str(kwargs.get("stage", "")),
                p_before=float(kwargs.get("p_before", 0.5)),
                p_candidate=float(kwargs.get("p_candidate", 0.5)),
                session_state=kwargs.get("session_state"),
                enabled=bool(kwargs.get("enabled", True)),
                max_total_shrink_pp=float(kwargs.get("max_total_shrink_pp", 2.0)),
                stage_cap_pp=kwargs.get("stage_cap_pp"),
            )

        ss = {
            "ml_confidence_governor_enabled": True,
            "ml_confidence_governor_live_enabled": True,
            "ml_confidence_governor_max_total_shrink_pp": 2.0,
        }
        out = run_ml_post_layers(
            p_map_used=0.65,
            p_series_used=_prob_win_series(0.65, 3),
            bo=3,
            trace_shadow_state_alite={"applied": True, "p_map_new": 0.50},
            trace_shadow_state_areal={},
            feature_flag_enabled_fn=lambda k, d=False: str(k) == "ff_ml_state_alite_live",
            prob_win_series_fn=_prob_win_series,
            session_state=ss,
            trace_pipeline_steps=[],
            live_kwargs=dict(_BASE_LIVE_KWARGS),
            marginal_kwargs={**_BASE_MARGINAL_KWARGS},
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
            apply_confidence_governor_fn=_gov_fn,
        )
        tr = out.get("trace_state_alite_live", {})
        gov = tr.get("governor", {}) if isinstance(tr.get("governor"), dict) else {}
        self.assertTrue(bool(gov.get("limited", False)), "Governor must limit the override")
        p_final = float(out["p_map_used"])
        # With 2pp budget, p_map should not move more than ~2pp from 0.65
        self.assertGreater(p_final, 0.63 - 0.01)

    # ------------------------------------------------------------------
    # Series coherence
    # ------------------------------------------------------------------

    def test_series_coherence_applied_for_bo3(self):
        """BO3 pipeline must run series_economic_coherence_post."""
        out = self._run(p_map=0.65, bo=3, p_series=0.40)
        coh = out.get("trace_series_coherence_post_info", {})
        self.assertIn("enabled", coh)
        self.assertIn("bo", coh)
        self.assertEqual(int(coh.get("bo", 0)), 3)

    def test_series_coherence_applied_for_bo5(self):
        out = self._run(p_map=0.65, bo=5, p_series=0.30)
        coh = out.get("trace_series_coherence_post_info", {})
        self.assertEqual(int(coh.get("bo", 0)), 5)

    def test_pipeline_order_contains_expected_stages(self):
        out = self._run(
            shadow_alite={"applied": True, "p_map_new": 0.58},
            feature_flag_fn=lambda k, d=False: str(k) == "ff_ml_state_alite_live",
        )
        order = list(out.get("trace_pipeline_order", []))
        self.assertIn("state_alite_live_override", order)
        self.assertIn("marginal_trust_guard", order)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_nan_inputs_return_nan_gracefully(self):
        out = self._run(p_map=float("nan"), p_series=float("nan"))
        # Should not raise; p_map_used may be nan but keys must exist
        self.assertIn("p_map_used", out)

    def test_bo1_skips_series_coherence(self):
        out = self._run(p_map=0.60, bo=1)
        coh = out.get("trace_series_coherence_post_info", {})
        # bo=1 is not in (3, 5), so coherence is not applied
        self.assertFalse(bool(coh.get("applied", False)))


if __name__ == "__main__":
    unittest.main()
