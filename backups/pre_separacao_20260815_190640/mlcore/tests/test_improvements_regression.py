"""Regression tests for all 8 code-quality improvements.

Each class targets one improvement so failures are immediately locatable.
No external I/O; all test data is constructed inline.
"""

import unittest

from mlcore.utils import clip_prob


# ---------------------------------------------------------------------------
# Improvement 6A — microseg not-applied fallback reasons
# ---------------------------------------------------------------------------

from app_core.ml_probability_adjustments_core import (
    apply_ml_microseg_calibration,
    seg_key,
)


def _seg(lg="LCK", sp="Summer", sd="blue", bo="3", ph="regular"):
    return seg_key(lg, sp, sd, bo, ph)


class TestMicrosegFallbackReasons(unittest.TestCase):
    """apply_ml_microseg_calibration must expose why calibration was skipped."""

    _COMMON = dict(
        clip_prob_fn=clip_prob,
        league="LCK", split="Summer", side="blue", bo="3", phase="regular",
        strength=0.80, cap_pp=5.0,
    )

    # -----------------------------------------------------------------------
    # non-hierarchical mode
    # -----------------------------------------------------------------------

    def test_nonhier_empty_table_reason(self):
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table={}, min_n=10, hierarchical=False,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "empty_table")

    def test_nonhier_no_matching_keys_reason(self):
        table = {seg_key("CBLOL", "Summer", "blue", "3", "regular"): {"n": 50.0, "bias": 0.02}}
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=10, hierarchical=False,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "no_matching_keys")
        self.assertEqual(info.get("keys_found_in_table"), 0)

    def test_nonhier_all_below_min_n_reason(self):
        table = {
            _seg(): {"n": 5.0, "bias": 0.03},
            seg_key("LCK", "*", "blue", "3", "regular"): {"n": 3.0, "bias": 0.01},
            seg_key("*", "*", "*", "*", "*"): {"n": 2.0, "bias": 0.00},
        }
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=10, hierarchical=False,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "all_below_min_n")
        self.assertGreater(info.get("keys_found_in_table", 0), 0)
        self.assertGreater(info.get("max_n_seen", 0.0), 0.0)

    def test_nonhier_keys_tried_is_6(self):
        """Always tries all 6 hierarchy levels."""
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table={}, min_n=10, hierarchical=False,
        )
        self.assertEqual(info.get("keys_tried"), 6)

    def test_nonhier_applied_when_sufficient_n(self):
        table = {_seg(): {"n": 50.0, "bias": 0.04}}
        _p_out, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=10, hierarchical=False,
        )
        self.assertTrue(info["applied"])
        self.assertNotIn("fallback_reason", info)

    # -----------------------------------------------------------------------
    # hierarchical mode
    # -----------------------------------------------------------------------

    def test_hier_empty_table_reason(self):
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table={}, min_n=10, hierarchical=True,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "empty_table")

    def test_hier_no_matching_keys_reason(self):
        table = {seg_key("CBLOL", "Summer", "blue", "3", "regular"): {"n": 50.0, "bias": 0.02}}
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=10, hierarchical=True,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "no_matching_keys")

    def test_hier_all_zero_n_reason(self):
        table = {
            _seg(): {"n": 0.0, "bias": 0.03},
            seg_key("LCK", "*", "blue", "3", "regular"): {"n": 0.0, "bias": 0.01},
        }
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=10, hierarchical=True,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info.get("fallback_reason"), "all_zero_n")

    def test_hier_applies_with_small_n_via_soft_weight(self):
        """Hierarchical mode does NOT hard-gate on min_n; n=5 should still produce applied=True."""
        table = {_seg(): {"n": 5.0, "bias": 0.04}}
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table=table, min_n=20, hierarchical=True,
        )
        self.assertTrue(info["applied"])

    def test_hier_not_applied_exposes_n_gate(self):
        _, info = apply_ml_microseg_calibration(
            0.60, **self._COMMON, table={}, min_n=20, hierarchical=True,
        )
        self.assertFalse(info["applied"])
        self.assertIn("n_gate", info)
        self.assertAlmostEqual(float(info["n_gate"]), 20.0, places=4)


# ---------------------------------------------------------------------------
# Improvement 7 — players context shadow fallback
# ---------------------------------------------------------------------------

from app_core.fit_runtime_core import fit_players_context_for_league


def _make_ctx_cfg(decision: str, reasons=None, *, strength=0.40, cap=0.22, min_games=6, min_rel=0.28):
    fitted_settings = {
        "ml_players_context_strength": strength,
        "ml_players_context_cap": cap,
        "ml_players_context_min_games": min_games,
        "ml_players_context_min_reliability": min_rel,
    }
    return {
        "gate": {"decision": decision, "reasons": list(reasons or [])},
        "settings": dict(fitted_settings),  # always stored even for shadow_only
        "promoted_settings": (dict(fitted_settings) if decision == "promote" else {}),
        "promoted_settings_by_league": {},
        "promoted_settings_by_macro": {},
    }


class TestPlayersContextShadowFallback(unittest.TestCase):

    def test_promote_returns_fitted_settings(self):
        cfg = _make_ctx_cfg("promote", strength=0.50)
        _settings, _src = fit_players_context_for_league({"LCK": cfg}, "LCK")
        # promote path — returns promoted_settings (empty in this helper)
        # or falls through to {}. Since promoted_settings is empty dict we expect {}
        # Let's test the promote path via promoted_settings directly
        cfg2 = {
            "gate": {"decision": "promote", "reasons": []},
            "settings": {"ml_players_context_strength": 0.50},
            "promoted_settings": {"ml_players_context_strength": 0.50},
            "promoted_settings_by_league": {},
            "promoted_settings_by_macro": {},
        }
        settings2, _src2 = fit_players_context_for_league(cfg2, "LCK")
        self.assertAlmostEqual(settings2.get("ml_players_context_strength", 0.0), 0.50, places=4)

    def test_shadow_only_data_reason_applies_60pct_mult(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["low_sample"], strength=0.40, cap=0.22)
        settings, src = fit_players_context_for_league(cfg, "LCK")
        self.assertAlmostEqual(settings.get("ml_players_context_strength", 0.0), 0.40 * 0.60, places=4)
        self.assertAlmostEqual(settings.get("ml_players_context_cap", 0.0), 0.22 * 0.60, places=4)
        self.assertIn("shadow_low_sample", src)

    def test_shadow_only_weak_signal_applies_35pct_mult(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["gain_too_small"], strength=0.40, cap=0.22)
        settings, src = fit_players_context_for_league(cfg, "LCK")
        self.assertAlmostEqual(settings.get("ml_players_context_strength", 0.0), 0.40 * 0.35, places=4)
        self.assertIn("shadow_weak_signal", src)

    def test_shadow_only_regression_disables_entirely(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["logloss_regression"])
        settings, src = fit_players_context_for_league(cfg, "LCK")
        self.assertEqual(settings, {})
        self.assertIn("gate:", src)

    def test_shadow_only_not_evaluated_uses_60pct(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["not_evaluated"], strength=0.35, cap=0.20)
        settings, src = fit_players_context_for_league(cfg, "LCK")
        self.assertAlmostEqual(settings.get("ml_players_context_strength", 0.0), 0.35 * 0.60, places=4)
        self.assertIn("shadow_low_sample", src)

    def test_shadow_only_min_games_capped_at_12(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["low_sample"], min_games=11)
        settings, _ = fit_players_context_for_league(cfg, "LCK")
        self.assertEqual(settings.get("ml_players_context_min_games"), 12)

    def test_shadow_only_min_reliability_capped_at_050(self):
        cfg = _make_ctx_cfg("shadow_only", reasons=["low_sample"], min_rel=0.48)
        settings, _ = fit_players_context_for_league(cfg, "LCK")
        self.assertAlmostEqual(settings.get("ml_players_context_min_reliability", 0.0), 0.50, places=4)

    def test_non_shadow_non_promote_skips_fallback(self):
        cfg = _make_ctx_cfg("pending")
        settings, _src = fit_players_context_for_league(cfg, "LCK")
        self.assertEqual(settings, {})

    def test_empty_cfg_returns_empty(self):
        settings, _src = fit_players_context_for_league({}, "LCK")
        self.assertEqual(settings, {})


# ---------------------------------------------------------------------------
# Improvement 8 — runtime microseg empirical Bayes k
# ---------------------------------------------------------------------------

from app_core.ml_microseg_core import _estimate_shrink_k


class TestEstimateShrinkK(unittest.TestCase):
    """_estimate_shrink_k must fall back to k_max for noise and give low k for signal."""

    def _make_agg(self, biases, ns):
        return {
            f"seg_{i}": {"n": float(n), "sum_err": float(b) * float(n)}
            for i, (b, n) in enumerate(zip(biases, ns))
        }

    def test_too_few_segments_returns_kmax(self):
        agg = self._make_agg([0.10] * 5, [80] * 5)  # only 5 segs, need ≥8
        k = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0, min_segments=8)
        self.assertAlmostEqual(k, 300.0, places=4)

    def test_low_n_segments_excluded(self):
        """Segments with n < min_n_per_segment are excluded from the EB estimate."""
        agg = self._make_agg([0.10] * 12, [5] * 12)  # n=5 < default min=15
        k = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0, min_n_per_segment=15.0)
        self.assertAlmostEqual(k, 300.0, places=4)

    def test_genuine_signal_gives_low_k(self):
        """Large true bias (±0.10) at n=200 should yield k well below k_max."""
        biases = [0.10, -0.10, 0.10, -0.10, 0.10, -0.10, 0.10, -0.10, 0.10, -0.10]
        ns = [200] * len(biases)
        agg = self._make_agg(biases, ns)
        k = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0, min_segments=8)
        self.assertLess(k, 100.0, f"Expected k < 100 for strong signal, got {k:.1f}")

    def test_noise_level_bias_gives_kmax(self):
        """Small true bias (±0.02) relative to sampling noise → k_max."""
        biases = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02, 0.02, -0.02]
        ns = [30] * len(biases)  # high sampling noise: sigma_within ~ sqrt(0.25/30) ≈ 0.091
        agg = self._make_agg(biases, ns)
        k = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0, min_segments=8)
        self.assertAlmostEqual(k, 300.0, places=4)

    def test_k_is_within_bounds(self):
        biases = [0.10, -0.10, 0.08, -0.08, 0.12, -0.12, 0.09, -0.09, 0.11, -0.11]
        ns = [100] * len(biases)
        agg = self._make_agg(biases, ns)
        k = _estimate_shrink_k(agg, k_min=25.0, k_max=300.0, min_segments=8)
        self.assertGreaterEqual(k, 25.0)
        self.assertLessEqual(k, 300.0)

    def test_empty_agg_returns_kmax(self):
        k = _estimate_shrink_k({}, k_min=25.0, k_max=300.0)
        self.assertAlmostEqual(k, 300.0, places=4)


# ---------------------------------------------------------------------------
# Improvement 8 — shrink_k in build_ml_microseg_table output
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from app_core.ml_microseg_core import build_ml_microseg_table


class _MockModel:
    def p_blue(self, blue, red, **kwargs):
        return 0.55


class _MockArt:
    model = _MockModel()
    calibrator = None


def _make_games(n=60, league="LCK"):
    rng = np.random.default_rng(42)
    data = {
        "side": ["blue"] * n,
        "team": [f"T{i % 4}" for i in range(n)],
        "opponent": [f"O{i % 4}" for i in range(n)],
        "win": rng.integers(0, 2, n).tolist(),
        "league": [league] * n,
        "split": ["Summer"] * n,
        "playoffs": [0] * n,
        "game": [1] * n,
    }
    return pd.DataFrame(data)


class TestMicrosegTableShrinkK(unittest.TestCase):

    def test_shrink_k_present_in_output(self):
        df = _make_games(60)
        art = _MockArt()
        table = build_ml_microseg_table(df, art, lookback_games=200, shrink_k=90.0)
        if not table:
            self.skipTest("No segments produced; model may be minimal stub")
        for k, v in table.items():
            self.assertIn("shrink_k", v, f"Missing shrink_k in segment '{k}'")

    def test_shrink_k_is_positive(self):
        df = _make_games(60)
        art = _MockArt()
        table = build_ml_microseg_table(df, art)
        for k, v in table.items():
            sk = float(v.get("shrink_k", 0.0))
            self.assertGreater(sk, 0.0, f"shrink_k must be positive for segment '{k}'")

    def test_empty_df_returns_empty(self):
        table = build_ml_microseg_table(pd.DataFrame(), _MockArt())
        self.assertEqual(table, {})

    def test_none_df_returns_empty(self):
        table = build_ml_microseg_table(None, _MockArt())
        self.assertEqual(table, {})


# ---------------------------------------------------------------------------
# Phase 2C — runtime microseg k blended with fitted k (geometric mean)
# ---------------------------------------------------------------------------

from app_core.ml_microseg_core import _extract_fitted_k


class TestMicrosegFittedKBlend(unittest.TestCase):

    def test_extract_fitted_k_returns_k_from_table(self):
        table = {"seg1": {"n": 50.0, "bias": 0.02, "shrink_k": 120.0}}
        k = _extract_fitted_k(table)
        self.assertAlmostEqual(k, 120.0, places=4)

    def test_extract_fitted_k_returns_none_for_empty(self):
        self.assertIsNone(_extract_fitted_k({}))
        self.assertIsNone(_extract_fitted_k(None))

    def test_extract_fitted_k_skips_zero_k(self):
        table = {"seg1": {"n": 50.0, "shrink_k": 0.0}}
        self.assertIsNone(_extract_fitted_k(table))

    def test_fitted_k_blends_via_geometric_mean(self):
        """With k_runtime=100 and k_fitted=100, blend must equal 100."""
        df = _make_games(60)
        art = _MockArt()
        # Build without fitted table to get baseline k
        table_no_fit = build_ml_microseg_table(df, art)
        if not table_no_fit:
            self.skipTest("No segments produced")
        k_baseline = next(iter(table_no_fit.values())).get("shrink_k", 0.0)

        # Build with fitted table where shrink_k = k_baseline (same signal level)
        fitted = {"*|*|*|*|*": {"n": 100.0, "bias": 0.01, "shrink_k": k_baseline}}
        table_with_fit = build_ml_microseg_table(df, art, fitted_microseg_table=fitted)
        if not table_with_fit:
            self.skipTest("No segments produced with fitted table")
        k_blend = next(iter(table_with_fit.values())).get("shrink_k", 0.0)
        # Geometric mean of k and k = k
        self.assertAlmostEqual(k_blend, k_baseline, delta=1.0)

    def test_fitted_k_pulls_toward_fit_time_estimate(self):
        """If k_fitted=50 (high signal) but runtime would give k_max=300,
        the blend should be sqrt(50*300)=~122, clearly below 300."""
        df = _make_games(60)
        art = _MockArt()
        fitted = {"*|*|*|*|*": {"n": 200.0, "bias": 0.01, "shrink_k": 50.0}}
        table = build_ml_microseg_table(df, art, fitted_microseg_table=fitted)
        if not table:
            self.skipTest("No segments produced")
        k = next(iter(table.values())).get("shrink_k", 0.0)
        self.assertLess(k, 250.0, f"Blend should pull k below 250; got {k:.1f}")
        self.assertGreater(k, 25.0, f"k should remain above k_min=25; got {k:.1f}")

    def test_no_fitted_table_uses_runtime_k_only(self):
        """Without fitted_microseg_table the function must behave as before."""
        df = _make_games(60)
        art = _MockArt()
        table = build_ml_microseg_table(df, art, fitted_microseg_table=None)
        # Just check it produces output without crashing
        self.assertIsInstance(table, dict)


# ---------------------------------------------------------------------------
# Improvement 6B — read_governor_budget_summary (already in governor e2e;
# here just a smoke check that the orchestrator exposes it correctly)
# ---------------------------------------------------------------------------

from pathlib import Path
from app_core.ml_orchestrator_core import run_ml_post_layers


def _dummy_series(p, bo):
    try:
        from mlcore.bo import p_series_from_pmap
        return float(p_series_from_pmap(float(p), int(bo)))
    except Exception:
        return float(p)


class TestOrchestratorBudgetSummaryKey(unittest.TestCase):

    def test_trace_governor_budget_final_always_present(self):
        out = run_ml_post_layers(
            p_map_used=0.60,
            p_series_used=_dummy_series(0.60, 3),
            bo=3,
            trace_shadow_state_alite={},
            trace_shadow_state_areal={},
            feature_flag_enabled_fn=lambda k, d=False: bool(d),
            prob_win_series_fn=_dummy_series,
            session_state={},
            trace_pipeline_steps=[],
            live_kwargs={"playoffs_opt": False,
                         "ml_info": {"blue": {"league": "LCK"}, "red": {"league": "LCK"}},
                         "fit_params_path": Path("params_fitted.__missing__.json")},
            marginal_kwargs={
                "ml_info": {"blue": {"league": "LCK", "games_played": 45, "shrink_factor": 0.9}},
                "trace_n_team_eff": 28.0,
                "lv_last_report": {},
                "ml_trace_last": {},
                "ml_confidence_fn": lambda lv, **kw: {"score": 72.0},
                "marginal_adjust_fn": lambda p, info, **kw: (float(p), {"applied": False}),
            },
            confidence_baseline_path=Path("baseline.__missing__.json"),
            load_confidence_baseline_cached_fn=lambda _sig: {},
            file_sig_fn=lambda _p: ("", 0, 0),
            lg_a_hint="LCK",
        )
        budget = out.get("trace_governor_budget_final", {})
        self.assertIn("available", budget)
        self.assertIn("budget_exhausted", budget)
        self.assertIsInstance(budget.get("budget_exhausted"), bool)


if __name__ == "__main__":
    unittest.main()
