"""Tests for app_core/ml_confidence_scoring_core.py.

ml_confidence_from_layer_validation is a complex scoring function with
~15 independent adjustment branches. Tests use a 'neutral' fixture
(score ≈ 61) so both positive and negative adjustments are observable
without hitting the 0 / 100 clamp.

Neutral base breakdown:
  start=50, gain=0.003 (+5), n_total=400 (+6), logloss=0.68 (-5),
  n_team_eff=12 (+5), no trace, no baseline → score=61
"""

import math
import unittest
from typing import Any, Dict, Optional

from app_core.ml_confidence_scoring_core import ml_confidence_from_layer_validation


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _report(gain: float = 0.003, n_total: int = 400, logloss: float = 0.68,
            league: Optional[str] = None) -> Dict[str, Any]:
    """Build a layer_validation report. Defaults give a mid-range score."""
    fused = {"gain_vs_team_logloss": gain, "n_total": n_total, "logloss": logloss}
    r: Dict[str, Any] = {"models": {"fused_active": fused}}
    if league:
        r["by_league"] = {league: {"fused_active": fused}}
    return r


def _guards(**kw) -> Dict[str, Any]:
    return dict(kw)


def _trace(guards: Optional[Dict] = None, **layers) -> Dict[str, Any]:
    t: Dict[str, Any] = {}
    if guards is not None:
        t["guards"] = guards
    t.update(layers)
    return t


def _neutral() -> Dict[str, Any]:
    """Return kwargs giving score ≈ 61 — room for both + and − adjustments."""
    return dict(
        report=_report(),          # gain +5, n_total +6, logloss -5 → net +6
        league_hint="",
        n_team_eff=12.0,           # +5
        ml_trace=None,
        confidence_baseline=None,
        # total = 50 + 6 + 5 = 61
    )


def _score(**overrides) -> float:
    kw = {**_neutral(), **overrides}
    return ml_confidence_from_layer_validation(**kw)["score"]


def _out(**overrides) -> Dict[str, Any]:
    kw = {**_neutral(), **overrides}
    return ml_confidence_from_layer_validation(**kw)


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------

class TestOutputContract(unittest.TestCase):

    def test_required_keys_present(self):
        out = _out()
        for key in ("grade", "score", "source", "n_total", "logloss",
                    "gain_vs_team_logloss", "pipeline_adj", "pipeline_meta_method",
                    "pipeline_summary", "baseline_adj", "baseline_source",
                    "baseline_band"):
            self.assertIn(key, out, f"missing key: {key}")

    def test_score_clamped_to_100(self):
        out = _out(confidence_baseline={
            "global": {"score_bias": 9999.0}, "by_league": {},
        })
        self.assertLessEqual(out["score"], 100.0)

    def test_score_clamped_to_0(self):
        out = _out(confidence_baseline={
            "global": {"score_bias": -9999.0}, "by_league": {},
        })
        self.assertGreaterEqual(out["score"], 0.0)

    def test_grade_values(self):
        self.assertIn(_out()["grade"], ("A (Alta)", "B (Media)", "C (Baixa)"))

    def test_grade_a_when_score_ge_80(self):
        # neutral=61, C-trained +10, A-real +8, A-lite +6 = 85
        out = _out(ml_trace=_trace(
            guards=_guards(
                state_areal_live_on=True, state_areal_live_applied=True,
                state_alite_live_on=True, state_alite_live_applied=True,
                meta_c_lite_live_on=True, meta_c_lite_live_applied=True,
            ),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        ))
        self.assertGreaterEqual(out["score"], 80.0)
        self.assertEqual(out["grade"], "A (Alta)")

    def test_grade_c_when_score_below_65(self):
        # neutral=61 → C
        out = _out()
        self.assertLess(out["score"], 65.0)
        self.assertEqual(out["grade"], "C (Baixa)")

    def test_grade_b_between_65_and_80(self):
        # neutral 61 + C-trained +10 = 71
        out = _out(ml_trace=_trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        ))
        self.assertGreaterEqual(out["score"], 65.0)
        self.assertLess(out["score"], 80.0)
        self.assertEqual(out["grade"], "B (Media)")

    def test_score_is_float(self):
        self.assertIsInstance(_out()["score"], float)

    def test_score_always_finite(self):
        for rep in (None, {}, _report(), {"error": "x"}):
            s = ml_confidence_from_layer_validation(report=rep)["score"]
            self.assertTrue(math.isfinite(s))


# ---------------------------------------------------------------------------
# 2. Source selection
# ---------------------------------------------------------------------------

class TestSourceSelection(unittest.TestCase):

    def test_source_global_when_no_league_match(self):
        out = _out(report=_report(), league_hint="MISSING")
        self.assertEqual(out["source"], "global")

    def test_source_league_when_league_present_in_report(self):
        out = _out(report=_report(league="LCK"), league_hint="LCK")
        self.assertEqual(out["source"], "liga:LCK")

    def test_source_global_when_report_none(self):
        out = ml_confidence_from_layer_validation(report=None)
        # no report → falls to else branch → score -15, source stays "global"
        self.assertEqual(out["source"], "global")


# ---------------------------------------------------------------------------
# 3. Gain adjustment (holding n_total/logloss/nte fixed via neutral)
# ---------------------------------------------------------------------------

class TestGainAdjustment(unittest.TestCase):

    def _isolated(self, gain):
        """Only gain varies; others are the neutral values."""
        r = _report(gain=gain, n_total=400, logloss=0.68)
        return ml_confidence_from_layer_validation(
            report=r, n_team_eff=12.0, ml_trace=None, confidence_baseline=None,
        )["score"]

    def test_high_gain_ge_0015_gives_highest_score(self):
        self.assertGreater(self._isolated(0.020), self._isolated(0.010))

    def test_medium_gain_ge_0008_higher_than_small(self):
        self.assertGreater(self._isolated(0.010), self._isolated(0.003))

    def test_small_positive_gain_higher_than_negative(self):
        self.assertGreater(self._isolated(0.003), self._isolated(-0.010))

    def test_negative_gain_reduces_score_below_start(self):
        # 50 - 20 (neg gain) + 6 (n_total=400) - 5 (logloss=0.68) + 5 (nte=12) = 36
        self.assertLess(self._isolated(-0.010), 50.0)

    def test_nan_gain_deducts(self):
        # 50 - 8 (nan) + 6 - 5 + 5 = 48
        self.assertLess(self._isolated(float("nan")), 50.0)

    def test_gain_returned_in_output(self):
        out = _out(report=_report(gain=0.015))
        self.assertAlmostEqual(out["gain_vs_team_logloss"], 0.015, places=5)

    def test_nan_gain_returned_as_none(self):
        out = _out(report=_report(gain=float("nan")))
        self.assertIsNone(out["gain_vs_team_logloss"])


# ---------------------------------------------------------------------------
# 4. n_total adjustment
# ---------------------------------------------------------------------------

class TestNTotalAdjustment(unittest.TestCase):

    def _isolated(self, n_total):
        r = _report(gain=0.003, n_total=n_total, logloss=0.68)
        return ml_confidence_from_layer_validation(
            report=r, n_team_eff=float("nan"), ml_trace=None, confidence_baseline=None,
        )["score"]

    def test_large_n_ge1500_highest(self):
        self.assertGreater(self._isolated(2000), self._isolated(800))

    def test_medium_n_ge700_mid(self):
        self.assertGreater(self._isolated(800), self._isolated(400))

    def test_small_n_ge300_low(self):
        self.assertGreater(self._isolated(400), self._isolated(50))

    def test_tiny_n_below300_subtracts(self):
        # 50 +5 (gain) -8 (tiny n) -5 (logloss) = 42 < 50
        self.assertLess(self._isolated(50), 50.0)

    def test_n_total_returned(self):
        out = _out(report=_report(n_total=800))
        self.assertEqual(out["n_total"], 800)


# ---------------------------------------------------------------------------
# 5. Logloss adjustment
# ---------------------------------------------------------------------------

class TestLoglossAdjustment(unittest.TestCase):

    def _isolated(self, logloss):
        r = _report(gain=0.003, n_total=400, logloss=logloss)
        return ml_confidence_from_layer_validation(
            report=r, n_team_eff=float("nan"), ml_trace=None, confidence_baseline=None,
        )["score"]

    def test_low_logloss_le062_highest(self):
        self.assertGreater(self._isolated(0.60), self._isolated(0.64))

    def test_medium_logloss_le066_mid(self):
        self.assertGreater(self._isolated(0.64), self._isolated(0.70))

    def test_high_logloss_above066_subtracts(self):
        # 50 +5 +6 -5 = 56; with nte=nan: 50 +5 +6 -5 = 56 (neutral) – wait: no nte here
        # 50 +5 (gain) +6 (n400) -5 (logloss 0.70) = 56, not < 50
        # Let's compare it's LESS than the low logloss case
        self.assertLess(self._isolated(0.70), self._isolated(0.60))

    def test_logloss_returned(self):
        out = _out(report=_report(logloss=0.62))
        self.assertAlmostEqual(out["logloss"], 0.62, places=5)

    def test_nan_logloss_returned_as_none(self):
        out = _out(report=_report(logloss=float("nan")))
        self.assertIsNone(out["logloss"])


# ---------------------------------------------------------------------------
# 6. n_team_eff adjustment
# ---------------------------------------------------------------------------

class TestNTeamEffAdjustment(unittest.TestCase):

    def _isolated(self, nte):
        return ml_confidence_from_layer_validation(
            report=_report(), n_team_eff=nte, ml_trace=None, confidence_baseline=None,
        )["score"]

    def test_large_nte_ge25_highest(self):
        self.assertGreater(self._isolated(30.0), self._isolated(12.0))

    def test_medium_nte_ge12_mid(self):
        self.assertGreater(self._isolated(12.0), self._isolated(5.0))

    def test_small_nte_below12_subtracts(self):
        # should score less than no adjustment (nan)
        self.assertLess(self._isolated(5.0), self._isolated(float("nan")))

    def test_nan_nte_no_adjustment(self):
        with_nan = self._isolated(float("nan"))
        # neutral score without nte: 50 +5 +6 -5 = 56
        self.assertAlmostEqual(with_nan, 56.0)


# ---------------------------------------------------------------------------
# 7. Pipeline trace adjustments — use neutral base (score=61)
# ---------------------------------------------------------------------------

class TestPipelineAdjustments(unittest.TestCase):

    def _diff(self, trace) -> float:
        """Score difference: with_trace - neutral_base."""
        return _score(ml_trace=trace) - _score(ml_trace=None)

    def test_areal_applied_adds_positive(self):
        t = _trace(guards=_guards(state_areal_live_on=True, state_areal_live_applied=True))
        self.assertGreater(self._diff(t), 0)

    def test_areal_on_not_applied_subtracts(self):
        t = _trace(guards=_guards(state_areal_live_on=True, state_areal_live_applied=False))
        self.assertLess(self._diff(t), 0)

    def test_areal_applied_note_in_summary(self):
        t = _trace(guards=_guards(state_areal_live_on=True, state_areal_live_applied=True))
        self.assertIn("A-real ativo", _out(ml_trace=t)["pipeline_summary"])

    def test_alite_applied_adds_positive(self):
        t = _trace(guards=_guards(state_alite_live_on=True, state_alite_live_applied=True))
        self.assertGreater(self._diff(t), 0)

    def test_meta_c_trained_adds_positive(self):
        t = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        )
        self.assertGreater(self._diff(t), 0)

    def test_meta_c_trained_more_than_fallback(self):
        trained = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        )
        fallback = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "simple_blend"},
        )
        self.assertGreater(_score(ml_trace=trained), _score(ml_trace=fallback))

    def test_meta_c_not_applied_subtracts(self):
        t = _trace(guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=False))
        self.assertLess(self._diff(t), 0)

    def test_meta_c_trained_note_in_summary(self):
        t = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        )
        self.assertIn("C treinado", _out(ml_trace=t)["pipeline_summary"])

    def test_meta_c_aggressive_reduces_score(self):
        normal = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1", "diff_pp": 5.0},
        )
        aggressive = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1", "diff_pp": 20.0},
        )
        self.assertGreater(_score(ml_trace=normal), _score(ml_trace=aggressive))

    def test_consistency_guard_subtracts(self):
        t = _trace(guards=_guards(consistency_guard_triggered=True))
        self.assertLess(self._diff(t), 0)
        self.assertIn("consistency guard", _out(ml_trace=t)["pipeline_summary"])

    def test_patch_guard_refit_bigger_penalty_than_warn(self):
        refit = _trace(guards=_guards(patch_guard_applied=True, patch_guard_status="REFIT_REQUIRED"))
        warn = _trace(guards=_guards(patch_guard_applied=True, patch_guard_status="WARN"))
        self.assertLess(_score(ml_trace=refit), _score(ml_trace=warn))

    def test_patch_guard_refit_note_in_summary(self):
        t = _trace(guards=_guards(patch_guard_applied=True, patch_guard_status="REFIT"))
        self.assertIn("patch REFIT", _out(ml_trace=t)["pipeline_summary"])

    def test_wf_correction_small_shrink_adds(self):
        t = _trace(guards=_guards(wf_correction_applied=True, wf_shrink=0.05))
        self.assertGreater(self._diff(t), 0)
        self.assertIn("WF correction", _out(ml_trace=t)["pipeline_summary"])

    def test_high_coverage_adds(self):
        high = _trace(fusion={"coverage_eff": 0.85})
        low = _trace(fusion={"coverage_eff": 0.20})
        self.assertGreater(_score(ml_trace=high), _score(ml_trace=low))

    def test_players_adj_many_adds_more_than_few(self):
        many = _trace(players_adjustment={"enabled": True, "n_players_eff": 25})
        few = _trace(players_adjustment={"enabled": True, "n_players_eff": 3})
        self.assertGreater(_score(ml_trace=many), _score(ml_trace=few))

    def test_layer_unc_low_actionability_subtracts_vs_high(self):
        low_act = _trace(
            guards=_guards(layer_uncertainty_live_on=True, layer_uncertainty_live_applied=True),
            layer_uncertainty_live={"applied": True, "diff_pp": 1.0, "actionability_zone": "low"},
        )
        high_act = _trace(
            guards=_guards(layer_uncertainty_live_on=True, layer_uncertainty_live_applied=True),
            layer_uncertainty_live={"applied": True, "diff_pp": 1.0, "actionability_zone": "high"},
        )
        self.assertLess(_score(ml_trace=low_act), _score(ml_trace=high_act))

    def test_pipeline_adj_positive_when_areal_applied(self):
        t = _trace(guards=_guards(state_areal_live_on=True, state_areal_live_applied=True))
        out = _out(ml_trace=t)
        self.assertGreater(out["pipeline_adj"], 0.0)

    def test_pipeline_summary_length_bounded(self):
        t = _trace(guards=_guards(
            state_areal_live_on=True, state_areal_live_applied=True,
            state_alite_live_on=True, state_alite_live_applied=True,
            meta_c_lite_live_on=True, meta_c_lite_live_applied=True,
            consistency_guard_triggered=True,
            patch_guard_applied=True, patch_guard_status="WARN",
            wf_correction_applied=True, wf_shrink=0.05,
        ), meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"})
        self.assertLessEqual(len(_out(ml_trace=t)["pipeline_summary"]), 280)

    def test_bad_trace_type_does_not_crash(self):
        out = ml_confidence_from_layer_validation(
            report=_report(), ml_trace="not_a_dict",
        )
        self.assertIn("score", out)


# ---------------------------------------------------------------------------
# 8. Confidence baseline adjustment
# ---------------------------------------------------------------------------

class TestBaselineAdjustment(unittest.TestCase):

    def test_league_baseline_used_when_available(self):
        cb = {"global": {"score_bias": 5.0},
              "by_league": {"LCK": {"score_bias": 12.0, "quality_band": "A"}}}
        out = _out(league_hint="LCK", confidence_baseline=cb)
        self.assertEqual(out["baseline_source"], "liga:LCK")
        self.assertAlmostEqual(out["baseline_adj"], 12.0)

    def test_global_baseline_used_when_no_league_match(self):
        cb = {"global": {"score_bias": 5.0}, "by_league": {}}
        out = _out(league_hint="LCK", confidence_baseline=cb)
        self.assertEqual(out["baseline_source"], "global")
        self.assertAlmostEqual(out["baseline_adj"], 5.0)

    def test_positive_baseline_increases_score(self):
        cb = {"global": {"score_bias": 10.0}, "by_league": {}}
        self.assertGreater(_score(confidence_baseline=cb), _score(confidence_baseline=None))

    def test_negative_baseline_decreases_score(self):
        cb = {"global": {"score_bias": -10.0}, "by_league": {}}
        self.assertLess(_score(confidence_baseline=cb), _score(confidence_baseline=None))

    def test_baseline_adj_clamped_at_15(self):
        cb = {"global": {"score_bias": 999.0}, "by_league": {}}
        self.assertAlmostEqual(_out(confidence_baseline=cb)["baseline_adj"], 15.0)

    def test_baseline_adj_clamped_at_minus_15(self):
        cb = {"global": {"score_bias": -999.0}, "by_league": {}}
        self.assertAlmostEqual(_out(confidence_baseline=cb)["baseline_adj"], -15.0)

    def test_no_baseline_gives_zero_adj(self):
        out = _out(confidence_baseline=None)
        self.assertAlmostEqual(out["baseline_adj"], 0.0)
        self.assertIsNone(out["baseline_source"])

    def test_error_baseline_gives_zero_adj(self):
        out = _out(confidence_baseline={"error": "unavailable"})
        self.assertAlmostEqual(out["baseline_adj"], 0.0)

    def test_baseline_band_returned(self):
        cb = {"global": {"score_bias": 5.0},
              "by_league": {"LCK": {"score_bias": 3.0, "quality_band": "B+"}}}
        out = _out(league_hint="LCK", confidence_baseline=cb)
        self.assertEqual(out["baseline_band"], "B+")


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_all_none_does_not_crash(self):
        out = ml_confidence_from_layer_validation(report=None)
        self.assertIn("score", out)
        self.assertIn("grade", out)

    def test_empty_dicts_do_not_crash(self):
        out = ml_confidence_from_layer_validation(
            report={}, league_hint="", n_team_eff=float("nan"),
            ml_trace={}, confidence_baseline={},
        )
        self.assertIn("score", out)

    def test_score_always_finite_various_reports(self):
        for rep in (None, {}, _report(), {"error": "x"}):
            s = ml_confidence_from_layer_validation(report=rep)["score"]
            self.assertTrue(math.isfinite(s), f"score not finite for report={rep!r}")

    def test_pipeline_meta_method_returned_for_c_trained(self):
        t = _trace(
            guards=_guards(meta_c_lite_live_on=True, meta_c_lite_live_applied=True),
            meta_c_lite_live={"applied": True, "method": "meta_blend_c_model_v1"},
        )
        out = _out(ml_trace=t)
        self.assertIsNotNone(out["pipeline_meta_method"])
        self.assertIn("meta_blend_c_model", out["pipeline_meta_method"])

    def test_pipeline_meta_method_none_when_not_applied(self):
        out = _out(ml_trace=None)
        self.assertIsNone(out["pipeline_meta_method"])


if __name__ == "__main__":
    unittest.main()
