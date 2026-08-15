"""End-to-end tests for confidence governor budget accounting.

Covers:
- Stage cap (62.5 % rule): state overrides cannot monopolise the budget
- Sequential budget accumulation across multiple stages
- read_governor_budget_summary correctness
- Budget exhaustion detection
- Moves AWAY from 50 % are free (no budget consumed)
- Disabled governor passes everything through unchanged
"""

import unittest
from typing import Any, Dict

from app_core.confidence_governor_core import (
    apply_confidence_governor,
    read_governor_budget_summary,
)


def _fresh_ss(**extra) -> Dict[str, Any]:
    ss: Dict[str, Any] = {}
    ss.update(extra)
    return ss


class TestApplyConfidenceGovernor(unittest.TestCase):
    # ------------------------------------------------------------------
    # Basic shrink accounting
    # ------------------------------------------------------------------

    def test_no_shrink_consumed_when_candidate_equals_prior(self):
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.65,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertAlmostEqual(out["requested_shrink_pp"], 0.0, places=6)
        self.assertAlmostEqual(out["applied_shrink_pp"], 0.0, places=6)

    def test_move_away_from_50_consumes_no_budget(self):
        """Increasing edge (moving AWAY from 50 %) should use 0 pp budget."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.60, p_candidate=0.70,  # edge grows
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertAlmostEqual(out["requested_shrink_pp"], 0.0, places=6)
        self.assertAlmostEqual(out["applied_shrink_pp"], 0.0, places=6)
        self.assertAlmostEqual(out["p_out"], 0.70, places=4)

    def test_shrink_toward_50_consumes_budget(self):
        """5pp shrink (0.65→0.60) should consume 5pp and pass through."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertAlmostEqual(out["requested_shrink_pp"], 5.0, places=4)
        self.assertAlmostEqual(out["applied_shrink_pp"], 5.0, places=4)
        self.assertAlmostEqual(out["p_out"], 0.60, places=4)
        self.assertFalse(out["limited"])

    def test_p_out_written_to_session_state(self):
        """Governor must persist state in session_state for sequential calls."""
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertIn("_ml_confidence_governor", ss)
        gov_state = ss["_ml_confidence_governor"]
        self.assertAlmostEqual(gov_state["used_shrink_pp"], 5.0, places=4)

    # ------------------------------------------------------------------
    # Budget cap / limiting
    # ------------------------------------------------------------------

    def test_excess_shrink_is_limited(self):
        """10pp request with 8pp budget: limited=True, only 8pp applied."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.60, p_candidate=0.50,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        # 0.60 → edge=10pp; 0.50 → edge=0pp; request=10pp (float may be ~9.9999...)
        self.assertAlmostEqual(out["requested_shrink_pp"], 10.0, places=4)
        self.assertAlmostEqual(out["applied_shrink_pp"], 8.0, places=4)
        self.assertTrue(out["limited"])
        p_out = out["p_out"]
        # Edge should be reduced by 8pp: 10pp → 2pp → p = 0.52
        self.assertAlmostEqual(p_out, 0.52, delta=0.005)

    def test_stage_cap_pp_limits_single_stage(self):
        """stage_cap_pp=3 allows at most 3pp even when budget has 8pp."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.50,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
            stage_cap_pp=3.0,
        )
        self.assertAlmostEqual(out["applied_shrink_pp"], 3.0, places=4)
        self.assertTrue(out["limited"])

    def test_stage_cap_does_not_add_to_budget(self):
        """stage_cap_pp is an additional constraint, not extra budget."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.55, p_candidate=0.52,
            session_state=ss, enabled=True, max_total_shrink_pp=1.0,
            stage_cap_pp=5.0,  # larger than budget — budget wins
        )
        # request = 3pp, budget = 1pp, cap = 5pp → allowed = min(1, 5) = 1pp
        self.assertAlmostEqual(out["applied_shrink_pp"], 1.0, places=4)
        self.assertTrue(out["limited"])

    # ------------------------------------------------------------------
    # Sequential multi-stage accounting
    # ------------------------------------------------------------------

    def test_two_stages_accumulate_budget(self):
        """Second stage sees reduced remaining budget from first stage."""
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,  # uses 5pp
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        out2 = apply_confidence_governor(
            stage="s2", p_before=0.60, p_candidate=0.50,  # requests 10pp
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        # Only 3pp remaining → limited
        self.assertTrue(out2["limited"])
        self.assertAlmostEqual(out2["applied_shrink_pp"], 3.0, places=4)

    def test_state_override_stage_cap_reserves_budget_for_marginal(self):
        """Orchestrator guarantees marginal_trust_guard has at least 37.5 % of budget.

        alite and areal are mutually exclusive in the orchestrator. Each is
        capped at 62.5 % of the total budget (5pp on an 8pp budget). After
        one state override stage, at least 3pp remain for marginal_trust_guard.
        """
        ss = _fresh_ss()
        cap_pp = 8.0 * 0.625  # = 5.0

        # One state override stage (e.g. alite) requests all 8pp but is capped at 5pp.
        out_state = apply_confidence_governor(
            stage="state_alite_live_override",
            p_before=0.58, p_candidate=0.50,  # 8pp request
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
            stage_cap_pp=cap_pp,
        )
        self.assertTrue(out_state["limited"])
        self.assertAlmostEqual(out_state["applied_shrink_pp"], cap_pp, places=4)

        # marginal_trust_guard runs without cap — still has 3pp remaining
        out_mg = apply_confidence_governor(
            stage="marginal_trust_guard",
            p_before=0.55, p_candidate=0.50,  # 5pp request
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        remaining = 8.0 - cap_pp  # 3pp
        self.assertAlmostEqual(out_mg["applied_shrink_pp"], remaining, places=4)

    # ------------------------------------------------------------------
    # Disabled governor
    # ------------------------------------------------------------------

    def test_disabled_governor_passes_through(self):
        """When enabled=False, p_out = p_candidate regardless of budget."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.50,
            session_state=ss, enabled=False, max_total_shrink_pp=8.0,
        )
        self.assertAlmostEqual(out["p_out"], 0.50, places=4)
        self.assertFalse(out["limited"])

    # ------------------------------------------------------------------
    # Invalid inputs
    # ------------------------------------------------------------------

    def test_nan_inputs_return_gracefully(self):
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=float("nan"), p_candidate=float("nan"),
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertIn("reason", out)
        self.assertEqual(out.get("reason"), "invalid_inputs")

    def test_none_session_state_does_not_crash(self):
        out = apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=None, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertIn("p_out", out)
        self.assertAlmostEqual(out["p_out"], 0.60, places=4)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def test_reset_clears_state(self):
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
            reset=True,
        )
        gov = ss.get("_ml_confidence_governor", {})
        self.assertAlmostEqual(gov.get("used_shrink_pp", -1.0), 0.0, places=6)

    # ------------------------------------------------------------------
    # Symmetry: below 50 % works the same as above 50 %
    # ------------------------------------------------------------------

    def test_below_50_symmetry(self):
        """p_before=0.35, p_candidate=0.40: shrink is 5pp, same as 0.65→0.60."""
        ss = _fresh_ss()
        out = apply_confidence_governor(
            stage="s1", p_before=0.35, p_candidate=0.40,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        self.assertAlmostEqual(out["requested_shrink_pp"], 5.0, places=4)
        self.assertAlmostEqual(out["applied_shrink_pp"], 5.0, places=4)
        self.assertAlmostEqual(out["p_out"], 0.40, places=4)


class TestReadGovernorBudgetSummary(unittest.TestCase):

    def test_empty_state_returns_not_available(self):
        summary = read_governor_budget_summary({})
        self.assertFalse(summary["available"])
        self.assertFalse(summary["budget_exhausted"])

    def test_none_session_state(self):
        summary = read_governor_budget_summary(None)
        self.assertFalse(summary["available"])

    def test_summary_after_single_stage(self):
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        summary = read_governor_budget_summary(ss)
        self.assertTrue(summary["available"])
        self.assertTrue(summary["enabled"])
        self.assertAlmostEqual(summary["used_shrink_pp"], 5.0, places=4)
        self.assertAlmostEqual(summary["remaining_shrink_pp"], 3.0, places=4)
        self.assertFalse(summary["budget_exhausted"])
        self.assertEqual(summary["n_stages_applied"], 1)
        self.assertEqual(summary["n_stages_limited"], 0)

    def test_budget_exhausted_flag(self):
        ss = _fresh_ss()
        # Use full 8pp in one shot (p_before=0.58, p_candidate=0.50 → 8pp)
        apply_confidence_governor(
            stage="s1", p_before=0.58, p_candidate=0.50,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        summary = read_governor_budget_summary(ss)
        self.assertTrue(summary["budget_exhausted"])

    def test_summary_reflects_multiple_stages(self):
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.62,  # 3pp
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        apply_confidence_governor(
            stage="s2", p_before=0.62, p_candidate=0.60,  # 2pp
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        summary = read_governor_budget_summary(ss)
        self.assertAlmostEqual(summary["used_shrink_pp"], 5.0, places=4)
        self.assertEqual(summary["n_stages_applied"], 2)

    def test_n_stages_limited_counts_correctly(self):
        ss = _fresh_ss()
        # Stage 1: uses 7pp (not limited)
        apply_confidence_governor(
            stage="s1", p_before=0.57, p_candidate=0.50,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        # Stage 2: requests 7pp but only 1pp left → limited
        apply_confidence_governor(
            stage="s2", p_before=0.57, p_candidate=0.50,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        summary = read_governor_budget_summary(ss)
        self.assertEqual(summary["n_stages_limited"], 1)

    def test_summary_does_not_mutate_state(self):
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        used_before = ss["_ml_confidence_governor"]["used_shrink_pp"]
        read_governor_budget_summary(ss)
        used_after = ss["_ml_confidence_governor"]["used_shrink_pp"]
        self.assertAlmostEqual(used_before, used_after, places=9)

    def test_required_keys_present(self):
        ss = _fresh_ss()
        apply_confidence_governor(
            stage="s1", p_before=0.65, p_candidate=0.60,
            session_state=ss, enabled=True, max_total_shrink_pp=8.0,
        )
        summary = read_governor_budget_summary(ss)
        for key in ("available", "enabled", "max_total_shrink_pp",
                    "used_shrink_pp", "remaining_shrink_pp",
                    "budget_exhausted", "n_stages_applied",
                    "n_stages_limited", "stages", "state_key"):
            self.assertIn(key, summary, f"Missing key: {key}")


if __name__ == "__main__":
    unittest.main()
