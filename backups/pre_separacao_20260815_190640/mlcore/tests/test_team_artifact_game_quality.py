"""Tests for mlcore/team_artifact_game_quality.py — game quality signal helpers."""
import unittest

from mlcore.team_artifact_game_quality import (
    _data_completeness_state,
    _data_completeness_weight,
    _expected_quality_for_winner,
    _norm_margin,
    _qwl_adjusted_signal,
)


# ---------------------------------------------------------------------------
# _norm_margin
# ---------------------------------------------------------------------------

class TestNormMargin(unittest.TestCase):

    def test_zero_gives_zero(self):
        self.assertAlmostEqual(_norm_margin(0.0, 100.0), 0.0)

    def test_positive_at_scale_gives_one(self):
        self.assertAlmostEqual(_norm_margin(100.0, 100.0), 1.0)

    def test_negative_at_scale_gives_minus_one(self):
        self.assertAlmostEqual(_norm_margin(-100.0, 100.0), -1.0)

    def test_exceeds_scale_clamped_to_one(self):
        self.assertAlmostEqual(_norm_margin(200.0, 100.0), 1.0)

    def test_below_neg_scale_clamped_to_minus_one(self):
        self.assertAlmostEqual(_norm_margin(-200.0, 100.0), -1.0)

    def test_half_scale(self):
        self.assertAlmostEqual(_norm_margin(50.0, 100.0), 0.5)

    def test_nan_gives_zero(self):
        self.assertAlmostEqual(_norm_margin(float("nan"), 100.0), 0.0)

    def test_zero_scale_gives_zero(self):
        self.assertAlmostEqual(_norm_margin(50.0, 0.0), 0.0)

    def test_negative_scale_gives_zero(self):
        self.assertAlmostEqual(_norm_margin(50.0, -10.0), 0.0)

    def test_invalid_value_gives_zero(self):
        self.assertAlmostEqual(_norm_margin(None, 100.0), 0.0)


# ---------------------------------------------------------------------------
# _data_completeness_state
# ---------------------------------------------------------------------------

class TestDataCompletenessState(unittest.TestCase):

    def test_complete_string(self):
        self.assertEqual(_data_completeness_state("complete"), "complete")

    def test_full_maps_to_complete(self):
        self.assertEqual(_data_completeness_state("full"), "complete")

    def test_true_maps_to_complete(self):
        self.assertEqual(_data_completeness_state("1"), "complete")

    def test_partial_string(self):
        self.assertEqual(_data_completeness_state("partial"), "partial")

    def test_false_maps_to_partial(self):
        self.assertEqual(_data_completeness_state("0"), "partial")

    def test_no_maps_to_partial(self):
        self.assertEqual(_data_completeness_state("no"), "partial")

    def test_empty_gives_unknown(self):
        self.assertEqual(_data_completeness_state(""), "unknown")

    def test_none_gives_unknown(self):
        self.assertEqual(_data_completeness_state(None), "unknown")

    def test_garbage_gives_unknown(self):
        self.assertEqual(_data_completeness_state("random_text"), "unknown")

    def test_case_insensitive(self):
        self.assertEqual(_data_completeness_state("COMPLETE"), "complete")
        self.assertEqual(_data_completeness_state("Partial"), "partial")


# ---------------------------------------------------------------------------
# _data_completeness_weight
# ---------------------------------------------------------------------------

class TestDataCompletenessWeight(unittest.TestCase):

    def test_complete_gives_one(self):
        self.assertAlmostEqual(_data_completeness_weight("complete"), 1.00)

    def test_partial_gives_0_82(self):
        self.assertAlmostEqual(_data_completeness_weight("partial"), 0.82)

    def test_unknown_gives_0_92(self):
        self.assertAlmostEqual(_data_completeness_weight("unknown"), 0.92)

    def test_empty_gives_unknown_weight(self):
        self.assertAlmostEqual(_data_completeness_weight(""), 0.92)

    def test_complete_highest(self):
        self.assertGreater(_data_completeness_weight("complete"), _data_completeness_weight("unknown"))
        self.assertGreater(_data_completeness_weight("unknown"), _data_completeness_weight("partial"))


# ---------------------------------------------------------------------------
# _expected_quality_for_winner
# ---------------------------------------------------------------------------

class TestExpectedQualityForWinner(unittest.TestCase):

    def test_half_probability_gives_base(self):
        result = _expected_quality_for_winner(0.5, base=0.3, slope=0.4)
        self.assertAlmostEqual(result, 0.3, places=5)

    def test_high_probability_increases_expected(self):
        result_high = _expected_quality_for_winner(0.8, base=0.3, slope=0.4)
        result_mid = _expected_quality_for_winner(0.5, base=0.3, slope=0.4)
        self.assertGreater(result_high, result_mid)

    def test_result_clamped_below_0_95(self):
        result = _expected_quality_for_winner(1.0, base=0.8, slope=0.5)
        self.assertLessEqual(result, 0.95)

    def test_result_clamped_above_zero(self):
        result = _expected_quality_for_winner(0.0, base=0.0, slope=0.0)
        self.assertGreaterEqual(result, 0.0)

    def test_clipped_p_at_one(self):
        result = _expected_quality_for_winner(2.0, base=0.3, slope=0.4)
        self.assertLessEqual(result, 0.95)

    def test_favorite_wins_expected_more_decisive(self):
        low = _expected_quality_for_winner(0.55, base=0.3, slope=0.4)
        high = _expected_quality_for_winner(0.80, base=0.3, slope=0.4)
        self.assertLess(low, high)


# ---------------------------------------------------------------------------
# _qwl_adjusted_signal
# ---------------------------------------------------------------------------

class TestQwlAdjustedSignal(unittest.TestCase):

    def _call(self, q_pos=0.7, p_winner=0.65, surplus_mix=0.5, opp_ctx=0.0):
        return _qwl_adjusted_signal(
            q_pos, p_winner,
            expected_base=0.3, expected_slope=0.4,
            surplus_mix=surplus_mix,
            opponent_context_mix=opp_ctx,
        )

    def test_returns_four_values(self):
        result = self._call()
        self.assertEqual(len(result), 4)

    def test_q_adjusted_in_unit_interval(self):
        q_adj, _, _, _ = self._call()
        self.assertGreaterEqual(q_adj, 0.0)
        self.assertLessEqual(q_adj, 1.0)

    def test_q_expected_in_unit_interval(self):
        _, q_exp, _, _ = self._call()
        self.assertGreaterEqual(q_exp, 0.0)
        self.assertLessEqual(q_exp, 0.95)

    def test_surplus_non_negative(self):
        _, _, q_surplus, _ = self._call()
        self.assertGreaterEqual(q_surplus, 0.0)

    def test_zero_surplus_mix_returns_q_pos(self):
        q_adj, _, _, _ = self._call(q_pos=0.7, surplus_mix=0.0)
        self.assertAlmostEqual(q_adj, 0.7, places=5)

    def test_ctx_mult_clamped(self):
        _, _, _, ctx = self._call(p_winner=0.3, opp_ctx=0.5)
        self.assertGreaterEqual(ctx, 0.60)
        self.assertLessEqual(ctx, 1.40)

    def test_underdog_win_ctx_above_one(self):
        # p_winner=0.3 → underdog winning → ctx_mult > 1.0 with positive opp_ctx
        _, _, _, ctx = self._call(p_winner=0.3, opp_ctx=0.5)
        self.assertGreater(ctx, 1.0)

    def test_favorite_win_ctx_below_one(self):
        # p_winner=0.8 → favorite winning → ctx_mult < 1.0 with positive opp_ctx
        _, _, _, ctx = self._call(p_winner=0.8, opp_ctx=0.5)
        self.assertLess(ctx, 1.0)

    def test_high_quality_gives_positive_surplus_when_above_expected(self):
        q_pos = 0.95
        _q_adj, _q_exp, q_surplus, _ = self._call(q_pos=q_pos, p_winner=0.6, surplus_mix=0.5)
        self.assertGreater(q_surplus, 0.0)

    def test_zero_quality_gives_zero_surplus(self):
        _q_adj, _q_exp, q_surplus, _ = self._call(q_pos=0.0, surplus_mix=0.5)
        self.assertAlmostEqual(q_surplus, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
