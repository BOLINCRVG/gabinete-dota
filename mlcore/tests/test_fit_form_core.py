"""Tests for mlcore/fit_form_core.py — form blend helper."""
import math
import unittest

from mlcore.fit_samples_core import Sample
from mlcore.fit_form_core import _apply_form_blend_on_sample


def _make_sample(
    *,
    form_n_a: int = 10,
    form_n_b: int = 10,
    form_wins_a: int = 7,
    form_wins_b: int = 3,
    same_league: bool = True,
) -> Sample:
    return Sample(
        month="2025-03",
        gameid="test001",
        league="LCK",
        patch="14.1",
        split="Spring",
        macro="KR",
        side="blue",
        bo_bucket="bo1",
        phase="regular",
        y=1,
        p_team=0.6,
        p_players=0.55,
        n_team=500.0,
        n_players=300.0,
        form_n_a=form_n_a,
        form_n_b=form_n_b,
        form_wins_a=form_wins_a,
        form_wins_b=form_wins_b,
        same_league=same_league,
    )


class TestApplyFormBlendOnSample(unittest.TestCase):

    def test_disabled_returns_p_model(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6,
            sample=s,
            enabled=False,
            max_weight=0.5,
            knee_games=6.0,
            beta_prior=2.0,
            signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.6, places=5)

    def test_returns_float(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertIsInstance(result, float)

    def test_result_strictly_between_0_and_1(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_result_is_finite(self):
        s = _make_sample()
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertTrue(math.isfinite(result))

    def test_strong_form_a_shifts_up(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=9, form_wins_b=1)
        result = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.75, knee_games=4.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertGreater(result, 0.5)

    def test_strong_form_b_shifts_down(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=1, form_wins_b=9)
        result = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.75, knee_games=4.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertLess(result, 0.5)

    def test_zero_form_games_returns_p_model(self):
        s = _make_sample(form_n_a=0, form_n_b=0, form_wins_a=0, form_wins_b=0)
        result = _apply_form_blend_on_sample(
            p_model=0.65, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.65, places=5)

    def test_equal_form_stays_near_p_model(self):
        s = _make_sample(form_n_a=10, form_n_b=10, form_wins_a=5, form_wins_b=5)
        result = _apply_form_blend_on_sample(
            p_model=0.6, sample=s, enabled=True,
            max_weight=0.5, knee_games=6.0, beta_prior=2.0, signal_power=1.0,
        )
        self.assertAlmostEqual(result, 0.6, places=2)

    def test_high_max_weight_allows_larger_shift(self):
        s = _make_sample(form_n_a=20, form_n_b=20, form_wins_a=18, form_wins_b=2)
        r_low = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.1, knee_games=2.0, beta_prior=1.5, signal_power=1.0,
        )
        r_high = _apply_form_blend_on_sample(
            p_model=0.5, sample=s, enabled=True,
            max_weight=0.9, knee_games=2.0, beta_prior=1.5, signal_power=1.0,
        )
        self.assertGreater(r_high, r_low)


if __name__ == "__main__":
    unittest.main()
