"""Tests for mlcore/elo.py — EloModel and EloParams."""
import unittest

import pandas as pd

from mlcore.elo import EloModel, EloParams, _p_from_diff_points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(**kwargs) -> EloModel:
    """Fresh EloModel with no L2/priors by default for clean math."""
    defaults = dict(
        l2_team=0.0, l2_league=0.0, l2_tier=0.0, l2_side=0.0,
        tier_prior_weight=0.0, league_prior_weight=0.0,
        league_strength_cap_abs=0.0, tier_strength_cap_abs=0.0,
    )
    defaults.update(kwargs)
    return EloModel(EloParams(**defaults))


def _fit_one(model: EloModel, blue: str, red: str, blue_win: int,
             blue_league: str = "LCK", red_league: str = "LCK") -> None:
    model.update(blue, red, blue_win, blue_league=blue_league, red_league=red_league)


# ---------------------------------------------------------------------------
# _p_from_diff_points (module-level helper)
# ---------------------------------------------------------------------------

class TestPFromDiffPoints(unittest.TestCase):

    def test_zero_diff_returns_half(self):
        self.assertAlmostEqual(_p_from_diff_points(0.0, 400.0), 0.5)

    def test_positive_diff_above_half(self):
        p = _p_from_diff_points(200.0, 400.0)
        self.assertGreater(p, 0.5)

    def test_negative_diff_below_half(self):
        p = _p_from_diff_points(-200.0, 400.0)
        self.assertLess(p, 0.5)

    def test_antisymmetric(self):
        p_pos = _p_from_diff_points(100.0, 400.0)
        p_neg = _p_from_diff_points(-100.0, 400.0)
        self.assertAlmostEqual(p_pos + p_neg, 1.0, places=10)

    def test_large_positive_near_1(self):
        p = _p_from_diff_points(2000.0, 400.0)
        self.assertGreater(p, 0.99)

    def test_large_negative_near_0(self):
        p = _p_from_diff_points(-2000.0, 400.0)
        self.assertLess(p, 0.01)


# ---------------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------------

class TestBasicAccessors(unittest.TestCase):

    def setUp(self):
        self.model = _model()

    def test_rating_raw_defaults_to_init_rating(self):
        self.assertAlmostEqual(self.model.rating_raw("T1"), 0.0)

    def test_n_games_defaults_to_zero(self):
        self.assertEqual(self.model.n_games("T1"), 0)

    def test_meta_defaults_to_other(self):
        m = self.model.meta("Unknown")
        self.assertEqual(m["league"], "OTHER")
        self.assertEqual(m["tier"], "OTHER")

    def test_residual_eff_zero_gp_returns_zero(self):
        # shrink = 0/(0+alpha) = 0 → residual_eff = rating * 0 = 0
        self.assertAlmostEqual(self.model.residual_eff("T1"), 0.0)

    def test_residual_eff_large_gp_approaches_rating(self):
        m = _model(alpha=80.0)
        m.ratings["T1"] = 100.0
        m.games_played["T1"] = 10_000  # shrink = 10000/10080 ≈ 0.992
        self.assertGreater(m.residual_eff("T1"), 99.0)

    def test_residual_eff_at_alpha_is_half(self):
        m = _model(alpha=80.0)
        m.ratings["T1"] = 1.0
        m.games_played["T1"] = 80  # n == alpha → shrink = 0.5
        self.assertAlmostEqual(m.residual_eff("T1"), 0.5, places=6)


# ---------------------------------------------------------------------------
# p_blue — basic probability
# ---------------------------------------------------------------------------

class TestPBlue(unittest.TestCase):

    def setUp(self):
        self.model = _model()

    def test_equal_teams_returns_half(self):
        p = self.model.p_blue("T1", "Gen.G")
        self.assertAlmostEqual(p, 0.5)

    def test_stronger_blue_returns_above_half(self):
        m = _model(alpha=0.0)  # alpha=0 → no shrink
        m.ratings["Blue"] = 50.0
        m.ratings["Red"] = -50.0
        p = m.p_blue("Blue", "Red")
        self.assertGreater(p, 0.5)

    def test_stronger_red_returns_below_half(self):
        m = _model(alpha=0.0)
        m.ratings["Blue"] = -50.0
        m.ratings["Red"] = 50.0
        p = m.p_blue("Blue", "Red")
        self.assertLess(p, 0.5)

    def test_side_adv_shifts_probability(self):
        m = _model(alpha=0.0)
        p_no_adv = m.p_blue("A", "B")
        m.side_adv_points = 30.0
        p_with_adv = m.p_blue("A", "B")
        self.assertGreater(p_with_adv, p_no_adv)

    def test_p_in_0_1(self):
        m = _model(alpha=0.0)
        m.ratings["A"] = 999.0
        m.ratings["B"] = -999.0
        p = m.p_blue("A", "B")
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)


# ---------------------------------------------------------------------------
# update — ratings change correctly
# ---------------------------------------------------------------------------

class TestUpdate(unittest.TestCase):

    def test_blue_win_increases_blue_rating(self):
        m = _model()
        r_before = m.rating_raw("T1")
        m.update("T1", "Gen.G", 1, blue_league="LCK", red_league="LCK")
        self.assertGreater(m.rating_raw("T1"), r_before)

    def test_blue_win_decreases_red_rating(self):
        m = _model()
        r_before = m.rating_raw("Gen.G")
        m.update("T1", "Gen.G", 1, blue_league="LCK", red_league="LCK")
        self.assertLess(m.rating_raw("Gen.G"), r_before)

    def test_red_win_decreases_blue_rating(self):
        m = _model()
        m.update("T1", "Gen.G", 0, blue_league="LCK", red_league="LCK")
        self.assertLess(m.rating_raw("T1"), 0.0)

    def test_games_played_increments(self):
        m = _model()
        m.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        self.assertEqual(m.n_games("A"), 1)
        self.assertEqual(m.n_games("B"), 1)

    def test_update_returns_predicted_p(self):
        m = _model()
        p = m.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_same_league_no_league_strength_update(self):
        m = _model()
        m.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        # Same league → league_strength should NOT be updated
        # LCK key may be set to 0.0 on initialization but should not diverge
        lck_val = m.league_strength.get("LCK", 0.0)
        self.assertAlmostEqual(lck_val, 0.0, places=6)

    def test_diff_leagues_updates_league_strength(self):
        m = _model(k_league=6.0)
        m.update("T1", "Cloud9", 1, blue_league="LCK", red_league="LCS")
        lck = m.league_strength.get("LCK", 0.0)
        lcs = m.league_strength.get("LCS", 0.0)
        # Blue (LCK) won → LCK strength increases, LCS decreases
        self.assertGreater(lck, 0.0)
        self.assertLess(lcs, 0.0)

    def test_same_tier_no_tier_strength_update(self):
        m = _model(k_tier=3.0)
        # Both T1 leagues
        m.update("T1", "JDG", 1, blue_league="LCK", red_league="LPL",
                 blue_tier="T1", red_tier="T1")
        t1_val = m.tier_strength.get("T1", 0.0)
        self.assertAlmostEqual(t1_val, 0.0, places=6)

    def test_diff_tiers_updates_tier_strength(self):
        m = _model(k_tier=3.0)
        m.update("T1", "FlyQuest", 1, blue_league="LCK", red_league="LCS",
                 blue_tier="T1", red_tier="T2")
        t1 = m.tier_strength.get("T1", 0.0)
        t2 = m.tier_strength.get("T2", 0.0)
        self.assertGreater(t1, 0.0)
        self.assertLess(t2, 0.0)

    def test_k_decay_reduces_update_after_many_games(self):
        m_fresh = _model(alpha=80.0)
        m_veteran = _model(alpha=80.0)
        # Veteran has many games → k_decay is low → smaller update
        m_veteran.games_played["A"] = 10_000
        m_veteran.games_played["B"] = 10_000
        m_fresh.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        m_veteran.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        self.assertGreater(abs(m_fresh.rating_raw("A")), abs(m_veteran.rating_raw("A")))

    def test_strong_favourite_smaller_update_than_equal(self):
        # Favourite winning is expected → smaller rating gain than equal teams
        m_fav = _model(alpha=0.0)
        m_fav.ratings["Favourite"] = 200.0
        m_fav.ratings["Underdog"] = -200.0
        m_fav.update("Favourite", "Underdog", 1, blue_league="LCK", red_league="LCK")
        fav_gain = m_fav.rating_raw("Favourite") - 200.0

        m_eq = _model(alpha=0.0)
        m_eq.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        eq_gain = m_eq.rating_raw("A")  # started at 0.0

        self.assertLess(fav_gain, eq_gain)

    def test_upset_gives_large_update(self):
        m = _model(alpha=0.0)
        m.ratings["Favourite"] = 200.0
        m.ratings["Underdog"] = -200.0
        before = m.rating_raw("Underdog")
        # Underdog (red) wins
        m.update("Favourite", "Underdog", 0, blue_league="LCK", red_league="LCK")
        after = m.rating_raw("Underdog")
        # Upset → large positive update for underdog
        self.assertGreater(after - before, 5.0)


# ---------------------------------------------------------------------------
# base_strength and components
# ---------------------------------------------------------------------------

class TestBaseStrengthAndComponents(unittest.TestCase):

    def test_base_strength_fresh_team_is_zero(self):
        m = _model()
        self.assertAlmostEqual(m.base_strength("New"), 0.0)

    def test_base_strength_after_wins(self):
        m = _model(alpha=0.0)
        m.ratings["A"] = 50.0
        self.assertGreater(m.base_strength("A"), 0.0)

    def test_components_keys_present(self):
        m = _model()
        c = m.components("T1")
        expected = {
            "team", "league", "tier", "residual_raw", "games_played",
            "shrink_factor", "residual_eff", "league_strength", "tier_strength",
            "base_strength", "base_strength_uncapped",
        }
        for k in expected:
            self.assertIn(k, c)

    def test_components_base_strength_matches_base_strength_method(self):
        m = _model(alpha=0.0)
        m.ratings["T1"] = 30.0
        m.games_played["T1"] = 5
        c = m.components("T1")
        self.assertAlmostEqual(c["base_strength"], m.base_strength("T1"), places=6)

    def test_rank_caps_not_applied_by_default(self):
        m = _model(alpha=0.0)
        m.ratings["A"] = 1000.0
        m.rank_caps_by_tier["T1"] = 10.0
        # Without apply_rank_caps → cap not applied
        bs = m.base_strength("A", tier="T1", apply_rank_caps=False)
        self.assertGreater(bs, 10.0)

    def test_rank_caps_applied_when_flag_set(self):
        m = _model(alpha=0.0)
        m.ratings["A"] = 1000.0
        m.rank_caps_by_tier["T1"] = 10.0
        bs = m.base_strength("A", tier="T1", apply_rank_caps=True)
        self.assertAlmostEqual(bs, 10.0)


# ---------------------------------------------------------------------------
# fit — sequential DataFrame
# ---------------------------------------------------------------------------

class TestFit(unittest.TestCase):

    def _games_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"blue": "A", "red": "B", "blue_win": 1, "blue_league": "LCK", "red_league": "LCK"},
            {"blue": "B", "red": "A", "blue_win": 0, "blue_league": "LCK", "red_league": "LCK"},
            {"blue": "A", "red": "B", "blue_win": 1, "blue_league": "LCK", "red_league": "LCK"},
        ])

    def test_fit_increments_games(self):
        m = _model()
        m.fit(self._games_df())
        self.assertEqual(m.n_games("A"), 3)
        self.assertEqual(m.n_games("B"), 3)

    def test_fit_a_wins_more_a_stronger(self):
        m = _model(alpha=0.0)
        m.fit(self._games_df())
        self.assertGreater(m.rating_raw("A"), m.rating_raw("B"))

    def test_fit_with_legacy_league_column(self):
        df = pd.DataFrame([
            {"blue": "X", "red": "Y", "blue_win": 1, "league": "LPL"},
        ])
        m = _model()
        m.fit(df)  # should not crash
        self.assertEqual(m.n_games("X"), 1)

    def test_fit_interleague_updates_league_strength(self):
        m = _model(k_league=6.0)
        df = pd.DataFrame([
            {"blue": "T1", "red": "C9", "blue_win": 1, "blue_league": "LCK", "red_league": "LCS"},
            {"blue": "T1", "red": "C9", "blue_win": 1, "blue_league": "LCK", "red_league": "LCS"},
        ])
        m.fit(df)
        self.assertGreater(m.league_strength.get("LCK", 0.0), 0.0)
        self.assertLess(m.league_strength.get("LCS", 0.0), 0.0)


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------

class TestSerialisationRoundTrip(unittest.TestCase):

    def test_round_trip_preserves_ratings(self):
        m = _model()
        m.ratings["T1"] = 42.5
        m.games_played["T1"] = 10
        d = m.to_dict()
        m2 = EloModel.from_dict(d)
        self.assertAlmostEqual(m2.rating_raw("T1"), 42.5)
        self.assertEqual(m2.n_games("T1"), 10)

    def test_round_trip_preserves_league_strength_relative(self):
        # from_dict calls _center_offsets() which centers to mean=0.
        # Relative differences are preserved even if absolute values shift.
        m = _model()
        m.league_strength["LCK"] = 15.0
        m.league_strength["LCS"] = -5.0
        d = m.to_dict()
        m2 = EloModel.from_dict(d)
        diff_before = 15.0 - (-5.0)  # 20pp
        diff_after = m2.league_strength.get("LCK", 0.0) - m2.league_strength.get("LCS", 0.0)
        self.assertAlmostEqual(diff_after, diff_before, places=4)

    def test_round_trip_preserves_params_k(self):
        m = EloModel(EloParams(k=18.0))
        d = m.to_dict()
        m2 = EloModel.from_dict(d)
        self.assertAlmostEqual(m2.params.k, 18.0)

    def test_from_dict_empty_returns_fresh_model(self):
        m = EloModel.from_dict({})
        self.assertAlmostEqual(m.rating_raw("T1"), 0.0)

    def test_to_dict_has_required_keys(self):
        m = _model()
        d = m.to_dict()
        for k in ("params", "ratings", "games_played", "league_strength", "tier_strength"):
            self.assertIn(k, d)

    def test_round_trip_side_adv(self):
        m = _model()
        m.side_adv_points = 8.5
        d = m.to_dict()
        m2 = EloModel.from_dict(d)
        self.assertAlmostEqual(m2.side_adv_points, 8.5)


# ---------------------------------------------------------------------------
# set_team_meta
# ---------------------------------------------------------------------------

class TestSetTeamMeta(unittest.TestCase):

    def test_meta_lookup_uses_set_meta(self):
        m = _model()
        m.set_team_meta({"T1": {"league": "LCK", "tier": "T1"}})
        meta = m.meta("T1")
        self.assertEqual(meta["league"], "LCK")
        self.assertEqual(meta["tier"], "T1")

    def test_missing_team_meta_still_returns_other(self):
        m = _model()
        m.set_team_meta({"T1": {"league": "LCK", "tier": "T1"}})
        self.assertEqual(m.meta("UnknownTeam")["tier"], "OTHER")


# ---------------------------------------------------------------------------
# Dynamic K (disabled by default, enabled via flag)
# ---------------------------------------------------------------------------

class TestDynamicK(unittest.TestCase):

    def test_dynamic_k_disabled_no_effect(self):
        m = _model(dynamic_k_enabled=False)
        # Should behave exactly like static k
        m.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        self.assertEqual(m.n_games("A"), 1)

    def test_dynamic_k_enabled_fresh_team_larger_update(self):
        m_static = _model(dynamic_k_enabled=False, alpha=0.0)
        m_dynamic = _model(dynamic_k_enabled=True, alpha=0.0,
                           dynamic_k_uncertainty_mult_max=1.5,
                           dynamic_k_games_prior=30.0)
        m_static.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        m_dynamic.update("A", "B", 1, blue_league="LCK", red_league="LCK")
        # Dynamic K with uncertainty multiplier → larger update for fresh teams
        self.assertGreaterEqual(abs(m_dynamic.rating_raw("A")), abs(m_static.rating_raw("A")))


if __name__ == "__main__":
    unittest.main()
