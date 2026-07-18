"""Tests for mlcore/players_layer.py — norm_pos, PlayersArtifact, compute_delta_and_confidence."""
import unittest
from datetime import date

from mlcore.players_layer import (
    PlayersArtifact,
    PlayerState,
    POS_CANON,
    compute_delta_and_confidence,
    lineup_for_game,
    norm_pos,
    _interleague_pair_scale,
    _lane_weight_for,
    _league_weight_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _art(**kwargs) -> PlayersArtifact:
    """Fresh artifact with optional overrides."""
    return PlayersArtifact(**kwargs)


def _art_with_player(player: str, pos: str, r: float, gp: int = 20, league: str = "LCK") -> PlayersArtifact:
    art = _art()
    key = f"{player}|||{pos}"
    art.players[key] = PlayerState(r=r, gp=gp, last_league=league, last_tier="T1")
    return art


_BLUE = {"Top": "p1", "Jungle": "p2", "Mid": "p3", "Bot": "p4", "Support": "p5"}
_RED = {"Top": "p6", "Jungle": "p7", "Mid": "p8", "Bot": "p9", "Support": "p10"}


# ---------------------------------------------------------------------------
# norm_pos
# ---------------------------------------------------------------------------

class TestNormPos(unittest.TestCase):

    def test_top_aliases(self):
        for alias in ("top", "Top", "TOP", "t"):
            self.assertEqual(norm_pos(alias), "Top")

    def test_jungle_aliases(self):
        for alias in ("jng", "jg", "jungle", "j", "Jungle"):
            self.assertEqual(norm_pos(alias), "Jungle")

    def test_mid_aliases(self):
        for alias in ("mid", "Mid", "MID", "m"):
            self.assertEqual(norm_pos(alias), "Mid")

    def test_bot_aliases(self):
        for alias in ("bot", "adc", "bottom", "b", "Bot", "ADC"):
            self.assertEqual(norm_pos(alias), "Bot")

    def test_support_aliases(self):
        for alias in ("sup", "support", "s", "Support", "SUP"):
            self.assertEqual(norm_pos(alias), "Support")

    def test_unknown_returns_empty(self):
        self.assertEqual(norm_pos("carry"), "")
        self.assertEqual(norm_pos("fill"), "")

    def test_none_returns_empty(self):
        self.assertEqual(norm_pos(None), "")

    def test_empty_string_returns_empty(self):
        self.assertEqual(norm_pos(""), "")


# ---------------------------------------------------------------------------
# _interleague_pair_scale
# ---------------------------------------------------------------------------

class TestInterleaguePairScale(unittest.TestCase):

    def test_same_macro_returns_1(self):
        self.assertEqual(_interleague_pair_scale("LCK", "LCK"), 1.0)

    def test_same_region_different_leagues_returns_1(self):
        # Both map to KR macro
        self.assertAlmostEqual(_interleague_pair_scale("LCK", "LCK Challengers"), 1.0, places=5)

    def test_cn_kr_high_base(self):
        # CN/KR is the highest cross-macro pair
        scale = _interleague_pair_scale("LPL", "LCK")
        self.assertGreater(scale, 0.92)

    def test_other_pair_lower_base(self):
        scale_other = _interleague_pair_scale("SL (LatAm)", "VCS")
        scale_cn_kr = _interleague_pair_scale("LPL", "LCK")
        self.assertLess(scale_other, scale_cn_kr)

    def test_gp_support_increases_scale(self):
        # Same pair, more GP → higher scale (convergence toward 1.0)
        s_low = _interleague_pair_scale("LPL", "LEC", gp=5)
        s_high = _interleague_pair_scale("LPL", "LEC", gp=200)
        self.assertGreater(s_high, s_low)

    def test_scale_clamped_below_086(self):
        scale = _interleague_pair_scale("SL (LatAm)", "VCS", gp=0)
        self.assertGreaterEqual(scale, 0.86)

    def test_scale_clamped_above_0975(self):
        scale = _interleague_pair_scale("LPL", "LCK", gp=5000, last_tier="T1", cur_tier="T1")
        self.assertLessEqual(scale, 0.975)

    def test_same_tier_bonus_applied(self):
        s_no_tier = _interleague_pair_scale("LPL", "LEC", gp=50)
        s_tier = _interleague_pair_scale("LPL", "LEC", gp=50, last_tier="T1", cur_tier="T1")
        self.assertGreaterEqual(s_tier, s_no_tier)


# ---------------------------------------------------------------------------
# _lane_weight_for / _league_weight_for
# ---------------------------------------------------------------------------

class TestWeightHelpers(unittest.TestCase):

    def test_lane_weight_no_dict_returns_1(self):
        self.assertEqual(_lane_weight_for("Mid", None), 1.0)

    def test_lane_weight_known_key(self):
        w = _lane_weight_for("Mid", {"Mid": 1.5})
        self.assertAlmostEqual(w, 1.5)

    def test_lane_weight_bot_adc_alias(self):
        w = _lane_weight_for("Bot", {"ADC": 1.2})
        self.assertAlmostEqual(w, 1.2)

    def test_league_weight_no_dict_returns_1(self):
        self.assertEqual(_league_weight_for("LCK", None), 1.0)

    def test_league_weight_exact_match(self):
        w = _league_weight_for("LCK", {"LCK": 1.3})
        self.assertAlmostEqual(w, 1.3)

    def test_league_weight_missing_key_returns_1(self):
        w = _league_weight_for("UNKNOWNLEAGUE", {"LCK": 1.3})
        self.assertAlmostEqual(w, 1.0)


# ---------------------------------------------------------------------------
# PlayersArtifact.prior_for / get_state
# ---------------------------------------------------------------------------

class TestPlayersArtifactPriorAndState(unittest.TestCase):

    def test_prior_for_unknown_returns_0(self):
        art = _art()
        self.assertAlmostEqual(art.prior_for("Faker", "Mid"), 0.0)

    def test_prior_for_known_key(self):
        art = _art(prior_scores={"Faker|||Mid": {"prior": 0.25}})
        self.assertAlmostEqual(art.prior_for("Faker", "Mid"), 0.25)

    def test_prior_for_non_dict_value_returns_0(self):
        art = _art(prior_scores={"Faker|||Mid": 0.5})
        self.assertAlmostEqual(art.prior_for("Faker", "Mid"), 0.0)

    def test_get_state_initializes_at_prior(self):
        art = _art(prior_scores={"p1|||Mid": {"prior": 0.3}})
        st = art.get_state("p1", "Mid")
        self.assertAlmostEqual(st.r, 0.3)

    def test_get_state_returns_existing(self):
        art = _art_with_player("p1", "Mid", r=0.8, gp=10)
        st = art.get_state("p1", "Mid")
        self.assertAlmostEqual(st.r, 0.8)
        self.assertEqual(st.gp, 10)

    def test_get_state_side_effect_stores_state(self):
        art = _art()
        art.get_state("NewPlayer", "Top")
        self.assertIn("NewPlayer|||Top", art.players)


# ---------------------------------------------------------------------------
# PlayersArtifact.r_effective (shrink only, no decay)
# ---------------------------------------------------------------------------

class TestREffective(unittest.TestCase):

    def test_zero_gp_returns_near_prior(self):
        art = _art(shrink_m=20.0)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=2.0, gp=0)
        r_eff, _gp, _ = art.r_effective_at("p", "Mid", as_of=None, match_league=None)
        # shrink=0/(0+20)=0 → r_eff = prior + 0*(r-prior) = 0.0
        self.assertAlmostEqual(r_eff, 0.0, places=4)

    def test_large_gp_returns_near_r(self):
        art = _art(shrink_m=20.0)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=0.5, gp=2000)
        r_eff, _gp, _ = art.r_effective_at("p", "Mid", as_of=None, match_league=None)
        self.assertAlmostEqual(r_eff, 0.5, places=2)

    def test_gp_midpoint(self):
        art = _art(shrink_m=20.0)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=1.0, gp=20)
        # shrink = 20/(20+20)=0.5 → r_eff = 0 + 0.5*(1.0-0)=0.5
        r_eff, _gp, _ = art.r_effective_at("p", "Mid", as_of=None, match_league=None)
        self.assertAlmostEqual(r_eff, 0.5, places=4)

    def test_unknown_player_returns_zero_zero(self):
        art = _art()
        r_eff, gp, _dbg = art.r_effective_at("ghost", "Top", as_of=None, match_league=None)
        self.assertAlmostEqual(r_eff, 0.0)
        self.assertEqual(gp, 0)

    def test_empty_pos_returns_zero(self):
        art = _art()
        r_eff, _gp, _dbg = art.r_effective_at("Faker", "", as_of=None, match_league=None)
        self.assertAlmostEqual(r_eff, 0.0)

    def test_debug_keys_present(self):
        art = _art_with_player("p", "Mid", r=0.3, gp=50)
        _, _, dbg = art.r_effective_at("p", "Mid", as_of=None, match_league=None)
        for key in ("r_eff", "gp", "prior", "r_shrunk", "transfer_scale"):
            self.assertIn(key, dbg)


# ---------------------------------------------------------------------------
# PlayersArtifact.r_effective_at — decay
# ---------------------------------------------------------------------------

class TestREffectiveAtDecay(unittest.TestCase):

    def test_decay_reduces_rating_from_prior(self):
        art = _art(half_life_days=180.0, shrink_m=1.0)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=1.0, gp=100, last_date="2024-01-01")
        # Query 180 days later → decay factor = 0.5
        as_of = date(2024, 7, 1)  # ~181 days
        r_eff, _, _ = art.r_effective_at("p", "Mid", as_of=as_of, match_league=None)
        # r after decay ≈ 0 + (1.0 - 0) * 0.5 = 0.5; then shrink≈1 at gp=100
        self.assertLess(r_eff, 1.0)
        self.assertGreater(r_eff, 0.0)

    def test_no_last_date_no_decay(self):
        # With no last_date there is no decay; shrink ≈ 1 at very high gp.
        art = _art(half_life_days=180.0, shrink_m=1.0)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=0.8, gp=100_000, last_date=None)
        as_of = date(2025, 1, 1)
        r_eff, _, _ = art.r_effective_at("p", "Mid", as_of=as_of, match_league=None)
        self.assertAlmostEqual(r_eff, 0.8, places=3)

    def test_transfer_penalty_on_upgrade(self):
        # Player was in T2 league, now playing in T1 — should be penalized
        art = _art(shrink_m=1.0, transfer_alpha=0.7)
        key = "p|||Mid"
        art.players[key] = PlayerState(r=1.0, gp=100, last_tier="T2", last_league="LCS")
        r_no_penalty, _, _ = art.r_effective_at("p", "Mid", as_of=None, match_league=None)
        r_penalty, _, _ = art.r_effective_at("p", "Mid", as_of=None, match_league="LCK")
        self.assertLessEqual(r_penalty, r_no_penalty)

    def test_transfer_no_tier_penalty_on_downgrade(self):
        # _transfer_scale returns 1.0 when moving to a weaker tier (T1→T2).
        # We verify this by calling _transfer_scale directly, since r_effective_at
        # also applies an interleague cross-macro scale that can reduce the rating.
        art = _art(transfer_alpha=0.7)
        # T1=priority 0, T2=priority 1 → d = 0-1 = -1 ≤ 0 → returns 1.0
        scale = art._transfer_scale(last_tier="T1", cur_tier="T2")
        self.assertAlmostEqual(scale, 1.0)


# ---------------------------------------------------------------------------
# PlayersArtifact.update_game
# ---------------------------------------------------------------------------

class TestUpdateGame(unittest.TestCase):

    def setUp(self):
        self.art = _art(k=0.15, shrink_m=0.0)  # shrink_m=0 → no shrink, see prior
        self.game_date = date(2025, 1, 15)
        self.blue = {"Top": "b1", "Jungle": "b2", "Mid": "b3", "Bot": "b4", "Support": "b5"}
        self.red = {"Top": "r1", "Jungle": "r2", "Mid": "r3", "Bot": "r4", "Support": "r5"}

    def test_gp_increments(self):
        self.art.update_game(game_date=self.game_date, blue_lineup=self.blue, red_lineup=self.red, blue_win=1)
        for pos in POS_CANON:
            bp = self.blue[pos]
            rp = self.red[pos]
            self.assertEqual(self.art.get_state(bp, pos).gp, 1)
            self.assertEqual(self.art.get_state(rp, pos).gp, 1)

    def test_blue_win_increases_blue_rating(self):
        self.art.update_game(game_date=self.game_date, blue_lineup=self.blue, red_lineup=self.red, blue_win=1)
        for pos in POS_CANON:
            # Blue won → blue rating should go up (error = 1 - sigmoid(0)=0.5 > 0)
            self.assertGreater(self.art.get_state(self.blue[pos], pos).r, 0.0)
            self.assertLess(self.art.get_state(self.red[pos], pos).r, 0.0)

    def test_red_win_decreases_blue_rating(self):
        self.art.update_game(game_date=self.game_date, blue_lineup=self.blue, red_lineup=self.red, blue_win=0)
        for pos in POS_CANON:
            self.assertLess(self.art.get_state(self.blue[pos], pos).r, 0.0)
            self.assertGreater(self.art.get_state(self.red[pos], pos).r, 0.0)

    def test_last_date_set(self):
        self.art.update_game(game_date=self.game_date, blue_lineup=self.blue, red_lineup=self.red, blue_win=1)
        st = self.art.get_state("b1", "Top")
        self.assertEqual(st.last_date, "2025-01-15")

    def test_league_stored(self):
        self.art.update_game(game_date=self.game_date, league="LCK", blue_lineup=self.blue, red_lineup=self.red, blue_win=1)
        st = self.art.get_state("b3", "Mid")
        self.assertEqual(st.last_league, "LCK")

    def test_missing_position_skipped(self):
        blue = {"Top": "b1"}  # only Top
        red = {"Top": "r1"}
        self.art.update_game(game_date=self.game_date, blue_lineup=blue, red_lineup=red, blue_win=1)
        self.assertEqual(self.art.get_state("b1", "Top").gp, 1)
        # Mid not in lineup → not updated
        self.assertEqual(self.art.get_state("b3", "Mid").gp, 0)

    def test_repeated_games_accumulate(self):
        for i in range(5):
            self.art.update_game(
                game_date=date(2025, 1, i + 1),
                blue_lineup=self.blue,
                red_lineup=self.red,
                blue_win=1,
            )
        self.assertEqual(self.art.get_state("b1", "Top").gp, 5)


# ---------------------------------------------------------------------------
# compute_delta_and_confidence
# ---------------------------------------------------------------------------

class TestComputeDeltaAndConfidence(unittest.TestCase):

    def _full_art(self, r_blue: float = 0.3, r_red: float = -0.2) -> PlayersArtifact:
        art = _art(shrink_m=1.0)
        for pos in POS_CANON:
            art.players[f"b{pos}|||{pos}"] = PlayerState(r=r_blue, gp=200)
            art.players[f"r{pos}|||{pos}"] = PlayerState(r=r_red, gp=200)
        return art

    def _blue_lineup(self) -> dict:
        return {pos: f"b{pos}" for pos in POS_CANON}

    def _red_lineup(self) -> dict:
        return {pos: f"r{pos}" for pos in POS_CANON}

    def test_output_keys_present(self):
        art = self._full_art()
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        for k in ("delta", "coverage", "coverage_total", "avg_gp_min", "details"):
            self.assertIn(k, out)

    def test_coverage_total_is_5(self):
        art = self._full_art()
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertEqual(out["coverage_total"], 5)

    def test_full_lineup_coverage_5(self):
        art = self._full_art()
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertEqual(out["coverage"], 5)

    def test_delta_positive_when_blue_stronger(self):
        art = self._full_art(r_blue=0.5, r_red=-0.5)
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertGreater(out["delta"], 0.0)

    def test_delta_negative_when_red_stronger(self):
        art = self._full_art(r_blue=-0.5, r_red=0.5)
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertLess(out["delta"], 0.0)

    def test_delta_zero_when_equal(self):
        art = self._full_art(r_blue=0.0, r_red=0.0)
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertAlmostEqual(out["delta"], 0.0, places=6)

    def test_partial_lineup_lower_coverage(self):
        art = self._full_art()
        blue = {"Top": "bTop", "Mid": "bMid"}
        red = {"Top": "rTop", "Mid": "rMid"}
        out = compute_delta_and_confidence(art, blue_lineup=blue, red_lineup=red)
        self.assertEqual(out["coverage"], 2)

    def test_empty_lineup_zero_coverage(self):
        art = _art()
        out = compute_delta_and_confidence(art, blue_lineup={}, red_lineup={})
        self.assertEqual(out["coverage"], 0)
        self.assertAlmostEqual(out["delta"], 0.0)

    def test_adc_alias_normalized(self):
        art = _art(shrink_m=1.0)
        art.players["bBot|||Bot"] = PlayerState(r=0.4, gp=100)
        art.players["rBot|||Bot"] = PlayerState(r=-0.4, gp=100)
        # Pass "ADC" as key — should be normalized to "Bot"
        blue = {"ADC": "bBot"}
        red = {"ADC": "rBot"}
        out = compute_delta_and_confidence(art, blue_lineup=blue, red_lineup=red)
        self.assertEqual(out["coverage"], 1)
        self.assertGreater(out["delta"], 0.0)

    def test_lane_weights_scale_delta(self):
        art = self._full_art(r_blue=0.5, r_red=0.0)
        out_default = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        out_scaled = compute_delta_and_confidence(
            art,
            blue_lineup=self._blue_lineup(),
            red_lineup=self._red_lineup(),
            lane_weights={pos: 2.0 for pos in POS_CANON},
        )
        self.assertAlmostEqual(out_scaled["delta"], out_default["delta"] * 2.0, places=4)

    def test_require_both_sides_true_skips_one_sided(self):
        art = _art(shrink_m=1.0)
        art.players["b1|||Mid"] = PlayerState(r=1.0, gp=100)
        # Red has no Mid player
        blue = {"Mid": "b1"}
        red = {}
        out = compute_delta_and_confidence(art, blue_lineup=blue, red_lineup=red, require_both_sides=True)
        self.assertEqual(out["coverage"], 0)

    def test_require_both_sides_false_does_not_crash(self):
        # require_both_sides=False skips the early-continue, but ok=bool(pb and pr)
        # still needs both sides for counting — coverage stays 0 when one side missing.
        art = _art(shrink_m=1.0)
        art.players["b1|||Mid"] = PlayerState(r=1.0, gp=100)
        blue = {"Mid": "b1"}
        red = {}
        out = compute_delta_and_confidence(art, blue_lineup=blue, red_lineup=red, require_both_sides=False)
        self.assertEqual(out["coverage"], 0)  # still 0 — both sides needed for counting

    def test_freshness_in_output_with_as_of(self):
        art = self._full_art()
        for k in art.players:
            art.players[k].last_date = "2025-01-01"
        out = compute_delta_and_confidence(
            art,
            blue_lineup=self._blue_lineup(),
            red_lineup=self._red_lineup(),
            as_of=date(2025, 6, 1),
        )
        self.assertIn("freshness_avg", out)
        self.assertGreaterEqual(out["freshness_avg"], 0.0)
        self.assertLessEqual(out["freshness_avg"], 1.0)

    def test_transfer_payload_does_not_crash(self):
        art = self._full_art()
        out = compute_delta_and_confidence(
            art,
            blue_lineup=self._blue_lineup(),
            red_lineup=self._red_lineup(),
            transfer_payload={"enabled": False},
        )
        self.assertIn("player_portability_summary", out)

    def test_details_has_5_entries(self):
        art = self._full_art()
        out = compute_delta_and_confidence(art, blue_lineup=self._blue_lineup(), red_lineup=self._red_lineup())
        self.assertEqual(len(out["details"]), 5)


# ---------------------------------------------------------------------------
# lineup_for_game
# ---------------------------------------------------------------------------

class TestLineupForGame(unittest.TestCase):

    def _make_df(self):
        import pandas as pd
        rows = []
        for i, pos in enumerate(POS_CANON, 1):
            rows.append({"gameid": "g1", "side": "blue", "pos_can": pos, "player_key": f"b{i}"})
            rows.append({"gameid": "g1", "side": "red", "pos_can": pos, "player_key": f"r{i}"})
        return pd.DataFrame(rows)

    def test_blue_and_red_returned(self):
        df = self._make_df()
        blue, red = lineup_for_game(df, "g1")
        self.assertEqual(len(blue), 5)
        self.assertEqual(len(red), 5)

    def test_unknown_gameid_returns_empty(self):
        df = self._make_df()
        blue, red = lineup_for_game(df, "NOPE")
        self.assertEqual(blue, {})
        self.assertEqual(red, {})

    def test_empty_dataframe_returns_empty(self):
        import pandas as pd
        blue, _red = lineup_for_game(pd.DataFrame(), "g1")
        self.assertEqual(blue, {})

    def test_player_keys_correct(self):
        df = self._make_df()
        blue, red = lineup_for_game(df, "g1")
        self.assertEqual(blue["Mid"], "b3")
        self.assertEqual(red["Mid"], "r3")


# ---------------------------------------------------------------------------
# PlayersArtifact.to_dict / round-trip
# ---------------------------------------------------------------------------

class TestArtifactSerialisation(unittest.TestCase):

    def test_to_dict_has_players_key(self):
        art = _art_with_player("Faker", "Mid", r=0.5, gp=10)
        d = art.to_dict()
        self.assertIn("players", d)
        self.assertIn("Faker|||Mid", d["players"])

    def test_to_dict_float_values(self):
        art = _art_with_player("Faker", "Mid", r=0.5, gp=10)
        d = art.to_dict()
        entry = d["players"]["Faker|||Mid"]
        self.assertIsInstance(entry["r"], float)
        self.assertIsInstance(entry["gp"], int)


if __name__ == "__main__":
    unittest.main()
