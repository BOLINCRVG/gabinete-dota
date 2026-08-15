import unittest

from app_core.state_strength_areal_core import compute_state_shadow_areal


def _mk_artifact(
    *,
    blue_last="2026-02-20",
    red_last="2026-02-20",
    asof="2026-02-20",
    by_league=None,
    roster_cont_avg_blue=1.0,
    roster_cont_avg_red=1.0,
    roster_cont_n_blue=12,
    roster_cont_n_red=12,
    last_league_blue="LCK",
    last_league_red="LCK",
    mu_blue=0.8,
    mu_red=-0.4,
    qwl_blue=0.0,
    qwl_red=0.0,
    qwl_surplus_blue=0.0,
    qwl_surplus_red=0.0,
    qwl_n_blue=0,
    qwl_n_red=0,
    qwl_rel_blue=1.0,
    qwl_rel_red=1.0,
    online_journal_tail=None,
):
    art = {
        "version": "team_state_artifact_v9",
        "params": {
            "qwl_expected_base": 0.12,
            "qwl_expected_slope": 0.6,
            "qwl_surplus_mix": 0.7,
            "qwl_tempo_weight": 0.08,
            "qwl_conversion_weight": 0.12,
            "qwl_throw_weight": 1.0,
            "qwl_comeback_weight": 0.65,
            "qwl_phase_dynamic_strength": 0.35,
            "qwl_opponent_context_mix": 0.3,
            "qwl_surplus_mix_by_league": [],
            "qwl_surplus_mix_by_patch": [],
            "qwl_throw_weight_by_league": [],
            "qwl_throw_weight_by_patch": [],
            "qwl_comeback_weight_by_league": [],
            "qwl_comeback_weight_by_patch": [],
            "qwl_phase_dynamic_strength_by_league": [],
            "qwl_phase_dynamic_strength_by_patch": [],
            "qwl_opponent_context_mix_by_league": [],
            "qwl_opponent_context_mix_by_patch": [],
        },
        "meta": {
            "update_mode": "incremental",
            "last_processed_game": {"gameid": "g1", "date": asof},
            "online_journal_tail_count": 1,
            "quality_adjust_avg": 0.31,
            "quality_expected_avg": 0.16,
            "quality_surplus_avg": 0.27,
            "quality_context_multiplier_avg": 0.99,
            "quality_context_boost_games": 80,
            "quality_context_cut_games": 120,
            "quality_tempo_avg": 0.02,
            "quality_conversion_avg": 0.45,
            "quality_throw_avg": -0.06,
            "quality_comeback_avg": 0.03,
        },
        "teams": {
            "teama": {
                "team": "Team A",
                "mu": mu_blue,
                "sigma": 0.6,
                "gp": 20,
                "last_date": blue_last,
                "last_league": last_league_blue,
                "roster_cont_avg": roster_cont_avg_blue,
                "roster_cont_n": roster_cont_n_blue,
                "qwl_signed_avg": qwl_blue,
                "qwl_surplus_signed_avg": qwl_surplus_blue,
                "qwl_n": qwl_n_blue,
                "qwl_data_reliability_avg": qwl_rel_blue,
            },
            "teamb": {
                "team": "Team B",
                "mu": mu_red,
                "sigma": 0.7,
                "gp": 20,
                "last_date": red_last,
                "last_league": last_league_red,
                "roster_cont_avg": roster_cont_avg_red,
                "roster_cont_n": roster_cont_n_red,
                "qwl_signed_avg": qwl_red,
                "qwl_surplus_signed_avg": qwl_surplus_red,
                "qwl_n": qwl_n_red,
                "qwl_data_reliability_avg": qwl_rel_red,
            },
        },
    }
    if isinstance(by_league, dict):
        art["by_league"] = by_league
    if isinstance(online_journal_tail, list):
        art["online_journal_tail"] = list(online_journal_tail)
    return art


class TestStateStrengthArealCore(unittest.TestCase):
    def test_areal_applies_with_valid_artifact(self):
        out = compute_state_shadow_areal(
            p_map_old=0.55,
            p_series_old=0.62,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            state_artifact=_mk_artifact(),
            trace_uncertainty_info={"n_eff": 22.0, "prior_games": 20.0},
            n_team_eff=15.0,
            n_players_eff=5.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("method")), "state_shadow_areal_v9")
        self.assertIn("weights", out)
        self.assertIn("state_recency", out.get("weights", {}))
        self.assertIn("precision_state_weight", out.get("weights", {}))

    def test_stale_state_damps_move(self):
        kwargs = dict(
            p_map_old=0.55,
            p_series_old=0.62,
            bo=5,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            trace_uncertainty_info={"n_eff": 22.0, "prior_games": 20.0},
            n_team_eff=18.0,
            n_players_eff=6.0,
        )
        fresh = compute_state_shadow_areal(state_artifact=_mk_artifact(blue_last="2026-02-19", red_last="2026-02-20", asof="2026-02-20"), **kwargs)
        stale = compute_state_shadow_areal(state_artifact=_mk_artifact(blue_last="2025-12-01", red_last="2025-12-02", asof="2026-02-20"), **kwargs)
        self.assertTrue(fresh.get("applied"))
        self.assertTrue(stale.get("applied"))
        self.assertGreater(abs(float(fresh.get("diff_pp_map") or 0.0)), abs(float(stale.get("diff_pp_map") or 0.0)))
        self.assertGreater(float(fresh.get("weights", {}).get("state_recency", 0.0)), float(stale.get("weights", {}).get("state_recency", 0.0)))

    def test_league_mismatch_damps_state_weight(self):
        kwargs = dict(
            p_map_old=0.52,
            p_series_old=0.55,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            trace_uncertainty_info={"n_eff": 30.0, "prior_games": 20.0},
            n_team_eff=25.0,
            n_players_eff=8.0,
        )
        art = _mk_artifact()
        same = compute_state_shadow_areal(state_artifact=art, league_blue="LCK", league_red="LCK", **kwargs)
        cross = compute_state_shadow_areal(state_artifact=art, league_blue="LEC", league_red="LEC", **kwargs)
        self.assertTrue(same.get("applied"))
        self.assertTrue(cross.get("applied"))
        self.assertGreater(float(same.get("weights", {}).get("league", 0.0)), float(cross.get("weights", {}).get("league", 0.0)))
        self.assertGreater(
            abs(float(same.get("diff_pp_map") or 0.0)),
            abs(float(cross.get("diff_pp_map") or 0.0)),
        )

    def test_interregional_shared_foreign_event_damps_state_even_more(self):
        kwargs = dict(
            p_map_old=0.5861365558989118,
            p_series_old=0.6583390391564214,
            bo=5,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LPL",
            league_red="CBLOL",
            trace_uncertainty_info={"n_eff": 0.380592893286499, "prior_games": 13.2},
            n_team_eff=0.0,
            n_players_eff=0.0,
        )
        aligned = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.09971250213575561,
                mu_red=0.3129942415901323,
                roster_cont_avg_blue=1.0,
                roster_cont_avg_red=0.990909090909091,
                roster_cont_n_blue=50,
                roster_cont_n_red=22,
                asof="2026-03-17",
                blue_last="2026-03-17",
                red_last="2026-03-17",
                last_league_blue="LPL",
                last_league_red="CBLOL",
                by_league={"LPL": {"teams": 10, "games": 260}, "CBLOL": {"teams": 10, "games": 150}},
            ),
            **kwargs,
        )
        shared_foreign = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.09971250213575561,
                mu_red=0.3129942415901323,
                roster_cont_avg_blue=1.0,
                roster_cont_avg_red=0.990909090909091,
                roster_cont_n_blue=50,
                roster_cont_n_red=22,
                asof="2026-03-17",
                blue_last="2026-03-17",
                red_last="2026-03-17",
                last_league_blue="FST",
                last_league_red="FST",
                by_league={"LPL": {"teams": 10, "games": 260}, "CBLOL": {"teams": 10, "games": 150}},
            ),
            **kwargs,
        )
        self.assertTrue(aligned.get("applied"))
        self.assertTrue(shared_foreign.get("applied"))
        self.assertGreaterEqual(
            abs(float(aligned.get("diff_pp_map") or 0.0)),
            abs(float(shared_foreign.get("diff_pp_map") or 0.0)),
        )
        self.assertGreaterEqual(
            float(aligned.get("weights", {}).get("final", 0.0)),
            float(shared_foreign.get("weights", {}).get("final", 0.0)),
        )

    def test_cross_strength_interregional_caps_adverse_move_against_stronger_side(self):
        out = compute_state_shadow_areal(
            p_map_old=0.84,
            p_series_old=0.92,
            bo=5,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="CBLOL",
            state_artifact=_mk_artifact(
                mu_blue=-0.20,
                mu_red=0.35,
                roster_cont_avg_blue=1.0,
                roster_cont_avg_red=1.0,
                roster_cont_n_blue=24,
                roster_cont_n_red=24,
                asof="2026-03-17",
                blue_last="2026-03-17",
                red_last="2026-03-17",
                last_league_blue="LCK",
                last_league_red="CBLOL",
                by_league={"LCK": {"teams": 10, "games": 260}, "CBLOL": {"teams": 10, "games": 150}},
            ),
            trace_uncertainty_info={"n_eff": 12.0, "prior_games": 13.2},
            n_team_eff=16.0,
            n_players_eff=6.0,
        )
        self.assertTrue(out.get("applied"))
        adverse_pp = (0.84 - float(out.get("p_map_new"))) * 100.0
        self.assertLessEqual(adverse_pp, 0.900001)

    def test_cross_tier_interregional_damps_state_more_than_same_tier_interregional(self):
        kwargs = dict(
            p_map_old=0.5861365558989118,
            p_series_old=0.6583390391564214,
            bo=5,
            team_blue="Team A",
            team_red="Team B",
            trace_uncertainty_info={"n_eff": 0.38, "prior_games": 13.2},
            n_team_eff=0.0,
            n_players_eff=0.0,
        )
        same_tier = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.10,
                mu_red=0.31,
                asof="2026-03-17",
                blue_last="2026-03-17",
                red_last="2026-03-17",
                last_league_blue="LPL",
                last_league_red="LEC",
                by_league={"LPL": {"teams": 10, "games": 260}, "LEC": {"teams": 10, "games": 230}},
            ),
            league_blue="LPL",
            league_red="LEC",
            **kwargs,
        )
        cross_tier = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.10,
                mu_red=0.31,
                asof="2026-03-17",
                blue_last="2026-03-17",
                red_last="2026-03-17",
                last_league_blue="LPL",
                last_league_red="CBLOL",
                by_league={"LPL": {"teams": 10, "games": 260}, "CBLOL": {"teams": 10, "games": 150}},
            ),
            league_blue="LPL",
            league_red="CBLOL",
            **kwargs,
        )
        self.assertTrue(same_tier.get("applied"))
        self.assertTrue(cross_tier.get("applied"))
        self.assertGreater(
            abs(float(same_tier.get("diff_pp_map") or 0.0)),
            abs(float(cross_tier.get("diff_pp_map") or 0.0)),
        )
        self.assertTrue(bool((cross_tier.get("state") or {}).get("cross_tier_interregional", False)))

    def test_sparse_league_coverage_damps_state_weight(self):
        kwargs = dict(
            p_map_old=0.52,
            p_series_old=0.55,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            trace_uncertainty_info={"n_eff": 30.0, "prior_games": 20.0},
            n_team_eff=25.0,
            n_players_eff=8.0,
        )
        dense = compute_state_shadow_areal(
            state_artifact=_mk_artifact(by_league={"LCK": {"teams": 10, "games": 220}}),
            **kwargs,
        )
        sparse = compute_state_shadow_areal(
            state_artifact=_mk_artifact(by_league={"LCK": {"teams": 3, "games": 12}}),
            **kwargs,
        )
        self.assertTrue(dense.get("applied"))
        self.assertTrue(sparse.get("applied"))
        self.assertGreater(
            float(dense.get("weights", {}).get("league_coverage", 0.0)),
            float(sparse.get("weights", {}).get("league_coverage", 0.0)),
        )
        self.assertGreater(
            abs(float(dense.get("diff_pp_map") or 0.0)),
            abs(float(sparse.get("diff_pp_map") or 0.0)),
        )

    def test_roster_state_reliability_damps_state_weight(self):
        kwargs = dict(
            p_map_old=0.54,
            p_series_old=0.57,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            trace_uncertainty_info={"n_eff": 30.0, "prior_games": 20.0},
            n_team_eff=25.0,
            n_players_eff=8.0,
        )
        stable = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                roster_cont_avg_blue=0.95,
                roster_cont_avg_red=0.92,
                roster_cont_n_blue=18,
                roster_cont_n_red=18,
            ),
            **kwargs,
        )
        unstable = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                roster_cont_avg_blue=0.25,
                roster_cont_avg_red=0.30,
                roster_cont_n_blue=18,
                roster_cont_n_red=18,
            ),
            **kwargs,
        )
        self.assertTrue(stable.get("applied"))
        self.assertTrue(unstable.get("applied"))
        self.assertGreater(
            float(stable.get("weights", {}).get("roster_state", 0.0)),
            float(unstable.get("weights", {}).get("roster_state", 0.0)),
        )
        self.assertGreater(
            abs(float(stable.get("diff_pp_map") or 0.0)),
            abs(float(unstable.get("diff_pp_map") or 0.0)),
        )

    def test_exposes_qwl_trace_from_artifact(self):
        out = compute_state_shadow_areal(
            p_map_old=0.55,
            p_series_old=0.62,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            state_artifact=_mk_artifact(),
            trace_uncertainty_info={"n_eff": 22.0, "prior_games": 20.0},
            n_team_eff=15.0,
            n_players_eff=5.0,
        )
        qwl = out.get("qwl_trace") if isinstance(out.get("qwl_trace"), dict) else {}
        self.assertTrue(bool(qwl.get("enabled", False)))
        params = qwl.get("params") if isinstance(qwl.get("params"), dict) else {}
        self.assertAlmostEqual(float(params.get("qwl_surplus_mix")), 0.7, places=6)
        meta = qwl.get("artifact_meta") if isinstance(qwl.get("artifact_meta"), dict) else {}
        self.assertAlmostEqual(float(meta.get("quality_adjust_avg")), 0.31, places=6)

    def test_qwl_team_edge_pushes_state_direction(self):
        kwargs = dict(
            p_map_old=0.50,
            p_series_old=0.50,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            trace_uncertainty_info={"n_eff": 28.0, "prior_games": 20.0},
            n_team_eff=20.0,
            n_players_eff=6.0,
        )
        base = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.0,
                mu_red=0.0,
                qwl_blue=0.0,
                qwl_red=0.0,
                qwl_surplus_blue=0.0,
                qwl_surplus_red=0.0,
                qwl_n_blue=0,
                qwl_n_red=0,
            ),
            **kwargs,
        )
        qwl = compute_state_shadow_areal(
            state_artifact=_mk_artifact(
                mu_blue=0.0,
                mu_red=0.0,
                qwl_blue=0.28,
                qwl_red=-0.20,
                qwl_surplus_blue=0.14,
                qwl_surplus_red=-0.10,
                qwl_n_blue=40,
                qwl_n_red=40,
                qwl_rel_blue=0.98,
                qwl_rel_red=0.97,
            ),
            **kwargs,
        )
        self.assertTrue(base.get("applied"))
        self.assertTrue(qwl.get("applied"))
        self.assertGreater(
            abs(float(qwl.get("diff_pp_map") or 0.0)),
            abs(float(base.get("diff_pp_map") or 0.0)),
        )
        self.assertGreater(float(qwl.get("weights", {}).get("qwl_n", 0.0)), 0.0)
        self.assertTrue(
            abs(float(qwl.get("weights", {}).get("qwl_logit_shift", 0.0))) > 0.0
        )

    def test_qwl_fallback_from_online_journal_tail(self):
        journal = [
            {
                "blue": "Team A",
                "red": "Team B",
                "y_blue": 1,
                "q_adjusted": 0.42,
                "q_surplus": 0.20,
                "q_tempo_term": 0.08,
                "q_conversion_term": 0.10,
                "q_throw_term": 0.0,
                "q_comeback_term": 0.02,
                "q_data_reliability": 0.98,
            },
            {
                "blue": "Team B",
                "red": "Team A",
                "y_blue": 0,
                "q_adjusted": 0.36,
                "q_surplus": 0.18,
                "q_tempo_term": -0.04,
                "q_conversion_term": -0.08,
                "q_throw_term": -0.10,
                "q_comeback_term": 0.00,
                "q_data_reliability": 0.97,
            },
        ]
        out = compute_state_shadow_areal(
            p_map_old=0.50,
            p_series_old=0.50,
            bo=3,
            team_blue="Team A",
            team_red="Team B",
            league_blue="LCK",
            league_red="LCK",
            state_artifact=_mk_artifact(
                mu_blue=0.0,
                mu_red=0.0,
                qwl_n_blue=0,
                qwl_n_red=0,
                online_journal_tail=journal,
            ),
            trace_uncertainty_info={"n_eff": 28.0, "prior_games": 20.0},
            n_team_eff=20.0,
            n_players_eff=6.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertGreater(float(out.get("weights", {}).get("qwl_n", 0.0)), 0.0)
        self.assertTrue(abs(float(out.get("weights", {}).get("qwl_logit_shift", 0.0))) > 0.0)


if __name__ == "__main__":
    unittest.main()
