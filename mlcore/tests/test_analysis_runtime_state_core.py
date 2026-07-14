from __future__ import annotations

import unittest

from app_core.analysis_runtime_state_core import (
    build_analysis_signature,
    build_consistency_analysis_signature,
    compute_sim_seed_series,
    resolve_active_line_texts,
)


class AnalysisRuntimeStateCoreTests(unittest.TestCase):
    def test_compute_sim_seed_series_updates_session(self) -> None:
        state = {
            "sim_seed_mode": "Deterministico (confronto+config)",
        }

        def _seed(*parts) -> int:
            return len(parts) * 10

        got = compute_sim_seed_series(
            state,
            stable_seed_fn=_seed,
            season_label="S16",
            team_a="A",
            team_b="B",
            bo=3,
            year_opt="2026",
            split_opt="All",
            playoffs_opt="all",
            league_mode="auto",
            map_mode_ui="avg",
            weight_mode="Recencia",
            half_life_days=180,
            n_sims=20000,
            totals_mode="map",
            ml_mode="map",
        )
        self.assertEqual(got, 140)
        self.assertEqual(state["_sim_seed_series"], 140)

    def test_resolve_active_line_texts_prefers_saved_in_results(self) -> None:
        state = {"lines_kills": "25.5", "combo_time_text": "32.5"}
        saved = {"lines_kills": "27.5", "combo_time_text": "31.5"}
        got = resolve_active_line_texts(state, flow_stage="results", saved_lines=saved)
        self.assertEqual(got["lines_kills"], "27.5")
        self.assertEqual(got["combo_time_text"], "31.5")

        got_select = resolve_active_line_texts(state, flow_stage="select", saved_lines=saved)
        self.assertEqual(got_select["lines_kills"], "25.5")
        self.assertEqual(got_select["combo_time_text"], "32.5")

    def test_analysis_signatures_are_stable(self) -> None:
        state = {
            "resumo_combo_use_mismatch": True,
            "fusion_team_weight_early": 0.58,
            "fusion_team_weight_mid": 0.75,
            "fusion_team_weight_playoffs": 0.82,
            "fusion_early_season_team_weight": 0.55,
            "fusion_season_knee_games": 18.0,
            "fusion_coverage_power": 1.5,
            "fusion_transfer_boost": 0.35,
            "fusion_divergence_pp_cap": 22.0,
            "fusion_divergence_low_coverage": 0.30,
            "fusion_divergence_shrink": 0.35,
            "ml_microseg_enabled": True,
            "ml_microseg_min_n": 30,
            "ml_microseg_strength": 0.60,
            "ml_microseg_shrink_k": 90.0,
            "ml_microseg_lookback_games": 1200,
            "ml_microseg_cap_pp": 8.0,
            "ml_microseg_hierarchical": True,
            "ml_microseg_hier_decay": 0.80,
            "ml_fused_post_cal_use_fitted": True,
            "ml_uncertainty_enabled": True,
            "ml_uncertainty_strength": 0.55,
            "ml_uncertainty_prior_games": 22.0,
            "ml_uncertainty_max_shrink": 0.65,
            "ml_uncertainty_exclude_leagues": "",
            "ml_marginal_trust_enabled": True,
            "ml_marginal_trust_strength": 0.70,
            "ml_marginal_trust_prior_games": 28.0,
            "ml_marginal_trust_max_shrink": 0.72,
            "ml_marginal_trust_cap_prob": 0.84,
            "ml_marginal_trust_score_soft": 68.0,
            "ml_marginal_trust_score_hard": 46.0,
            "ml_marginal_trust_exclude_leagues": "",
            "ml_marginal_trust_band_tuning": True,
            "ml_marginal_trust_band_coeffs": {"soft": {"a": 1.0}},
        }
        line_texts = {
            "lines_kills": "27.5",
            "lines_towers": "11.5",
            "lines_dragons": "",
            "lines_barons": "",
            "lines_inhib": "",
            "lines_time": "32.5",
            "teamA_kills_line_text": "",
            "teamB_kills_line_text": "",
            "hc_kills_text": "",
            "hc_towers_text": "",
            "hc_dragons_text": "",
            "combo_kills_text": "27.5",
            "combo_time_text": "32.5",
        }

        def _seed(*parts) -> int:
            return abs(hash(tuple(parts))) % 1000000

        sig_a = build_analysis_signature(
            state,
            stable_seed_fn=_seed,
            sanitize_marginal_band_coeffs_fn=lambda raw: {"soft": {"a": float((raw or {}).get("soft", {}).get("a", 1.0))}},
            season_label="S16",
            team_a="A",
            team_b="B",
            bo=3,
            year_opt="2026",
            split_opt="All",
            playoffs_opt="all",
            league_mode="auto",
            fixed_league="All",
            window_opt="All",
            weight_mode="Recencia",
            half_life_days=180,
            n_sims=20000,
            totals_mode="map",
            ml_mode="map",
            map_mode_ui="avg",
            line_texts=line_texts,
        )
        sig_b = build_consistency_analysis_signature(
            state,
            stable_seed_fn=_seed,
            sanitize_marginal_band_coeffs_fn=lambda raw: {"soft": {"a": float((raw or {}).get("soft", {}).get("a", 1.0))}},
            season_label="S16",
            team_a="A",
            team_b="B",
            bo=3,
            year_opt="2026",
            split_opt="All",
            playoffs_opt="all",
            league_mode="auto",
            fixed_league="All",
            window_opt="All",
            weight_mode="Recencia",
            half_life_days=180,
            n_sims=20000,
            totals_mode="map",
            map_mode_ui="avg",
            line_texts=line_texts,
        )
        self.assertIsInstance(sig_a, int)
        self.assertIsInstance(sig_b, int)
        self.assertNotEqual(sig_a, 0)
        self.assertNotEqual(sig_b, 0)


if __name__ == "__main__":
    unittest.main()
