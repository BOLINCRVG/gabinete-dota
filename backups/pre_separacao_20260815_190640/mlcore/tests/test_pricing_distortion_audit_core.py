import math
import unittest

from app_core.pricing_distortion_audit_core import (
    evaluate_pricing_distortion,
    extract_top_adverse_moves,
    merge_pricing_distortion_cases,
    perspective_probability,
    pick_reference_probability,
    seed_pricing_distortion_cases,
    stronger_side_from_leagues,
)


class TestPickReferenceProbability(unittest.TestCase):

    def test_prefers_p_fused(self):
        trace = {"probabilities": {"p_cal": 0.61, "p_fused": 0.64, "p_map_used": 0.58}}
        value, source = pick_reference_probability(trace)
        self.assertAlmostEqual(value, 0.64, places=6)
        self.assertEqual(source, "p_fused")

    def test_falls_back_to_p_cal(self):
        trace = {"probabilities": {"p_cal": 0.60}}
        value, source = pick_reference_probability(trace)
        self.assertAlmostEqual(value, 0.60, places=6)
        self.assertEqual(source, "p_cal")

    def test_none_trace_returns_nan(self):
        value, source = pick_reference_probability(None)
        self.assertTrue(math.isnan(value))
        self.assertEqual(source, "")

    def test_empty_probs_returns_nan(self):
        value, _source = pick_reference_probability({"probabilities": {}})
        self.assertTrue(math.isnan(value))


class TestSeedAndMergeDistortionCases(unittest.TestCase):

    def test_seed_returns_list(self):
        cases = seed_pricing_distortion_cases()
        self.assertIsInstance(cases, list)
        self.assertGreater(len(cases), 0)

    def test_seed_each_has_required_keys(self):
        for case in seed_pricing_distortion_cases():
            self.assertIn("team_a", case)
            self.assertIn("team_b", case)
            self.assertIn("bo", case)

    def test_merge_deduplicates(self):
        base = [{"team_a": "T1", "team_b": "GEN", "bo": 5}]
        extra = [{"team_a": "T1", "team_b": "GEN", "bo": 5}]
        merged = merge_pricing_distortion_cases(base, extra)
        self.assertEqual(len(merged), 1)

    def test_merge_combines_unique(self):
        base = [{"team_a": "T1", "team_b": "GEN", "bo": 5}]
        extra = [{"team_a": "JDG", "team_b": "BLG", "bo": 3}]
        merged = merge_pricing_distortion_cases(base, extra)
        self.assertEqual(len(merged), 2)

    def test_merge_skips_empty_teams(self):
        cases = [{"team_a": "", "team_b": "GEN", "bo": 5}]
        merged = merge_pricing_distortion_cases(cases)
        self.assertEqual(len(merged), 0)

    def test_merge_none_inputs(self):
        merged = merge_pricing_distortion_cases(None, None)
        self.assertEqual(merged, [])


class TestStrongerSideFromLeagues(unittest.TestCase):

    def test_lck_stronger_than_cblol(self):
        out = stronger_side_from_leagues("LCK", "CBLOL")
        self.assertEqual(out["stronger_side"], "a")
        self.assertGreater(out["strength_gap"], 0.0)

    def test_same_league_no_stronger_side(self):
        out = stronger_side_from_leagues("LCK", "LCK")
        self.assertEqual(out["stronger_side"], "")

    def test_returns_dict_with_required_keys(self):
        out = stronger_side_from_leagues("LEC", "LPL")
        for key in ("league_a", "league_b", "tier_a", "tier_b", "score_a", "score_b", "strength_gap", "stronger_side"):
            self.assertIn(key, out)


class TestPerspectiveProbability(unittest.TestCase):

    def test_side_a_returns_as_is(self):
        self.assertAlmostEqual(perspective_probability(0.7, "a"), 0.7, places=6)

    def test_side_b_inverts(self):
        self.assertAlmostEqual(perspective_probability(0.7, "b"), 0.3, places=6)

    def test_unknown_side_returns_nan(self):
        self.assertTrue(math.isnan(perspective_probability(0.7, "c")))

    def test_invalid_prob_returns_nan(self):
        self.assertTrue(math.isnan(perspective_probability(float("nan"), "a")))


class PricingDistortionAuditCoreTests(unittest.TestCase):
    def test_pick_reference_probability_prefers_fused(self):
        trace = {"probabilities": {"p_cal": 0.61, "p_fused": 0.64, "p_map_used": 0.58}}
        value, source = pick_reference_probability(trace)
        self.assertAlmostEqual(value, 0.64, places=6)
        self.assertEqual(source, "p_fused")

    def test_extract_top_adverse_moves_flips_for_team_b(self):
        steps = [
            {"stage": "state_areal_live_override", "before": 0.60, "after": 0.52, "applied": True},
            {"stage": "style_matchup_b_live", "before": 0.52, "after": 0.54, "applied": True},
        ]
        rows = extract_top_adverse_moves(steps, stronger_side="b")
        self.assertEqual(rows[0]["stage"], "style_matchup_b_live")
        self.assertGreater(float(rows[0]["adverse_pp"]), 1.9)

    def test_evaluate_pricing_distortion_flags_overcompression(self):
        trace = {
            "probabilities": {
                "p_fused": 0.68,
                "p_map_used": 0.54,
            },
            "pipeline": {
                "steps": [
                    {"stage": "season_form_blend", "before": 0.68, "after": 0.63, "applied": True},
                    {"stage": "state_areal_live_override", "before": 0.63, "after": 0.54, "applied": True},
                ]
            },
        }
        row = evaluate_pricing_distortion(
            team_a="JD Gaming",
            team_b="LOUD",
            league_a="LPL",
            league_b="CBLOL",
            trace_payload=trace,
        )
        self.assertEqual(row["status"], "review")
        self.assertEqual(row["problem_class"], "post_layers_overcompressed")
        self.assertIn("compressao_excessiva_pos_camadas", row["reasons"])

    def test_evaluate_pricing_distortion_flags_base_underpowered(self):
        trace = {
            "probabilities": {
                "p_fused": 0.53,
                "p_map_used": 0.52,
            },
            "pipeline": {"steps": []},
        }
        row = evaluate_pricing_distortion(
            team_a="JD Gaming",
            team_b="LOUD",
            league_a="LPL",
            league_b="CBLOL",
            trace_payload=trace,
        )
        self.assertEqual(row["status"], "review")
        self.assertEqual(row["problem_class"], "base_underpowered")
        self.assertIn("base_ja_sai_fraca_para_o_gap", row["reasons"])


if __name__ == "__main__":
    unittest.main()
