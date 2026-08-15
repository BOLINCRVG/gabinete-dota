import unittest

from app_core.interleague_baseline_core import build_interleague_pairs_table, infer_league_class_from_node


class TestInterleagueBaselineCore(unittest.TestCase):
    def test_infer_league_class_aligns_with_tier_config(self):
        self.assertEqual(
            infer_league_class_from_node("LEC", {"wf_n": 620, "quality_band": "ok"}),
            "major",
        )
        self.assertEqual(
            infer_league_class_from_node("CBLOL", {"wf_n": 493, "quality_band": "ok"}),
            "major",
        )
        self.assertEqual(
            infer_league_class_from_node("LCKC", {"wf_n": 1023, "quality_band": "ok"}),
            "minor",
        )
        self.assertEqual(
            infer_league_class_from_node("LFL", {"wf_n": 574, "quality_band": "ok"}),
            "minor",
        )

    def test_build_interleague_pairs_emits_reliability_metadata(self):
        by_league = {
            "LCK": {
                "score_bias": 0.35,
                "wf_n": 1038,
                "quality_band": "good",
                "score_q": 0.01,
                "fdr_reject": True,
            },
            "LPL": {
                "score_bias": 3.9151,
                "wf_n": 1553,
                "quality_band": "strong",
                "score_q": 0.001,
                "fdr_reject": True,
            },
            "LEC": {
                "score_bias": 0.0,
                "wf_n": 620,
                "quality_band": "ok",
                "score_q": 0.08,
                "fdr_reject": True,
            },
            "LFL": {
                "score_bias": -0.7,
                "wf_n": 574,
                "quality_band": "ok",
                "score_q": 0.22,
                "fdr_reject": False,
            },
        }
        league_class = {
            lg: infer_league_class_from_node(lg, node)
            for lg, node in by_league.items()
        }
        out = build_interleague_pairs_table(by_league, league_class_out=league_class)
        pair_major = out.get("league_pair:LCK->LPL|inter:1") or {}
        pair_mixed = out.get("league_pair:LCK->LFL|inter:1") or {}
        tier_pair = out.get("tier_pair:T1->ERL|inter:1") or {}
        macro_major = out.get("inter:KR->CN|inter:1") or {}
        pair_upper = out.get("league_pair:LEC->LFL|inter:1") or {}

        self.assertEqual(str(pair_major.get("class_relation")), "major_major")
        self.assertTrue(float(pair_major.get("pair_reliability", 0.0)) > 0.70)
        self.assertTrue(float(pair_major.get("source_reliability_a", 0.0)) > 0.70)
        self.assertEqual(str(pair_mixed.get("class_relation")), "mixed")
        self.assertTrue(float(pair_major.get("pair_reliability", 0.0)) > float(pair_mixed.get("pair_reliability", 1.0)))
        self.assertEqual(str(pair_major.get("strength_band_a")), "elite")
        self.assertEqual(str(pair_major.get("strength_band_b")), "elite")
        self.assertTrue(float(pair_mixed.get("strength_gap", 0.0)) > 0.20)
        self.assertTrue(float(pair_upper.get("strength_gap", 0.0)) < float(pair_mixed.get("strength_gap", 1.0)))
        self.assertEqual(str(tier_pair.get("source")), "interleague_tier_pair_meta_v1")
        self.assertTrue(float(tier_pair.get("pair_reliability", 0.0)) > 0.0)
        self.assertTrue(float(tier_pair.get("strength_gap", 0.0)) > 0.0)
        self.assertEqual(str(macro_major.get("source")), "interleague_pair_meta_v4")
        self.assertIn("major_major", dict(macro_major.get("class_mix") or {}))
        self.assertTrue(str(macro_major.get("dominant_strength_pair") or ""))


if __name__ == "__main__":
    unittest.main()
