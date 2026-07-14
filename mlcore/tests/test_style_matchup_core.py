import unittest
from dataclasses import dataclass

from app_core.style_matchup_core import (
    STYLE_MATCHUP_B_MODEL_FEATURES,
    apply_style_matchup_b_live,
    apply_style_matchup_b_lite,
    apply_style_matchup_b_model,
    build_style_matchup_b_model_features,
)


@dataclass
class _P:
    n_games: int = 10
    kills_for: float = 15.0
    kills_against: float = 12.0
    towers_for: float = 8.0
    towers_against: float = 6.0
    dragons_for: float = 2.6
    dragons_against: float = 1.8
    barons_for: float = 1.1
    barons_against: float = 0.7
    inhib_for: float = 1.3
    inhib_against: float = 0.8
    time_min: float = 30.0
    kills_for_sd: float = 4.0
    kills_against_sd: float = 3.5


class TestStyleMatchupCore(unittest.TestCase):
    def test_applies_when_profiles_valid(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0, towers_for=6.0, towers_against=8.0, dragons_for=1.6, dragons_against=2.5)
        out = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LPL",
            league_b="LPL",
            patch_status="ok",
            roster_changed=False,
        )
        self.assertTrue(out.get("applied"))
        self.assertGreater(float(out.get("p_after")), 0.55)
        self.assertIn("style", out)
        self.assertIn("terms", out["style"])

    def test_interregional_damps_move(self):
        pa = _P()
        pb = _P(kills_for=11.0, kills_against=16.0, towers_for=5.0, towers_against=9.0)
        same = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            patch_status="ok",
            roster_changed=False,
        )
        inter = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LEC",
            patch_status="ok",
            roster_changed=False,
        )
        self.assertTrue(same.get("applied"))
        self.assertTrue(inter.get("applied"))
        self.assertGreater(abs(float(same.get("diff_pp", 0.0))), abs(float(inter.get("diff_pp", 0.0))))

    def test_interregional_high_sample_relief_increases_move(self):
        pa_low = _P(n_games=4)
        pb_low = _P(n_games=4, kills_for=11.0, kills_against=16.0, towers_for=5.0, towers_against=9.0)
        pa_high = _P(n_games=18)
        pb_high = _P(n_games=18, kills_for=11.0, kills_against=16.0, towers_for=5.0, towers_against=9.0)
        low = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa_low,
            pB=pb_low,
            league_a="LCK",
            league_b="LEC",
            patch_status="ok",
            roster_changed=False,
            bo=3,
        )
        high = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa_high,
            pB=pb_high,
            league_a="LCK",
            league_b="LEC",
            patch_status="ok",
            roster_changed=False,
            bo=3,
        )
        self.assertTrue(low.get("applied"))
        self.assertTrue(high.get("applied"))
        self.assertGreater(abs(float(high.get("diff_pp", 0.0))), abs(float(low.get("diff_pp", 0.0))))

    def test_cross_tier_interregional_damps_more_than_same_tier_interregional(self):
        pa = _P(n_games=18)
        pb = _P(n_games=18, kills_for=11.0, kills_against=16.0, towers_for=5.0, towers_against=9.0)
        same_tier = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LPL",
            league_b="LEC",
            patch_status="ok",
            roster_changed=False,
            bo=5,
        )
        cross_tier = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LPL",
            league_b="CBLOL",
            patch_status="ok",
            roster_changed=False,
            bo=5,
        )
        self.assertTrue(same_tier.get("applied"))
        self.assertTrue(cross_tier.get("applied"))
        self.assertGreater(abs(float(same_tier.get("diff_pp", 0.0))), abs(float(cross_tier.get("diff_pp", 0.0))))

    def test_low_sample_profiles_skip(self):
        pa = _P(n_games=2)
        pb = _P(n_games=9)
        out = apply_style_matchup_b_lite(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertFalse(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "low_sample_profiles")

    def test_feature_builder_returns_vector(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        pa_r = _P(n_games=6, kills_for=17.0, kills_against=11.0)
        pb_r = _P(n_games=6, kills_for=11.0, kills_against=16.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            pA_recent=pa_r,
            pB_recent=pb_r,
            league_a="LCK",
            league_b="LCK",
            bo=5,
            is_playoffs=True,
        )
        self.assertTrue(feat.get("ok"))
        self.assertEqual(len(feat.get("vector") or []), len(STYLE_MATCHUP_B_MODEL_FEATURES))
        self.assertIn("style", feat)
        self.assertIn("weights", feat)
        self.assertIn("recent", (feat.get("style") or {}))
        feats = dict(feat.get("features") or {})
        self.assertEqual(float(feats.get("bo_is_5", 0.0)), 1.0)
        self.assertEqual(float(feats.get("is_playoffs", 0.0)), 1.0)
        self.assertEqual(float(feats.get("same_league", 0.0)), 1.0)
        macros = (((feat.get("style") or {}).get("macros")) or {})
        self.assertIn(str(macros.get("phase_bucket", "")), {"early", "mid", "late"})
        terms = ((feat.get("style") or {}).get("terms")) or {}
        self.assertIn("quality_objective_control_edge", terms)
        self.assertIn("quality_closeout_edge", terms)
        self.assertIn("quality_resilience_edge", terms)
        self.assertIn("quality_net_control_edge", terms)
        self.assertIn("quality_conversion_edge", terms)
        self.assertIn("quality_containment_edge", terms)

    def test_quality_terms_push_style_direction(self):
        # A has stronger objective control/closeout profile than B.
        pa = _P(
            kills_for=16.0,
            kills_against=9.0,
            towers_for=9.0,
            towers_against=4.5,
            dragons_for=3.1,
            dragons_against=1.1,
            barons_for=1.4,
            barons_against=0.4,
            inhib_for=1.7,
            inhib_against=0.5,
            time_min=29.0,
        )
        pb = _P(
            kills_for=11.0,
            kills_against=14.5,
            towers_for=5.2,
            towers_against=8.7,
            dragons_for=1.4,
            dragons_against=2.8,
            barons_for=0.5,
            barons_against=1.3,
            inhib_for=0.6,
            inhib_against=1.5,
            time_min=33.5,
        )
        feat = build_style_matchup_b_model_features(
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="TCL",
            league_b="TCL",
            bo=3,
            is_playoffs=False,
        )
        self.assertTrue(feat.get("ok"))
        terms = ((feat.get("style") or {}).get("terms")) or {}
        self.assertGreater(float(terms.get("quality_objective_control_edge", 0.0)), 0.0)
        self.assertGreater(float(terms.get("quality_closeout_edge", 0.0)), 0.0)
        self.assertGreater(float(terms.get("quality_resilience_edge", 0.0)), 0.0)
        self.assertGreater(float(terms.get("quality_net_control_edge", 0.0)), 0.0)
        self.assertGreater(float(terms.get("quality_conversion_edge", 0.0)), 0.0)
        self.assertGreater(float(terms.get("quality_containment_edge", 0.0)), 0.0)

    def test_model_apply_uses_trained_cfg(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v1",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 999,
            "oos_logloss": 0.66,
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("method")), "style_matchup_b_model_v1")
        # Intercept=0 -> sigmoid(0)=0.5
        self.assertAlmostEqual(float(out.get("p_after")), 0.5, places=6)

    def test_model_apply_runtime_guard_blocks_underperforming_global(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 1200,
            "oos_logloss": 0.70,
            "improvement_vs_lite": -0.010,
            "improvement_vs_anchor": -0.008,
            "runtime_guard": {
                "enabled": True,
                "min_n_total": 220,
                "min_improvement_vs_lite": -0.003,
                "min_improvement_vs_anchor": -0.0015,
                "allow_context_override": True,
            },
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertFalse(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "model_reliability_guard")
        guard = ((out.get("model_meta") or {}).get("guard")) or {}
        self.assertTrue(bool(guard.get("blocked", False)))

    def test_model_apply_runtime_guard_allows_strong_context_override(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 1400,
            "oos_logloss": 0.70,
            "improvement_vs_lite": -0.012,
            "improvement_vs_anchor": -0.009,
            "context_models": {
                "league:LCK": {
                    "version": "style_matchup_b_model_v7_ctx_override",
                    "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
                    "coef": [0.0 for _ in vec],
                    "intercept": 0.8,
                    "x_mean": [0.0 for _ in vec],
                    "x_std": [1.0 for _ in vec],
                    "n_total": 780,
                    "oos_logloss": 0.62,
                    "improvement_vs_lite": 0.011,
                    "improvement_vs_anchor": 0.008,
                }
            },
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str((out.get("model_meta") or {}).get("context_key")), "league:LCK")
        self.assertEqual(str(out.get("method")), "style_matchup_b_model_v7_ctx_override")
        guard = ((out.get("model_meta") or {}).get("guard")) or {}
        self.assertFalse(bool(guard.get("blocked", False)))

    def test_model_apply_prefers_context_submodel(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            patch_status="changed",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        base = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 999,
            "oos_logloss": 0.66,
        }
        cfg = dict(base)
        cfg["context_models"] = {
            "league:LCK|patch:refit": {
                "version": "style_matchup_b_model_v7_ctx",
                "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
                "coef": [0.0 for _ in vec],
                "intercept": 1.0,  # sigmoid(1)=0.731...
                "x_mean": [0.0 for _ in vec],
                "x_std": [1.0 for _ in vec],
                "n_total": 300,
                "oos_logloss": 0.60,
            }
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            patch_status="changed",
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str((out.get("model_meta") or {}).get("context_key")), "league:LCK|patch:refit")
        self.assertEqual(str(out.get("method")), "style_matchup_b_model_v7_ctx")
        self.assertTrue(bool(((out.get("model_meta") or {}).get("context_blend") or {}).get("used")))
        # v6 blends contextual submodel with global model in logit space, so
        # the result should move up from p_in, but not necessarily stay as
        # extreme as the raw contextual submodel sigmoid(intercept=1.0).
        self.assertNotAlmostEqual(float(out.get("p_after")), float(out.get("p_before")), places=6)
        cb = ((out.get("model_meta") or {}).get("context_blend") or {})
        self.assertGreater(float(cb.get("w_ctx_blend", 0.0)), 0.0)
        self.assertLessEqual(float(cb.get("w_ctx_blend", 0.0)), 1.0)
        os_ = ((out.get("model_meta") or {}).get("output_shrink") or {})
        self.assertTrue(bool(os_.get("used")))
        self.assertGreater(float(os_.get("value", 0.0)), 0.0)
        self.assertLessEqual(float(os_.get("value", 1.0)), 1.0)

    def test_model_apply_prefers_phase_context_submodel(self):
        pa = _P(time_min=28.0, kills_for=18.0, kills_against=12.0)
        pb = _P(time_min=30.0, kills_for=12.0, kills_against=18.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            patch_status="ok",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 999,
            "oos_logloss": 0.66,
            "context_models": {
                "league:LCK|phase:early": {
                    "version": "style_matchup_b_model_v7_ctx_phase",
                    "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
                    "coef": [0.0 for _ in vec],
                    "intercept": 0.8,
                    "x_mean": [0.0 for _ in vec],
                    "x_std": [1.0 for _ in vec],
                    "n_total": 420,
                    "oos_logloss": 0.60,
                    "improvement_vs_lite": 0.01,
                    "improvement_vs_anchor": 0.01,
                }
            },
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str((out.get("model_meta") or {}).get("context_key")), "league:LCK|phase:early")
        self.assertEqual(str(out.get("method")), "style_matchup_b_model_v7_ctx_phase")

    def test_model_apply_quality_scale_by_league_changes_output(self):
        pa = _P(
            kills_for=16.0,
            kills_against=9.0,
            towers_for=9.0,
            towers_against=4.0,
            dragons_for=3.0,
            dragons_against=1.1,
            barons_for=1.4,
            barons_against=0.4,
            inhib_for=1.6,
            inhib_against=0.5,
            time_min=29.0,
        )
        pb = _P(
            kills_for=11.0,
            kills_against=15.0,
            towers_for=5.0,
            towers_against=8.5,
            dragons_for=1.5,
            dragons_against=2.8,
            barons_for=0.5,
            barons_against=1.2,
            inhib_for=0.7,
            inhib_against=1.5,
            time_min=33.0,
        )
        feat = build_style_matchup_b_model_features(
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        idx_style = STYLE_MATCHUP_B_MODEL_FEATURES.index("style_score")
        coef = [0.0 for _ in vec]
        coef[idx_style] = 1.15
        cfg_base = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": coef,
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 1600,
            "oos_logloss": 0.64,
        }
        out_base = apply_style_matchup_b_model(
            model_cfg=cfg_base,
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(out_base.get("applied"))

        cfg_scaled = dict(cfg_base)
        cfg_scaled["quality_term_scale"] = {
            "global": 1.0,
            "min": 0.70,
            "max": 1.35,
            "map": {"league:LCK": 1.30},
        }
        out_scaled = apply_style_matchup_b_model(
            model_cfg=cfg_scaled,
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
        )
        self.assertTrue(out_scaled.get("applied"))
        self.assertGreater(float(out_scaled.get("p_after")), float(out_base.get("p_after")))
        q_meta = ((out_scaled.get("model_meta") or {}).get("quality_term_scale") or {})
        self.assertEqual(str(q_meta.get("selected_key")), "league:LCK")
        self.assertGreater(float(q_meta.get("scale", 1.0)), 1.0)

    def test_model_apply_quality_scale_uses_global_patch_key(self):
        pa = _P(
            kills_for=15.0,
            kills_against=10.0,
            towers_for=8.0,
            towers_against=5.0,
            dragons_for=2.7,
            dragons_against=1.4,
            barons_for=1.2,
            barons_against=0.5,
            inhib_for=1.3,
            inhib_against=0.6,
            time_min=32.0,
        )
        pb = _P(
            kills_for=11.0,
            kills_against=15.0,
            towers_for=5.3,
            towers_against=8.3,
            dragons_for=1.5,
            dragons_against=2.7,
            barons_for=0.5,
            barons_against=1.1,
            inhib_for=0.7,
            inhib_against=1.4,
            time_min=32.5,
        )
        feat = build_style_matchup_b_model_features(
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="TCL",
            league_b="TCL",
            patch_status="ok",
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        idx_style = STYLE_MATCHUP_B_MODEL_FEATURES.index("style_score")
        coef = [0.0 for _ in vec]
        coef[idx_style] = 1.10
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": coef,
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 1500,
            "oos_logloss": 0.65,
            "quality_term_scale": {
                "global": 1.0,
                "min": 0.88,
                "max": 1.15,
                "map": {"global|patch:stable": 1.10},
            },
        }
        out = apply_style_matchup_b_model(
            model_cfg=cfg,
            p_map_in=0.52,
            pA=pa,
            pB=pb,
            league_a="TCL",
            league_b="TCL",
            patch_status="ok",
        )
        self.assertTrue(out.get("applied"))
        q_meta = ((out.get("model_meta") or {}).get("quality_term_scale") or {})
        self.assertEqual(str(q_meta.get("selected_key")), "global|patch:stable")
        self.assertGreater(float(q_meta.get("scale", 1.0)), 1.0)

    def test_live_path_model_applied(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            bo=5,
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7_test",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.2,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 900,
            "oos_logloss": 0.64,
            "improvement_vs_lite": 0.01,
            "improvement_vs_anchor": 0.01,
        }
        out = apply_style_matchup_b_live(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            bo=5,
            n_team_eff=20.0,
            n_players_eff=8.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("b_pipeline_path")), "model_applied")
        self.assertFalse(bool(out.get("fallback_from_model", False)))
        self.assertFalse(bool(out.get("fallback_from_gate", False)))
        self.assertEqual(str(out.get("method")), "style_matchup_b_model_v7_test")

    def test_live_fallbacks_to_lite_when_model_not_applied(self):
        pa = _P()
        pb = _P(kills_for=12.0, kills_against=15.0)
        feat = build_style_matchup_b_model_features(
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            bo=5,
        )
        self.assertTrue(feat.get("ok"))
        vec = [float(v) for v in (feat.get("vector") or [])]
        cfg = {
            "enabled": True,
            "version": "style_matchup_b_model_v7_guarded",
            "features": list(STYLE_MATCHUP_B_MODEL_FEATURES),
            "coef": [0.0 for _ in vec],
            "intercept": 0.0,
            "x_mean": [0.0 for _ in vec],
            "x_std": [1.0 for _ in vec],
            "n_total": 1200,
            "oos_logloss": 0.70,
            "improvement_vs_lite": -0.010,
            "improvement_vs_anchor": -0.008,
            "runtime_guard": {
                "enabled": True,
                "min_n_total": 220,
                "min_improvement_vs_lite": -0.003,
                "min_improvement_vs_anchor": -0.0015,
                "allow_context_override": True,
            },
        }
        out = apply_style_matchup_b_live(
            model_cfg=cfg,
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            bo=5,
            n_team_eff=20.0,
            n_players_eff=8.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "style_matchup_b_lite")
        self.assertTrue(bool(out.get("fallback_from_model", False)))
        self.assertFalse(bool(out.get("fallback_from_gate", False)))
        self.assertEqual(str(out.get("fallback_reason")), "model_reliability_guard")
        self.assertEqual(str(out.get("b_pipeline_path")), "lite_fallback_model_not_applied")

    def test_live_fallbacks_to_lite_when_gate_blocks(self):
        pa = _P(n_games=4)
        pb = _P(n_games=4, kills_for=12.0, kills_against=15.0)
        out = apply_style_matchup_b_live(
            model_cfg={"enabled": True},
            p_map_in=0.55,
            pA=pa,
            pB=pb,
            league_a="LCK",
            league_b="LCK",
            bo=1,
            n_team_eff=5.0,
            n_players_eff=0.0,
        )
        self.assertTrue(out.get("applied"))
        self.assertEqual(str(out.get("reason")), "style_matchup_b_lite")
        self.assertFalse(bool(out.get("fallback_from_model", False)))
        self.assertTrue(bool(out.get("fallback_from_gate", False)))
        self.assertEqual(str(out.get("fallback_reason")), "bo1_low_evidence")
        self.assertEqual(str(out.get("b_pipeline_path")), "lite_fallback_gate_blocked")


if __name__ == "__main__":
    unittest.main()
