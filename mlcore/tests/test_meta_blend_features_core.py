"""Tests for mlcore/meta_blend_features_core.py — C-model gate and feature builder."""
import math
import unittest

from mlcore.meta_blend_features_core import (
    META_BLEND_C_MODEL_FEATURES,
    build_meta_blend_c_model_features,
    should_apply_meta_blend_c_model_gate,
)


# ---------------------------------------------------------------------------
# should_apply_meta_blend_c_model_gate
# ---------------------------------------------------------------------------

class TestShouldApplyMetaBlendCModelGate(unittest.TestCase):

    def _gate(self, **kw):
        return should_apply_meta_blend_c_model_gate(**kw)

    def test_output_has_required_keys(self):
        out = self._gate()
        for key in ("allow", "reason", "bo", "score", "threshold", "inputs"):
            self.assertIn(key, out)

    def test_allow_is_bool(self):
        out = self._gate()
        self.assertIsInstance(out["allow"], bool)

    def test_score_in_unit_interval(self):
        for re in (0.0, 0.5, 1.0):
            out = self._gate(recent_evidence=re, n_team_eff=20.0)
            self.assertGreaterEqual(out["score"], 0.0)
            self.assertLessEqual(out["score"], 1.0)

    def test_bo5_has_lower_threshold(self):
        out_bo1 = self._gate(bo=1)
        out_bo5 = self._gate(bo=5)
        self.assertLess(out_bo5["threshold"], out_bo1["threshold"])

    def test_bo5_with_high_evidence_allows(self):
        out = self._gate(bo=5, recent_evidence=0.9, n_team_eff=30.0, n_players_eff=5.0)
        self.assertTrue(out["allow"])

    def test_bo1_no_evidence_blocks(self):
        out = self._gate(bo=1, recent_evidence=0.0, n_team_eff=0.0, n_players_eff=0.0)
        self.assertFalse(out["allow"])

    def test_interregional_reduces_score(self):
        out_normal = self._gate(recent_evidence=0.8, n_team_eff=20.0, bo=3)
        out_inter = self._gate(recent_evidence=0.8, n_team_eff=20.0, bo=3, interregional=True)
        self.assertLessEqual(out_inter["score"], out_normal["score"])

    def test_patch_refit_reduces_score(self):
        out_normal = self._gate(recent_evidence=0.8, n_team_eff=20.0, bo=3)
        out_refit = self._gate(recent_evidence=0.8, n_team_eff=20.0, bo=3, patch_status="refit")
        self.assertLessEqual(out_refit["score"], out_normal["score"])

    def test_playoffs_adds_score(self):
        out_no = self._gate(recent_evidence=0.5, n_team_eff=15.0, bo=3, is_playoffs=False)
        out_yes = self._gate(recent_evidence=0.5, n_team_eff=15.0, bo=3, is_playoffs=True)
        self.assertGreater(out_yes["score"], out_no["score"])

    def test_reason_blocked_when_not_allow(self):
        out = self._gate(bo=1, recent_evidence=0.0)
        self.assertFalse(out["allow"])
        self.assertIn("blocked", out["reason"])

    def test_reason_allow_when_allowed(self):
        out = self._gate(bo=5, recent_evidence=1.0, n_team_eff=50.0, n_players_eff=10.0)
        self.assertTrue(out["allow"])
        self.assertIn("allow", out["reason"])

    def test_invalid_bo_defaults_to_one(self):
        out = self._gate(bo="bad")
        self.assertEqual(out["bo"], 1)

    def test_inputs_dict_present(self):
        out = self._gate(recent_evidence=0.6, n_team_eff=15.0)
        self.assertIsInstance(out["inputs"], dict)
        self.assertIn("recent_evidence", out["inputs"])


# ---------------------------------------------------------------------------
# build_meta_blend_c_model_features
# ---------------------------------------------------------------------------

class TestBuildMetaBlendCModelFeatures(unittest.TestCase):

    def _build(self, p_anchor=0.55, p_adjusted=0.60, **kw):
        return build_meta_blend_c_model_features(p_anchor=p_anchor, p_adjusted=p_adjusted, **kw)

    def test_ok_true_for_valid_probs(self):
        out = self._build()
        self.assertTrue(out["ok"])

    def test_invalid_anchor_returns_not_ok(self):
        out = build_meta_blend_c_model_features(p_anchor=float("nan"), p_adjusted=0.6)
        self.assertFalse(out["ok"])

    def test_invalid_adjusted_returns_not_ok(self):
        out = build_meta_blend_c_model_features(p_anchor=0.5, p_adjusted=float("inf"))
        self.assertFalse(out["ok"])

    def test_output_has_required_keys(self):
        out = self._build()
        for key in ("ok", "reason", "features", "vector"):
            self.assertIn(key, out)

    def test_vector_length_matches_feature_list(self):
        out = self._build()
        self.assertEqual(len(out["vector"]), len(META_BLEND_C_MODEL_FEATURES))

    def test_features_dict_has_all_keys(self):
        out = self._build()
        for k in META_BLEND_C_MODEL_FEATURES:
            self.assertIn(k, out["features"])

    def test_all_vector_values_are_finite(self):
        out = self._build(bo=3, n_team_eff=15.0, n_players_eff=3.0)
        for v in out["vector"]:
            self.assertTrue(math.isfinite(v), f"Non-finite vector value: {v}")

    def test_z_anchor_finite_for_valid_p(self):
        out = self._build(p_anchor=0.7)
        self.assertTrue(math.isfinite(out["features"]["z_anchor"]))

    def test_log1p_n_team_positive(self):
        out = self._build(n_team_eff=10.0)
        self.assertGreater(out["features"]["log1p_n_team"], 0.0)

    def test_log1p_n_team_zero_for_no_eff(self):
        out = self._build(n_team_eff=0.0)
        self.assertAlmostEqual(out["features"]["log1p_n_team"], 0.0)

    def test_bo_is_3_flag(self):
        out = self._build(bo=3)
        self.assertAlmostEqual(out["features"]["bo_is_3"], 1.0)
        self.assertAlmostEqual(out["features"]["bo_is_5"], 0.0)

    def test_bo_is_5_flag(self):
        out = self._build(bo=5)
        self.assertAlmostEqual(out["features"]["bo_is_5"], 1.0)
        self.assertAlmostEqual(out["features"]["bo_is_3"], 0.0)

    def test_playoffs_flag(self):
        out = self._build(is_playoffs=True)
        self.assertAlmostEqual(out["features"]["is_playoffs"], 1.0)

    def test_interregional_flag(self):
        out = self._build(interregional=True)
        self.assertAlmostEqual(out["features"]["interregional"], 1.0)

    def test_patch_refit_flag(self):
        out = self._build(patch_status="refit")
        self.assertAlmostEqual(out["features"]["patch_refit"], 1.0)

    def test_cons_guard_flag(self):
        out = self._build(consistency_guard_triggered=True, consistency_guard_drift_pp=10.0)
        self.assertAlmostEqual(out["features"]["cons_guard"], 1.0)
        self.assertGreater(out["features"]["cons_drift_scaled"], 0.0)

    def test_sign_z_delta_positive_when_adjusted_higher(self):
        out = self._build(p_anchor=0.50, p_adjusted=0.65)
        self.assertAlmostEqual(out["features"]["sign_z_delta"], 1.0)

    def test_sign_z_delta_negative_when_adjusted_lower(self):
        out = self._build(p_anchor=0.65, p_adjusted=0.50)
        self.assertAlmostEqual(out["features"]["sign_z_delta"], -1.0)

    def test_rank_sos_applied_flag(self):
        out = self._build(rank_sos_applied=True, rank_sos_weight=0.5, rank_sos_diff_pp=10.0, rank_sos_p=0.6)
        self.assertAlmostEqual(out["features"]["rank_sos_applied"], 1.0)
        self.assertAlmostEqual(out["features"]["rank_sos_weight"], 0.5)
        self.assertGreater(out["features"]["rank_sos_diff_scaled"], 0.0)

    def test_roster_cont_p_centered(self):
        out = self._build(roster_cont_target_p=0.7)
        self.assertAlmostEqual(out["features"]["roster_cont_p_centered"], 0.2, places=5)

    def test_meta_dict_present(self):
        out = self._build()
        self.assertIn("meta", out)
        self.assertIn("p_anchor", out["meta"])


# ---------------------------------------------------------------------------
# META_BLEND_C_MODEL_FEATURES constant
# ---------------------------------------------------------------------------

class TestMetaBlendCModelFeatures(unittest.TestCase):

    def test_is_list(self):
        self.assertIsInstance(META_BLEND_C_MODEL_FEATURES, list)

    def test_non_empty(self):
        self.assertGreater(len(META_BLEND_C_MODEL_FEATURES), 0)

    def test_all_strings(self):
        for f in META_BLEND_C_MODEL_FEATURES:
            self.assertIsInstance(f, str)

    def test_no_duplicates(self):
        self.assertEqual(len(META_BLEND_C_MODEL_FEATURES), len(set(META_BLEND_C_MODEL_FEATURES)))


if __name__ == "__main__":
    unittest.main()
