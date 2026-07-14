import json
import tempfile
import unittest
from pathlib import Path

from app_core.params_tuning_core import (
    enrich_params_with_areal_prob_tuning,
    enrich_params_with_ml_marginal_band_coeffs,
    is_valid_areal_prob_tuning,
    postprocess_params_file_tuning,
)


def _sanitize_band_coeffs(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for band in ("top", "core", "edge"):
        row = raw.get(band)
        if not isinstance(row, dict):
            continue
        clean = {}
        for key, val in row.items():
            try:
                clean[str(key)] = float(val)
            except Exception:
                continue
        if clean:
            out[band] = clean
    return out


class TestParamsTuningCore(unittest.TestCase):
    def test_is_valid_areal_prob_tuning(self):
        ok_cfg = {
            "version": "areal_prob_tuning_v1",
            "global": {
                "bayes_info_scale": 1.0,
                "bayes_mu_q_strength": 1.0,
                "quality_k_boost": 0.22,
                "qwl_surplus_mix": 0.70,
                "qwl_throw_weight": 1.20,
                "qwl_comeback_weight": 0.65,
                "qwl_phase_dynamic_strength": 0.35,
                "qwl_opponent_context_mix": 0.30,
            },
        }
        self.assertTrue(is_valid_areal_prob_tuning(ok_cfg))
        self.assertFalse(is_valid_areal_prob_tuning({}))
        self.assertFalse(is_valid_areal_prob_tuning({"version": "x", "global": {}}))

    def test_enrich_prefers_phase2b_and_preserves_existing_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = {
                "unknown_key": {"a": 1},
                "x": 2,
            }
            (root / "params_fitted.json").write_text(json.dumps(params), encoding="utf-8")
            phase2b = {
                "decision": "promote",
                "tuning_cfg": {
                    "version": "areal_prob_tuning_v2",
                    "global": {
                        "bayes_info_scale": 1.0,
                        "bayes_mu_q_strength": 1.0,
                        "quality_k_boost": 0.22,
                        "qwl_surplus_mix": 0.70,
                        "qwl_throw_weight": 1.20,
                        "qwl_comeback_weight": 0.65,
                        "qwl_phase_dynamic_strength": 0.35,
                        "qwl_opponent_context_mix": 0.30,
                    },
                },
            }
            (root / "phase2b_areal_tuning_report.json").write_text(json.dumps(phase2b), encoding="utf-8")

            out, src = enrich_params_with_areal_prob_tuning(dict(params), root=root)
            self.assertEqual(src, "phase2b_areal_tuning_report.json")
            self.assertIn("unknown_key", out)
            self.assertIn("areal_prob_tuning", out)
            self.assertEqual(out.get("areal_prob_tuning_source"), "phase2b_areal_tuning_report.json")

    def test_enrich_marginal_uses_params_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = {
                "ml_marginal_trust_band_coeffs": {
                    "top": {"strength_scale": 0.5},
                },
            }
            (root / "params_fitted.json").write_text(json.dumps(params), encoding="utf-8")
            out, src = enrich_params_with_ml_marginal_band_coeffs(
                dict(params),
                root=root,
                sanitize_band_coeffs_fn=_sanitize_band_coeffs,
            )
            self.assertEqual(src, "already_present")
            self.assertEqual(
                out.get("ml_marginal_trust_band_coeffs", {}).get("top", {}).get("strength_scale"),
                0.5,
            )

    def test_postprocess_updates_file_without_key_loss(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = {"k_legacy": 123, "nested": {"ok": True}}
            (root / "params_fitted.json").write_text(json.dumps(params), encoding="utf-8")
            phase2b = {
                "decision": "promote",
                "tuning_cfg": {
                    "version": "areal_prob_tuning_v3",
                    "global": {
                        "bayes_info_scale": 1.0,
                        "bayes_mu_q_strength": 1.0,
                        "quality_k_boost": 0.22,
                        "qwl_surplus_mix": 0.70,
                        "qwl_throw_weight": 1.20,
                        "qwl_comeback_weight": 0.65,
                        "qwl_phase_dynamic_strength": 0.35,
                        "qwl_opponent_context_mix": 0.30,
                    },
                },
            }
            phase4b = {
                "decision": "promote",
                "tuning_cfg": {
                    "band_coeffs": {
                        "top": {"strength_scale": 0.45},
                        "core": {"strength_scale": 0.80},
                    }
                },
            }
            (root / "phase2b_areal_tuning_report.json").write_text(json.dumps(phase2b), encoding="utf-8")
            (root / "phase4b_marginal_tuning_report.json").write_text(json.dumps(phase4b), encoding="utf-8")

            ok, reason = postprocess_params_file_tuning(
                root,
                sanitize_band_coeffs_fn=_sanitize_band_coeffs,
            )
            self.assertTrue(ok)
            self.assertIn("updated_from_", reason)
            out = json.loads((root / "params_fitted.json").read_text(encoding="utf-8"))
            self.assertEqual(out.get("k_legacy"), 123)
            self.assertEqual(out.get("nested", {}).get("ok"), True)
            self.assertIn("areal_prob_tuning", out)
            self.assertIn("ml_marginal_trust_band_coeffs", out)


if __name__ == "__main__":
    unittest.main()
