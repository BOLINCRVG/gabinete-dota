import unittest
from pathlib import Path
import importlib.util


def _load_gate_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "tools" / "run_phase3_breal_gate.py"
    spec = importlib.util.spec_from_file_location("run_phase3_breal_gate", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed_to_load_phase3_gate_module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase3BrealGate(unittest.TestCase):
    def setUp(self):
        self.mod = _load_gate_module()

    def _row(self, *, league: str, scale_applied: bool, is_b_model: bool = True) -> dict:
        return {
            "ok": True,
            "league": league,
            "applied": True,
            "is_b_model": bool(is_b_model),
            "context_used": True,
            "fallback_from_model": False,
            "model_guard_blocked": False,
            "diff_pp": 2.1,
            "quality_signal_abs_mean": 0.22,
            "quality_scale_key": "global|patch:stable|phase:mid",
            "quality_scale_applied": bool(scale_applied),
        }

    def test_shadow_when_quality_scale_not_applied_with_expected_signal(self):
        rows = [self._row(league="LCK", scale_applied=False) for _ in range(10)]
        dec, reasons, metrics, _ = self.mod._decide(rows)
        self.assertEqual(dec, "shadow_only")
        self.assertIn("low_quality_scale_apply_rate", reasons)
        self.assertEqual(int(metrics.get("quality_scale_expected_samples") or 0), 10)
        self.assertLess(float(metrics.get("quality_scale_expected_apply_rate") or 0.0), 0.60)

    def test_promote_when_quality_scale_applied(self):
        rows = [self._row(league="LCK", scale_applied=True) for _ in range(10)]
        dec, reasons, metrics, _ = self.mod._decide(rows)
        self.assertEqual(dec, "promote")
        self.assertNotIn("low_quality_scale_apply_rate", reasons)
        self.assertEqual(int(metrics.get("quality_scale_expected_samples") or 0), 10)
        self.assertGreaterEqual(float(metrics.get("quality_scale_expected_apply_rate") or 0.0), 1.0)

    def test_shadow_when_b_model_not_effectively_used(self):
        rows = [self._row(league="LCK", scale_applied=True, is_b_model=False) for _ in range(10)]
        dec, reasons, metrics, _ = self.mod._decide(rows)
        self.assertEqual(dec, "shadow_only")
        self.assertIn("low_b_model_applied_rate", reasons)
        self.assertLess(float(metrics.get("b_model_applied_rate") or 0.0), 0.40)


if __name__ == "__main__":
    unittest.main()
