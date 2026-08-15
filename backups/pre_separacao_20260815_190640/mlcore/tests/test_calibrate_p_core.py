"""Tests for app_core/calibrate_p_core.py — calibrate_p_from_artifact."""
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_core.calibrate_p_core import calibrate_p_from_artifact


def _ctx(art=None, path_exists=False):
    """Build a minimal ctx dict with stub callbacks."""
    def _file_sig(p):
        return p

    def _load_artifact_cached(sig):
        return art

    return {
        "_file_sig": _file_sig,
        "_load_artifact_cached": _load_artifact_cached,
    }


# ---------------------------------------------------------------------------
# calibrate_p_from_artifact — fallback paths
# ---------------------------------------------------------------------------

class TestCalibratePFallback(unittest.TestCase):

    def test_nan_p_raw_returns_nan(self):
        result = calibrate_p_from_artifact("nonexistent.json", float("nan"), ctx=_ctx())
        self.assertTrue(math.isnan(result))

    def test_inf_p_raw_returns_nan(self):
        result = calibrate_p_from_artifact("nonexistent.json", float("inf"), ctx=_ctx())
        self.assertTrue(math.isnan(result))

    def test_missing_file_returns_clipped_p(self):
        result = calibrate_p_from_artifact("nonexistent_path.json", 0.7, ctx=_ctx())
        self.assertAlmostEqual(result, 0.7, places=5)

    def test_missing_file_clips_above_1(self):
        result = calibrate_p_from_artifact("nonexistent.json", 1.5, ctx=_ctx())
        self.assertLessEqual(result, 1.0)

    def test_missing_file_clips_below_0(self):
        result = calibrate_p_from_artifact("nonexistent.json", -0.5, ctx=_ctx())
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# calibrate_p_from_artifact — with artifact file
# ---------------------------------------------------------------------------

class TestCalibratePWithArtifact(unittest.TestCase):

    def _make_ctx(self, calibrator):
        art = SimpleNamespace(calibrator=calibrator)
        return _ctx(art=art)

    def test_no_calibrator_returns_fallback(self):
        art = SimpleNamespace(calibrator=None)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = calibrate_p_from_artifact(path, 0.6, ctx=_ctx(art=art))
            self.assertAlmostEqual(result, 0.6, places=5)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_non_callable_calibrator_returns_fallback(self):
        art = SimpleNamespace(calibrator="not_a_calibrator")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = calibrate_p_from_artifact(path, 0.6, ctx=_ctx(art=art))
            self.assertAlmostEqual(result, 0.6, places=5)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_callable_calibrator_used(self):
        cal = SimpleNamespace(predict_proba=lambda p: 0.8)
        art = SimpleNamespace(calibrator=cal)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = calibrate_p_from_artifact(path, 0.6, ctx=_ctx(art=art))
            self.assertAlmostEqual(result, 0.8, places=5)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_calibrator_output_clipped(self):
        # calibrator returns value > 1.0 — must be clipped
        cal = SimpleNamespace(predict_proba=lambda p: 1.5)
        art = SimpleNamespace(calibrator=cal)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = calibrate_p_from_artifact(path, 0.6, ctx=_ctx(art=art))
            self.assertLessEqual(result, 1.0)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_calibrator_raising_returns_fallback(self):
        def bad_cal(p):
            raise ValueError("boom")
        cal = SimpleNamespace(predict_proba=bad_cal)
        art = SimpleNamespace(calibrator=cal)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result = calibrate_p_from_artifact(path, 0.65, ctx=_ctx(art=art))
            self.assertAlmostEqual(result, 0.65, places=5)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_result_is_float(self):
        result = calibrate_p_from_artifact("nonexistent.json", 0.55, ctx=_ctx())
        self.assertIsInstance(result, float)


if __name__ == "__main__":
    unittest.main()
