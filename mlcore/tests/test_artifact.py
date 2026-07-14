"""Tests for mlcore/artifact.py — MLArtifact serialization and version parsing."""
import json
import tempfile
import unittest
from pathlib import Path

from mlcore.artifact import MLArtifact, _parse_ver
from mlcore.calibration import PlattCalibrator
from mlcore.elo import EloModel


def _minimal_artifact_dict() -> dict:
    return {
        "version": "1.2.3",
        "model": {},
        "calibrator": {"a": 1.0, "b": 0.0},
        "meta": {"league": "LCK"},
    }


# ---------------------------------------------------------------------------
# _parse_ver
# ---------------------------------------------------------------------------

class TestParseVer(unittest.TestCase):

    def test_three_part_version(self):
        self.assertEqual(_parse_ver("1.2.3"), (1, 2, 3))

    def test_two_part_version(self):
        self.assertEqual(_parse_ver("2.5"), (2, 5, 0))

    def test_single_part_version(self):
        self.assertEqual(_parse_ver("3"), (3, 0, 0))

    def test_empty_string_returns_zeros(self):
        self.assertEqual(_parse_ver(""), (0, 0, 0))

    def test_invalid_returns_zeros(self):
        self.assertEqual(_parse_ver("bad.ver"), (0, 0, 0))

    def test_comparison_works(self):
        self.assertLess(_parse_ver("0.3.9"), _parse_ver("0.4.0"))
        self.assertGreater(_parse_ver("1.0.0"), _parse_ver("0.9.9"))

    def test_v_prefix_invalid(self):
        # "v1.2.3" has non-numeric first part → returns (0, 0, 0)
        self.assertEqual(_parse_ver("v1.2.3"), (0, 0, 0))

    def test_large_version(self):
        self.assertEqual(_parse_ver("10.20.30"), (10, 20, 30))


# ---------------------------------------------------------------------------
# MLArtifact.from_dict / to_dict roundtrip
# ---------------------------------------------------------------------------

class TestMLArtifactSerialize(unittest.TestCase):

    def _make(self) -> MLArtifact:
        return MLArtifact(
            version="1.0.0",
            model=EloModel.from_dict({}),
            calibrator=PlattCalibrator(a=1.2, b=-0.1),
            meta={"info": "test", "n": 42},
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("version", "model", "calibrator", "meta"):
            self.assertIn(key, d)

    def test_version_preserved(self):
        art = self._make()
        self.assertEqual(art.to_dict()["version"], "1.0.0")

    def test_meta_preserved(self):
        art = self._make()
        d = art.to_dict()
        self.assertEqual(d["meta"]["info"], "test")
        self.assertEqual(d["meta"]["n"], 42)

    def test_calibrator_a_preserved(self):
        art = self._make()
        d = art.to_dict()
        self.assertAlmostEqual(d["calibrator"]["a"], 1.2, places=5)

    def test_from_dict_roundtrip_version(self):
        art = MLArtifact.from_dict(_minimal_artifact_dict())
        self.assertEqual(art.version, "1.2.3")

    def test_from_dict_roundtrip_calibrator(self):
        d = _minimal_artifact_dict()
        d["calibrator"] = {"a": 0.95, "b": 0.05}
        art = MLArtifact.from_dict(d)
        self.assertAlmostEqual(art.calibrator.a, 0.95)
        self.assertAlmostEqual(art.calibrator.b, 0.05)

    def test_from_dict_missing_keys_returns_defaults(self):
        art = MLArtifact.from_dict({})
        self.assertIsNotNone(art.model)
        self.assertIsNotNone(art.calibrator)
        self.assertIsInstance(art.meta, dict)

    def test_full_roundtrip(self):
        original = self._make()
        restored = MLArtifact.from_dict(original.to_dict())
        self.assertEqual(original.version, restored.version)
        self.assertAlmostEqual(original.calibrator.a, restored.calibrator.a, places=8)
        self.assertAlmostEqual(original.calibrator.b, restored.calibrator.b, places=8)

    def test_meta_is_dict(self):
        art = MLArtifact.from_dict(_minimal_artifact_dict())
        self.assertIsInstance(art.meta, dict)


# ---------------------------------------------------------------------------
# MLArtifact.save_json / load_json
# ---------------------------------------------------------------------------

class TestMLArtifactIO(unittest.TestCase):

    def _make(self) -> MLArtifact:
        return MLArtifact(
            version="2.0.0",
            model=EloModel.from_dict({}),
            calibrator=PlattCalibrator(a=1.0, b=0.0),
            meta={"test": True},
        )

    def test_save_creates_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self._make().save_json(path)
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            self.assertIn("version", d)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_json_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            original = self._make()
            original.save_json(path)
            restored = MLArtifact.load_json(path)
            self.assertEqual(original.version, restored.version)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
