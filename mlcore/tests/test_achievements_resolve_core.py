"""Tests for app_core/achievements_resolve_core.py."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_core.achievements_resolve_core import _resolve_relpath, ach_layer_for_artifact


# ---------------------------------------------------------------------------
# _resolve_relpath
# ---------------------------------------------------------------------------

class TestResolveRelpath(unittest.TestCase):

    def test_absolute_path_returned_as_is(self):
        import os
        result = _resolve_relpath("/some/base/file.json", os.path.abspath("/other/achievements.json"))
        self.assertEqual(result, os.path.abspath("/other/achievements.json"))

    def test_relative_resolved_against_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_file = str(Path(tmpdir) / "model" / "artifact.json")
            result = _resolve_relpath(base_file, "achievements.json")
            self.assertTrue(result.startswith(str(Path(tmpdir) / "model")))
            self.assertTrue(result.endswith("achievements.json"))

    def test_relative_path_stays_in_same_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = str(Path(tmpdir) / "artifact.json")
            result = _resolve_relpath(base, "ach.json")
            self.assertEqual(Path(result).parent, Path(tmpdir))
            self.assertEqual(Path(result).name, "ach.json")


# ---------------------------------------------------------------------------
# ach_layer_for_artifact — missing file paths
# ---------------------------------------------------------------------------

class TestAchLayerMissingFile(unittest.TestCase):

    def _make_art(self, ach_path="", weights=None):
        meta = {"achievements": {"path": ach_path, "weights_points": weights or {}}}
        return SimpleNamespace(meta=meta)

    def _ctx(self):
        return {
            "_file_sig": lambda p: p,
            "_load_achievements_cached": lambda sig: {},
        }

    def test_no_ach_path_no_file_returns_disabled(self):
        art = self._make_art(ach_path="")
        result = ach_layer_for_artifact(art, "/nonexistent/artifact.json", ctx=self._ctx())
        self.assertFalse(result["enabled"])
        self.assertEqual(result["team_idx"], {})

    def test_nonexistent_ach_path_returns_disabled(self):
        art = self._make_art(ach_path="nonexistent_file.json")
        result = ach_layer_for_artifact(art, "/nonexistent/artifact.json", ctx=self._ctx())
        self.assertFalse(result["enabled"])

    def test_result_has_required_keys(self):
        art = self._make_art()
        result = ach_layer_for_artifact(art, "/nonexistent/artifact.json", ctx=self._ctx())
        for key in ("enabled", "team_idx", "weights_points"):
            self.assertIn(key, result)

    def test_override_path_nonexistent_falls_through(self):
        art = self._make_art()
        result = ach_layer_for_artifact(
            art, "/nonexistent/artifact.json",
            ach_override_path="/nonexistent/ach_override.json",
            ctx=self._ctx(),
        )
        self.assertFalse(result["enabled"])


if __name__ == "__main__":
    unittest.main()
