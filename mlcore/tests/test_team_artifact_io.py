import json
import unittest
from pathlib import Path

from mlcore.team_artifact_io import load_team_state_artifact, save_team_state_artifact


class TestLoadTeamStateArtifact(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        result = load_team_state_artifact("/nonexistent/path/artifact.json")
        self.assertEqual(result, {})

    def test_loads_valid_json(self, tmp_path=None):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"team": "T1", "wins": 3}, f)
            fname = f.name
        try:
            result = load_team_state_artifact(fname)
            self.assertEqual(result["team"], "T1")
            self.assertEqual(result["wins"], 3)
        finally:
            os.unlink(fname)

    def test_non_dict_json_returns_empty(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
            fname = f.name
        try:
            result = load_team_state_artifact(fname)
            self.assertEqual(result, {})
        finally:
            os.unlink(fname)

    def test_corrupt_json_returns_empty(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{not valid json")
            fname = f.name
        try:
            result = load_team_state_artifact(fname)
            self.assertEqual(result, {})
        finally:
            os.unlink(fname)


class TestSaveTeamStateArtifact(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "artifact.json"
            data = {"league": "LCK", "patch": "14.10", "score": 0.75}
            save_team_state_artifact(path, data)
            self.assertTrue(path.exists())
            loaded = load_team_state_artifact(path)
            self.assertEqual(loaded["league"], "LCK")
            self.assertAlmostEqual(loaded["score"], 0.75)

    def test_creates_parent_directories(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a" / "b" / "c" / "artifact.json"
            save_team_state_artifact(path, {"x": 1})
            self.assertTrue(path.exists())

    def test_overwrites_existing_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "artifact.json"
            save_team_state_artifact(path, {"v": 1})
            save_team_state_artifact(path, {"v": 2})
            loaded = load_team_state_artifact(path)
            self.assertEqual(loaded["v"], 2)

    def test_unicode_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "artifact.json"
            save_team_state_artifact(path, {"name": "팀리퀴드"})
            loaded = load_team_state_artifact(path)
            self.assertEqual(loaded["name"], "팀리퀴드")


if __name__ == "__main__":
    unittest.main()
