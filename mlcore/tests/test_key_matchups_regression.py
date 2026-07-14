import hashlib
import json
import unittest
from pathlib import Path

from mlcore.artifact import MLArtifact
from mlcore.bo import prob_win_series
from mlcore.regression_fixtures import resolve_odds_fixture_snapshot
from mlcore.utils import prob_to_decimal_odds


class KeyMatchupsRegressionTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.artifact_path = root / "ml_artifact.json"
        self.fixture_path = Path(__file__).resolve().parent / "fixtures" / "key_matchups_regression.json"

        if not self.artifact_path.exists():
            self.skipTest("ml_artifact.json not found in project root.")
        if not self.fixture_path.exists():
            self.skipTest("key_matchups_regression fixture not found.")

        self.art = MLArtifact.load_json(str(self.artifact_path))
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        art_sha = self._sha256(self.artifact_path)
        snap = resolve_odds_fixture_snapshot(payload, art_sha)
        self.assertIsNotNone(
            snap,
            "Fixture artifact hash mismatch. Run tools/update_regression_fixtures.py",
        )
        self.fixture = dict(snap or {})

    def _predict(self, team_a: str, team_b: str):
        p_raw_f = float(self.art.model.p_blue(team_a, team_b))
        p_raw_r = float(self.art.model.p_blue(team_b, team_a))
        p_cal_f = float(self.art.calibrator.predict_proba(p_raw_f))
        p_cal_r = float(self.art.calibrator.predict_proba(p_raw_r))
        p_map = 0.5 * (p_cal_f + (1.0 - p_cal_r))

        p_bo3 = float(prob_win_series(p_map, 3))
        p_bo5 = float(prob_win_series(p_map, 5))
        return {
            "p_map": p_map,
            "odd_map_a": float(prob_to_decimal_odds(p_map)),
            "odd_map_b": float(prob_to_decimal_odds(1.0 - p_map)),
            "p_bo3": p_bo3,
            "odd_bo3_a": float(prob_to_decimal_odds(p_bo3)),
            "odd_bo3_b": float(prob_to_decimal_odds(1.0 - p_bo3)),
            "p_bo5": p_bo5,
            "odd_bo5_a": float(prob_to_decimal_odds(p_bo5)),
            "odd_bo5_b": float(prob_to_decimal_odds(1.0 - p_bo5)),
        }

    def test_key_matchups_snapshot(self):
        tol = 1e-9
        for row in self.fixture.get("matchups", []):
            got = self._predict(str(row["team_a"]), str(row["team_b"]))
            self.assertAlmostEqual(got["p_map"], float(row["p_map"]), delta=tol)
            self.assertAlmostEqual(got["odd_map_a"], float(row["odd_map_a"]), delta=tol)
            self.assertAlmostEqual(got["odd_map_b"], float(row["odd_map_b"]), delta=tol)
            self.assertAlmostEqual(got["p_bo3"], float(row["p_bo3"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo3_a"], float(row["odd_bo3_a"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo3_b"], float(row["odd_bo3_b"]), delta=tol)
            self.assertAlmostEqual(got["p_bo5"], float(row["p_bo5"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo5_a"], float(row["odd_bo5_a"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo5_b"], float(row["odd_bo5_b"]), delta=tol)


if __name__ == "__main__":
    unittest.main()
