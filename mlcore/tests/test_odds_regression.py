import json
import hashlib
import unittest
from pathlib import Path

from mlcore.artifact import MLArtifact
from mlcore.bo import prob_win_series
from mlcore.regression_fixtures import resolve_odds_fixture_snapshot
from mlcore.utils import prob_to_decimal_odds
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol



class OddsRegressionTest(unittest.TestCase):
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
        self.fixture_path = Path(__file__).resolve().parent / "fixtures" / "odds_regression.json"

        if not self.artifact_path.exists():
            self.skipTest("ml_artifact.json not found in project root.")
        if not self.fixture_path.exists():
            self.skipTest("odds_regression fixture not found.")

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
        p_raw = 0.5 * (p_raw_f + (1.0 - p_raw_r))

        p_cal_f = float(self.art.calibrator.predict_proba(p_raw_f))
        p_cal_r = float(self.art.calibrator.predict_proba(p_raw_r))
        p_map = 0.5 * (p_cal_f + (1.0 - p_cal_r))

        p_bo3 = float(prob_win_series(p_map, 3))
        p_bo5 = float(prob_win_series(p_map, 5))
        return {
            "p_raw": p_raw,
            "p_map": p_map,
            "odd_map_a": float(prob_to_decimal_odds(p_map)),
            "odd_map_b": float(prob_to_decimal_odds(1.0 - p_map)),
            "p_bo3": p_bo3,
            "odd_bo3_a": float(prob_to_decimal_odds(p_bo3)),
            "p_bo5": p_bo5,
            "odd_bo5_a": float(prob_to_decimal_odds(p_bo5)),
        }

    @so_no_lol("fotografia do modelo do LoL (o Dota tem outro modelo)")
    def test_snapshot_matchups(self):
        # If artifact is intentionally retrained, update fixture with the same script used by the agent.
        tol = 1e-9
        for row in self.fixture.get("matchups", []):
            got = self._predict(str(row["team_a"]), str(row["team_b"]))
            self.assertAlmostEqual(got["p_map"], float(row["p_map"]), delta=tol)
            self.assertAlmostEqual(got["odd_map_a"], float(row["odd_map_a"]), delta=tol)
            self.assertAlmostEqual(got["odd_map_b"], float(row["odd_map_b"]), delta=tol)
            self.assertAlmostEqual(got["p_bo3"], float(row["p_bo3"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo3_a"], float(row["odd_bo3_a"]), delta=tol)
            self.assertAlmostEqual(got["p_bo5"], float(row["p_bo5"]), delta=tol)
            self.assertAlmostEqual(got["odd_bo5_a"], float(row["odd_bo5_a"]), delta=tol)

    @so_no_lol("fotografia do modelo do LoL (o Dota tem outro modelo)")
    def test_probability_consistency(self):
        for row in self.fixture.get("matchups", []):
            team_a = str(row["team_a"])
            team_b = str(row["team_b"])
            ab = self._predict(team_a, team_b)
            ba = self._predict(team_b, team_a)
            self.assertAlmostEqual(ab["p_map"] + ba["p_map"], 1.0, delta=1e-9)
            self.assertAlmostEqual(ab["p_bo3"] + (1.0 - ab["p_bo3"]), 1.0, delta=1e-12)
            self.assertAlmostEqual(ab["p_bo5"] + (1.0 - ab["p_bo5"]), 1.0, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
