import hashlib
import json
import unittest
from pathlib import Path

from mlcore.regression_fixtures import resolve_players_fixture_snapshot


class PlayersFusionRegressionTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def test_fixture_hashes_are_current(self):
        root = Path(__file__).resolve().parents[2]
        fx = root / "mlcore" / "tests" / "fixtures" / "players_fusion_regression.json"
        art = root / "ml_artifact.json"
        part = root / "players_artifact.json"
        csvp = root / "CSV_DIARIO" / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
        if not fx.exists() or not art.exists() or not part.exists() or not csvp.exists():
            self.skipTest("fixture/artifacts/csv missing")

        data = json.loads(fx.read_text(encoding="utf-8"))
        art_sha = self._sha256(art)
        part_sha = self._sha256(part)
        csv_sha = self._sha256(csvp)
        snap = resolve_players_fixture_snapshot(data, art_sha, part_sha, csv_sha)
        self.assertIsNotNone(
            snap,
            "Fixture hashes mismatch. Run tools/update_regression_fixtures.py",
        )

    def test_players_fused_differs_from_team_only_on_key_matchups(self):
        root = Path(__file__).resolve().parents[2]
        fx = root / "mlcore" / "tests" / "fixtures" / "players_fusion_regression.json"
        if not fx.exists():
            self.skipTest("fixture missing")
        data = json.loads(fx.read_text(encoding="utf-8"))
        art = root / "ml_artifact.json"
        part = root / "players_artifact.json"
        csvp = root / "CSV_DIARIO" / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
        if not art.exists() or not part.exists() or not csvp.exists():
            self.skipTest("artifacts/csv missing")
        snap = resolve_players_fixture_snapshot(
            data,
            self._sha256(art),
            self._sha256(part),
            self._sha256(csvp),
        )
        self.assertIsNotNone(
            snap,
            "Fixture hashes mismatch. Run tools/update_regression_fixtures.py",
        )
        rows = (snap or {}).get("matchups") or []
        self.assertGreaterEqual(len(rows), 3)

        moved = 0
        total_abs_delta = 0.0
        for r in rows:
            t = (r.get("team_only") or {})
            f = (r.get("players_fused") or {})
            pt = float(t.get("p_map"))
            pf = float(f.get("p_map"))
            self.assertTrue(0.0 < pt < 1.0)
            self.assertTrue(0.0 < pf < 1.0)
            total_abs_delta += abs(pf - pt)
            if abs(pf - pt) >= 0.01:
                moved += 1
        self.assertGreaterEqual(moved, 1, "players fusion should move at least 1 key matchup by >=1pp")
        self.assertGreaterEqual(total_abs_delta, 0.02, "players fusion total impact should stay material across key matchups")


if __name__ == "__main__":
    unittest.main()
