"""Tests for app_core/live_models_status_core.py."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app_core.live_models_status_core import (
    FreshnessStatus,
    assess_freshness,
    build_retrain_command,
    csv_signatures,
    manifest_path,
    read_manifest,
    retrain_status,
    update_manifest_after_training,
    write_manifest,
)


def _make_fake_csv(tmpdir: Path, name: str, n_games: int = 100) -> Path:
    p = tmpdir / name
    rows = [f"g{i},{i},XXX,2025-01-{1+i%28:02d},1,1,1,team,1,30" for i in range(n_games)]
    header = "gameid,participantid,league,date,year,split,playoffs,teamname,result,gamelength"
    p.write_text(header + "\n" + "\n".join(rows), encoding="utf-8")
    return p


class TestCSVSignatures(unittest.TestCase):

    def test_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=50)
            sigs = csv_signatures([csv], count_games=True)
            self.assertEqual(len(sigs), 1)
            self.assertEqual(sigs[0]["n_games"], 50)
            self.assertIsNotNone(sigs[0]["size"])

    def test_missing_file_no_size(self):
        sigs = csv_signatures(["__nonexistent__.csv"])
        self.assertEqual(len(sigs), 1)
        self.assertIsNone(sigs[0]["size"])


class TestManifestIO(unittest.TestCase):

    def test_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            payload = {"version": "live_models_manifest_v1", "n_games_at_train": 1000}
            write_manifest(md, payload)
            self.assertEqual(read_manifest(md)["n_games_at_train"], 1000)

    def test_missing_manifest_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_manifest(tmp), {})

    def test_manifest_path_format(self):
        p = manifest_path("/some/dir")
        self.assertTrue(str(p).endswith("manifest.json"))


class TestUpdateManifest(unittest.TestCase):

    def test_updates_manifest_with_signatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=80)
            md = tmp_p / "live_models"
            md.mkdir()
            manifest = update_manifest_after_training(md, [csv])
            self.assertEqual(manifest["n_games_at_train"], 80)
            self.assertIn("last_trained_at", manifest)


class TestAssessFreshness(unittest.TestCase):

    def test_no_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "missing_subdir"
            fs = assess_freshness(md, [])
            self.assertEqual(fs.state, "no_models")

    def test_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp)
            fs = assess_freshness(md, [])
            self.assertEqual(fs.state, "no_manifest")

    def test_fresh_after_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            fs = assess_freshness(md, [csv])
            self.assertEqual(fs.state, "fresh")
            self.assertEqual(fs.n_new_games, 0)

    def test_stale_after_new_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            # Recreate CSV with more games
            _make_fake_csv(tmp_p, "data.csv", n_games=500)
            fs = assess_freshness(md, [csv], stale_n_games=200)
            self.assertIn(fs.state, ("stale", "very_stale"))

    def test_very_stale_after_huge_new_games(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = _make_fake_csv(tmp_p, "data.csv", n_games=100)
            md = tmp_p / "live_models"
            md.mkdir()
            update_manifest_after_training(md, [csv])
            _make_fake_csv(tmp_p, "data.csv", n_games=1500)
            fs = assess_freshness(md, [csv], stale_n_games=200, very_stale_n_games=1000)
            self.assertEqual(fs.state, "very_stale")


class TestBuildRetrainCommand(unittest.TestCase):

    def test_command_contains_required_args(self):
        # 2026-07: o retreino aponta pro ORQUESTRADOR (fases 6/7/8/10 + manifest),
        # nao mais pro run_phase6_live_gate isolado (--save-models-dir).
        cmd = build_retrain_command(val_cutoff="2026-04-30")
        self.assertIn("--val-cutoff", cmd)
        self.assertIn("2026-04-30", cmd)
        self.assertTrue(any("retrain_live_models_all" in str(part) for part in cmd))
        self.assertIn("--models-root", cmd)


class TestRetrainStatus(unittest.TestCase):

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = retrain_status(tmp)
            self.assertIn("running", s)
            self.assertIn("log_exists", s)
            self.assertFalse(s["running"])


if __name__ == "__main__":
    unittest.main()
