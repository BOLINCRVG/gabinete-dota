import json
import tempfile
import unittest
from pathlib import Path

from mlcore.update_incremental import (
    _apply_promotion_reset,
    _extract_model_team_meta,
    _parse_dt,
    _resolve_csv_path,
    _safe_int,
    _safe_write_json,
    _should_skip_artifact_write,
)


class _ModelStub:
    def __init__(self) -> None:
        self.ratings = {"Blue": 12.5, "Red": -4.0}
        self.games_played = {"Blue": 31, "Red": 18}


class UpdateIncrementalTests(unittest.TestCase):
    def test_parse_dt_accepts_z_and_naive(self) -> None:
        dt_z = _parse_dt("2026-03-19T21:47:43Z")
        dt_naive = _parse_dt("2026-03-19T21:47:43")
        self.assertIsNotNone(dt_z)
        self.assertIsNotNone(dt_naive)
        self.assertEqual(dt_z.isoformat(), "2026-03-19T21:47:43+00:00")
        self.assertEqual(dt_naive.isoformat(), "2026-03-19T21:47:43+00:00")
        self.assertIsNone(_parse_dt("not-a-date"))

    def test_resolve_csv_path_directory_picks_newest_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "older.csv"
            newer = root / "newer.csv"
            older.write_text("a", encoding="utf-8")
            newer.write_text("b", encoding="utf-8")
            # datas EXPLICITAS: criados no mesmo instante, no Linux os dois
            # podem ter a mesma data e o "mais novo" vira sorteio
            import os, time
            agora = time.time()
            os.utime(older, (agora - 100, agora - 100))
            os.utime(newer, (agora, agora))
            self.assertEqual(_resolve_csv_path(root), newer)

    def test_safe_write_json_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "artifact.json"
            target.write_text(json.dumps({"v": 1}), encoding="utf-8")
            _safe_write_json(target, {"v": 2})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["v"], 2)
            bak = target.with_suffix(".json.bak")
            self.assertTrue(bak.exists())
            self.assertEqual(json.loads(bak.read_text(encoding="utf-8"))["v"], 1)

    def test_extract_model_team_meta_filters_non_dict_entries(self) -> None:
        class _MetaStub:
            team_meta = {"A": {"league": "LCK", "tier": "T1"}, "B": "bad", "C": None}

        got = _extract_model_team_meta(_MetaStub())
        self.assertEqual(got, {"A": {"league": "LCK", "tier": "T1"}})

    def test_apply_promotion_reset_only_on_stronger_tier(self) -> None:
        model = _ModelStub()
        changed = _apply_promotion_reset(model, team="Blue", prev_tier="T2", new_tier="T1")
        self.assertTrue(changed)
        self.assertEqual(model.ratings["Blue"], 0.0)
        self.assertEqual(model.games_played["Blue"], 0)

        model2 = _ModelStub()
        changed2 = _apply_promotion_reset(model2, team="Blue", prev_tier="T1", new_tier="T2")
        self.assertFalse(changed2)
        self.assertEqual(model2.ratings["Blue"], 12.5)
        self.assertEqual(model2.games_played["Blue"], 31)

    def test_safe_int_handles_float_like_strings(self) -> None:
        self.assertEqual(_safe_int("1.0"), 1)
        self.assertEqual(_safe_int(None, 7), 7)

    def test_skip_artifact_write_only_when_no_new_and_no_backfill(self) -> None:
        # 0 jogos novos e sem backfill -> pula a escrita (preserva mtime).
        self.assertTrue(_should_skip_artifact_write(0, False))
        # com jogos novos -> escreve.
        self.assertFalse(_should_skip_artifact_write(3, False))
        # 0 novos mas com backfill (correcoes) -> escreve (precisa reaplicar).
        self.assertFalse(_should_skip_artifact_write(0, True))


if __name__ == "__main__":
    unittest.main()
