import tempfile
import unittest
from pathlib import Path

from app_core.csv_prepare_core import csv_diag_path, csv_header_check_path


class TestCsvHeaderCheckPath(unittest.TestCase):

    def _make_csv(self, content: str) -> str:
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        Path(path).write_text(content, encoding="utf-8")
        return path

    def test_valid_headers_ok(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("gameid,participantid,teamname,date\n1,100,A,2026-01-01\n", encoding="utf-8")
            out = csv_header_check_path(str(p))
        self.assertTrue(out["ok"])
        self.assertEqual(out["missing_required"], [])
        self.assertEqual(out["n_columns"], 4)

    def test_missing_required_column_reported(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("gameid,teamname,date\n1,A,2026-01-01\n", encoding="utf-8")
            out = csv_header_check_path(str(p))
        self.assertTrue(out["ok"])
        self.assertIn("participantid", out["missing_required"])

    def test_missing_file_returns_error(self):
        out = csv_header_check_path("__nonexistent__.csv")
        self.assertFalse(out["ok"])
        self.assertIn("error", out)

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            out = csv_header_check_path(str(p))
        self.assertIsInstance(out, dict)

    def test_columns_list_present(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("gameid,participantid,teamname,date\n1,100,A,2026-01-01\n", encoding="utf-8")
            out = csv_header_check_path(str(p))
        self.assertIn("gameid", out["columns"])


class TestCsvDiagPath(unittest.TestCase):

    def _make_csv(self, lines):
        return "\n".join(lines)

    def test_rows_and_unique_gameids(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text(
                "\n".join([
                    "gameid,participantid,teamname,date",
                    "1,100,A,2026-01-01",
                    "1,200,B,2026-01-01",
                    "2,100,C,2026-01-01",
                ]),
                encoding="utf-8",
            )
            out = csv_diag_path(str(p))
        self.assertEqual(out["rows"], 3)
        self.assertEqual(out["unique_gameids"], 2)

    def test_team_rows_and_completeness(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text(
                "\n".join([
                    "gameid,participantid,teamname,date",
                    "1,100,A,2026-01-01",
                    "1,200,B,2026-01-01",
                    "2,100,C,2026-01-01",
                ]),
                encoding="utf-8",
            )
            out = csv_diag_path(str(p))
        self.assertEqual(out["team_rows"], 3)
        self.assertEqual(out["games_complete"], 1)
        self.assertEqual(out["games_incomplete"], 1)

    def test_missing_file_returns_empty(self):
        out = csv_diag_path("__nonexistent__.csv")
        self.assertEqual(out, {})

    def test_returns_dict(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.csv"
            p.write_text("gameid,participantid\n1,100\n", encoding="utf-8")
            out = csv_diag_path(str(p))
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
