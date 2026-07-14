import math
import os
import tempfile
import unittest

import pandas as pd

from mlcore.io_oracle import (
    _coerce_bool,
    _resolve_optional_columns,
    _row_float,
    _row_int,
    _row_text,
    load_oracle_elixir_games,
)


class TestCoerceBool(unittest.TestCase):

    def test_true_strings(self):
        for v in ("1", "true", "True", "yes", "YES", "y"):
            self.assertTrue(_coerce_bool(v), msg=repr(v))

    def test_false_strings(self):
        for v in ("0", "false", "False", "no", "n", ""):
            self.assertFalse(_coerce_bool(v), msg=repr(v))

    def test_bool_passthrough(self):
        self.assertTrue(_coerce_bool(True))
        self.assertFalse(_coerce_bool(False))

    def test_nan_returns_false(self):
        self.assertFalse(_coerce_bool(float("nan")))

    def test_numeric_string_truthy(self):
        self.assertTrue(_coerce_bool("2"))

    def test_numeric_string_zero(self):
        self.assertFalse(_coerce_bool("0.0"))


class TestResolveOptionalColumns(unittest.TestCase):

    def test_canonical_name_included(self):
        use_cols, rename = _resolve_optional_columns(["dragons", "teamkills"])
        self.assertIn("dragons", use_cols)
        self.assertEqual(rename, {})

    def test_alias_resolved_to_canonical(self):
        use_cols, rename = _resolve_optional_columns(["void_grubs"])
        self.assertIn("void_grubs", use_cols)
        self.assertEqual(rename.get("void_grubs"), "voidgrubs")

    def test_unknown_column_ignored(self):
        use_cols, rename = _resolve_optional_columns(["unknown_col"])
        self.assertEqual(use_cols, [])
        self.assertEqual(rename, {})

    def test_empty_input(self):
        use_cols, rename = _resolve_optional_columns([])
        self.assertEqual(use_cols, [])
        self.assertEqual(rename, {})

    def test_returns_tuple(self):
        result = _resolve_optional_columns(["dragons"])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestRowHelpers(unittest.TestCase):

    def _row(self, **kwargs):
        return pd.Series(kwargs)

    def test_row_float_present(self):
        row = self._row(gold=1500.5)
        self.assertAlmostEqual(_row_float(row, "gold"), 1500.5)

    def test_row_float_missing(self):
        row = self._row()
        self.assertIsNone(_row_float(row, "gold"))

    def test_row_float_nan(self):
        row = self._row(gold=float("nan"))
        self.assertIsNone(_row_float(row, "gold"))

    def test_row_float_string_number(self):
        row = self._row(gold="200.5")
        self.assertAlmostEqual(_row_float(row, "gold"), 200.5)

    def test_row_int_present(self):
        row = self._row(towers=5)
        self.assertEqual(_row_int(row, "towers"), 5)

    def test_row_int_missing(self):
        row = self._row()
        self.assertIsNone(_row_int(row, "towers"))

    def test_row_int_nan(self):
        row = self._row(towers=float("nan"))
        self.assertIsNone(_row_int(row, "towers"))

    def test_row_int_from_float(self):
        row = self._row(towers=3.0)
        self.assertEqual(_row_int(row, "towers"), 3)

    def test_row_text_present(self):
        row = self._row(league="LCK")
        self.assertEqual(_row_text(row, "league"), "LCK")

    def test_row_text_stripped(self):
        row = self._row(league="  LCK  ")
        self.assertEqual(_row_text(row, "league"), "LCK")

    def test_row_text_missing(self):
        row = self._row()
        self.assertIsNone(_row_text(row, "league"))

    def test_row_text_nan(self):
        row = self._row(league=float("nan"))
        self.assertIsNone(_row_text(row, "league"))

    def test_row_text_empty_string(self):
        row = self._row(league="")
        self.assertIsNone(_row_text(row, "league"))


class TestIoOracle(unittest.TestCase):
    def _write_csv(self, body: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(body)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_loader_maps_void_grubs_alias_and_extended_columns(self) -> None:
        csv = self._write_csv(
            "\n".join(
                [
                    "gameid,league,year,split,playoffs,date,game,patch,participantid,teamname,result,gamelength,void_grubs,datacompleteness,firstmidtower,firsttothreetowers,turretplates,elementaldrakes,golddiffat20,xpdiffat20,csdiffat20",
                    "g1,LCK,2026,S1,0,2026-01-01,1,14.1,100,BlueTeam,1,1800,3,complete,1,1,5,3,2500,1900,40",
                    "g1,LCK,2026,S1,0,2026-01-01,1,14.1,200,RedTeam,0,1800,1,partial,0,0,2,1,-2500,-1900,-40",
                ]
            )
        )

        games = load_oracle_elixir_games(csv)
        self.assertEqual(len(games), 1)

        row = games.iloc[0]
        self.assertEqual(float(row["blue_voidgrubs"]), 3.0)
        self.assertEqual(float(row["red_voidgrubs"]), 1.0)
        self.assertEqual(str(row["blue_datacompleteness"]), "complete")
        self.assertEqual(str(row["red_datacompleteness"]), "partial")
        self.assertEqual(int(row["blue_firstmidtower"]), 1)
        self.assertEqual(int(row["red_firstmidtower"]), 0)
        self.assertEqual(int(row["blue_firsttothreetowers"]), 1)
        self.assertEqual(int(row["red_firsttothreetowers"]), 0)
        self.assertEqual(float(row["blue_turretplates"]), 5.0)
        self.assertEqual(float(row["red_turretplates"]), 2.0)
        self.assertEqual(float(row["blue_elementaldrakes"]), 3.0)
        self.assertEqual(float(row["red_elementaldrakes"]), 1.0)
        self.assertEqual(float(row["blue_golddiffat20"]), 2500.0)
        self.assertEqual(float(row["red_golddiffat20"]), -2500.0)
        self.assertFalse(bool(row["playoffs"]))


if __name__ == "__main__":
    unittest.main()
