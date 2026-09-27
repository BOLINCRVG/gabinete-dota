# -*- coding: utf-8 -*-
"""Testes do baixador do Oracle's Elixir (fase 1 da campanha: ciclo de dados
1000% testado). O gdown é substituído por um dublê via sys.modules — nenhum
teste toca rede ou o CSV real."""
import importlib
import sys
import types
import unittest
from unittest import mock

import pandas as pd

from mlcore.tests._workspace_temp import workspace_tempdir
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol



def _csv_text(n_games: int, dmax: str) -> str:
    rows = ["gameid,date"]
    for i in range(n_games):
        rows.append(f"g{i},{dmax}")
    return "\n".join(rows)


class _FakeEntry:
    def __init__(self, path, fid="fid123"):
        self.path = path
        self.id = fid


def _fake_gdown(*, files, download_result=None, download_exc=None):
    mod = types.ModuleType("gdown")
    exc_mod = types.ModuleType("gdown.exceptions")

    class FileURLRetrievalError(Exception):
        pass

    exc_mod.FileURLRetrievalError = FileURLRetrievalError
    mod.exceptions = exc_mod

    def download_folder(url=None, skip_download=True, quiet=True):
        return files

    def download(id=None, output=None, quiet=True):
        if download_exc is not None:
            raise download_exc
        if download_result is not None:
            with open(output, "w", encoding="utf-8") as f:
                f.write(download_result)
            return output
        return None

    mod.download_folder = download_folder
    mod.download = download
    return mod, FileURLRetrievalError


class TestOeDownload(unittest.TestCase):
    def _run(self, tmp, gdown_mod, year=2026):
        """Importa o tool com CSV_DIR apontado pro tempdir e gdown dublê."""
        with mock.patch.dict(sys.modules, {"gdown": gdown_mod,
                                           "gdown.exceptions": gdown_mod.exceptions}):
            spec = importlib.util.spec_from_file_location(
                "oe_download_test", "tools/oe_download.py")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            m.CSV_DIR = tmp
            with mock.patch.object(m, "datetime") as dt:
                dt.now.return_value.year = year
                return m, m.main()

    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_update_writes_atomically_when_newer(self):
        gd, _ = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
            download_result=_csv_text(10, "2026-07-16"),
        )
        with workspace_tempdir("oe_download") as td:
            from pathlib import Path
            tmp = Path(td)
            target = tmp / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
            target.write_text(_csv_text(5, "2026-07-10"), encoding="utf-8")
            _, rc = self._run(tmp, gd)
            self.assertEqual(rc, 0)
            df = pd.read_csv(target)
            self.assertEqual(df["gameid"].nunique(), 10)

    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_regression_keeps_current_and_fails_loud(self):
        gd, _ = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
            download_result=_csv_text(3, "2026-07-01"),
        )
        with workspace_tempdir("oe_download") as td:
            from pathlib import Path
            tmp = Path(td)
            target = tmp / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
            target.write_text(_csv_text(5, "2026-07-10"), encoding="utf-8")
            _, rc = self._run(tmp, gd)
            self.assertEqual(rc, 1)  # RC_FAIL: regressão nunca é transitório
            df = pd.read_csv(target)
            self.assertEqual(df["gameid"].nunique(), 5)  # atual intocado

    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_quota_is_tempfail_75(self):
        gd, exc_cls = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
        )
        gd2, _ = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
            download_exc=exc_cls("Too many users"),
        )
        gd2.exceptions.FileURLRetrievalError = exc_cls
        with workspace_tempdir("oe_download") as td:
            from pathlib import Path
            _, rc = self._run(Path(td), gd2)
            self.assertEqual(rc, 75)

    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_year_fallback_january(self):
        """Virada de ano: sem arquivo 2027, segue no 2026 sem falhar."""
        gd, _ = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
            download_result=_csv_text(4, "2026-12-30"),
        )
        with workspace_tempdir("oe_download") as td:
            from pathlib import Path
            m, rc = self._run(Path(td), gd, year=2027)
            self.assertEqual(rc, 0)
            self.assertTrue((Path(td) / "2026_LoL_esports_match_data_from_OraclesElixir.csv").exists())

    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_garbage_download_is_tempfail_not_replacement(self):
        gd, _ = _fake_gdown(
            files=[_FakeEntry("2026_LoL_esports_match_data_from_OraclesElixir.csv")],
            download_result="<html>Quota exceeded</html>",
        )
        with workspace_tempdir("oe_download") as td:
            from pathlib import Path
            tmp = Path(td)
            target = tmp / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
            target.write_text(_csv_text(5, "2026-07-10"), encoding="utf-8")
            _, rc = self._run(tmp, gd)
            self.assertEqual(rc, 75)
            df = pd.read_csv(target)
            self.assertEqual(df["gameid"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
