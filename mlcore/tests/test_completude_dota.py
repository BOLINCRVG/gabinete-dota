# -*- coding: utf-8 -*-
"""Completude da base do Dota contra a Liquipedia (24/11/2026)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import completude_dota as CD  # noqa: E402

BASE = pd.DataFrame([{"gameid": "g1", "date": "2026-11-18 18:31:04", "teamname": t, "position": "team"}
                     for t in ("Team Spirit", "PVISION")])
RES = {"Team Spirit": "Team Spirit", "PARIVISION": "PVISION", "Aurora Gaming": "Aurora Gaming",
       "LGD Gaming": "LGD Gaming"}.get


class TestCompletudeDota(unittest.TestCase):

    def test_acha_o_que_falta(self):
        series = [{"time_a": "Team Spirit", "time_b": "PARIVISION", "inicio_utc": "2026-11-18 17:00:00+00", "placar": "2-1"},
                  {"time_a": "Aurora Gaming", "time_b": "LGD Gaming", "inicio_utc": "2026-11-23 16:50:00+00", "placar": "2-0"},
                  {"time_a": "Time Novo", "time_b": "LGD Gaming", "inicio_utc": "2026-11-23 10:00:00+00", "placar": "2-0"}]
        f = CD.conferir(series, BASE, lambda n: RES(n) or "")
        self.assertEqual([x["a"] for x in f], ["Aurora Gaming", "Time Novo"])
        self.assertIn("ainda não trouxe", f[0]["motivo"])
        self.assertIn("Time Novo", f[1]["motivo"])

    def test_sem_placar_e_wo_ou_adiada(self):
        series = [{"time_a": "Aurora Gaming", "time_b": "LGD Gaming", "inicio_utc": "2026-11-23 16:50:00+00", "placar": ""}]
        f = CD.conferir(series, BASE, lambda n: RES(n) or "")
        self.assertIn("W.O. ou adiada", f[0]["motivo"])


if __name__ == "__main__":
    unittest.main()
