# -*- coding: utf-8 -*-
"""Calibração global do sd do TEMPO (gate ACCEPT 2026-07-17, F5).

Trava a constante validada no cego (mudou = re-rodar
tools/eval_totals_league_dispersion.py e re-pinar aqui com o novo número)
e o comportamento: só o sd do tempo escala; média e demais mercados intactos.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlcore.lines_engine import (  # noqa: E402
    TIME_SD_CALIBRATION,
    MatchupTotals,
    apply_time_sd_calibration,
)


class TestTimeSdCalibration(unittest.TestCase):
    def test_constant_locked_to_gate_value(self) -> None:
        self.assertAlmostEqual(TIME_SD_CALIBRATION, 1.4461, places=4)

    def test_scales_only_time_sd(self) -> None:
        totals = {
            "time": MatchupTotals(mean=32.0, sd=2.0, dist="normal"),
            "kills": MatchupTotals(mean=27.0, sd=4.0, dist="normal"),
            "dragons": MatchupTotals(mean=4.6, sd=2.14, dist="poisson"),
        }
        out = apply_time_sd_calibration(totals)
        self.assertAlmostEqual(out["time"].sd, 2.0 * TIME_SD_CALIBRATION, places=6)
        self.assertAlmostEqual(out["time"].mean, 32.0, places=6)
        self.assertIs(out["kills"], totals["kills"])
        self.assertIs(out["dragons"], totals["dragons"])

    def test_input_dict_not_mutated(self) -> None:
        totals = {"time": MatchupTotals(mean=30.0, sd=1.5, dist="normal")}
        apply_time_sd_calibration(totals)
        self.assertAlmostEqual(totals["time"].sd, 1.5, places=6)

    def test_noop_without_time(self) -> None:
        totals = {"kills": MatchupTotals(mean=27.0, sd=4.0, dist="normal")}
        self.assertEqual(apply_time_sd_calibration(totals), totals)

    def test_noop_on_non_normal_or_nan(self) -> None:
        t_poisson = {"time": MatchupTotals(mean=30.0, sd=1.5, dist="poisson")}
        self.assertEqual(apply_time_sd_calibration(t_poisson), t_poisson)
        t_nan = {"time": MatchupTotals(mean=30.0, sd=float("nan"), dist="normal")}
        self.assertEqual(apply_time_sd_calibration(t_nan), t_nan)


if __name__ == "__main__":
    unittest.main()
