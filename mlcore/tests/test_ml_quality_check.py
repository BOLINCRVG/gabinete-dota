"""Testes do alarme de qualidade pos-retrain (tools/ml_quality_check.py)."""
from __future__ import annotations

import unittest

try:
    from tools.ml_quality_check import FAIL_DELTA, REPIN_DELTA, WARN_DELTA, compare_to_baseline
except ImportError as _sem_ferramenta:  # ferramenta so do LoL: no Dota o modulo e pulado
    import unittest as _u
    raise _u.SkipTest("tools.ml_quality_check nao existe neste repositorio: " + str(_sem_ferramenta))


def _snap(ll: float) -> dict:
    return {"geral_cal": {"logloss": ll, "ece": 0.03, "acc": 0.67, "n": 5000}}


class TestCompareToBaseline(unittest.TestCase):
    BASE = _snap(0.6021)

    def test_incidente_holdout180_dispara_fail(self) -> None:
        # Valores reais do incidente 2026-06: baseline 0.6021 -> degradado 0.6907.
        r = compare_to_baseline(_snap(0.6907), self.BASE)
        self.assertEqual(r["verdict"], "fail")
        self.assertGreater(r["delta_logloss"], FAIL_DELTA)

    def test_ruido_e_ok(self) -> None:
        r = compare_to_baseline(_snap(0.6021 + WARN_DELTA / 2), self.BASE)
        self.assertEqual(r["verdict"], "ok")

    def test_faixa_warn(self) -> None:
        r = compare_to_baseline(_snap(0.6021 + (WARN_DELTA + FAIL_DELTA) / 2), self.BASE)
        self.assertEqual(r["verdict"], "warn")

    def test_melhora_repina_baseline(self) -> None:
        r = compare_to_baseline(_snap(0.6021 - 2 * REPIN_DELTA), self.BASE)
        self.assertEqual(r["verdict"], "ok_improved")

    def test_sem_baseline(self) -> None:
        r = compare_to_baseline(_snap(0.60), None)
        self.assertEqual(r["verdict"], "no_baseline")

    def test_snapshot_invalido(self) -> None:
        r = compare_to_baseline({"geral_cal": {}}, self.BASE)
        self.assertEqual(r["verdict"], "error")

    def test_limiar_exato_fail(self) -> None:
        r = compare_to_baseline(_snap(0.6021 + FAIL_DELTA), self.BASE)
        self.assertEqual(r["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
