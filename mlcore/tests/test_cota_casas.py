"""A cota da OddsPapi no robo das casas (15/11/2026): 250 chamadas/mes para LoL, Dota e CS."""
import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]
_ESP = "dota" if (_RAIZ / "dota_paths.py").exists() else "lol"


def _robo():
    argv = sys.argv
    sys.argv = ["coletar_casas.py", _ESP]
    try:
        spec = importlib.util.spec_from_file_location("coletar_casas_cota", _RAIZ / "tools" / "coletar_casas.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = argv


class TestCota(unittest.TestCase):
    def test_orcamento(self):
        r = _robo()
        self.assertEqual(r.cota({"subscriptions": [{"is_active": True, "request_limit": 250, "request_count": 301}]}),
                         (250, 301))
        self.assertEqual(r.orcamento(250, 0, date(2026, 12, 1)), 2)
        self.assertEqual(r.orcamento(250, 301, date(2026, 11, 15)), 0)
        self.assertEqual(r.orcamento(None, None, date(2026, 12, 1)), 2)

    def test_plano(self):
        r = _robo()
        fx = [{"fixtureId": f"f{i}", "tournamentId": 10 + i // 2, "startTime": f"2026-12-01T{10 + i:02d}:00"}
              for i in range(8)]
        self.assertEqual(r.plano_de_coleta(fx[:2], 3), ("jogos", fx[:2]))
        self.assertEqual(r.plano_de_coleta(fx, 1), ("lotes", [[10, 11, 12]]))
        self.assertEqual(r.plano_de_coleta(fx, 0), ("nada", []))

    def test_catalogo_versionado(self):
        c = _robo().carregar_catalogo()
        self.assertGreater(len(c), 1000)
        self.assertIn(("kills", "total"), {(v[0], v[1]) for v in c.values() if isinstance(v, tuple)})


if __name__ == "__main__":
    unittest.main()
