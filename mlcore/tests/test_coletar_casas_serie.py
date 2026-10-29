"""O robo das casas le os mercados da SERIE do catalogo (29/10/2026)."""
import importlib.util
import sys
import unittest
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "tools" / "coletar_casas.py"


def _robo():
    argv = sys.argv
    sys.argv = ["coletar_casas.py", "lol"]
    try:
        spec = importlib.util.spec_from_file_location("coletar_casas_teste", _P)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = argv


def _m(mid, nome, periodo, handicap, outs):
    return {"marketId": mid, "marketName": nome, "period": periodo, "handicap": handicap,
            "marketLength": len(outs), "outcomes": [{"outcomeId": i, "outcomeName": n} for i, n in outs]}


class TestCatalogoDaSerie(unittest.TestCase):
    def test_mercados_da_serie(self):
        r = _robo()
        cat = r.catalogo_de_linhas([
            _m(1, "Total Maps Over Under", "result", 2.5, [(11, "Over"), (12, "Under")]),
            _m(2, "Kills - Over Under", "result", 55.5, [(21, "Over"), (22, "Under")]),
            _m(3, "Correct Map Score", "result", 0.0, [(31, "2:0"), (32, "2:1"), (33, "1:2"), (34, "0:2")]),
            _m(4, "Kills - Over Under First Map", "p1", 27.5, [(41, "Over"), (42, "Under")]),
            _m(5, "Winner", "result", 0.0, [(51, "1"), (52, "2")]),
        ])
        self.assertEqual(cat["1"], ("maps", "total", 0, 2.5, ("11", "12"), None))
        self.assertEqual(cat["2"], ("kills", "total", 0, 55.5, ("21", "22"), None))
        self.assertEqual(sorted(c[3] for c in cat["3"]), [1.0, 2.0])
        self.assertEqual(cat["4"][2], 1)
        self.assertNotIn("5", cat)
        odds = {"bookmakerOdds": {"stake": {"markets": {
            "3": {"outcomes": {k: {"players": {"0": {"price": v}}} for k, v in
                               (("31", 2.1), ("32", 4.0), ("33", 5.0), ("34", 6.5))}}}}}}
        lin = r.linhas_das_casas(odds, cat)["stake"]
        self.assertIn(["maps", "placar", 0, 2.0, 2.1, 6.5, None], lin)
        self.assertIn(["maps", "placar", 0, 1.0, 4.0, 5.0, None], lin)


if __name__ == "__main__":
    unittest.main()
