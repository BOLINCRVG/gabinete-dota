"""O vigia dos dados (15/11/2026): falha quando uma fonte fica velha."""
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location("vigia_teste", Path(__file__).resolve().parents[2] / "tools" / "vigia.py")
vigia = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vigia)


class TestVigia(unittest.TestCase):
    def test_conferir(self):
        agora = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
        self.assertAlmostEqual(vigia.idade_h("2026-11-15T03:00:00-03:00", agora), 6.0)
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            (raiz / "novo.json").write_text(json.dumps({"lido_em": "2026-11-15T10:00:00+00:00"}), encoding="utf-8")
            (raiz / "velho.json").write_text(json.dumps({"lido_em": "2026-11-05T10:00:00+00:00"}), encoding="utf-8")
            r = vigia.conferir(["novo.json:lido_em:12", "velho.json:lido_em:12", "velho.json:lido_em:12:opcional",
                                "nao_existe.json:lido_em:12"], agora, raiz)
        self.assertEqual([x[3] for x in r], ["ok", "velho", "velho (opcional)", "ilegivel"])


if __name__ == "__main__":
    unittest.main()
