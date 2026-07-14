import re
import unicodedata
import unittest

from app_core.navigation import TAB_ALL
from app_core.ui_labels import MARKET_TAB_NAMES


class UILabelsIntegrityTest(unittest.TestCase):
    @staticmethod
    def _norm(s: str) -> str:
        txt = unicodedata.normalize("NFKD", str(s or ""))
        txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
        return re.sub(r"[^A-Za-z0-9]+", "", txt).lower()

    @staticmethod
    def _looks_like_mojibake(s: str) -> bool:
        txt = str(s or "")
        if any(m in txt for m in ("\ufffd", "Ã", "Â")):
            return True
        if "??" in txt:
            return True
        return bool(re.search(r"[A-Za-zÀ-ÿ]\?[A-Za-zÀ-ÿ]", txt))

    def test_tabs_no_duplicates_by_semantics(self):
        norms = [self._norm(v) for v in TAB_ALL]
        self.assertEqual(len(norms), len(set(norms)), "Duplicate tabs by normalized label")
        # Consistency tab: renamed from 'Consistência de Mercado (Laplace)'
        # to 'Consistência das linhas' as part of the 2026-05 UI clarity pass.
        consistency = [v for v in norms
                       if "consistencia" in v and ("linhas" in v or "mercado" in v)]
        self.assertEqual(len(consistency), 1, "Expected single consistency tab")

    def test_market_tabs_no_duplicates_by_semantics(self):
        norms = [self._norm(v) for v in MARKET_TAB_NAMES]
        self.assertEqual(len(norms), len(set(norms)), "Duplicate market tabs by normalized label")

    def test_labels_no_mojibake_markers(self):
        for label in list(TAB_ALL) + list(MARKET_TAB_NAMES):
            self.assertFalse(self._looks_like_mojibake(label), f"Mojibake label: {label!r}")


if __name__ == "__main__":
    unittest.main()
