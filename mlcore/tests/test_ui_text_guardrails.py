from __future__ import annotations

import re
import unittest
from pathlib import Path

from app_core.ui_labels import MARKET_TAB_NAMES
from app_core.text_normalization import _fix_mojibake_text, _canonicalize_tab_labels


class UITestMojibakeGuardrails(unittest.TestCase):
    def _load_source(self) -> str:
        p = Path("plays_app.py")
        return p.read_text(encoding="utf-8")

    def test_plays_app_is_valid_utf8(self) -> None:
        src = self._load_source()
        self.assertTrue(len(src) > 1000)

    def test_fix_mojibake_examples(self) -> None:
        fix = _fix_mojibake_text
        self.assertEqual(fix("Consist?ncia de Mercado (Laplace)"), "Consistência de Mercado (Laplace)")
        self.assertEqual(fix("Hist?rico + Laplace"), "Histórico + Laplace")
        self.assertEqual(fix("Seletor R?pido"), "Seletor Rápido")
        self.assertEqual(fix("An?lise em dia para os par?metros atuais."), "Análise em dia para os parâmetros atuais.")
        self.assertEqual(fix("Visão Geral"), "Visão Geral")

    def test_tab_dedup_with_broken_variants(self) -> None:
        fix = _fix_mojibake_text
        canonicalize = _canonicalize_tab_labels
        tab_all = [
            "Visão Geral",
            "Live-Bet",
            "Consistência de Mercado (Laplace)",
            "Series",
            "Players",
            "Campeões",
            "Rankings",
            "Parâmetros",
            "Identidade",
        ]
        tabs = [
            "Visão Geral",
            "Consist?ncia de Mercado (Laplace)",
            "Consistência de Mercado (Laplace)",
            "Consist�ncia de Mercado (Laplace)",
            "Series",
        ]
        out = canonicalize(tabs, tab_all)
        norm = lambda t: fix(str(t)).strip().casefold()
        cons_count = sum(1 for t in out if "consistência de mercado (laplace)" == norm(t))
        self.assertEqual(cons_count, 1)
        self.assertEqual(out[1], "Consistência de Mercado (Laplace)")

    def test_market_tab_labels_are_canonical(self) -> None:
        src = self._load_source()
        # list(_MARKET_TAB_NAMES) moved to consistencia_tab_core.py when TAB_CONS was extracted
        cons_src = Path("app_core/consistencia_tab_core.py").read_text(encoding="utf-8")
        self.assertTrue(
            "list(_MARKET_TAB_NAMES)" in src or "list(_MARKET_TAB_NAMES)" in cons_src,
            "list(_MARKET_TAB_NAMES) not found in plays_app.py or consistencia_tab_core.py",
        )
        self.assertIn("Seletor Rápido", MARKET_TAB_NAMES)
        self.assertIn("Dragões", MARKET_TAB_NAMES)
        self.assertIn("Dragões Elementais", MARKET_TAB_NAMES)
        self.assertIn("Barões", MARKET_TAB_NAMES)
        self.assertNotIn("Seletor R?pido", MARKET_TAB_NAMES)

    def test_common_broken_ui_tokens_not_present(self) -> None:
        src = self._load_source()
        # Ignore the correction dictionary block itself; broken tokens are expected there as keys.
        src_no_dict = re.sub(r"rep = \{.*?\n        \}", "", src, flags=re.S)
        banned = [
            "Consist?ncia de Mercado (Laplace)",
            "Fluxo R?pido",
            "A??o",
            "Seletor R?pido",
        ]
        for tok in banned:
            self.assertNotIn(tok, src_no_dict, f"broken token still present outside mapper: {tok}")

    def test_tab_refs_use_canonical_tab_all_indices(self) -> None:
        src = self._load_source()
        # TAB constants are imported from app_core.tab_dispatcher and aliased locally
        self.assertIn("_TAB_VISAO = TAB_VISAO", src)
        self.assertIn("_TAB_CONS = TAB_CONS", src)
        self.assertIn("_TAB_PARAMS = TAB_PARAMS", src)
        self.assertIn("if _TAB_PARAMS in tab_refs:", src)
        self.assertIn("if _TAB_VISAO in tab_refs:", src)
        self.assertIn("if _TAB_CONS in tab_refs:", src)
        self.assertTrue(
            ('_TAB_LIVEBET = "Live-Bet"' in src)
            or ('for _tab_name in ("Rankings", "Jogos", "Series", "StatsCamps", "Campeões", "Players", "Identidade", "Live-Bet"):' in src)
            or ('if nav_page == "Live-Bet":' in src)
        )


if __name__ == "__main__":
    unittest.main()
