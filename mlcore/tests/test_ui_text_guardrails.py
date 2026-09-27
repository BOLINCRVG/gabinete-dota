from __future__ import annotations

import re
import unittest
from pathlib import Path

from app_core.ui_labels import MARKET_TAB_NAMES
from app_core.text_normalization import _fix_mojibake_text, _canonicalize_tab_labels
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol



class UITestMojibakeGuardrails(unittest.TestCase):
    def _load_source(self) -> str:
        p = Path("plays_app.py")
        return p.read_text(encoding="utf-8")

    @so_no_lol("usa o plays_app.py, o app do LoL")
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

    @so_no_lol("usa o plays_app.py, o app do LoL")
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

    @so_no_lol("usa o plays_app.py, o app do LoL")
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

    @so_no_lol("usa o plays_app.py, o app do LoL")
    def test_tab_refs_use_canonical_tab_all_indices(self) -> None:
        src = self._load_source()
        # TAB constants are imported from app_core.tab_dispatcher and aliased locally
        self.assertIn("_TAB_VISAO = TAB_VISAO", src)
        self.assertIn("_TAB_CONS = TAB_CONS", src)
        self.assertIn("_TAB_PARAMS = TAB_PARAMS", src)
        # O que esta trava protege e a BUSCA PELA CONSTANTE: as tres tabs
        # principais tem de sair de `tab_refs` pela constante canonica, nunca
        # por texto solto nem por posicao na lista.
        #
        # Em 13/09/2026 a forma mudou de `if X in tab_refs:` para
        # `tab_refs.get(X)`, porque as tres passaram a ser desenhadas
        # isoladas (uma aba quebrada nao pode mais derrubar o site). As duas
        # formas respeitam a regra; travar a forma literal reprovava a
        # mudanca sem ter nada a ver com o que ela quer impedir.
        for _nome in ("_TAB_PARAMS", "_TAB_VISAO", "_TAB_CONS"):
            self.assertTrue(
                ("if {} in tab_refs:".format(_nome) in src)
                or ("tab_refs.get({})".format(_nome) in src)
                or ("tab_refs[{}]".format(_nome) in src),
                "{} nao e usado como chave de tab_refs".format(_nome))
        self.assertTrue(
            ('_TAB_LIVEBET = "Live-Bet"' in src)
            or ('for _tab_name in ("Rankings", "Jogos", "Series", "StatsCamps", "Campeões", "Players", "Identidade", "Live-Bet"):' in src)
            or ('if nav_page == "Live-Bet":' in src)
        )


if __name__ == "__main__":
    unittest.main()
