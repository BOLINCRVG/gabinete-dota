"""Resumo default-lines behavior.

Contrato (2026-06-11, regra do usuário): PADRÃO + digitadas, JUNTAS.
- Campo vazio → padrão FIXO da métrica (27.5 kills, ...), nunca um valor de um
  confronto anterior (sem "fantasma 25.5").
- Campo com linhas → padrão + digitadas (digitou 28.5 → analisa 27.5 E 28.5).
  Quem corta linha que não presta é o teto de odd (resumo_max_odd) no
  build_metric_rows, não a remoção manual do padrão.
"""
from __future__ import annotations

import unittest

from app_core.resumo_runtime_core import RESUMO_DEFAULT_LINES, merge_with_default_lines


class TestResumoDefaultLines(unittest.TestCase):

    def test_empty_shows_standard_default(self):
        self.assertEqual(merge_with_default_lines("kills", []), [27.5])
        self.assertEqual(merge_with_default_lines("towers", []), [11.5])
        self.assertEqual(merge_with_default_lines("dragons", []), [4.5])
        self.assertEqual(merge_with_default_lines("barons", []), [1.5])
        self.assertEqual(merge_with_default_lines("inhibitors", []), [1.5])
        self.assertEqual(merge_with_default_lines("time", []), [32.5])

    def test_typed_lines_join_the_default(self):
        # Usuário digitou 5.5 → analisa o padrão 11.5 E a 5.5 (regra 2026-06-11:
        # "quero o 27.5 e o 28.5", padrão nunca some).
        self.assertEqual(merge_with_default_lines("towers", [5.5]), [5.5, 11.5])
        self.assertEqual(merge_with_default_lines("towers", [12.5]), [11.5, 12.5])
        # O caso exato do pedido: digitou 28.5 → mostra 27.5 E 28.5.
        self.assertEqual(merge_with_default_lines("kills", [28.5]), [27.5, 28.5])

    def test_multiple_custom_lines_join_default_sorted(self):
        self.assertEqual(merge_with_default_lines("kills", [30.5, 26.5]), [26.5, 27.5, 30.5])
        # Digitar o padrão junto não duplica.
        self.assertEqual(merge_with_default_lines("kills", [27.5, 28.5]), [27.5, 28.5])

    def test_typing_the_default_is_not_duplicated(self):
        self.assertEqual(merge_with_default_lines("kills", [27.5]), [27.5])

    def test_unknown_metric_keeps_typed_only(self):
        self.assertEqual(merge_with_default_lines("unknown", []), [])
        self.assertEqual(merge_with_default_lines("unknown", [3.5]), [3.5])

    def test_default_is_a_copy_not_shared_reference(self):
        # Mutating the returned list must not corrupt the module constant.
        out = merge_with_default_lines("kills", [])
        out.append(99.0)
        self.assertEqual(RESUMO_DEFAULT_LINES["kills"], [27.5])


if __name__ == "__main__":
    unittest.main()
