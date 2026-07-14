import unittest

from app_core.navigation import TAB_ALL, TAB_GROUPS, pick_tabs


class TestPickTabs(unittest.TestCase):

    def test_completo_returns_all_tabs(self):
        tabs = pick_tabs("Completo")
        self.assertEqual(tabs, TAB_ALL)

    def test_fluxo_rapido_returns_list(self):
        tabs = pick_tabs("Fluxo Rápido")
        self.assertIsInstance(tabs, list)
        self.assertGreater(len(tabs), 0)
        self.assertIn("Visão Geral", tabs)

    def test_mercados_returns_list(self):
        tabs = pick_tabs("Mercados")
        self.assertIsInstance(tabs, list)
        self.assertIn("Visão Geral", tabs)

    def test_analise_includes_analise_de_linha(self):
        tabs = pick_tabs("Análise")
        self.assertIn("Análise de Linha", tabs)

    def test_unknown_mode_returns_all_tabs(self):
        tabs = pick_tabs("UnknownMode")
        self.assertEqual(tabs, list(TAB_ALL))

    def test_empty_string_returns_all_tabs(self):
        tabs = pick_tabs("")
        self.assertEqual(tabs, list(TAB_ALL))

    def test_returns_new_list_not_reference(self):
        tabs1 = pick_tabs("Completo")
        tabs1.append("extra")
        tabs2 = pick_tabs("Completo")
        self.assertNotIn("extra", tabs2)


class TestTabConstants(unittest.TestCase):

    def test_tab_all_is_list(self):
        self.assertIsInstance(TAB_ALL, list)

    def test_tab_all_not_empty(self):
        self.assertGreater(len(TAB_ALL), 0)

    def test_tab_groups_keys_include_completo(self):
        self.assertIn("Completo", TAB_GROUPS)

    def test_completo_group_is_tab_all(self):
        self.assertIs(TAB_GROUPS["Completo"], TAB_ALL)

    def test_visao_geral_in_all_groups(self):
        for group_tabs in TAB_GROUPS.values():
            self.assertIn("Visão Geral", group_tabs)


if __name__ == "__main__":
    unittest.main()
