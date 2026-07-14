import unittest

from app_core.text_normalization import (
    _canonicalize_tab_label,
    _canonicalize_tab_labels,
    _fix_mojibake_text,
    _looks_like_mojibake,
    _normalize_text_key,
)

CANONICAL_TABS = [
    "Visão Geral",
    "Configurações",
    "Análise",
    "Histórico",
    "Navegação",
]


class TestLooksLikeMojibake(unittest.TestCase):
    def test_clean_portuguese_ok(self):
        self.assertFalse(_looks_like_mojibake("Visão Geral"))
        self.assertFalse(_looks_like_mojibake("Análise de dados"))
        self.assertFalse(_looks_like_mojibake("hello world"))

    def test_replacement_char_detected(self):
        self.assertTrue(_looks_like_mojibake("Configura�es"))

    def test_c3_marker_detected(self):
        # Ã is the Latin Extended-A marker common in Latin-1 → UTF-8 mojibake
        self.assertTrue(_looks_like_mojibake("Ã£o"))

    def test_double_question_mark_detected(self):
        self.assertTrue(_looks_like_mojibake("Configura??es"))

    def test_letter_question_letter_detected(self):
        self.assertTrue(_looks_like_mojibake("Configura?es"))


class TestFixMojibakeText(unittest.TestCase):
    def test_clean_string_unchanged(self):
        self.assertEqual(_fix_mojibake_text("Visão Geral"), "Visão Geral")

    def test_double_question_configuracoes(self):
        self.assertEqual(_fix_mojibake_text("Configura??es"), "Configurações")

    def test_double_question_calibracoes(self):
        self.assertEqual(_fix_mojibake_text("Calibra??es"), "Calibrações")

    def test_double_question_parametros(self):
        self.assertEqual(_fix_mojibake_text("Par?metros"), "Parâmetros")

    def test_replacement_char_nao(self):
        self.assertEqual(_fix_mojibake_text("N�o"), "Não")

    def test_replacement_char_historico(self):
        self.assertEqual(_fix_mojibake_text("Hist?rico"), "Histórico")

    def test_non_string_passthrough(self):
        self.assertEqual(_fix_mojibake_text(42), 42)
        self.assertIsNone(_fix_mojibake_text(None))

    def test_atencao_replacement(self):
        self.assertEqual(_fix_mojibake_text("ATEN??O"), "ATENÇÃO")

    def test_visao_replacement(self):
        self.assertEqual(_fix_mojibake_text("Vis?o"), "Visão")

    def test_analise_replacement(self):
        self.assertEqual(_fix_mojibake_text("An?lise"), "Análise")


class TestNormalizeTextKey(unittest.TestCase):
    def test_removes_accents(self):
        self.assertEqual(_normalize_text_key("Visão"), "visao")
        self.assertEqual(_normalize_text_key("Análise"), "analise")

    def test_casefolds(self):
        self.assertEqual(_normalize_text_key("VISÃO GERAL"), "visaogeral")

    def test_removes_spaces_and_punctuation(self):
        self.assertEqual(_normalize_text_key("Visão Geral!"), "visaogeral")

    def test_empty_string(self):
        self.assertEqual(_normalize_text_key(""), "")

    def test_none(self):
        self.assertEqual(_normalize_text_key(None), "")

    def test_already_normalized(self):
        self.assertEqual(_normalize_text_key("abc123"), "abc123")

    def test_repairs_mojibake_first(self):
        # "Configura??es" → "Configurações" → "configuracoes"
        self.assertEqual(_normalize_text_key("Configura??es"), "configuracoes")


class TestCanonicalizeTabLabel(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(
            _canonicalize_tab_label("Visão Geral", CANONICAL_TABS), "Visão Geral"
        )

    def test_mojibake_repaired_and_matched(self):
        result = _canonicalize_tab_label("Configura??es", CANONICAL_TABS)
        self.assertEqual(result, "Configurações")

    def test_close_fuzzy_match(self):
        # "Historico" (no accent) should match "Histórico" after normalize
        result = _canonicalize_tab_label("Hist?rico", CANONICAL_TABS)
        self.assertEqual(result, "Histórico")

    def test_no_match_returns_fixed_input(self):
        result = _canonicalize_tab_label("ZZZ Unknown Tab", CANONICAL_TABS)
        self.assertEqual(result, "ZZZ Unknown Tab")

    def test_empty_string(self):
        self.assertEqual(_canonicalize_tab_label("", CANONICAL_TABS), "")


class TestCanonicalizeTabLabels(unittest.TestCase):
    def test_list_with_mojibake(self):
        tabs = ["Vis?o Geral", "Configura??es", "An?lise"]
        result = _canonicalize_tab_labels(tabs, CANONICAL_TABS)
        self.assertEqual(result, ["Visão Geral", "Configurações", "Análise"])

    def test_deduplicates(self):
        # "Visão Geral" and "Vis?o Geral" both normalize to the same canonical
        tabs = ["Visão Geral", "Vis?o Geral"]
        result = _canonicalize_tab_labels(tabs, CANONICAL_TABS)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Visão Geral")

    def test_empty_list(self):
        self.assertEqual(_canonicalize_tab_labels([], CANONICAL_TABS), [])

    def test_preserves_order_and_filters_duplicates(self):
        tabs = ["Histórico", "Análise", "Histórico"]
        result = _canonicalize_tab_labels(tabs, CANONICAL_TABS)
        self.assertEqual(result, ["Histórico", "Análise"])


if __name__ == "__main__":
    unittest.main()
