import unittest

from app_core.statscamps_state_core import (
    ensure_statscamps_filter_state,
    get_statscamps_threshold_text,
    normalize_statscamps_select_state,
    reset_statscamps_league_dependents,
    sync_statscamps_delta_targets,
)
from app_core.tab_state_core import normalize_select_key


class TestEnsureStatscampsFilterState(unittest.TestCase):

    def test_initializes_missing_keys(self):
        state = {}
        ensure_statscamps_filter_state(state)
        self.assertIn("sc_year", state)
        self.assertIn("sc_league", state)
        self.assertIn("sc_split", state)
        self.assertIn("sc_playoffs", state)

    def test_missing_keys_set_to_empty(self):
        state = {}
        ensure_statscamps_filter_state(state)
        self.assertEqual(state["sc_year"], "")
        self.assertEqual(state["sc_league"], "")

    def test_none_value_replaced(self):
        state = {"sc_year": None}
        ensure_statscamps_filter_state(state)
        self.assertEqual(state["sc_year"], "")

    def test_valid_value_unchanged(self):
        state = {"sc_year": "2025", "sc_league": "LCK", "sc_split": "Spring", "sc_playoffs": ""}
        ensure_statscamps_filter_state(state)
        self.assertEqual(state["sc_year"], "2025")
        self.assertEqual(state["sc_league"], "LCK")


class TestResetStatscampsLeagueDependents(unittest.TestCase):

    def test_resets_split_and_deltas(self):
        state = {"sc_split": "Spring", "sc_delta_a": "A", "sc_delta_b": "B", "sc_compare_sel": ["A", "B"]}
        reset_statscamps_league_dependents(state)
        self.assertEqual(state["sc_split"], "")
        self.assertEqual(state["sc_delta_a"], "")
        self.assertEqual(state["sc_delta_b"], "")

    def test_compare_sel_cleared(self):
        state = {"sc_compare_sel": ["A", "B"], "sc_split": "Spring", "sc_delta_a": "A", "sc_delta_b": "B"}
        reset_statscamps_league_dependents(state)
        self.assertEqual(state["sc_compare_sel"], [])


class TestNormalizeStatscampsSelectState(unittest.TestCase):

    def test_normalizes_after_ensure(self):
        state = {}
        ensure_statscamps_filter_state(state)
        normalize_statscamps_select_state(state, years=["2025"], splits=["Spring"])
        self.assertIn("sc_year", state)
        self.assertIn("sc_split", state)
        self.assertIn("sc_playoffs", state)

    def test_invalid_year_replaced(self):
        state = {"sc_year": "1900"}
        normalize_statscamps_select_state(state, years=["2025"], splits=["Spring"])
        self.assertEqual(state["sc_year"], "")

    def test_valid_year_kept(self):
        state = {"sc_year": "2025"}
        normalize_statscamps_select_state(state, years=["2025"], splits=["Spring"])
        self.assertEqual(state["sc_year"], "2025")

    def test_invalid_playoffs_replaced(self):
        state = {"sc_playoffs": "INVALID"}
        normalize_statscamps_select_state(state, years=["2025"], splits=["Spring"])
        self.assertEqual(state["sc_playoffs"], "")


class TestSplitNotClobberedByYearNormalize(unittest.TestCase):
    """Regressão: bug 'seleciono o split e ele ignora'.

    A causa era chamar o normalize completo com splits=[] ANTES do selectbox de
    Ano (as opções de split ainda nem foram calculadas). Com splits=[] a lista
    permitida vira só [""], então o split escolhido era zerado a cada rerun.
    O fix é normalizar SÓ o ano nesse ponto.
    """

    def test_full_normalize_with_empty_splits_wipes_split(self):
        # Documenta o comportamento que causava o bug.
        state = {"sc_year": "2025", "sc_split": "Summer", "sc_playoffs": ""}
        normalize_statscamps_select_state(state, years=["2025"], splits=[])
        self.assertEqual(state["sc_split"], "")  # split foi zerado (era o bug)

    def test_year_only_normalize_preserves_split(self):
        # O fix: normalizar só o ano não pode tocar no split.
        state = {"sc_year": "2025", "sc_split": "Summer", "sc_playoffs": ""}
        normalize_select_key(state, "sc_year", ["", "2025"])
        self.assertEqual(state["sc_split"], "Summer")  # preservado
        self.assertEqual(state["sc_year"], "2025")

    def test_split_normalized_with_real_options_is_kept(self):
        # Mais abaixo no fluxo, com as opções reais, o split válido sobrevive.
        state = {"sc_year": "2025", "sc_split": "Summer", "sc_playoffs": ""}
        normalize_statscamps_select_state(state, years=["2025"], splits=["Spring", "Summer"])
        self.assertEqual(state["sc_split"], "Summer")


class TestGetStatscampsThresholdText(unittest.TestCase):

    def test_returns_state_value(self):
        state = {"sc_th_kills": "24.5"}
        result = get_statscamps_threshold_text(state, key="sc_th_kills", default="23.5")
        self.assertEqual(result, "24.5")

    def test_missing_key_returns_default(self):
        state = {}
        result = get_statscamps_threshold_text(state, key="sc_th_kills", default="23.5")
        self.assertEqual(result, "23.5")

    def test_none_value_returns_default(self):
        state = {"sc_th_kills": None}
        result = get_statscamps_threshold_text(state, key="sc_th_kills", default="23.5")
        self.assertEqual(result, "23.5")

    def test_returns_string(self):
        state = {"sc_th_kills": "25.0"}
        result = get_statscamps_threshold_text(state, key="sc_th_kills", default="23.5")
        self.assertIsInstance(result, str)


class TestSyncStatscampsDeltaTargets(unittest.TestCase):

    def test_valid_values_kept(self):
        state = {"sc_delta_a": "A", "sc_delta_b": "X"}
        a, b = sync_statscamps_delta_targets(state, selected=["A", "B"], default_a="A", default_b="B")
        self.assertEqual(a, "A")
        self.assertEqual(b, "B")

    def test_invalid_a_replaced_by_default(self):
        state = {"sc_delta_a": "INVALID", "sc_delta_b": "B"}
        a, _b = sync_statscamps_delta_targets(state, selected=["A", "B"], default_a="A", default_b="B")
        self.assertEqual(a, "A")

    def test_invalid_b_replaced_by_default(self):
        state = {"sc_delta_a": "A", "sc_delta_b": "INVALID"}
        _a, b = sync_statscamps_delta_targets(state, selected=["A", "B"], default_a="A", default_b="B")
        self.assertEqual(b, "B")

    def test_returns_tuple_of_strings(self):
        state = {}
        a, b = sync_statscamps_delta_targets(state, selected=["A", "B"], default_a="A", default_b="B")
        self.assertIsInstance(a, str)
        self.assertIsInstance(b, str)

    def test_updates_state(self):
        state = {"sc_delta_a": "BAD"}
        sync_statscamps_delta_targets(state, selected=["A", "B"], default_a="A", default_b="B")
        self.assertEqual(state["sc_delta_a"], "A")


if __name__ == "__main__":
    unittest.main()
