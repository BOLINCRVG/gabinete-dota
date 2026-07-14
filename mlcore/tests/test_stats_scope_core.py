"""Tests for app_core/stats_scope_core.py — scope resolution helpers."""
import unittest
from dataclasses import dataclass
from typing import Any, Optional

from app_core.stats_scope_core import _resolve_scope_filters, scope_label_from_meta


@dataclass
class _Filters:
    year: Optional[int] = None
    league: Optional[str] = None
    split: Optional[str] = None
    playoffs: Optional[bool] = None


def _make_filters_cls(**kwargs):
    return _Filters(**kwargs)


class TestResolveScopeFiltersManual(unittest.TestCase):
    """When session_state has sc_league, sc_year, and sc_split set."""

    def _resolve(self, ss, active=None, ml_info=None, map_mode_ui="avg"):
        if active is None:
            active = _Filters()
        return _resolve_scope_filters(
            session_state=ss,
            filters_cls=_make_filters_cls,
            active_filters=active,
            ml_info=ml_info,
            map_mode_ui=map_mode_ui,
        )

    def test_manual_flag_true_when_all_set(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertTrue(result["manual"])

    def test_manual_flag_false_when_league_missing(self):
        ss = {"sc_league": "", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertFalse(result["manual"])

    def test_manual_flag_false_when_year_missing(self):
        ss = {"sc_league": "LCK", "sc_year": "", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertFalse(result["manual"])

    def test_manual_flag_false_when_split_missing(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": ""}
        result = self._resolve(ss)
        self.assertFalse(result["manual"])

    def test_manual_league_returned(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertEqual(result["league"], "LCK")

    def test_manual_year_returned(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertEqual(result["year"], "2024")

    def test_manual_split_returned(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertEqual(result["split"], "Spring")

    def test_manual_playoffs_playoffs(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring", "sc_playoffs": "Playoffs"}
        result = self._resolve(ss)
        self.assertTrue(result["filters"].playoffs)

    def test_manual_playoffs_regular(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring", "sc_playoffs": "Regular"}
        result = self._resolve(ss)
        self.assertFalse(result["filters"].playoffs)

    def test_manual_playoffs_empty_none(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring", "sc_playoffs": ""}
        result = self._resolve(ss)
        self.assertIsNone(result["filters"].playoffs)

    def test_manual_map_selector_from_session(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring", "sc_map": "Map2"}
        result = self._resolve(ss)
        self.assertEqual(result["map_selector"], "Map2")

    def test_manual_map_selector_default_all_maps(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertEqual(result["map_selector"], "All maps")

    def test_returns_dict_with_filters_key(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertIn("filters", result)

    def test_filters_instance_is_filters(self):
        ss = {"sc_league": "LCK", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertIsInstance(result["filters"], _Filters)

    def test_manual_filters_league_set(self):
        ss = {"sc_league": "LPL", "sc_year": "2024", "sc_split": "Spring"}
        result = self._resolve(ss)
        self.assertEqual(result["filters"].league, "LPL")


class TestResolveScopeFiltersAuto(unittest.TestCase):
    """Auto mode: session_state incomplete, fall back to active_filters + ml_info."""

    def _resolve(self, ss=None, active=None, ml_info=None, map_mode_ui="avg"):
        if ss is None:
            ss = {}
        if active is None:
            active = _Filters()
        return _resolve_scope_filters(
            session_state=ss,
            filters_cls=_make_filters_cls,
            active_filters=active,
            ml_info=ml_info,
            map_mode_ui=map_mode_ui,
        )

    def test_auto_mode_manual_false(self):
        result = self._resolve()
        self.assertFalse(result["manual"])

    def test_auto_uses_active_filters_league(self):
        active = _Filters(league="LEC")
        result = self._resolve(active=active)
        self.assertEqual(result["league"], "LEC")

    def test_auto_uses_ml_info_league_when_same(self):
        ml_info = {"blue": {"league": "LCK"}, "red": {"league": "LCK"}}
        result = self._resolve(ml_info=ml_info)
        self.assertEqual(result["league"], "LCK")

    def test_auto_ignores_ml_info_league_when_different(self):
        ml_info = {"blue": {"league": "LCK"}, "red": {"league": "LPL"}}
        result = self._resolve(ml_info=ml_info)
        self.assertEqual(result["league"], "")

    def test_auto_map_mode_from_param(self):
        result = self._resolve(map_mode_ui="Map1")
        self.assertEqual(result["map_selector"], "Map1")

    def test_auto_playoffs_none_returns_empty(self):
        active = _Filters(playoffs=None)
        result = self._resolve(active=active)
        self.assertEqual(result["playoffs"], "")

    def test_auto_playoffs_true_returns_playoffs_string(self):
        active = _Filters(playoffs=True)
        result = self._resolve(active=active)
        self.assertEqual(result["playoffs"], "Playoffs")

    def test_auto_playoffs_false_returns_regular(self):
        active = _Filters(playoffs=False)
        result = self._resolve(active=active)
        self.assertEqual(result["playoffs"], "Regular")

    def test_auto_ml_info_none_does_not_crash(self):
        result = self._resolve(ml_info=None)
        self.assertFalse(result["manual"])

    def test_auto_ml_info_broken_does_not_crash(self):
        result = self._resolve(ml_info={"blue": None, "red": "garbage"})
        self.assertFalse(result["manual"])


class TestScopeLabelFromMeta(unittest.TestCase):

    def test_none_input_returns_recorte_ativo(self):
        label = scope_label_from_meta(None)
        self.assertIn("Recorte ativo", label)

    def test_empty_dict_returns_recorte_ativo(self):
        label = scope_label_from_meta({})
        self.assertIn("Recorte ativo", label)

    def test_manual_suffix(self):
        label = scope_label_from_meta({"manual": True, "league": "LCK", "year": "2024", "split": "Spring"})
        self.assertIn("manual", label)

    def test_auto_suffix(self):
        label = scope_label_from_meta({"manual": False, "league": "LCK"})
        self.assertIn("auto", label)

    def test_league_in_label(self):
        label = scope_label_from_meta({"league": "LCK", "year": "2024", "split": "Spring"})
        self.assertIn("LCK", label)

    def test_year_in_label(self):
        label = scope_label_from_meta({"league": "LCK", "year": "2024", "split": "Spring"})
        self.assertIn("2024", label)

    def test_returns_string(self):
        self.assertIsInstance(scope_label_from_meta({"league": "LEC"}), str)

    def test_empty_values_excluded(self):
        label = scope_label_from_meta({"league": "", "year": "", "split": ""})
        self.assertIn("Recorte ativo", label)


if __name__ == "__main__":
    unittest.main()
