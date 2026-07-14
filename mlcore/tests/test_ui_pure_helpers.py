"""Tests for pure helper functions scattered across app_core UI modules."""
import math
import unittest

from app_core.csv_prepare_ui_core import _csv_prepare_status_chip, _csv_prepare_status_icon
from app_core.opponent_profile_ui_core import _format_metric, _format_pct, _format_z, _line_options


class TestCsvPrepareStatusHelpers(unittest.TestCase):

    def test_chip_ok(self):
        self.assertEqual(_csv_prepare_status_chip("ok"), "OK")

    def test_chip_error(self):
        self.assertEqual(_csv_prepare_status_chip("error"), "ERRO")

    def test_chip_running(self):
        self.assertEqual(_csv_prepare_status_chip("running"), "RODANDO")

    def test_chip_pending(self):
        self.assertEqual(_csv_prepare_status_chip("pending"), "PENDENTE")

    def test_chip_unknown_uppercased(self):
        self.assertEqual(_csv_prepare_status_chip("custom"), "CUSTOM")

    def test_chip_none_fallback(self):
        result = _csv_prepare_status_chip(None)
        self.assertIsInstance(result, str)

    def test_chip_case_insensitive(self):
        self.assertEqual(_csv_prepare_status_chip("OK"), "OK")
        self.assertEqual(_csv_prepare_status_chip("ERROR"), "ERRO")

    def test_icon_ok(self):
        self.assertEqual(_csv_prepare_status_icon("ok"), "[OK]")

    def test_icon_error(self):
        self.assertEqual(_csv_prepare_status_icon("error"), "[X]")

    def test_icon_running(self):
        self.assertEqual(_csv_prepare_status_icon("running"), "[...]")

    def test_icon_pending(self):
        self.assertEqual(_csv_prepare_status_icon("pending"), "[ ]")

    def test_icon_unknown_fallback(self):
        self.assertEqual(_csv_prepare_status_icon("unknown"), "[ ]")

    def test_icon_returns_string(self):
        self.assertIsInstance(_csv_prepare_status_icon("ok"), str)


class TestOpponentProfileFormatHelpers(unittest.TestCase):

    def _fmt_num(self, v, d):
        return f"{float(v):.{int(d)}f}"

    def _min_to_mmss(self, v):
        m = int(float(v))
        s = int((float(v) - m) * 60)
        return f"{m:02d}:{s:02d}"

    def test_line_options_empty(self):
        result = _line_options([])
        self.assertEqual(result, [""])

    def test_line_options_with_values(self):
        result = _line_options([25.5, 29.5])
        self.assertIn("25.5", result)
        self.assertIn("29.5", result)
        self.assertEqual(result[0], "")

    def test_line_options_none(self):
        result = _line_options(None)
        self.assertEqual(result, [""])

    def test_format_metric_numeric(self):
        result = _format_metric("kills", 25.5, min_to_mmss_fn=self._min_to_mmss, format_num_fn=self._fmt_num)
        self.assertEqual(result, "25.5")

    def test_format_metric_time(self):
        result = _format_metric("time", 30.5, min_to_mmss_fn=self._min_to_mmss, format_num_fn=self._fmt_num)
        self.assertIn("30", result)

    def test_format_metric_invalid(self):
        result = _format_metric("kills", "abc", min_to_mmss_fn=self._min_to_mmss, format_num_fn=self._fmt_num)
        self.assertEqual(result, "-")

    def test_format_pct_numeric(self):
        result = _format_pct(55.0, format_num_fn=self._fmt_num)
        self.assertIn("55.0%", result)

    def test_format_pct_invalid(self):
        result = _format_pct("bad", format_num_fn=self._fmt_num)
        self.assertEqual(result, "-")

    def test_format_z_numeric(self):
        result = _format_z(1.23, format_num_fn=self._fmt_num)
        self.assertEqual(result, "1.23")

    def test_format_z_invalid(self):
        result = _format_z(None, format_num_fn=self._fmt_num)
        self.assertEqual(result, "-")

    def test_format_z_returns_string(self):
        result = _format_z(0.5, format_num_fn=self._fmt_num)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
