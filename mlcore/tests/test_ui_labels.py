"""Tests for app_core/ui_labels.py — pure formatting helpers."""
import math
import unittest

from app_core.ui_labels import (
    fmt_bytes_short,
    fmt_line_disp,
    fmt_num,
    fmt_prob_pct,
    fmt_recortes_disp,
    metric_label_pt,
    min_to_mmss,
    side_label_pt,
)


class TestFmtProbPct(unittest.TestCase):

    def test_half_is_fifty_percent(self):
        self.assertEqual(fmt_prob_pct(0.5), "50.0%")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_prob_pct(float("nan")), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_prob_pct(None), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_prob_pct(float("inf")), "-")

    def test_zero_is_zero_percent(self):
        self.assertEqual(fmt_prob_pct(0.0), "0.0%")

    def test_one_is_hundred_percent(self):
        self.assertEqual(fmt_prob_pct(1.0), "100.0%")

    def test_returns_string(self):
        self.assertIsInstance(fmt_prob_pct(0.6), str)

    def test_includes_percent_sign(self):
        self.assertIn("%", fmt_prob_pct(0.75))


class TestFmtNum(unittest.TestCase):

    def test_two_decimals_default(self):
        self.assertEqual(fmt_num(1.23456), "1.23")

    def test_zero_decimals(self):
        self.assertEqual(fmt_num(1.7, 0), "2")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_num(float("nan")), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_num(None), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_num(float("inf")), "-")

    def test_returns_string(self):
        self.assertIsInstance(fmt_num(3.14), str)

    def test_three_decimals(self):
        self.assertEqual(fmt_num(1.23456, 3), "1.235")


class TestMinToMmss(unittest.TestCase):

    def test_30_minutes(self):
        self.assertEqual(min_to_mmss(30.0), "30:00")

    def test_32_half_minutes(self):
        self.assertEqual(min_to_mmss(32.5), "32:30")

    def test_nan_returns_dash(self):
        self.assertEqual(min_to_mmss(float("nan")), "-")

    def test_none_returns_dash(self):
        self.assertEqual(min_to_mmss(None), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(min_to_mmss(float("inf")), "-")

    def test_returns_string(self):
        self.assertIsInstance(min_to_mmss(25.0), str)

    def test_zero_is_zero_zero(self):
        self.assertEqual(min_to_mmss(0.0), "00:00")

    def test_one_and_half_minutes(self):
        self.assertEqual(min_to_mmss(1.5), "01:30")


class TestMetricLabelPt(unittest.TestCase):

    def test_kills(self):
        self.assertEqual(metric_label_pt("kills"), "Kills")

    def test_towers(self):
        self.assertEqual(metric_label_pt("towers"), "Torres")

    def test_dragons(self):
        self.assertEqual(metric_label_pt("dragons"), "Dragões")

    def test_barons(self):
        self.assertEqual(metric_label_pt("barons"), "Barões")

    def test_inhibitors(self):
        self.assertEqual(metric_label_pt("inhibitors"), "Inibidores")

    def test_time(self):
        self.assertEqual(metric_label_pt("time"), "Tempo")

    def test_unknown_returns_titlecased(self):
        result = metric_label_pt("unknown_metric")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_case_insensitive(self):
        self.assertEqual(metric_label_pt("KILLS"), metric_label_pt("kills"))


class TestSideLabelPt(unittest.TestCase):

    def test_over(self):
        self.assertEqual(side_label_pt("over"), "Acima")

    def test_under(self):
        self.assertEqual(side_label_pt("under"), "Abaixo")

    def test_unknown_returns_titlecased(self):
        result = side_label_pt("other")
        self.assertIsInstance(result, str)

    def test_case_insensitive(self):
        self.assertEqual(side_label_pt("OVER"), side_label_pt("over"))


class TestFmtLineDisp(unittest.TestCase):

    def test_non_time_integer_line(self):
        result = fmt_line_disp("kills", 3.0)
        self.assertEqual(result, "3")

    def test_non_time_half_line(self):
        result = fmt_line_disp("kills", 3.5)
        self.assertIn("3.5", result)

    def test_time_metric_returns_mmss(self):
        result = fmt_line_disp("time", 30.0)
        self.assertEqual(result, "30:00")

    def test_returns_string(self):
        self.assertIsInstance(fmt_line_disp("kills", 5.0), str)

    def test_invalid_value_returns_string(self):
        result = fmt_line_disp("kills", "invalid")
        self.assertIsInstance(result, str)


class TestFmtRecortesDisp(unittest.TestCase):

    def test_basic_list(self):
        result = fmt_recortes_disp(["Liga", "2025"])
        self.assertIn("Liga", result)
        self.assertIn("2025", result)

    def test_empty_list_returns_dash(self):
        self.assertEqual(fmt_recortes_disp([]), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_recortes_disp(None), "-")

    def test_ano_normalized(self):
        result = fmt_recortes_disp(["ano"])
        self.assertIn("Ano", result)

    def test_returns_string(self):
        self.assertIsInstance(fmt_recortes_disp(["X"]), str)


class TestFmtBytesShort(unittest.TestCase):

    def test_zero_bytes(self):
        self.assertEqual(fmt_bytes_short(0), "0 B")

    def test_one_kb(self):
        result = fmt_bytes_short(1024)
        self.assertIn("KB", result)

    def test_one_mb(self):
        result = fmt_bytes_short(1024 * 1024)
        self.assertIn("MB", result)

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_bytes_short(float("nan")), "-")

    def test_negative_returns_dash(self):
        self.assertEqual(fmt_bytes_short(-1), "-")

    def test_returns_string(self):
        self.assertIsInstance(fmt_bytes_short(512), str)

    def test_small_bytes_no_unit_scaling(self):
        result = fmt_bytes_short(500)
        self.assertIn("B", result)
        self.assertNotIn("KB", result)


if __name__ == "__main__":
    unittest.main()
