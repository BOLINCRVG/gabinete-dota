"""Tests for app_core/ui_labels.py — pure formatting helpers."""
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


# ---------------------------------------------------------------------------
# fmt_prob_pct
# ---------------------------------------------------------------------------

class TestFmtProbPct(unittest.TestCase):

    def test_half(self):
        self.assertEqual(fmt_prob_pct(0.5), "50.0%")

    def test_one(self):
        self.assertEqual(fmt_prob_pct(1.0), "100.0%")

    def test_zero(self):
        self.assertEqual(fmt_prob_pct(0.0), "0.0%")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_prob_pct(None), "-")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_prob_pct(float("nan")), "-")

    def test_one_decimal_place(self):
        self.assertEqual(fmt_prob_pct(0.643), "64.3%")

    def test_ends_with_percent(self):
        self.assertTrue(fmt_prob_pct(0.7).endswith("%"))


# ---------------------------------------------------------------------------
# fmt_num
# ---------------------------------------------------------------------------

class TestFmtNum(unittest.TestCase):

    def test_default_two_decimals(self):
        self.assertEqual(fmt_num(3.14159), "3.14")

    def test_zero_decimals(self):
        self.assertEqual(fmt_num(3.7, nd=0), "4")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_num(None), "-")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_num(float("nan")), "-")

    def test_integer_value(self):
        self.assertEqual(fmt_num(5.0, nd=1), "5.0")

    def test_custom_decimals(self):
        self.assertEqual(fmt_num(1.23456, nd=4), "1.2346")


# ---------------------------------------------------------------------------
# min_to_mmss
# ---------------------------------------------------------------------------

class TestMinToMmss(unittest.TestCase):

    def test_30_min(self):
        self.assertEqual(min_to_mmss(30.0), "30:00")

    def test_28_5_min(self):
        # 28.5 minutes = 28:30
        self.assertEqual(min_to_mmss(28.5), "28:30")

    def test_zero(self):
        self.assertEqual(min_to_mmss(0.0), "00:00")

    def test_none_returns_dash(self):
        self.assertEqual(min_to_mmss(None), "-")

    def test_nan_returns_dash(self):
        self.assertEqual(min_to_mmss(float("nan")), "-")

    def test_format_always_two_digits(self):
        result = min_to_mmss(5.0)
        self.assertEqual(result, "05:00")


# ---------------------------------------------------------------------------
# metric_label_pt
# ---------------------------------------------------------------------------

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

    def test_unknown_returns_title(self):
        result = metric_label_pt("custom_metric")
        self.assertEqual(result, "Custom_Metric")

    def test_case_insensitive(self):
        self.assertEqual(metric_label_pt("KILLS"), "Kills")


# ---------------------------------------------------------------------------
# side_label_pt
# ---------------------------------------------------------------------------

class TestSideLabelPt(unittest.TestCase):

    def test_over(self):
        self.assertEqual(side_label_pt("over"), "Acima")

    def test_under(self):
        self.assertEqual(side_label_pt("under"), "Abaixo")

    def test_case_insensitive(self):
        self.assertEqual(side_label_pt("OVER"), "Acima")

    def test_unknown_returns_title(self):
        self.assertEqual(side_label_pt("draw"), "Draw")


# ---------------------------------------------------------------------------
# fmt_line_disp
# ---------------------------------------------------------------------------

class TestFmtLineDisp(unittest.TestCase):

    def test_time_shows_mmss(self):
        result = fmt_line_disp("time", 28.5)
        self.assertEqual(result, "28:30")

    def test_integer_value_no_decimal(self):
        result = fmt_line_disp("kills", 5.0)
        self.assertEqual(result, "5")

    def test_fractional_value_two_decimals(self):
        result = fmt_line_disp("kills", 5.5)
        self.assertEqual(result, "5.5")

    def test_invalid_value_returns_str(self):
        result = fmt_line_disp("kills", "n/a")
        self.assertEqual(result, "n/a")

    def test_trailing_zeros_stripped(self):
        result = fmt_line_disp("kills", 5.10)
        self.assertEqual(result, "5.1")


# ---------------------------------------------------------------------------
# fmt_recortes_disp
# ---------------------------------------------------------------------------

class TestFmtRecortesDisp(unittest.TestCase):

    def test_empty_returns_dash(self):
        self.assertEqual(fmt_recortes_disp([]), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_recortes_disp(None), "-")

    def test_single_item(self):
        self.assertEqual(fmt_recortes_disp(["15"]), "15")

    def test_multiple_items_comma_separated(self):
        result = fmt_recortes_disp(["15", "5"])
        self.assertEqual(result, "15, 5")

    def test_ano_normalized(self):
        result = fmt_recortes_disp(["ano"])
        self.assertEqual(result, "Ano")

    def test_uppercase_ano_normalized(self):
        result = fmt_recortes_disp(["ANO"])
        self.assertEqual(result, "Ano")


# ---------------------------------------------------------------------------
# fmt_bytes_short
# ---------------------------------------------------------------------------

class TestFmtBytesShort(unittest.TestCase):

    def test_bytes(self):
        self.assertEqual(fmt_bytes_short(512), "512 B")

    def test_kilobytes(self):
        self.assertEqual(fmt_bytes_short(1024), "1.0 KB")

    def test_megabytes(self):
        result = fmt_bytes_short(1024 * 1024)
        self.assertEqual(result, "1.0 MB")

    def test_gigabytes(self):
        result = fmt_bytes_short(1024 ** 3)
        self.assertEqual(result, "1.0 GB")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_bytes_short(None), "-")

    def test_negative_returns_dash(self):
        self.assertEqual(fmt_bytes_short(-100), "-")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_bytes_short(float("nan")), "-")

    def test_non_numeric_returns_dash(self):
        self.assertEqual(fmt_bytes_short("abc"), "-")

    def test_zero_bytes(self):
        self.assertEqual(fmt_bytes_short(0), "0 B")


if __name__ == "__main__":
    unittest.main()
