import math
import unittest

from app_core.shared_formatters import (
    fmt_delta_pp,
    fmt_num,
    fmt_pct,
    fmt_status,
    fmt_value,
)


class TestFmtNum(unittest.TestCase):
    def test_positive_value(self):
        self.assertEqual(fmt_num(23.5), "23.5")

    def test_zero(self):
        self.assertEqual(fmt_num(0), "0.0")

    def test_negative(self):
        self.assertEqual(fmt_num(-1.5), "-1.5")

    def test_zero_decimals(self):
        self.assertEqual(fmt_num(23.7, 0), "24")

    def test_two_decimals(self):
        self.assertEqual(fmt_num(1.0, 2), "1.00")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_num(float("nan")), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_num(float("inf")), "-")

    def test_invalid_string_returns_dash(self):
        self.assertEqual(fmt_num("abc"), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_num(None), "-")

    def test_numeric_string(self):
        self.assertEqual(fmt_num("35.5"), "35.5")


class TestFmtPct(unittest.TestCase):
    def test_normal_value(self):
        self.assertEqual(fmt_pct(62.5), "62.5%")

    def test_zero(self):
        self.assertEqual(fmt_pct(0.0), "0.0%")

    def test_hundred(self):
        self.assertEqual(fmt_pct(100.0), "100.0%")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_pct(float("nan")), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_pct(float("inf")), "-")

    def test_two_decimals(self):
        self.assertEqual(fmt_pct(33.333, 2), "33.33%")


class TestFmtDeltaPp(unittest.TestCase):
    def test_positive_delta(self):
        self.assertEqual(fmt_delta_pp(5.2), "+5.2pp")

    def test_negative_delta(self):
        self.assertEqual(fmt_delta_pp(-3.1), "-3.1pp")

    def test_zero_has_plus(self):
        self.assertEqual(fmt_delta_pp(0.0), "+0.0pp")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_delta_pp(float("nan")), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_delta_pp(float("inf")), "-")

    def test_two_decimals(self):
        self.assertEqual(fmt_delta_pp(1.25, 2), "+1.25pp")


class TestFmtStatus(unittest.TestCase):
    def test_above_55_is_favoravel(self):
        self.assertEqual(fmt_status(55.0), "Favorável")
        self.assertEqual(fmt_status(70.0), "Favorável")
        self.assertEqual(fmt_status(100.0), "Favorável")

    def test_below_45_is_desfavoravel(self):
        self.assertEqual(fmt_status(45.0), "Desfavorável")
        self.assertEqual(fmt_status(30.0), "Desfavorável")
        self.assertEqual(fmt_status(0.0), "Desfavorável")

    def test_between_45_and_55_is_neutro(self):
        self.assertEqual(fmt_status(50.0), "Neutro")
        self.assertEqual(fmt_status(45.1), "Neutro")
        self.assertEqual(fmt_status(54.9), "Neutro")

    def test_nan_returns_em_dash(self):
        self.assertEqual(fmt_status(float("nan")), "—")

    def test_inf_returns_em_dash(self):
        self.assertEqual(fmt_status(float("inf")), "—")


class TestFmtValue(unittest.TestCase):
    def test_whole_number_no_decimal(self):
        self.assertEqual(fmt_value(5.0), "5")
        self.assertEqual(fmt_value(0.0), "0")
        self.assertEqual(fmt_value(-3.0), "-3")

    def test_fractional_with_decimal(self):
        self.assertEqual(fmt_value(5.3), "5.3")
        self.assertEqual(fmt_value(0.7), "0.7")

    def test_nan_returns_dash(self):
        self.assertEqual(fmt_value(float("nan")), "-")

    def test_inf_returns_dash(self):
        self.assertEqual(fmt_value(float("inf")), "-")

    def test_invalid_returns_dash(self):
        self.assertEqual(fmt_value("xyz"), "-")

    def test_none_returns_dash(self):
        self.assertEqual(fmt_value(None), "-")

    def test_two_decimals(self):
        self.assertEqual(fmt_value(3.14159, 2), "3.14")

    def test_integer_input(self):
        self.assertEqual(fmt_value(7), "7")


if __name__ == "__main__":
    unittest.main()
