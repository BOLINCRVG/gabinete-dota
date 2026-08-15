"""Tests for app_core/params_tab_core.py — pure helper _norm_gate_decision."""
import unittest

from app_core.params_tab_core import _norm_gate_decision


class TestNormGateDecision(unittest.TestCase):

    def test_promote_returns_promote(self):
        self.assertEqual(_norm_gate_decision("promote"), "promote")

    def test_shadow_only_returns_shadow_only(self):
        self.assertEqual(_norm_gate_decision("shadow_only"), "shadow_only")

    def test_reject_returns_reject(self):
        self.assertEqual(_norm_gate_decision("reject"), "reject")

    def test_unknown_value_returns_unknown(self):
        self.assertEqual(_norm_gate_decision("something_else"), "unknown")

    def test_uppercase_promote_returns_promote(self):
        self.assertEqual(_norm_gate_decision("PROMOTE"), "promote")

    def test_mixed_case_shadow_only(self):
        self.assertEqual(_norm_gate_decision("Shadow_Only"), "shadow_only")

    def test_none_returns_unknown(self):
        self.assertEqual(_norm_gate_decision(None), "unknown")

    def test_empty_string_returns_unknown(self):
        self.assertEqual(_norm_gate_decision(""), "unknown")

    def test_returns_string(self):
        self.assertIsInstance(_norm_gate_decision("promote"), str)

    def test_whitespace_stripped(self):
        self.assertEqual(_norm_gate_decision("  promote  "), "promote")


if __name__ == "__main__":
    unittest.main()
