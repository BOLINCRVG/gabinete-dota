"""Tests for app_core/live_bet_analysis_core.py — pure helper functions."""
import unittest

from app_core.live_bet_analysis_core import (
    _draft_call_score,
    _profile_bucket_score,
)


class TestProfileBucketScore(unittest.TestCase):

    def test_neutro_returns_zero(self):
        self.assertEqual(_profile_bucket_score("Neutro"), 0.0)

    def test_over_returns_positive_half(self):
        self.assertEqual(_profile_bucket_score("Over"), 0.5)

    def test_muito_over_returns_one(self):
        self.assertEqual(_profile_bucket_score("Muito Over"), 1.0)

    def test_under_returns_negative_half(self):
        self.assertEqual(_profile_bucket_score("Under"), -0.5)

    def test_muito_under_returns_negative_one(self):
        self.assertEqual(_profile_bucket_score("Muito Under"), -1.0)

    def test_unknown_returns_zero(self):
        self.assertEqual(_profile_bucket_score("Unknown"), 0.0)

    def test_empty_string_returns_zero(self):
        self.assertEqual(_profile_bucket_score(""), 0.0)

    def test_none_returns_zero(self):
        self.assertEqual(_profile_bucket_score(None), 0.0)

    def test_leading_trailing_spaces_stripped(self):
        self.assertEqual(_profile_bucket_score("  Over  "), 0.5)

    def test_returns_float(self):
        self.assertIsInstance(_profile_bucket_score("Neutro"), float)


class TestDraftCallScore(unittest.TestCase):

    def test_favorece_over_returns_one(self):
        self.assertEqual(_draft_call_score("Favorece over kills"), 1.0)

    def test_mais_keyword_returns_one(self):
        self.assertEqual(_draft_call_score("mais torres"), 1.0)

    def test_alonga_returns_one(self):
        self.assertEqual(_draft_call_score("alonga o jogo"), 1.0)

    def test_vies_over_returns_half(self):
        self.assertEqual(_draft_call_score("vies over leve"), 0.5)

    def test_vies_mais_returns_half(self):
        self.assertEqual(_draft_call_score("vies mais"), 0.5)

    def test_vies_longo_returns_half(self):
        self.assertEqual(_draft_call_score("vies longo"), 0.5)

    def test_favorece_under_returns_neg_one(self):
        self.assertEqual(_draft_call_score("favorece under tempo"), -1.0)

    def test_menos_keyword_returns_neg_one(self):
        self.assertEqual(_draft_call_score("menos matares"), -1.0)

    def test_encurta_returns_neg_one(self):
        self.assertEqual(_draft_call_score("encurta o jogo"), -1.0)

    def test_vies_under_returns_neg_half(self):
        self.assertEqual(_draft_call_score("vies under leve"), -0.5)

    def test_vies_menos_returns_neg_half(self):
        self.assertEqual(_draft_call_score("vies menos"), -0.5)

    def test_vies_curto_returns_neg_half(self):
        self.assertEqual(_draft_call_score("vies curto"), -0.5)

    def test_unknown_returns_zero(self):
        self.assertEqual(_draft_call_score("neutro"), 0.0)

    def test_empty_returns_zero(self):
        self.assertEqual(_draft_call_score(""), 0.0)

    def test_none_returns_zero(self):
        self.assertEqual(_draft_call_score(None), 0.0)

    def test_case_insensitive(self):
        self.assertEqual(_draft_call_score("FAVORECE OVER"), 1.0)

    def test_returns_float(self):
        self.assertIsInstance(_draft_call_score("neutro"), float)


if __name__ == "__main__":
    unittest.main()
