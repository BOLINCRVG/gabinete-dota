"""Tests for app_core/live_bet_draft_core.py — pure draft resolution helpers."""
import unittest
from typing import Optional

from app_core.live_bet_draft_core import _resolve_pick_names, resolve_compact_draft_sides


def _exact_fuzzy(token: str, vocab: list, threshold: float):
    """Stub that matches only exact (case-sensitive) strings."""
    if token in vocab:
        return (token, [])
    return (None, [v for v in vocab if v.startswith(token[:2])][:3])


def _always_none_fuzzy(token: str, vocab: list, threshold: float):
    """Stub that never matches."""
    return (None, [])


class TestResolvePickNames(unittest.TestCase):

    def test_matched_tokens_returned(self):
        vocab = ["Jinx", "Ahri", "Lux"]
        resolved, unmatched = _resolve_pick_names(
            tokens=["Jinx", "Ahri"],
            vocab=vocab,
            fuzzy_match_one_fn=_exact_fuzzy,
        )
        self.assertEqual(resolved, ["Jinx", "Ahri"])
        self.assertEqual(unmatched, [])

    def test_unmatched_token_kept_as_is(self):
        resolved, unmatched = _resolve_pick_names(
            tokens=["UnknownChamp"],
            vocab=[],
            fuzzy_match_one_fn=_always_none_fuzzy,
        )
        self.assertEqual(resolved, ["UnknownChamp"])
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["entrada"], "UnknownChamp")

    def test_unmatched_entry_has_sugestoes(self):
        _, unmatched = _resolve_pick_names(
            tokens=["Unknown"],
            vocab=["Jhin"],
            fuzzy_match_one_fn=_always_none_fuzzy,
        )
        self.assertIn("sugestoes", unmatched[0])

    def test_empty_tokens_returns_empty(self):
        resolved, unmatched = _resolve_pick_names(
            tokens=[],
            vocab=["Jinx"],
            fuzzy_match_one_fn=_exact_fuzzy,
        )
        self.assertEqual(resolved, [])
        self.assertEqual(unmatched, [])

    def test_returns_tuple(self):
        result = _resolve_pick_names(
            tokens=["Ahri"],
            vocab=["Ahri"],
            fuzzy_match_one_fn=_exact_fuzzy,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_mixed_matched_and_unmatched(self):
        vocab = ["Jinx", "Ahri"]
        resolved, unmatched = _resolve_pick_names(
            tokens=["Jinx", "Nobody"],
            vocab=vocab,
            fuzzy_match_one_fn=_exact_fuzzy,
        )
        self.assertEqual(resolved[0], "Jinx")
        self.assertEqual(resolved[1], "Nobody")
        self.assertEqual(len(unmatched), 1)


class TestResolveCompactDraftSides(unittest.TestCase):

    def _resolve(self, text, vocab, blue_team="A", team_a="A", team_b="B", fuzzy=None):
        if fuzzy is None:
            fuzzy = _exact_fuzzy
        return resolve_compact_draft_sides(
            text=text,
            vocab=vocab,
            blue_team=blue_team,
            team_a=team_a,
            team_b=team_b,
            fuzzy_match_one_fn=fuzzy,
        )

    def test_returns_dict(self):
        result = self._resolve("", [])
        self.assertIsInstance(result, dict)

    def test_required_keys_present(self):
        result = self._resolve("", [])
        for key in ("resolved", "unmatched", "blue_side_team", "red_side_team", "picks_a", "picks_b"):
            self.assertIn(key, result)

    def test_blue_side_team_is_blue_team(self):
        result = self._resolve("", [], blue_team="TeamA", team_a="TeamA", team_b="TeamB")
        self.assertEqual(result["blue_side_team"], "TeamA")

    def test_red_side_team_opposite_of_blue(self):
        result = self._resolve("", [], blue_team="TeamA", team_a="TeamA", team_b="TeamB")
        self.assertEqual(result["red_side_team"], "TeamB")

    def test_blue_side_b_swaps_picks(self):
        vocab = ["Jinx", "Ahri", "Lux", "Thresh", "Blitzcrank",
                 "Zed", "Yasuo", "Orianna", "Kaisa", "Syndra"]
        tokens = " ".join(vocab[:10])
        result = self._resolve(tokens, vocab, blue_team="TeamB", team_a="TeamA", team_b="TeamB")
        # When blue_side_team == team_b, picks_a should be red (5:10) and picks_b should be blue (0:5)
        self.assertEqual(result["picks_b"], list(vocab[:5]))
        self.assertEqual(result["picks_a"], list(vocab[5:10]))

    def test_blue_side_a_assigns_normally(self):
        vocab = ["Jinx", "Ahri", "Lux", "Thresh", "Blitzcrank",
                 "Zed", "Yasuo", "Orianna", "Kaisa", "Syndra"]
        tokens = " ".join(vocab[:10])
        result = self._resolve(tokens, vocab, blue_team="TeamA", team_a="TeamA", team_b="TeamB")
        self.assertEqual(result["picks_a"], list(vocab[:5]))
        self.assertEqual(result["picks_b"], list(vocab[5:10]))

    def test_max_ten_picks_respected(self):
        vocab = [f"Champ{i}" for i in range(15)]
        tokens = " ".join(vocab[:15])
        result = self._resolve(tokens, vocab)
        self.assertLessEqual(len(result["resolved"]), 10)

    def test_empty_text_returns_empty_picks(self):
        result = self._resolve("", [])
        self.assertEqual(result["picks_a"], [])
        self.assertEqual(result["picks_b"], [])

    def test_resolved_list_type(self):
        result = self._resolve("", [])
        self.assertIsInstance(result["resolved"], list)

    def test_unmatched_list_type(self):
        result = self._resolve("", [])
        self.assertIsInstance(result["unmatched"], list)


if __name__ == "__main__":
    unittest.main()
