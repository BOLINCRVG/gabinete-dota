"""Tests for the UI cache wrappers."""
from __future__ import annotations

import unittest

import pandas as pd

from app_core.ui_cache_core import (
    cached_champion_knowledge_registry,
    cached_champion_vocab,
    clear_ui_cache,
)


class TestChampionRegistryCache(unittest.TestCase):

    def setUp(self):
        clear_ui_cache()

    def test_returns_dict_with_meta(self):
        reg = cached_champion_knowledge_registry()
        self.assertIsInstance(reg, dict)
        # Always returns dict with at least these keys (even if file missing)
        self.assertIn("champions", reg)
        self.assertIn("aliases", reg)

    def test_second_call_returns_same_object(self):
        r1 = cached_champion_knowledge_registry()
        r2 = cached_champion_knowledge_registry()
        # Cache resource → same instance
        self.assertIs(r1, r2)


class TestChampionVocabCache(unittest.TestCase):

    def setUp(self):
        clear_ui_cache()

    def test_empty_df_returns_empty(self):
        self.assertEqual(cached_champion_vocab(None), [])
        self.assertEqual(cached_champion_vocab(pd.DataFrame()), [])

    def test_without_champion_column_returns_empty(self):
        df = pd.DataFrame({"foo": ["bar"]})
        self.assertEqual(cached_champion_vocab(df), [])

    def test_builds_sorted_unique_vocab(self):
        df = pd.DataFrame({"champion": ["Ahri", "Jinx", "Ahri", "Lee Sin", ""]})
        vocab = cached_champion_vocab(df)
        self.assertEqual(vocab, ["Ahri", "Jinx", "Lee Sin"])

    def test_same_champion_set_returns_same_cache(self):
        df1 = pd.DataFrame({"champion": ["Ahri", "Jinx", "Ahri"]})
        df2 = pd.DataFrame({"champion": ["Jinx", "Ahri"]})
        v1 = cached_champion_vocab(df1)
        v2 = cached_champion_vocab(df2)
        # Different DataFrames, same unique set → cache hit
        self.assertEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
