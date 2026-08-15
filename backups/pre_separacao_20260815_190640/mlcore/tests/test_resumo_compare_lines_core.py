"""Tests for resumo_compare_lines_core — sweet-spot grouping + sparkline."""
from __future__ import annotations

import unittest

from app_core.resumo_compare_lines_core import build_line_comparisons, sparkline


def _row(metric, lado, line, odd_hist):
    return {
        "_metric": metric, "Lado": lado, "_line": line,
        "Odd (Hist)": odd_hist,
        "Mercado": f"{metric} ({line})", "Linha": str(line),
    }


class TestBuildLineComparisons(unittest.TestCase):

    def test_groups_same_metric_multiple_lines(self):
        rows = [
            _row("kills", "Under", 27.5, 1.78),
            _row("kills", "Under", 28.5, 1.62),
            _row("kills", "Under", 29.5, 1.50),
        ]
        groups = build_line_comparisons(rows)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["metric"], "kills")
        self.assertEqual(g["side"], "Under")
        self.assertEqual(len(g["lines"]), 3)

    def test_lines_sorted_ascending(self):
        rows = [
            _row("kills", "Under", 29.5, 1.50),
            _row("kills", "Under", 27.5, 1.78),
        ]
        g = build_line_comparisons(rows)[0]
        self.assertEqual([l["line"] for l in g["lines"]], [27.5, 29.5])

    def test_best_idx_highest_prob(self):
        # Under 27.5 @ 1.78 → p=0.562; 29.5 @ 1.50 → p=0.667 (higher)
        rows = [
            _row("kills", "Under", 27.5, 1.78),
            _row("kills", "Under", 29.5, 1.50),
        ]
        g = build_line_comparisons(rows)[0]
        self.assertEqual(g["lines"][g["best_idx"]]["line"], 29.5)

    def test_single_line_not_grouped(self):
        rows = [_row("kills", "Under", 27.5, 1.78)]
        self.assertEqual(build_line_comparisons(rows), [])

    def test_ml_skipped(self):
        rows = [
            {"_metric": "00_ml_A", "Lado": "T1", "_line": 0.0, "Odd (Hist)": 1.5,
             "Mercado": "ML", "Linha": "-"},
            {"_metric": "00_ml_B", "Lado": "GenG", "_line": 0.0, "Odd (Hist)": 2.5,
             "Mercado": "ML", "Linha": "-"},
        ]
        self.assertEqual(build_line_comparisons(rows), [])

    def test_over_and_under_separate_groups(self):
        rows = [
            _row("kills", "Over", 27.5, 2.0),
            _row("kills", "Over", 28.5, 2.2),
            _row("kills", "Under", 27.5, 1.8),
            _row("kills", "Under", 28.5, 1.6),
        ]
        groups = build_line_comparisons(rows)
        self.assertEqual(len(groups), 2)
        sides = {g["side"] for g in groups}
        self.assertEqual(sides, {"Over", "Under"})

    def test_missing_odd_handled(self):
        rows = [
            _row("towers", "Under", 11.5, None),
            _row("towers", "Under", 12.5, 1.7),
        ]
        g = build_line_comparisons(rows)[0]
        # best should be the one with a valid prob
        self.assertEqual(g["lines"][g["best_idx"]]["line"], 12.5)


class TestSparkline(unittest.TestCase):

    def test_ascending(self):
        s = sparkline([0.1, 0.5, 0.9])
        self.assertEqual(len(s), 3)
        # First should be lowest tick, last highest
        self.assertEqual(s[0], "▁")
        self.assertEqual(s[-1], "█")

    def test_flat_values_midtick(self):
        s = sparkline([0.5, 0.5, 0.5])
        self.assertEqual(len(s), 3)

    def test_none_becomes_space(self):
        s = sparkline([0.1, None, 0.9])
        self.assertEqual(s[1], " ")

    def test_empty_returns_empty(self):
        self.assertEqual(sparkline([]), "")
        self.assertEqual(sparkline([None, None]), "")


if __name__ == "__main__":
    unittest.main()
