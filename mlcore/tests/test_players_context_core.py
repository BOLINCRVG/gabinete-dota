import unittest
from types import SimpleNamespace

import pandas as pd

from app_core.players_context_core import compute_players_context_quality_signal


class PlayersContextCoreTest(unittest.TestCase):
    def _build_df(self) -> pd.DataFrame:
        rows = []
        base_date = pd.Timestamp("2026-02-01")
        for i in range(10):
            d = (base_date + pd.Timedelta(days=i)).isoformat()
            # Team A: stronger quality profile.
            rows.append(
                {
                    "date": d,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LCK",
                    "team": "A",
                    "win": 1 if i != 3 else 0,
                    "game_time_min": 30.0 + (i % 3),
                    "kills_for": 17 + (i % 3),
                    "kills_against": 8 + (i % 2),
                    "towers_for": 9 + (i % 2),
                    "towers_against": 3,
                    "dragons_for": 3,
                    "dragons_against": 1,
                    "barons_for": 1,
                    "barons_against": 0,
                    "inhibitors_for": 2,
                    "inhibitors_against": 0,
                    "golddiffat15_for": 1200.0 if i != 2 else -700.0,  # includes one comeback spot
                }
            )
            # Team B: weaker quality profile.
            rows.append(
                {
                    "date": d,
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": False,
                    "league": "LPL",
                    "team": "B",
                    "win": 0 if i != 6 else 1,
                    "game_time_min": 33.5 + (i % 3),
                    "kills_for": 9 + (i % 2),
                    "kills_against": 16 + (i % 3),
                    "towers_for": 3,
                    "towers_against": 9 + (i % 2),
                    "dragons_for": 1,
                    "dragons_against": 3,
                    "barons_for": 0,
                    "barons_against": 1,
                    "inhibitors_for": 0,
                    "inhibitors_against": 2,
                    "golddiffat15_for": -1100.0 if i != 7 else 650.0,  # includes one throw spot
                }
            )
        return pd.DataFrame(rows)

    def test_quality_signal_pushes_better_team(self):
        df = self._build_df()
        f = SimpleNamespace(year=2026, split="Spring", playoffs=False, league=None)
        out = compute_players_context_quality_signal(
            team_games=df,
            team_a="A",
            team_b="B",
            filters=f,
            as_of_date="2026-03-01",
            enabled=True,
            lookback_games=12,
            min_games=6,
            strength=0.55,
            cap=0.30,
            interregional=False,
        )
        self.assertTrue(bool(out.get("applied", False)))
        self.assertGreater(float(out.get("edge_score", 0.0)), 0.0)
        self.assertGreater(float(out.get("delta", 0.0)), 0.0)
        self.assertGreaterEqual(int(out.get("n_min", 0) or 0), 6)

    def test_interregional_multiplier_shrinks_delta(self):
        df = self._build_df()
        f = {"year": 2026, "split": "Spring", "playoffs": False, "league": None}
        base = compute_players_context_quality_signal(
            team_games=df,
            team_a="A",
            team_b="B",
            filters=f,
            as_of_date="2026-03-01",
            enabled=True,
            lookback_games=12,
            min_games=6,
            strength=0.55,
            cap=0.30,
            interregional=False,
        )
        inter = compute_players_context_quality_signal(
            team_games=df,
            team_a="A",
            team_b="B",
            filters=f,
            as_of_date="2026-03-01",
            enabled=True,
            lookback_games=12,
            min_games=6,
            strength=0.55,
            cap=0.30,
            interregional=True,
            interregional_mult=0.70,
        )
        self.assertTrue(bool(base.get("applied", False)))
        self.assertTrue(bool(inter.get("applied", False)))
        self.assertLess(abs(float(inter.get("delta", 0.0))), abs(float(base.get("delta", 0.0))))

    def test_low_sample_blocks_application(self):
        df = self._build_df().head(4).copy()
        out = compute_players_context_quality_signal(
            team_games=df,
            team_a="A",
            team_b="B",
            filters={"year": 2026},
            as_of_date="2026-03-01",
            enabled=True,
            lookback_games=4,
            min_games=6,
        )
        self.assertFalse(bool(out.get("applied", False)))
        self.assertEqual(str(out.get("reason")), "low_sample")


if __name__ == "__main__":
    unittest.main()

