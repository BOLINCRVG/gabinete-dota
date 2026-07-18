"""Tests for the Phase 8 first-X market modules.

Covers:
  - feature builder produces correct schema
  - feature target resolution from the BLUE row
  - FirstMarketModel fit/predict/save/load roundtrip on synthetic data
  - runtime helper predict_first_market with toggles and side mapping
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mlcore.live_ml.first_markets import (
    FIRST_FEATURE_COLS,
    FIRST_MARKETS,
    FirstMarketModel,
    build_first_market_features,
)


def _make_synthetic_team_rows(n_games: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic team_rows DataFrame with two participants per gameid."""
    rng = np.random.default_rng(int(seed))
    rows = []
    for gid in range(n_games):
        # P(blue wins first_blood) ~ p_pre[gid]
        p_pre = float(rng.uniform(0.3, 0.7))
        # Higher p_pre means blue more likely
        outcomes = {
            "firstblood": int(rng.random() < p_pre),
            "firstdragon": int(rng.random() < (0.5 + 0.3 * (p_pre - 0.5))),
            "firsttower": int(rng.random() < (0.5 + 0.5 * (p_pre - 0.5))),
            "firstherald": int(rng.random() < (0.5 + 0.4 * (p_pre - 0.5))),
            "firstbaron": int(rng.random() < (0.5 + 0.8 * (p_pre - 0.5))),
        }
        # Two rows: blue (pid 100) and red (pid 200), complementary outcomes
        for pid, side in ((100, "Blue"), (200, "Red")):
            row = {
                "gameid": f"GAME_{gid:04d}",
                "participantid": pid,
                "side": side,
                "league": "LCK" if gid % 2 == 0 else "LFL",
                "game": 1,
                "playoffs": "0",
            }
            for col, blue_won in outcomes.items():
                row[col] = float(blue_won if pid == 100 else (1 - blue_won))
            rows.append(row)
    return pd.DataFrame(rows)


class TestFeatureBuilder(unittest.TestCase):

    def test_schema_size(self):
        team_rows = _make_synthetic_team_rows(n_games=50)
        p_map = {f"GAME_{i:04d}": 0.55 for i in range(50)}
        samples = build_first_market_features(team_rows, market="first_blood",
                                               p_map_pre_by_gameid_blue=p_map)
        self.assertEqual(len(samples), 50)
        self.assertEqual(samples[0].features.shape, (len(FIRST_FEATURE_COLS),))

    def test_target_taken_from_blue_row(self):
        # Custom: blue won FB in game 0
        team_rows = pd.DataFrame([
            {"gameid": "G", "participantid": 100, "side": "Blue", "league": "LCK",
             "game": 1, "playoffs": "0", "firstblood": 1.0},
            {"gameid": "G", "participantid": 200, "side": "Red", "league": "LCK",
             "game": 1, "playoffs": "0", "firstblood": 0.0},
        ])
        samples = build_first_market_features(team_rows, market="first_blood",
                                               p_map_pre_by_gameid_blue={"G": 0.55})
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].target, 1)

    def test_skip_inconsistent_rows(self):
        team_rows = pd.DataFrame([
            {"gameid": "G", "participantid": 100, "side": "Blue", "league": "LCK",
             "game": 1, "playoffs": "0", "firstblood": 1.0},
            {"gameid": "G", "participantid": 200, "side": "Red", "league": "LCK",
             "game": 1, "playoffs": "0", "firstblood": 1.0},  # both =1 → invalid
        ])
        samples = build_first_market_features(team_rows, market="first_blood",
                                               p_map_pre_by_gameid_blue={"G": 0.55})
        self.assertEqual(len(samples), 0)

    def test_unknown_market_raises(self):
        team_rows = _make_synthetic_team_rows(n_games=10)
        with self.assertRaises(ValueError):
            build_first_market_features(team_rows, market="not_a_market",
                                         p_map_pre_by_gameid_blue={})

    def test_missing_column_returns_empty(self):
        # team_rows without "firstblood" column
        team_rows = pd.DataFrame([{"gameid": "G", "participantid": 100,
                                     "side": "Blue", "league": "LCK", "game": 1, "playoffs": "0"}])
        samples = build_first_market_features(team_rows, market="first_blood",
                                               p_map_pre_by_gameid_blue={"G": 0.5})
        self.assertEqual(len(samples), 0)


class TestFirstMarketModelFitPredict(unittest.TestCase):

    def test_fit_and_predict(self):
        team_rows = _make_synthetic_team_rows(n_games=300, seed=42)
        p_map = {f"GAME_{i:04d}": float(0.3 + 0.4 * (i / 300)) for i in range(300)}
        samples = build_first_market_features(team_rows, market="first_baron",
                                               p_map_pre_by_gameid_blue=p_map)
        split = int(0.7 * len(samples))
        train, val = samples[:split], samples[split:]
        model = FirstMarketModel(market="first_baron")
        model.fit(train, val)
        # Predictions are well-formed probabilities
        for s in val[:5]:
            p = model.predict_blue(s)
            self.assertTrue(0.0 < p < 1.0)
        # Batch matches single
        batch = model.predict_blue_batch(val[:5])
        for i, s in enumerate(val[:5]):
            self.assertAlmostEqual(model.predict_blue(s), batch[i], places=6)

    def test_save_load_roundtrip(self):
        team_rows = _make_synthetic_team_rows(n_games=200, seed=1)
        p_map = {f"GAME_{i:04d}": 0.55 for i in range(200)}
        samples = build_first_market_features(team_rows, market="first_blood",
                                               p_map_pre_by_gameid_blue=p_map)
        model = FirstMarketModel(market="first_blood")
        model.fit(samples[:150], samples[150:])
        with TemporaryDirectory() as td:
            path = Path(td) / "first_blood"
            model.save(path)
            self.assertTrue((path / "lgb_model.txt").exists())
            self.assertTrue((path / "metadata.json").exists())
            m2 = FirstMarketModel.load(path)
            # Predictions equal
            for s in samples[150:155]:
                self.assertAlmostEqual(model.predict_blue(s), m2.predict_blue(s), places=6)
            # Base rate preserved
            self.assertAlmostEqual(m2.base_rate_blue, model.base_rate_blue, places=6)


class TestRuntimeHelper(unittest.TestCase):

    def test_disabled_returns_none(self):
        from app_core.live_first_markets_runtime_core import predict_first_market
        r = predict_first_market(session_state={}, market="first_blood")
        self.assertIsNone(r["p_team_a"])
        self.assertEqual(r["reason"], "disabled")

    def test_unknown_market_returns_none(self):
        from app_core.live_first_markets_runtime_core import predict_first_market
        r = predict_first_market(
            session_state={"ml_live_state_model_enabled": True},
            market="not_a_market",
        )
        self.assertIsNone(r["p_team_a"])
        self.assertEqual(r["reason"], "market_not_active")

    def test_with_real_model_returns_probability(self):
        models_dir = Path(__file__).resolve().parent.parent.parent / "live_models" / "first_markets"
        if not (models_dir / "first_blood" / "pregame" / "metadata.json").exists():
            self.skipTest("first_markets models not available")
        from app_core.live_first_markets_runtime_core import predict_first_market, clear_first_market_model_cache
        clear_first_market_model_cache()
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(models_dir.parent),
        }
        r = predict_first_market(
            session_state=ss, market="first_blood",
            p_map_pre=0.65, league="LCK", bo=5, is_playoffs=True,
        )
        p = r["p_team_a"]
        self.assertIsNotNone(p)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)
        # With p_map_pre=0.65 (blue favored), p_team_a should be > 0.5 most likely
        self.assertGreater(p, 0.4)  # weak check, conservative

    def test_team_a_is_red_flips_probability(self):
        models_dir = Path(__file__).resolve().parent.parent.parent / "live_models" / "first_markets"
        if not (models_dir / "first_blood" / "pregame" / "metadata.json").exists():
            self.skipTest("first_markets models not available")
        from app_core.live_first_markets_runtime_core import predict_first_market, clear_first_market_model_cache
        clear_first_market_model_cache()
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(models_dir.parent),
        }
        r_blue = predict_first_market(
            session_state=ss, market="first_blood",
            p_map_pre=0.55, league="LCK", bo=5, is_playoffs=True,
            team_a_is_blue=True,
        )
        r_red = predict_first_market(
            session_state=ss, market="first_blood",
            p_map_pre=0.55, league="LCK", bo=5, is_playoffs=True,
            team_a_is_blue=False,
        )
        self.assertAlmostEqual(r_blue["p_team_a"] + r_red["p_team_a"], 1.0, places=6)

    def test_available_first_markets(self):
        from app_core.live_first_markets_runtime_core import available_first_markets
        models_dir = Path(__file__).resolve().parent.parent.parent / "live_models" / "first_markets"
        if not models_dir.exists():
            self.skipTest("first_markets dir not present")
        markets = available_first_markets(models_dir)
        # All 5 should be available (they all passed the gate)
        self.assertEqual(set(markets), set(FIRST_MARKETS))


if __name__ == "__main__":
    unittest.main()
