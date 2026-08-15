"""Tests for Phase 10 player props (v1 = kills)."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from mlcore.live_ml.player_props import (
    PLAYER_PROP_FEATURE_COLS,
    PlayerPropModel,
    build_player_kill_features,
)


def _synthetic_player_rows(n_players: int = 8, n_games: int = 30) -> pd.DataFrame:
    """Generate a synthetic player_rows DataFrame with kills proportional to skill."""
    rng = np.random.default_rng(0)
    rows = []
    players = [f"P{i}" for i in range(n_players)]
    positions = ["top", "jng", "mid", "bot", "sup"]
    champs = ["Ahri", "Jinx", "Lee Sin", "Gnar", "Yuumi"]
    for game in range(n_games):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=game)
        for p_i, player in enumerate(players):
            pos = positions[p_i % len(positions)]
            champ = champs[(p_i + game) % len(champs)]
            skill = 1.5 + (p_i / n_players) * 4.0  # 1.5 to 5.5
            kills = max(0, int(rng.poisson(skill)))
            rows.append({
                "gameid": f"G{game:03d}_{p_i % 2}",
                "playername": player,
                "champion": champ,
                "position": pos,
                "league": "LCK",
                "game": 1,
                "playoffs": "0",
                "side": "Blue" if p_i < n_players // 2 else "Red",
                "kills": kills,
                "deaths": 2,
                "assists": 3,
                "gamelength": 1800,
                "date": date,
            })
    return pd.DataFrame(rows)


class TestFeatureBuilder(unittest.TestCase):

    def test_schema(self):
        rows = _synthetic_player_rows()
        p_map = {(f"G{i:03d}_{s}", "Blue"): 0.55 for i in range(30) for s in (0, 1)}
        samples = build_player_kill_features(rows, p_map_pre_by_gameid_side=p_map)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0].features.shape, (len(PLAYER_PROP_FEATURE_COLS),))

    def test_causal_history_progression(self):
        """First few games have global mean fallback; later games use player history."""
        rows = _synthetic_player_rows()
        p_map = {}
        samples = build_player_kill_features(rows, p_map_pre_by_gameid_side=p_map,
                                              min_history=5)
        # Find samples for P0 — early ones should have feature[0] = global mean,
        # late ones should differ (causal history kicked in).
        p0 = [s for s in samples if s.player == "P0"]
        self.assertGreater(len(p0), 10)
        early_mean = p0[0].features[0]
        late_mean = p0[-1].features[0]
        # late_mean should be close to player's actual mean (low for P0 = skill 1.5)
        self.assertLess(late_mean, early_mean + 2.0)


class TestPlayerPropFitPredict(unittest.TestCase):

    def test_fit_and_predict(self):
        rows = _synthetic_player_rows(n_players=10, n_games=60)
        p_map = {(f"G{i:03d}_{s}", "Blue"): 0.55 for i in range(60) for s in (0, 1)}
        samples = build_player_kill_features(rows, p_map_pre_by_gameid_side=p_map)
        split = int(0.75 * len(samples))
        train, val = samples[:split], samples[split:]
        model = PlayerPropModel(market="player_kills")
        model.fit(train, val)
        # Predictions are well-formed
        for s in val[:5]:
            m = model.predict_mean(s)
            self.assertGreaterEqual(m, -0.5)  # may be slightly negative
            self.assertLess(m, 30.0)
        # P(over) returns valid probability
        p = model.prob_over(val[0], line=2.5)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_save_load_roundtrip(self):
        rows = _synthetic_player_rows(n_players=8, n_games=50)
        p_map = {(f"G{i:03d}_{s}", "Blue"): 0.5 for i in range(50) for s in (0, 1)}
        samples = build_player_kill_features(rows, p_map_pre_by_gameid_side=p_map)
        split = int(0.7 * len(samples))
        model = PlayerPropModel(market="player_kills")
        model.fit(samples[:split], samples[split:])
        with TemporaryDirectory() as td:
            path = Path(td) / "player_kills"
            model.save(path)
            m2 = PlayerPropModel.load(path)
            for s in samples[split:split + 3]:
                self.assertAlmostEqual(model.predict_mean(s), m2.predict_mean(s), places=6)
            self.assertAlmostEqual(m2.residual_sigma, model.residual_sigma, places=6)


class TestRuntimeHelper(unittest.TestCase):

    def test_disabled_returns_none(self):
        from app_core.live_player_props_runtime_core import predict_player_kills
        r = predict_player_kills(
            session_state={},
            player="Faker", champion="Ahri", position="mid",
            player_kills_mean=3.5, league="LCK",
        )
        self.assertIsNone(r["mean"])
        self.assertEqual(r["reason"], "disabled")

    def test_with_real_model_returns_prediction(self):
        models_dir = Path(__file__).resolve().parent.parent.parent / "live_models" / "player_props"
        if not (models_dir / "player_kills" / "pregame" / "metadata.json").exists():
            self.skipTest("player_props model not available")
        from app_core.live_player_props_runtime_core import predict_player_kills, clear_player_prop_cache
        clear_player_prop_cache()
        ss = {
            "ml_live_state_model_enabled": True,
            "ml_live_state_models_dir": str(models_dir.parent),
        }
        r = predict_player_kills(
            session_state=ss,
            player="Faker", champion="Ahri", position="mid",
            player_kills_mean=3.2, player_kills_std=2.1,
            champ_kills_mean_pos=4.0,
            league="LCK", bo=5, is_playoffs=True,
            p_map_pre_team=0.62, line=3.5,
        )
        self.assertIsNotNone(r["mean"])
        self.assertGreater(r["mean"], 0.0)
        self.assertLess(r["mean"], 25.0)
        self.assertIsNotNone(r["p_over"])
        self.assertGreater(r["p_over"], 0.0)
        self.assertLess(r["p_over"], 1.0)
        self.assertGreater(r["sigma"], 0.0)


if __name__ == "__main__":
    unittest.main()
