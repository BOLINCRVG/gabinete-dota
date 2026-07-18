"""Tests for mlcore/live_ml/market_features.py and market_models.py."""
import unittest

import numpy as np
import pandas as pd

from mlcore.live_ml import (
    LiveMarketModel, LiveMarketSample, MARKET_FEATURE_COLS,
    SUPPORTED_MARKETS, build_market_features,
)


def _make_samples(n=300, market="total_kills", target_base=28.0, target_slope=0.001, seed=0):
    import random
    random.seed(seed)
    samples = []
    for i in range(n):
        gold_d = random.gauss(0, 2000)
        feat = np.array([
            0.0,                         # z_pre
            gold_d, abs(gold_d),         # gold features
            random.gauss(0, 1500), random.gauss(0, 30),
            random.uniform(5, 25), random.gauss(0, 3),
            random.uniform(5, 25), random.uniform(5, 30),
            0.0, 1.0, 0.0,
        ])
        target = max(5.0, target_base + target_slope * gold_d + random.gauss(0, 4))
        samples.append(LiveMarketSample(
            gameid=f"g{i}", minute=15, market=market,
            features=feat, target=target,
        ))
    return samples


class TestMarketFeatureSchema(unittest.TestCase):

    def test_feature_cols_non_empty(self):
        self.assertGreater(len(MARKET_FEATURE_COLS), 0)

    def test_supported_markets(self):
        for m in ("total_kills", "total_towers", "total_dragons",
                  "total_barons", "gamelength_min"):
            self.assertIn(m, SUPPORTED_MARKETS)


class TestBuildMarketFeatures(unittest.TestCase):

    def _df(self):
        return pd.DataFrame([
            {
                "gameid": "g1", "participantid": 100,
                "league": "LCK", "playoffs": 1, "game": 5,
                "teamkills": 18, "towers": 7, "dragons": 3, "barons": 1, "gamelength": 1800,
                "goldat15": 30000, "xpat15": 33000, "csat15": 500,
                "killsat15": 5, "deathsat15": 2, "assistsat15": 8,
                "opp_goldat15": 28000, "opp_xpat15": 31000, "opp_csat15": 480,
                "opp_killsat15": 2, "opp_deathsat15": 5, "opp_assistsat15": 3,
            },
            {
                "gameid": "g1", "participantid": 200,
                "league": "LCK", "playoffs": 1, "game": 5,
                "teamkills": 12, "towers": 4, "dragons": 1, "barons": 0, "gamelength": 1800,
                "goldat15": 28000, "xpat15": 31000, "csat15": 480,
                "killsat15": 2, "deathsat15": 5, "assistsat15": 3,
                "opp_goldat15": 30000, "opp_xpat15": 33000, "opp_csat15": 500,
                "opp_killsat15": 5, "opp_deathsat15": 2, "opp_assistsat15": 8,
            },
        ])

    def test_total_kills_target(self):
        samples = build_market_features(
            self._df(), minute=15, market="total_kills",
            p_map_pre_by_gameid_blue={"g1": 0.60},
        )
        self.assertEqual(len(samples), 1)
        self.assertAlmostEqual(samples[0].target, 30.0)  # 18 + 12

    def test_total_towers_target(self):
        samples = build_market_features(
            self._df(), minute=15, market="total_towers",
            p_map_pre_by_gameid_blue={"g1": 0.5},
        )
        self.assertAlmostEqual(samples[0].target, 11.0)  # 7 + 4

    def test_gamelength_converted_to_minutes(self):
        samples = build_market_features(
            self._df(), minute=15, market="gamelength_min",
            p_map_pre_by_gameid_blue={"g1": 0.5},
        )
        self.assertAlmostEqual(samples[0].target, 30.0)  # 1800 / 60

    def test_invalid_market_raises(self):
        with self.assertRaises(ValueError):
            build_market_features(self._df(), minute=15, market="not_a_market",
                                   p_map_pre_by_gameid_blue={})

    def test_invalid_minute_raises(self):
        with self.assertRaises(ValueError):
            build_market_features(self._df(), minute=7, market="total_kills",
                                   p_map_pre_by_gameid_blue={})


class TestLiveMarketModelFit(unittest.TestCase):

    def test_fit_returns_self(self):
        samples = _make_samples(n=300)
        train, val = samples[:240], samples[240:]
        m = LiveMarketModel(market="total_kills", minute=15)
        self.assertIs(m.fit(train, val), m)

    def test_fit_sets_residual_sigma(self):
        samples = _make_samples(n=300)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:240], samples[240:])
        self.assertGreater(m.residual_sigma, 0.0)

    def test_invalid_market_raises(self):
        with self.assertRaises(ValueError):
            LiveMarketModel(market="bad_market", minute=15)


class TestPredictAndProbOver(unittest.TestCase):

    def test_predict_mean_is_float(self):
        samples = _make_samples(n=300)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:240], samples[240:])
        pred = m.predict_mean(samples[250])
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

    def test_prob_over_in_bounds(self):
        samples = _make_samples(n=300)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:240], samples[240:])
        p = m.prob_over(samples[250], line=28.5)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_prob_over_decreases_with_higher_line(self):
        samples = _make_samples(n=300)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:240], samples[240:])
        p_low = m.prob_over(samples[250], line=10.0)
        p_high = m.prob_over(samples[250], line=50.0)
        self.assertGreater(p_low, p_high)

    def test_predict_batch(self):
        samples = _make_samples(n=300)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:240], samples[240:])
        preds = m.predict_mean_batch(samples[240:250])
        self.assertEqual(len(preds), 10)


class TestSerialization(unittest.TestCase):

    def test_roundtrip(self):
        import tempfile
        samples = _make_samples(n=200)
        m = LiveMarketModel(market="total_kills", minute=15)
        m.fit(samples[:160], samples[160:])
        pred_before = m.predict_mean(samples[170])
        sigma_before = m.residual_sigma
        with tempfile.TemporaryDirectory() as tmp:
            m.save(tmp)
            m2 = LiveMarketModel.load(tmp)
            self.assertEqual(m2.market, m.market)
            self.assertEqual(m2.minute, m.minute)
            self.assertAlmostEqual(m2.residual_sigma, sigma_before, places=4)
            self.assertAlmostEqual(m2.predict_mean(samples[170]), pred_before, places=4)


if __name__ == "__main__":
    unittest.main()
