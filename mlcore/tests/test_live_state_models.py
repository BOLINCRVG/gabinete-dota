"""Tests for mlcore/live_ml/ — state-to-win models for live predictions."""
import unittest

import numpy as np
import pandas as pd

from mlcore.live_ml import (
    LiveSnapshot, StateToWinModel, build_snapshot_features,
)
from mlcore.live_ml.features import feature_names, _encode_tier


def _make_snapshots(n: int = 200, minute: int = 15, seed: int = 0):
    np.random.seed(seed)
    snaps = []
    for i in range(n):
        side = "Blue" if i % 2 == 0 else "Red"
        # Synthetic: large gold_diff favors winning
        gold_d = np.random.randn() * 1500
        true_p = 1.0 / (1.0 + np.exp(-gold_d / 2000.0))
        y = 1 if np.random.rand() < true_p else 0
        feat = np.array([
            0.0,            # z_pre
            float(gold_d),  # gold_diff
            float(np.random.randn() * 1000),  # xp_diff
            float(np.random.randn() * 50),     # cs_diff
            float(np.random.randn() * 3),      # kills_diff
            float(np.random.randn() * 3),      # deaths_diff
            float(np.random.randn() * 5),      # assists_diff
            0.0,            # league_tier
            1.0,            # bo
            0.0,            # playoffs
        ])
        snaps.append(LiveSnapshot(
            gameid=f"g{i}", minute=minute, side=side, p_pre=0.5,
            features=feat, y=int(y),
        ))
    return snaps


class TestFeatureNames(unittest.TestCase):

    def test_canonical_order(self):
        names = feature_names()
        self.assertEqual(names[0], "z_pre")
        self.assertIn("gold_diff", names)
        self.assertIn("league_tier", names)

    def test_no_duplicates(self):
        names = feature_names()
        self.assertEqual(len(names), len(set(names)))


class TestEncodeTier(unittest.TestCase):

    def test_lck_t1(self):
        self.assertEqual(_encode_tier("LCK"), 0)

    def test_unknown_returns_valid_int(self):
        result = _encode_tier("Zzz_NotARealLeague")
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 5)


class TestBuildSnapshotFeatures(unittest.TestCase):

    def _team_df(self):
        return pd.DataFrame([
            {
                "gameid": "g1", "side": "Blue", "participantid": 100, "result": 1,
                "league": "LCK", "playoffs": 0, "game": 1,
                "goldat15": 30000, "xpat15": 33000, "csat15": 500,
                "killsat15": 5, "deathsat15": 2, "assistsat15": 8,
                "opp_goldat15": 28000, "opp_xpat15": 31000, "opp_csat15": 480,
                "opp_killsat15": 2, "opp_deathsat15": 5, "opp_assistsat15": 3,
            },
            {
                "gameid": "g1", "side": "Red", "participantid": 200, "result": 0,
                "league": "LCK", "playoffs": 0, "game": 1,
                "goldat15": 28000, "xpat15": 31000, "csat15": 480,
                "killsat15": 2, "deathsat15": 5, "assistsat15": 3,
                "opp_goldat15": 30000, "opp_xpat15": 33000, "opp_csat15": 500,
                "opp_killsat15": 5, "opp_deathsat15": 2, "opp_assistsat15": 8,
            },
        ])

    def test_build_creates_snapshots(self):
        df = self._team_df()
        p_pre = {("g1", "Blue"): 0.60, ("g1", "Red"): 0.40}
        snaps = build_snapshot_features(df, minute=15, p_map_pre_by_gameid_side=p_pre)
        self.assertEqual(len(snaps), 2)

    def test_features_shape(self):
        df = self._team_df()
        p_pre = {("g1", "Blue"): 0.60, ("g1", "Red"): 0.40}
        snaps = build_snapshot_features(df, minute=15, p_map_pre_by_gameid_side=p_pre)
        self.assertEqual(snaps[0].features.shape[0], len(feature_names()))

    def test_drops_missing_p_pre(self):
        df = self._team_df()
        p_pre = {("g1", "Blue"): 0.60}  # no Red entry
        snaps = build_snapshot_features(df, minute=15, p_map_pre_by_gameid_side=p_pre)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].side, "Blue")

    def test_invalid_minute_raises(self):
        with self.assertRaises(ValueError):
            build_snapshot_features(pd.DataFrame(), minute=7,
                                     p_map_pre_by_gameid_side={})


class TestStateToWinModel(unittest.TestCase):

    def test_fit_with_val_set(self):
        train = _make_snapshots(n=300, seed=1)
        val = _make_snapshots(n=100, seed=2)
        model = StateToWinModel(minute=15)
        model.fit(train, val)
        self.assertIsNotNone(model.lgb_model)
        self.assertGreater(model.best_iteration, 0)

    def test_predict_in_bounds(self):
        train = _make_snapshots(n=200, seed=3)
        val = _make_snapshots(n=60, seed=4)
        model = StateToWinModel(minute=15)
        model.fit(train, val)
        preds = model.predict_proba(val[:10])
        self.assertTrue(np.all(preds > 0.0))
        self.assertTrue(np.all(preds < 1.0))

    def test_predict_one(self):
        train = _make_snapshots(n=200, seed=5)
        val = _make_snapshots(n=60, seed=6)
        model = StateToWinModel(minute=15)
        model.fit(train, val)
        p = model.predict_proba_one(val[0])
        self.assertIsInstance(p, float)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_no_train_raises(self):
        model = StateToWinModel(minute=15)
        with self.assertRaises(ValueError):
            model.fit([])

    def test_metadata_populated(self):
        train = _make_snapshots(n=200, seed=7)
        val = _make_snapshots(n=60, seed=8)
        model = StateToWinModel(minute=15)
        model.fit(train, val)
        self.assertIn("feature_importance", model.train_metadata)
        self.assertIn("best_iteration", model.train_metadata)
        self.assertIn("n_train", model.train_metadata)


class TestSerialization(unittest.TestCase):

    def test_save_load_roundtrip(self):
        import tempfile
        train = _make_snapshots(n=200, seed=9)
        val = _make_snapshots(n=60, seed=10)
        model = StateToWinModel(minute=15)
        model.fit(train, val)
        p_before = model.predict_proba(val[:5])
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            restored = StateToWinModel.load(tmp)
            p_after = restored.predict_proba(val[:5])
        for a, b in zip(p_before, p_after):
            self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
