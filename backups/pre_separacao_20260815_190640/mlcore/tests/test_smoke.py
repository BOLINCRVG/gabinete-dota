import os
import unittest
import math

from mlcore.io_oracle import load_oracle_elixir_games
from mlcore.elo import EloModel
from mlcore.calibration import fit_platt_from_raw_probs
from mlcore.utils import logloss

class SmokeTest(unittest.TestCase):
    def test_loader_and_model(self):
        # Uses an env var to point to a csv, so the test can run in your repo without bundling data.
        csv = os.environ.get("MLCORE_CSV")
        if not csv:
            self.skipTest("Set MLCORE_CSV to run this smoke test with a real CSV.")
        games = load_oracle_elixir_games(csv)
        self.assertTrue(len(games) > 100)
        model = EloModel()
        model.fit(games.iloc[:5000])
        p_raw = [model.p_blue(r.blue, r.red) for r in games.iloc[5000:5200].itertuples(index=False)]
        y = [int(r.blue_win) for r in games.iloc[5000:5200].itertuples(index=False)]
        cal = fit_platt_from_raw_probs(y, p_raw)
        p_cal = [cal.predict_proba(p) for p in p_raw]
        ll = float(logloss(y, p_cal))
        self.assertTrue(math.isfinite(ll) and ll >= 0.0)  # ensure pipeline runs end-to-end

if __name__ == "__main__":
    unittest.main()
