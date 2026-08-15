import unittest

from app_core.ml_marginal_guard_core import run_marginal_trust_guard


def _conf_fn(*_args, **_kwargs):
    return {"score": 72.0, "grade": "B", "source": "test"}


def _adj_fn(p_in, *_args, **_kwargs):
    return (float(p_in) + 0.01, {"enabled": True, "applied": True, "league": "LCK", "reason": "ok", "shrink": 0.1, "cap_hi": 0.84})


def _no_adj_fn(p_in, *_args, **_kwargs):
    return (float(p_in), {"enabled": True, "applied": False, "reason": "excluded_league", "league": "LCK"})


class TestRunMarginalTrustGuard(unittest.TestCase):

    def _run(self, p_map=0.60, p_series=0.65, adj_fn=None):
        return run_marginal_trust_guard(
            p_map_used=p_map,
            p_series_used=p_series,
            bo=3,
            ml_info={},
            league_hint="LCK",
            trace_n_team_eff=40.0,
            lv_last_report={},
            ml_trace_last={},
            confidence_baseline={},
            session_state={},
            ml_confidence_fn=_conf_fn,
            marginal_adjust_fn=adj_fn or _adj_fn,
            prob_win_series_fn=lambda p, _bo: p,
        )

    def test_p_map_updated(self):
        out = self._run()
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.61, places=6)

    def test_p_series_updated(self):
        out = self._run()
        self.assertAlmostEqual(float(out.get("p_series_used")), 0.61, places=6)

    def test_pipeline_step_stage(self):
        step = self._run().get("pipeline_step") or {}
        self.assertEqual(str(step.get("stage")), "marginal_trust_guard")

    def test_pipeline_step_applied(self):
        step = self._run().get("pipeline_step") or {}
        self.assertTrue(bool(step.get("applied", False)))

    def test_returns_dict(self):
        self.assertIsInstance(self._run(), dict)

    def test_required_keys_present(self):
        out = self._run()
        for key in ("p_map_used", "p_series_used", "pipeline_step", "trace_marginal_trust_info"):
            self.assertIn(key, out)

    def test_no_adjustment_when_not_applied(self):
        out = self._run(adj_fn=_no_adj_fn)
        self.assertAlmostEqual(float(out.get("p_map_used")), 0.60, places=6)

    def test_step_not_applied_when_no_change(self):
        out = self._run(adj_fn=_no_adj_fn)
        step = out.get("pipeline_step") or {}
        self.assertFalse(bool(step.get("applied", True)))


if __name__ == "__main__":
    unittest.main()
