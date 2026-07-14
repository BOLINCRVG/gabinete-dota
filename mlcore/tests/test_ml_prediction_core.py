"""Tests for app_core/ml_prediction_core.py.

Both compute_ml_core_prediction and compute_ml_post_layers accept
callbacks for all impure operations, making them purely testable
without any Streamlit state or file I/O.
"""

import math
import unittest
from datetime import date
from typing import Any, Dict, Optional

from app_core.ml_prediction_core import (
    compute_ml_core_prediction,
    compute_ml_post_layers,
)


# ---------------------------------------------------------------------------
# Stub builders
# ---------------------------------------------------------------------------

def _leagues_ok(ta, tb, year, lmode, fixed):
    return {"league_a": "LCK", "league_b": "LCK", "tier_a": "S", "tier_b": "S"}


def _leagues_fail(ta, tb, year, lmode, fixed):
    raise OSError("network error")


def _p_map_v2_ok(artifact_path, blue, red, as_of, **kw):
    return {"p_raw": 0.60, "p_cal": 0.58}


def _p_map_v2_none(artifact_path, blue, red, as_of, **kw):
    return None


def _p_map_v2_symmetric(artifact_path, blue, red, as_of, **kw):
    # Simulate slightly asymmetric forward/reverse
    if blue == "T1":
        return {"p_raw": 0.62, "p_cal": 0.60}
    else:  # reversed
        return {"p_raw": 0.42, "p_cal": 0.44}


def _microseg_identity(p, *, league, split, side, bo, phase, table,
                       min_n, strength, cap_pp, hierarchical, hierarchy_decay):
    return p, {"applied": True, "delta_pp": 0.0}


def _microseg_shifts(p, *, league, split, side, bo, phase, table,
                     min_n, strength, cap_pp, hierarchical, hierarchy_decay):
    return min(0.99, p + 0.03), {"applied": True, "delta_pp": 3.0}


def _microseg_mult(p, *, league, split, side, bo, phase, table,
                   min_n, strength, cap_pp, hierarchical, hierarchy_decay):
    # Shift DEPENDENTE de p (multiplicativo) — diferente de um shift constante,
    # NÃO cancela na simetrização de saída (é o que prova que microseg ainda
    # afeta o resultado mesmo com a simetria ligada).
    return min(0.99, max(0.01, p * 1.15)), {"applied": True, "delta_pp": 1.0}


def _microseg_raises(*a, **kw):
    raise ValueError("table error")


def _fused_postcal_identity(p, *, league_hint, use_fitted):
    return p, {"applied": False}


def _fused_postcal_shifts(p, *, league_hint, use_fitted):
    return min(0.99, p + 0.02), {"applied": True, "method": "platt"}


def _fused_postcal_mult(p, *, league_hint, use_fitted):
    # p-dependente (sobrevive à simetrização), ver _microseg_mult.
    return min(0.99, max(0.01, p * 1.15)), {"applied": True, "method": "platt"}


def _fused_postcal_raises(p, *, league_hint, use_fitted):
    raise TypeError("bad type")


def _build_microseg_returns(table_dict):
    def _fn(*, csv_sig, art_sig, lookback_games, shrink_k):
        return table_dict
    return _fn


def _load_fitted_params(sig):
    return {
        "microseg_calibration": {
            "table": {"LCK|Summer|blue|3|regular": {"bias": 0.01, "n": 100}}
        }
    }


def _load_fitted_params_empty(sig):
    return {}


def _file_sig(path):
    return (str(path), 0, 0)


def _default_core_kwargs(**overrides):
    kwargs = dict(
        team_a="T1",
        team_b="T2",
        artifact_path="/data/artifact.json",
        as_of_date=date(2024, 6, 1),
        achievement_path=None,
        ml_runtime_cfg={
            "ml_microseg_enabled": True,
            "ml_microseg_lookback_games": 1200,
            "ml_microseg_shrink_k": 90.0,
            "ml_microseg_min_n": 30,
            "ml_microseg_strength": 0.5,
            "ml_microseg_cap_pp": 3.0,
            "ml_microseg_hierarchical": True,
            "ml_microseg_hier_decay": 0.80,
            "ml_fused_post_cal_use_fitted": True,
        },
        filter_year=2024,
        filter_league="LCK",
        filter_split="Summer",
        filter_playoffs=None,
        bo=3,
        league_mode="fixed",
        csv_sig=("csv", 0, 0),
        art_sig=("art", 0, 0),
        fit_params_path=None,
        resolve_leagues_fn=_leagues_ok,
        p_map_v2_fn=_p_map_v2_ok,
        build_microseg_table_fn=_build_microseg_returns({}),
        apply_microseg_fn=_microseg_identity,
        apply_fused_postcal_fn=_fused_postcal_identity,
        load_fitted_params_fn=None,
        file_sig_fn=None,
    )
    kwargs.update(overrides)
    return kwargs


def _default_post_kwargs(**overrides):
    kwargs = dict(
        ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"},
                 "p_raw": 0.60, "p_cal": 0.58},
        p_map_cal=0.62,
        bo=3,
        ml_runtime_cfg={
            "wf_apply_ml_correction": True,
            "wf_ml_corr_strength": 0.60,
            "ml_players_context_use_fitted": True,
        },
        wf_report={"league": "LCK", "correction": 0.01},
        wf_correct_p_map_fn=lambda rep, p, *, league_hint, strength: (p + 0.01, {"applied": True}),
        fit_scales_for_league_fn=lambda league: (1.1, 0.9, "LCK"),
        fit_players_context_for_league_fn=lambda league: ({"enabled": True}, "fitted"),
        prob_win_series_fn=lambda p, bo: p ** 2 + (1 - p) ** 2 * p,  # simple BO3 approx
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Tests: compute_ml_core_prediction
# ---------------------------------------------------------------------------

class TestComputeMlCorePrediction(unittest.TestCase):

    def test_happy_path_returns_required_keys(self):
        out = compute_ml_core_prediction(**_default_core_kwargs())
        for key in ("ml_info", "p_map_raw", "p_map_cal", "league_a", "league_b",
                    "tier_a", "tier_b", "microseg_info", "fused_post_cal_info"):
            self.assertIn(key, out)

    def test_p_map_finite_when_v2_returns_value(self):
        out = compute_ml_core_prediction(**_default_core_kwargs())
        self.assertTrue(math.isfinite(out["p_map_raw"]))
        self.assertTrue(math.isfinite(out["p_map_cal"]))

    def test_p_map_nan_when_v2_returns_none(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_none,
        ))
        self.assertFalse(math.isfinite(out["p_map_raw"]))
        self.assertFalse(math.isfinite(out["p_map_cal"]))
        self.assertIsNone(out["ml_info"])

    def test_symmetry_averaging(self):
        """p_cal must be averaged with 1 - reverse p_cal."""
        # Forward: p_cal=0.60, Reverse: p_cal=0.44 → expected 0.5*(0.60 + 1-0.44) = 0.5*1.16 = 0.58
        out = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_microseg_fn=_microseg_identity,
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        self.assertAlmostEqual(out["p_map_cal"], 0.5 * (0.60 + (1.0 - 0.44)), places=5)
        self.assertAlmostEqual(out["p_map_raw"], 0.5 * (0.62 + (1.0 - 0.42)), places=5)

    def test_league_resolution_populated(self):
        out = compute_ml_core_prediction(**_default_core_kwargs())
        self.assertEqual(out["league_a"], "LCK")
        self.assertEqual(out["league_b"], "LCK")
        self.assertEqual(out["tier_a"], "S")

    def test_league_resolution_failure_does_not_crash(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            resolve_leagues_fn=_leagues_fail,
        ))
        self.assertEqual(out["league_a"], "")
        self.assertEqual(out["league_b"], "")
        self.assertTrue(math.isfinite(out["p_map_cal"]) or math.isnan(out["p_map_cal"]))

    def test_microseg_disabled_skips_calibration(self):
        cfg = dict(_default_core_kwargs()["ml_runtime_cfg"])
        cfg["ml_microseg_enabled"] = False
        called = []

        def _spy_microseg(*a, **kw):
            called.append(1)
            return _microseg_shifts(*a, **kw)

        compute_ml_core_prediction(**_default_core_kwargs(
            ml_runtime_cfg=cfg,
            apply_microseg_fn=_spy_microseg,
        ))
        self.assertEqual(len(called), 0, "microseg must not be called when disabled")

    def test_microseg_enabled_calls_calibration(self):
        called = []

        def _spy(p, **kw):
            called.append(p)
            return _microseg_identity(p, **kw)

        compute_ml_core_prediction(**_default_core_kwargs(apply_microseg_fn=_spy))
        # Chamado nas DUAS direções (forward + reverse) para simetrizar a saída.
        self.assertEqual(len(called), 2)

    def test_microseg_shifts_p_map_cal(self):
        # Core ASSIMÉTRICO (p_fwd != p_rev) + shift p-dependente: o microseg
        # sobrevive à simetrização e move p_map_cal.
        out_id = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_microseg_fn=_microseg_identity,
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        out_shift = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_microseg_fn=_microseg_mult,
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        self.assertGreater(out_shift["p_map_cal"], out_id["p_map_cal"])

    def test_constant_side_shift_cancels_under_symmetry(self):
        # Comportamento CORRETO da simetrização: um shift CONSTANTE (viés de lado
        # blue) é removido — pré-jogo você não sabe quem é blue, então não deve
        # contar. Documenta por que o teste antigo (shift constante) mudou.
        out_id = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_microseg_fn=_microseg_identity,
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        out_const = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_microseg_fn=_microseg_shifts,   # +0.03 constante
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        self.assertAlmostEqual(out_const["p_map_cal"], out_id["p_map_cal"], places=9)

    def test_microseg_exception_swallowed(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            apply_microseg_fn=_microseg_raises,
        ))
        # p_map_cal should still be finite (from symmetry averaging)
        self.assertTrue(math.isfinite(out["p_map_cal"]))

    def test_microseg_info_returned(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            apply_microseg_fn=_microseg_identity,
        ))
        self.assertIsInstance(out["microseg_info"], dict)

    def test_fused_postcal_applied(self):
        # Core assimétrico + fused p-dependente (sobrevive à simetrização).
        out_no_shift = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_fused_postcal_fn=_fused_postcal_identity,
        ))
        out_shift = compute_ml_core_prediction(**_default_core_kwargs(
            p_map_v2_fn=_p_map_v2_symmetric,
            apply_fused_postcal_fn=_fused_postcal_mult,
        ))
        self.assertGreater(out_shift["p_map_cal"], out_no_shift["p_map_cal"])

    def test_fused_postcal_exception_swallowed(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            apply_fused_postcal_fn=_fused_postcal_raises,
        ))
        self.assertTrue(math.isfinite(out["p_map_cal"]))

    def test_fitted_microseg_table_loaded_when_available(self):
        called = []

        def _spy_build(*, csv_sig, art_sig, lookback_games, shrink_k):
            called.append(1)
            return {}

        compute_ml_core_prediction(**_default_core_kwargs(
            fit_params_path="/data/params.json",
            load_fitted_params_fn=_load_fitted_params,
            file_sig_fn=_file_sig,
            build_microseg_table_fn=_spy_build,
        ))
        # Fitted table was loaded → live build should NOT be called
        self.assertEqual(len(called), 0, "live build must be skipped when fitted table is available")

    def test_live_microseg_build_called_when_no_fitted(self):
        called = []

        def _spy_build(*, csv_sig, art_sig, lookback_games, shrink_k):
            called.append(1)
            return {}

        compute_ml_core_prediction(**_default_core_kwargs(
            fit_params_path="/data/params.json",
            load_fitted_params_fn=_load_fitted_params_empty,
            file_sig_fn=_file_sig,
            build_microseg_table_fn=_spy_build,
        ))
        self.assertEqual(len(called), 1, "live build must be called when fitted table is empty")

    def test_p_map_values_in_valid_range(self):
        out = compute_ml_core_prediction(**_default_core_kwargs())
        if math.isfinite(out["p_map_cal"]):
            self.assertGreater(out["p_map_cal"], 0.0)
            self.assertLess(out["p_map_cal"], 1.0)
        if math.isfinite(out["p_map_raw"]):
            self.assertGreater(out["p_map_raw"], 0.0)
            self.assertLess(out["p_map_raw"], 1.0)

    def test_microseg_source_tagged_as_live_when_no_fitted(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            fit_params_path=None,
            apply_microseg_fn=_microseg_identity,
        ))
        if isinstance(out.get("microseg_info"), dict):
            self.assertEqual(out["microseg_info"].get("source"), "live")

    def test_microseg_source_tagged_as_fitted_when_loaded(self):
        out = compute_ml_core_prediction(**_default_core_kwargs(
            fit_params_path="/data/params.json",
            load_fitted_params_fn=_load_fitted_params,
            file_sig_fn=_file_sig,
            apply_microseg_fn=_microseg_identity,
        ))
        if isinstance(out.get("microseg_info"), dict):
            self.assertEqual(out["microseg_info"].get("source"), "fitted")


# ---------------------------------------------------------------------------
# Tests: compute_ml_post_layers
# ---------------------------------------------------------------------------

class TestComputeMlPostLayers(unittest.TestCase):

    def test_returns_required_keys(self):
        out = compute_ml_post_layers(**_default_post_kwargs())
        for key in ("p_map_cal", "p_series", "wf_corr_info",
                    "fit_team_scale", "fit_players_scale", "fit_league_used",
                    "players_context_cfg", "players_context_src"):
            self.assertIn(key, out)

    def test_wf_correction_applied_when_enabled(self):
        out_no_wf = compute_ml_post_layers(**_default_post_kwargs(
            ml_runtime_cfg={
                "wf_apply_ml_correction": False,
                "wf_ml_corr_strength": 0.60,
                "ml_players_context_use_fitted": True,
            },
        ))
        out_wf = compute_ml_post_layers(**_default_post_kwargs())
        self.assertGreater(out_wf["p_map_cal"], out_no_wf["p_map_cal"])

    def test_wf_correction_skipped_when_disabled(self):
        called = []

        def _spy(rep, p, *, league_hint, strength):
            called.append(p)
            return p + 0.01, {}

        compute_ml_post_layers(**_default_post_kwargs(
            ml_runtime_cfg={
                "wf_apply_ml_correction": False,
                "wf_ml_corr_strength": 0.60,
                "ml_players_context_use_fitted": True,
            },
            wf_correct_p_map_fn=_spy,
        ))
        self.assertEqual(len(called), 0)

    def test_wf_correction_skipped_when_p_map_nan(self):
        called = []

        def _spy(rep, p, *, league_hint, strength):
            called.append(p)
            return p + 0.01, {}

        compute_ml_post_layers(**_default_post_kwargs(
            p_map_cal=float("nan"),
            wf_correct_p_map_fn=_spy,
        ))
        self.assertEqual(len(called), 0)

    def test_wf_corr_info_stored(self):
        out = compute_ml_post_layers(**_default_post_kwargs())
        self.assertIsInstance(out["wf_corr_info"], dict)
        self.assertTrue(out["wf_corr_info"].get("applied"))

    def test_p_series_computed_from_prob_win_series(self):
        # WF adds 0.01 so p_map_cal after WF = 0.66
        p_after_wf = 0.63  # 0.62 + 0.01
        expected_series = p_after_wf ** 2 + (1 - p_after_wf) ** 2 * p_after_wf

        out = compute_ml_post_layers(**_default_post_kwargs())
        self.assertAlmostEqual(out["p_series"], expected_series, places=5)

    def test_p_series_nan_when_p_map_nan(self):
        out = compute_ml_post_layers(**_default_post_kwargs(
            p_map_cal=float("nan"),
        ))
        self.assertFalse(math.isfinite(out["p_series"]))

    def test_fit_scales_from_callback(self):
        out = compute_ml_post_layers(**_default_post_kwargs(
            fit_scales_for_league_fn=lambda league: (1.5, 0.7, "LPL"),
        ))
        self.assertAlmostEqual(out["fit_team_scale"], 1.5)
        self.assertAlmostEqual(out["fit_players_scale"], 0.7)
        self.assertEqual(out["fit_league_used"], "LPL")

    def test_players_context_from_callback(self):
        ctx = {"enabled": True, "max_weight": 0.6}
        out = compute_ml_post_layers(**_default_post_kwargs(
            fit_players_context_for_league_fn=lambda league: (ctx, "fitted"),
        ))
        self.assertEqual(out["players_context_cfg"], ctx)
        self.assertEqual(out["players_context_src"], "fitted")

    def test_league_hint_extracted_from_ml_info_when_same(self):
        captured = []

        def _wf(rep, p, *, league_hint, strength):
            captured.append(league_hint)
            return p, {}

        compute_ml_post_layers(**_default_post_kwargs(wf_correct_p_map_fn=_wf))
        self.assertEqual(captured[0], "LCK")

    def test_league_hint_empty_when_leagues_differ(self):
        captured = []

        def _wf(rep, p, *, league_hint, strength):
            captured.append(league_hint)
            return p, {}

        ml_info = {"blue": {"league": "LCK"}, "red": {"league": "LPL"}}
        compute_ml_post_layers(**_default_post_kwargs(ml_info=ml_info, wf_correct_p_map_fn=_wf))
        self.assertEqual(captured[0], "")

    def test_wf_exception_swallowed(self):
        def _raises(rep, p, *, league_hint, strength):
            raise ValueError("bad")

        out = compute_ml_post_layers(**_default_post_kwargs(
            wf_correct_p_map_fn=_raises,
        ))
        self.assertTrue(math.isfinite(out["p_map_cal"]))

    def test_fit_scales_exception_returns_defaults(self):
        def _raises(league):
            raise RuntimeError("no config")

        out = compute_ml_post_layers(**_default_post_kwargs(
            fit_scales_for_league_fn=_raises,
        ))
        self.assertAlmostEqual(out["fit_team_scale"], 1.0)
        self.assertAlmostEqual(out["fit_players_scale"], 1.0)
        self.assertEqual(out["fit_league_used"], "")

    def test_none_ml_info_does_not_crash(self):
        out = compute_ml_post_layers(**_default_post_kwargs(ml_info=None))
        self.assertIn("p_map_cal", out)
        self.assertIn("p_series", out)

    def test_error_ml_info_excluded_from_league_hint(self):
        captured = []

        def _wf(rep, p, *, league_hint, strength):
            captured.append(league_hint)
            return p, {}

        ml_info = {"__error__": True, "blue": {"league": "LCK"}, "red": {"league": "LCK"}}
        compute_ml_post_layers(**_default_post_kwargs(ml_info=ml_info, wf_correct_p_map_fn=_wf))
        self.assertEqual(captured[0], "")


if __name__ == "__main__":
    unittest.main()
