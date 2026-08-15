import json
import math
import unittest
from pathlib import Path
from typing import Any, Dict

from app_core.ml_orchestrator_core import run_ml_post_layers
from app_core.ml_pre_pipeline_core import run_ml_pre_pipeline
from app_core.ml_trace_payload_core import build_ml_trace_payload
from mlcore.bo import prob_win_series


def _f(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return float(x) if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _copy_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


class TestMlParityGolden(unittest.TestCase):
    def setUp(self):
        self.fixture_path = Path(__file__).resolve().parent / "fixtures" / "ml_parity_golden.json"
        if not self.fixture_path.exists():
            self.skipTest("ml_parity_golden fixture not found.")
        self.fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.tol = float(self.fixture.get("tolerance", 1e-8) or 1e-8)

    def _feature_flag_fn(self, flags: Dict[str, Any]):
        ff_map = dict(flags) if isinstance(flags, dict) else {}

        def _ff(key: str, default: bool = False) -> bool:
            return bool(ff_map.get(str(key), bool(default)))

        return _ff

    def _build_live_layers_stub(self, scenario: Dict[str, Any]):
        layers = list(scenario.get("post_live_layers", [])) if isinstance(scenario.get("post_live_layers"), list) else []

        stage_flag_key = {
            "style_matchup_b_live": "ff_ml_style_b_lite_live",
            "league_rank_sos_live": "ff_ml_league_rank_sos_live",
            "roster_continuity_live": "ff_ml_roster_continuity_live",
            "hier_calibration_live": "ff_ml_hier_calibration_live",
            "meta_blend_c_lite_live": "ff_ml_meta_c_lite_live",
            "layer_uncertainty_live": "ff_ml_layer_uncertainty_live",
        }
        stage_trace_key = {
            "style_matchup_b_live": "trace_style_b_lite_live",
            "league_rank_sos_live": "trace_league_rank_sos_live",
            "roster_continuity_live": "trace_roster_continuity_live",
            "hier_calibration_live": "trace_hier_calibration_live",
            "meta_blend_c_lite_live": "trace_meta_c_lite_live",
            "layer_uncertainty_live": "trace_layer_uncertainty_live",
        }

        def _run_live_layers(**kwargs):
            bo_i = int(kwargs.get("bo", 1) or 1)
            ff = kwargs.get("feature_flag_enabled_fn")
            ff_fn = ff if callable(ff) else (lambda _k, d=False: bool(d))
            prob_fn = kwargs.get("prob_win_series_fn")
            prob_series = prob_fn if callable(prob_fn) else (lambda p, _bo: float(p))

            p_map = _f(kwargs.get("p_map_used"), float("nan"))
            p_series = _f(kwargs.get("p_series_used"), float("nan"))
            steps = kwargs.get("trace_pipeline_steps")
            if not isinstance(steps, list):
                steps = []

            traces: Dict[str, Dict[str, Any]] = {
                "trace_style_b_lite_live": {
                    "enabled": bool(ff_fn("ff_ml_style_b_lite_live", True)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                },
                "trace_league_rank_sos_live": {
                    "enabled": bool(ff_fn("ff_ml_league_rank_sos_live", False)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                },
                "trace_roster_continuity_live": {
                    "enabled": bool(ff_fn("ff_ml_roster_continuity_live", False)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                },
                "trace_hier_calibration_live": {
                    "enabled": bool(ff_fn("ff_ml_hier_calibration_live", False)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                },
                "trace_meta_c_lite_live": {
                    "enabled": bool(ff_fn("ff_ml_meta_c_lite_live", False)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                },
                "trace_layer_uncertainty_live": {
                    "enabled": bool(ff_fn("ff_ml_layer_uncertainty_live", True)),
                    "applied": False,
                    "reason": "disabled_by_flag",
                    "p_before": (float(p_map) if math.isfinite(float(p_map)) else None),
                    "p_after": None,
                    "p_series_before": (float(p_series) if math.isfinite(float(p_series)) else None),
                    "p_series_after": None,
                },
            }

            for row in layers:
                if not isinstance(row, dict):
                    continue
                stage = str(row.get("stage", "") or "").strip()
                if not stage:
                    continue
                flag_key = stage_flag_key.get(stage, "")
                if flag_key and (not bool(ff_fn(flag_key, False))):
                    continue

                before_map = float(p_map) if math.isfinite(float(p_map)) else float("nan")
                before_series = float(p_series) if math.isfinite(float(p_series)) else float("nan")
                after_map = _f(row.get("p_after"), before_map)
                after_series = _f(row.get("p_series_after"), float("nan"))
                if not math.isfinite(after_series):
                    try:
                        after_series = float(prob_series(float(after_map), int(bo_i)))
                    except Exception:
                        after_series = float(before_series)
                applied = bool(row.get("applied", True))

                step = {
                    "stage": stage,
                    "before": (float(before_map) if math.isfinite(float(before_map)) else None),
                    "after": (float(after_map) if math.isfinite(float(after_map)) else None),
                    "applied": bool(applied),
                    "enabled": True,
                    "source": "golden_stub",
                }
                steps.append(step)

                trace = {
                    "enabled": True,
                    "applied": bool(applied),
                    "method": str(row.get("method", "golden_stub") or "golden_stub"),
                    "reason": str(row.get("reason", "") or ""),
                    "p_before": (float(before_map) if math.isfinite(float(before_map)) else None),
                    "p_after": (float(after_map) if math.isfinite(float(after_map)) else None),
                    "diff_pp": (
                        float(100.0 * (float(after_map) - float(before_map)))
                        if math.isfinite(float(before_map)) and math.isfinite(float(after_map))
                        else None
                    ),
                }
                if stage == "layer_uncertainty_live":
                    trace["p_series_before"] = (
                        float(before_series) if math.isfinite(float(before_series)) else None
                    )
                    trace["p_series_after"] = (
                        float(after_series) if math.isfinite(float(after_series)) else None
                    )
                    trace["diff_pp_series"] = (
                        float(100.0 * (float(after_series) - float(before_series)))
                        if math.isfinite(float(before_series)) and math.isfinite(float(after_series))
                        else None
                    )
                if stage == "league_rank_sos_live":
                    trace["p_rank"] = (
                        float(row.get("p_rank"))
                        if row.get("p_rank") is not None
                        else (float(after_map) if math.isfinite(float(after_map)) else None)
                    )
                    trace["weight"] = float(row.get("weight", 0.20) or 0.20)
                if stage == "roster_continuity_live":
                    trace["target_p"] = (
                        float(row.get("target_p"))
                        if row.get("target_p") is not None
                        else (float(after_map) if math.isfinite(float(after_map)) else None)
                    )
                    trace["weight"] = float(row.get("weight", 0.20) or 0.20)
                if stage == "hier_calibration_live":
                    trace["scope_used"] = str(row.get("scope_used", "pair") or "pair")
                    trace["bias_adj"] = float(row.get("bias_adj", 0.0) or 0.0)
                if stage == "meta_blend_c_lite_live":
                    trace["p_anchor"] = (
                        float(row.get("p_anchor"))
                        if row.get("p_anchor") is not None
                        else (float(before_map) if math.isfinite(float(before_map)) else None)
                    )

                trace_key = stage_trace_key.get(stage)
                if trace_key:
                    traces[trace_key] = dict(trace)
                if applied:
                    p_map = float(after_map)
                    p_series = float(after_series)

            return {
                "p_map_used": float(p_map),
                "p_series_used": float(p_series),
                "_ff_ml_style_b_lite_live": bool(ff_fn("ff_ml_style_b_lite_live", True)),
                "_ff_ml_league_rank_sos_live": bool(ff_fn("ff_ml_league_rank_sos_live", False)),
                "_ff_ml_roster_continuity_live": bool(ff_fn("ff_ml_roster_continuity_live", False)),
                "_ff_ml_hier_calibration_live": bool(ff_fn("ff_ml_hier_calibration_live", False)),
                "_ff_ml_meta_c_lite_live": bool(ff_fn("ff_ml_meta_c_lite_live", False)),
                "_ff_ml_layer_uncertainty_live": bool(ff_fn("ff_ml_layer_uncertainty_live", True)),
                "trace_style_b_lite_live": dict(traces["trace_style_b_lite_live"]),
                "trace_league_rank_sos_live": dict(traces["trace_league_rank_sos_live"]),
                "trace_roster_continuity_live": dict(traces["trace_roster_continuity_live"]),
                "trace_hier_calibration_live": dict(traces["trace_hier_calibration_live"]),
                "trace_meta_c_lite_live": dict(traces["trace_meta_c_lite_live"]),
                "trace_layer_uncertainty_live": dict(traces["trace_layer_uncertainty_live"]),
                "trace_pipeline_steps": list(steps),
            }

        return _run_live_layers

    def _build_marginal_stub(self, scenario: Dict[str, Any]):
        marginal_cfg = _copy_dict(scenario.get("marginal"))

        def _run_marginal(**kwargs):
            bo_i = int(kwargs.get("bo", 1) or 1)
            prob_fn = kwargs.get("prob_win_series_fn")
            prob_series = prob_fn if callable(prob_fn) else (lambda p, _bo: float(p))

            p_in = _f(kwargs.get("p_map_used"), float("nan"))
            ps_in = _f(kwargs.get("p_series_used"), float("nan"))
            enabled = bool(marginal_cfg.get("enabled", True))
            applied = bool(marginal_cfg.get("applied", False)) if enabled else False

            p_out = float(p_in)
            ps_out = float(ps_in)
            if applied:
                p_out = _f(marginal_cfg.get("p_out"), p_out)
                ps_out = _f(marginal_cfg.get("p_series_out"), float("nan"))
                if not math.isfinite(float(ps_out)):
                    try:
                        ps_out = float(prob_series(float(p_out), int(bo_i)))
                    except Exception:
                        ps_out = float(ps_in)

            info = {
                "enabled": bool(enabled),
                "applied": bool(applied),
                "reason": str(marginal_cfg.get("reason", "not_applied") or "not_applied"),
                "league": str(marginal_cfg.get("league", kwargs.get("league_hint", "")) or ""),
                "tier": str(marginal_cfg.get("tier", "") or ""),
                "band": str(marginal_cfg.get("band", "") or ""),
                "p_in": (float(p_in) if math.isfinite(float(p_in)) else None),
                "p_out": (float(p_out) if math.isfinite(float(p_out)) else None),
            }

            out = {
                "p_map_used": float(p_out),
                "p_series_used": float(ps_out),
                "trace_marginal_trust_info": info,
                "caption_applied": "",
                "caption_not_applied": "",
            }
            if applied:
                out["pipeline_step"] = {
                    "stage": "marginal_trust_guard",
                    "before": (float(p_in) if math.isfinite(float(p_in)) else None),
                    "after": (float(p_out) if math.isfinite(float(p_out)) else None),
                    "applied": True,
                    "enabled": True,
                    "source": "golden_stub",
                }
            return out

        return _run_marginal

    def _build_trace_ctx(
        self,
        scenario: Dict[str, Any],
        pre_out: Dict[str, Any],
        post_out: Dict[str, Any],
    ) -> Dict[str, Any]:
        league = str(scenario.get("league", "") or "")
        trace_ctx: Dict[str, Any] = {
            "generated_at": "2026-01-01T00:00:00",
            "season_label": str(scenario.get("season", "2026")),
            "teamA": str(scenario.get("teamA", "")),
            "teamB": str(scenario.get("teamB", "")),
            "bo": int(scenario.get("bo", 1) or 1),
            "_input_signature_hash": str(scenario.get("id", "")),
            "_input_signature_payload": {"scenario_id": str(scenario.get("id", ""))},
            "year_opt": str(scenario.get("year_opt", "2026")),
            "split_opt": str(scenario.get("split_opt", "S1")),
            "playoffs_opt": str(scenario.get("playoffs_opt", "")),
            "league_mode": "fixed",
            "fixed_league": league,
            "ml_engine": "mlcore_v2",
            "p_map_raw": _f(scenario.get("p_map_raw"), float("nan")),
            "p_map_cal": _f(scenario.get("p_map_cal"), float("nan")),
            "p_map_used": _f(post_out.get("p_map_used"), float("nan")),
            "p_series_used": _f(post_out.get("p_series_used"), float("nan")),
            "trace_p_map_team": _f(pre_out.get("trace_p_map_team"), float("nan")),
            "trace_p_map_players": _f(pre_out.get("trace_p_map_players"), float("nan")),
            "trace_p_map_fused": _f(pre_out.get("trace_p_map_fused"), float("nan")),
            "trace_n_team_eff": _f(pre_out.get("trace_n_team_eff"), float("nan")),
            "trace_n_players_eff": _f(pre_out.get("trace_n_players_eff"), float("nan")),
            "trace_fusion_info": _copy_dict(pre_out.get("trace_fusion_info")),
            "trace_uncertainty_info": _copy_dict(pre_out.get("trace_uncertainty_info")),
            "_delta_mode": str(pre_out.get("_delta_mode", "") or ""),
            "_delta_cap": _f(pre_out.get("_delta_cap"), 0.0),
            "_delta_slope": _f(pre_out.get("_delta_slope"), 0.0),
            "lineup_mode": str(pre_out.get("lineup_mode", "") or ""),
            "match_league": str(pre_out.get("match_league", "") or ""),
            "delta_players": _f(pre_out.get("delta_players"), 0.0),
            "delta_players_eff": _f(pre_out.get("delta_players_eff"), 0.0),
            "trace_form_info": _copy_dict(pre_out.get("trace_form_info")),
            "trace_hybrid_info": _copy_dict(pre_out.get("trace_hybrid_info")),
            "trace_conf_guard_info": _copy_dict(pre_out.get("trace_conf_guard_info")),
            "trace_patch_guard_info": _copy_dict(pre_out.get("trace_patch_guard_info")),
            "trace_series_gamma_info": _copy_dict(pre_out.get("trace_series_gamma_info")),
            "trace_series_coherence_info": _copy_dict(pre_out.get("trace_series_coherence_info")),
            "_guard_triggered": bool(pre_out.get("_guard_triggered", False)),
            "_guard_dpp": _f(pre_out.get("_guard_dpp"), float("nan")),
            "_ff_ml_series_bo_dplite_live": bool(pre_out.get("_ff_ml_series_bo_dplite_live", False)),
            "trace_series_bo_dplite_live": _copy_dict(pre_out.get("trace_series_bo_dplite_live")),
            "_ff_ml_state_areal_live": bool(post_out.get("_ff_ml_state_areal_live", False)),
            "_ff_ml_state_alite_live": bool(post_out.get("_ff_ml_state_alite_live", False)),
            "_ff_ml_style_b_lite_live": bool(post_out.get("_ff_ml_style_b_lite_live", False)),
            "_ff_ml_league_rank_sos_live": bool(post_out.get("_ff_ml_league_rank_sos_live", False)),
            "_ff_ml_roster_continuity_live": bool(post_out.get("_ff_ml_roster_continuity_live", False)),
            "_ff_ml_hier_calibration_live": bool(post_out.get("_ff_ml_hier_calibration_live", False)),
            "_ff_ml_meta_c_lite_live": bool(post_out.get("_ff_ml_meta_c_lite_live", False)),
            "_ff_ml_layer_uncertainty_live": bool(post_out.get("_ff_ml_layer_uncertainty_live", False)),
            "trace_state_alite_live": _copy_dict(post_out.get("trace_state_alite_live")),
            "trace_state_areal_live": _copy_dict(post_out.get("trace_state_areal_live")),
            "trace_style_b_lite_live": _copy_dict(post_out.get("trace_style_b_lite_live")),
            "trace_league_rank_sos_live": _copy_dict(post_out.get("trace_league_rank_sos_live")),
            "trace_roster_continuity_live": _copy_dict(post_out.get("trace_roster_continuity_live")),
            "trace_hier_calibration_live": _copy_dict(post_out.get("trace_hier_calibration_live")),
            "trace_meta_c_lite_live": _copy_dict(post_out.get("trace_meta_c_lite_live")),
            "trace_layer_uncertainty_live": _copy_dict(post_out.get("trace_layer_uncertainty_live")),
            "trace_marginal_trust_info": _copy_dict(post_out.get("trace_marginal_trust_info")),
            "trace_pipeline_steps": list(post_out.get("trace_pipeline_steps", []))
            if isinstance(post_out.get("trace_pipeline_steps"), list)
            else [],
            "trace_pipeline_order": list(post_out.get("trace_pipeline_order", []))
            if isinstance(post_out.get("trace_pipeline_order"), list)
            else [],
            "consistency_guard_on": True,
            "marginal_trust_guard_on": True,
            "_wf_info": {},
            "trace_shadow_ml_vnext": {},
            "trace_shadow_state_alite": _copy_dict(scenario.get("trace_shadow_state_alite")),
            "trace_shadow_state_areal": _copy_dict(scenario.get("trace_shadow_state_areal")),
        }
        return trace_ctx

    def _extract_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        probs = payload.get("probabilities") if isinstance(payload.get("probabilities"), dict) else {}
        odds = payload.get("odds") if isinstance(payload.get("odds"), dict) else {}
        pipe = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
        guards = payload.get("guards") if isinstance(payload.get("guards"), dict) else {}

        state_areal = payload.get("state_areal_live") if isinstance(payload.get("state_areal_live"), dict) else {}
        state_alite = payload.get("state_alite_live") if isinstance(payload.get("state_alite_live"), dict) else {}
        style = payload.get("style_b_lite_live") if isinstance(payload.get("style_b_lite_live"), dict) else {}
        rank = payload.get("league_rank_sos_live") if isinstance(payload.get("league_rank_sos_live"), dict) else {}
        roster = payload.get("roster_continuity_live") if isinstance(payload.get("roster_continuity_live"), dict) else {}
        hier = payload.get("hier_calibration_live") if isinstance(payload.get("hier_calibration_live"), dict) else {}
        meta = payload.get("meta_c_lite_live") if isinstance(payload.get("meta_c_lite_live"), dict) else {}
        layer_unc = payload.get("layer_uncertainty_live") if isinstance(payload.get("layer_uncertainty_live"), dict) else {}
        series = payload.get("series_bo_dplite_live") if isinstance(payload.get("series_bo_dplite_live"), dict) else {}
        marginal = payload.get("marginal_trust_guard") if isinstance(payload.get("marginal_trust_guard"), dict) else {}

        return {
            "probabilities": {
                "p_map_used": probs.get("p_map_used"),
                "p_series_used": probs.get("p_series_used"),
            },
            "odds": {
                "teamA_map": odds.get("teamA_map"),
                "teamB_map": odds.get("teamB_map"),
            },
            "pipeline_order": list(pipe.get("order", [])) if isinstance(pipe.get("order"), list) else [],
            "guards": {
                "consistency_guard_triggered": guards.get("consistency_guard_triggered"),
                "patch_guard_applied": guards.get("patch_guard_applied"),
                "state_areal_live_applied": guards.get("state_areal_live_applied"),
                "state_alite_live_applied": guards.get("state_alite_live_applied"),
                "style_b_lite_live_applied": guards.get("style_b_lite_live_applied"),
                "league_rank_sos_live_applied": guards.get("league_rank_sos_live_applied"),
                "roster_continuity_live_applied": guards.get("roster_continuity_live_applied"),
                "hier_calibration_live_applied": guards.get("hier_calibration_live_applied"),
                "meta_c_lite_live_applied": guards.get("meta_c_lite_live_applied"),
                "layer_uncertainty_live_applied": guards.get("layer_uncertainty_live_applied"),
                "series_bo_dplite_live_applied": guards.get("series_bo_dplite_live_applied"),
                "marginal_trust_guard_applied": guards.get("marginal_trust_guard_applied"),
            },
            "layers": {
                "state_areal_live": {
                    "applied": bool(state_areal.get("applied", False)),
                    "p_map_after": state_areal.get("p_map_after"),
                },
                "state_alite_live": {
                    "applied": bool(state_alite.get("applied", False)),
                    "p_map_after": state_alite.get("p_map_after"),
                },
                "style_b_lite_live": {
                    "applied": bool(style.get("applied", False)),
                    "p_after": style.get("p_after"),
                },
                "league_rank_sos_live": {
                    "applied": bool(rank.get("applied", False)),
                    "p_after": rank.get("p_after"),
                },
                "roster_continuity_live": {
                    "applied": bool(roster.get("applied", False)),
                    "p_after": roster.get("p_after"),
                },
                "hier_calibration_live": {
                    "applied": bool(hier.get("applied", False)),
                    "p_after": hier.get("p_after"),
                },
                "meta_c_lite_live": {
                    "applied": bool(meta.get("applied", False)),
                    "p_after": meta.get("p_after"),
                },
                "layer_uncertainty_live": {
                    "applied": bool(layer_unc.get("applied", False)),
                    "p_after": layer_unc.get("p_after"),
                },
                "series_bo_dplite_live": {
                    "applied": bool(series.get("applied", False)),
                    "p_after": series.get("p_after"),
                },
                "marginal_trust_guard": {
                    "applied": bool(marginal.get("applied", False)),
                    "p_out": marginal.get("p_out"),
                },
            },
        }

    def _assert_close(self, got: Any, exp: Any, path: str = ""):
        loc = path or "root"
        if isinstance(exp, dict):
            self.assertTrue(isinstance(got, dict), f"{loc}: expected dict, got {type(got).__name__}")
            for key, exp_val in exp.items():
                self.assertIn(key, got, f"{loc}: missing key '{key}'")
                self._assert_close(got.get(key), exp_val, path=f"{loc}.{key}")
            return
        if isinstance(exp, list):
            self.assertTrue(isinstance(got, list), f"{loc}: expected list, got {type(got).__name__}")
            self.assertEqual(len(got), len(exp), f"{loc}: list length mismatch")
            for idx, exp_val in enumerate(exp):
                self._assert_close(got[idx], exp_val, path=f"{loc}[{idx}]")
            return
        if isinstance(exp, bool):
            self.assertEqual(bool(got), bool(exp), f"{loc}: bool mismatch")
            return
        if exp is None:
            self.assertIsNone(got, f"{loc}: expected None, got {got!r}")
            return
        if isinstance(exp, (int, float)):
            self.assertIsNotNone(got, f"{loc}: expected numeric {exp}, got None")
            self.assertAlmostEqual(float(got), float(exp), delta=self.tol, msg=f"{loc}: numeric mismatch")
            return
        self.assertEqual(got, exp, f"{loc}: value mismatch")

    def _run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        engine_out = _copy_dict(scenario.get("engine_out"))
        pre_live_out = _copy_dict(scenario.get("pre_live_out"))

        pre_out = run_ml_pre_pipeline(
            pre_engine_kwargs={"p_map_used": engine_out.get("p_map_used", 0.5)},
            pre_live_kwargs={"lineup_mode": str(engine_out.get("lineup_mode", "off"))},
            run_pre_engine_fn=(lambda **_k: dict(engine_out)),
            run_pre_live_fn=(lambda **_k: dict(pre_live_out)),
        )

        ff_fn = self._feature_flag_fn(_copy_dict(scenario.get("feature_flags")))
        live_stub = self._build_live_layers_stub(scenario)
        marginal_stub = self._build_marginal_stub(scenario)

        league = str(scenario.get("league", "") or "")
        post_out = run_ml_post_layers(
            p_map_used=pre_out.get("p_map_used"),
            p_series_used=pre_out.get("p_series_used"),
            bo=int(scenario.get("bo", 1) or 1),
            trace_shadow_state_alite=_copy_dict(scenario.get("trace_shadow_state_alite")),
            trace_shadow_state_areal=_copy_dict(scenario.get("trace_shadow_state_areal")),
            feature_flag_enabled_fn=ff_fn,
            prob_win_series_fn=prob_win_series,
            session_state={},
            trace_pipeline_steps=list(pre_out.get("trace_pipeline_steps", []))
            if isinstance(pre_out.get("trace_pipeline_steps"), list)
            else [],
            live_kwargs={
                "playoffs_opt": str(scenario.get("playoffs_opt", "")),
                "ml_info": {"blue": {"league": league}, "red": {"league": league}},
                "teamA": str(scenario.get("teamA", "")),
                "teamB": str(scenario.get("teamB", "")),
            },
            marginal_kwargs={
                "ml_info": {"blue": {"league": league}},
            },
            lg_a_hint=league,
            run_live_layers_fn=live_stub,
            run_marginal_trust_guard_fn=marginal_stub,
            caption_fn=None,
        )

        trace_ctx = self._build_trace_ctx(scenario, pre_out, post_out)
        payload = build_ml_trace_payload(
            ctx=trace_ctx,
            odd_from_prob_fn=lambda p: 1.0 / float(p),
        )
        return payload

    def test_ml_parity_golden_scenarios(self):
        scenarios = self.fixture.get("scenarios", [])
        self.assertTrue(isinstance(scenarios, list) and len(scenarios) > 0, "No scenarios in ml_parity_golden fixture.")

        for row in scenarios:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id", "unknown"))
            expected = _copy_dict(row.get("expected"))
            with self.subTest(scenario=sid):
                payload = self._run_scenario(row)
                got_snapshot = self._extract_snapshot(payload)
                self._assert_close(got_snapshot, expected, path=sid)


if __name__ == "__main__":
    unittest.main()
