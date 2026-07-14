from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from app_core.ml_session_runtime_core import (
    apply_fit_runtime_state_to_session,
    apply_players_context_runtime_override,
    build_combined_trace_payload,
    clear_feature_trace_error,
    clear_ml_fused_post_cal_info,
    clear_ml_microseg_info,
    clear_wf_ml_corr_info,
    consume_pending_apply_param_values,
    get_achievement_path,
    get_analysis_gate_state,
    get_feature_trace_state,
    get_fit_runtime_snapshot,
    get_fit_scale_state,
    get_ml_engine_runtime_settings,
    get_ml_fused_post_cal_info,
    get_ml_microseg_info,
    get_ml_scope,
    get_mlcore_cache_state,
    get_ml_trace_snapshot,
    get_pending_apply_param_values,
    get_pre_delivery_report,
    get_wf_ml_corr_info,
    persist_combined_trace_file,
    set_achievement_path,
    set_analysis_signature,
    set_consistency_analysis_signature,
    set_feature_trace_error,
    set_feature_trace_events,
    set_fit_base_scales,
    set_fit_runtime_labels,
    set_fit_used_scales,
    set_latest_ml_trace,
    set_ml_fused_post_cal_info,
    set_ml_microseg_info,
    set_ml_scope,
    set_mlcore_cache_state,
    set_pending_apply_param_values,
    set_pre_delivery_report,
    set_wf_ml_corr_info,
)


class MlSessionRuntimeCoreTests(unittest.TestCase):
    def test_build_combined_trace_payload_includes_meta(self) -> None:
        session_state = {
            "_ml_trace_last": {"engine": "mlcore_v2"},
            "_ml_trace_markets": [{"mercado": "Kills"}],
            "_ml_trace_players": [{"player": "A"}],
            "_feature_trace_events": [{"feature": "x"}],
            "season_label": "2026",
            "teamA": "A",
            "teamB": "B",
            "bo": 3,
            "year_opt": "2026",
            "split_opt": "S1",
            "playoffs_opt": "Regular",
            "map_mode": "avg",
            "league_mode": "auto",
            "ml_engine": "mlcore_v2",
            "ml_microseg_enabled": True,
            "ml_microseg_strength": 0.6,
            "wf_half_life_days": 120.0,
            "series_gamma_on": True,
        }

        payload = build_combined_trace_payload(
            session_state,
            build_combined_trace_fn=lambda **kwargs: dict(kwargs),
            app_version="app-test",
            generated_at="2026-03-09T21:00:00",
        )

        self.assertEqual(payload["meta"]["app_version"], "app-test")
        self.assertEqual(payload["meta"]["teams"]["bo"], 3)
        self.assertEqual(payload["meta"]["filters"]["year"], "2026")
        self.assertTrue(payload["meta"]["calibration"]["microseg_enabled"])

    def test_persist_combined_trace_file_writes_session_and_disk(self) -> None:
        session_state = {
            "_ml_trace_last": {},
            "_ml_trace_markets": [],
            "_ml_trace_players": [],
            "_feature_trace_events": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = persist_combined_trace_file(
                session_state,
                app_root=tmpdir,
                build_combined_trace_fn=lambda **kwargs: {"ok": True},
                app_version="app-test",
            )
            self.assertTrue((Path(tmpdir) / "ml_trace_last.json").exists())
            self.assertEqual(session_state["_ml_trace_combined"]["ok"], True)
            written = json.loads((Path(tmpdir) / "ml_trace_last.json").read_text(encoding="utf-8"))
            self.assertEqual(written["ok"], True)
            self.assertEqual(payload["ok"], True)

    def test_apply_fit_runtime_state_to_session_uses_defaults_on_failure(self) -> None:
        session_state = {}
        out = apply_fit_runtime_state_to_session(
            session_state,
            {},
            build_fit_runtime_state_fn=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            sanitize_marginal_band_coeffs_fn=lambda raw: {},
        )
        self.assertEqual(out, {})
        self.assertEqual(session_state["_fit_team_scale"], 1.0)
        self.assertEqual(session_state["_fit_series_gamma_cfg"], {"global": {}, "by_league": {}, "by_macro": {}})

    def test_fit_runtime_snapshot_and_trace_snapshot_helpers(self) -> None:
        session_state = {
            "_fit_team_scale": 1.15,
            "_fit_players_context_cfg": {"global": {"enabled": True}},
            "_ml_trace_last": {"p_map_used": 0.61},
            "_ml_trace_markets": [{"mercado": "ML"}],
            "_ml_trace_players": [{"player": "mid"}],
            "_feature_trace_events": [{"feature": "state"}],
            "_p_ml_scope": 0.61,
        }
        fit_snapshot = get_fit_runtime_snapshot(session_state)
        self.assertEqual(fit_snapshot["_fit_team_scale"], 1.15)
        self.assertEqual(fit_snapshot["_fit_players_context_cfg"], {"global": {"enabled": True}})

        trace_snapshot = get_ml_trace_snapshot(session_state)
        self.assertEqual(trace_snapshot["_ml_trace_last"], {"p_map_used": 0.61})
        self.assertEqual(trace_snapshot["_ml_trace_markets"], [{"mercado": "ML"}])
        self.assertEqual(trace_snapshot["_p_ml_scope"], 0.61)

        latest = set_latest_ml_trace(session_state, {"p_map_used": 0.64})
        self.assertEqual(latest, {"p_map_used": 0.64})
        self.assertEqual(session_state["_ml_trace_last"], {"p_map_used": 0.64})

    def test_feature_trace_state_helpers(self) -> None:
        session_state = {}
        rows = set_feature_trace_events(
            session_state,
            [{"feature": "shadow", "event": "on"}],
            ok_at="2026-03-10T10:00:00",
        )
        self.assertEqual(rows, [{"feature": "shadow", "event": "on"}])
        self.assertEqual(session_state["_feature_trace_events"], rows)
        self.assertEqual(session_state["_feature_trace_last_ok_at"], "2026-03-10T10:00:00")
        self.assertEqual(session_state["_feature_trace_last_error"], "")

        msg = set_feature_trace_error(
            session_state,
            "boom",
            error_at="2026-03-10T10:01:00",
        )
        self.assertEqual(msg, "boom")
        self.assertEqual(session_state["_feature_trace_last_error"], "boom")
        self.assertEqual(session_state["_feature_trace_last_error_at"], "2026-03-10T10:01:00")

        state = get_feature_trace_state(session_state)
        self.assertEqual(state["events"], rows)
        self.assertEqual(state["last_error"], "boom")
        self.assertEqual(state["last_error_at"], "2026-03-10T10:01:00")

        clear_feature_trace_error(session_state)
        self.assertEqual(session_state["_feature_trace_last_error"], "")

    def test_pre_delivery_and_analysis_gate_helpers(self) -> None:
        session_state = {}
        report = set_pre_delivery_report(session_state, {"ok": True, "total_count": 4})
        self.assertEqual(report["ok"], True)
        self.assertEqual(get_pre_delivery_report(session_state)["total_count"], 4)

        self.assertEqual(set_analysis_signature(session_state, 123), 123)
        self.assertEqual(set_consistency_analysis_signature(session_state, 456), 456)
        gate_state = get_analysis_gate_state(session_state)
        self.assertEqual(gate_state["analysis_sig_last"], 123)
        self.assertEqual(gate_state["consistency_analysis_sig_last"], 456)

    def test_runtime_scalar_helpers(self) -> None:
        session_state = {}
        self.assertTrue(math.isnan(get_ml_scope(session_state)))
        self.assertAlmostEqual(set_ml_scope(session_state, 0.61), 0.61)
        self.assertAlmostEqual(get_ml_scope(session_state), 0.61)

        self.assertEqual(set_achievement_path(session_state, " C:/tmp/achievements.json "), "C:/tmp/achievements.json")
        self.assertEqual(get_achievement_path(session_state), "C:/tmp/achievements.json")

        vals = set_pending_apply_param_values(session_state, {"a": 1, "b": 2})
        self.assertEqual(vals, {"a": 1, "b": 2})
        self.assertEqual(get_pending_apply_param_values(session_state), {"a": 1, "b": 2})
        self.assertEqual(consume_pending_apply_param_values(session_state), {"a": 1, "b": 2})
        self.assertEqual(get_pending_apply_param_values(session_state), {})

        set_fit_base_scales(session_state, team_scale=1.2, players_scale=0.9)
        set_fit_used_scales(session_state, team_scale=1.1, players_scale=0.8)
        scale_state = get_fit_scale_state(session_state)
        self.assertEqual(scale_state["team_scale"], 1.2)
        self.assertEqual(scale_state["players_scale"], 0.9)
        self.assertEqual(scale_state["team_scale_used"], 1.1)
        self.assertEqual(scale_state["players_scale_used"], 0.8)

    def test_ml_runtime_cache_and_info_helpers(self) -> None:
        session_state = {"ml_as_of_date": "2026-03-10"}
        cfg = get_ml_engine_runtime_settings(session_state)
        self.assertEqual(cfg["ml_engine"], "mlcore_v2")
        self.assertEqual(cfg["ml_as_of_date"], "2026-03-10")
        self.assertEqual(cfg["ml_engine_alert_pp"], 12.0)

        payload = set_mlcore_cache_state(session_state, ("k", 1), {"p_cal": 0.61})
        self.assertEqual(payload["p_cal"], 0.61)
        cache_state = get_mlcore_cache_state(session_state)
        self.assertEqual(cache_state["key"], ("k", 1))
        self.assertEqual(cache_state["value"]["p_cal"], 0.61)

        self.assertEqual(set_ml_microseg_info(session_state, {"source": "fitted"})["source"], "fitted")
        self.assertEqual(get_ml_microseg_info(session_state)["source"], "fitted")
        clear_ml_microseg_info(session_state)
        self.assertEqual(get_ml_microseg_info(session_state), {})

        self.assertEqual(set_ml_fused_post_cal_info(session_state, {"alpha": 1})["alpha"], 1)
        self.assertEqual(get_ml_fused_post_cal_info(session_state)["alpha"], 1)
        clear_ml_fused_post_cal_info(session_state)
        self.assertEqual(get_ml_fused_post_cal_info(session_state), {})

        self.assertEqual(set_wf_ml_corr_info(session_state, {"strength": 0.6})["strength"], 0.6)
        self.assertEqual(get_wf_ml_corr_info(session_state)["strength"], 0.6)
        clear_wf_ml_corr_info(session_state)
        self.assertEqual(get_wf_ml_corr_info(session_state), {})

    def test_fit_labels_and_players_context_override_helpers(self) -> None:
        session_state = {}
        labels = set_fit_runtime_labels(
            session_state,
            scale_league_used="LCK",
            players_context_source_used="global",
        )
        self.assertEqual(labels["scale_league_used"], "LCK")
        self.assertEqual(session_state["_fit_scale_league_used"], "LCK")
        self.assertEqual(session_state["_fit_players_context_source_used"], "global")

        apply_players_context_runtime_override(
            session_state,
            {"ml_players_context_enabled": True, "ml_players_context_lookback_games": 30},
            "league",
            use_fitted=True,
        )
        self.assertTrue(session_state["ml_players_context_enabled"])
        self.assertEqual(session_state["ml_players_context_lookback_games"], 30)

        apply_players_context_runtime_override(session_state, {}, "gate:reject", use_fitted=True)
        self.assertFalse(session_state["ml_players_context_enabled"])


if __name__ == "__main__":
    unittest.main()
