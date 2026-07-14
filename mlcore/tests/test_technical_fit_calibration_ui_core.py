import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app_core.technical_fit_calibration_ui_core import (
    apply_best_form_settings_to_session,
    build_fit_summary_snapshot,
    resolve_refit_players_artifact_path,
)


class TechnicalFitCalibrationUICoreTests(unittest.TestCase):
    def test_build_fit_summary_snapshot_uses_defaults(self) -> None:
        snap = build_fit_summary_snapshot(
            fit_live={"best": {"logloss": 0.5}, "generated_at": "2026-03-11T10:00:00"},
            fit_scale_state={"team_scale_used": 1.2, "players_scale_used": 0.9},
            scale_league_used="LCK",
            form_league_used="LCK",
        )
        self.assertEqual(snap["team_scale_used"], 1.2)
        self.assertEqual(snap["players_scale_used"], 0.9)
        self.assertEqual(snap["scale_league_used"], "LCK")
        self.assertEqual(snap["generated_at"], "2026-03-11T10:00:00")

    def test_apply_best_form_settings_to_session_updates_state(self) -> None:
        state = {}
        apply_best_form_settings_to_session(
            state,
            {
                "ml_form_blend_enabled": False,
                "ml_form_blend_max_weight": 0.5,
                "ml_form_blend_knee_games": 8,
                "ml_form_blend_beta": 1.5,
                "ml_form_blend_signal_power": 0.8,
                "ml_form_blend_parity_strength": 0.2,
                "ml_form_use_fitted": False,
            },
        )
        self.assertFalse(state["ml_form_blend_enabled"])
        self.assertEqual(state["ml_form_blend_max_weight"], 0.5)
        self.assertFalse(state["ml_form_use_fitted"])

    def test_resolve_refit_players_artifact_path_prefers_artifact_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app_root = Path(tmp)
            artifact_path = app_root / "ml_artifact.json"
            players_path = app_root / "players_alt.json"
            artifact_path.write_text("{}", encoding="utf-8")
            players_path.write_text("{}", encoding="utf-8")
            out = resolve_refit_players_artifact_path(
                app_root=app_root,
                artifact_path=str(artifact_path),
                load_artifact_fn=lambda _sig: SimpleNamespace(meta={"players_layer": {"players_artifact": str(players_path)}}),
                artifact_sig_fn=lambda path: path,
            )
        self.assertEqual(out, str(players_path))

    def test_resolve_refit_players_artifact_path_falls_back_to_artifact_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            app_root = base / "app"
            app_root.mkdir(parents=True, exist_ok=True)
            artifact_dir = base / "custom_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "ml_artifact.json"
            sibling_players = artifact_dir / "players_artifact.json"
            artifact_path.write_text("{}", encoding="utf-8")
            sibling_players.write_text("{}", encoding="utf-8")
            out = resolve_refit_players_artifact_path(
                app_root=app_root,
                artifact_path=str(artifact_path),
                load_artifact_fn=lambda _sig: SimpleNamespace(meta={}),
                artifact_sig_fn=lambda path: path,
            )
        self.assertEqual(out, str(sibling_players))


if __name__ == "__main__":
    unittest.main()
