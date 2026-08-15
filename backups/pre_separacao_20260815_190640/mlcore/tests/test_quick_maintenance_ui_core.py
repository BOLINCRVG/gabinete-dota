import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.quick_maintenance_ui_core import (
    build_quick_walkforward_kwargs,
    resolve_quick_maintenance_players_artifact_path,
    resolve_quick_maintenance_players_output_path,
)


class QuickMaintenanceUICoreTests(unittest.TestCase):
    def test_resolve_players_artifact_path_prefers_injected_resolver(self) -> None:
        with workspace_tempdir("quick_maintenance_ui_core") as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "custom"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "ml_artifact.json"
            players_path = artifact_dir / "players_custom.json"
            artifact_path.write_text("{}", encoding="utf-8")
            players_path.write_text("{}", encoding="utf-8")

            out = resolve_quick_maintenance_players_artifact_path(
                root=root,
                artifact_path=str(artifact_path),
                resolve_players_artifact_path_fn=lambda _artifact_path, _root: str(players_path),
            )

        self.assertEqual(out, players_path)

    def test_resolve_players_output_path_falls_back_to_artifact_sibling(self) -> None:
        with workspace_tempdir("quick_maintenance_ui_core") as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "custom"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "ml_artifact.json"
            artifact_path.write_text("{}", encoding="utf-8")

            out = resolve_quick_maintenance_players_output_path(
                root=root,
                artifact_path=str(artifact_path),
                resolve_players_artifact_path_fn=lambda _artifact_path, _root: str(root / "players_artifact.json"),
            )

        self.assertEqual(out, artifact_dir / "players_artifact.json")

    def test_build_quick_walkforward_kwargs_includes_active_artifacts_and_scales(self) -> None:
        kwargs = build_quick_walkforward_kwargs(
            csv_list=["a.csv", "", "b.csv"],
            wf_cfg={"min_train_games": 1234, "k": 21.5, "scale": 410.0},
            artifact_path="artifact.json",
            players_artifact_path="players.json",
            params_fitted_path="params.json",
            fit_scale_state={"team_scale": 1.25, "players_scale": 0.85},
        )

        self.assertEqual(kwargs["csv_paths"], ("a.csv", "b.csv"))
        self.assertEqual(kwargs["artifact_path"], "artifact.json")
        self.assertEqual(kwargs["players_artifact_path"], "players.json")
        self.assertEqual(kwargs["params_fitted_path"], "params.json")
        self.assertAlmostEqual(kwargs["team_scale"], 1.25, places=9)
        self.assertAlmostEqual(kwargs["players_scale"], 0.85, places=9)


if __name__ == "__main__":
    unittest.main()
