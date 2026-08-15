import unittest
from pathlib import Path

from mlcore.tests._workspace_temp import workspace_tempdir
from app_core.csv_bootstrap_runtime_core import (
    apply_bootstrap_selection,
    build_runtime_config_defaults,
    consume_auto_run_incremental,
    consume_auto_run_maintenance,
    consume_bootstrap_gate_state,
    consume_csv_prepare_refresh,
    csv_bootstrap_fingerprint,
    default_artifact_for,
    discover_season_csvs,
    list_csv_candidates,
    mark_bootstrap_gate_completed,
    reconcile_bootstrap_maintenance_status,
    request_auto_run_incremental,
    request_auto_run_maintenance,
    request_csv_runtime_refresh,
    reset_bootstrap_selection,
    resolve_active_csv_and_artifact,
    resolve_ml_paths,
    resolve_season_context,
    season_from_year,
    should_run_bootstrap_auto_maintenance,
)


class TestCSVBootstrapRuntimeCore(unittest.TestCase):
    def test_list_csv_candidates_sorted_newest_first(self):
        with workspace_tempdir("csv_bootstrap_runtime_core") as td:
            root = Path(td)
            older = root / "a.csv"
            newer = root / "b.csv"
            older.write_text("x", encoding="utf-8")
            newer.write_text("xx", encoding="utf-8")
            older.touch()
            newer.touch()
            out = list_csv_candidates(root)
            self.assertEqual([x["name"] for x in out], ["b.csv", "a.csv"])

    def test_season_discovery_and_default_artifact(self):
        with workspace_tempdir("csv_bootstrap_runtime_core") as td:
            root = Path(td)
            (root / "2025_LoL_esports_match_data_from_OraclesElixir.csv").write_text("", encoding="utf-8")
            (root / "2026_LoL_esports_match_data_from_OraclesElixir.csv").write_text("", encoding="utf-8")
            (root / "ml_artifact.json").write_text("{}", encoding="utf-8")
            seasons = discover_season_csvs(root)
            self.assertEqual([x["season"] for x in seasons], [season_from_year(2025), season_from_year(2026)])
            self.assertTrue(default_artifact_for(root, "S16", 2026).endswith("ml_artifact.json"))

    def test_csv_bootstrap_fingerprint_uses_file_sig(self):
        def _sig(path: str):
            return (path, 123, 456)

        fp = csv_bootstrap_fingerprint("daily.csv", ["h1.csv", "", "h2.csv"], "ml_artifact.json", file_sig_fn=_sig)
        self.assertEqual(fp["csv_sig"], ["daily.csv", 123, 456])
        self.assertEqual(fp["artifact_sig"], ["ml_artifact.json", 123, 456])
        self.assertEqual(fp["hist_csvs"], ["h1.csv", "h2.csv"])

    def test_apply_reset_and_resolve_bootstrap_selection(self):
        ss = {}
        apply_bootstrap_selection(ss, "daily.csv", ["2014.csv", "", "2015.csv"], clear_selector_flag=True)
        self.assertEqual(ss["boot_daily_csv"], "daily.csv")
        self.assertEqual(ss["boot_hist_csvs"], ["2014.csv", "2015.csv"])
        self.assertEqual(ss["cfg_csv_override"], "daily.csv")
        self.assertFalse(ss["cfg_auto_paths"])
        self.assertTrue(ss["boot_done"])
        self.assertFalse(ss["boot_show_selector"])

        daily, hist, art = resolve_ml_paths(ss, Path("C:/tmp"))
        self.assertEqual(daily, "daily.csv")
        self.assertEqual(hist, ["2014.csv", "2015.csv"])
        self.assertEqual(art, "C:\\tmp\\ml_artifact.json")

        reset_bootstrap_selection(ss, show_selector=True)
        self.assertFalse(ss["cfg_auto_paths"])
        self.assertTrue(ss["boot_show_selector"])
        self.assertEqual(ss["_csv_bootstrap_done_fp"], "")

    def test_runtime_refresh_and_auto_run_helpers(self):
        ss = {}
        flags = request_csv_runtime_refresh(ss, prepare=True, bootstrap=True, hold_screen=True, force_maint=True)
        self.assertTrue(flags["prepare"])
        self.assertTrue(consume_csv_prepare_refresh(ss))
        gate_state = consume_bootstrap_gate_state(ss)
        self.assertTrue(gate_state["force_once"])
        self.assertTrue(gate_state["force_maint"])
        self.assertFalse(gate_state["force_full"])  # nao pedido aqui
        self.assertTrue(gate_state["pending_open"])

        # force_full (botao "Manutencao completa"): one-shot, consumido junto.
        ss2: dict = {}
        flags2 = request_csv_runtime_refresh(ss2, bootstrap=True, force_maint=True, force_full=True)
        self.assertTrue(flags2["force_full"])
        gate2 = consume_bootstrap_gate_state(ss2)
        self.assertTrue(gate2["force_maint"])
        self.assertTrue(gate2["force_full"])
        # consumido (pop) -> some na segunda leitura
        self.assertFalse(consume_bootstrap_gate_state(ss2)["force_full"])
        self.assertEqual(mark_bootstrap_gate_completed(ss, "fp-1"), "fp-1")
        self.assertEqual(ss["_csv_bootstrap_done_fp"], "fp-1")
        self.assertFalse(ss["_csv_bootstrap_hold_screen"])

        self.assertTrue(request_auto_run_incremental(ss))
        self.assertTrue(consume_auto_run_incremental(ss))
        self.assertFalse(consume_auto_run_incremental(ss))

        auto_maint = request_auto_run_maintenance(ss, {"severity": "warning"})
        self.assertTrue(auto_maint["enabled"])
        consumed = consume_auto_run_maintenance(ss)
        self.assertTrue(consumed["enabled"])
        self.assertEqual(consumed["status"], {"severity": "warning"})

    def test_reconcile_status_and_policy(self):
        status, note = reconcile_bootstrap_maintenance_status(
            {
                "need_retrain_params": True,
                "alerts": ["params stale"],
                "severity": "weird",
                "has_issues": False,
            },
            stage="before",
        )
        self.assertTrue(status["has_issues"])
        self.assertEqual(status["severity"], "warning")
        self.assertTrue(status["flags_any_need"])
        self.assertIn("[before]", note)
        self.assertTrue(should_run_bootstrap_auto_maintenance(status))
        self.assertTrue(should_run_bootstrap_auto_maintenance({}, force_maint=True))
        self.assertFalse(should_run_bootstrap_auto_maintenance({"severity": "warning"}))

    def test_runtime_config_defaults_and_active_paths(self):
        seasons = [{"season": "S15", "year": 2025, "csv_path": "C:/data/2025.csv"}]
        ss = {"cfg_auto_paths": True}
        defaults = build_runtime_config_defaults(ss, app_root=Path("C:/repo"), seasons=seasons)
        self.assertTrue(defaults["auto_paths"])
        self.assertTrue(str(defaults["csv_override"]).endswith("2025_LoL_esports_match_data_from_OraclesElixir.csv"))
        self.assertTrue(str(defaults["artifact_override"]).endswith("ml_artifact.json"))

        csv_path, artifact_path = resolve_active_csv_and_artifact(
            {"cfg_auto_paths": True},
            app_root=Path("C:/repo"),
            season_label="S15",
            season_year=2025,
            season_csv_path="C:/data/2025.csv",
            seasons_available=True,
        )
        self.assertEqual(csv_path, "C:/data/2025.csv")
        self.assertTrue(artifact_path.endswith("ml_artifact.json"))

        csv_path2, _ = resolve_active_csv_and_artifact(
            {"cfg_auto_paths": False, "cfg_csv_override": "C:/manual.csv"},
            app_root=Path("C:/repo"),
            season_label="S15",
            season_year=2025,
            season_csv_path="C:/data/2025.csv",
            seasons_available=True,
        )
        self.assertEqual(csv_path2, "C:/manual.csv")

        csv_path3, _ = resolve_active_csv_and_artifact(
            {
                "cfg_auto_paths": False,
                "cfg_csv_override": "C:/repo/CSV_DIARIO/2026_LoL_esports_match_data_from_OraclesElixir.csv",
            },
            app_root=Path("C:/repo"),
            season_label="S15",
            season_year=2025,
            season_csv_path="C:/data/2025.csv",
            seasons_available=True,
        )
        self.assertEqual(csv_path3, "C:/data/2025.csv")

    def test_dead_artifact_override_falls_back_to_repo_default(self):
        """Regressão 2026-07-15 (nuvem): cfg_artifact_override de OUTRA máquina
        (caminho absoluto morto, ex.: C:\\... no Linux) era usado cru — o motor
        v2 recebia artifact inexistente e o ML sumia mudo. Override morto deve
        cair no padrão do repo."""
        with workspace_tempdir("bootstrap_runtime_core") as td:
            root = Path(td)
            (root / "ml_artifact.json").write_text("{}", encoding="utf-8")
            _, artifact_path = resolve_active_csv_and_artifact(
                {
                    "cfg_auto_paths": False,
                    "cfg_artifact_override": r"C:\maquina\de\outra\pessoa\ml_artifact.json"
                    if not Path(r"C:\maquina").exists()
                    else str(root / "nao_existe" / "ml_artifact.json"),
                },
                app_root=root,
                season_label="S16",
                season_year=2026,
                season_csv_path="",
                seasons_available=False,
            )
            self.assertEqual(Path(artifact_path), root / "ml_artifact.json")

    def test_resolve_season_context(self):
        seasons = [
            {"season": "S14", "year": 2024, "csv_path": "C:/data/2024.csv"},
            {"season": "S15", "year": 2025, "csv_path": "C:/data/2025.csv"},
        ]
        ctx = resolve_season_context(seasons, selected_label="S14")
        self.assertTrue(ctx["has_seasons"])
        self.assertEqual(ctx["selected_label"], "S14")
        self.assertEqual(ctx["season_year"], 2024)
        self.assertEqual(ctx["selected_idx"], 0)

        ctx2 = resolve_season_context(seasons, selected_label="__invalid__")
        self.assertEqual(ctx2["selected_label"], "S15")
        self.assertEqual(ctx2["default_idx"], 1)

        ctx3 = resolve_season_context([], selected_label=None)
        self.assertFalse(ctx3["has_seasons"])
        self.assertEqual(ctx3["selected_label"], "S15")
        self.assertEqual(ctx3["season_year"], 2025)


if __name__ == "__main__":
    unittest.main()
