import tempfile
import unittest
from pathlib import Path

from app_core.csv_bootstrap_selector_ui_core import resolve_bootstrap_auto_selection


class CsvBootstrapSelectorUICoreTests(unittest.TestCase):
    def test_resolve_bootstrap_auto_selection_prefers_daily_directory(self) -> None:
        plan = resolve_bootstrap_auto_selection(
            session_state={},
            root=Path("C:/repo"),
            list_csv_candidates_fn=lambda folder: (
                [{"name": "daily.csv", "path": "C:/repo/CSV_DIARIO/daily.csv"}]
                if str(folder).endswith("CSV_DIARIO")
                else [{"name": "hist.csv", "path": "C:/repo/CSV_HISTORICO/hist.csv"}]
            ),
        )
        self.assertTrue(plan["auto_apply"])
        self.assertEqual(plan["daily_choice"], "C:/repo/CSV_DIARIO/daily.csv")
        self.assertEqual(plan["hist_choice"], ["C:/repo/CSV_HISTORICO/hist.csv"])

    def test_resolve_bootstrap_auto_selection_uses_cfg_override_when_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_csv = Path(tmp) / "manual.csv"
            cfg_csv.write_text("x", encoding="utf-8")
            plan = resolve_bootstrap_auto_selection(
                session_state={"cfg_csv_override": str(cfg_csv)},
                root=Path(tmp),
                list_csv_candidates_fn=lambda _folder: [],
            )
        self.assertTrue(plan["auto_apply"])
        self.assertEqual(plan["daily_choice"], str(cfg_csv))

    def test_resolve_bootstrap_auto_selection_respects_force_selector(self) -> None:
        plan = resolve_bootstrap_auto_selection(
            session_state={"boot_show_selector": True},
            root=Path("C:/repo"),
            list_csv_candidates_fn=lambda _folder: [{"name": "daily.csv", "path": "C:/repo/daily.csv"}],
        )
        self.assertFalse(plan["auto_apply"])


if __name__ == "__main__":
    unittest.main()
