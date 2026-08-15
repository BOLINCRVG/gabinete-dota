import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_gate_module():
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "tools" / "run_phase2b_areal_tuning_gate.py"
    spec = importlib.util.spec_from_file_location("run_phase2b_areal_tuning_gate", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed_to_load_phase2b_gate_module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase2BArealGate(unittest.TestCase):
    def test_full_speed_report_keeps_qwl_grid_dimensions(self) -> None:
        mod = _load_gate_module()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "CSV_DIARIO").mkdir(parents=True, exist_ok=True)
            csv_path = root / "CSV_DIARIO" / "sample.csv"
            csv_path.write_text("gameid\n1\n", encoding="utf-8")

            def _fake_artifact(*_args, **_kwargs):
                return {
                    "meta": {
                        "eval_metrics": {
                            "n": 500,
                            "logloss": 0.50,
                            "brier": 0.25,
                            "by_league": {"LCK": {"n": 500, "logloss": 0.50, "brier": 0.25}},
                            "by_patch": {"16.1": {"n": 250, "logloss": 0.50, "brier": 0.25}},
                        }
                    }
                }

            old_root = mod.ROOT
            mod.ROOT = root
            mod.build_team_state_artifact = _fake_artifact
            try:
                with patch("sys.argv", ["run_phase2b_areal_tuning_gate.py", "--speed", "full", "--out", "out.json"]):
                    rc = mod.main()
                self.assertEqual(rc, 0)
                report = json.loads((root / "out.json").read_text(encoding="utf-8"))
            finally:
                mod.ROOT = old_root

        self.assertIn("grid", report)
        self.assertEqual(report["grid"]["qwl_throw_weight"], [0.9, 1.0, 1.2])
        self.assertEqual(report["grid"]["qwl_comeback_weight"], [0.55, 0.65, 0.85])
        self.assertEqual(report["grid"]["qwl_phase_dynamic_strength"], [0.2, 0.35, 0.5])
        self.assertEqual(report["grid"]["qwl_opponent_context_mix"], [0.15, 0.3, 0.45])


if __name__ == "__main__":
    unittest.main()
