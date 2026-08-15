import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_tool_module(filename: str, module_name: str):
    root = Path(__file__).resolve().parents[2]
    mod_path = root / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed_to_load_module:{filename}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPhase5HierInterSameGates(unittest.TestCase):
    def setUp(self):
        self.mod_c = _load_tool_module("run_phase5c_hier_interleague_gate.py", "run_phase5c_hier_interleague_gate")
        self.mod_d = _load_tool_module("run_phase5d_hier_sameleague_gate.py", "run_phase5d_hier_sameleague_gate")

    def _run_main(self, mod, compare_payload: dict, baseline_payload: dict):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            cmp_path = tdp / "compare.json"
            base_path = tdp / "baseline.json"
            out_path = tdp / "report.json"
            cmp_path.write_text(json.dumps(compare_payload), encoding="utf-8")
            base_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

            argv_prev = list(sys.argv)
            try:
                sys.argv = [
                    str(getattr(mod, "__file__", "tool.py")),
                    "--compare",
                    str(cmp_path),
                    "--baseline",
                    str(base_path),
                    "--out",
                    str(out_path),
                ]
                rc = int(mod.main())
            finally:
                sys.argv = argv_prev

            rep = json.loads(out_path.read_text(encoding="utf-8"))
            return rc, rep

    def test_phase5c_promote_on_noninferior_with_gain(self):
        compare = {
            "rows_total": 600,
            "global": {
                "n": 600,
                "post": {"n": 600},
                "gain_logloss": 0.0004,
                "gain_brier": 0.0002,
                "gain_ece": 0.0006,
            },
            "by_segment_class": {
                "inter_major_major": {"n": 120, "gain_logloss": 0.0004, "gain_brier": 0.0002, "gain_ece": 0.0006},
                "inter_mixed": {"n": 200, "gain_logloss": 0.0003, "gain_brier": 0.0001, "gain_ece": 0.0005},
                "inter_minor_minor": {"n": 280, "gain_logloss": 0.0002, "gain_brier": 0.0001, "gain_ece": 0.0004},
            },
        }
        baseline = {"thresholds": {}}
        rc, rep = self._run_main(self.mod_c, compare, baseline)
        self.assertEqual(rc, 0)
        self.assertEqual(str(rep.get("decision")), "promote")

    def test_phase5c_reject_on_global_noninferior_fail(self):
        compare = {
            "rows_total": 700,
            "global": {
                "n": 700,
                "post": {"n": 700},
                "gain_logloss": -0.02,
                "gain_brier": 0.0,
                "gain_ece": 0.0,
            },
            "by_segment_class": {
                "inter_major_major": {"n": 120, "gain_logloss": 0.0, "gain_brier": 0.0, "gain_ece": 0.0},
                "inter_mixed": {"n": 200, "gain_logloss": 0.0, "gain_brier": 0.0, "gain_ece": 0.0},
                "inter_minor_minor": {"n": 280, "gain_logloss": 0.0, "gain_brier": 0.0, "gain_ece": 0.0},
            },
        }
        baseline = {"thresholds": {}}
        rc, rep = self._run_main(self.mod_c, compare, baseline)
        self.assertEqual(rc, 2)
        self.assertEqual(str(rep.get("decision")), "reject")

    def test_phase5d_promote_when_global_is_neutral(self):
        compare = {
            "rows_total": 2200,
            "global": {
                "n": 2200,
                "post": {"n": 2200},
                "gain_logloss": -1e-6,
                "gain_brier": -1e-6,
                "gain_ece": -1e-5,
            },
            "by_segment_class": {
                "same_major": {"n": 400, "gain_logloss": 0.0, "gain_brier": 0.0, "gain_ece": 0.0},
                "same_minor": {"n": 1800, "gain_logloss": 0.0, "gain_brier": 0.0, "gain_ece": 0.0},
            },
        }
        baseline = {"thresholds": {}}
        rc, rep = self._run_main(self.mod_d, compare, baseline)
        self.assertEqual(rc, 0)
        self.assertEqual(str(rep.get("decision")), "promote")
        checks = rep.get("checks") if isinstance(rep.get("checks"), list) else []
        gate = next((c for c in checks if str(c.get("check")) == "global_has_practical_gain_or_neutral"), {})
        self.assertTrue(bool(gate.get("ok")))

    def test_phase5d_shadow_when_segment_noninferior_fails(self):
        compare = {
            "rows_total": 2600,
            "global": {
                "n": 2600,
                "post": {"n": 2600},
                "gain_logloss": 0.0004,
                "gain_brier": 0.0002,
                "gain_ece": 0.0006,
            },
            "by_segment_class": {
                "same_major": {"n": 500, "gain_logloss": 0.0002, "gain_brier": 0.0001, "gain_ece": 0.0002},
                "same_minor": {"n": 2100, "gain_logloss": -0.02, "gain_brier": 0.0, "gain_ece": 0.0},
            },
        }
        baseline = {"thresholds": {}}
        rc, rep = self._run_main(self.mod_d, compare, baseline)
        self.assertEqual(rc, 1)
        self.assertEqual(str(rep.get("decision")), "shadow_only")


if __name__ == "__main__":
    unittest.main()
