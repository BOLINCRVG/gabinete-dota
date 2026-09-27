import contextlib
import importlib.util
import json
import shutil
import unittest
import uuid
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol



ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = ROOT / "evidence" / "_tmp_tests"


@contextlib.contextmanager
def _tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    td = TEST_TMP_ROOT / f"run_stabilization_gate_{uuid.uuid4().hex}"
    td.mkdir(parents=True, exist_ok=True)
    try:
        yield td
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _load_module():
    mod_path = ROOT / "tools" / "run_stabilization_gate.py"
    spec = importlib.util.spec_from_file_location("run_stabilization_gate", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load tools/run_stabilization_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RunStabilizationGateTests(unittest.TestCase):
    @so_no_lol("testa uma ferramenta de tools/ que so existe no LoL")
    def test_quick_profile_runs_code_error_scan_check(self) -> None:
        mod = _load_module()
        seen_cmds: list[list[str]] = []
        original_run_cmd = mod._run_cmd
        try:
            def _fake_run_cmd(cmd, *, timeout, cwd=None):
                seen_cmds.append([str(x) for x in cmd])
                return {
                    "ok": True,
                    "returncode": 0,
                    "output": "",
                    "cmd": [str(x) for x in cmd],
                }

            mod._run_cmd = _fake_run_cmd
            with _tempdir() as td:
                out_path = td / "report.json"
                rc = mod.main(["--profile", "quick", "--timeout", "120", "--out", str(out_path)])
                self.assertEqual(rc, 0)
                report = json.loads(out_path.read_text(encoding="utf-8"))
        finally:
            mod._run_cmd = original_run_cmd

        checks = list(report.get("checks", []))
        names = [str(row.get("check", "")) for row in checks]
        self.assertIn("code_error_scan:quick", names)

        scan_row = next(row for row in checks if str(row.get("check", "")) == "code_error_scan:quick")
        scan_cmd = [str(x) for x in list(scan_row.get("cmd", []))]
        self.assertIn("tools/find_code_errors.py", scan_cmd)
        self.assertIn("--profile", scan_cmd)
        self.assertIn("quick", scan_cmd)
        self.assertEqual(scan_cmd.count("--path"), len(mod.SCANNER_TARGETS))
        for target in mod.SCANNER_TARGETS:
            self.assertIn(str(target), scan_cmd)
        self.assertTrue(any("tools/find_code_errors.py" in " ".join(cmd) for cmd in seen_cmds))


if __name__ == "__main__":
    unittest.main()
