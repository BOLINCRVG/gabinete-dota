import unittest
from unittest.mock import patch

from app_core.governance_runtime_core import (
    append_governance_log,
    combine_governance_decisions,
    normalize_governance_decision,
    run_governance_commands,
    run_governance_operation,
)


class TestNormalizeGovernanceDecision(unittest.TestCase):

    def test_promote_uppercase_normalized(self):
        self.assertEqual(normalize_governance_decision("PROMOTE"), "promote")

    def test_shadow_only_kept(self):
        self.assertEqual(normalize_governance_decision("shadow_only"), "shadow_only")

    def test_reject_kept(self):
        self.assertEqual(normalize_governance_decision("reject"), "reject")

    def test_unknown_value_returns_unknown(self):
        self.assertEqual(normalize_governance_decision("other"), "unknown")

    def test_none_returns_unknown(self):
        self.assertEqual(normalize_governance_decision(None), "unknown")

    def test_empty_string_returns_unknown(self):
        self.assertEqual(normalize_governance_decision(""), "unknown")

    def test_returns_string(self):
        self.assertIsInstance(normalize_governance_decision("promote"), str)


class TestCombineGovernanceDecisions(unittest.TestCase):

    def test_reject_wins_over_promote(self):
        got = combine_governance_decisions({"stop_rules": "promote", "anchors": "shadow_only", "phase1": "reject"})
        self.assertEqual(got["final"], "reject")

    def test_shadow_only_over_promote(self):
        got = combine_governance_decisions({"a": "promote", "b": "shadow_only"})
        self.assertEqual(got["final"], "shadow_only")

    def test_all_promote_gives_promote(self):
        got = combine_governance_decisions({"a": "promote", "b": "promote"})
        self.assertEqual(got["final"], "promote")

    def test_message_contains_parts(self):
        got = combine_governance_decisions({"stop_rules": "promote", "phase1": "reject"})
        self.assertIn("stop_rules=promote", got["message"])

    def test_empty_gives_unknown(self):
        got = combine_governance_decisions({})
        self.assertEqual(got["final"], "unknown")

    def test_returns_dict_with_required_keys(self):
        got = combine_governance_decisions({"x": "promote"})
        self.assertIn("final", got)
        self.assertIn("final_label", got)
        self.assertIn("parts", got)
        self.assertIn("message", got)


class TestRunGovernanceCommands(unittest.TestCase):

    def test_collects_logs(self):
        class _CP:
            returncode = 0
            stdout = "ok"
            stderr = ""

        with patch("app_core.governance_runtime_core.subprocess.run", return_value=_CP()):
            got = run_governance_commands([["python", "tools/x.py"]], cwd=".")
        self.assertEqual(len(got["logs"]), 1)
        self.assertIn("rc=0", got["logs"][0])


class TestRunGovernanceOperationAndAppendLog(unittest.TestCase):

    def test_run_operation_promote(self):
        class _CP:
            returncode = 0
            stdout = "done"
            stderr = ""

        with patch("app_core.governance_runtime_core.subprocess.run", return_value=_CP()):
            got = run_governance_operation(promote=True, cwd=".", tag="x")
        self.assertTrue(got["ok"])
        self.assertIn("promote_baseline.py", got["log_line"])

    def test_append_governance_log_merges(self):
        class _CP:
            returncode = 0
            stdout = "done"
            stderr = ""

        with patch("app_core.governance_runtime_core.subprocess.run", return_value=_CP()):
            got = run_governance_operation(promote=True, cwd=".", tag="x")
        merged = append_governance_log("prev", got["log_line"], now_iso="2026-03-10T10:00:00")
        self.assertIn("prev", merged["last_log"])
        self.assertIn("promote_baseline.py", merged["last_log"])
        self.assertEqual(merged["last_run"], "2026-03-10T10:00:00")


class TestAppendGovernanceLog(unittest.TestCase):

    def test_merges_two_logs(self):
        merged = append_governance_log("first_log", "second_log", now_iso="2026-01-01T00:00:00")
        self.assertIn("first_log", merged["last_log"])
        self.assertIn("second_log", merged["last_log"])

    def test_none_previous_uses_line(self):
        merged = append_governance_log(None, "only_line", now_iso="2026-01-01")
        self.assertEqual(merged["last_log"], "only_line")

    def test_none_line_uses_previous(self):
        merged = append_governance_log("prev_log", None, now_iso="2026-01-01")
        self.assertEqual(merged["last_log"], "prev_log")

    def test_timestamp_stored(self):
        merged = append_governance_log(None, "line", now_iso="2026-05-25T12:00:00")
        self.assertEqual(merged["last_run"], "2026-05-25T12:00:00")

    def test_returns_dict(self):
        result = append_governance_log("a", "b", now_iso="2026-01-01")
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
