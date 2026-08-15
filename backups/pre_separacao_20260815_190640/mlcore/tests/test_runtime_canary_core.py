import unittest

from app_core.runtime_canary_core import evaluate_runtime_canary


class RuntimeCanaryCoreTests(unittest.TestCase):

    def test_promotes_when_all_clean(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 0, "warn_count": 0}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertEqual(payload["decision"], "promote")

    def test_rejects_on_matchup_reviews(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 1, "rows": [{"status": "review"}]},
            identity_report={"summary": {"error_count": 0, "warn_count": 0}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertEqual(payload["decision"], "reject")
        self.assertIn("matchup_integrity_review", payload["reasons"])

    def test_rejects_on_identity_errors(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 2, "warn_count": 0}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertEqual(payload["decision"], "reject")
        self.assertIn("identity_registry_errors", payload["reasons"])

    def test_shadow_only_on_identity_warns(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 0, "warn_count": 1}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertEqual(payload["decision"], "shadow_only")
        self.assertIn("identity_registry_warns", payload["reasons"])

    def test_shadow_only_on_missing_reports(self):
        payload = evaluate_runtime_canary(matchup_report={}, identity_report={}, csv_gate_report={})
        self.assertEqual(payload["decision"], "shadow_only")

    def test_csv_gate_reject_propagates(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 0, "warn_count": 0}},
            csv_gate_report={"decision": "reject"},
        )
        self.assertEqual(payload["decision"], "reject")

    def test_pass_reason_when_all_ok(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 0, "warn_count": 0}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertIn("pass", payload["reasons"])

    def test_components_present(self):
        payload = evaluate_runtime_canary(
            matchup_report={"review_count": 0, "rows": []},
            identity_report={"summary": {"error_count": 0, "warn_count": 0}},
            csv_gate_report={"decision": "promote"},
        )
        self.assertIn("components", payload)
        self.assertIn("matchup_review_count", payload["components"])

    def test_returns_dict(self):
        result = evaluate_runtime_canary(matchup_report={}, identity_report={}, csv_gate_report={})
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
