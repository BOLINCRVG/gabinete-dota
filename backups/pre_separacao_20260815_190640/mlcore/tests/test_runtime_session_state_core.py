import unittest

from app_core.runtime_session_state_core import (
    get_cal_diag_last_report,
    get_daily_pipeline_report,
    get_governance_last_log,
    get_lv_last_report,
    get_ml_last_log,
    get_ml_confidence_baseline_last_report,
    get_quick_maint_last_log,
    get_team_state_artifact_last_report,
    get_wf_last_report,
    get_wf_last_report_at,
    set_cal_diag_last_report,
    set_daily_pipeline_report,
    set_governance_last_log,
    set_lv_last_report,
    set_ml_last_log,
    set_ml_confidence_baseline_last_report,
    set_quick_maint_last_log,
    set_team_state_artifact_last_report,
    set_wf_last_report,
    set_wf_last_report_at,
)


class RuntimeSessionStateCoreTests(unittest.TestCase):
    def test_report_roundtrip(self) -> None:
        state = {}
        set_wf_last_report(state, {"ok": True}, at="2026-03-10T10:00:00")
        set_lv_last_report(state, {"ok": False}, at="2026-03-10T11:00:00")
        set_cal_diag_last_report(state, {"models": {"fused": {}}}, at="2026-03-10T11:30:00")
        set_ml_confidence_baseline_last_report(state, {"by_league": {"LCK": {}}}, at="2026-03-10T11:40:00")
        set_team_state_artifact_last_report(state, {"meta": {"teams": 10}}, at="2026-03-10T11:50:00")
        set_daily_pipeline_report(state, {"ok": True, "steps": []}, at="2026-03-10T12:00:00")
        self.assertEqual(get_wf_last_report(state), {"ok": True})
        self.assertEqual(get_lv_last_report(state), {"ok": False})
        self.assertEqual(get_cal_diag_last_report(state), {"models": {"fused": {}}})
        self.assertEqual(get_ml_confidence_baseline_last_report(state), {"by_league": {"LCK": {}}})
        self.assertEqual(get_team_state_artifact_last_report(state), {"meta": {"teams": 10}})
        self.assertEqual(get_daily_pipeline_report(state), {"ok": True, "steps": []})
        self.assertEqual(get_wf_last_report_at(state), "2026-03-10T10:00:00")
        self.assertEqual(set_wf_last_report_at(state, "2026-03-10T12:00:00"), "2026-03-10T12:00:00")

    def test_log_roundtrip(self) -> None:
        state = {}
        set_ml_last_log(state, "ml")
        set_quick_maint_last_log(state, "quick")
        set_governance_last_log(state, "gov")
        self.assertEqual(get_ml_last_log(state), "ml")
        self.assertEqual(get_quick_maint_last_log(state), "quick")
        self.assertEqual(get_governance_last_log(state), "gov")
