import unittest

from app_core.governance_report_core import (
    build_anchor_cases_rows,
    build_leakage_file_rows,
    build_parts_rows,
    build_phase12_by_league_rows,
    build_stop_rules_banner,
    build_stop_rules_global_band_rows,
    build_stop_rules_league_band_rows,
    build_stop_rules_major_rows,
    decision_banner_parts,
)


class GovernanceReportCoreTests(unittest.TestCase):
    def test_decision_banner_parts(self):
        got = decision_banner_parts("promote", prefix="Fase 1")
        self.assertEqual(got["message_type"], "success")
        self.assertIn("Fase 1", got["message"])

    def test_build_anchor_cases_rows(self):
        rows = build_anchor_cases_rows({"cases": [{"name": "A", "ok": True, "market": "ML", "side": "A", "odd_value": 1.9}]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Caso"], "A")

    def test_build_leakage_file_rows(self):
        rows = build_leakage_file_rows({"files": [{"path": "c:/tmp/a.csv", "ok": True, "rows": 10, "errors": ["e1"], "warnings": ["w1"]}]})
        self.assertEqual(rows[0]["Arquivo"], "a.csv")

    def test_build_parts_rows(self):
        rows = build_parts_rows({"x": "promote"})
        self.assertEqual(rows, [{"Check": "x", "Decision": "promote"}])

    def test_build_phase12_by_league_rows(self):
        rows = build_phase12_by_league_rows({"by_league": {"LCK": {"rows": 10, "gain_logloss": 0.01, "off": {"logloss": 0.5}, "on": {"logloss": 0.49}}}})
        self.assertEqual(rows[0]["Liga"], "LCK")

    def test_build_stop_rules_helpers(self):
        payload = {
            "promotion_decision": {
                "decision": "promote",
                "reasons": ["r1"],
                "checks": {
                    "major_leagues": [
                        {"league": "LCK", "rows": 20, "gain_logloss_vs_base": 0.01, "fail_drawdown": False}
                    ]
                },
            },
            "monitoring": {
                "global": {
                    "favorite_odd_bands_final": [{"band": "1.5-1.8", "n": 5, "calibration_gap_pp": 1.2, "winrate_actual": 0.7, "p_mean": 0.68}],
                    "favorite_odd_bands_base": [{"band": "1.5-1.8", "calibration_gap_pp": 1.6}],
                },
                "by_league": {
                    "LCK": {
                        "favorite_odd_bands_final": [{"band": "1.5-1.8", "n": 3, "calibration_gap_pp": 1.0, "winrate_actual": 0.66, "p_mean": 0.64}]
                    }
                },
            },
        }
        banner = build_stop_rules_banner(payload)
        self.assertEqual(banner["message_type"], "success")
        self.assertEqual(len(build_stop_rules_major_rows(payload)), 1)
        self.assertEqual(len(build_stop_rules_global_band_rows(payload)), 1)
        self.assertEqual(len(build_stop_rules_league_band_rows(payload, "LCK")), 1)


if __name__ == "__main__":
    unittest.main()
