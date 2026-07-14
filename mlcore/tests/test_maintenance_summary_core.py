from __future__ import annotations

import unittest

from app_core.maintenance_summary_core import build_maintenance_quality_status, build_quick_maintenance_alerts


class MaintenanceSummaryCoreTests(unittest.TestCase):
    def test_build_quick_maintenance_alerts_maps_status_messages(self) -> None:
        got = build_quick_maintenance_alerts(
            {
                "alerts": [
                    "Drift de dados: CSV diario mais novo que o `ml_artifact.json`.",
                    "Parametros desatualizados vs artifact.",
                    "Walk-forward desatualizado (>7 dias).",
                    "Retreino de modelo pendente (flag interna).",
                ],
                "need_retrain_model": True,
                "need_retrain_params": True,
                "need_walkforward": True,
            }
        )
        self.assertEqual(
            got["alerts"],
            [
                "Retreinar modelo: CSV diario esta mais novo que o ml_artifact.json.",
                "Retreinar parametros: params_fitted.json esta desatualizado vs artifact.",
                "Rodar walk-forward: ultimo relatorio tem mais de 7 dias.",
            ],
        )
        self.assertFalse(got["ok"])

    def test_build_quick_maintenance_alerts_prefers_running_message_for_params(self) -> None:
        got = build_quick_maintenance_alerts(
            {
                "alerts": [
                    "Parametros desatualizados vs artifact.",
                    "Retreino de parametros em execucao (auto_full, context_models).",
                ],
                "need_retrain_params": True,
                "params_fit_running": True,
            }
        )
        self.assertEqual(got["alerts"], ["Retreino de parametros em execucao."])

    def test_build_quick_maintenance_alerts_falls_back_to_generic(self) -> None:
        got = build_quick_maintenance_alerts(
            {
                "alerts": [],
                "need_retrain_model": True,
                "need_retrain_params": False,
                "need_walkforward": True,
            }
        )
        self.assertIn("Retreinar modelo: manutencao pendente.", got["alerts"])
        self.assertIn("Rodar walk-forward: manutencao pendente.", got["alerts"])

    def test_build_quick_maintenance_alerts_ok(self) -> None:
        got = build_quick_maintenance_alerts({"alerts": []})
        self.assertEqual(got["alerts"], [])
        self.assertTrue(got["ok"])

    def test_build_maintenance_quality_status_running(self) -> None:
        got = build_maintenance_quality_status(
            {
                "severity": "warning",
                "need_retrain_model": False,
                "need_retrain_params": True,
                "need_walkforward": True,
                "params_fit_running": True,
                "params_fit_runtime": {"summary": "context_models | players_context_fit | candidate_eval | global | 4/8 | eta~435s"},
            }
        )
        self.assertEqual(got["model_base"]["state"], "ready")
        self.assertEqual(got["operational"]["state"], "running")
        self.assertIn("4/8", got["operational"]["text"])
        self.assertEqual(got["deep"]["state"], "pending")

    def test_build_maintenance_quality_status_ready_with_patch_warn(self) -> None:
        got = build_maintenance_quality_status(
            {
                "severity": "warning",
                "need_retrain_model": False,
                "need_retrain_params": False,
                "need_walkforward": False,
                "need_patch_recalibration": True,
            }
        )
        self.assertEqual(got["operational"]["state"], "ready_warn")


if __name__ == "__main__":
    unittest.main()
