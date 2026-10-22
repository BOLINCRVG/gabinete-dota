import unittest

try:
    from tools.model_health_report import generate_health_report
except ImportError as _sem_ferramenta:  # ferramenta so do LoL: no Dota o modulo e pulado
    import unittest as _u
    raise _u.SkipTest("tools.model_health_report nao existe neste repositorio: " + str(_sem_ferramenta))


class TestModelHealthReport(unittest.TestCase):
    def test_generate_health_report_no_data(self):
        rep = generate_health_report(
            csv_paths=["__missing_file__.csv"],
            artifact_path="ml_artifact.json",
            players_artifact_path="players_artifact.json",
            min_train_games=10,
            bins=5,
            min_league_games=5,
        )
        self.assertIn("walkforward", rep)
        self.assertIn("calibration", rep)
        # With missing file, expected error shape from internals.
        self.assertTrue(
            (rep["walkforward"].get("error") is not None)
            or (rep["calibration"].get("error") is not None)
        )


if __name__ == "__main__":
    unittest.main()

