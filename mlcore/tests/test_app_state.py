import unittest

from app_core.app_state import (
    APP_STATE_DEFAULTS,
    apply_excellent_value_preset,
    apply_recommended_preset,
    apply_twenty_ten_preset,
    bootstrap_state,
    get_state_schema,
    get_bool,
    get_float,
    get_int,
    get_str,
)


class TestAppState(unittest.TestCase):
    def test_bootstrap_defaults_migrations_constraints(self):
        ss = {
            "resumo_mix_w_hist": 33,
            "resumo_max_odd": 99.0,
        }
        defaults = {
            "resumo_mix_w_laplace": 40,
            "resumo_max_odd": 1.72,
            "resumo_req_h2h": True,
        }
        migrations = {"resumo_mix_w_hist": "resumo_mix_w_laplace"}
        constraints = {"resumo_max_odd": (1.01, 10.0), "resumo_mix_w_laplace": (0.0, 100.0)}

        bootstrap_state(ss, defaults=defaults, migrations=migrations, constraints=constraints)
        self.assertEqual(ss["resumo_mix_w_laplace"], 33)
        self.assertEqual(ss["resumo_req_h2h"], True)
        self.assertEqual(ss["resumo_max_odd"], 10.0)

    def test_getters(self):
        ss = {"a": "true", "b": "2.5", "c": "7", "d": 10}
        self.assertTrue(get_bool(ss, "a", False))
        self.assertAlmostEqual(get_float(ss, "b", 0.0), 2.5, places=4)
        self.assertEqual(get_int(ss, "c", 0), 7)
        self.assertEqual(get_str(ss, "d", ""), "10")

    def test_apply_recommended_preset(self):
        ss = {}
        apply_recommended_preset(ss, "Padrao")
        self.assertEqual(ss["cfg_profile"], "Padrão")
        self.assertEqual(ss["ml_engine"], "mlcore_v2")
        self.assertEqual(ss["league_mode"], "auto")
        self.assertIn("n_sims", ss)

    def test_apply_excellent_value_preset(self):
        ss = {}
        apply_excellent_value_preset(ss)
        self.assertEqual(ss["ml_engine_compare"], True)
        self.assertEqual(ss["resumo_req_h2h"], False)
        self.assertEqual(ss["half_life_days"], 90)

    def test_apply_twenty_ten_preset(self):
        ss = {}
        apply_twenty_ten_preset(ss)
        self.assertEqual(ss["cfg_profile"], "20/10")
        self.assertEqual(ss["resumo_max_odd"], 3.00)
        self.assertEqual(ss["resumo_req_h2h"], False)
        self.assertEqual(ss["fusion_transfer_boost"], 0.60)

    def test_get_state_schema(self):
        sch = get_state_schema()
        self.assertIn("defaults", sch)
        self.assertIn("migrations", sch)
        self.assertIn("constraints", sch)
        self.assertEqual(sch["defaults"].get("resumo_max_odd"), APP_STATE_DEFAULTS.get("resumo_max_odd"))


if __name__ == "__main__":
    unittest.main()

