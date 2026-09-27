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


class ThresholdDefaultsSingleSourceTests(unittest.TestCase):
    """Passo 1 do mapa (21/07): os limiares default sc_th_* em APP_STATE_DEFAULTS
    DERIVAM do registro central (antes eram literais à mão, duplicata do
    default_thresholds). Trava a fonte única: mudar um limiar no registro
    reflete no estado sozinho."""

    def test_sc_th_derivam_do_registro(self):
        from app_core.app_state import APP_STATE_DEFAULTS
        from app_core.stats_config import STATS_MARKET_CONFIG
        for cfg in STATS_MARKET_CONFIG.values():
            tk = str(cfg["threshold_key"])
            self.assertEqual(APP_STATE_DEFAULTS.get(tk), str(cfg["default_thresholds"]),
                             f"{tk} não bate com o registro")

    def test_valores_historicos_preservados(self):
        from app_core.app_state import APP_STATE_DEFAULTS
        self.assertEqual(APP_STATE_DEFAULTS.get("sc_th_kills"), "23.5,25.5,27.5,29.5")
        # sc_th_time era "" ate 13/09/2026, quando o dono pediu o recorte de
        # TEMPO na tela de stats do campeonato: "pode ser o padrao de 32:30 a
        # linha padrao". 32.5 = 32:30, e bate com o default_line do mercado,
        # que ja era 32.5 desde sempre — a lista de limiares e que estava vazia,
        # e por isso nenhuma coluna "% Tempo >" aparecia.
        self.assertEqual(APP_STATE_DEFAULTS.get("sc_th_time"), "32.5")

    def test_nao_ha_sc_th_extra_nem_faltando(self):
        from app_core.app_state import APP_STATE_DEFAULTS
        from app_core.stats_config import STATS_MARKET_CONFIG
        no_estado = {k for k in APP_STATE_DEFAULTS if k.startswith("sc_th_")}
        no_registro = {str(cfg["threshold_key"]) for cfg in STATS_MARKET_CONFIG.values()}
        self.assertEqual(no_estado, no_registro)


if __name__ == "__main__":
    unittest.main()

