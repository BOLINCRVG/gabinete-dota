import unittest

from app_core.visao_geral_runtime_core import (
    apply_runtime_tuning_to_session,
    build_input_signature_payload,
    build_microseg_caption,
    compute_input_signature,
    resolve_shared_league_hint,
)


class VisaoGeralRuntimeCoreTests(unittest.TestCase):
    def test_build_microseg_caption(self):
        text = build_microseg_caption({"applied": True, "key": "x", "source": "live", "bo": 3, "phase": "late", "n": 10, "bias": 0.03})
        self.assertIn("Microseg calibracao", text)
        self.assertIn("+3.0 pp", text)

    def test_build_microseg_caption_never_crashes_on_bad_n(self):
        """Bug de produção 18/07: um `n` que `int()` recusa derrubava a Visão
        Geral inteira. A caption nunca pode quebrar, seja qual for o formato."""
        self.assertIn("n=35", build_microseg_caption(
            {"applied": True, "key": "x", "n": [10, 20, 5], "bias": 0.0}))
        for bad in ({}, None, "abc", [], object(), {"a": 1}, (1, 2), b"xx"):
            txt = build_microseg_caption({"applied": True, "key": "x", "n": bad, "bias": 0.0})
            self.assertIn("Microseg calibracao", txt)  # não levanta

    def test_caption_sobrevive_a_infinito_e_nan(self):
        """REGRESSÃO do fix de 18/07: ele capturava só (TypeError, ValueError).
        `int(inf)` levanta OverflowError — ou seja, o app AINDA caía por esse
        caminho. Achado em 19/07 ao investigar a causa raiz."""
        for bad in (float("inf"), float("-inf"), float("nan"), [float("inf"), 1]):
            txt = build_microseg_caption({"applied": True, "key": "x", "n": bad, "bias": 0.0})
            self.assertIn("n=0", txt, f"{bad} deveria virar 0, não derrubar")

    def test_n_valido_continua_exato(self):
        """A blindagem não pode falsificar o número quando ele é legítimo."""
        for entrada, esperado in ((10, "n=10"), (10.9, "n=10"), ("42", "n=42"),
                                  (0, "n=0"), (-5, "n=-5")):
            txt = build_microseg_caption(
                {"applied": True, "key": "x", "n": entrada, "bias": 0.0})
            self.assertIn(esperado, txt)

    def test_anomalia_de_n_fica_registrada_para_diagnostico(self):
        """Como a causa raiz de 18/07 nunca foi provada, o formato inesperado
        precisa ficar gravado — na próxima ocorrência vira teste real."""
        from app_core.visao_geral_runtime_core import _MICROSEG_N_ANOMALIAS
        _MICROSEG_N_ANOMALIAS.clear()
        build_microseg_caption({"applied": True, "key": "x", "n": float("nan"), "bias": 0.0})
        self.assertTrue(_MICROSEG_N_ANOMALIAS, "anomalia deveria ter sido registrada")
        self.assertEqual(_MICROSEG_N_ANOMALIAS[0]["tipo"], "float")

    def test_resolve_shared_league_hint(self):
        self.assertEqual(resolve_shared_league_hint(ml_info={"blue": {"league": "LCK"}, "red": {"league": "LCK"}}), "LCK")
        self.assertEqual(resolve_shared_league_hint(ml_info={}, fit_league_hint="LPL"), "LPL")
        self.assertEqual(
            resolve_shared_league_hint(
                ml_info={"blue": {"league": "LCK"}, "red": {"league": "LPL"}},
                match_league="INTER",
                league_a="LCK",
                league_b="LPL",
            ),
            "",
        )

    def test_build_and_hash_input_signature(self):
        payload = build_input_signature_payload(
            matchup={"bo": 3},
            filters={"year": "2026"},
            config={"engine": "mlcore_v2"},
            lineup={"mode": "ml"},
            versions={"csv": {"path": "x"}},
        )
        sig = compute_input_signature(payload)
        self.assertIn("matchup", sig["payload"])
        self.assertEqual(len(sig["sha256"]), 64)

    def test_apply_runtime_tuning_to_session(self):
        state = {}
        out = apply_runtime_tuning_to_session(
            session_state=state,
            league_hint="LCK",
            bo=3,
            confidence_baseline={},
            confidence_governor_budget_fn=lambda league_hint, confidence_baseline=None: {"budget_pp": 7.5},
            series_coherence_post_cap_fn=lambda league_hint, bo, confidence_baseline=None: {"max_dev_pp": 14.0},
            governor_default_pp=8.0,
        )
        self.assertEqual(state["_ml_confidence_governor_budget_runtime_pp"], 7.5)
        self.assertEqual(state["_ml_series_coherence_post_max_dev_pp_runtime"], 14.0)
        self.assertEqual(out["gov_budget_pp"], 7.5)


if __name__ == "__main__":
    unittest.main()
