import logging
import unittest

from app_core.streamlit_apptest_core import new_apptest_for_file, quiet_streamlit_bare_mode_warnings, run_apptest_quiet


class UISmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
        logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
        quiet_streamlit_bare_mode_warnings()

    def test_streamlit_app_renders_main_controls(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at = new_apptest_for_file("plays_app.py")
        run_apptest_quiet(at, 180)

        self.assertEqual(len(at.exception), 0, f"App raised exceptions: {at.exception}")

        radios = list(at.radio)
        self.assertGreaterEqual(len(radios), 1)
        radio_labels = [str(r.label) for r in radios]
        self.assertTrue(("P\u00e1gina" in radio_labels) or ("Pagina" in radio_labels))

        sels = {s.key: s for s in at.selectbox}
        btn_labels = [str(b.label) for b in at.button]
        in_match_screen = ("teamA" in sels and "teamB" in sels)
        in_bootstrap_screen = ("Entrar no app" in btn_labels)
        self.assertTrue(
            in_match_screen or in_bootstrap_screen,
            f"Neither matchup controls nor bootstrap continue button found. selectbox_keys={list(sels.keys())} buttons={btn_labels}",
        )
        if in_match_screen:
            self.assertIn("Analisar Confronto", btn_labels)

    def test_players_and_champions_filter_keys_exist(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at = new_apptest_for_file("plays_app.py")
        try:
            at.session_state["ui_page_mode"] = "Completo"
            at.session_state["selected_tab_names"] = [
                "Vis\u00e3o Geral",
                "Consist\u00eancia de Mercado (Laplace)",
                "Series",
                "Players",
                "Campe\u00f5es",
                "Rankings",
                "Par\u00e2metros",
            ]
        except Exception:
            pass
        run_apptest_quiet(at, 180)

        self.assertEqual(len(at.exception), 0, f"App raised exceptions: {at.exception}")

        sels = {s.key: s for s in at.selectbox}
        expected_keys = {
            "ch_year",
            "ch_league",
            "ch_split",
            "ch_playoffs",
            "pl_year",
            "pl_league",
            "pl_split",
            "pl_playoffs",
        }
        missing = sorted(k for k in expected_keys if (k not in sels) and (k not in at.session_state))
        self.assertFalse(missing, f"Missing Players/Campe\u00f5es filter keys: {missing}")

    def test_e2e_tab_focus_players_and_champions(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at_players = new_apptest_for_file("plays_app.py")
        try:
            at_players.session_state["ui_page_mode"] = "Completo"
            at_players.session_state["selected_tab_names"] = ["Players"]
        except Exception:
            pass
        run_apptest_quiet(at_players, 180)
        self.assertEqual(len(at_players.exception), 0, f"Players focus raised exceptions: {at_players.exception}")
        sels_p = {s.key: s for s in at_players.selectbox}
        self.assertTrue(("pl_year" in sels_p) or ("pl_year" in at_players.session_state))
        self.assertTrue(("pl_league" in sels_p) or ("pl_league" in at_players.session_state))

        at_ch = new_apptest_for_file("plays_app.py")
        try:
            at_ch.session_state["ui_page_mode"] = "Completo"
            at_ch.session_state["selected_tab_names"] = ["Campe\u00f5es"]
        except Exception:
            pass
        run_apptest_quiet(at_ch, 180)
        self.assertEqual(len(at_ch.exception), 0, f"Campe\u00f5es focus raised exceptions: {at_ch.exception}")
        sels_c = {s.key: s for s in at_ch.selectbox}
        self.assertTrue(("ch_year" in sels_c) or ("ch_year" in at_ch.session_state))
        self.assertTrue(("ch_league" in sels_c) or ("ch_league" in at_ch.session_state))

    def test_e2e_tab_focus_series(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at_series = new_apptest_for_file("plays_app.py")
        try:
            at_series.session_state["ui_page_mode"] = "Completo"
            at_series.session_state["selected_tab_names"] = ["Series"]
        except Exception:
            pass
        run_apptest_quiet(at_series, 180)
        self.assertEqual(len(at_series.exception), 0, f"Series focus raised exceptions: {at_series.exception}")
        sels = {s.key: s for s in at_series.selectbox}
        if "bo" not in sels:
            btn_labels = [str(b.label) for b in at_series.button]
            bootstrap_selector = ("season_label" in sels) and ("cfg_profile" in sels)
            if ("Entrar no app" in btn_labels) or bootstrap_selector:
                self.skipTest("series controls gated by CSV bootstrap screen")
        self.assertIn("bo", sels)

    def test_e2e_tab_focus_live_bet(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at_live = new_apptest_for_file("plays_app.py")
        try:
            at_live.session_state["nav_main"] = "Live-Bet"
        except Exception:
            pass
        run_apptest_quiet(at_live, 180)
        self.assertEqual(len(at_live.exception), 0, f"Live-Bet focus raised exceptions: {at_live.exception}")
        sels = {s.key: s for s in at_live.selectbox}
        if ("lb_team_a" not in sels) and ("lb_team_a" not in at_live.session_state):
            btn_labels = [str(b.label) for b in at_live.button]
            bootstrap_selector = ("season_label" in sels) and ("cfg_profile" in sels)
            if ("Entrar no app" in btn_labels) or bootstrap_selector:
                self.skipTest("live-bet controls gated by CSV bootstrap screen")
        self.assertTrue(("lb_team_a" in sels) or ("lb_team_a" in at_live.session_state))
        self.assertTrue(("lb_championship" in sels) or ("lb_championship" in at_live.session_state))

    def test_e2e_functional_ml_map_used(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at = new_apptest_for_file("plays_app.py")
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App raised exceptions on bootstrap: {at.exception}")

        sels = {s.key: s for s in at.selectbox}
        if "teamA" not in sels or "teamB" not in sels:
            self.skipTest("team selectors unavailable")

        opts_a = [str(x) for x in list(getattr(sels["teamA"], "options", []) or []) if str(x)]
        opts_b = [str(x) for x in list(getattr(sels["teamB"], "options", []) or []) if str(x)]
        if len(opts_a) < 2 or len(opts_b) < 2:
            self.skipTest("not enough teams in selector options")

        t1 = opts_a[0]
        t2 = next((x for x in opts_b if x != t1), None)
        if not t2:
            self.skipTest("could not find distinct teams for smoke")

        # Force directly into results flow to validate core output signal.
        at.session_state["teamA"] = t1
        at.session_state["teamB"] = t2
        at.session_state["bo"] = 3
        at.session_state["flow_stage"] = "results"
        at.session_state["_analysis_force_once"] = True
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App raised exceptions in functional run: {at.exception}")

        self.assertIn("_p_map_used", at.session_state, "session_state missing _p_map_used")
        p = at.session_state["_p_map_used"]
        self.assertIsNotNone(p, "_p_map_used is None")
        p = float(p)
        self.assertTrue(0.0 < p < 1.0, f"_p_map_used out of bounds: {p}")

    def test_e2e_tab_switch_preserves_match_state(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at = new_apptest_for_file("plays_app.py")
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App raised exceptions on bootstrap: {at.exception}")

        sels = {s.key: s for s in at.selectbox}
        if "teamA" not in sels or "teamB" not in sels:
            self.skipTest("team selectors unavailable")
        opts_a = [str(x) for x in list(getattr(sels["teamA"], "options", []) or []) if str(x)]
        opts_b = [str(x) for x in list(getattr(sels["teamB"], "options", []) or []) if str(x)]
        if len(opts_a) < 2 or len(opts_b) < 2:
            self.skipTest("not enough teams in selector options")

        t1 = opts_a[0]
        t2 = next((x for x in opts_b if x != t1), None)
        if not t2:
            self.skipTest("could not find distinct teams")

        at.session_state["teamA"] = t1
        at.session_state["teamB"] = t2
        at.session_state["bo"] = 3
        at.session_state["flow_stage"] = "results"
        at.session_state["_analysis_force_once"] = True
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App exceptions before tab switch: {at.exception}")

        # Simula troca de abas e volta para validar estabilidade do estado principal.
        at.session_state["selected_tab_names"] = ["Consist\u00eancia de Mercado (Laplace)"]
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App exceptions on Consistência tab: {at.exception}")

        at.session_state["selected_tab_names"] = ["Vis\u00e3o Geral"]
        at.run(timeout=180)
        self.assertEqual(len(at.exception), 0, f"App exceptions returning to Visão Geral: {at.exception}")

        self.assertIn("teamA", at.session_state)
        self.assertIn("teamB", at.session_state)
        self.assertIn("bo", at.session_state)
        self.assertEqual(str(at.session_state["teamA"]), t1)
        self.assertEqual(str(at.session_state["teamB"]), t2)
        self.assertEqual(int(at.session_state["bo"] or 0), 3)

    def test_market_selector_state_persists_across_rerun(self):
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        at = new_apptest_for_file("plays_app.py")
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App raised exceptions on bootstrap: {at.exception}")

        sels = {s.key: s for s in at.selectbox}
        if "teamA" not in sels or "teamB" not in sels:
            self.skipTest("team selectors unavailable")
        opts_a = [str(x) for x in list(getattr(sels["teamA"], "options", []) or []) if str(x)]
        opts_b = [str(x) for x in list(getattr(sels["teamB"], "options", []) or []) if str(x)]
        if len(opts_a) < 2 or len(opts_b) < 2:
            self.skipTest("not enough teams in selector options")

        t1 = opts_a[0]
        t2 = next((x for x in opts_b if x != t1), None)
        if not t2:
            self.skipTest("could not find distinct teams")

        at.session_state["teamA"] = t1
        at.session_state["teamB"] = t2
        at.session_state["bo"] = 3
        at.session_state["flow_stage"] = "results"
        at.session_state["_analysis_force_once"] = True
        at.session_state["consistency_market_tab"] = "Resumo"
        at.session_state["vg_odds_market_tab"] = "ML + Totais"
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App exceptions before persistence check: {at.exception}")
        self.assertEqual(str(at.session_state["consistency_market_tab"] or ""), "Resumo")
        self.assertEqual(str(at.session_state["vg_odds_market_tab"] or ""), "ML + Totais")

        at.session_state["_analysis_force_once"] = True
        run_apptest_quiet(at, 180)
        self.assertEqual(len(at.exception), 0, f"App exceptions after rerun persistence check: {at.exception}")
        self.assertEqual(str(at.session_state["consistency_market_tab"] or ""), "Resumo")
        self.assertEqual(str(at.session_state["vg_odds_market_tab"] or ""), "ML + Totais")


if __name__ == "__main__":
    unittest.main()
