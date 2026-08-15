from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import math

from mlcore.state_team_artifact import (
    TeamState,
    _apply_league_switch_recenter,
    _apply_patch_switch_recenter,
    _apply_roster_recenter,
    _days_between_iso,
    _effective_k,
    _file_sig_info,
    _glicko_lite_p_blue,
    _resolve_effective_params,
    _resolve_scoped_param,
    _safe_iso_date,
    _same_source_path_order,
    _team_state_from_dict,
    build_team_state_artifact,
    load_team_state_artifact,
    save_team_state_artifact,
)


class ApplyPatchSwitchRecenterTests(unittest.TestCase):
    """Regra stateful extraída do loop (20/07, provada por equivalence_check):
    ao trocar de patch, o mu é puxado para o prior da liga."""

    def test_sem_troca_nao_faz_nada(self):
        st = TeamState(team="A", key="a", mu=1.0)
        sw, rec = _apply_patch_switch_recenter(st, patch_switched=False, patch_mu_recenter=0.11, patch_sigma_mult=1.0, league_prior_mu=0.0)
        self.assertEqual((sw, rec), (0, 0))
        self.assertEqual(st.mu, 1.0)

    def test_troca_recentra_mu_para_o_prior(self):
        st = TeamState(team="A", key="a", mu=1.0)
        sw, rec = _apply_patch_switch_recenter(st, patch_switched=True, patch_mu_recenter=0.11, patch_sigma_mult=1.0, league_prior_mu=0.0)
        self.assertEqual((sw, rec), (1, 1))
        self.assertLess(st.mu, 1.0)  # puxou pra baixo (prior=0)
        self.assertGreater(st.mu, 0.0)

    def test_shrink_zero_conta_switch_mas_nao_recentra(self):
        st = TeamState(team="A", key="a", mu=1.0)
        sw, rec = _apply_patch_switch_recenter(st, patch_switched=True, patch_mu_recenter=0.0, patch_sigma_mult=1.0, league_prior_mu=0.0)
        self.assertEqual((sw, rec), (1, 0))
        self.assertEqual(st.mu, 1.0)


class ApplyLeagueSwitchRecenterTests(unittest.TestCase):
    def test_mesma_liga_nao_recentra(self):
        st = TeamState(team="A", key="a", mu=1.0, last_league="LCK")
        inc = _apply_league_switch_recenter(st, league="LCK", league_switch_mu_recenter=0.12, league_prior_mu=0.0)
        self.assertEqual(inc, 0)
        self.assertEqual(st.mu, 1.0)

    def test_troca_de_liga_recentra(self):
        st = TeamState(team="A", key="a", mu=1.0, last_league="LPL")
        inc = _apply_league_switch_recenter(st, league="LCK", league_switch_mu_recenter=0.12, league_prior_mu=0.0)
        self.assertEqual(inc, 1)
        self.assertLess(st.mu, 1.0)

    def test_sem_last_league_nao_recentra(self):
        st = TeamState(team="A", key="a", mu=1.0, last_league="")
        inc = _apply_league_switch_recenter(st, league="LCK", league_switch_mu_recenter=0.12, league_prior_mu=0.0)
        self.assertEqual(inc, 0)


class ApplyRosterRecenterTests(unittest.TestCase):
    _KW = dict(roster_mu_recenter=0.28, roster_cont_hard_reset=0.25,
               roster_hard_reset_extra_sigma=0.18, max_sigma=2.25, league_prior_mu=0.0)

    def test_cc_none_nao_conta(self):
        st = TeamState(team="A", key="a", mu=1.0)
        self.assertEqual(_apply_roster_recenter(st, None, **self._KW), (0, 0, 0, 0))

    def test_roster_igual_conta_evento_sem_mudanca(self):
        st = TeamState(team="A", key="a", mu=1.0)
        ev, ewc, hard, mu = _apply_roster_recenter(st, 1.0, **self._KW)
        self.assertEqual((ev, ewc, hard, mu), (1, 0, 0, 0))
        self.assertEqual(st.mu, 1.0)

    def test_mudanca_parcial_recentra(self):
        st = TeamState(team="A", key="a", mu=1.0, gp=40)
        ev, ewc, hard, mu = _apply_roster_recenter(st, 0.6, **self._KW)
        self.assertEqual((ev, ewc, hard), (1, 1, 0))
        self.assertEqual(mu, 1)
        self.assertLess(st.mu, 1.0)

    def test_troca_total_e_hard_reset(self):
        # cc <= roster_cont_hard_reset (0.25) -> hard reset + infla sigma
        st = TeamState(team="A", key="a", mu=1.0, sigma=1.0, gp=40)
        sigma0 = st.sigma
        ev, ewc, hard, mu = _apply_roster_recenter(st, 0.10, **self._KW)
        self.assertEqual((ev, ewc, hard, mu), (1, 1, 1, 1))
        self.assertGreater(st.sigma, sigma0)  # sigma inflado no hard reset


class GlickoLitePBlueTests(unittest.TestCase):
    """CORE de cada jogo (extraído do loop, 20/07, provado por equivalence_check):
    P(blue) via diferença de mu atenuada pela incerteza. Retorna (g_b,g_r,denom,p)."""

    def test_forcas_iguais_da_meio(self):
        _, _, _, p = _glicko_lite_p_blue(mu_b=0.0, mu_r=0.0, sigma_b=0.5, sigma_r=0.5)
        self.assertAlmostEqual(p, 0.5, places=9)

    def test_mu_maior_favorece_blue(self):
        _, _, _, p = _glicko_lite_p_blue(mu_b=1.0, mu_r=-1.0, sigma_b=0.5, sigma_r=0.5)
        self.assertGreater(p, 0.5)

    def test_simetria_troca_de_lado(self):
        _, _, _, p_ab = _glicko_lite_p_blue(mu_b=0.6, mu_r=0.2, sigma_b=0.4, sigma_r=0.7)
        _, _, _, p_ba = _glicko_lite_p_blue(mu_b=0.2, mu_r=0.6, sigma_b=0.7, sigma_r=0.4)
        self.assertAlmostEqual(p_ab + p_ba, 1.0, places=9)

    def test_g_factor_encolhe_com_sigma(self):
        # adversário mais incerto (sigma alto) -> g menor -> favorito puxa menos
        _, _, _, p_certo = _glicko_lite_p_blue(mu_b=1.0, mu_r=0.0, sigma_b=0.3, sigma_r=0.3)
        _, _, _, p_incerto = _glicko_lite_p_blue(mu_b=1.0, mu_r=0.0, sigma_b=0.3, sigma_r=2.0)
        self.assertLess(p_incerto, p_certo)

    def test_valores_intermediarios_finitos(self):
        g_b, g_r, denom, p = _glicko_lite_p_blue(mu_b=0.3, mu_r=-0.1, sigma_b=0.8, sigma_r=1.2)
        for v in (g_b, g_r, denom, p):
            self.assertTrue(math.isfinite(v))
        self.assertGreater(denom, 1.0)  # sqrt(1 + sigmas^2)


class EffectiveKTests(unittest.TestCase):
    """Taxa de aprendizado por jogo (extraída do loop)."""

    def test_dentro_dos_limites(self):
        k = _effective_k(base_k=0.22, league="LCK", sigma_b=1.0, sigma_r=1.0, base_sigma=1.0)
        self.assertGreaterEqual(k, 0.04)
        self.assertLessEqual(k, 0.55)

    def test_mais_incerteza_move_mais(self):
        k_baixo = _effective_k(base_k=0.22, league="LCK", sigma_b=0.3, sigma_r=0.3, base_sigma=1.0)
        k_alto = _effective_k(base_k=0.22, league="LCK", sigma_b=2.0, sigma_r=2.0, base_sigma=1.0)
        self.assertGreater(k_alto, k_baixo)

    def test_clip_no_teto(self):
        # base_k absurdo é limitado a 0.55
        k = _effective_k(base_k=99.0, league="LCK", sigma_b=1.0, sigma_r=1.0, base_sigma=1.0)
        self.assertEqual(k, 0.55)

    def test_clip_no_piso(self):
        k = _effective_k(base_k=0.0, league="LCK", sigma_b=0.1, sigma_r=0.1, base_sigma=1.0)
        self.assertEqual(k, 0.04)


class ResolveScopedParamTests(unittest.TestCase):
    """Helper extraído do build (19/07): global -> override liga -> mistura patch.
    Antes era nested (intestável); a rede de equivalência provou 856 times iguais."""

    def test_sem_override_devolve_global(self):
        self.assertEqual(_resolve_scoped_param(1.0, {}, {}, "LCK", "14.1"), 1.0)

    def test_liga_sobrescreve_global(self):
        v = _resolve_scoped_param(1.0, {"LCK": 2.0}, {}, "LCK", "14.1")
        self.assertEqual(v, 2.0)

    def test_patch_mistura_65_35_por_cima(self):
        # global 1.0 (sem liga), patch 3.0 -> 0.65*1 + 0.35*3 = 1.70
        v = _resolve_scoped_param(1.0, {}, {"14.1": 3.0}, "LCK", "14.1")
        self.assertAlmostEqual(v, 1.70, places=9)

    def test_liga_e_patch_juntos(self):
        # liga leva a base pra 2.0, depois patch mistura: 0.65*2 + 0.35*4 = 2.70
        v = _resolve_scoped_param(1.0, {"LCK": 2.0}, {"14.1": 4.0}, "LCK", "14.1")
        self.assertAlmostEqual(v, 2.70, places=9)

    def test_chave_normaliza_maiuscula(self):
        # a chave do map é casada via _norm_upper_key
        v = _resolve_scoped_param(1.0, {"LCK": 2.0}, {}, "lck", "14.1")
        self.assertEqual(v, 2.0)


class ResolveEffectiveParamsTests(unittest.TestCase):
    """Resolve todos os params efetivos de um jogo via specs (global,by_lg,by_pt,lo,hi)."""

    def test_aplica_clip_quando_ha_limites(self):
        specs = {"x": (5.0, {}, {}, 0.0, 1.0)}  # global 5 mas clip [0,1]
        self.assertEqual(_resolve_effective_params(specs, "LCK", "14.1")["x"], 1.0)

    def test_sem_limites_nao_clipa(self):
        specs = {"x": (5.0, {}, {}, None, None)}
        self.assertEqual(_resolve_effective_params(specs, "LCK", "14.1")["x"], 5.0)

    def test_resolve_multiplos_de_uma_vez(self):
        specs = {
            "a": (1.0, {"LCK": 2.0}, {}, None, None),
            "b": (0.5, {}, {"14.1": 0.9}, 0.0, 1.0),
        }
        out = _resolve_effective_params(specs, "LCK", "14.1")
        self.assertEqual(out["a"], 2.0)
        self.assertAlmostEqual(out["b"], 0.65 * 0.5 + 0.35 * 0.9, places=9)


def _games_df(rows: list[tuple[str, str, str, str, int, str, str]]) -> pd.DataFrame:
    # gameid, date, blue, red, blue_win, league, patch
    out = []
    for gid, dt, blue, red, blue_win, league, patch_v in rows:
        out.append(
            {
                "gameid": gid,
                "date": dt,
                "blue": blue,
                "red": red,
                "blue_win": int(blue_win),
                "league": league,
                "patch": patch_v,
                "split": "S1",
                "playoffs": False,
            }
        )
    return pd.DataFrame(out)


class TestStateTeamArtifactIncremental(unittest.TestCase):
    def test_helper_date_and_source_guards_handle_malformed_inputs(self) -> None:
        self.assertEqual(_safe_iso_date("2026-01-02"), "2026-01-02")
        self.assertIsNone(_safe_iso_date(["2026-01-02"]))
        self.assertEqual(_days_between_iso("2026-01-01", "2026-01-03"), 2)
        self.assertEqual(_days_between_iso("2026-01-01", "bad-date"), 0)
        self.assertTrue(_same_source_path_order([{"path": "a.csv"}], [{"path": "a.csv"}]))
        self.assertFalse(_same_source_path_order([{"path": "a.csv"}], ["a.csv"]))  # type: ignore[list-item]

    def test_team_state_from_dict_sanitizes_invalid_numeric_fields(self) -> None:
        st = _team_state_from_dict(
            {
                "team": "Alpha",
                "gp": "nan",
                "wins": "inf",
                "losses": -3,
                "roster_cont_n": "bad",
                "qwl_n": object(),
                "last_lineup": {"Top": "player_top", "Mid": ""},
            }
        )
        self.assertIsNotNone(st)
        assert st is not None
        self.assertEqual(st.team, "Alpha")
        self.assertEqual(st.gp, 0)
        self.assertEqual(st.wins, 0)
        self.assertEqual(st.losses, 0)
        self.assertEqual(st.roster_cont_n, 0)
        self.assertEqual(st.qwl_n, 0)
        self.assertEqual(st.last_lineup, {"Top": "player_top"})

    def test_load_and_save_team_state_artifact_handle_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            art_path = os.path.join(td, "team_state_artifact.json")
            with open(art_path, "w", encoding="utf-8") as f:
                f.write("{invalid")
            self.assertEqual(load_team_state_artifact(art_path), {})

            artifact = {"meta": {"games_used": 3}, "teams": {"a": {"team": "A"}}}
            save_team_state_artifact(art_path, artifact)
            self.assertEqual(load_team_state_artifact(art_path), artifact)
            self.assertFalse(os.path.exists(art_path + ".tmp"))

    def test_file_sig_info_reports_real_file_stats(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            path = tf.name
            tf.write(b"abc")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

        sig = _file_sig_info(path)
        self.assertTrue(bool(sig.get("exists")))
        self.assertEqual(int(sig.get("size") or 0), 3)
        self.assertTrue(int(sig.get("mtime_ns") or 0) > 0)

    def test_incremental_seed_with_malformed_source_files_falls_back_cleanly(self) -> None:
        df_seed = _games_df([("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1")])
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            seed_art = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True)

        meta = seed_art.get("meta") or {}
        meta["source_files"] = 123
        seed_art["meta"] = meta

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True, seed_artifact=seed_art, incremental=True)

        rep_meta = rep.get("meta") or {}
        self.assertNotIn("error", rep)
        self.assertTrue(bool(rep_meta.get("seed_used")))
        self.assertEqual(int(rep_meta.get("games_used") or 0), 0)

    def test_incremental_append_only_uses_seed_and_skips_seen_games(self) -> None:
        df_seed = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1"),
                ("g2", "2026-01-02", "A", "C", 0, "LCK", "14.1"),
            ]
        )
        df_full = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1"),
                ("g2", "2026-01-02", "A", "C", 0, "LCK", "14.1"),
                ("g3", "2026-01-03", "B", "C", 1, "LCK", "14.1"),
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            seed_art = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True)
        self.assertEqual((seed_art.get("meta") or {}).get("games_used"), 2)
        self.assertTrue(isinstance(seed_art.get("processed_gameids"), list) and len(seed_art.get("processed_gameids")) == 2)

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_full), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ) as m_players:
            rep = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True, seed_artifact=seed_art, incremental=True)

        meta = rep.get("meta") or {}
        self.assertEqual(rep.get("version"), "team_state_artifact_v9")
        self.assertEqual(meta.get("update_mode"), "incremental_from_seed")
        self.assertTrue(bool(meta.get("seed_used")))
        self.assertTrue(bool(meta.get("incremental_fastpath")))
        self.assertEqual(int(meta.get("games_used") or 0), 1)
        self.assertEqual(int(meta.get("games_skipped_seen") or 0), 2)
        self.assertEqual(int(meta.get("candidate_new_games") or 0), 1)
        self.assertEqual(int(meta.get("processing_games") or 0), 1)
        self.assertTrue(m_players.called)
        gameids = m_players.call_args.kwargs.get("gameids")
        self.assertEqual(set(gameids or set()), {"g3"})
        self.assertIn("last_processed_game", meta)
        self.assertIn("online_journal_tail_count", meta)

    def test_backfill_unseen_older_game_forces_rebuild(self) -> None:
        df_seed = _games_df(
            [
                ("g1", "2026-01-02", "A", "B", 1, "LPL", "14.1"),
                ("g2", "2026-01-03", "A", "C", 0, "LPL", "14.1"),
            ]
        )
        df_with_backfill = _games_df(
            [
                ("g0", "2026-01-01", "B", "C", 1, "LPL", "14.1"),  # unseen older game
                ("g1", "2026-01-02", "A", "B", 1, "LPL", "14.1"),
                ("g2", "2026-01-03", "A", "C", 0, "LPL", "14.1"),
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            seed_art = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True)

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_with_backfill), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True, seed_artifact=seed_art, incremental=True)

        meta = rep.get("meta") or {}
        self.assertEqual(meta.get("update_mode"), "full_rebuild_backfill_detected")
        self.assertFalse(bool(meta.get("seed_used")))
        self.assertTrue(bool(meta.get("backfill_detected")))
        self.assertEqual(int(meta.get("backfill_unknown_games") or 0), 1)
        self.assertEqual(str(meta.get("backfill_oldest_date")), "2026-01-01")
        self.assertEqual(int(meta.get("games_used") or 0), 3)
        self.assertEqual(int(meta.get("games_skipped_seen") or 0), 0)
        self.assertFalse(bool(meta.get("incremental_fastpath")))

    def test_no_source_change_reuses_seed_without_reloading_csvs(self) -> None:
        df_seed = _games_df([("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1")])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            csv_path = tf.name
            tf.write(b"x")
        self.addCleanup(lambda: os.path.exists(csv_path) and os.remove(csv_path))

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            seed_art = build_team_state_artifact([csv_path], include_players_for_roster=True)

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df_seed) as m_games, patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ) as m_players:
            rep = build_team_state_artifact([csv_path], include_players_for_roster=True, seed_artifact=seed_art, incremental=True)

        meta = rep.get("meta") or {}
        self.assertEqual(str(meta.get("update_mode")), "incremental_no_source_change_seed_reused")
        self.assertEqual(str(meta.get("source_scan_mode")), "none_seed_reuse")
        self.assertEqual(int(meta.get("games_used") or 0), 0)
        self.assertFalse(m_games.called)
        self.assertFalse(m_players.called)

    def test_only_last_csv_changed_uses_last_csv_scan_mode(self) -> None:
        df_hist = _games_df([("h1", "2026-01-01", "H1", "H2", 1, "LEC", "14.1")])
        df_daily_seed = _games_df([("d1", "2026-01-02", "A", "B", 1, "LCK", "14.1")])
        df_daily_new = _games_df([("d1", "2026-01-02", "A", "B", 1, "LCK", "14.1"), ("d2", "2026-01-03", "A", "C", 0, "LCK", "14.1")])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf_hist, tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False
        ) as tf_daily:
            hist_path = tf_hist.name
            daily_path = tf_daily.name
            tf_hist.write(b"h")
            tf_daily.write(b"d")
        self.addCleanup(lambda: os.path.exists(hist_path) and os.remove(hist_path))
        self.addCleanup(lambda: os.path.exists(daily_path) and os.remove(daily_path))

        def _loader_seed(path: str, keep_playoffs=None):
            return df_hist if str(path) == str(hist_path) else df_daily_seed

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", side_effect=_loader_seed), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            seed_art = build_team_state_artifact([hist_path, daily_path], include_players_for_roster=True)

        # Change only daily file signature (append-like)
        with open(daily_path, "ab") as f:
            f.write(b"more")

        called_paths: list[str] = []

        def _loader_inc(path: str, keep_playoffs=None):
            called_paths.append(str(path))
            return df_daily_new if str(path) == str(daily_path) else df_hist

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", side_effect=_loader_inc), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(
                [hist_path, daily_path],
                include_players_for_roster=True,
                seed_artifact=seed_art,
                incremental=True,
            )

        meta = rep.get("meta") or {}
        self.assertEqual(str(meta.get("source_scan_mode")), "last_csv_only_incremental")
        self.assertEqual(int(meta.get("csv_scanned_count") or 0), 1)
        self.assertEqual(str(meta.get("lineup_source_scan_mode")), "last_csv_only_incremental")
        self.assertEqual(int(meta.get("lineup_csv_scanned_count") or 0), 1)
        self.assertEqual(list(meta.get("lineup_source_csvs_scanned") or []), [str(daily_path)])
        self.assertEqual(int(meta.get("games_used") or 0), 1)
        self.assertEqual([str(daily_path)], called_paths)

    def test_patch_switch_triggers_patch_mu_recenter_events(self) -> None:
        df = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1"),
                ("g2", "2026-01-02", "A", "B", 0, "LCK", "14.2"),
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(["dummy.csv"], include_players_for_roster=False)
        meta = rep.get("meta") or {}
        self.assertGreaterEqual(int(meta.get("patch_switch_events") or 0), 1)
        self.assertGreaterEqual(int(meta.get("patch_mu_recenter_events") or 0), 1)

    def test_roster_hard_reset_triggers_recenter_and_sigma_event(self) -> None:
        df = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LPL", "14.1"),
                ("g2", "2026-01-02", "A", "B", 0, "LPL", "14.1"),
            ]
        )

        def _lineup_side_effect(_pr: pd.DataFrame, gid: str):
            if str(gid) == "g1":
                return (
                    {"Top": "a_top", "Jungle": "a_jng", "Mid": "a_mid", "Bot": "a_bot", "Support": "a_sup"},
                    {"Top": "b_top", "Jungle": "b_jng", "Mid": "b_mid", "Bot": "b_bot", "Support": "b_sup"},
                )
            return (
                {"Top": "x_top", "Jungle": "x_jng", "Mid": "x_mid", "Bot": "x_bot", "Support": "x_sup"},
                {"Top": "y_top", "Jungle": "y_jng", "Mid": "y_mid", "Bot": "y_bot", "Support": "y_sup"},
            )

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame([{"gameid": "g1"}])
        ), patch("mlcore.state_team_artifact.lineup_for_game", side_effect=_lineup_side_effect):
            rep = build_team_state_artifact(["dummy.csv"], include_players_for_roster=True)
        meta = rep.get("meta") or {}
        self.assertGreaterEqual(int(meta.get("roster_events_with_change") or 0), 1)
        self.assertGreaterEqual(int(meta.get("roster_mu_recenter_events") or 0), 1)
        self.assertGreaterEqual(int(meta.get("roster_hard_reset_events") or 0), 1)

    def test_quality_of_result_boost_increases_state_update(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "gameid": "q1",
                    "date": "2026-01-01",
                    "blue": "A",
                    "red": "B",
                    "blue_win": 1,
                    "league": "LCK",
                    "patch": "14.1",
                    "split": "S1",
                    "playoffs": False,
                    "teamkills": None,
                    "blue_teamkills": 26.0,
                    "red_teamkills": 8.0,
                    "blue_dragons": 4.0,
                    "red_dragons": 0.0,
                    "blue_barons": 2.0,
                    "red_barons": 0.0,
                    "blue_towers": 10.0,
                    "red_towers": 2.0,
                    "blue_firstblood": 1,
                    "red_firstblood": 0,
                    "blue_firstdragon": 1,
                    "red_firstdragon": 0,
                    "blue_firstbaron": 1,
                    "red_firstbaron": 0,
                    "blue_firsttower": 1,
                    "red_firsttower": 0,
                    "blue_golddiffat10": 2100.0,
                    "red_golddiffat10": -2100.0,
                }
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_noq = build_team_state_artifact(["dummy.csv"], include_players_for_roster=False, quality_k_boost=0.0)
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_q = build_team_state_artifact(["dummy.csv"], include_players_for_roster=False, quality_k_boost=0.45)

        teams_noq = rep_noq.get("teams") or {}
        teams_q = rep_q.get("teams") or {}
        mu_a_noq = float((teams_noq.get("a") or {}).get("mu") or 0.0)
        mu_a_q = float((teams_q.get("a") or {}).get("mu") or 0.0)
        self.assertGreater(mu_a_q, mu_a_noq)
        meta_q = rep_q.get("meta") or {}
        self.assertGreaterEqual(int(meta_q.get("quality_signal_used_games") or 0), 1)
        self.assertGreaterEqual(int(meta_q.get("quality_adjust_used_games") or 0), 1)
        q_a = (teams_q.get("a") or {})
        q_b = (teams_q.get("b") or {})
        self.assertIn("qwl_signed_avg", q_a)
        self.assertIn("qwl_n", q_a)
        self.assertGreater(float(q_a.get("qwl_signed_avg") or 0.0), 0.0)
        self.assertLess(float(q_b.get("qwl_signed_avg") or 0.0), 0.0)
        self.assertGreaterEqual(int(q_a.get("qwl_n") or 0), 1)

    def test_qwl_opponent_adjustment_reduces_expected_favorite_overboost(self) -> None:
        rows = [
            {
                "gameid": "q0",
                "date": "2026-01-01",
                "blue": "A",
                "red": "B",
                "blue_win": 1,
                "league": "LCK",
                "patch": "14.1",
                "split": "S1",
                "playoffs": False,
            },
            {
                "gameid": "q1",
                "date": "2026-01-02",
                "blue": "A",
                "red": "B",
                "blue_win": 1,
                "league": "LCK",
                "patch": "14.1",
                "split": "S1",
                "playoffs": False,
            },
            {
                "gameid": "q2",
                "date": "2026-01-03",
                "blue": "A",
                "red": "B",
                "blue_win": 1,
                "league": "LCK",
                "patch": "14.1",
                "split": "S1",
                "playoffs": False,
            },
            {
                "gameid": "q3",
                "date": "2026-01-04",
                "blue": "A",
                "red": "B",
                "blue_win": 1,
                "league": "LCK",
                "patch": "14.1",
                "split": "S1",
                "playoffs": False,
                "teamkills": None,
                "blue_teamkills": 24.0,
                "red_teamkills": 7.0,
                "blue_dragons": 4.0,
                "red_dragons": 0.0,
                "blue_barons": 2.0,
                "red_barons": 0.0,
                "blue_towers": 10.0,
                "red_towers": 2.0,
                "blue_firstblood": 1,
                "red_firstblood": 0,
                "blue_firstdragon": 1,
                "red_firstdragon": 0,
                "blue_firstbaron": 1,
                "red_firstbaron": 0,
                "blue_firsttower": 1,
                "red_firsttower": 0,
                "blue_golddiffat10": 1900.0,
                "red_golddiffat10": -1900.0,
            },
        ]
        df = pd.DataFrame(rows)

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_mix0 = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_surplus_mix=0.0,
            )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_mix1 = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_surplus_mix=0.95,
            )

        mu_a_mix0 = float((((rep_mix0.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        mu_a_mix1 = float((((rep_mix1.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        self.assertGreater(mu_a_mix0, mu_a_mix1)
        meta_mix1 = rep_mix1.get("meta") or {}
        self.assertGreaterEqual(int(meta_mix1.get("quality_adjust_used_games") or 0), 1)
        self.assertGreaterEqual(int(meta_mix1.get("quality_adjust_surplus_games") or 0), 1)

    def test_qwl_causal_v2_terms_penalize_throw_and_reward_closeout(self) -> None:
        row_win = {
            "gameid": "qc_win",
            "date": "2026-01-01",
            "blue": "A",
            "red": "B",
            "blue_win": 1,
            "league": "LCK",
            "patch": "14.1",
            "split": "S1",
            "playoffs": False,
            "teamkills": None,
            "blue_teamkills": 21.0,
            "red_teamkills": 10.0,
            "blue_dragons": 3.0,
            "red_dragons": 1.0,
            "blue_barons": 2.0,
            "red_barons": 0.0,
            "blue_towers": 9.0,
            "red_towers": 3.0,
            "blue_firstblood": 1,
            "red_firstblood": 0,
            "blue_firstdragon": 1,
            "red_firstdragon": 0,
            "blue_firstbaron": 1,
            "red_firstbaron": 0,
            "blue_firsttower": 1,
            "red_firsttower": 0,
            "blue_golddiffat10": 2200.0,
            "red_golddiffat10": -2200.0,
            "gamelength": 1820.0,
        }
        row_throw = dict(row_win)
        row_throw.update(
            {
                "gameid": "qc_throw",
                "date": "2026-01-02",
                "blue_win": 0,
                "blue_dragons": 1.0,
                "red_dragons": 3.0,
                "blue_barons": 0.0,
                "red_barons": 2.0,
                "blue_towers": 4.0,
                "red_towers": 9.0,
                "blue_firstbaron": 0,
                "red_firstbaron": 1,
                "blue_teamkills": 12.0,
                "red_teamkills": 20.0,
                "gamelength": 1740.0,
            }
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row_win])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_win = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row_throw])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_throw = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )

        j_win = (rep_win.get("online_journal_tail") or [{}])[-1]
        j_throw = (rep_throw.get("online_journal_tail") or [{}])[-1]
        self.assertGreater(float(j_win.get("q_conversion_term") or 0.0), 0.0)
        self.assertLess(float(j_throw.get("q_conversion_term") or 0.0), 0.0)
        self.assertGreater(float(j_win.get("q_tempo_term") or 0.0), 0.0)
        self.assertLess(float(j_throw.get("q_tempo_term") or 0.0), 0.0)

        mu_a_win = float((((rep_win.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        mu_a_throw = float((((rep_throw.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        self.assertGreater(mu_a_win, mu_a_throw)

        meta_throw = rep_throw.get("meta") or {}
        self.assertGreaterEqual(int(meta_throw.get("quality_tempo_used_games") or 0), 1)
        self.assertGreaterEqual(int(meta_throw.get("quality_conversion_used_games") or 0), 1)
        self.assertGreaterEqual(int(meta_throw.get("quality_throw_used_games") or 0), 1)

    def test_qwl_early15_and_xp_sources_are_tracked_in_journal(self) -> None:
        row_xp = {
            "gameid": "qxp",
            "date": "2026-01-01",
            "blue": "A",
            "red": "B",
            "blue_win": 1,
            "league": "LCK",
            "patch": "14.1",
            "split": "S1",
            "playoffs": False,
            "blue_teamkills": 16.0,
            "red_teamkills": 11.0,
            "blue_dragons": 3.0,
            "red_dragons": 2.0,
            "blue_barons": 1.0,
            "red_barons": 0.0,
            "blue_towers": 8.0,
            "red_towers": 4.0,
            "blue_firstblood": 1,
            "red_firstblood": 0,
            "blue_firstdragon": 1,
            "red_firstdragon": 0,
            "blue_firstbaron": 1,
            "red_firstbaron": 0,
            "blue_firsttower": 1,
            "red_firsttower": 0,
            # force lead fallback to xp15 (no gold diff columns)
            "blue_xpdiffat15": 1600.0,
            "red_xpdiffat15": -1600.0,
            "blue_csdiffat15": 24.0,
            "red_csdiffat15": -24.0,
            "gamelength": 1880.0,
        }

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row_xp])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.35,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )

        j = (rep.get("online_journal_tail") or [{}])[-1]
        self.assertIn("q_early_term", j)
        self.assertIn("q_objective_term", j)
        self.assertIn("q_kill_term", j)
        self.assertEqual(str(j.get("q_lead_source") or ""), "xp15")
        self.assertTrue(abs(float(j.get("q_early_term") or 0.0)) > 0.0)

    def test_qwl_datacompleteness_partial_reduces_update_strength(self) -> None:
        row_complete = {
            "gameid": "qdc_complete",
            "date": "2026-01-01",
            "blue": "A",
            "red": "B",
            "blue_win": 1,
            "league": "LCK",
            "patch": "14.1",
            "split": "S1",
            "playoffs": False,
            "blue_datacompleteness": "complete",
            "red_datacompleteness": "complete",
            "blue_teamkills": 24.0,
            "red_teamkills": 8.0,
            "blue_dragons": 4.0,
            "red_dragons": 1.0,
            "blue_barons": 2.0,
            "red_barons": 0.0,
            "blue_towers": 10.0,
            "red_towers": 2.0,
            "blue_firstblood": 1,
            "red_firstblood": 0,
            "blue_firstdragon": 1,
            "red_firstdragon": 0,
            "blue_firstbaron": 1,
            "red_firstbaron": 0,
            "blue_firsttower": 1,
            "red_firsttower": 0,
            "blue_golddiffat20": 3200.0,
            "red_golddiffat20": -3200.0,
            "gamelength": 1840.0,
        }
        row_partial = dict(row_complete)
        row_partial.update(
            {
                "gameid": "qdc_partial",
                "blue_datacompleteness": "partial",
                "red_datacompleteness": "partial",
            }
        )

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row_complete])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_complete = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row_partial])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_partial = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.45,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )

        mu_complete = float((((rep_complete.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        mu_partial = float((((rep_partial.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        self.assertGreater(mu_complete, mu_partial)

        meta_complete = rep_complete.get("meta") or {}
        meta_partial = rep_partial.get("meta") or {}
        self.assertEqual(int(meta_complete.get("quality_data_partial_games") or 0), 0)
        self.assertGreaterEqual(int(meta_partial.get("quality_data_partial_games") or 0), 1)
        self.assertGreater(
            float(meta_complete.get("quality_data_reliability_avg") or 0.0),
            float(meta_partial.get("quality_data_reliability_avg") or 0.0),
        )

    def test_qwl_uses_gold20_and_elemental_fallback_terms(self) -> None:
        row = {
            "gameid": "q20",
            "date": "2026-01-01",
            "blue": "A",
            "red": "B",
            "blue_win": 1,
            "league": "LCK",
            "patch": "14.1",
            "split": "S1",
            "playoffs": False,
            "blue_datacompleteness": "complete",
            "red_datacompleteness": "complete",
            "blue_teamkills": 20.0,
            "red_teamkills": 10.0,
            # no dragon columns to force elementaldrakes fallback
            "blue_elementaldrakes": 3.0,
            "red_elementaldrakes": 1.0,
            "blue_barons": 1.0,
            "red_barons": 0.0,
            "blue_towers": 9.0,
            "red_towers": 4.0,
            "blue_turretplates": 7.0,
            "red_turretplates": 3.0,
            "blue_firsttower": 1,
            "red_firsttower": 0,
            "blue_firstmidtower": 1,
            "red_firstmidtower": 0,
            "blue_firsttothreetowers": 1,
            "red_firsttothreetowers": 0,
            "blue_golddiffat10": 1000.0,
            "red_golddiffat10": -1000.0,
            "blue_golddiffat15": 1700.0,
            "red_golddiffat15": -1700.0,
            "blue_golddiffat20": 2600.0,
            "red_golddiffat20": -2600.0,
            "gamelength": 1980.0,
        }

        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=pd.DataFrame([row])), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                quality_k_boost=0.35,
                qwl_tempo_weight=0.15,
                qwl_conversion_weight=0.25,
            )

        j = (rep.get("online_journal_tail") or [{}])[-1]
        self.assertEqual(str(j.get("q_lead_source") or ""), "gold20")
        self.assertEqual(str(j.get("q_objective_dragon_source") or ""), "elementaldrakes")
        self.assertAlmostEqual(float(j.get("q_data_reliability") or 0.0), 1.0, places=6)
        self.assertEqual(str(j.get("q_data_completeness_blue") or ""), "complete")

    def test_bayes_sigma_update_shrinks_uncertainty_with_information(self) -> None:
        df = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1"),
                ("g2", "2026-01-02", "A", "B", 1, "LCK", "14.1"),
                ("g3", "2026-01-03", "A", "B", 0, "LCK", "14.1"),
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_bayes = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                mu_update_mode="legacy_k_v1",
                sigma_update_mode="bayes_logit_v1",
                bayes_info_scale=1.0,
            )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_legacy = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                mu_update_mode="legacy_k_v1",
                sigma_update_mode="legacy_decay_v1",
            )
        a_bayes = float(((rep_bayes.get("teams") or {}).get("a") or {}).get("sigma") or 10.0)
        a_legacy = float(((rep_legacy.get("teams") or {}).get("a") or {}).get("sigma") or 10.0)
        self.assertLess(a_bayes, a_legacy)
        params = rep_bayes.get("params") or {}
        self.assertEqual(str(params.get("sigma_update_mode")), "bayes_logit_v1")

    def test_mu_update_mode_bayes_joint_tracks_meta_and_differs_from_legacy(self) -> None:
        df = _games_df(
            [
                ("g1", "2026-01-01", "A", "B", 1, "LCK", "14.1"),
                ("g2", "2026-01-02", "A", "B", 0, "LCK", "14.1"),
                ("g3", "2026-01-03", "A", "B", 1, "LCK", "14.1"),
            ]
        )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_joint = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                mu_update_mode="bayes_joint_v1",
                sigma_update_mode="bayes_logit_v1",
            )
        with patch("mlcore.state_team_artifact.load_oracle_elixir_games", return_value=df), patch(
            "mlcore.state_team_artifact.load_match_player_rows", return_value=pd.DataFrame()
        ):
            rep_legacy = build_team_state_artifact(
                ["dummy.csv"],
                include_players_for_roster=False,
                mu_update_mode="legacy_k_v1",
                sigma_update_mode="bayes_logit_v1",
            )

        meta_joint = rep_joint.get("meta") or {}
        meta_legacy = rep_legacy.get("meta") or {}
        self.assertGreater(int(meta_joint.get("bayes_joint_updates") or 0), 0)
        self.assertEqual(int(meta_joint.get("legacy_k_updates") or 0), 0)
        self.assertGreater(int(meta_legacy.get("legacy_k_updates") or 0), 0)
        self.assertEqual(int(meta_legacy.get("bayes_joint_updates") or 0), 0)

        p_joint = rep_joint.get("params") or {}
        self.assertEqual(str(p_joint.get("mu_update_mode")), "bayes_joint_v1")
        self.assertTrue("bayes_mu_q_strength" in p_joint)

        mu_joint = float((((rep_joint.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        mu_legacy = float((((rep_legacy.get("teams") or {}).get("a") or {}).get("mu") or 0.0))
        self.assertNotAlmostEqual(mu_joint, mu_legacy, places=6)


if __name__ == "__main__":
    unittest.main()


