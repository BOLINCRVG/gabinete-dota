import unittest

import pandas as pd

from app_core.players_core import (
    build_player_market_rows,
    build_player_means_rows,
    build_recent_player_stats_rows,
)


def _odd_from_prob(p: float):
    try:
        pv = float(p)
    except Exception:
        return None
    if pv <= 0.0 or pv >= 1.0:
        return None
    return 1.0 / pv


def _fmt_num(v, nd=2):
    try:
        return f"{float(v):.{int(nd)}f}"
    except Exception:
        return "-"


def _fmt_pct(p):
    try:
        return f"{100.0 * float(p):.1f}%"
    except Exception:
        return "-"


class TestPlayersCore(unittest.TestCase):
    def setUp(self):
        # 4 jogos por player, mesma lane para manter teste determinístico.
        self.dfb = pd.DataFrame(
            [
                {"player": "A_TOP", "pos": "top", "kills": 2, "deaths": 3, "assists": 4},
                {"player": "A_TOP", "pos": "top", "kills": 3, "deaths": 2, "assists": 5},
                {"player": "A_TOP", "pos": "top", "kills": 1, "deaths": 4, "assists": 6},
                {"player": "A_TOP", "pos": "top", "kills": 4, "deaths": 1, "assists": 7},
                {"player": "B_TOP", "pos": "top", "kills": 5, "deaths": 2, "assists": 3},
                {"player": "B_TOP", "pos": "top", "kills": 4, "deaths": 3, "assists": 2},
                {"player": "B_TOP", "pos": "top", "kills": 6, "deaths": 1, "assists": 4},
                {"player": "B_TOP", "pos": "top", "kills": 3, "deaths": 4, "assists": 5},
            ]
        )
        self.lane_rows = [{"Lane": "TOP", "Pos": "top", "A": "A_TOP", "B": "B_TOP"}]
        self.dfb_h2h = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "player": "A_TOP", "pos": "top", "side": "Blue", "team": "TA", "champ": "Gnar", "result": 1, "kills": 2, "deaths": 3, "assists": 4, "gamelength": 1800, "teamkills": 10, "damagetochampions": 10000},
                {"gameid": "g1", "date": "2026-01-01", "player": "B_TOP", "pos": "top", "side": "Red", "team": "TB", "champ": "Jax", "result": 0, "kills": 5, "deaths": 2, "assists": 3, "gamelength": 1800, "teamkills": 12, "damagetochampions": 12000},
                {"gameid": "g2", "date": "2026-01-05", "player": "A_TOP", "pos": "top", "side": "Blue", "team": "TA", "champ": "Gnar", "result": 1, "kills": 3, "deaths": 2, "assists": 5, "gamelength": 1850, "teamkills": 11, "damagetochampions": 11000},
                {"gameid": "g2", "date": "2026-01-05", "player": "B_TOP", "pos": "top", "side": "Red", "team": "TB", "champ": "Jax", "result": 0, "kills": 4, "deaths": 3, "assists": 2, "gamelength": 1850, "teamkills": 13, "damagetochampions": 13000},
                {"gameid": "g3", "date": "2026-01-10", "player": "A_TOP", "pos": "top", "side": "Blue", "team": "TA", "champ": "Gnar", "result": 0, "kills": 1, "deaths": 4, "assists": 6, "gamelength": 1900, "teamkills": 9, "damagetochampions": 9000},
                {"gameid": "g3", "date": "2026-01-10", "player": "B_TOP", "pos": "top", "side": "Red", "team": "TB", "champ": "Jax", "result": 1, "kills": 6, "deaths": 1, "assists": 4, "gamelength": 1900, "teamkills": 14, "damagetochampions": 14000},
            ]
        )

    def test_build_player_means_rows(self):
        rows = build_player_means_rows(
            dfb=self.dfb,
            lane_rows=self.lane_rows,
            team_a="TEAM_A",
            team_b="TEAM_B",
            format_num_fn=_fmt_num,
        )
        self.assertEqual(len(rows), 2)
        players = sorted(r["Player"] for r in rows)
        self.assertEqual(players, ["A_TOP", "B_TOP"])
        for r in rows:
            self.assertEqual(r["Amostra"], 4)
            self.assertNotEqual(r["Kills (media)"], "-")
            self.assertNotEqual(r["Assists (sd)"], "-")

    def test_build_player_market_rows(self):
        per_player_lines = {
            0: {
                "A": {"Kills": [2.5], "Deaths": [], "Assists": []},
                "B": {"Kills": [4.5], "Deaths": [], "Assists": []},
            }
        }
        out_rows, fair_rows = build_player_market_rows(
            dfb=self.dfb,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TEAM_A",
            team_b="TEAM_B",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
        )
        # 2 players * 1 linha * (Acima/Abaixo)
        self.assertEqual(len(out_rows), 4)
        self.assertEqual(len(fair_rows), 2)
        lados = sorted(set(r["Lado"] for r in out_rows))
        self.assertEqual(lados, ["Abaixo", "Acima"])
        for r in out_rows:
            self.assertNotEqual(r["Odd (Modelo)"], "-")
            self.assertIn(r["Player"], ["A_TOP", "B_TOP"])
            self.assertIn("Obs Odds", r)
            self.assertTrue("H2H" in str(r["Obs Odds"]))
        for r in fair_rows:
            self.assertIn("%", r["Acima (%)"])
            self.assertNotEqual(r["Odd Abaixo"], "-")

    def test_build_player_means_rows_empty_sample(self):
        lane_rows = [{"Lane": "TOP", "Pos": "top", "A": "NOPE_A", "B": "NOPE_B"}]
        rows = build_player_means_rows(
            dfb=self.dfb,
            lane_rows=lane_rows,
            team_a="TEAM_A",
            team_b="TEAM_B",
            format_num_fn=_fmt_num,
        )
        self.assertEqual(rows, [])

    def test_build_player_market_rows_invalid_lines_ignored(self):
        per_player_lines = {
            0: {
                "A": {"Kills": ["abc", float("nan"), float("inf")], "Deaths": [], "Assists": []},
                "B": {"Kills": [None], "Deaths": [], "Assists": []},
            }
        }
        out_rows, fair_rows = build_player_market_rows(
            dfb=self.dfb,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TEAM_A",
            team_b="TEAM_B",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
        )
        self.assertEqual(out_rows, [])
        self.assertEqual(fair_rows, [])

    def test_build_player_market_rows_sd_floor_and_zero_weights(self):
        # sd=0 para kills (tudo igual), deve aplicar piso 0.35 internamente.
        df_flat = pd.DataFrame(
            [
                {"player": "A_TOP", "pos": "top", "kills": 2, "deaths": 1, "assists": 1},
                {"player": "A_TOP", "pos": "top", "kills": 2, "deaths": 2, "assists": 2},
                {"player": "B_TOP", "pos": "top", "kills": 3, "deaths": 1, "assists": 1},
                {"player": "B_TOP", "pos": "top", "kills": 3, "deaths": 2, "assists": 2},
            ]
        )
        per_player_lines = {
            0: {
                "A": {"Kills": [2.5], "Deaths": [], "Assists": []},
                "B": {"Kills": [2.5], "Deaths": [], "Assists": []},
            }
        }
        out_rows, fair_rows = build_player_market_rows(
            dfb=df_flat,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TEAM_A",
            team_b="TEAM_B",
            wf_m=0.0,
            wf_h=0.0,
            wf_s=0.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
        )
        self.assertGreater(len(out_rows), 0)
        self.assertGreater(len(fair_rows), 0)
        # Fallback de pesos: Odd (Mix) continua disponível.
        for r in out_rows:
            self.assertNotEqual(r["Odd (Mix)"], "-")
        # SD com piso deve aparecer como 0.35 em pelo menos uma linha.
        self.assertTrue(any(r.get("SD") == "0.35" for r in fair_rows))

    def test_build_player_market_rows_trace_and_max_odd_gate(self):
        per_player_lines = {
            0: {
                "A": {"Kills": [2.5], "Deaths": [], "Assists": []},
                "B": {"Kills": [4.5], "Deaths": [], "Assists": []},
            }
        }
        out_rows, fair_rows = build_player_market_rows(
            dfb=self.dfb,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TEAM_A",
            team_b="TEAM_B",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
            max_odd=1.20,  # força vários cenários de fallback/bloqueio
            require_model=False,
        )
        self.assertEqual(len(fair_rows), 2)
        self.assertEqual(len(out_rows), 4)
        for r in out_rows:
            self.assertIn("_calib_reason", r)
            self.assertIn("_calib_source", r)
            self.assertIn("_odd_model_raw", r)
            self.assertIn("_odd_hist_raw", r)
            self.assertIn("_odd_mix_raw", r)
            self.assertIn(r["_calib_reason"], ["mix", "fallback_model", "fallback_hist", "no_valid_odd_under_max"])
            self.assertIn(r["_calib_source"], ["mix", "model", "hist", "none"])

    def test_build_player_market_rows_h2h_used_when_sample_meets_min(self):
        per_player_lines = {0: {"A": {"Kills": [2.5], "Deaths": [], "Assists": []}, "B": {"Kills": [4.5], "Deaths": [], "Assists": []}}}
        out_rows, _ = build_player_market_rows(
            dfb=self.dfb_h2h,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TA",
            team_b="TB",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
            min_h2h_games=3,
        )
        self.assertEqual(len(out_rows), 4)
        self.assertTrue(all(r.get("_h2h_used") is True for r in out_rows))
        self.assertTrue(all(r.get("H2H") != "-" for r in out_rows))
        self.assertTrue(any("H2H considerado no Mix" in str(r.get("Obs Mix", "")) for r in out_rows))

    def test_build_player_market_rows_h2h_excluded_when_sample_below_min(self):
        per_player_lines = {0: {"A": {"Kills": [2.5], "Deaths": [], "Assists": []}, "B": {"Kills": [4.5], "Deaths": [], "Assists": []}}}
        out_rows, _ = build_player_market_rows(
            dfb=self.dfb_h2h,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TA",
            team_b="TB",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
            min_h2h_games=4,
        )
        self.assertEqual(len(out_rows), 4)
        self.assertTrue(all(r.get("_h2h_used") is False for r in out_rows))
        self.assertTrue(any("H2H nao considerado no Mix" in str(r.get("Obs Mix", "")) for r in out_rows))

    def test_build_player_market_rows_obs_mix_explains_none(self):
        per_player_lines = {0: {"A": {"Kills": [2.5], "Deaths": [], "Assists": []}, "B": {"Kills": [4.5], "Deaths": [], "Assists": []}}}
        out_rows, _ = build_player_market_rows(
            dfb=self.dfb_h2h,
            lane_rows=self.lane_rows,
            per_player_lines_by_idx=per_player_lines,
            team_a="TA",
            team_b="TB",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=_fmt_pct,
            max_odd=1.01,
            require_model=False,
            min_h2h_games=3,
        )
        self.assertEqual(len(out_rows), 4)
        self.assertTrue(any("Odd Mix indisponivel" in str(r.get("Obs Mix", "")) for r in out_rows))

    def test_build_recent_player_stats_rows(self):
        rows = build_recent_player_stats_rows(
            dfb=self.dfb_h2h,
            lane_rows=self.lane_rows,
            team_a="TA",
            team_b="TB",
            format_num_fn=_fmt_num,
            max_games_per_player=2,
        )
        self.assertGreaterEqual(len(rows), 2)
        r0 = rows[0]
        for col in ["Lane", "Time", "Player", "Campeão", "Resultado", "Tempo", "K/D/A", "KDA", "KP%", "Dano", "Oponente", "Data"]:
            self.assertIn(col, r0)


if __name__ == "__main__":
    unittest.main()
