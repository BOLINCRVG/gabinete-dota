import math
import unittest

import pandas as pd

from app_core.players_core import build_player_market_rows, build_player_means_rows
from app_core.resumo_core import (
    build_combo_rows,
    build_metric_rows,
    build_ml_rows,
    finalize_resumo_dataframe,
    make_pricing_adapters,
    wr_laplace,
    counts_ml_totals,
)


def _odd_from_prob(p):
    try:
        pv = float(p)
    except Exception:
        return None
    if (not math.isfinite(pv)) or pv <= 0.0 or pv >= 1.0:
        return None
    return 1.0 / pv


def _fmt_num(v, nd=2):
    try:
        return f"{float(v):.{int(nd)}f}"
    except Exception:
        return "-"


class TestE2ESmokeFlow(unittest.TestCase):
    def test_resumo_and_players_core_flow(self):
        pricing = make_pricing_adapters(
            laplace_pick="Liga Ano",
            w_model=40.0,
            w_hist=40.0,
            w_stats=20.0,
            floor_odd=1.10,
        )

        # ---------- Resumo core flow ----------
        df_league_year = pd.DataFrame(
            {
                "win": [1, 1, 0, 1, 0, 1, 1, 0],
                "total_kills": [24, 26, 30, 21, 28, 33, 27, 25],
            }
        )
        df_h2h_year = pd.DataFrame({"win": [1, 0], "total_kills": [27, 29]})
        df_empty = pd.DataFrame({"win": [], "total_kills": []})

        rows = []
        rows.extend(
            build_ml_rows(
                p_ml_row=0.62,
                team_a="TEAM_A",
                team_b="TEAM_B",
                req_liga=False,
                req_h2h=False,
                req_model=True,
                min_sample_liga=0,
                min_sample_h2h=0,
                max_odd=5.0,
                df_league_year=df_league_year,
                df_league_15=df_league_year,
                df_league_10=df_league_year,
                df_league_5=df_league_year,
                df_h2h_year=df_h2h_year,
                df_h2h_15=df_h2h_year,
                df_h2h_10=df_empty,
                df_h2h_5=df_empty,
                odd_from_prob_fn=_odd_from_prob,
                laplace_odds_fn=pricing["laplace_odds_fn"],
                laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
                odd_mix_fn=pricing["odd_mix_fn"],
                fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
                fmt_recortes_disp_fn=lambda xs: ",".join(xs),
            )
        )

        rows.extend(
            build_metric_rows(
                lines_by_metric={"kills": [26.5]},
                ensure_lines_fn=lambda _m, ls: ls,
                metric_col_map={"kills": "total_kills"},
                totals_for_lines={"kills": object()},
                total_over_prob_fn=lambda _tot, _line: 0.56,
                odd_from_prob_fn=_odd_from_prob,
                req_liga=False,
                req_h2h=False,
                req_model=True,
                min_sample_liga=0,
                min_sample_h2h=0,
                max_odd=5.0,
                df_league_year=df_league_year,
                df_league_15=df_league_year,
                df_league_10=df_league_year,
                df_league_5=df_league_year,
                df_h2h_year=df_h2h_year,
                df_h2h_15=df_h2h_year,
                df_h2h_10=df_empty,
                df_h2h_5=df_empty,
                laplace_odds_fn=pricing["laplace_odds_fn"],
                laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
                odd_mix_fn=pricing["odd_mix_fn"],
                fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
                fmt_recortes_disp_fn=lambda xs: ",".join(xs),
                fmt_line_disp_fn=lambda _m, ln: f"{float(ln):.1f}",
                metric_label_fn=lambda _m: "Kills",
                side_label_fn=lambda s: "Acima" if str(s) == "Over" else "Abaixo",
                sc_badge_fn=lambda _m, _s, _ln: "-",
                sc_pct_fn=lambda _m, _s, _ln: 55.0,
            )
        )

        dfA = pd.DataFrame({"win": [1, 1, 0, 1], "total_kills": [24, 28, 27, 30]})
        dfB = pd.DataFrame({"win": [0, 0, 1, 0], "total_kills": [31, 29, 26, 30]})

        def _combo_bases(side_key: str):
            base = dfA if side_key == "A" else dfB
            return {
                "l_year": base,
                "l_5": base,
                "l_10": base,
                "l_15": base,
                "h_year": df_h2h_year,
                "h_5": df_h2h_year,
                "h_10": df_h2h_year,
                "h_15": df_h2h_year,
            }

        rows.extend(
            build_combo_rows(
                combo_lines_by_metric={"kills": [28.5]},
                metric_col_map={"kills": "total_kills"},
                p_ml_combo=0.62,
                team_a="TEAM_A",
                team_b="TEAM_B",
                combo_hist_w=0.4,
                combo_style_w=0.2,
                combo_form_w=0.2,
                combo_use_mismatch=True,
                req_liga=False,
                req_h2h=False,
                req_model=True,
                min_sample_liga=0,
                min_sample_h2h=0,
                max_odd=8.0,
                get_combo_bases_fn=_combo_bases,
                wr_laplace_fn=wr_laplace,
                counts_ml_totals_fn=counts_ml_totals,
                totals_for_lines={"kills": object()},
                total_over_prob_fn=lambda _tot, _line: 0.53,
                odd_from_prob_fn=_odd_from_prob,
                laplace_odds_fn=pricing["laplace_odds_fn"],
                laplace_pick_odd_fn=pricing["laplace_pick_odd_fn"],
                odd_mix_fn=lambda om, oh, sc, mb: pricing["odd_mix_fn"](om, oh, sc, model_boost=mb),
                fmt_line_disp_fn=lambda _m, ln: f"{float(ln):.1f}",
                metric_label_fn=lambda _m: "Kills",
                fmt_recortes_disp_fn=lambda xs: ",".join(xs),
                fmt_laplace_filtered_fn=pricing["fmt_laplace_filtered_fn"],
                sc_badge_fn=lambda _m, _s, _ln: "-",
                sc_pct_fn=lambda _m, _s, _ln: 52.0,
            )
        )

        self.assertGreater(len(rows), 0)
        self.assertTrue(any("Mercado" in r for r in rows))
        self.assertTrue(any("_calib_reason" in r for r in rows))

        df_resumo = finalize_resumo_dataframe(pd.DataFrame(rows), format_num_fn=lambda v, d: _fmt_num(v, d))
        self.assertFalse(df_resumo.empty)
        self.assertIn("Mercado", df_resumo.columns)
        self.assertIn("Odd (Mix)", df_resumo.columns)

        # ---------- Players core flow ----------
        df_players = pd.DataFrame(
            [
                {"player": "A_TOP", "pos": "top", "kills": 2, "deaths": 3, "assists": 4},
                {"player": "A_TOP", "pos": "top", "kills": 3, "deaths": 2, "assists": 5},
                {"player": "B_TOP", "pos": "top", "kills": 5, "deaths": 2, "assists": 3},
                {"player": "B_TOP", "pos": "top", "kills": 4, "deaths": 3, "assists": 2},
            ]
        )
        lane_rows = [{"Lane": "TOP", "Pos": "top", "A": "A_TOP", "B": "B_TOP"}]
        means = build_player_means_rows(
            dfb=df_players,
            lane_rows=lane_rows,
            team_a="TEAM_A",
            team_b="TEAM_B",
            format_num_fn=_fmt_num,
        )
        self.assertEqual(len(means), 2)

        out_rows, fair_rows = build_player_market_rows(
            dfb=df_players,
            lane_rows=lane_rows,
            per_player_lines_by_idx={0: {"A": {"Kills": [2.5], "Deaths": [], "Assists": []}, "B": {"Kills": [4.5], "Deaths": [], "Assists": []}}},
            team_a="TEAM_A",
            team_b="TEAM_B",
            wf_m=40.0,
            wf_h=40.0,
            wf_s=20.0,
            odd_from_prob_fn=_odd_from_prob,
            format_num_fn=_fmt_num,
            format_pct_fn=lambda p: f"{100.0 * float(p):.1f}%",
            max_odd=5.0,
            require_model=False,
        )
        self.assertEqual(len(fair_rows), 2)
        self.assertEqual(len(out_rows), 4)
        self.assertTrue(all("_calib_reason" in r for r in out_rows))
        self.assertTrue(all("_calib_source" in r for r in out_rows))


if __name__ == "__main__":
    unittest.main()

