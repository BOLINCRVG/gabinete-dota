import unittest
import math

import pandas as pd

from app_core.resumo_core import make_pricing_adapters, build_ml_rows, build_metric_rows


class TestResumoCore(unittest.TestCase):
    def test_make_pricing_adapters(self):
        ad = make_pricing_adapters(
            laplace_pick="Liga 15",
            w_model=40.0,
            w_hist=40.0,
            w_stats=20.0,
            floor_odd=1.10,
        )
        self.assertIn("laplace_odds_fn", ad)
        self.assertIn("laplace_pick_odd_fn", ad)
        self.assertIn("odd_mix_fn", ad)
        self.assertIn("fmt_laplace_filtered_fn", ad)

        odd_lap = ad["laplace_odds_fn"](10, 10)
        self.assertAlmostEqual(float(odd_lap), 1.10, places=2)

        odds_map = {
            "Liga_15_Over": 1.70,
            "Liga_10_Over": 1.80,
            "H2H_15_Over": 1.90,
        }
        pick = ad["laplace_pick_odd_fn"](odds_map, "Over")
        self.assertAlmostEqual(float(pick), 1.70, places=2)

        mix = ad["odd_mix_fn"](1.80, 2.00, 60.0)
        self.assertIsNotNone(mix)
        self.assertGreater(float(mix), 1.0)

        # Clean, scope-correct, jargon-free format (no "mu recortes"/"pick=").
        txt = ad["fmt_laplace_filtered_fn"](odds_map, "Liga", "Over", 2.5)
        self.assertIn("méd ", str(txt))          # average of qualifying recortes
        self.assertNotIn("pick=", str(txt))      # no global-pick leak into the line
        self.assertNotIn("mu recortes", str(txt))

    def test_build_ml_rows_regression_shape_and_trace(self):
        ad = make_pricing_adapters(
            laplace_pick="Liga Ano",
            w_model=40.0,
            w_hist=40.0,
            w_stats=20.0,
            floor_odd=1.10,
        )

        df_league_year = pd.DataFrame({"win": [1, 1, 0, 1, 0, 1]})
        df_empty = pd.DataFrame({"win": []})

        def _odd_from_prob(p):
            try:
                pv = float(p)
            except Exception:
                return None
            if (not math.isfinite(pv)) or pv <= 0.0 or pv >= 1.0:
                return None
            return 1.0 / pv

        rows = build_ml_rows(
            p_ml_row=0.60,
            team_a="TEAM_A",
            team_b="TEAM_B",
            req_liga=False,
            req_h2h=False,
            req_model=True,
            min_sample_liga=0,
            min_sample_h2h=0,
            max_odd=3.0,
            df_league_year=df_league_year,
            df_league_15=df_league_year,
            df_league_10=df_league_year,
            df_league_5=df_league_year,
            df_h2h_year=df_empty,
            df_h2h_15=df_empty,
            df_h2h_10=df_empty,
            df_h2h_5=df_empty,
            odd_from_prob_fn=_odd_from_prob,
            laplace_odds_fn=ad["laplace_odds_fn"],
            laplace_pick_odd_fn=ad["laplace_pick_odd_fn"],
            odd_mix_fn=ad["odd_mix_fn"],
            fmt_laplace_filtered_fn=ad["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=lambda xs: ",".join(xs),
        )
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r["Mercado"], "ML (Mapa)")
            self.assertIn("_calib_reason", r)
            self.assertIn("_calib_source", r)
            self.assertIn(r["_calib_source"], ["mix", "model", "hist"])
            self.assertIsNotNone(r["Odd (Mix)"])

    def test_build_metric_rows_hist_fallback_without_model(self):
        ad = make_pricing_adapters(
            laplace_pick="Liga Ano",
            w_model=40.0,
            w_hist=40.0,
            w_stats=20.0,
            floor_odd=1.10,
        )
        df_league_year = pd.DataFrame({"total_kills": [24, 26, 30, 21, 28, 33]})
        df_empty = pd.DataFrame({"total_kills": []})

        rows = build_metric_rows(
            lines_by_metric={"kills": [25.5]},
            ensure_lines_fn=lambda _m, ls: ls,
            metric_col_map={"kills": "total_kills"},
            totals_for_lines={},
            total_over_prob_fn=lambda _tot, _line: 0.5,
            odd_from_prob_fn=lambda p: 1.0 / float(p) if p and 0.0 < float(p) < 1.0 else None,
            req_liga=False,
            req_h2h=False,
            req_model=False,
            min_sample_liga=0,
            min_sample_h2h=0,
            max_odd=3.0,
            df_league_year=df_league_year,
            df_league_15=df_league_year,
            df_league_10=df_league_year,
            df_league_5=df_league_year,
            df_h2h_year=df_empty,
            df_h2h_15=df_empty,
            df_h2h_10=df_empty,
            df_h2h_5=df_empty,
            laplace_odds_fn=ad["laplace_odds_fn"],
            laplace_pick_odd_fn=ad["laplace_pick_odd_fn"],
            odd_mix_fn=ad["odd_mix_fn"],
            fmt_laplace_filtered_fn=ad["fmt_laplace_filtered_fn"],
            fmt_recortes_disp_fn=lambda xs: ",".join(xs),
            fmt_line_disp_fn=lambda _m, ln: f"{float(ln):.1f}",
            metric_label_fn=lambda _m: "Kills",
            side_label_fn=lambda s: "Acima" if str(s) == "Over" else "Abaixo",
            sc_badge_fn=lambda _m, _s, _ln: "-",
            sc_pct_fn=lambda _m, _s, _ln: None,
        )
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("_calib_reason", r)
            self.assertIn("_calib_source", r)
            self.assertIn(r["_calib_source"], ["mix", "hist"])
            self.assertIsNotNone(r["Odd (Mix)"])


if __name__ == "__main__":
    unittest.main()
