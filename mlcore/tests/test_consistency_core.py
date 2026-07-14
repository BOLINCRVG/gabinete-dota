import unittest

import pandas as pd

from app_core.consistency_core import (
    panel_title_with_suffix,
    fmt_metric_value,
    team_metric_stats,
    last_rows_metric,
    build_overall_tables,
    h2h_numeric_series,
    recent_rows_with_hc_markers,
    h2h_rows_verbose,
    has_metric_history,
    normalize_p_ml_scope,
    build_ml_totals_model_payload,
    build_ml_totals_laplace_payload,
    build_audit_base_options,
    build_audit_market_options,
    build_audit_view,
    build_simple_audit_view,
    build_quick_metric_options,
    quick_team_frame,
    quick_outcome_probs,
    build_quick_selector_rows,
)


class TestConsistencyCore(unittest.TestCase):
    def test_panel_title_with_suffix(self):
        self.assertEqual(panel_title_with_suffix("Kills"), "Kills (combined)")
        self.assertEqual(panel_title_with_suffix("Kills (combined)"), "Kills (combined)")

    def test_team_metric_stats_and_h2h(self):
        df = pd.DataFrame({"date": [1, 2], "opponent": ["A", "B"], "total_kills": [25, 35]})
        st = team_metric_stats(df, "total_kills")
        self.assertEqual(st["games"], 2)
        self.assertAlmostEqual(st["avg"], 30.0, places=6)
        h = h2h_numeric_series(df, "total_kills")
        self.assertEqual(int(h.shape[0]), 2)

    def test_last_rows_metric(self):
        df = pd.DataFrame(
            {
                "date": [1, 2, 3],
                "opponent": ["A", "B", "C"],
                "league": ["Cup 1", "Cup 2", "Cup 3"],
                "total_kills": [20, 21, 22],
            }
        )
        rows = last_rows_metric(df, "total_kills", 2, metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0].startswith("B - "))
        self.assertTrue(rows[1].startswith("C - "))
        self.assertIn("Cup 2", rows[0])
        self.assertIn("Cup 3", rows[1])

    def test_recent_rows_with_hc_markers_flips_line_for_other_team(self):
        df = pd.DataFrame(
            {
                "date": [1, 2],
                "opponent": ["B", "C"],
                "league": ["LPL", "LPL"],
                "kills_for": [12, 8],
                "kills_against": [10, 10],
            }
        )
        rows_a, lines_a = recent_rows_with_hc_markers(
            df,
            shown_team="A",
            hc_team_selected="A",
            hc_lines=[1.5],
            fallback_col_name="kills_for",
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
            n=10,
        )
        rows_b, lines_b = recent_rows_with_hc_markers(
            df,
            shown_team="B",
            hc_team_selected="A",
            hc_lines=[1.5],
            fallback_col_name="kills_for",
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
            n=10,
        )
        self.assertEqual(lines_a, [1.5])
        self.assertEqual(lines_b, [-1.5])
        self.assertIn("HC: V", rows_a[0])
        self.assertTrue(any("HC: X" in row for row in rows_b))

    def test_h2h_rows_verbose_and_history_detection(self):
        df = pd.DataFrame(
            {
                "date": [1],
                "opponent": ["B"],
                "league": ["LPL"],
                "total_dragons": [6],
                "dragons_for": [4],
                "dragons_against": [2],
                "kills_for": [18],
                "kills_against": [11],
            }
        )
        total_rows = h2h_rows_verbose(
            df,
            metric_col="total_dragons",
            team_a="A",
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
            n=10,
        )
        kill_rows = h2h_rows_verbose(
            df,
            metric_col="kills_for",
            team_a="A",
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
            n=10,
        )
        self.assertIn("total=6", total_rows[0])
        self.assertIn("A=4", total_rows[0])
        self.assertIn("A=18", kill_rows[0])
        self.assertTrue(has_metric_history(df, metric_col="total_dragons"))
        self.assertFalse(has_metric_history(pd.DataFrame({"x": [1]}), metric_col="total_dragons"))

    def test_build_overall_tables_shape(self):
        s = {"games": 10, "avg": 30.0, "med": 29.0, "min": 20.0, "max": 40.0}
        cur, all_df = build_overall_tables(
            "A",
            "B",
            "total_kills",
            s,
            s,
            s,
            s,
            num_formatter=lambda v, d: str(int(v)),
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
        )
        self.assertEqual(list(cur.columns), ["Métrica", "A", "B"])
        self.assertEqual(list(all_df.columns), ["Métrica", "A", "B"])
        self.assertEqual(int(cur.shape[0]), 5)
        self.assertEqual(int(all_df.shape[0]), 5)

    def test_build_overall_tables_deduplicates_empty_or_equal_team_labels(self):
        s = {"games": 10, "avg": 30.0, "med": 29.0, "min": 20.0, "max": 40.0}
        cur_empty, _ = build_overall_tables(
            "",
            "",
            "total_kills",
            s,
            s,
            s,
            s,
            num_formatter=lambda v, d: str(int(v)),
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
        )
        self.assertEqual(list(cur_empty.columns), ["Métrica", "Time A", "Time B"])

        cur_same, _ = build_overall_tables(
            "Team X",
            "Team X",
            "total_kills",
            s,
            s,
            s,
            s,
            num_formatter=lambda v, d: str(int(v)),
            metric_formatter=lambda c, v, d: fmt_metric_value(c, v, d),
        )
        self.assertEqual(list(cur_same.columns), ["Métrica", "Team X", "Team X (2)"])

    def test_normalize_p_ml_scope(self):
        self.assertAlmostEqual(normalize_p_ml_scope(0.7), 0.7, places=6)
        self.assertEqual(normalize_p_ml_scope(2.0), 0.0)
        self.assertEqual(normalize_p_ml_scope("x"), 0.0)

    def test_build_ml_totals_model_payload(self):
        def _raw(metric, lines, total_obj, p):
            return pd.DataFrame({"metric": [metric], "n_lines": [len(lines)], "p": [float(p)]})

        def _disp(df, _label):
            out = df.copy()
            out["ok"] = 1
            return out

        out = build_ml_totals_model_payload(
            team_a="A",
            team_b="B",
            totals_for_lines={"kills": object(), "time": object()},
            combo_kills_lines=[29.5],
            combo_time_lines=[],
            p_ml_scope=0.65,
            lines_table_with_p_ml_fn=_raw,
            display_ml_totals_table_fn=_disp,
        )
        self.assertTrue(out["has_any_lines"])
        self.assertEqual(len(out["tabs"]), 2)
        self.assertTrue(out["tabs"][0]["has_lines"])
        self.assertFalse(out["tabs"][1]["has_lines"])
        self.assertIn("ok", out["tabs"][0]["table_a"].columns)

    def test_build_ml_totals_laplace_payload(self):
        df = pd.DataFrame(
            {
                "win": [1, 0, 1, 1],
                "total_kills": [28, 32, 30, 27],
                "game_time_min": [31, 35, 33, 29],
            }
        )

        def _event_block(
            df_year,
            _df_15,
            _df_10,
            _df_5,
            _h_year,
            _h15,
            _h10,
            _h5,
            event_fn,
            label_yes,
            label_no,
            floor_odd,
        ):
            ok = event_fn(df_year).fillna(False).astype(bool)
            return pd.DataFrame(
                {
                    "label_yes": [label_yes],
                    "label_no": [label_no],
                    "n": [int(ok.shape[0])],
                    "k": [int(ok.sum())],
                    "floor": [float(floor_odd)],
                }
            )

        out = build_ml_totals_laplace_payload(
            team_a="A",
            team_b="B",
            combo_kills_lines=[29.5],
            combo_time_lines=[],
            df_a_all=df,
            df_b_all=df,
            df_h2h_a=df,
            df_h2h_b=df,
            event_block_fn=_event_block,
            floor_odd=1.10,
        )
        self.assertTrue(out["has_any_lines"])
        self.assertEqual(len(out["rows"]), 1)
        row = out["rows"][0]
        self.assertIn("tbl_over_a", row)
        self.assertIn("tbl_under_b", row)
        self.assertEqual(row["tbl_over_a"].iloc[0]["label_yes"], "ML&Over")

    def test_build_audit_options_and_view(self):
        df = pd.DataFrame(
            {
                "date": [1, 2],
                "league": ["LPL", "LPL"],
                "split": ["S1", "S1"],
                "playoffs": [False, False],
                "team": ["A", "A"],
                "opponent": ["B", "C"],
                "total_kills": [25, 31],
                "game_time_min": [30.5, 32.0],
            }
        )
        opts = build_audit_base_options("A", "B", df, df, df, df, df, df, df, df, df, df)
        self.assertGreaterEqual(len(opts), 10)
        self.assertIn("todos confrontos", opts[-1][0])
        mopts = build_audit_market_options("A", "B", [29.5], [11.5], [4.5], [1.5], [1.5], [32.5], [15.5], [14.5])
        self.assertIn("Total Kills", mopts)
        view = build_audit_view(df, "total_kills", 29.5, lambda x: f"{x:.1f}")
        self.assertIn("OU", view.columns)
        self.assertIn("Campeonato", view.columns)
        self.assertIn("Abaixo", set(view["OU"].astype(str)))
        self.assertIn("Acima", set(view["OU"].astype(str)))
        simple_view = build_simple_audit_view(df, "game_time_min", 31.0, lambda x: f"{x:.1f}")
        self.assertIn("Campeonato", simple_view.columns)
        self.assertIn("Valor", simple_view.columns)
        self.assertIn("Data", simple_view.columns)

    def test_quick_selector_helpers(self):
        df_a = pd.DataFrame(
            {
                "win": [1, 0, 1, 1],
                "result": [1, 0, 1, 1],
                "side": ["blue", "blue", "red", "blue"],
                "total_kills": [25, 30, 28, 31],
            }
        )
        df_h = pd.DataFrame(
            {
                "win": [1, 0],
                "result": [1, 0],
                "side": ["blue", "red"],
                "total_kills": [27, 29],
            }
        )
        mopts = build_quick_metric_options("A", "B", [29.5], [11.5], [4.5], [1.5], [1.5], [32.5], [15.5], [14.5], "A")
        self.assertIn("Kills (Total)", mopts)
        team_df, h2h_df = quick_team_frame(True, df_a, df_a, df_h, df_h)
        self.assertEqual(int(team_df.shape[0]), 4)
        p_outcome, p_side = quick_outcome_probs(0.7, True, True, team_df, "blue")
        self.assertGreater(p_outcome, 0.0)
        self.assertIsNotNone(p_side)
        rows = build_quick_selector_rows(
            df_team=team_df,
            df_h2h_team=h2h_df,
            value_col="total_kills",
            line_f=29.5,
            side_key="blue",
            want_win=True,
            p_outcome=float(p_outcome),
            p_side_ctx=p_side,
            p_over=0.55,
            laplace_pick="Liga 15",
            mix_w_model=40.0,
            mix_w_laplace=40.0,
            mix_w_sc=20.0,
            metric_key="kills",
            format_num_fn=lambda x, d: "-" if x is None else f"{float(x):.{int(d)}f}",
            line_formatter_fn=lambda mk, lv: f"{lv:g}",
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("Odd (Modelo)", rows[0])


if __name__ == "__main__":
    unittest.main()
