from __future__ import annotations

import unittest

import pandas as pd

from app_core.championship_stats_core import (
    apply_scope_filters,
    build_relevant_championship_options,
    build_championship_stats_table,
    build_threshold_lines_by_metric,
    compute_team_metric_pct,
    enrich_resumo_rows_with_stats_detail,
    format_matchup_stats_detail,
)


class ChampionshipStatsCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {"team": "Gen.G", "win": 1, "side": "blue", "game_time_min": 31.5, "firstblood": 1, "firstherald": 1, "firsttower": 1, "firstdragon": 1, "firstbaron": 1, "gold_for": 60000, "kills_for": 18, "deaths_for": 12, "towers_for": 9, "total_gold": 118000, "total_kills": 30, "total_towers": 15, "total_dragons": 5, "total_nashors": 2, "total_inhibitors": 2},
                {"team": "Gen.G", "win": 0, "side": "red", "game_time_min": 29.0, "firstblood": 0, "firstherald": 0, "firsttower": 0, "firstdragon": 1, "firstbaron": 0, "gold_for": 54000, "kills_for": 12, "deaths_for": 17, "towers_for": 4, "total_gold": 111000, "total_kills": 29, "total_towers": 11, "total_dragons": 4, "total_nashors": 1, "total_inhibitors": 1},
                {"team": "T1", "win": 1, "side": "blue", "game_time_min": 33.0, "firstblood": 1, "firstherald": 0, "firsttower": 1, "firstdragon": 0, "firstbaron": 1, "gold_for": 62000, "kills_for": 20, "deaths_for": 16, "towers_for": 8, "total_gold": 123000, "total_kills": 36, "total_towers": 13, "total_dragons": 5, "total_nashors": 2, "total_inhibitors": 2},
                {"team": "T1", "win": 0, "side": "red", "game_time_min": 28.0, "firstblood": 0, "firstherald": 1, "firsttower": 0, "firstdragon": 0, "firstbaron": 0, "gold_for": 51000, "kills_for": 10, "deaths_for": 18, "towers_for": 3, "total_gold": 108000, "total_kills": 28, "total_towers": 10, "total_dragons": 3, "total_nashors": 1, "total_inhibitors": 1},
            ]
        )

    def test_apply_scope_filters(self) -> None:
        out = apply_scope_filters(self.df, side_filter="Blue", win_loss_filter="Wins")
        self.assertEqual(int(out.shape[0]), 2)
        self.assertTrue((out["side"] == "blue").all())

    def test_build_stats_table_with_custom_line(self) -> None:
        out = build_championship_stats_table(self.df, threshold_lines_by_metric={"kills": [29.5, 35.5]})
        self.assertIn("% Kills > 29.5", out.columns)
        self.assertIn("% Kills > 35.5", out.columns)
        geng = out[out["Team"] == "Gen.G"].iloc[0]
        self.assertAlmostEqual(float(geng["% Kills > 29.5"]), 50.0, places=4)
        self.assertAlmostEqual(float(geng["% Kills > 35.5"]), 0.0, places=4)

    def test_build_threshold_lines_by_metric_injects_custom(self) -> None:
        session_state = {"sc_th_kills": "23.5,25.5,29.5"}
        out = build_threshold_lines_by_metric(
            session_state=session_state,
            parse_lines_fn=lambda txt: [float(x.strip()) for x in str(txt).split(",") if str(x).strip()],
            parse_time_lines_fn=lambda txt: [float(x.strip()) for x in str(txt).split(",") if str(x).strip()],
            custom_metric_label="Kills",
            extra_lines=[35.5],
        )
        self.assertIn(35.5, out["kills"])

    def test_compute_team_metric_pct(self) -> None:
        self.assertAlmostEqual(float(compute_team_metric_pct(self.df, team="T1", metric_key="kills", side_key="over", line_val=35.5) or 0.0), 50.0, places=4)
        self.assertAlmostEqual(float(compute_team_metric_pct(self.df, team="Gen.G", metric_key="kills", side_key="under", line_val=29.5) or 0.0), 50.0, places=4)

    def test_enrich_resumo_rows_with_stats_detail(self) -> None:
        rows = [{"Mercado": "Kills", "Lado": "Acima", "_metric": "kills", "_line": 29.5}]
        out = enrich_resumo_rows_with_stats_detail(rows, df_scope=self.df, team_a="Gen.G", team_b="T1")
        self.assertEqual(out[0]["Stats Detail"], "Gen.G 50% | T1 50%")

    def test_format_matchup_stats_detail(self) -> None:
        self.assertEqual(format_matchup_stats_detail(team_a="Gen.G", team_b="T1", pct_a=50.0, pct_b=62.0), "Gen.G 50% | T1 62%")

    def test_build_relevant_championship_options_prefers_recent_team_championships(self) -> None:
        df = self.df.copy()
        df["league"] = ["LCK", "LCK", "First Stand", "LCK"]
        df["date"] = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-03-10", "2026-01-03"])
        df["year"] = [2026, 2026, 2026, 2026]
        df["split"] = ["Spring", "Spring", "Spring", "Spring"]
        df["playoffs"] = [False, False, False, False]
        options, selected = build_relevant_championship_options(
            df,
            filters_cls=lambda **kwargs: type("F", (), kwargs)(),
            active_filters=type("AF", (), {"year": 2026, "split": "Spring", "playoffs": False})(),
            apply_filters_fn=lambda data, f: data[
                (data["year"] == getattr(f, "year", None))
                & (data["split"] == getattr(f, "split", None))
                & (data["playoffs"] == getattr(f, "playoffs", None))
            ],
            team_a="Gen.G",
            team_b="T1",
        )
        self.assertEqual(selected, "Auto")
        self.assertEqual(options[0], "Auto")
        self.assertEqual(options[1], "First Stand")


if __name__ == "__main__":
    unittest.main()
