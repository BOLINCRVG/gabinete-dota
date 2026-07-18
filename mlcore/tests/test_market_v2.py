"""Tests for the Mercado v2 panel core and helpers."""
from __future__ import annotations

import math
import unittest

import pandas as pd

from app_core.market_window_analysis_core import (
    HighlightedSample,
    build_alert_lines,
    compute_window_stats,
    highlight_context_samples,
)


def _make_league_scope(n_teams: int = 6, n_games_per_team: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic team_games-like DataFrame: each team plays each opponent once."""
    import random
    rng = random.Random(int(seed))
    teams = [f"Team{i}" for i in range(n_teams)]
    rows = []
    for ti, team in enumerate(teams):
        opps = [t for t in teams if t != team]
        for gi in range(n_games_per_team):
            opp = opps[gi % len(opps)]
            # team kills correlate with team's "skill" (lower index = more kills)
            base_kills = max(0.0, 28.0 - 1.5 * ti + rng.uniform(-4.0, 4.0))
            base_towers = max(0.0, 7.5 - 0.4 * ti + rng.uniform(-2.0, 2.0))
            rows.append({
                "team": team, "opponent": opp,
                "league": "TST", "year": 2026, "split": "Spring",
                "playoffs": False,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=gi + ti * 5),
                "kills_for": base_kills,
                "towers_for": base_towers,
                "game_time_min": 30.0 + rng.uniform(-3.0, 5.0),
            })
    return pd.DataFrame(rows)


class TestComputeWindowStats(unittest.TestCase):

    def test_windows_returned_in_order(self):
        league = _make_league_scope()
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Over",
        )
        self.assertEqual([s.label for s in stats], ["Ano todo", "Últimos 10", "Últimos 5"])

    def test_hit_rate_consistent_with_n(self):
        league = _make_league_scope()
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Over",
        )
        for s in stats:
            if s.n > 0:
                expected = 100.0 * s.hits / s.n
                self.assertAlmostEqual(s.hit_rate_pct, expected, places=4)

    def test_empty_team_returns_neutral(self):
        league = _make_league_scope()
        empty_team_df = league.iloc[0:0]
        stats = compute_window_stats(
            team_df=empty_team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Over",
        )
        for s in stats:
            self.assertEqual(s.n, 0)
            self.assertTrue(math.isnan(s.hit_rate_pct))
            self.assertEqual(s.tone, "neutral")

    def test_under_direction(self):
        league = _make_league_scope()
        team_df = league[league["team"] == "Team5"]  # weakest, mostly under
        stats = compute_window_stats(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Under",
        )
        # Team5 has lower kills → Under rate should be high
        ano_todo = stats[0]
        self.assertGreater(ano_todo.hit_rate_pct, 50.0)

    def test_opp_baseline_is_computed(self):
        league = _make_league_scope()
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Over",
        )
        for s in stats:
            if s.n > 0:
                self.assertTrue(math.isfinite(s.opp_avg_baseline))

    def test_tone_assignment(self):
        league = _make_league_scope()
        # Find a strong team with high hit rate (>=70) for green
        team_df = league[league["team"] == "Team0"]
        stats = compute_window_stats(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=20.0, direction="Over",  # very low line
        )
        # With low line, hit rates will be high
        self.assertIn(stats[0].tone, ("green", "yellow", "red"))


class TestHighlightSamples(unittest.TestCase):

    def test_returns_three_buckets(self):
        league = _make_league_scope()
        team_df = league[league["team"] == "Team0"]
        out = highlight_context_samples(
            team_df=team_df, league_scope_df=league,
            metric_col="kills_for", line=27.5, direction="Over",
        )
        self.assertIn("impressive_hits", out)
        self.assertIn("weak_hits", out)
        self.assertIn("surprising_misses", out)
        for _k, lst in out.items():
            self.assertIsInstance(lst, list)
            for s in lst:
                self.assertIsInstance(s, HighlightedSample)


class TestBuildAlertLines(unittest.TestCase):

    def test_alert_for_impressive_hit_over(self):
        samples = {
            "impressive_hits": [HighlightedSample(
                date="2026-04-01", opponent="WeakOpp",
                value=30.0, opp_baseline_value=15.0,
                side="hit", note="vs WeakOpp (very_under, baseline 15.0)",
            )],
            "weak_hits": [],
            "surprising_misses": [],
        }
        lines = build_alert_lines(team_name="Gen.G", samples=samples, direction="Over")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["tone"], "good")
        self.assertIn("Gen.G", lines[0]["text"])

    def test_alert_for_weak_hit_warn_tone(self):
        samples = {
            "impressive_hits": [],
            "weak_hits": [HighlightedSample(
                date="2026-04-01", opponent="OverOpp",
                value=29.0, opp_baseline_value=32.0,
                side="hit", note="vs OverOpp (very_over, baseline 32.0)",
            )],
            "surprising_misses": [],
        }
        lines = build_alert_lines(team_name="T1", samples=samples, direction="Over")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["tone"], "warn")

    def test_no_samples_no_alerts(self):
        samples = {"impressive_hits": [], "weak_hits": [], "surprising_misses": []}
        lines = build_alert_lines(team_name="X", samples=samples, direction="Over")
        self.assertEqual(lines, [])


class _FakeColumn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _FakeSt:
    def __init__(self):
        self.markdowns = []
        self.captions = []
        self.warnings = []
        self.infos = []
        self.metrics = []
        self.session_state = {}
        self.images = []
    def container(self, *a, **k): return _FakeColumn()
    def expander(self, *a, **k): return _FakeColumn()
    def spinner(self, *a, **k): return _FakeColumn()
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_FakeColumn() for _ in range(n))
    def markdown(self, t, **k): self.markdowns.append(str(t))
    def caption(self, t, **k): self.captions.append(str(t))
    def warning(self, t, **k): self.warnings.append(str(t))
    def info(self, t, **k): self.infos.append(str(t))
    def error(self, t, **k): pass
    def success(self, t, **k): pass
    def image(self, url, **k): self.images.append(str(url))
    def metric(self, label, value, **k): self.metrics.append({"label": label, "value": value})
    def number_input(self, label, *, key=None, value=0.0, **k):
        return self.session_state.get(key, value) if key else value
    def button(self, label, *, key=None, **k):
        return False
    def selectbox(self, label, options, *, key=None, index=0, **k):
        opts = list(options) or [""]
        v = opts[index if 0 <= index < len(opts) else 0]
        if key:
            self.session_state[key] = v
        return v
    def radio(self, label, options, *, key=None, index=0, **k):
        opts = list(options) or [""]
        v = opts[index if 0 <= index < len(opts) else 0]
        if key:
            self.session_state[key] = v
        return v


class TestPanelRender(unittest.TestCase):
    """Smoke test: full panel renders without raising."""

    def test_render_full_panel(self):
        from app_core.market_v2_panel_ui_core import render_market_v2_panel
        league = _make_league_scope()
        team_a, team_b = "Team0", "Team3"
        df_a = league[league["team"] == team_a]
        df_b = league[league["team"] == team_b]
        fake = _FakeSt()
        render_market_v2_panel(
            st=fake, session_state=fake.session_state,
            team_a=team_a, team_b=team_b,
            league_a="TST", league_b="TST",
            df_team_a=df_a, df_team_b=df_b, league_scope_df=league,
            p_map=0.58, p_md3=0.62, p_md5=0.65,
        )
        # At least header + some markdowns rendered
        self.assertGreater(len(fake.markdowns), 5)

    def test_render_empty_teams_shows_info(self):
        from app_core.market_v2_panel_ui_core import render_market_v2_panel
        league = _make_league_scope()
        fake = _FakeSt()
        render_market_v2_panel(
            st=fake, session_state=fake.session_state,
            team_a="", team_b="",
            df_team_a=pd.DataFrame(), df_team_b=pd.DataFrame(), league_scope_df=league,
        )
        self.assertTrue(any("Selecione" in m for m in fake.infos))


if __name__ == "__main__":
    unittest.main()
