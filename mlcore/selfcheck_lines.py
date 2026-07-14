from __future__ import annotations

import argparse
import math
from typing import List, Optional


from .stats_teams import Filters, load_team_rows, build_team_games
from .lines_engine import (
    build_profile,
    matchup_expected_totals,
    normalize_map_mode,
    parse_time_line,
    prob_over_from_total,
    fair_odds_from_p,
    simulate_series_total,
    prob_over_series_from_sims,
    try_load_p_map_from_artifact,
)


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def _fmt_odds(x: float) -> str:
    if x is None or not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.2f}"


def _print_section(title: str) -> None:
    print("\n" + title)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Self-check lines (totals + combo markets) using OE CSV team-rows.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--teamA", required=True, help="Blue / Team A")
    ap.add_argument("--teamB", required=True, help="Red / Team B")

    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--league", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--playoffs", choices=["all", "true", "false"], default="all")
    ap.add_argument("--league-mode", choices=["all", "fixed", "auto"], default="auto")

    ap.add_argument("--map-mode", default="avg", help="avg | 1..5 | m1..m5")
    ap.add_argument("--half-life", type=float, default=180.0)

    ap.add_argument("--bo", type=int, default=1, choices=[1, 3, 5])
    ap.add_argument(
        "--series-total",
        action="store_true",
        help="Interpret provided lines as SERIES totals (sum across maps played).",
    )
    ap.add_argument("--artifact", default=None, help="Optional path to ml_artifact.json to derive p_map")
    ap.add_argument("--p-map", type=float, default=None, help="Optional p(map win) for TeamA (override)")
    ap.add_argument("--sims", type=int, default=20000)

    ap.add_argument("--kills", nargs="*", default=[])
    ap.add_argument("--towers", nargs="*", default=[])
    ap.add_argument("--dragons", nargs="*", default=[])
    ap.add_argument("--barons", nargs="*", default=[])
    ap.add_argument("--inhib", nargs="*", default=[])
    ap.add_argument("--time", nargs="*", default=[])

    args = ap.parse_args(argv)

    playoffs = None if args.playoffs == "all" else (args.playoffs == "true")
    filters = Filters(year=args.year, league=args.league, split=args.split, playoffs=playoffs)

    map_mode = normalize_map_mode(args.map_mode)

    team_rows = load_team_rows(args.csv)
    team_games = build_team_games(team_rows)

    def run_for_map(map_mode_str: str):
        mm = normalize_map_mode(map_mode_str)
        pA = build_profile(team_games, team=args.teamA, filters=filters, league_mode=args.league_mode, map_mode=mm, half_life_days=args.half_life)
        pB = build_profile(team_games, team=args.teamB, filters=filters, league_mode=args.league_mode, map_mode=mm, half_life_days=args.half_life)
        totals = matchup_expected_totals(pA, pB)
        return pA, pB, totals

    # Always compute AVG (gol.gg-style)
    _pA_avg, _pB_avg, totals_avg = run_for_map("avg")

    # Optionally compute per-map mode too
    if map_mode != "avg":
        _, _, totals_map = run_for_map(map_mode)
    else:
        _, _, totals_map = None, None, None

    # p_map
    p_map = args.p_map
    p_raw = None
    p_cal = None
    if p_map is None and args.artifact:
        got = try_load_p_map_from_artifact(args.artifact, args.teamA, args.teamB)
        if got:
            p_raw, p_cal = got
            p_map = p_cal
    if p_map is None:
        p_map = 0.5

    def p_win_series(p_map_blue: float, bo: int) -> float:
        """P(blue wins series) assuming i.i.d map win prob p_map_blue."""
        p = float(p_map_blue)
        if bo == 1:
            return p
        if bo == 3:
            # win in 2 or 3 maps: p^2 + 2*p^2*(1-p)
            return p * p + 2.0 * p * p * (1.0 - p)
        if bo == 5:
            # sum_{k=3..5} C(5,k) p^k (1-p)^{5-k}
            q = 1.0 - p
            return (10.0 * p**3 * q**2) + (5.0 * p**4 * q) + (p**5)
        return p

    p_ml = p_win_series(p_map, int(args.bo))

    print(f"[selfcheck_lines] teams: {args.teamA} vs {args.teamB}")
    print(f"[selfcheck_lines] filters: year={args.year} league={args.league} split={args.split} playoffs={args.playoffs} league_mode={args.league_mode}")
    print(f"[selfcheck_lines] half_life_days: {args.half_life}")
    print(f"[selfcheck_lines] map_mode: {args.map_mode} (avg always printed)")
    if args.artifact:
        print(f"[selfcheck_lines] artifact: {args.artifact}")
    if p_raw is not None:
        print(f"[selfcheck_lines] p_map_raw={_fmt(p_raw)} p_map_cal={_fmt(p_cal)}")
    print(f"[selfcheck_lines] bo={args.bo}")
    print(f"[selfcheck_lines] p_map_used={_fmt(p_map)}  fair_odds={_fmt_odds(fair_odds_from_p(p_map))}")
    if int(args.bo) in (3, 5):
        print(f"[selfcheck_lines] p_series_used={_fmt(p_ml)} fair_odds={_fmt_odds(fair_odds_from_p(p_ml))}")

    def show_totals_block(title: str, totals):
        _print_section(title)
        for k in ["kills", "towers", "dragons", "barons", "inhibitors", "time"]:
            t = totals[k]
            if k == "time":
                print(f"  {k:10s} mean={_fmt(t.mean,2)}min  sd={_fmt(t.sd,2)}  dist={t.dist}")
            else:
                print(f"  {k:10s} mean={_fmt(t.mean,2)}  sd={_fmt(t.sd,2)}  dist={t.dist}")

    show_totals_block("[AVG MAPS] expected totals (gol.gg style)", totals_avg)
    if totals_map is not None:
        show_totals_block(f"[{map_mode.upper()}] expected totals", totals_map)

    def eval_lines(metric: str, lines: List[str], totals):
        if not lines:
            return
        _print_section(f"[{metric}] lines")
        for s in lines:
            if metric == "time":
                line = parse_time_line(s)
            else:
                line = float(str(s).replace(",", "."))
            p_over = prob_over_from_total(totals[metric], line)
            p_under = 1.0 - p_over if math.isfinite(p_over) else float("nan")
            print(
                f"  line={s:>6}  p_over={_fmt(p_over)} (odds={_fmt_odds(fair_odds_from_p(p_over))})"
                f"  p_under={_fmt(p_under)} (odds={_fmt_odds(fair_odds_from_p(p_under))})"
            )

            # combo (independence baseline)
            # For MAP lines, ML is a MAP market -> use p_map.
            p_ml_over = p_map * p_over if math.isfinite(p_over) else float("nan")
            p_ml_under = p_map * p_under if math.isfinite(p_under) else float("nan")
            print(
                f"          ML+OVER  p={_fmt(p_ml_over)} odds={_fmt_odds(fair_odds_from_p(p_ml_over))}"
                f" | ML+UNDER p={_fmt(p_ml_under)} odds={_fmt_odds(fair_odds_from_p(p_ml_under))}"
            )

    # choose which totals to price lines against:
    totals_for_lines = totals_map if totals_map is not None else totals_avg

    def eval_series_lines(metric: str, lines: List[str], totals):
        if not lines:
            return
        _print_section(f"[{metric}] SERIES lines (bo{args.bo}, sims={args.sims})")
        sims = simulate_series_total(bo=args.bo, p_blue_map=p_map, total=totals[metric], n_sims=args.sims)
        print(f"  sims_mean={_fmt(float(sims.mean()),2)}")
        for s in lines:
            if metric == "time":
                line = parse_time_line(s)
            else:
                line = float(str(s).replace(",", "."))
            p_over = prob_over_series_from_sims(sims, line)
            p_under = 1.0 - p_over if math.isfinite(p_over) else float("nan")
            print(
                f"  line={s:>6}  p_over={_fmt(p_over)} (odds={_fmt_odds(fair_odds_from_p(p_over))})"
                f"  p_under={_fmt(p_under)} (odds={_fmt_odds(fair_odds_from_p(p_under))})"
            )

            # combo baseline (independence)
            p_ml_over = p_ml * p_over if math.isfinite(p_over) else float("nan")
            p_ml_under = p_ml * p_under if math.isfinite(p_under) else float("nan")
            print(
                f"          ML+OVER  p={_fmt(p_ml_over)} odds={_fmt_odds(fair_odds_from_p(p_ml_over))}"
                f" | ML+UNDER p={_fmt(p_ml_under)} odds={_fmt_odds(fair_odds_from_p(p_ml_under))}"
            )

    if args.series_total:
        if args.bo == 1:
            print("\n[selfcheck_lines] NOTE: --series-total with bo1 is the same as map totals.")
        eval_series_lines("kills", args.kills, totals_for_lines)
        eval_series_lines("towers", args.towers, totals_for_lines)
        eval_series_lines("dragons", args.dragons, totals_for_lines)
        eval_series_lines("barons", args.barons, totals_for_lines)
        eval_series_lines("inhibitors", args.inhib, totals_for_lines)
        eval_series_lines("time", args.time, totals_for_lines)
    else:
        eval_lines("kills", args.kills, totals_for_lines)
        eval_lines("towers", args.towers, totals_for_lines)
        eval_lines("dragons", args.dragons, totals_for_lines)
        eval_lines("barons", args.barons, totals_for_lines)
        eval_lines("inhibitors", args.inhib, totals_for_lines)
        eval_lines("time", args.time, totals_for_lines)

    # series totals summary (always handy when bo>1)
    if args.bo in (3, 5):
        _print_section(f"[SERIES TOTALS] simulated summary for bo{args.bo} (sims={args.sims})")
        for metric in ["kills", "towers", "dragons", "barons", "inhibitors", "time"]:
            sims = simulate_series_total(bo=args.bo, p_blue_map=p_map, total=totals_for_lines[metric], n_sims=args.sims)
            print(f"  {metric:10s}: mean={_fmt(float(sims.mean()),2)}")


if __name__ == "__main__":
    main()
