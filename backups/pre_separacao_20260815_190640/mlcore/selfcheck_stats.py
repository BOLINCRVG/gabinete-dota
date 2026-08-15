from __future__ import annotations

import argparse
from pathlib import Path

from .stats_teams import (
    Filters,
    build_team_games,
    compare_two_teams,
    compare_two_teams_league_mode,
    load_team_rows,
)


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "-"
    return f"{x:.2f}"


def _print_block(title: str, s: dict) -> None:
    print(f"\n[{title}] {s['team']} | games={s['games']} | recent15={s.get('recent_wl_15','')}")
    print(f"  win_rate:          {_fmt(100*s['win_rate'])}%")
    print(f"  avg_total_kills:   {_fmt(s['avg_total_kills'])}")
    print(f"  avg_total_towers:  {_fmt(s['avg_total_towers'])}")
    print(f"  avg_total_dragons: {_fmt(s['avg_total_dragons'])}")
    print(f"  avg_total_nashors: {_fmt(s['avg_total_nashors'])}")
    print(f"  avg_total_inhib:   {_fmt(s['avg_total_inhibitors'])}")
    print(f"  avg_game_time(min):{_fmt(s['avg_game_time_min'])}")
    print(f"  kills_for/against: {_fmt(s['avg_kills_for'])} / {_fmt(s['avg_kills_against'])}")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m mlcore.selfcheck_stats",
        description="Quick sanity-check for team stats (Gol.GG-like) on OracleElixir CSV.",
    )
    p.add_argument("--csv", required=True, help="Path to OracleElixir CSV (2024/2025/2026 etc.)")
    p.add_argument("--teamA", required=True, help="Team A name (exact match to CSV teamname)")
    p.add_argument("--teamB", required=True, help="Team B name (exact match to CSV teamname)")
    p.add_argument("--year", type=int, default=None, help="Filter by year (e.g., 2025)")
    p.add_argument("--league", default=None, help="Filter by league (exact match, e.g., 'LTA N')")
    p.add_argument(
        "--league-mode",
        default="fixed",
        choices=["fixed", "auto", "all"],
        help=(
            "How to apply the league filter: "
            "fixed=use --league for both teams (legacy), "
            "auto=each team uses its own most common league, "
            "all=no league filtering"
        ),
    )
    p.add_argument("--split", default=None, help="Filter by split (exact match)")
    p.add_argument(
        "--playoffs",
        default=None,
        choices=["all", "true", "false"],
        help="Filter playoffs: all/true/false",
    )
    p.add_argument("--half-life", type=float, default=180.0, help="Recency half-life in days (default: 180)")

    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    playoffs = None
    if args.playoffs == "true":
        playoffs = True
    elif args.playoffs == "false":
        playoffs = False

    f = Filters(year=args.year, league=args.league, split=args.split, playoffs=playoffs)

    team_rows = load_team_rows(csv_path)
    team_games = build_team_games(team_rows)

    # If league_mode is fixed, we keep the old behaviour.
    # If league_mode is auto, we pick each team's most common league within the base filtered set.
    # If league_mode is all, we ignore league filtering entirely.
    if args.league_mode == "fixed":
        res = compare_two_teams(team_games, args.teamA, args.teamB, filters=f, half_life_days=args.half_life)
    else:
        res = compare_two_teams_league_mode(
            team_games,
            args.teamA,
            args.teamB,
            filters=f,
            league_mode=args.league_mode,
            half_life_days=args.half_life,
        )

    print("[selfcheck_stats] csv:", csv_path)
    print("[selfcheck_stats] filters:", f)
    print("[selfcheck_stats] half_life_days:", args.half_life)

    for bucket in ["all", "last5", "last10", "last15"]:
        _print_block(f"A/{bucket}", res["A"][bucket])
        _print_block(f"B/{bucket}", res["B"][bucket])


if __name__ == "__main__":
    main()
