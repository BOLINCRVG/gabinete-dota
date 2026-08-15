from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "mlcore"

import argparse
from datetime import date, datetime, timezone
from typing import Any, Dict

from .artifact import MLArtifact
from .bo import p_series_from_pmap
from .achievements import load_achievements, build_team_index, team_boost_points
from .tiers import tier_base_points, league_base_points, tier_priority

def _team_meta_from_artifact(art: MLArtifact) -> Dict[str, Dict[str, Any]]:
    meta = dict(art.meta or {})
    tm = meta.get("team_meta") or {}
    if isinstance(tm, dict) and tm:
        return {str(k): dict(v) for k, v in tm.items() if isinstance(v, dict)}
    model = getattr(art, "model", None)
    model_tm = getattr(model, "team_meta", None) if model is not None else None
    if isinstance(model_tm, dict) and model_tm:
        return {str(k): dict(v) for k, v in model_tm.items() if isinstance(v, dict)}
    return {}


def _strength(team: str, model: Any, team_meta: Dict[str, Dict[str, Any]]) -> float:
    m = team_meta.get(team) or {}
    league = str(m.get("league", "OTHER") or "OTHER")
    tier = str(m.get("tier", "OTHER") or "OTHER")
    base_strength_fn = getattr(model, "base_strength", None)
    if callable(base_strength_fn):
        try:
            return float(base_strength_fn(team, league=league, tier=tier, apply_rank_caps=False))
        except TypeError:
            try:
                return float(base_strength_fn(team, league=league, tier=tier))
            except Exception:
                pass
        except Exception:
            pass
    ratings = dict(getattr(model, "ratings", {}) or {}) if model is not None else {}
    return float(ratings.get(team, 0.0)) + tier_base_points(tier) + league_base_points(league)

def _p_from_diff_points(diff_points: float, scale_points: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** (-float(diff_points) / float(scale_points))))


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()

def main() -> None:
    ap = argparse.ArgumentParser(description="Self-check an ML artifact (ranking + example prediction).")
    ap.add_argument("--artifact", default="ml_artifact.json", help="Path to ml_artifact.json")
    ap.add_argument("--top", type=int, default=20, help="Top N teams")
    ap.add_argument("--example-blue", default="100 Thieves", help="Example blue team")
    ap.add_argument("--example-red", default="3BL Esports", help="Example red team")
    ap.add_argument("--bo", type=int, default=1, help="Best-of (1/3/5)")
    ap.add_argument("--asof", default="", help="As-of date for achievements (YYYY-MM-DD). Default: today")
    args = ap.parse_args()

    art = MLArtifact.load_json(args.artifact)

    meta = dict(art.meta or {})
    team_meta = _team_meta_from_artifact(art)
    ratings = dict(getattr(art.model, "ratings", {}) or {})
    scale = float(getattr(getattr(art.model, "params", None), "scale_points", 400.0))

    print("[selfcheck] artifact version:", art.version)
    if "metrics" in meta:
        print("[selfcheck] metrics:", meta["metrics"])

    # Achievements
    ach_cfg = dict((meta.get("achievements") or {}))
    weights = dict(ach_cfg.get("weights_points") or {})
    ach_path = ach_cfg.get("path") or "achievements.json"
    ach_data = load_achievements(ach_path)
    # Fallback: manual weights in achievements.json (config.fixed_weights_points)
    if not weights:
        weights = dict((ach_data.get('config') or {}).get('fixed_weights_points') or {})
    team_idx = build_team_index(ach_data)

    if args.asof.strip():
        try:
            as_of = datetime.fromisoformat(args.asof.strip()).date()
        except Exception:
            as_of = _today_utc()
    else:
        as_of = _today_utc()

    def total_with_ach(team: str) -> float:
        base = _strength(team, art.model, team_meta)
        if not weights:
            return base
        return base + team_boost_points(team, as_of, team_idx, weights)

    # Top
    rows = []
    teams = set(ratings.keys()) | set(team_meta.keys())
    for team in teams:
        m = team_meta.get(team) or {}
        league = str(m.get("league", "OTHER"))
        tier = str(m.get("tier", "OTHER"))
        base = _strength(team, art.model, team_meta)
        ach = team_boost_points(team, as_of, team_idx, weights) if weights else 0.0
        rows.append((team, base + ach, base, ach, league, tier))
    rows_raw = sorted(rows, key=lambda x: x[1], reverse=True)
    rows = sorted(rows, key=lambda x: (tier_priority(x[5]), -x[1], x[0]))

    print("\nTop GLOBAL ranking (tier travado: T1 > T2 > ERL > MINOR):")
    for team, total, base, ach, league, tier in rows[: int(args.top)]:
        print(f"  {team:30s} {total:8.2f}  base={base:7.2f} ach={ach:6.2f}  league={league:6s} tier={tier:6s}")

    print("\nTop GLOBAL (raw, sem trava de tier):")
    for team, total, base, ach, league, tier in rows_raw[: int(args.top)]:
        print(f"  {team:30s} {total:8.2f}  base={base:7.2f} ach={ach:6.2f}  league={league:6s} tier={tier:6s}")

    if weights:
        topw = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
        print("\nAchievement weights (points, top abs):")
        for fk, w in topw:
            print(f"  {fk:16s} {w:8.2f}")
    else:
        print("\nAchievement weights: (none)  -> run train_offline with --learn-ach and a non-empty achievements.json")

    # Example prediction (map)
    blue = args.example_blue
    red = args.example_red

    base_b = _strength(blue, art.model, team_meta)
    base_r = _strength(red, art.model, team_meta)
    ach_b = team_boost_points(blue, as_of, team_idx, weights) if weights else 0.0
    ach_r = team_boost_points(red, as_of, team_idx, weights) if weights else 0.0

    sb = base_b + ach_b
    sr = base_r + ach_r
    mb = team_meta.get(blue) or {}
    mr = team_meta.get(red) or {}
    diff = float(sb - sr + float(getattr(art.model, "side_adv_points", 0.0) or 0.0))
    p_raw = float("nan")
    try:
        p_raw = float(
            art.model.p_blue(
                blue,
                red,
                blue_league=str(mb.get("league", "OTHER") or "OTHER"),
                red_league=str(mr.get("league", "OTHER") or "OTHER"),
                blue_tier=str(mb.get("tier", "OTHER") or "OTHER"),
                red_tier=str(mr.get("tier", "OTHER") or "OTHER"),
            )
        )
    except Exception:
        p_raw = _p_from_diff_points(diff, scale)
    p_cal = art.calibrator.predict_proba(p_raw)
    p_series = p_series_from_pmap(float(p_cal), int(args.bo))
    print("\nExample prediction (uses team_meta league/tier inferred at training):")
    print(f"  {blue} (blue) [{mb.get('league','?')}/{mb.get('tier','?')}] vs {red} (red) [{mr.get('league','?')}/{mr.get('tier','?')}]")
    print(f"  as_of={as_of.isoformat()}  base_blue={base_b:.2f} base_red={base_r:.2f}  ach_blue={ach_b:.2f} ach_red={ach_r:.2f}")
    print(f"  strength_blue={sb:.2f} strength_red={sr:.2f} diff={diff:.2f}")
    print(f"  p_map_raw={p_raw:.3f}  p_map_cal={p_cal:.3f}  p_series(bo{int(args.bo)})={p_series:.3f}")

if __name__ == "__main__":
    main()
