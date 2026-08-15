from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .calibration_diagnostics import run_calibration_diagnostics
from .io_oracle import load_oracle_elixir_games
from .elo import EloModel, EloParams
from .utils import logloss, brier, clip_prob

_WF_REQUIRED_COLS = ("gameid", "date", "blue", "red", "blue_win", "league")


def _empty_walkforward_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gameid": pd.Series(dtype="string"),
            "date": pd.Series(dtype="string"),
            "blue": pd.Series(dtype="string"),
            "red": pd.Series(dtype="string"),
            "blue_win": pd.Series(dtype="int64"),
            "league": pd.Series(dtype="string"),
        }
    )


def _normalize_walkforward_input_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _empty_walkforward_frame()

    d = df.copy()
    for col in _WF_REQUIRED_COLS:
        if col not in d.columns:
            d[col] = pd.NA

    d = d[list(_WF_REQUIRED_COLS)].copy()
    d["gameid"] = d["gameid"].astype("string").fillna("").str.strip()
    d["date"] = d["date"].astype("string").fillna("").str.strip()
    d["blue"] = d["blue"].astype("string").fillna("").str.strip()
    d["red"] = d["red"].astype("string").fillna("").str.strip()
    d["league"] = d["league"].astype("string").fillna("").str.strip()
    d["league"] = d["league"].where(d["league"] != "", "OTHER")

    d["blue_win"] = pd.to_numeric(d["blue_win"], errors="coerce")
    d = d.dropna(subset=["blue_win"])
    if d.empty:
        return _empty_walkforward_frame()
    d["blue_win"] = (d["blue_win"].astype(float) > 0.5).astype(int)

    d = d[(d["gameid"] != "") & (d["date"] != "") & (d["blue"] != "") & (d["red"] != "")]
    if d.empty:
        return _empty_walkforward_frame()
    return d.reset_index(drop=True)


def _p_from_diff(diff_points: float, scale: float) -> float:
    scale = float(scale) if float(scale) > 0 else 400.0
    p = 1.0 / (1.0 + 10.0 ** (-float(diff_points) / scale))
    return float(clip_prob(float(p)))


def _month_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.year):04d}-{int(ts.month):02d}"


def _roi_proxy_fold(y: List[int], p: List[float]) -> float:
    # Proxy only: assumes fair payout from model itself (not bookmaker odds).
    # Positive/negative indicates calibration + sharpness interaction.
    if not y or not p or len(y) != len(p):
        return 0.0
    pnl = 0.0
    for yt, pr in zip(y, p):
        pa = float(pr)
        pb = 1.0 - pa
        if pa >= pb:
            odd = 1.0 / pa
            won = int(yt) == 1
        else:
            odd = 1.0 / pb
            won = int(yt) == 0
        pnl += (odd - 1.0) if won else -1.0
    return float(pnl / len(y))


def _fit_model(train_df: pd.DataFrame, k: float, scale: float) -> EloModel:
    m = EloModel(EloParams(k=float(k), scale_points=float(scale), init_rating=0.0))
    fit = train_df.copy()
    fit["blue_league"] = fit["league"].astype(str)
    fit["red_league"] = fit["league"].astype(str)
    fit["blue_tier"] = fit["league"].astype(str).map(lambda x: "OTHER")
    fit["red_tier"] = fit["league"].astype(str).map(lambda x: "OTHER")
    m.fit(fit[["blue", "red", "blue_win", "blue_league", "red_league", "blue_tier", "red_tier"]])
    return m


def _walkforward_from_active_calibration(cal: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cal, dict):
        return {"error": "invalid_calibration_payload"}
    if cal.get("error"):
        return {"error": str(cal.get("error"))}

    models = dict(cal.get("models") or {})
    active = dict(models.get("fused_active") or models.get("fused_raw") or {})
    monthly_src = list(cal.get("monthly_fused") or [])
    by_league_src = dict(cal.get("by_league") or {})

    folds: List[Dict[str, Any]] = []
    for row in monthly_src:
        if not isinstance(row, dict):
            continue
        folds.append(
            {
                "month": str(row.get("month") or ""),
                "n": int(row.get("n") or 0),
                "logloss": row.get("logloss"),
                "brier": row.get("brier"),
                "ece": row.get("ece"),
                "mean_p": row.get("mean_p"),
                "emp_rate": row.get("emp_rate"),
            }
        )

    by_league: Dict[str, Dict[str, Any]] = {}
    for lg, payload in by_league_src.items():
        if not isinstance(payload, dict):
            continue
        src = dict(payload.get("fused_active") or payload.get("fused_raw") or {})
        if not src:
            continue
        by_league[str(lg)] = {
            "n_total": int(src.get("n") or 0),
            "months": None,
            "logloss_mean": src.get("logloss"),
            "brier_mean": src.get("brier"),
            "ece_mean": src.get("ece"),
            "roi_proxy_mean": None,
        }

    overall = {
        "folds": len(folds),
        "n_total": int(active.get("n") or 0),
        "logloss_mean": active.get("logloss"),
        "brier_mean": active.get("brier"),
        "ece_mean": active.get("ece"),
        "roi_proxy_mean": None,
    }
    return {
        "generated_at": str(cal.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "mode": "active_model",
        "source": "calibration_diagnostics",
        "overall": overall,
        "folds": folds,
        "by_league": by_league,
    }


def run_walkforward(
    csv_paths: List[str],
    *,
    holdout_months: int = 1,
    min_train_games: int = 1500,
    k: float = 26.0,
    scale: float = 400.0,
    artifact_path: str | None = None,
    players_artifact_path: str | None = None,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    bins: int = 10,
    min_league_games: int = 250,
    params_fitted_path: str | None = None,
) -> Dict[str, Any]:
    art_p = Path(str(artifact_path or "")).expanduser() if artifact_path else None
    players_p = Path(str(players_artifact_path or "")).expanduser() if players_artifact_path else None
    if art_p and players_p and art_p.exists() and players_p.exists():
        cal = run_calibration_diagnostics(
            artifact_path=str(art_p),
            players_artifact_path=str(players_p),
            csv_paths=[str(Path(p).expanduser()) for p in (csv_paths or [])],
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            bins=int(bins),
            min_league_games=int(min_league_games),
            params_fitted_path=str(Path(str(params_fitted_path)).expanduser()) if params_fitted_path else None,
        )
        return _walkforward_from_active_calibration(cal)

    dfs = []
    for p in csv_paths:
        df = load_oracle_elixir_games(p, keep_playoffs=None)
        df_norm = _normalize_walkforward_input_df(df)
        if not df_norm.empty:
            dfs.append(df_norm)
    if not dfs:
        return {"error": "no_data"}

    g = dfs[0].copy() if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).copy()
    g = g.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last").reset_index(drop=True)
    g["month"] = g["date"].apply(_month_key)

    months = sorted(g["month"].unique().tolist())
    folds: List[Dict[str, Any]] = []
    league_rows: List[Dict[str, Any]] = []

    for i in range(1, len(months)):
        test_month = months[i]
        train = g[g["month"] < test_month].copy()
        test = g[g["month"] == test_month].copy()
        if len(train) < int(min_train_games) or test.empty:
            continue

        m = _fit_model(train, k=float(k), scale=float(scale))
        y: List[int] = []
        p: List[float] = []
        leagues: List[str] = []
        for r in test.itertuples(index=False):
            pb = float(m.p_blue(str(r.blue), str(r.red), blue_league=str(r.league), red_league=str(r.league)))
            y.append(int(r.blue_win))
            p.append(float(pb))
            leagues.append(str(r.league or "OTHER"))

        ll = float(logloss(y, p))
        br = float(brier(y, p))
        roi = float(_roi_proxy_fold(y, p))
        folds.append({"month": test_month, "n": len(y), "logloss": ll, "brier": br, "roi_proxy": roi})

        # per league for drift/roi proxy
        tmp = pd.DataFrame({"league": leagues, "y": y, "p": p})
        for lg, gg in tmp.groupby("league"):
            yl = [int(v) for v in gg["y"].tolist()]
            pl = [float(v) for v in gg["p"].tolist()]
            if not yl:
                continue
            ll_l = float(logloss(yl, pl))
            league_rows.append(
                {
                    "month": test_month,
                    "league": str(lg),
                    "n": len(yl),
                    "logloss": ll_l,
                    "drift_vs_month_global": float(ll_l - ll),
                    "roi_proxy": float(_roi_proxy_fold(yl, pl)),
                }
            )

    by_league: Dict[str, Dict[str, Any]] = {}
    if league_rows:
        dfl = pd.DataFrame(league_rows)
        for lg, gg in dfl.groupby("league"):
            by_league[str(lg)] = {
                "n_total": int(gg["n"].sum()),
                "months": int(gg["month"].nunique()),
                "logloss_mean": float((gg["logloss"] * gg["n"]).sum() / max(1, gg["n"].sum())),
                "drift_mean": float((gg["drift_vs_month_global"] * gg["n"]).sum() / max(1, gg["n"].sum())),
                "roi_proxy_mean": float((gg["roi_proxy"] * gg["n"]).sum() / max(1, gg["n"].sum())),
            }

    overall = {
        "folds": len(folds),
        "n_total": int(sum(f["n"] for f in folds)),
        "logloss_mean": float(sum(f["logloss"] * f["n"] for f in folds) / max(1, sum(f["n"] for f in folds))) if folds else None,
        "brier_mean": float(sum(f["brier"] * f["n"] for f in folds) / max(1, sum(f["n"] for f in folds))) if folds else None,
        "roi_proxy_mean": float(sum(f["roi_proxy"] * f["n"] for f in folds) / max(1, sum(f["n"] for f in folds))) if folds else None,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "elo_proxy",
        "source": "legacy_elo_proxy",
        "overall": overall,
        "folds": folds,
        "by_league": by_league,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward monthly report (drift + ROI proxy by league).")
    ap.add_argument("--csv", required=True, action="append")
    ap.add_argument("--out", default="walkforward_report.json")
    ap.add_argument("--min-train-games", type=int, default=1500)
    ap.add_argument("--k", type=float, default=26.0)
    ap.add_argument("--scale", type=float, default=400.0)
    ap.add_argument("--artifact", default="", help="Optional ml_artifact.json to evaluate active model instead of legacy Elo proxy")
    ap.add_argument("--players-artifact", default="", help="Optional players_artifact.json paired with --artifact")
    ap.add_argument("--team-scale", type=float, default=1.0)
    ap.add_argument("--players-scale", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-league-games", type=int, default=250)
    ap.add_argument("--params-fitted", default="", help="Optional params_fitted.json for active-model diagnostics")
    args = ap.parse_args()

    rep = run_walkforward(
        [str(Path(x).expanduser()) for x in args.csv],
        min_train_games=int(args.min_train_games),
        k=float(args.k),
        scale=float(args.scale),
        artifact_path=str(args.artifact or ""),
        players_artifact_path=str(args.players_artifact or ""),
        team_scale=float(args.team_scale),
        players_scale=float(args.players_scale),
        bins=int(args.bins),
        min_league_games=int(args.min_league_games),
        params_fitted_path=str(args.params_fitted or ""),
    )
    out = Path(str(args.out)).expanduser()
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[walkforward] saved:", str(out))
    print(json.dumps(rep.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()
