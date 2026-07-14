from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import math
import json
import pandas as pd

from .artifact import MLArtifact
from .calibration import fit_platt_from_raw_probs
from .fit_ml_params import _build_lineup_cache, _fuse_probs_by_precision, _make_samples
from .io_oracle import load_oracle_elixir_games
from .players_layer import PlayersArtifact, load_match_player_rows
from .utils import brier, logloss


def _load_games(csv_paths: List[str]) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for p in csv_paths:
        d = load_oracle_elixir_games(str(p), keep_playoffs=None)
        if d is None or d.empty:
            continue
        # Avoid concat dtype FutureWarning from empty/all-NA frames.
        d2 = d.dropna(how="all").dropna(axis=1, how="all")
        if d2.empty or len(d2.columns) == 0:
            continue
        chunks.append(d2)
    if not chunks:
        return pd.DataFrame()
    g = chunks[0].copy() if len(chunks) == 1 else pd.concat(chunks, ignore_index=True, join="inner")
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).copy()
    g = g.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last").reset_index(drop=True)
    return g


def _ece(y: List[int], p: List[float], bins: int = 10) -> float:
    if not y or not p or len(y) != len(p):
        return float("nan")
    b = max(2, int(bins))
    acc = 0.0
    n_all = float(len(y))
    for i in range(b):
        lo = i / b
        hi = (i + 1) / b
        idx = [j for j, pp in enumerate(p) if (pp >= lo and (pp < hi if i < b - 1 else pp <= hi))]
        if not idx:
            continue
        py = sum(float(y[j]) for j in idx) / float(len(idx))
        pp = sum(float(p[j]) for j in idx) / float(len(idx))
        acc += (float(len(idx)) / n_all) * abs(py - pp)
    return float(acc)


def _read_json_file_safe(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_logit(p: float) -> float:
    x = float(max(1e-9, min(1.0 - 1e-9, float(p))))
    return float(math.log(x / (1.0 - x)))


def _safe_sigmoid(z: float) -> float:
    zz = float(max(-35.0, min(35.0, float(z))))
    return float(1.0 / (1.0 + math.exp(-zz)))


def _seg_key(league: str, split: str, side: str, bo: str, phase: str) -> str:
    return (
        f"{str(league or '').strip()}|"
        f"{str(split or '').strip()}|"
        f"{str(side or '').strip().lower()}|"
        f"{str(bo or '').strip()}|"
        f"{str(phase or '').strip().lower()}"
    )


def _apply_microseg_calibration(
    p_in: float,
    *,
    league: str,
    split: str,
    side: str,
    bo: str,
    phase: str,
    table: Dict[str, Dict[str, float]],
    min_n: int = 30,
    strength: float = 0.60,
    cap_pp: float = 8.0,
    hierarchical: bool = True,
    hierarchy_decay: float = 0.80,
) -> float:
    p0 = float(max(1e-9, min(1.0 - 1e-9, float(p_in))))
    z0 = float(_safe_logit(p0))
    keys = [
        _seg_key(league, split, side, bo, phase),
        _seg_key(league, "*", side, bo, phase),
        _seg_key(league, "*", side, "*", phase),
        _seg_key(league, "*", "*", "*", phase),
        _seg_key(league, "*", "*", "*", "*"),
        _seg_key("*", "*", "*", "*", "*"),
    ]

    if not bool(hierarchical):
        for k in keys:
            d = table.get(k) if isinstance(table, dict) else None
            if not isinstance(d, dict):
                continue
            n = float(d.get("n", 0.0) or 0.0)
            if n < float(max(1, int(min_n))):
                continue
            b = float(d.get("bias", 0.0) or 0.0)
            cap = abs(float(cap_pp)) / 100.0
            if cap > 0:
                b = max(-cap, min(cap, b))
            p_bias = float(max(1e-9, min(1.0 - 1e-9, p0 + b)))
            delta = float(_safe_logit(p_bias) - z0)
            z1 = float(z0 + float(max(0.0, min(1.0, strength))) * delta)
            return float(max(1e-9, min(1.0 - 1e-9, _safe_sigmoid(z1))))
        return p0

    dec = float(max(0.30, min(1.00, float(hierarchy_decay))))
    n_gate = float(max(1, int(min_n)))
    cands: List[tuple[str, float, float]] = []
    for lvl, k in enumerate(keys):
        d = table.get(k) if isinstance(table, dict) else None
        if not isinstance(d, dict):
            continue
        n = float(d.get("n", 0.0) or 0.0)
        if n <= 0.0:
            continue
        b = float(d.get("bias", 0.0) or 0.0)
        w_n = float(n / (n + n_gate))
        w_lvl = float(dec ** int(lvl))
        w = float(max(0.0, w_n * w_lvl))
        if w <= 0.0:
            continue
        cands.append((k, b, w))
    if not cands:
        return p0

    w_sum = float(sum(x[2] for x in cands))
    if w_sum <= 0.0:
        return p0
    b_mix = float(sum(x[1] * x[2] for x in cands) / w_sum)
    cap = abs(float(cap_pp)) / 100.0
    if cap > 0:
        b_mix = max(-cap, min(cap, b_mix))
    p_bias = float(max(1e-9, min(1.0 - 1e-9, p0 + b_mix)))
    delta = float(_safe_logit(p_bias) - z0)
    z1 = float(z0 + float(max(0.0, min(1.0, strength))) * delta)
    return float(max(1e-9, min(1.0 - 1e-9, _safe_sigmoid(z1))))


def _reliability_bins(y: List[int], p: List[float], bins: int = 10) -> List[Dict[str, Any]]:
    if not y or not p or len(y) != len(p):
        return []
    b = max(2, int(bins))
    rows: List[Dict[str, Any]] = []
    for i in range(b):
        lo = i / b
        hi = (i + 1) / b
        idx = [j for j, pp in enumerate(p) if (pp >= lo and (pp < hi if i < b - 1 else pp <= hi))]
        if not idx:
            rows.append(
                {
                    "bin": f"[{lo:.2f},{hi:.2f}]",
                    "n": 0,
                    "mean_p": None,
                    "emp_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_p = float(sum(float(p[j]) for j in idx) / float(len(idx)))
        emp = float(sum(float(y[j]) for j in idx) / float(len(idx)))
        rows.append(
            {
                "bin": f"[{lo:.2f},{hi:.2f}]",
                "n": len(idx),
                "mean_p": mean_p,
                "emp_rate": emp,
                "gap": float(emp - mean_p),
            }
        )
    return rows


def _metrics(y: List[int], p: List[float], bins: int = 10) -> Dict[str, Any]:
    if not y or not p:
        return {"n": 0, "logloss": None, "brier": None, "ece": None}
    return {
        "n": len(y),
        "logloss": float(logloss(y, p)),
        "brier": float(brier(y, p)),
        "ece": float(_ece(y, p, bins=bins)),
    }


def _oof_platt_by_month(
    df: pd.DataFrame,
    *,
    y_col: str,
    p_col: str,
    month_col: str = "month",
    min_train_n: int = 1200,
) -> pd.Series:
    if df is None or df.empty or y_col not in df.columns or p_col not in df.columns or month_col not in df.columns:
        return pd.Series(dtype=float)
    out = pd.to_numeric(df[p_col], errors="coerce").astype(float).clip(lower=1e-9, upper=1.0 - 1e-9).copy()
    months = sorted({str(x) for x in df[month_col].astype(str).tolist() if str(x).strip()})
    for m in months:
        train = df[df[month_col].astype(str) < str(m)]
        test_idx = df[df[month_col].astype(str) == str(m)].index
        if len(test_idx) <= 0:
            continue
        if len(train) < int(max(80, int(min_train_n))):
            continue
        y_train = [int(v) for v in pd.to_numeric(train[y_col], errors="coerce").fillna(0).astype(int).tolist()]
        p_train = [float(v) for v in pd.to_numeric(train[p_col], errors="coerce").fillna(0.5).astype(float).tolist()]
        try:
            cal = fit_platt_from_raw_probs(y_train, p_train, l2=1e-3, max_iter=220)
            p_test_raw = pd.to_numeric(df.loc[test_idx, p_col], errors="coerce").fillna(0.5).astype(float).clip(lower=1e-9, upper=1.0 - 1e-9)
            out.loc[test_idx] = p_test_raw.map(lambda x: float(max(1e-9, min(1.0 - 1e-9, cal.predict_proba(float(x))))))
        except Exception:
            continue
    return out


def run_calibration_diagnostics(
    *,
    artifact_path: str,
    players_artifact_path: str,
    csv_paths: List[str],
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    bins: int = 10,
    min_league_games: int = 250,
    params_fitted_path: str | None = None,
    use_microseg_fitted: bool = True,
    microseg_min_n: int = 30,
    microseg_strength: float = 0.60,
    microseg_cap_pp: float = 8.0,
    microseg_hierarchical: bool = True,
    microseg_hier_decay: float = 0.80,
    use_oof_platt_fused: bool = True,
    oof_platt_min_train_n: int = 1200,
) -> Dict[str, Any]:
    g = _load_games(csv_paths)
    if g.empty:
        return {"error": "no_games"}

    art = MLArtifact.load_json(str(artifact_path))
    p_art = PlayersArtifact.load_json(str(players_artifact_path))
    w_lineup = float((((art.meta or {}).get("players_layer") or {}).get("w_lineup") or 0.85))

    gameids = set(g["gameid"].astype(str).tolist())
    pr = load_match_player_rows(csv_paths, keep_playoffs=None, gameids=gameids)
    lineup_cache = _build_lineup_cache(pr)
    samples = _make_samples(
        art=art,
        p_art=p_art,
        games=g,
        lineup_cache=lineup_cache,
        w_lineup=w_lineup,
    )
    if not samples:
        return {"error": "no_samples"}

    microseg_table: Dict[str, Dict[str, float]] = {}
    if bool(use_microseg_fitted) and params_fitted_path:
        prm = _read_json_file_safe(Path(str(params_fitted_path)).expanduser())
        ms_pack = prm.get("microseg_calibration") if isinstance(prm.get("microseg_calibration"), dict) else {}
        ms_tbl = ms_pack.get("table") if isinstance(ms_pack.get("table"), dict) else {}
        if isinstance(ms_tbl, dict):
            microseg_table = ms_tbl

    rows: List[Dict[str, Any]] = []
    for s in samples:
        p_team = float(s.p_team)
        p_players = float(s.p_players)
        p_fused_raw = float(
            _fuse_probs_by_precision(
                p_team=p_team,
                p_players=p_players,
                n_team=float(s.n_team),
                n_players=float(s.n_players),
                team_scale=float(team_scale),
                players_scale=float(players_scale),
            )
        )
        p_fused = float(p_fused_raw)
        if microseg_table:
            p_fused = _apply_microseg_calibration(
                p_fused_raw,
                league=str(s.league),
                split=str(s.split),
                side=str(s.side or "blue"),
                bo=str(s.bo_bucket or "1"),
                phase=str(s.phase or "regular"),
                table=microseg_table,
                min_n=int(microseg_min_n),
                strength=float(microseg_strength),
                cap_pp=float(microseg_cap_pp),
                hierarchical=bool(microseg_hierarchical),
                hierarchy_decay=float(microseg_hier_decay),
            )
        rows.append(
            {
                "month": str(s.month),
                "league": str(s.league),
                "split": str(s.split),
                "side": str(s.side),
                "bo": str(s.bo_bucket),
                "phase": str(s.phase),
                "y": int(s.y),
                "p_team": p_team,
                "p_players": p_players,
                "p_fused_raw": float(p_fused_raw),
                "p_fused": p_fused,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "no_rows"}

    if bool(use_oof_platt_fused):
        try:
            df["p_fused_oof_platt"] = _oof_platt_by_month(
                df,
                y_col="y",
                p_col="p_fused",
                month_col="month",
                min_train_n=int(oof_platt_min_train_n),
            )
        except Exception:
            df["p_fused_oof_platt"] = pd.to_numeric(df.get("p_fused"), errors="coerce").fillna(0.5).astype(float)
    else:
        df["p_fused_oof_platt"] = pd.to_numeric(df.get("p_fused"), errors="coerce").fillna(0.5).astype(float)

    out_models: Dict[str, Any] = {}
    for model_key, p_col in [
        ("team_only", "p_team"),
        ("players_only", "p_players"),
        ("fused_raw", "p_fused"),
        ("fused_active", "p_fused_oof_platt"),
    ]:
        y = [int(x) for x in df["y"].tolist()]
        p = [float(x) for x in df[p_col].tolist()]
        m = _metrics(y, p, bins=bins)
        m["bins"] = _reliability_bins(y, p, bins=bins)
        out_models[model_key] = m

    monthly_rows: List[Dict[str, Any]] = []
    for month, gm in df.groupby("month"):
        y = [int(x) for x in gm["y"].tolist()]
        p = [float(x) for x in gm["p_fused"].tolist()]
        if not y:
            continue
        mm = _metrics(y, p, bins=bins)
        monthly_rows.append(
            {
                "month": str(month),
                "n": int(mm.get("n") or 0),
                "logloss": mm.get("logloss"),
                "brier": mm.get("brier"),
                "ece": mm.get("ece"),
                "mean_p": float(gm["p_fused"].mean()),
                "emp_rate": float(gm["y"].mean()),
            }
        )

    by_league: Dict[str, Any] = {}
    for lg, gl in df.groupby("league"):
        if int(gl.shape[0]) < int(min_league_games):
            continue
        yl = [int(x) for x in gl["y"].tolist()]
        by_league[str(lg)] = {}
        for model_key, p_col in [
            ("team_only", "p_team"),
            ("players_only", "p_players"),
            ("fused_raw", "p_fused"),
            ("fused_active", "p_fused_oof_platt"),
        ]:
            pl = [float(x) for x in gl[p_col].tolist()]
            mm = _metrics(yl, pl, bins=bins)
            mm["bins"] = _reliability_bins(yl, pl, bins=bins)
            by_league[str(lg)][model_key] = mm

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_samples": int(df.shape[0]),
        "bins": int(max(2, int(bins))),
        "active_scales": {"team_scale": float(team_scale), "players_scale": float(players_scale)},
        "calibration_mode": {
            "use_microseg_fitted": bool(bool(microseg_table)),
            "params_fitted_path": (str(Path(str(params_fitted_path)).expanduser()) if params_fitted_path else ""),
            "microseg_min_n": int(microseg_min_n),
            "microseg_strength": float(microseg_strength),
            "microseg_cap_pp": float(microseg_cap_pp),
            "microseg_hierarchical": bool(microseg_hierarchical),
            "microseg_hier_decay": float(microseg_hier_decay),
            "use_oof_platt_fused": bool(use_oof_platt_fused),
            "oof_platt_min_train_n": int(oof_platt_min_train_n),
        },
        "models": out_models,
        "monthly_fused": monthly_rows,
        "by_league": by_league,
    }
