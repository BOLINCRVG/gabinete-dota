from __future__ import annotations

# Can run both as:
#   python -m mlcore.train_offline ...
# or (from repo root):
#   python .\mlcore\train_offline.py ...

import math
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "mlcore"

import argparse
from datetime import datetime, timezone, date
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

from .io_oracle import load_oracle_elixir_games
from .elo import EloModel, EloParams
from .league_sos import sos_league_base_points
from .calibration import fit_platt_from_raw_probs, PlattCalibrator
from .artifact import MLArtifact
from .utils import logloss, brier, clip_prob
from .tiers import tier_for_league, tier_priority, is_international_league
from .achievements import load_achievements, learn_weights_points
from .players_layer import (
    PlayersArtifact,
    load_match_player_rows,
    lineup_for_game,
    compute_delta_and_confidence,
    load_bytournament_stats,
    fit_prior_beta,
    fit_w_lineup,
    logit as _logit,
    sigmoid as _sigmoid,
)
from .walkforward_report import run_walkforward

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _date_split_holdout(games: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame, datetime | None]:
    """Split train/holdout by keeping the last N days for holdout.

    Returns (train_df, test_df, cutoff_dt_utc).
    """
    if games.empty:
        return games, games, None
    df = games.sort_values(["date", "gameid"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    max_dt = df["date"].max()
    if pd.isna(max_dt):
        return df, df.iloc[0:0].copy(), None

    cutoff = max_dt - pd.Timedelta(days=int(holdout_days))
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    # If holdout is too small (common early season), keep a minimal tail.
    if len(test) < 200 and len(df) >= 800:
        test = df.tail(200).copy()
        train = df.iloc[: max(0, len(df) - 200)].copy()
        cutoff = test["date"].min()

    cutoff_dt = None
    try:
        cutoff_dt = cutoff.to_pydatetime().replace(tzinfo=timezone.utc)
    except Exception:
        cutoff_dt = None

    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff_dt


def _infer_team_domestic_league_by_year(games: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """Infer each team's *domestic* league per year (from match rows).

    We ignore international events (Worlds/MSI/EUM/etc.) when building this mapping.

    Output:
        team -> {year -> league}
    """
    if games is None or games.empty:
        return {}

    needed = {"blue", "red", "league", "year"}
    if not needed.issubset(set(games.columns)):
        return {}

    # Count (team, year, league) occurrences excluding international events
    counts: Dict[tuple[str, int, str], int] = {}
    for r in games.itertuples(index=False):
        lg = str(getattr(r, "league", "") or "").strip()
        if not lg or is_international_league(lg):
            continue
        try:
            y = int(r.year)
        except Exception:
            continue
        b = str(getattr(r, "blue", "") or "").strip()
        rr = str(getattr(r, "red", "") or "").strip()
        if b:
            k = (b, y, lg)
            counts[k] = counts.get(k, 0) + 1
        if rr:
            k = (rr, y, lg)
            counts[k] = counts.get(k, 0) + 1

    out: Dict[str, Dict[int, str]] = {}
    # For each (team, year), pick league with max count
    tmp: Dict[tuple[str, int], Dict[str, int]] = {}
    for (t, y, lg), c in counts.items():
        # NOTE: assignment evaluates RHS first; ensure bucket exists before reading.
        key = (t, y)
        bucket = tmp.setdefault(key, {})
        bucket[lg] = bucket.get(lg, 0) + int(c)

    for (t, y), d in tmp.items():
        best_lg = max(d.items(), key=lambda kv: kv[1])[0]
        out.setdefault(t, {})[int(y)] = str(best_lg)

    return out


def _infer_team_meta_latest(team_domestic_by_year: Dict[str, Dict[int, str]]) -> Dict[str, Dict[str, str]]:
    """Build a stable team_meta (league/tier) using the most recent year available."""
    out: Dict[str, Dict[str, str]] = {}
    for team, by_year in (team_domestic_by_year or {}).items():
        if not by_year:
            continue
        try:
            y = max(int(k) for k in by_year.keys())
        except Exception:
            continue
        lg = str(by_year.get(y) or "OTHER")
        tr = str(tier_for_league(lg))
        out[str(team)] = {"league": lg, "tier": tr}
    return out


def _resolve_home_league(
    *,
    team: str,
    year: int | None,
    match_league: str,
    domestic_by_year: Dict[str, Dict[int, str]],
    team_meta_latest: Dict[str, Dict[str, str]],
) -> str:
    """Resolve the best domestic league for a team in that year.

    Priority:
      1) inferred (team, year) domestic league (from non-event matches)
      2) match_league itself, if it's not an international event
      3) latest known domestic league (team_meta_latest)
      4) OTHER
    """
    t = str(team)
    lg_match = str(match_league or "").strip()
    if year is not None:
        try:
            y = int(year)
        except Exception:
            y = None
        if y is not None:
            lg = (domestic_by_year.get(t) or {}).get(y)
            if lg:
                return str(lg)
    if lg_match and (not is_international_league(lg_match)):
        return lg_match
    m = team_meta_latest.get(t) or {}
    lg2 = str(m.get("league") or "").strip()
    return lg2 or "OTHER"


def _augment_games_with_home_meta(
    df: pd.DataFrame,
    *,
    domestic_by_year: Dict[str, Dict[int, str]],
    team_meta_latest: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
    """Add blue_league/red_league + blue_tier/red_tier columns."""
    if df is None or df.empty:
        return df

    out = df.copy()
    bl_list: List[str] = []
    rl_list: List[str] = []
    bt_list: List[str] = []
    rt_list: List[str] = []

    for r in out.itertuples(index=False):
        b = str(getattr(r, "blue", "") or "")
        rr = str(getattr(r, "red", "") or "")
        lg_match = str(getattr(r, "league", "") or "OTHER")
        y = None
        try:
            y = int(r.year)
        except Exception:
            y = None

        bl = _resolve_home_league(team=b, year=y, match_league=lg_match, domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)
        rl = _resolve_home_league(team=rr, year=y, match_league=lg_match, domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)

        bt = str(tier_for_league(bl))
        rt = str(tier_for_league(rl))

        bl_list.append(bl)
        rl_list.append(rl)
        bt_list.append(bt)
        rt_list.append(rt)

    out["blue_league"] = bl_list
    out["red_league"] = rl_list
    out["blue_tier"] = bt_list
    out["red_tier"] = rt_list
    return out


def _p_from_diff_points(diff_points: float, scale_points: float = 400.0) -> float:
    """Base-10 Elo logistic (same as EloModel uses)."""
    try:
        denom = float(scale_points) if float(scale_points) > 0 else 400.0
        p = float(1.0 / (1.0 + 10.0 ** (-(float(diff_points) / denom))))
        return float(clip_prob(p))
    except Exception:
        return 0.5


def _safe_int_year(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        y = int(v)
        return y if y > 0 else None
    except Exception:
        return None


def _reliability_ece(y_true: List[int], p_pred: List[float], n_bins: int = 10) -> Dict[str, Any]:
    if not y_true or not p_pred or len(y_true) != len(p_pred):
        return {"ece": None, "bins": []}
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    n = len(y_true)
    for i in range(int(n_bins)):
        lo = i / float(n_bins)
        hi = (i + 1) / float(n_bins)
        idx = [j for j, p in enumerate(p_pred) if (p >= lo and (p < hi or (i == n_bins - 1 and p <= hi)))]
        if not idx:
            continue
        py = float(sum(float(y_true[j]) for j in idx) / len(idx))
        pp = float(sum(float(p_pred[j]) for j in idx) / len(idx))
        w = float(len(idx) / n)
        gap = abs(py - pp)
        ece += w * gap
        bins.append({"lo": lo, "hi": hi, "n": len(idx), "p_mean": pp, "y_mean": py, "gap": gap})
    return {"ece": float(ece), "bins": bins}


def _maybe_reset_on_promotion(
    model: EloModel,
    team: str,
    new_tier: str,
    *,
    last_tier: Dict[str, str],
) -> None:
    """Reset residual rating when a team moves to a *stronger* tier.

    This is a practical guardrail: teams that dominate weak leagues should not
    keep an extreme residual when entering a much stronger tier.

    BUG CRÍTICO corrigido 24/07 (achado com ele): comparar com o tier ANTERIOR
    fazia todo time de T1 que joga um torneio MINOR pontual perder o histórico ao
    voltar. Ex.: LCK -> KeSPA Cup (MINOR) -> LCK era lido como "promoção" e zerava
    tudo. Os 10 times da LCK ficaram com games_played 2-14 em vez de 300-500
    (Gen.G: 5 resets, o último 2 dias antes do fim dos dados -> gp=2), destruindo
    rating e previsões da liga inteira. Agora comparamos com o MELHOR tier que o
    time já alcançou: voltar a um tier onde ele já esteve NÃO é promoção; subir
    pela primeira vez (LCKC -> LCK de verdade) continua resetando.
    """
    t = str(team)
    nt = str(new_tier or "OTHER")
    ot = str(last_tier.get(t) or "")
    if not ot:
        last_tier[t] = nt
        return

    if nt != ot:
        try:
            if tier_priority(nt) < tier_priority(ot):
                # Promotion -> reset residual evidence
                if hasattr(model, "ratings") and isinstance(model.ratings, dict):
                    model.ratings[t] = 0.0
                if hasattr(model, "games_played") and isinstance(model.games_played, dict):
                    model.games_played[t] = 0
        except Exception:
            # Guardrail-only path; keep training running even if one team metadata is malformed.
            return

    # Guarda o MELHOR tier já visto (menor priority), não o último: é o que
    # distingue promoção real de ida-e-volta em torneio.
    try:
        if not ot or tier_priority(nt) < tier_priority(ot):
            last_tier[t] = nt
    except (TypeError, ValueError):
        last_tier[t] = nt


def _top_global_v04(model: EloModel, team_meta: Dict[str, Dict[str, str]], n: int = 20, tier_lock: bool = True) -> List[Tuple[str, int, str, str]]:
    """Top ranking display helper (v0.4+)."""
    rows: List[Tuple[str, int, str, str, float]] = []
    for team in (model.ratings or {}).keys():
        m = team_meta.get(str(team)) or {}
        lg = str(m.get("league") or "OTHER")
        tr = str(m.get("tier") or "OTHER")
        s = float(model.base_strength(str(team), league=lg, tier=tr, apply_rank_caps=True))
        gp = int((model.games_played or {}).get(str(team), 0) or 0)
        rows.append((str(team), gp, lg, tr, s))

    if tier_lock:
        rows.sort(key=lambda x: (tier_priority(x[3]), -x[4]))
    else:
        rows.sort(key=lambda x: -x[4])

    out: List[Tuple[str, int, str, str]] = [(t, gp, lg, tr) for (t, gp, lg, tr, _s) in rows[: int(n)]]
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # Dota tem time com nome em chines/cirilico. No Windows o console e cp1252 e
    # o print do ranking final estourava UnicodeEncodeError — o artifact JA estava
    # salvo, mas o processo saia com erro e qualquer wrapper (robo, atualizador)
    # lia isso como "treino falhou". Visto em 16/08/2026 no gate do TI.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Train ML core artifact from OracleElixir CSV (offline).")
    ap.add_argument("--csv", required=True, action="append", help="Path to OracleElixir match_data CSV. Can be passed multiple times.")
    ap.add_argument("--out", default="ml_artifact.json", help="Output artifact path (json)")
    ap.add_argument("--holdout-days", type=int, default=180, help="Hold out last N days for metrics/calibration")
    ap.add_argument(
        "--keep-playoffs",
        nargs="?",
        const="auto",
        choices=["auto", "yes", "no", "all"],
        default="auto",
        help="Filter playoffs rows. If you pass '--keep-playoffs' without a value, it behaves like 'auto'.",
    )
    ap.add_argument("--k", type=float, default=28.0, help="Elo K-factor. 28 validado no OOS puro (prequencial): ganho real pequeno same-league, inter ~flat. k>28 ganha so in-sample (frozen otimista).")
    ap.add_argument("--scale", type=float, default=300.0, help="Elo scale_points (denominator). 300 validado no caminho CALIBRADO (logloss 0.6112->0.6029, ECE ~0.038 ok). scale nao causa overfit in-sample (so reescala prob); ganho confirmado tambem no OOS cru.")
    ap.add_argument("--alpha", type=float, default=60.0, help="Shrink/k-decay coupling: inference shrink n/(n+alpha) + k-decay alpha/(alpha+n). Lower = more confident/stable. 60 validated best on the faithful artifact path (NOTA 7.40 vs 7.38@80, melhor ECE); prequential harness preferred ~25 but that was a false positive (sem calibracao).")
    ap.add_argument(
        "--unc-scale-c", type=float, default=0.0,
        help="Incerteza -> probabilidade mais chata (forma do Glicko). Infla a escala do "
             "Elo por sqrt(1+c*(u_blue+u_red)), u=n0/(n0+n). Quem tem pouco jogo puxa a "
             "previsao pra perto de 50%%. 0 = desligado (legado). Fixar no TREINO, nao na "
             "predicao: assim o calibrador reabsorve a parte global e sobra o efeito direcionado.",
    )
    ap.add_argument(
        "--unc-n-prior", type=float, default=50.0,
        help="Jogos para um time contar como 'conhecido' no fator de incerteza (u=n0/(n0+n)). "
             "50 vem do degrau medido no TI 2026. So tem efeito com --unc-scale-c > 0.",
    )
    ap.add_argument(
        "--sos-league",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive league base points from interleague Bradley-Terry (SOS) and bake into the artifact. "
             "Validated: improves cross-region (interleague) OOS logloss. Use --no-sos-league for ablation.",
    )
    ap.add_argument("--sos-scale", type=float, default=800.0, help="SOS league points = scale * BT theta. 800 re-validado no Elo novo (caminho calibrado): GERAL ll 0.6029->0.6021, inter ll 0.5875->0.5838, inter ECE 0.0424->0.0361 (melhora calib). 1000 piora ECE. Prior data-derived (BT) nao overfita como o cap.")
    ap.add_argument("--sos-min-games", type=int, default=20, help="Min interleague games for a league to get a data-derived prior.")
    ap.add_argument("--league-cap", type=float, default=75.0, help="league_strength_cap_abs (learned offset cap). 75 validado no caminho fiel (NOTA 7.40->7.43, melhor ECE inter). >90 melhora logloss mas degrada calibracao/seguranca (overfit do cross-region esparso).")
    ap.add_argument("--league-prior-weight", type=float, default=0.10, help="Peso do prior estatico de liga (league_base_points). Com SOS data-derived isso pode ser redundante; testar 0.")
    ap.add_argument("--tier-prior-weight", type=float, default=0.30, help="Peso do prior estatico de tier (tier_base_points).")
    ap.add_argument(
        "--dynamic-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dynamic K multiplier (Glicko-style uncertainty + volatility). Default ON: "
             "validated as a clear gain on the faithful artifact path AND pure prequential OOS "
             "in BOTH segments (inter-liga 0.6212->0.6091, mesma-liga 0.6407->0.6242). "
             "Use --no-dynamic-k to disable.",
    )
    ap.add_argument("--dynamic-k-games-prior", type=float, default=30.0, help="Prior games for dynamic uncertainty multiplier.")
    ap.add_argument("--dynamic-k-unc-max", type=float, default=1.80, help="Max uncertainty multiplier when games are scarce.")
    ap.add_argument("--dynamic-k-vol-ref", type=float, default=0.30, help="Reference volatility (abs error EWMA) for dynamic K.")
    ap.add_argument("--dynamic-k-vol-min", type=float, default=0.80, help="Min volatility multiplier for dynamic K.")
    ap.add_argument("--dynamic-k-vol-max", type=float, default=1.35, help="Max volatility multiplier for dynamic K.")
    ap.add_argument("--vol-ewma-beta", type=float, default=0.08, help="EWMA beta for per-team volatility tracking.")
    ap.add_argument("--mov-lambda", type=float, default=0.5, help="MOVDA (arXiv 2506.00348): peso do termo margem-de-vitoria (diff de kills) no update do Elo. 0.5 validado 2026-07-02 por DOIS metodos: caminho fiel (logloss 0.5903->0.5824, ECE 0.0342->0.0221, NOTA 8.43->8.59) E prequencial OOS puro (+0.06 nota @0.25-0.5; lambda>=1 degrada). 0=off (comportamento pre-MOVDA).")
    ap.add_argument("--core", choices=["elo", "whr"], default="whr",
                    help="Motor dos ratings. DEFAULT=whr desde 2026-07-07 (deploy da alavanca N: "
                         "fusion scorecard 8.31->9.34; rollback = --core elo). "
                         "'whr' = Whole-History Rating (Coulom 2008, alavanca N, "
                         "ACCEPT duplo 2026-07-06: +0.0114 logloss prequencial e +0.59 nota no caminho "
                         "fiel c/ Platt vs Elo+MOVDA). Exporta ratings pro mesmo formato de artefato "
                         "(params alpha=0/league_prior_weight=1/scale=400 reconstroem o protocolo validado). "
                         "MOVDA/dynamic-k não se aplicam ao fit whr.")
    ap.add_argument("--whr-w2", type=float, default=60.0,
                    help="Drift do WHR (Elo^2/dia). 60 tunado prequencialmente em 2025, eval cego 2026.")
    ap.add_argument("--calibrate", choices=["auto","yes","no"], default="auto", help="Platt calibration on holdout raw probs")
    ap.add_argument("--min-holdout", type=int, default=500, help="Min holdout size to fit calibrator when --calibrate=auto")
    ap.add_argument("--min-cal-gain", type=float, default=0.0, help="Minimum holdout logloss improvement required to keep calibration")

    # Players / lineup layer (optional)
    ap.add_argument(
        "--fit-players",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build players_artifact.json + fit w_lineup on holdout. Default ON: um DEPLOY sem isso "
             "derruba o meta.players_layer e o runtime cai no peso de UI (regressao silenciosa na fusao). "
             "Use --no-fit-players SO para experimentos Elo-only rapidos (nao deployar).",
    )
    ap.add_argument("--players-out", default="players_artifact.json", help="Output players artifact path (json)")
    ap.add_argument("--players-stats", default=None, help="OracleElixir byTournament player stats CSV (optional prior)")
    ap.add_argument("--players-half-life", type=float, default=180.0, help="Players rating half-life (days)")
    ap.add_argument("--players-k", type=float, default=0.15, help="Players online update step (k)")
    ap.add_argument("--players-shrink-m", type=float, default=20.0, help="Shrink m for effective rating (gp/(gp+m))")
    ap.add_argument("--wlineup-l2", type=float, default=0.0, help="L2 regularization for w_lineup")
    ap.add_argument("--wlineup-max-abs", type=float, default=3.0, help="Clip fitted w_lineup to [-max,+max] to avoid overfit spikes")

    # Achievements
    ap.add_argument("--achievements", default="achievements.json", help="Path to achievements.json (optional)")
    ap.add_argument("--learn-ach", action="store_true", help="Learn achievement weights from achievements.json (if any entries)")
    ap.add_argument("--ach-l2", type=float, default=0.001, help="L2 regularization for ach weights learning")
    ap.add_argument("--ach-iters", type=int, default=80, help="Gradient descent iterations for ach weights learning")
    ap.add_argument("--ach-lr", type=float, default=0.5, help="Learning rate for ach weights learning")
    ap.add_argument("--ach-max-abs", type=float, default=250.0, help="Clamp abs(weight) for stability")
    ap.add_argument("--walkforward-report", default="", help="Optional output json path for monthly walk-forward report")

    args = ap.parse_args()

    if int(args.holdout_days) < 1:
        raise SystemExit("--holdout-days must be >= 1")
    if float(args.scale) <= 0:
        raise SystemExit("--scale must be > 0")
    if float(args.k) <= 0:
        raise SystemExit("--k must be > 0")
    if float(args.dynamic_k_games_prior) <= 0:
        raise SystemExit("--dynamic-k-games-prior must be > 0")
    if float(args.dynamic_k_unc_max) < 1.0:
        raise SystemExit("--dynamic-k-unc-max must be >= 1.0")
    if float(args.dynamic_k_vol_ref) <= 0:
        raise SystemExit("--dynamic-k-vol-ref must be > 0")
    if float(args.dynamic_k_vol_min) <= 0:
        raise SystemExit("--dynamic-k-vol-min must be > 0")
    if float(args.dynamic_k_vol_max) < float(args.dynamic_k_vol_min):
        raise SystemExit("--dynamic-k-vol-max must be >= --dynamic-k-vol-min")
    if not (0.0 < float(args.vol_ewma_beta) <= 1.0):
        raise SystemExit("--vol-ewma-beta must be in (0, 1]")

    # Resolve keep_playoffs behavior
    keep_playoffs: Optional[bool] = None
    if args.keep_playoffs == "yes":
        keep_playoffs = True
    elif args.keep_playoffs == "no":
        keep_playoffs = False
    elif args.keep_playoffs == "all":
        keep_playoffs = None
    else:
        keep_playoffs = None  # 'auto'

    csv_paths = [str(Path(p).expanduser()) for p in (args.csv or [])]
    dfs: List[pd.DataFrame] = []
    for p in csv_paths:
        df = load_oracle_elixir_games(p, keep_playoffs=keep_playoffs)
        if df.empty:
            continue
        df2 = df.dropna(how="all").dropna(axis=1, how="all")
        if df2.empty or len(df2.columns) == 0:
            continue
        df2["_src_csv"] = str(p)
        dfs.append(df2)

    games = pd.concat(dfs, ignore_index=True, join="inner") if dfs else pd.DataFrame()
    if games.empty:
        raise SystemExit("Nenhum jogo carregado dos CSVs informados.")

    # Normalize / dedupe
    games["gameid"] = games["gameid"].astype(str)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).reset_index(drop=True)
    games = games.drop_duplicates(subset=["gameid"], keep="last").reset_index(drop=True)
    games = games.sort_values(["date", "gameid"]).reset_index(drop=True)

    data_min_dt = games["date"].min()
    data_max_dt = games["date"].max()

    # Train/holdout split
    train_df, test_df, cutoff_dt = _date_split_holdout(games, int(args.holdout_days))

    # Infer domestic league per team-year (for league/tier-aware updates)
    domestic_by_year = _infer_team_domestic_league_by_year(games)
    team_meta_latest = _infer_team_meta_latest(domestic_by_year)

    # Add home league/tier columns to train/holdout
    train_fit = _augment_games_with_home_meta(train_df, domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)
    test_fit = _augment_games_with_home_meta(test_df, domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)

    # -----------------------------
    # Fit Elo residuals + learned offsets (v0.4+)
    # -----------------------------
    model = EloModel(
        EloParams(
            k=float(args.k),
            scale_points=float(args.scale),
            alpha=float(args.alpha),
            unc_scale_c=float(args.unc_scale_c),
            unc_n_prior=float(args.unc_n_prior),
            init_rating=0.0,
            league_prior_weight=float(args.league_prior_weight),
            tier_prior_weight=float(args.tier_prior_weight),
            league_strength_cap_abs=float(args.league_cap),
            dynamic_k_enabled=bool(args.dynamic_k),
            dynamic_k_games_prior=float(args.dynamic_k_games_prior),
            dynamic_k_uncertainty_mult_max=float(args.dynamic_k_unc_max),
            dynamic_k_volatility_ref=float(args.dynamic_k_vol_ref),
            dynamic_k_volatility_mult_min=float(args.dynamic_k_vol_min),
            dynamic_k_volatility_mult_max=float(args.dynamic_k_vol_max),
            volatility_ewma_beta=float(args.vol_ewma_beta),
            mov_lambda=float(getattr(args, "mov_lambda", 0.0) or 0.0),
        )
    )
    model.set_team_meta(team_meta_latest)

    # SOS league strength (data-derived priors, baked into the artifact). Computed
    # from TRAIN interleague games (Bradley-Terry, controls for strength of
    # schedule) and set BEFORE the fit so the learned offset stays consistent with
    # it (no double counting). See mlcore/league_sos.py. Validated to improve
    # cross-region OOS logloss; falls back to global static priors per-league.
    if bool(getattr(args, "sos_league", True)):
        try:
            _inter = train_fit[train_fit["blue_league"].astype(str) != train_fit["red_league"].astype(str)]
            _pairs = list(zip(
                _inter["blue_league"].astype(str), _inter["red_league"].astype(str),
                pd.to_numeric(_inter["blue_win"], errors="coerce").fillna(0).astype(float),
            ))
            sos_map = sos_league_base_points(_pairs, scale=float(args.sos_scale), min_games=int(args.sos_min_games))
            model.league_base_points_map = sos_map
            print(f"[train_offline] SOS league priors: {len(sos_map)} leagues (data-derived).")
        except Exception as e:
            model_warnings_sos = f"sos_league_failed: {type(e).__name__}: {e}"
            print("[train_offline]", model_warnings_sos)

    last_tier: Dict[str, str] = {}

    train_fit = train_fit.sort_values(["date", "gameid"]).reset_index(drop=True)
    if str(getattr(args, "core", "elo")) == "whr":
        # ------------------ WHR (alavanca N) ------------------
        # Fit MAP do histórico completo; depois EXPORTA os ratings finais pro
        # EloModel acima. Pra reconstruir EXATAMENTE o protocolo validado
        # (p = logistic10((R_a-R_b)/400), prior de liga = SOS no 1º ponto):
        #   ratings_armazenados = R_whr - pontos_da_liga (o p_blue soma o prior
        #   de volta via league_prior_weight=1.0), alpha=0 (sem shrink),
        #   tier_prior_weight=0, scale=400, dynk/MOVDA off.
        from .whr_core import WHRFitter
        whr = WHRFitter(w2_elo_per_day=float(args.whr_w2),
                        league_prior_points=dict(model.league_base_points_map))
        _n_fit = 0
        # Predições PREQUENCIAIS (prever ANTES de adicionar o jogo) do último
        # ano de treino — é nelas que o Platt do artefato é fitado (a receita
        # validada no caminho fiel). Fitar no holdout de 30d overfita o
        # intercepto da janela (bug 2026-07-07: média p_blue 0.573, ECE 0.089).
        _preq_cut = train_fit["date"].max() - pd.Timedelta(days=365)
        whr_preq_y: List[int] = []
        whr_preq_p: List[float] = []
        for r in train_fit.itertuples(index=False):
            day = int(r.date.toordinal())
            _b, _r = str(r.blue), str(r.red)
            if r.date >= _preq_cut and whr.n_games.get(_b, 0) >= 5 and whr.n_games.get(_r, 0) >= 5:
                # o achatamento por incerteza tem que entrar AQUI tambem, senao o
                # Platt e fitado numa distribuicao que o runtime nao usa e a parte
                # global do efeito vira vies em vez de ser reabsorvida
                _p_preq = float(whr.p_blue(_b, _r))
                if float(args.unc_scale_c) > 0.0:
                    from .elo import flatten_p_by_uncertainty
                    _p_preq = flatten_p_by_uncertainty(
                        _p_preq, whr.n_games.get(_b, 0), whr.n_games.get(_r, 0),
                        float(args.unc_scale_c), float(args.unc_n_prior))
                whr_preq_p.append(_p_preq)
                whr_preq_y.append(int(r.blue_win))
            whr.add_game(_b, _r, int(r.blue_win), day,
                         blue_league=str(r.blue_league), red_league=str(r.red_league))
            whr.refresh_pair(_b, _r, iters=2)
            _n_fit += 1
            if _n_fit % 400 == 0:
                whr.full_sweeps(sweeps=1, tol_elo=0.0)
        _sweeps = whr.full_sweeps(sweeps=12, tol_elo=0.5, iters=1)
        print(f"[train_offline] WHR fit: {_n_fit} jogos, convergiu em {_sweeps} varreduras finais; "
              f"{len(whr_preq_y)} predições prequenciais p/ calibração.")

        # Injeta no EloModel (mesmo formato de artefato/runtime). Os ratings
        # vão COMPLETOS (prior de liga já embutido pelo 1º ponto do WHR) e
        # league_prior_weight=0: p_blue vira diferença pura de ratings, IMUNE
        # a hints de liga divergentes entre chamadores (bug flagrado 2026-07-07:
        # com lpw=1.0 e rating descontado, cada régua/tela reconstruía uma soma
        # diferente — ECE 0.019 no fusion mas 0.091 no alarme). Custo aceito:
        # time NUNCA visto no treino perde o prior de liga (fica rating 0).
        for _t, _r_elo in whr.ratings_elo().items():
            model.ratings[_t] = float(_r_elo)
            model.games_played[_t] = int(whr.n_games.get(_t, 0))
        model.params.alpha = 0.0
        model.params.league_prior_weight = 0.0
        model.params.tier_prior_weight = 0.0
        model.params.scale_points = 400.0
        model.params.dynamic_k_enabled = False
        model.params.mov_lambda = 0.0
        args.scale = 400.0  # calibração/holdout usam a mesma escala do runtime
    else:
        for r in train_fit.itertuples(index=False):
            blue = str(r.blue)
            red = str(r.red)
            bw = int(r.blue_win)
            bl = str(r.blue_league)
            rl = str(r.red_league)
            bt = str(r.blue_tier)
            rt = str(r.red_tier)

            _maybe_reset_on_promotion(model, blue, bt, last_tier=last_tier)
            _maybe_reset_on_promotion(model, red, rt, last_tier=last_tier)

            # MOV (diff de kills do blue) — só tem efeito com --mov-lambda > 0.
            _mov_val = None
            try:
                _bk = float(getattr(r, "blue_teamkills"))
                _rk = float(getattr(r, "red_teamkills"))
                if math.isfinite(_bk) and math.isfinite(_rk):
                    _mov_val = _bk - _rk
            except (TypeError, ValueError, AttributeError):
                _mov_val = None

            model.update(blue, red, bw, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt, mov=_mov_val)

    # Center offsets and compute ranking caps (ranking-only guardrail)
    model_warnings: List[str] = []
    try:
        model._center_offsets()
    except Exception as e:
        model_warnings.append(f"center_offsets_failed: {type(e).__name__}: {e}")
    try:
        model.rank_caps_by_tier = model.compute_rank_caps(team_meta_latest)
    except Exception as e:
        model_warnings.append(f"compute_rank_caps_failed: {type(e).__name__}: {e}")

    # -----------------------------
    # Achievements layer (optional)
    # -----------------------------
    ach_path = Path(str(args.achievements)).expanduser()
    ach_data = load_achievements(ach_path)
    weights_points: Dict[str, float] = {}
    ach_report: Dict[str, Any] = {"enabled": False}

    if bool(args.learn_ach) and (ach_data.get("entries") or []):
        try:
            # Wrap model to satisfy learn_weights_points expectations
            class _WrapModelForAch:
                def __init__(self, m: EloModel, team_meta: Dict[str, Dict[str, str]]):
                    self._m = m
                    self._tm = dict(team_meta or {})

                def meta(self, team: str) -> Dict[str, str]:
                    return dict(self._tm.get(str(team)) or {})

                def total_strength(self, team: str, league: str | None = None, tier: str | None = None) -> float:
                    return float(self._m.base_strength(str(team), league=league, tier=tier, apply_rank_caps=False))

            wrap = _WrapModelForAch(model, team_meta_latest)

            train_rows: List[Dict[str, Any]] = []
            for r in train_fit.itertuples(index=False):
                train_rows.append(
                    {
                        "blue": str(r.blue),
                        "red": str(r.red),
                        "blue_win": int(r.blue_win),
                        "date": r.date,
                    }
                )

            weights_points, ach_report = learn_weights_points(
                train_rows=train_rows,
                model=wrap,
                achievements_data=ach_data,
                init_weights=None,
                scale_points=float(args.scale),
                l2=float(args.ach_l2),
                iters=int(args.ach_iters),
                lr=float(args.ach_lr),
                max_abs=float(args.ach_max_abs),
                verbose=True,
            )
            ach_report["enabled"] = True
        except Exception as e:
            weights_points = {}
            ach_report = {"enabled": False, "error": str(e)}

    # -----------------------------
    # Holdout metrics / calibration
    # -----------------------------
    y: List[int] = []
    p_raw: List[float] = []
    holdout_gameids: List[str] = []

    team_idx = None
    if not test_fit.empty:
        from .achievements import build_team_index, team_boost_points
        team_idx = build_team_index(ach_data)

        for r in test_fit.itertuples(index=False):
            b = str(r.blue)
            rr = str(r.red)
            bl = str(r.blue_league)
            rl = str(r.red_league)
            bt = str(r.blue_tier)
            rt = str(r.red_tier)

            # Base diff (includes learned league/tier offsets + shrink residual)
            sb = float(model.base_strength(b, league=bl, tier=bt, apply_rank_caps=False))
            sr = float(model.base_strength(rr, league=rl, tier=rt, apply_rank_caps=False))
            diff = float(sb - sr + float(getattr(model, "side_adv_points", 0.0)))

            # Achievements diff (optional)
            ach_diff = 0.0
            if weights_points and team_idx is not None:
                as_of = r.date
                if isinstance(as_of, datetime):
                    as_of_d = as_of.date()
                elif isinstance(as_of, date):
                    as_of_d = as_of
                else:
                    as_of_d = datetime.now(timezone.utc).date()
                ach_diff = float(team_boost_points(b, as_of_d, team_idx, weights_points) - team_boost_points(rr, as_of_d, team_idx, weights_points))
            diff = float(diff + ach_diff)

            y.append(int(r.blue_win))
            p_raw.append(_p_from_diff_points(diff, float(args.scale)))
            holdout_gameids.append(str(r.gameid))

    cal = PlattCalibrator()
    calibrated = False
    cal_reason = "disabled"
    # Core WHR: Platt fitado nas predições PREQUENCIAIS do último ano de treino
    # (receita do caminho fiel; ECE validada). O holdout curto fica só como
    # métrica OOS — fitar nele overfita o intercepto da janela.
    # --calibrate no é RESPEITADO (Dota mediu Platt piorando no blind e deploya
    # raw; antes este branch ignorava a flag — bug corrigido 2026-07-09).
    if (str(getattr(args, "core", "elo")) == "whr" and args.calibrate != "no"
            and len(locals().get("whr_preq_y") or []) >= 500):
        cal = fit_platt_from_raw_probs(whr_preq_y, whr_preq_p)
        calibrated = True
        cal_reason = f"whr_prequential_n={len(whr_preq_y)}"
    elif args.calibrate == "yes" or (args.calibrate == "auto" and len(y) >= int(args.min_holdout)):
        if len(y) > 0:
            cal_try = fit_platt_from_raw_probs(y, p_raw)
            p_cal_try = [float(cal_try.predict_proba(p)) for p in p_raw]
            ll_raw = float(logloss(y, p_raw))
            ll_cal = float(logloss(y, p_cal_try))
            gain = float(ll_raw - ll_cal)
            if gain >= float(args.min_cal_gain):
                cal = cal_try
                calibrated = True
                cal_reason = f"accepted_gain={gain:.6f}"
            else:
                cal = PlattCalibrator()
                calibrated = False
                cal_reason = f"rejected_gain={gain:.6f}<min_cal_gain={float(args.min_cal_gain):.6f}"
    else:
        cal_reason = "not_requested_or_insufficient_holdout"

    p_cal = [cal.predict_proba(p) for p in p_raw] if len(p_raw) else []

    rel_raw = _reliability_ece(y, p_raw, n_bins=10) if len(y) else {"ece": None, "bins": []}
    rel_cal = _reliability_ece(y, p_cal, n_bins=10) if len(y) else {"ece": None, "bins": []}

    metrics = {
        "holdout_n": len(y),
        "logloss_raw": float(logloss(y, p_raw)) if len(y) else None,
        "logloss_cal": float(logloss(y, p_cal)) if len(y) else None,
        "brier_raw": float(brier(y, p_raw)) if len(y) else None,
        "brier_cal": float(brier(y, p_cal)) if len(y) else None,
        "calibrated": bool(calibrated),
        "calibration_reason": str(cal_reason),
        "min_cal_gain": float(args.min_cal_gain),
        "cal_a": float(cal.a),
        "cal_b": float(cal.b),
        "ece_raw": rel_raw.get("ece"),
        "ece_cal": rel_cal.get("ece"),
        "reliability_bins_raw": rel_raw.get("bins"),
        "reliability_bins_cal": rel_cal.get("bins"),
    }

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_min_date": data_min_dt.date().isoformat() if pd.notna(data_min_dt) else None,
        "data_max_date": data_max_dt.date().isoformat() if pd.notna(data_max_dt) else None,
        "cutoff_date": cutoff_dt.date().isoformat() if cutoff_dt is not None and pd.notna(cutoff_dt) else None,
        "n_games_total": len(games),
        "n_games_train": len(train_df),
        "n_games_holdout": len(test_df),
        "n_teams": len(team_meta_latest),
        "csv": list(csv_paths),
        "keep_playoffs": keep_playoffs,
        "holdout_days": int(args.holdout_days),
        "core": str(getattr(args, "core", "elo")),
        "whr_w2": float(getattr(args, "whr_w2", 0.0)) if str(getattr(args, "core", "elo")) == "whr" else None,
        "params": {
            "k": float(args.k),
            "scale_points": float(args.scale),
            "alpha": float(getattr(getattr(model, "params", None), "alpha", 80.0) or 80.0),
            "k_league": float(getattr(getattr(model, "params", None), "k_league", 6.0) or 6.0),
            "k_tier": float(getattr(getattr(model, "params", None), "k_tier", 3.0) or 3.0),
            "dynamic_k_enabled": bool(getattr(getattr(model, "params", None), "dynamic_k_enabled", False)),
            "dynamic_k_games_prior": float(getattr(getattr(model, "params", None), "dynamic_k_games_prior", 30.0) or 30.0),
            "dynamic_k_uncertainty_mult_max": float(
                getattr(getattr(model, "params", None), "dynamic_k_uncertainty_mult_max", 1.80) or 1.80
            ),
            "dynamic_k_volatility_ref": float(getattr(getattr(model, "params", None), "dynamic_k_volatility_ref", 0.30) or 0.30),
            "dynamic_k_volatility_mult_min": float(
                getattr(getattr(model, "params", None), "dynamic_k_volatility_mult_min", 0.80) or 0.80
            ),
            "dynamic_k_volatility_mult_max": float(
                getattr(getattr(model, "params", None), "dynamic_k_volatility_mult_max", 1.35) or 1.35
            ),
            "volatility_ewma_beta": float(getattr(getattr(model, "params", None), "volatility_ewma_beta", 0.08) or 0.08),
        },
        "team_meta": team_meta_latest,
        "metrics": metrics,
        "tiers_note": "v0.4+: base_strength = residual_eff + league_strength(league) + tier_strength(tier); residual_eff uses shrink n/(n+alpha).",
        "model_warnings": model_warnings,
        "achievements": {
            "path": str(ach_path),
            "weights_points": weights_points,
            "learn_report": ach_report,
            "note": "achievements.json is external; weights are learned + stored in artifact meta",
        },
        "note": "v0.4.0: Elo residuals + learned league/tier offsets (no fixed baselines) + shrink(alpha) + optional achievements + optional Platt calibration + optional players layer.",
    }

    # Optional: walk-forward report to monitor drift and quality by league over time.
    if str(args.walkforward_report or "").strip():
        try:
            wf = run_walkforward(csv_paths, min_train_games=max(500, int(args.min_holdout)), k=float(args.k), scale=float(args.scale))
            wf_out = Path(str(args.walkforward_report)).expanduser()
            wf_out.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
            meta["walkforward_report"] = {
                "path": str(wf_out),
                "overall": wf.get("overall", {}),
                "generated_at": wf.get("generated_at"),
            }
        except Exception as e:
            meta["walkforward_report"] = {"error": str(e)}

    # -----------------------------
    # Optional: players layer (lineup adjustment)
    # -----------------------------
    if bool(args.fit_players):
        try:
            # Load player rows once (all CSVs), then filter by gameids
            player_rows = load_match_player_rows(csv_paths, keep_playoffs=keep_playoffs)

            # Fit prior from byTournament stats (optional)
            prior_beta = None
            prior_scores = {}
            prior_meta = {}
            if args.players_stats:
                try:
                    stats = load_bytournament_stats(str(args.players_stats))
                    prior_beta, prior_scores = fit_prior_beta(
                        stats,
                        player_rows=player_rows,
                        games=train_df[["gameid", "blue_win"]],
                        l2=1e-3,
                        iters=60,
                    )
                    prior_meta = {
                        "source": str(args.players_stats),
                        "features": ["z_gd10", "z_xpd10", "z_csd10", "z_dmgp", "z_dthp"],
                        "beta": prior_beta,
                        "n_rows": len(stats) if stats is not None else 0,
                    }
                except Exception as e:
                    prior_beta = None
                    prior_scores = {}
                    prior_meta = {"source": str(args.players_stats), "error": str(e)}

            # Build players artifact using TRAIN games only (chronological)
            p_art = PlayersArtifact(
                half_life_days=float(args.players_half_life),
                k=float(args.players_k),
                shrink_m=float(args.players_shrink_m),
                prior_scores=prior_scores,
                prior_meta=prior_meta,
            )

            train_g = train_df.sort_values("date").copy()
            train_gids = set(train_g["gameid"].astype(str).tolist())
            pr_train = player_rows[player_rows["gameid"].astype(str).isin(train_gids)].copy()

            for rr in train_g.itertuples(index=False):
                gid = str(rr.gameid)
                blue = str(rr.blue)
                red = str(rr.red)
                bw = int(rr.blue_win)

                b_players, r_players = lineup_for_game(pr_train, gid)
                p_art.update_game(
                    game_date=pd.to_datetime(rr.date).date(),
                    league=str(getattr(rr, "league", "") or "") or None,
                    blue_lineup=b_players,
                    red_lineup=r_players,
                    blue_win=bw,
                )

            # Fit w_lineup using HOLDOUT games (model diffs as features)
            pl_meta: Dict[str, Any] = {"enabled": True}

            test_g = test_df.sort_values("date").copy()
            test_gids = set(test_g["gameid"].astype(str).tolist())
            pr_hold = player_rows[player_rows["gameid"].astype(str).isin(test_gids)].copy()

            ys_pl: List[int] = []
            logit_base_pl: List[float] = []
            delta_pl: List[float] = []
            cov_pl: List[float] = []

            for rr in test_g.itertuples(index=False):
                gid = str(rr.gameid)
                blue = str(rr.blue)
                red = str(rr.red)
                bw = int(rr.blue_win)

                blue_lineup, red_lineup = lineup_for_game(pr_hold, gid)

                # Players delta in logit units + coverage (0..1)
                res = compute_delta_and_confidence(
                    p_art,
                    blue_lineup=blue_lineup,
                    red_lineup=red_lineup,
                    as_of=pd.to_datetime(rr.date).date(),
                    match_league=str(getattr(rr, "league", "") or "") or None,
                )
                d = float(res.get("delta", 0.0) or 0.0)
                cov = float(res.get("coverage", 0) or 0.0) / float(res.get("coverage_total", 5) or 5.0)
                d_eff = float(d * cov)

                # Use ML core raw p (un-calibrated) as baseline feature
                # NOTE: for holdout we use match league as fallback, but prefer inferred home leagues when possible.
                yy = _safe_int_year(getattr(rr, "year", None))
                bl = _resolve_home_league(team=blue, year=yy, match_league=str(getattr(rr, "league", "") or "OTHER"), domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)
                rl = _resolve_home_league(team=red, year=yy, match_league=str(getattr(rr, "league", "") or "OTHER"), domestic_by_year=domestic_by_year, team_meta_latest=team_meta_latest)
                bt = str(tier_for_league(bl))
                rt = str(tier_for_league(rl))

                p0 = float(model.p_blue(blue, red, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt))
                ys_pl.append(int(bw))
                logit_base_pl.append(float(_logit(p0)))
                delta_pl.append(float(d_eff))
                cov_pl.append(float(cov))

            w_fit_raw = float(fit_w_lineup(y=ys_pl, logit_base=logit_base_pl, delta=delta_pl, l2=float(args.wlineup_l2)))
            wmax = max(0.0, float(args.wlineup_max_abs))
            # Piso em 0: w_lineup NEGATIVO significa "apostar contra o lineup melhor" —
            # sem base teórica, é ruído do holdout curto. Visto na prática em 2026-07-02:
            # após o MOVDA o core absorveu parte do sinal de desempenho e o fit do holdout
            # escolheu w=-0.094, que PIOROU a fusão na season inteira (0.5858→0.5937).
            # Com piso, o pior caso é a fusão == core (players neutro). w_fit_raw fica
            # registrado no meta para observabilidade.
            w_fit = max(0.0, min(wmax, float(w_fit_raw)))

            # Optional: calibrator after applying lineup adjustment (Platt on holdout)
            cal_after_lineup = None
            if len(ys_pl) >= int(args.min_holdout) and any(abs(x) > 1e-9 for x in delta_pl):
                p1_raw = [float(_sigmoid(lb + float(w_fit) * di)) for lb, di in zip(logit_base_pl, delta_pl)]
                cal2 = fit_platt_from_raw_probs(ys_pl, p1_raw)
                cal_after_lineup = {"a": float(cal2.a), "b": float(cal2.b)}
            # Save players artifact
            p_out = Path(str(args.players_out)).expanduser()
            p_art.save_json(str(p_out))

            pl_meta.update(
                {
                    "players_artifact": str(p_out),
                    "w_lineup": float(w_fit),
                    "w_lineup_raw": float(w_fit_raw),
                    "w_lineup_clipped": bool(abs(float(w_fit_raw)) > abs(float(w_fit))),
                    "wlineup_max_abs": float(args.wlineup_max_abs),
                    "l2": float(args.wlineup_l2),
                    "avg_coverage": float(sum(cov_pl) / len(cov_pl)) if cov_pl else 0.0,
                    "cal_after_lineup": cal_after_lineup,
                    "holdout_n_used": len(ys_pl),
                    "n_train_games": len(train_g),
                    "n_holdout_games": len(test_g),
                    "prior": prior_meta,
                }
            )
            meta["players_layer"] = pl_meta
        except Exception as e:
            meta["players_layer"] = {"enabled": False, "error": str(e)}
            print("[train_offline] players_layer FAILED:", str(e))

    # Save artifact
    art = MLArtifact(version="0.4.0", model=model, calibrator=cal, meta=meta)
    art.save_json(args.out)

    print("[train_offline] saved:", args.out)
    print(json.dumps(metrics, indent=2))

    print("\nTop GLOBAL ranking (tier travado: T1 > T2 > ERL > MINOR):")
    for team, g, league, tier in _top_global_v04(model, team_meta_latest, 20, tier_lock=True):
        print(f"{team:30s}  gp={g:5d}  {league:10s}  {tier:6s}")

if __name__ == "__main__":
    main()
