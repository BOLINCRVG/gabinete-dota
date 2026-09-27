from __future__ import annotations

"""Retrain-check heuristics for ML artifact.

Goal: tell you when the artifact is likely stale enough that (full) re-training helps.

This file is also used by `mlcore_app.py`, so it exports:
- CheckConfig
- run_check

Design note (important):
- We DO NOT recommend full retraining just because the calibrator "drifted".
  Calibration can drift early in a season with very small samples. We surface drift
  as a diagnostic, but the retrain decision is based on:
    (a) age of artifact,
    (b) volume of new games since last train,
    (c) recent calibrated logloss degrading vs the artifact's own holdout logloss.
"""

import sys

# Console cp1252 no Windows derruba print com acento — e este modulo agora
# imprime a mensagem de "CSV parado", que tem travessao. O proprio guarda
# test_sem_falha_silenciosa pegou isto no meu codigo novo (16/08/2026).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .artifact import MLArtifact
from .io_oracle import load_oracle_elixir_games
from .tiers import tier_for_league
from .utils import logloss


# -----------------------
# Config + helpers
# -----------------------


def _resolve_csv_path(p: Path) -> Path:
    """Accept a CSV file path OR a directory. If directory, picks newest *.csv."""
    try:
        p = Path(p)
    except Exception:
        return p
    if p.exists() and p.is_dir():
        cands = sorted(p.glob('*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)
        if not cands:
            raise FileNotFoundError(f"No .csv files found in directory: {p}")
        chosen = cands[0]
        print(f"[retrain_check] csv_path is a directory; using newest CSV: {chosen}")
        return chosen
    return p

@dataclass
class CheckConfig:
    # Policy thresholds (used by mlcore_app sliders)
    max_days: int = 14                  # retrain if artifact older than this
    # CSV PARADO: o dado em si envelhecendo, mesmo com o artifact recem-treinado.
    # Achado 16/08/2026 (pergunta dele sobre a data de referencia): a data de
    # decaimento e HOJE, nao o ultimo jogo do CSV — o que esta CERTO, senao jogo
    # velho pareceria novo. Mas se o CSV congela, `hoje` avanca e envelhece tudo
    # sozinho, e NENHUM dos outros gatilhos dispara: o artifact continua novo e
    # "jogos novos" fica em zero. Com half-life de 60d, 14 dias parado ja tira 15%
    # do peso do dado mais recente, e 30 dias tiram 29% — sem sintoma visivel.
    # A cota do Google Drive ja travou o upload dele varias vezes: e cenario real.
    max_csv_stale_days: int = 5
    min_new_games: int = 1200           # retrain if >= this many new maps since last train

    # Quality window (recent tail)
    recent_n: int = 2000                # evaluate recent logloss on last N games by date
    drift_logloss_delta: float = 0.025  # retrain if recent logloss worse than train-holdout by >= this

    # Guardrail: don't draw strong conclusions from tiny samples
    min_recent_eval: int = 250          # minimum number of games to evaluate quality-based retrain


def _parse_iso_dt(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _team_meta_from_artifact(art: MLArtifact) -> Dict[str, Dict[str, str]]:
    # Prefer model.team_meta (newer), fallback to meta.team_meta
    tm = getattr(getattr(art, "model", None), "team_meta", None)
    if isinstance(tm, dict) and tm:
        out: Dict[str, Dict[str, str]] = {}
        for t, m in tm.items():
            mm = m or {}
            out[str(t)] = {"league": str(mm.get("league", "OTHER")), "tier": str(mm.get("tier", "OTHER"))}
        return out

    tm2 = (art.meta or {}).get("team_meta") or {}
    if isinstance(tm2, dict) and tm2:
        out2: Dict[str, Dict[str, str]] = {}
        for t, m in tm2.items():
            mm = m or {}
            out2[str(t)] = {"league": str(mm.get("league", "OTHER")), "tier": str(mm.get("tier", "OTHER"))}
        return out2

    return {}


def _recent_metrics(
    art: MLArtifact,
    games: pd.DataFrame,
    team_meta: Dict[str, Dict[str, str]],
    *,
    recent_n: int,
) -> Tuple[int, float, float]:
    """Return (n, recent_logloss_cal, drift_p95(|p_cal - p_raw|))."""
    df = games.sort_values(["date", "gameid"]).tail(int(recent_n)).copy()
    if df.empty:
        return 0, float("nan"), float("nan")

    y: List[int] = []
    p_raw: List[float] = []
    p_cal: List[float] = []

    for r in df.itertuples(index=False):
        b = str(r.blue)
        rr = str(r.red)
        lg = str(getattr(r, "league", "OTHER") or "OTHER")
        mb = team_meta.get(b) or {}
        mr = team_meta.get(rr) or {}

        bl = str(mb.get("league") or lg or "OTHER")
        rl = str(mr.get("league") or lg or "OTHER")
        bt = str(mb.get("tier") or tier_for_league(bl))
        rt = str(mr.get("tier") or tier_for_league(rl))

        pr = float(art.model.p_blue(b, rr, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt))
        try:
            pc = float(art.calibrator.predict_proba(pr))
        except Exception:
            pc = pr

        y.append(int(r.blue_win))
        p_raw.append(pr)
        p_cal.append(pc)

    ll = float(logloss(y, p_cal)) if y else float("nan")

    diffs = sorted(abs(a - b) for a, b in zip(p_cal, p_raw))
    if not diffs:
        return len(y), ll, float("nan")

    # p95
    k = round(0.95 * (len(diffs) - 1))
    drift_p95 = float(diffs[max(0, min(len(diffs) - 1, k))])

    return len(y), ll, drift_p95


# -----------------------
# Main decision function
# -----------------------

def check_retrain_needed(
    *,
    artifact_path: str,
    csv_path: str,
    config: Optional[CheckConfig] = None,
    keep_playoffs: bool = True,
) -> Dict[str, Any]:
    cfg = config or CheckConfig()

    art = MLArtifact.load_json(artifact_path)
    games = load_oracle_elixir_games(csv_path, keep_playoffs=keep_playoffs)
    if games.empty:
        return {"need_retrain": False, "reasons": ["csv has 0 games"], "stats": {}}

    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).reset_index(drop=True)

    team_meta = _team_meta_from_artifact(art)
    trained_at_meta = _parse_iso_dt(str((art.meta or {}).get("trained_at") or ""))
    try:
        art_mtime = datetime.fromtimestamp(Path(artifact_path).stat().st_mtime, tz=timezone.utc)
    except Exception:
        art_mtime = datetime.now(timezone.utc)
    # Use whichever is more recent: meta timestamp or file modification time.
    trained_at = art_mtime if trained_at_meta is None else max(trained_at_meta, art_mtime)
    # "Jogos novos" tem que ser contado a partir do que os RATINGS viram, nao do
    # fim dos dados: quando o artifact vem de um train_offline, o holdout ficou de
    # fora do fit e esses jogos SAO novos para o modelo. Usar data_max_date aqui
    # subestimava a defasagem — foi assim que o incidente holdout-180 (junho/2026)
    # passou 4 dias despercebido, com os ratings congelados em dezembro.
    # Artifact sem a chave nova cai em data_max_date, comportamento de antes.
    _m = art.meta or {}
    data_max = (_parse_iso_dt(str(_m.get("ratings_max_date") or ""))
                or _parse_iso_dt(str(_m.get("data_max_date") or "")))

    now = datetime.now(timezone.utc)
    days_since = int((now - trained_at).total_seconds() / 86400)

    # New games count since artifact's max date
    if data_max is None:
        new_games = len(games)
    else:
        new_games = int((games["date"] > pd.Timestamp(data_max)).sum())

    # Recent quality metrics (tail)
    n_recent, ll_recent, drift_p95 = _recent_metrics(art, games, team_meta, recent_n=int(cfg.recent_n))

    # Baseline (at train time) - compare recent vs holdout-calibrated
    baseline_ll = None
    try:
        baseline_ll = float(((art.meta or {}).get("metrics") or {}).get("logloss_cal"))
    except Exception:
        baseline_ll = None

    ll_delta = float("nan")
    if (baseline_ll is not None) and (ll_recent == ll_recent):
        ll_delta = float(ll_recent - baseline_ll)

    # Quantos dias faz que o CSV nao recebe jogo novo (independe do artifact).
    csv_stale_days = None
    csv_last_date = "?"
    try:
        _ult = pd.to_datetime(games["date"], errors="coerce").max()
        if pd.notna(_ult):
            csv_last_date = _ult.date().isoformat()
            csv_stale_days = int((datetime.now(timezone.utc).date() - _ult.date()).days)
    except (KeyError, TypeError, ValueError):
        csv_stale_days = None

    reasons: List[str] = []

    # Policy-based triggers
    if days_since >= int(cfg.max_days):
        reasons.append(f"artifact is {days_since} days old (>= {cfg.max_days})")
    if new_games >= int(cfg.min_new_games):
        reasons.append(f"{new_games} new games since last train (>= {cfg.min_new_games})")
    if csv_stale_days is not None and csv_stale_days >= int(cfg.max_csv_stale_days):
        reasons.append(
            f"CSV parado ha {csv_stale_days} dias (ultimo jogo {csv_last_date}); o decaimento "
            f"usa HOJE como referencia, entao o dado perde peso sozinho — reponha o CSV"
        )

    # Quality-based trigger (only if we have enough recent samples)
    if (baseline_ll is not None) and (ll_delta == ll_delta) and (n_recent >= int(cfg.min_recent_eval)):
        if ll_delta >= float(cfg.drift_logloss_delta):
            reasons.append(
                f"recent calibrated logloss drifted by +{ll_delta:.4f} (>= {cfg.drift_logloss_delta}) "
                f"vs train-holdout ({baseline_ll:.4f} -> {ll_recent:.4f})"
            )

    stats: Dict[str, Any] = {
        "days_since_train": days_since,
        "new_games": new_games,
        "csv_stale_days": csv_stale_days,
        "csv_last_date": csv_last_date,

        "n_total_games": len(games),
        "n_recent_eval": int(n_recent),
        "recent_logloss_cal": float(ll_recent) if ll_recent == ll_recent else None,
        "train_holdout_logloss_cal": float(baseline_ll) if baseline_ll is not None else None,
        "delta_logloss_cal": float(ll_delta) if ll_delta == ll_delta else None,
        "drift_p95_abs_pcal_minus_praw": float(drift_p95) if drift_p95 == drift_p95 else None,
        "artifact_trained_at": str((art.meta or {}).get("trained_at")),
        "artifact_data_max_date": str((art.meta or {}).get("data_max_date")),
    }

    # We *surface* calibration drift, but we don't force a full retrain because of it.
    suggest_recalibrate = False
    if (drift_p95 == drift_p95) and (n_recent >= int(cfg.min_recent_eval)):
        # Heuristic: if drift is very large, recalibration (only) is often enough.
        # We don't attach a hard threshold here; we just expose the metric.
        suggest_recalibrate = True

    return {
        "need_retrain": bool(reasons),
        "suggest_recalibrate": bool(suggest_recalibrate),
        "reasons": reasons,
        "stats": stats,
    }


# -----------------------
# mlcore_app-compatible runner
# -----------------------

def run_check(
    *,
    artifact_path: Path,
    csv_path: Path,
    config: Optional[CheckConfig] = None,
    keep_playoffs: Optional[bool] = None,
    exit_code: bool = True,
) -> int:
    """Run check and print human-friendly output. Returns 2 if retrain recommended, else 0."""
    csv_path = _resolve_csv_path(csv_path)
    out = check_retrain_needed(
        artifact_path=str(artifact_path),
        csv_path=str(csv_path),
        config=config,
        keep_playoffs=True if keep_playoffs is None else bool(keep_playoffs),
    )

    need = bool(out.get("need_retrain"))
    reasons = out.get("reasons") or []
    stats = out.get("stats") or {}

    print("need_retrain:", "YES" if need else "NO")
    for k in [
        "days_since_train",
        "new_games",
        "n_recent_eval",
        "train_holdout_logloss_cal",
        "recent_logloss_cal",
        "delta_logloss_cal",
        "drift_p95_abs_pcal_minus_praw",
    ]:
        if k in stats:
            print(f"  - {k}: {stats.get(k)}")

    if reasons:
        print("reasons:")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("reasons: (none)")

    if out.get("suggest_recalibrate"):
        print("note: calibration drift metric is available above; consider recalibrating if it stays high over a larger sample.")

    if exit_code:
        return 2 if need else 0
    return 0


def main() -> None:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Check whether ml_artifact.json likely needs full retraining.")
    ap.add_argument("--artifact", required=True, help="Path to ml_artifact.json")
    ap.add_argument("--csv", required=True, help="Path to daily CSV (OracleElixir format)")
    ap.add_argument("--max-days", type=int, default=CheckConfig.max_days, help="Retrain if artifact older than this many days.")
    ap.add_argument("--min-new-games", type=int, default=CheckConfig.min_new_games, help="Retrain if >= this many new games since last train.")
    ap.add_argument("--recent-n", type=int, default=CheckConfig.recent_n, help="Recent tail window (N games) for quality checks.")
    ap.add_argument("--drift-delta", type=float, default=CheckConfig.drift_logloss_delta, help="Retrain if recent logloss worsened by >= this.")
    ap.add_argument("--min-recent-eval", type=int, default=CheckConfig.min_recent_eval, help="Min games needed for quality-based trigger.")
    ap.add_argument(
        "--keep-playoffs",
        default="auto",
        choices=["auto", "yes", "no"],
        help="Keep playoffs games (auto=yes).",
    )
    ap.add_argument("--exit-code", action="store_true", help="Exit 2 if retrain recommended, else 0.")

    args = ap.parse_args()
    keep_playoffs = True if args.keep_playoffs in ("auto", "yes") else False

    cfg = CheckConfig(
        max_days=int(args.max_days),
        min_new_games=int(args.min_new_games),
        recent_n=int(args.recent_n),
        drift_logloss_delta=float(args.drift_delta),
        min_recent_eval=int(args.min_recent_eval),
    )

    code = run_check(
        artifact_path=Path(args.artifact),
        csv_path=_resolve_csv_path(Path(args.csv)),
        config=cfg,
        keep_playoffs=keep_playoffs,
        exit_code=True,
    )
    if args.exit_code:
        sys.exit(code)


if __name__ == "__main__":
    main()