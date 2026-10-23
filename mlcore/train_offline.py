from __future__ import annotations

# Can run both as:
#   python -m mlcore.train_offline ...
# or (from repo root):
#   python .\mlcore\train_offline.py ...

import math
import os
import sys

# Console cp1252 no Windows derruba qualquer print com acento/emoji — e o estouro
# vem no fim, depois do trabalho todo feito. Ver tools/find_silent_failures.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

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


def _mov_do_jogo(r: Any, sinal: str) -> Optional[float]:
    """Margem de vitória do BLUE, no sinal escolhido. None = não medível.

    None NÃO é zero: quem chama trata "sem sinal" caindo no ganhou/perdeu puro
    e CONTANDO o caso. Devolver 0.0 aqui seria dizer "empatou em margem", que é
    uma afirmação — e é exatamente a família de falha silenciosa das dívidas
    D5/D8/D9 (ausência de dado virando número plausível).

    Ordem de qualidade medida no eval cego 2026 (tools/eval_whr_mov.py):
    gd15 (+0.0099) >> kills (+0.0037) > goldpm (+0.0032) > torres/ouro (+0.0022).
    O ouro aos 15 ganha porque mede ANTES da bola de neve: depois disso a margem
    vira consequência do próprio resultado, não medida de qualidade.
    """
    def _f(nome: str) -> Optional[float]:
        try:
            v = float(getattr(r, nome))
        except (TypeError, ValueError, AttributeError):
            return None
        return v if math.isfinite(v) else None

    def _dif(stem: str) -> Optional[float]:
        b, v = _f(f"blue_{stem}"), _f(f"red_{stem}")
        return None if b is None or v is None else b - v

    if sinal == "gd15":
        return _f("blue_golddiffat15")
    if sinal == "kills":
        return _dif("teamkills")
    if sinal == "towers":
        return _dif("towers")
    # 'gold'/'goldpm' ficaram de fora: `totalgold` não está em
    # io_oracle.FLOAT_STEMS, então não chega até aqui. Ver o comentário no
    # argumento --whr-mov.
    return None


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
    league: str = "",
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

    # Torneio INTERNACIONAL (MSI/Worlds/EWC...) nao e promocao: um time da LCS
    # (T2) que vai ao MSI (T1) continua sendo o mesmo time. Sem isto, todo time
    # que joga internacional perdia o historico (medido 24/07: 61 times, FlyQuest
    # e Team Liquid com 8 resets cada). Nao mexe no last_tier tambem — senao a
    # VOLTA pra liga domestica pareceria rebaixamento/promocao no proximo evento.
    try:
        if league and is_international_league(str(league)):
            return
    except (TypeError, ValueError):
        pass

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
        "--apply-holdout", action="store_true",
        help="Depois de calibrar, aplica os jogos do holdout aos RATINGS. Sem isto o modelo "
             "publicado fica --holdout-days dias atrasado de forma permanente quando a esteira "
             "retreina todo dia (bug do Dota, 16/08/2026: metade do field do TI aparecia como "
             "'time desconhecido' so porque os jogos deles eram posteriores ao corte). Nao "
             "contamina a calibracao: o Platt e amostrado prequencialmente ANTES desta etapa.",
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
    # ---- margem de vitória no WHR (resultado fracionário) ----
    # DESLIGADA por default: 'off' deixa o fit BIT A BIT o de antes de 09/2026.
    # Prova cega em 2026 (tools/eval_whr_mov.py): gd15/lam=0.35/folga=0.15/w2=90
    # -> logloss +0.0117 (IC95 [+0.0078,+0.0157]), acerto 65.0%->65.7%, ganho em
    # 9/9 meses e 28/34 ligas. Ver evidence/eval_whr_mov_*.log.
    # 'gold'/'goldpm' NÃO entram aqui de propósito: io_oracle.FLOAT_STEMS não
    # carrega `totalgold`, então eles voltariam None em 100% dos jogos — uma
    # opção que não faz nada é pior que uma opção que não existe. Os dois foram
    # os PIORES no eval cego (+0.0022 e +0.0032 contra +0.0099 do gd15), então
    # não vale mexer no io_oracle por eles.
    ap.add_argument("--whr-mov", choices=["off", "gd15", "kills", "towers"],
                    default="off",
                    help="Sinal de margem no fit WHR. gd15 (ouro aos 15 min) foi o melhor "
                         "no eval cego, por medir qualidade ANTES da bola de neve.")
    ap.add_argument("--whr-mov-lambda", type=float, default=0.35,
                    help="Peso da surpresa de margem. Só vale com --whr-mov != off.")
    ap.add_argument("--whr-mov-folga", type=float, default=0.15,
                    help="Vitória normal vale 1-folga, para sobrar teto e o modelo "
                         "poder PREMIAR domínio (sem folga, 72%% dos jogos saturam).")
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
    ap.add_argument("--pscore-out", default="players_pscore.json",
                    help="Onde gravar a nota de jogador POR DESEMPENHO (por rota).")
    ap.add_argument("--no-pscore", action="store_true",
                    help="Nao monta a camada de jogadores por desempenho. "
                         "E o caminho de volta: sem ela o modelo fica como antes.")
    # PADRAO AINDA E 'media', de proposito. O 'ffa' MEDE melhor (+0,00921
    # contra +0,00402), mas ainda nao foi conferido de ponta a ponta no app
    # dirigido. Trocar o padrao antes disso seria subir no escuro.
    ap.add_argument("--pscore-rating", choices=("ffa", "media"), default="media",
                    help="Como a nota de desempenho vira rating do jogador. "
                         "'ffa' = confronto de todos contra todos (Plackett-Luce); "
                         "'media' = media acumulada (o comportamento antigo). "
                         "Medido em 11/09/2026, contra a base real servida: "
                         "media +0,00402, ffa +0,00921.")
    ap.add_argument("--pscore-max-abs", type=float, default=0.50,
                    help="Teto do peso da camada por desempenho. O ajuste converge "
                         "para ~0,066; o teto so existe para holdout curto nao "
                         "escolher um peso absurdo.")
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
    # Receita da margem, para gravar no artefato. None = treinou sem margem.
    _whr_mov_meta: Optional[Dict[str, Any]] = None
    if str(getattr(args, "core", "elo")) == "whr":
        # ------------------ WHR (alavanca N) ------------------
        # Fit MAP do histórico completo; depois EXPORTA os ratings finais pro
        # EloModel acima. Pra reconstruir EXATAMENTE o protocolo validado
        # (p = logistic10((R_a-R_b)/400), prior de liga = SOS no 1º ponto):
        #   ratings_armazenados = R_whr - pontos_da_liga (o p_blue soma o prior
        #   de volta via league_prior_weight=1.0), alpha=0 (sem shrink),
        #   tier_prior_weight=0, scale=400, dynk/MOVDA off.
        from .whr_core import WHRFitter, ajustar_expectativa_mov

        # ---- margem de vitória: DUAS passadas ----
        # A 1ª roda SEM margem só para colher (Δr, MOV) e ajustar E[MOV|Δr]. A
        # expectativa TEM que vir do Δr deste modelo: usar a de outro motor faria
        # o resíduo carregar o viés do outro. Custo: o fit do WHR roda 2x — só
        # quando --whr-mov != off.
        _mov_sinal = str(getattr(args, "whr_mov", "off") or "off")
        _mov_fit: Optional[Dict[str, float]] = None
        if _mov_sinal != "off":
            _pre = WHRFitter(w2_elo_per_day=float(args.whr_w2),
                             league_prior_points=dict(model.league_base_points_map))
            _xs: List[float] = []
            _ys: List[float] = []
            for r in train_fit.itertuples(index=False):
                _b, _r = str(r.blue), str(r.red)
                _bl, _rl = str(r.blue_league), str(r.red_league)
                _m = _mov_do_jogo(r, _mov_sinal)
                if _m is not None:
                    _xs.append(_pre.rating_nat(_b, _bl) - _pre.rating_nat(_r, _rl))
                    _ys.append(float(_m))
                _pre.add_game(_b, _r, int(r.blue_win), int(r.date.toordinal()),
                              blue_league=_bl, red_league=_rl)
                _pre.refresh_pair(_b, _r, iters=2)
            _mov_fit = ajustar_expectativa_mov(_xs, _ys)
            print(f"[train_offline] WHR margem '{_mov_sinal}': E[MOV|dr] "
                  f"a={_mov_fit['a']:.3f} b={_mov_fit['b']:.3f} c={_mov_fit['c']:.3f} "
                  f"desvio={_mov_fit['s']:.3f} n={_mov_fit['n']} "
                  f"(sem sinal em {len(train_fit) - len(_xs)} jogos)")

        whr = WHRFitter(w2_elo_per_day=float(args.whr_w2),
                        league_prior_points=dict(model.league_base_points_map),
                        mov_lambda=(float(args.whr_mov_lambda) if _mov_sinal != "off" else 0.0),
                        mov_folga=(float(args.whr_mov_folga) if _mov_sinal != "off" else 0.0),
                        mov_fit=_mov_fit)
        _n_fit = 0
        # Predições PREQUENCIAIS (prever ANTES de adicionar o jogo) do último
        # ano de treino — é nelas que o Platt do artefato é fitado (a receita
        # validada no caminho fiel). Fitar no holdout de 30d overfita o
        # intercepto da janela (bug 2026-07-07: média p_blue 0.573, ECE 0.089).
        _preq_cut = train_fit["date"].max() - pd.Timedelta(days=365)
        whr_preq_y: List[int] = []
        whr_preq_p: List[float] = []
        # O gameid vai junto para que outras camadas possam ser ajustadas
        # NESTE conjunto — sao previsoes feitas ANTES de o modelo ver o jogo,
        # entao sao fora da amostra por construcao, e sao ~9x o holdout de 30d.
        whr_preq_gid: List[str] = []
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
                whr_preq_gid.append(str(getattr(r, "gameid", "")))
            whr.add_game(_b, _r, int(r.blue_win), day,
                         blue_league=str(r.blue_league), red_league=str(r.red_league),
                         mov=(_mov_do_jogo(r, _mov_sinal) if _mov_sinal != "off" else None))
            whr.refresh_pair(_b, _r, iters=2)
            _n_fit += 1
            if _n_fit % 400 == 0:
                whr.full_sweeps(sweeps=1, tol_elo=0.0)
        _sweeps = whr.full_sweeps(sweeps=12, tol_elo=0.5, iters=1)
        print(f"[train_offline] WHR fit: {_n_fit} jogos, convergiu em {_sweeps} varreduras finais; "
              f"{len(whr_preq_y)} predições prequenciais p/ calibração.")
        if _mov_sinal != "off":
            print(f"[train_offline] WHR margem aplicada em {whr.n_mov_usados} jogos, "
                  f"ausente em {whr.n_mov_faltando} (esses caem no ganhou/perdeu puro).")
            _whr_mov_meta = {
                "sinal": _mov_sinal,
                "lambda": float(args.whr_mov_lambda),
                "folga": float(args.whr_mov_folga),
                "expectativa": dict(_mov_fit or {}),
                "jogos_com_margem": int(whr.n_mov_usados),
                "jogos_sem_margem": int(whr.n_mov_faltando),
            }

        # DEFASAGEM DO HOLDOUT (achado 16/08/2026): os ratings so viam ate o
        # corte, ou seja, o modelo publicado ficava --holdout-days DIAS ATRASADO
        # de forma PERMANENTE quando a esteira retreina todo dia (caso do Dota).
        # Medido: Iron Wing tinha 16 jogos em 2026 e games_played=0, porque todos
        # eram posteriores ao corte; no TI isso virava metade do torneio "time
        # desconhecido". E o meta gravava data_max_date = fim de TUDO, entao o
        # update_incremental achava que estava em dia e nao aplicava nada.
        # Aqui os jogos do holdout entram nos ratings DEPOIS da coleta
        # prequencial — o Platt ja foi amostrado fora da amostra, nao contamina.
        if bool(getattr(args, "apply_holdout", False)) and len(test_df) > 0:
            _test_fit = _augment_games_with_home_meta(
                test_df, domestic_by_year=domestic_by_year,
                team_meta_latest=team_meta_latest,
            ).sort_values(["date", "gameid"]).reset_index(drop=True)
            _n_tail = 0
            for r in _test_fit.itertuples(index=False):
                whr.add_game(str(r.blue), str(r.red), int(r.blue_win),
                             int(r.date.toordinal()),
                             blue_league=str(r.blue_league), red_league=str(r.red_league))
                whr.refresh_pair(str(r.blue), str(r.red), iters=2)
                _n_tail += 1
            whr.full_sweeps(sweeps=8, tol_elo=0.5, iters=1)
            print(f"[train_offline] holdout aplicado aos ratings: +{_n_tail} jogos "
                  f"(o modelo conhece ate data_max_date, nao so ate o corte).")

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

            _maybe_reset_on_promotion(model, blue, bt, last_tier=last_tier, league=bl)
            _maybe_reset_on_promotion(model, red, rt, last_tier=last_tier, league=rl)

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
        # ATE ONDE OS RATINGS FORAM DE VERDADE (16/08/2026). data_max_date grava o
        # fim de TODOS os dados, mas o fit so ve ate o corte do holdout — a menos
        # que --apply-holdout esteja ligado. Com os dois iguais, o modelo esta em
        # dia; diferentes, a distancia entre eles E a defasagem do modelo.
        # Quem consome: update_incremental (para saber o que ainda falta aplicar)
        # e o detector de defasagem, que antes eram enganados pelo data_max_date.
        "ratings_max_date": (
            data_max_dt.date().isoformat()
            if bool(getattr(args, "apply_holdout", False)) and pd.notna(data_max_dt)
            else (cutoff_dt.date().isoformat()
                  if cutoff_dt is not None and pd.notna(cutoff_dt)
                  else (data_max_dt.date().isoformat() if pd.notna(data_max_dt) else None))
        ),
        "n_games_total": len(games),
        "n_games_train": len(train_df),
        "n_games_holdout": len(test_df),
        "n_teams": len(team_meta_latest),
        "csv": list(csv_paths),
        "keep_playoffs": keep_playoffs,
        "holdout_days": int(args.holdout_days),
        "core": str(getattr(args, "core", "elo")),
        "whr_w2": float(getattr(args, "whr_w2", 0.0)) if str(getattr(args, "core", "elo")) == "whr" else None,
        # Margem de vitória: fica GRAVADA para o artefato dizer sozinho com que
        # receita foi treinado. Sem isto, descobrir se um modelo usou margem
        # exigiria adivinhar pelo comando que alguém rodou meses atrás.
        "whr_mov": _whr_mov_meta,
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

    # ------------------------------------------------------------------
    # CAMADA DE JOGADORES POR DESEMPENHO (nao pelo resultado da partida)
    #
    # A camada acima move a nota com `y - p`: ganhou o time, sobe todo mundo.
    # Esta aqui avalia COMO o jogador jogou, por rota. Medido na regua honesta
    # em 11/09/2026 (8 dobras, n=8.494, pareado, contra a base REAL servida):
    #   producao hoje ............. 7,09
    #   + camada antiga ........... 7,13   +0,00233
    #   + esta camada ............. 7,20   +0,00388  IC [+0,00270, +0,00506]
    #
    # DOIS CORTES, e confundi-los vazaria futuro:
    #   - os PESOS por rota so enxergam jogos anteriores ao corte do treino;
    #   - as NOTAS usadas para AJUSTAR o peso tambem param no corte, senao o
    #     peso seria escolhido com a resposta do holdout na mao.
    # Depois de ajustado o peso, as notas SALVAS sao refeitas com tudo, porque
    # a partida a prever ainda nao aconteceu.
    # ------------------------------------------------------------------
    if bool(args.fit_players) and not bool(args.no_pscore):
        try:
            from mlcore.players_pscore_core import (
                carregar_linhas as _ps_carregar,
                construir as _ps_construir,
            )

            from mlcore.players_pscore_core import (
                ajustar as _ps_ajustar,
                calcular_pscore as _ps_calc,
                deltas_por_jogo as _ps_deltas,
            )

            _corte = cutoff_dt.date() if (cutoff_dt is not None and pd.notna(cutoff_dt)) else None
            _df_ps = _ps_carregar(csv_paths)

            # ONDE O PESO E AJUSTADO, e por que NAO no holdout de 30 dias.
            #
            # A primeira versao ajustava nos ~984 jogos do holdout e achava
            # w=0,032. A regua honesta, ajustando em 8.494 jogos, converge para
            # w=0,066 — e nas dobras pequenas ela tambem oscilava (0,146 na
            # primeira, de 1.146 jogos). Ou seja: 984 jogos nao bastam, o peso
            # saia pela metade, e a camada entregaria menos do que foi medido.
            #
            # As predicoes PREQUENCIAIS do ultimo ano de treino resolvem: sao
            # feitas ANTES de o modelo ver cada jogo, logo sao fora da amostra
            # por construcao, e sao ~9x mais. E o mesmo conjunto onde o Platt
            # do artefato ja e ajustado.
            #
            # E o delta de cada jogo usa a nota PONTO NO TEMPO do jogador: so
            # o que ele tinha feito ANTES daquele jogo.
            _aj = _ps_ajustar(_df_ps, ate=_corte)
            _ps_valores = _ps_calc(_df_ps, _aj)

            # COMO A NOTA DE DESEMPENHO VIRA RATING DO JOGADOR
            #
            # 'media' (PADRAO hoje): media acumulada das notas anteriores dele.
            # 'ffa':                os 10 do jogo sao ordenados pela nota e o
            #                       rating e atualizado por esse confronto, com
            #                       Plackett-Luce. Sabe CONTRA QUEM ele jogou.
            #
            # MEDIDO PAREADO, ffa CONTRA 'media' — o que esta servido — nos
            # mesmos 8.494 jogos, 8 dobras, todos fora da amostra, escalacao
            # ANTERIOR (ponto-no-tempo fiel). 13/09/2026:
            #   ffa completo (mu - 3*sigma) .. +0,00534  IC [+0,00418, +0,00649]
            #   so experiencia (placebo) ..... +0,00276  IC [+0,00095, +0,00456]
            #   so habilidade (mu puro) ...... +0,00219  IC [+0,00139, +0,00299]
            # Os tres passam BH e Holm. Nota: 7,43 contra 7,20.
            #
            # O QUE O GANHO E, SEM ENFEITE: 52% vem de EXPERIENCIA, nao de
            # habilidade — a nota e `mu - 3*sigma`, o sigma encolhe com o
            # numero de jogos, e isso premia elenco rodado. Sinal real e
            # disponivel antes da partida, nao e vazamento, mas nao e o que o
            # nome sugere. Os outros 41% sao habilidade de fato, e essa parte
            # sozinha ja passaria na regua.
            #
            # CORRECAO REGISTRADA: uma medicao anterior dizia "~72% e
            # experiencia". Aquela conta usava OUTRA base (sem camada de
            # jogadores nenhuma), nao a servida. Contra a servida, e 52%.
            # Numero de comparacao so vale junto com a base dele.
            if str(args.pscore_rating) == "ffa":
                from mlcore.players_skill_ffa import (
                    construir_notas as _ffa_construir,
                    deltas_por_jogo as _ffa_deltas,
                )
                _antes_ffa, _notas_ffa = _ffa_construir(_df_ps, _ps_valores)
                _delta_gid = _ffa_deltas(_df_ps, _antes_ffa)
            else:
                _delta_gid = _ps_deltas(_df_ps, _ps_valores)

            _ys: List[int] = []
            _lb: List[float] = []
            _dl: List[float] = []
            _n_sem_cobertura = 0
            _fonte_peso = "prequencial"

            _preq = list(zip(whr_preq_gid, whr_preq_y, whr_preq_p)) if (
                str(args.core) == "whr" and whr_preq_gid) else []
            for _gid, _y, _pp in _preq:
                _d = _delta_gid.get(str(_gid))
                if _d is None:
                    _n_sem_cobertura += 1
                    continue
                _pp = min(max(float(_pp), 1e-6), 1.0 - 1e-6)
                _ys.append(int(_y))
                _lb.append(float(_logit(_pp)))
                _dl.append(float(_d))

            if len(_ys) < int(args.min_holdout):
                # Sem prequencial (outro motor, ou poucos jogos): cai no
                # holdout, que e menor mas continua fora da amostra.
                _fonte_peso = "holdout"
                _ys, _lb, _dl = [], [], []
                _n_sem_cobertura = 0
                for rr in test_df.sort_values("date").itertuples(index=False):
                    _d = _delta_gid.get(str(rr.gameid))
                    if _d is None:
                        _n_sem_cobertura += 1
                        continue
                    _yy = _safe_int_year(getattr(rr, "year", None))
                    _mlg = str(getattr(rr, "league", "") or "OTHER")
                    _bl = _resolve_home_league(team=str(rr.blue), year=_yy, match_league=_mlg,
                                               domestic_by_year=domestic_by_year,
                                               team_meta_latest=team_meta_latest)
                    _rl = _resolve_home_league(team=str(rr.red), year=_yy, match_league=_mlg,
                                               domestic_by_year=domestic_by_year,
                                               team_meta_latest=team_meta_latest)
                    _pp = float(model.p_blue(str(rr.blue), str(rr.red),
                                             blue_league=_bl, red_league=_rl,
                                             blue_tier=str(tier_for_league(_bl)),
                                             red_tier=str(tier_for_league(_rl))))
                    _pp = min(max(_pp, 1e-6), 1.0 - 1e-6)
                    _ys.append(int(rr.blue_win))
                    _lb.append(float(_logit(_pp)))
                    _dl.append(float(_d))

            if len(_ys) < int(args.min_holdout):
                raise ValueError(
                    "so {} jogos utilizaveis para ajustar o peso, abaixo do minimo {}".format(
                        len(_ys), int(args.min_holdout)))

            _w_raw = float(fit_w_lineup(y=_ys, logit_base=_lb, delta=_dl,
                                        l2=float(args.wlineup_l2)))
            # Piso em 0 pelo mesmo motivo da camada antiga: peso NEGATIVO quer
            # dizer "apostar contra quem jogou melhor", o que nao tem base e na
            # pratica e ruido de holdout curto. Com piso, o pior caso e a
            # camada ficar neutra. O valor cru fica registrado.
            _w = max(0.0, min(float(args.pscore_max_abs), _w_raw))

            # (2) notas com TUDO, que sao as que vao servir
            _pac = _ps_construir(csv_paths, ate=_corte, ate_notas=None, df=_df_ps)
            if str(args.pscore_rating) == "ffa":
                # As notas salvas passam a ser as do confronto de todos contra
                # todos. O FORMATO nao muda — segue chave "jogador|||Rota" com
                # o campo "n" — entao o app nao precisa mudar nada para ler.
                _pac["notas"] = _notas_ffa
                _pac["rating"] = "ffa"
            else:
                _pac["rating"] = "media"
            _pac["w"] = float(_w)
            _pac["w_raw"] = float(_w_raw)

            _ps_out = Path(str(args.pscore_out)).expanduser()
            _ps_out.parent.mkdir(parents=True, exist_ok=True)
            _ps_out.write_text(json.dumps(_pac, ensure_ascii=False), encoding="utf-8")

            meta["players_pscore"] = {
                "enabled": True,
                "arquivo": str(_ps_out),
                "w": float(_w),
                "w_raw": float(_w_raw),
                "w_com_piso": bool(_w_raw < 0.0),
                "corte_pesos": (_corte.isoformat() if _corte else None),
                "rotas": len(_pac.get("pesos_por_rota") or {}),
                "atributos": list(_pac.get("atributos") or []),
                "n_jogadores": len(_pac.get("notas") or {}),
                "rating": str(args.pscore_rating),
                "fonte_do_peso": _fonte_peso,
                "jogos_usados": len(_ys),
                "jogos_sem_delta": int(_n_sem_cobertura),
                "beta": float(_pac.get("beta", 0.0) or 0.0),
            }
            print("[train_offline] players_pscore [{}]: w={:.4f} (cru {:.4f}) | "
                  "{} jogadores | peso ajustado em {} jogos ({}), {} sem delta".format(
                      args.pscore_rating, _w, _w_raw, len(_pac.get("notas") or {}),
                      len(_ys), _fonte_peso, _n_sem_cobertura))
        except Exception as e:
            meta["players_pscore"] = {"enabled": False, "error": str(e)}
            print("[train_offline] players_pscore FALHOU:", str(e))

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
