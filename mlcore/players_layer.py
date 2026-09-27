from __future__ import annotations

"""Players layer (Lineup adjustment) for ML.

Design goals
- Incremental: update ratings using only new games.
- Stable: decay toward a prior, plus shrink for low sample.
- Robust: if data is missing, return neutral delta and low confidence.

Core idea
We maintain a scalar rating per (player, position). Ratings live in **logit units**:

  p_pos = sigmoid(r_blue - r_red)
  r_blue += k * (y - p_pos)
  r_red  -= k * (y - p_pos)

where y=1 if BLUE won the game (same y used by ML core).

For inference we compute the lineup delta:

  Δ_players = Σ_pos (r_eff(blue_player,pos) - r_eff(red_player,pos))

and mix in logit space:

  logit(p_final_raw) = logit(p_base_raw) + w_lineup * Δ_players

The scalar w_lineup is fitted offline on holdout (see train_offline integration).

Prior / fallback
Optionally we can load an OracleElixir "byTournament" player stats table and fit
a small logistic model to turn those stats into a prior score per player+pos.
If the prior isn't available, the prior defaults to 0.
"""

import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from mlcore.league_transfer_features_core import build_lineup_transfer_summary
from .league_context import league_macro_key
from .tiers import tier_for_league, tier_priority
from .utils import sigmoid
from .utils import logit  # noqa: F401 - re-exported (train_offline imports it from here)


# -----------------------------
# Helpers
# -----------------------------

POS_CANON = ["Top", "Jungle", "Mid", "Bot", "Support"]


def norm_pos(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not s:
        return ""
    if s in ("top", "t"):
        return "Top"
    if s in ("jng", "jg", "jungle", "j"):
        return "Jungle"
    if s in ("mid", "m"):
        return "Mid"
    if s in ("bot", "adc", "bottom", "b"):
        return "Bot"
    if s in ("sup", "support", "s"):
        return "Support"
    return ""


def _league_macro_key(league: str) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _lane_weight_for(pos: str, lane_weights: Optional[Dict[str, float]]) -> float:
    if not isinstance(lane_weights, dict):
        return 1.0
    p = str(pos or "").strip()
    if not p:
        return 1.0
    aliases = [p, p.upper(), p.lower()]
    if p == "Bot":
        aliases += ["ADC", "adc", "Adc"]
    for k in aliases:
        if k in lane_weights:
            fv = lane_weights.get(k)
            try:
                return float(fv or 1.0)
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _league_weight_for(match_league: Optional[str], league_weights: Optional[Dict[str, float]]) -> float:
    if not isinstance(league_weights, dict):
        return 1.0
    lg = str(match_league or "").strip()
    macro = _league_macro_key(lg)
    keys = [lg, lg.upper(), macro, f"macro:{macro}", "DEFAULT", "default"]
    for k in keys:
        if k in league_weights:
            fv = league_weights.get(k)
            try:
                return float(fv or 1.0)
            except (TypeError, ValueError):
                return 1.0
    return 1.0


def _interleague_pair_scale(
    last_league: Optional[str],
    match_league: Optional[str],
    *,
    gp: int = 0,
    last_tier: Optional[str] = None,
    cur_tier: Optional[str] = None,
) -> float:
    """Evidence-aware transfer prior for cross-macro league moves.

    The base macro-pair scale stays conservative, but portability increases when the
    player's own evidence is strong. This avoids treating a long-tenured T1 player
    the same way as a low-sample player in interleague contexts.
    """
    la = _league_macro_key(str(last_league or ""))
    lb = _league_macro_key(str(match_league or ""))
    if not la or not lb or la == lb:
        return 1.0
    pair = tuple(sorted([la, lb]))
    base = 0.91
    if "OTHER" in pair:
        base = 0.88
    elif pair in {("CN", "KR")}:
        base = 0.94
    elif pair in {("EMEA", "NA")}:
        base = 0.92
    elif pair in {("APAC", "CN"), ("APAC", "KR"), ("APAC", "EMEA"), ("APAC", "NA"), ("BR", "EMEA"), ("BR", "NA")}:
        base = 0.90
    elif pair in {("APAC", "BR"), ("BR", "CN"), ("BR", "KR")}:
        base = 0.88
    gp_v = max(0.0, float(gp or 0))
    gp_support = float(gp_v / (gp_v + 80.0))
    scale = float(base + ((1.0 - base) * gp_support))
    if last_tier and cur_tier and str(last_tier) == str(cur_tier):
        scale += float(0.015 * (0.35 + (0.65 * gp_support)))
        if str(cur_tier) == "T1" and pair in {("CN", "KR"), ("EMEA", "NA")}:
            scale += float(0.010 * (0.30 + (0.70 * gp_support)))
    return float(max(0.86, min(0.975, scale)))


def _player_key(player: str, pos: str) -> str:
    return f"{player}|||{pos}"


def _parse_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    dt = pd.to_datetime(v, errors="coerce")
    if not isinstance(dt, pd.Timestamp) or pd.isna(dt):
        return None
    return dt.date()




def safe_json_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.exists():
        try:
            bak = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, bak)
        except OSError:
            pass
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


# -----------------------------
# Artifact
# -----------------------------


@dataclass
class PlayerState:
    r: float = 0.0
    gp: int = 0
    last_date: Optional[str] = None  # ISO date
    last_league: Optional[str] = None
    last_tier: Optional[str] = None


class PlayersArtifact:
    """Persisted ratings per (player, pos) + optional prior."""

    def __init__(
        self,
        *,
        version: str = "1.1",
        half_life_days: float = 180.0,
        k: float = 0.15,
        shrink_m: float = 20.0,
        transfer_alpha: float = 0.7,
        tier_k_weights: Optional[Dict[str, float]] = None,
        prior_scores: Optional[Dict[str, Dict[str, Any]]] = None,
        prior_meta: Optional[Dict[str, Any]] = None,
        players: Optional[Dict[str, PlayerState]] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        self.version = str(version)
        self.half_life_days = float(half_life_days)
        self.k = float(k)
        self.shrink_m = float(shrink_m)
        self.transfer_alpha = float(transfer_alpha)
        # How much a single game should move ratings, by tier (higher tier => stronger signal).
        # Defaults are conservative to avoid a player "inflating" in weak leagues.
        self.tier_k_weights = dict(tier_k_weights or {
            "T1": 1.00,
            "T2": 0.85,
            "ERL": 0.65,
            "MINOR": 0.50,
            "OTHER": 0.70,
        })
        self.prior_scores = prior_scores or {}  # key -> {"prior": float, "gp": int}
        self.prior_meta = prior_meta or {}
        self.players: Dict[str, PlayerState] = players or {}
        self.updated_at = updated_at

    # --- persistence
    @staticmethod
    def load_json(path: str) -> "PlayersArtifact":
        p = Path(path)
        if not p.exists():
            return PlayersArtifact()
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        players_raw = d.get("players") or {}
        players: Dict[str, PlayerState] = {}
        for k, v in players_raw.items():
            if not isinstance(v, dict):
                continue
            players[str(k)] = PlayerState(
                r=float(v.get("r", 0.0) or 0.0),
                gp=int(v.get("gp", 0) or 0),
                last_date=(str(v.get("last_date")) if v.get("last_date") else None),
                last_league=(str(v.get("last_league")) if v.get("last_league") else None),
                last_tier=(str(v.get("last_tier")) if v.get("last_tier") else None),
            )
        return PlayersArtifact(
            version=str(d.get("version", "1.1")),
            half_life_days=float(d.get("half_life_days", 180.0) or 180.0),
            k=float(d.get("k", 0.15) or 0.15),
            shrink_m=float(d.get("shrink_m", 20.0) or 20.0),
            transfer_alpha=float(d.get("transfer_alpha", 0.7) or 0.7),
            tier_k_weights=(d.get("tier_k_weights") or None),
            prior_scores=(d.get("prior_scores") or {}),
            prior_meta=(d.get("prior_meta") or {}),
            players=players,
            updated_at=(d.get("updated_at") or None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "half_life_days": self.half_life_days,
            "k": self.k,
            "shrink_m": self.shrink_m,
            "transfer_alpha": self.transfer_alpha,
            "tier_k_weights": self.tier_k_weights,
            "prior_meta": self.prior_meta,
            "prior_scores": self.prior_scores,
            "players": {
                k: {
                    "r": v.r,
                    "gp": v.gp,
                    "last_date": v.last_date,
                    "last_league": v.last_league,
                    "last_tier": v.last_tier,
                }
                for k, v in self.players.items()
            },
        }

    def save_json(self, path: str) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        safe_json_write(Path(path), self.to_dict())

    # --- rating access
    def prior_for(self, player: str, pos: str) -> float:
        k = _player_key(player, pos)
        v = self.prior_scores.get(k)
        if isinstance(v, dict):
            try:
                return float(v.get("prior", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def get_state(self, player: str, pos: str) -> PlayerState:
        key = _player_key(player, pos)
        st = self.players.get(key)
        if st is None:
            # initialize at prior
            st = PlayerState(r=float(self.prior_for(player, pos)), gp=0, last_date=None)
            self.players[key] = st
        return st

    def _apply_decay_inplace(self, st: PlayerState, *, prior: float, new_date: date) -> None:
        if st.last_date is None or not self.half_life_days or self.half_life_days <= 0:
            return
        old = _parse_date(st.last_date)
        if old is None:
            return
        dt = (new_date - old).days
        if dt <= 0:
            return
        d = 0.5 ** (float(dt) / float(self.half_life_days))
        st.r = float(prior + (st.r - prior) * d)

    def _decayed_value(self, *, r: float, prior: float, last_date: Optional[str], as_of: Optional[date]) -> float:
        if as_of is None or last_date is None or not self.half_life_days or self.half_life_days <= 0:
            return float(r)
        old = _parse_date(last_date)
        if old is None:
            return float(r)
        dt = (as_of - old).days
        if dt <= 0:
            return float(r)
        d = 0.5 ** (float(dt) / float(self.half_life_days))
        return float(prior + (float(r) - float(prior)) * d)

    def _k_multiplier_for_league(self, league: Optional[str]) -> float:
        tier = tier_for_league(str(league) if league else None)
        try:
            return float(self.tier_k_weights.get(str(tier), self.tier_k_weights.get("OTHER", 1.0)))
        except (TypeError, ValueError):
            return 1.0

    def _transfer_scale(self, *, last_tier: Optional[str], cur_tier: Optional[str]) -> float:
        """Penalty when a player moves UP into a stronger tier.

        This avoids the classic failure mode: a player farms a weak league,
        gets a big rating, then immediately looks like "Faker" in a major league.

        If the player moves down (strong -> weak), we do not penalize.
        """
        lt = str(last_tier or "OTHER")
        ct = str(cur_tier or "OTHER")
        # smaller priority means stronger tier (T1=0 is strongest)
        d = int(tier_priority(lt)) - int(tier_priority(ct))
        if d <= 0:
            return 1.0
        a = max(0.0, float(self.transfer_alpha))
        return float(1.0 / (1.0 + a * float(d)))

    def r_effective(self, player: str, pos: str) -> Tuple[float, int]:
        """Back-compat: effective rating without *extra* decay/transfer at query time."""
        return self.r_effective_at(player, pos, as_of=None, match_league=None)[:2]

    def r_effective_at(
        self,
        player: str,
        pos: str,
        *,
        as_of: Optional[date],
        match_league: Optional[str],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return (r_eff, gp, debug) with:

        - decay from last_date -> as_of
        - GP shrink
        - transfer penalty when moving UP tiers
        """
        pos = norm_pos(pos)
        if not pos:
            return 0.0, 0, {"used": False}

        pl = str(player or "").strip()
        if not pl:
            return 0.0, 0, {"used": False}

        st = self.get_state(pl, pos)
        prior = float(self.prior_for(pl, pos))
        gp = int(st.gp or 0)

        r_decayed = self._decayed_value(r=float(st.r), prior=prior, last_date=st.last_date, as_of=as_of)

        m = max(1e-9, float(self.shrink_m))
        shrink = float(gp) / (float(gp) + m)
        r_shrunk = float(prior + (float(r_decayed) - float(prior)) * shrink)

        cur_tier = tier_for_league(str(match_league) if match_league else None) if match_league else None
        last_tier = st.last_tier or (tier_for_league(st.last_league) if st.last_league else None)
        tscale = self._transfer_scale(last_tier=last_tier, cur_tier=cur_tier) if cur_tier else 1.0
        inter_scale = (
            _interleague_pair_scale(
                st.last_league,
                match_league,
                gp=gp,
                last_tier=last_tier,
                cur_tier=cur_tier,
            )
            if match_league
            else 1.0
        )
        total_scale = float(max(0.0, min(1.0, float(tscale) * float(inter_scale))))
        r_eff = float(prior + (float(r_shrunk) - float(prior)) * total_scale)

        dbg = {
            "used": True,
            "prior": float(prior),
            "r": float(st.r),
            "r_decayed": float(r_decayed),
            "r_shrunk": float(r_shrunk),
            "r_eff": float(r_eff),
            "gp": int(gp),
            "last_date": st.last_date,
            "last_league": st.last_league,
            "last_tier": last_tier,
            "cur_tier": cur_tier,
            "transfer_scale": float(tscale),
            "interleague_scale": float(inter_scale),
            "interleague_gp_support": float(max(0.0, min(1.0, float(gp) / (float(gp) + 80.0)))) if gp > 0 else 0.0,
            "total_transfer_scale": float(total_scale),
        }
        return r_eff, gp, dbg

    # --- updates
    def update_game(
        self,
        *,
        game_date: date,
        league: Optional[str] = None,
        blue_lineup: Dict[str, str],
        red_lineup: Dict[str, str],
        blue_win: int,
    ) -> None:
        """Apply one game update using the 5v5 lineup."""
        y = 1.0 if int(blue_win) == 1 else 0.0
        for pos in POS_CANON:
            pb = str(blue_lineup.get(pos, "") or "").strip()
            pr = str(red_lineup.get(pos, "") or "").strip()
            if not pb or not pr:
                continue
            sb = self.get_state(pb, pos)
            sr = self.get_state(pr, pos)
            prior_b = float(self.prior_for(pb, pos))
            prior_r = float(self.prior_for(pr, pos))
            # decay to current date
            self._apply_decay_inplace(sb, prior=prior_b, new_date=game_date)
            self._apply_decay_inplace(sr, prior=prior_r, new_date=game_date)

            p = sigmoid(float(sb.r) - float(sr.r))
            err = (y - p)
            k = float(self.k) * float(self._k_multiplier_for_league(league))
            sb.r = float(sb.r + k * err)
            sr.r = float(sr.r - k * err)
            sb.gp = int(sb.gp or 0) + 1
            sr.gp = int(sr.gp or 0) + 1
            sb.last_date = game_date.isoformat()
            sr.last_date = game_date.isoformat()
            if league:
                sb.last_league = str(league)
                sr.last_league = str(league)
                try:
                    sb.last_tier = tier_for_league(str(league))
                    sr.last_tier = tier_for_league(str(league))
                except Exception:
                    pass


# -----------------------------
# OracleElixir match_data helpers
# -----------------------------


MATCH_PLAYER_USECOLS = [
    "gameid",
    "date",
    "league",
    "year",
    "split",
    "playoffs",
    "participantid",
    "side",
    "teamname",
    "playerid",
    "playername",
    "position",
    "result",
]


def load_match_player_rows(
    csv_paths: List[str],
    *,
    keep_playoffs: bool | None = None,
    gameids: Optional[set[str]] = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Load player rows (participantid 1..10) from one or more OE match_data CSVs.

    If gameids is provided, only keep those games (chunk-filtered).
    """
    dfs: List[pd.DataFrame] = []
    gset = set([str(x) for x in gameids]) if gameids else None

    for p in (csv_paths or []):
        # chunked read to allow filtering by gameid set
        try:
            try:
                it = pd.read_csv(p, low_memory=False, usecols=MATCH_PLAYER_USECOLS, chunksize=int(chunksize))
            except ValueError as e:
                # Older dumps might not have playerid; retry without it.
                msg = str(e)
                if "Usecols do not match columns" in msg and "playerid" in msg:
                    usecols = [c for c in MATCH_PLAYER_USECOLS if c != "playerid"]
                    it = pd.read_csv(p, low_memory=False, usecols=usecols, chunksize=int(chunksize))
                else:
                    raise
        except Exception:
            # fallback to full read
            df = pd.read_csv(p, low_memory=False)
            it = [df]

        for chunk in it:
            dfc = chunk.copy()
            if "participantid" not in dfc.columns:
                continue
            dfc = dfc[pd.to_numeric(dfc["participantid"], errors="coerce").isin(list(range(1, 11)))]
            if dfc.empty:
                continue
            if keep_playoffs is False and "playoffs" in dfc.columns:
                pl = pd.to_numeric(dfc["playoffs"], errors="coerce").fillna(0).astype(int).astype(bool)
                dfc = dfc[~pl]
            if gset is not None and "gameid" in dfc.columns:
                dfc["gameid"] = dfc["gameid"].astype(str)
                dfc = dfc[dfc["gameid"].isin(gset)]
            if dfc.empty:
                continue
            dfs.append(dfc)

    if not dfs:
        return pd.DataFrame(columns=[c for c in MATCH_PLAYER_USECOLS if c != "result"])

    df = pd.concat(dfs, ignore_index=True)
    # normalize
    df["gameid"] = df["gameid"].astype(str)
    df["teamname"] = df.get("teamname", "").astype(str).str.strip()
    df["playerid"] = df.get("playerid", "").astype(str).str.strip()
    df["playername"] = df.get("playername", "").astype(str).str.strip()
    df["position"] = df.get("position", "").astype(str)
    df["pos_can"] = df["position"].apply(norm_pos)
    df = df[df["pos_can"].isin(POS_CANON)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Prefer stable identity: playerid when present; fallback to playername.
    df["player_key"] = df["playerid"]
    df.loc[df["player_key"].astype(str).str.strip().eq(""), "player_key"] = df["playername"]

    df = df.dropna(subset=["date", "gameid", "teamname", "player_key", "pos_can"]).copy()
    df = df[df["player_key"].astype(str).str.strip().ne("")].copy()

    # prefer stable side: if side missing, infer via participantid
    if "side" not in df.columns:
        df["side"] = ""
    df["side"] = df["side"].astype(str).str.strip().str.lower()
    pid = pd.to_numeric(df["participantid"], errors="coerce").fillna(0).astype(int)
    df.loc[df["side"].eq(""), "side"] = pid.map(lambda x: "blue" if 1 <= x <= 5 else "red")
    df["side"] = df["side"].replace({"blue": "blue", "red": "red"})

    # winner label: result=1 for the winner's players
    if "result" in df.columns:
        df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0).astype(int)
    else:
        df["result"] = 0

    # keep last occurrence per player row if duplicates (overlapping CSVs)
    df = df.sort_values(["date", "gameid", "participantid"]).drop_duplicates(
        subset=["gameid", "participantid"], keep="last"
    )
    return df.reset_index(drop=True)


# ÍNDICE DAS ESCALAÇÕES, montado uma vez por quadro.
#
# POR QUE (23/09/2026). O perfil de UMA passada do app deu 808 segundos, e
# 660 deles estavam aqui: `lineup_for_game` era chamada 29.178 vezes (uma por
# jogo do histórico) e, a cada chamada, convertia e varria a coluna `gameid`
# inteira — ~290 mil linhas. São bilhões de comparações de texto para
# responder o que um índice responde de imediato. O custo crescia ao
# quadrado, e ele atualiza o CSV todos os dias.
#
# O CACHE SEGURA O PRÓPRIO QUADRO, e não só o endereço dele.
#
# A primeira versão usava `(id(quadro), len(quadro))` como chave. Um teste
# reprovou na suíte e passou sozinho — a assinatura clássica de lixo entre
# testes. A causa era real e séria: o Python REAPROVEITA endereços de memória.
# Quando um quadro morria e outro nascia no mesmo endereço com o mesmo
# tamanho, a chave batia e o índice velho era devolvido — escalação de OUTRO
# jogo, com toda a confiança. Isso é bem pior que lentidão.
#
# Guardando uma referência ao quadro, ele não pode ser coletado enquanto
# estiver em cache, e o endereço não pode ser reaproveitado. Fica UM só,
# porque o app trabalha com um quadro de cada vez e cada um pesa dezenas de MB.
_INDICE_ESCALACOES: Dict[str, Any] = {}


def _montar_indice_escalacoes(player_rows: pd.DataFrame) -> Dict[str, Tuple[Dict[str, str], Dict[str, str]]]:
    """gameid -> (escalação azul, escalação vermelha), numa passada só.

    Mantém a MESMA regra da versão que varria linha a linha: para cada lado e
    posição vale a PRIMEIRA linha encontrada, na ordem do quadro.
    """
    fora: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {}
    colunas = ("gameid", "side", "pos_can", "player_key")
    if any(c not in player_rows.columns for c in colunas):
        return fora
    for gid, lado, pos, chave in zip(
        player_rows["gameid"].astype(str),
        player_rows["side"],
        player_rows["pos_can"],
        player_rows["player_key"].astype(str),
    ):
        if pos not in POS_CANON:
            continue
        par = fora.get(gid)
        if par is None:
            par = ({}, {})
            fora[gid] = par
        alvo = par[0] if lado == "blue" else (par[1] if lado == "red" else None)
        if alvo is None or pos in alvo:
            continue            # primeira linha vence, como antes
        alvo[pos] = str(chave).strip()
    return fora


def lineup_for_game(player_rows: pd.DataFrame, gameid: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (blue_lineup, red_lineup) dict[pos]->player for one gameid."""
    if player_rows is None or player_rows.empty:
        return {}, {}
    guardado = _INDICE_ESCALACOES.get("quadro")
    if guardado is None or guardado is not player_rows:
        _INDICE_ESCALACOES["quadro"] = player_rows      # segura a referência
        _INDICE_ESCALACOES["indice"] = _montar_indice_escalacoes(player_rows)
    indice = _INDICE_ESCALACOES["indice"]
    azul, vermelho = indice.get(str(gameid), ({}, {}))
    # Cópias: quem chama pode mexer no que recebe, e o índice é compartilhado.
    return dict(azul), dict(vermelho)


def compute_delta_and_confidence(
    art: PlayersArtifact,
    *,
    blue_lineup: Dict[str, str],
    red_lineup: Dict[str, str],
    as_of: Optional[date] = None,
    match_league: Optional[str] = None,
    blue_league: Optional[str] = None,
    red_league: Optional[str] = None,
    require_both_sides: bool = True,
    lane_weights: Optional[Dict[str, float]] = None,
    league_weights: Optional[Dict[str, float]] = None,
    transfer_payload: Optional[Dict[str, Any]] = None,
    team_a: Optional[str] = None,
    team_b: Optional[str] = None,
    team_aliases_a: Optional[List[str]] = None,
    team_aliases_b: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Δ_players plus debug metrics.

    Returns:
      {
        "delta": float,
        "coverage": int,              # positions with usable players
        "coverage_total": 5,
        "avg_gp_min": float,
        "freshness_avg": float,       # 0..1, recency of used player ratings
        "details": [{pos, blue, red, r_blue, r_red, gp_blue, gp_red, used}]
      }
    """
    delta = 0.0
    used = 0
    gp_mins: List[float] = []
    freshness_vals: List[float] = []
    details: List[Dict[str, Any]] = []
    shared_league = str(match_league or "").strip()
    blue_league_s = str(blue_league or shared_league or "").strip()
    red_league_s = str(red_league or shared_league or "").strip()
    league_weight_blue = float(_league_weight_for(blue_league_s or shared_league, league_weights))
    league_weight_red = float(_league_weight_for(red_league_s or shared_league, league_weights))
    if blue_league_s and red_league_s and blue_league_s != red_league_s:
        league_weight = float(math.sqrt(max(0.0, league_weight_blue) * max(0.0, league_weight_red)))
    else:
        league_weight = float(league_weight_blue if blue_league_s else league_weight_red)

    # Accept flexible lineup keys (e.g., app may pass "ADC" while model uses "Bot").
    def _norm_lineup(d: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            posk = norm_pos(k)
            val = str(v or "").strip()
            if posk and val:
                out[posk] = val
        return out

    blue_n = _norm_lineup(blue_lineup)
    red_n = _norm_lineup(red_lineup)

    for pos in POS_CANON:
        pb = str(blue_n.get(pos, "") or "").strip()
        pr = str(red_n.get(pos, "") or "").strip()
        if require_both_sides and (not pb or not pr):
            details.append({"pos": pos, "blue": pb, "red": pr, "used": False})
            continue
        r_b, gp_b, dbg_b = art.r_effective_at(pb, pos, as_of=as_of, match_league=(blue_league_s or shared_league)) if pb else (0.0, 0, {"used": False})
        r_r, gp_r, dbg_r = art.r_effective_at(pr, pos, as_of=as_of, match_league=(red_league_s or shared_league)) if pr else (0.0, 0, {"used": False})
        ok = bool(pb and pr)
        lane_weight = float(_lane_weight_for(pos, lane_weights))
        if ok:
            lane_delta_raw = float(r_b - r_r)
            lane_delta_weighted = float(lane_delta_raw * lane_weight * league_weight)
            delta += lane_delta_weighted
            used += 1
            gp_mins.append(float(min(gp_b, gp_r)))
            # Recency confidence for this lane: stale ratings should contribute less confidence.
            if as_of is not None:
                def _fresh_from_last(last_date: Any) -> float:
                    ld = _parse_date(last_date)
                    if ld is None:
                        return 0.0
                    dd = max(0, int((as_of - ld).days))
                    hl = max(1.0, float(getattr(art, "half_life_days", 180.0) or 180.0) * 1.5)
                    return float(0.5 ** (float(dd) / float(hl)))
                fb = _fresh_from_last(dbg_b.get("last_date"))
                fr = _fresh_from_last(dbg_r.get("last_date"))
                freshness_vals.append(float(min(fb, fr)))
            else:
                freshness_vals.append(1.0)
        details.append(
            {
                "pos": pos,
                "blue": pb,
                "red": pr,
                "r_blue": float(r_b),
                "r_red": float(r_r),
                "lane_weight": float(lane_weight),
                "league_weight": float(league_weight),
                "league_weight_blue": float(league_weight_blue),
                "league_weight_red": float(league_weight_red),
                "blue_league": str(blue_league_s or shared_league),
                "red_league": str(red_league_s or shared_league),
                "lane_delta_raw": float((float(r_b) - float(r_r)) if ok else 0.0),
                "lane_delta_weighted": float((float(r_b) - float(r_r)) * lane_weight * league_weight) if ok else 0.0,
                "gp_blue": int(gp_b),
                "gp_red": int(gp_r),
                "blue_dbg": dbg_b,
                "red_dbg": dbg_r,
                "used": ok,
            }
        )

    avg_gp_min = float(sum(gp_mins) / len(gp_mins)) if gp_mins else 0.0
    freshness_avg = float(sum(freshness_vals) / len(freshness_vals)) if freshness_vals else 0.0
    out = {
        "delta": float(delta),
        "coverage": int(used),
        "coverage_total": 5,
        "avg_gp_min": float(avg_gp_min),
        "freshness_avg": float(max(0.0, min(1.0, freshness_avg))),
        "league_weight": float(league_weight),
        "league_weight_blue": float(league_weight_blue),
        "league_weight_red": float(league_weight_red),
        "blue_league": str(blue_league_s or shared_league),
        "red_league": str(red_league_s or shared_league),
        "details": details,
    }
    try:
        lineup_transfer_summary = build_lineup_transfer_summary(
            payload=(transfer_payload if isinstance(transfer_payload, dict) else None),
            team_a=team_a,
            team_b=team_b,
            blue_lineup=blue_n,
            red_lineup=red_n,
            team_aliases_a=team_aliases_a,
            team_aliases_b=team_aliases_b,
        )
    except Exception:
        lineup_transfer_summary = {"enabled": False, "applied": False, "reason": "error", "delta_adj": 0.0}
    out["player_portability_summary"] = dict(lineup_transfer_summary) if isinstance(lineup_transfer_summary, dict) else {}
    out["delta_transfer_adj"] = float((lineup_transfer_summary or {}).get("delta_adj", 0.0) or 0.0) if isinstance(lineup_transfer_summary, dict) else 0.0
    return out


# -----------------------------
# Prior fitting from byTournament stats
# -----------------------------


def _parse_pct(x: Any) -> float:
    s = str(x or "").strip().replace("%", "")
    if not s:
        return float("nan")
    try:
        return float(s) / 100.0
    except Exception:
        return float("nan")


def load_bytournament_stats(path: str) -> pd.DataFrame:
    """Load OE byTournament player stats (Copy/Download This Table)."""
    df = pd.read_csv(path, low_memory=False)
    # expected columns: Player, Team, Pos, GP, GD10, XPD10, CSD10, DMG%, DTH%
    rename = {"Player": "player", "Team": "team", "Pos": "pos", "GP": "gp"}
    for k, v in list(rename.items()):
        if k in df.columns:
            df = df.rename(columns={k: v})
    if "pos" in df.columns:
        df["pos_can"] = df["pos"].apply(norm_pos)
    else:
        df["pos_can"] = ""
    df = df[df["pos_can"].isin(POS_CANON)].copy()
    if df.empty:
        return pd.DataFrame(columns=["player", "team", "pos_can", "gp"])

    # numeric conversions
    if "gp" in df.columns:
        df["gp"] = pd.to_numeric(df["gp"], errors="coerce").fillna(0).astype(int)
    else:
        df["gp"] = 0

    def colpick(*names: str) -> Optional[str]:
        for n in names:
            if n in df.columns:
                return n
        return None

    c_gd10 = colpick("GD10")
    c_xpd10 = colpick("XPD10")
    c_csd10 = colpick("CSD10")
    c_dmg = colpick("DMG%", "DMG_PCT")
    c_dth = colpick("DTH%", "DTH_PCT")

    df["gd10"] = pd.to_numeric(df[c_gd10], errors="coerce") if c_gd10 else float("nan")
    df["xpd10"] = pd.to_numeric(df[c_xpd10], errors="coerce") if c_xpd10 else float("nan")
    df["csd10"] = pd.to_numeric(df[c_csd10], errors="coerce") if c_csd10 else float("nan")
    df["dmgp"] = df[c_dmg].apply(_parse_pct) if c_dmg else float("nan")
    df["dthp"] = df[c_dth].apply(_parse_pct) if c_dth else float("nan")

    # z-score per position (robust-ish)
    out_rows: List[pd.DataFrame] = []
    for _pos, g in df.groupby("pos_can"):
        gg = g.copy()
        for c in ["gd10", "xpd10", "csd10", "dmgp", "dthp"]:
            med = gg[c].median(skipna=True)
            gg[c] = gg[c].fillna(med if pd.notna(med) else 0.0)
        for c in ["gd10", "xpd10", "csd10", "dmgp", "dthp"]:
            mu = float(gg[c].mean()) if gg.shape[0] else 0.0
            sd = float(gg[c].std(ddof=0)) if gg.shape[0] else 0.0
            if not (sd > 1e-9):
                gg[f"z_{c}"] = 0.0
            else:
                gg[f"z_{c}"] = (gg[c] - mu) / sd
        out_rows.append(gg)

    out = pd.concat(out_rows, ignore_index=True) if out_rows else df.copy()
    out["player"] = out.get("player", "").astype(str).str.strip()
    out["team"] = out.get("team", "").astype(str).str.strip()
    out = out[["player", "team", "pos_can", "gp", "z_gd10", "z_xpd10", "z_csd10", "z_dmgp", "z_dthp"]].copy()
    return out


def fit_prior_beta(
    stats: pd.DataFrame,
    *,
    player_rows: pd.DataFrame,
    games: pd.DataFrame,
    l2: float = 1e-3,
    iters: int = 50,
) -> Tuple[Optional[List[float]], Dict[str, Dict[str, Any]]]:
    """Fit beta for prior from byTournament stats.

    We build a game feature vector as the **sum across positions** of
    (feat(blue_player) - feat(red_player)) for 5 features.

    Returns:
      beta (len=5) or None if insufficient signal,
      prior_scores mapping: key -> {prior, gp}
    """
    if stats is None or stats.empty:
        return None, {}
    if player_rows is None or player_rows.empty:
        return None, {}
    if games is None or games.empty:
        return None, {}

    feats = ["z_gd10", "z_xpd10", "z_csd10", "z_dmgp", "z_dthp"]

    # Map (playername,pos) -> stable identity (player_key) from match_data.
    # byTournament stats don't carry playerid; this bridge lets the prior
    # initialize the same person even when names change.
    namepos_to_key: Dict[Tuple[str, str], str] = {}
    try:
        if player_rows is not None and not player_rows.empty and "player_key" in player_rows.columns:
            tmp = player_rows[["playername", "pos_can", "player_key"]].copy()
            tmp["_pn"] = tmp["playername"].astype(str).str.strip().str.lower()
            tmp["_pk"] = tmp["player_key"].astype(str).str.strip()
            tmp = tmp[tmp["_pn"].ne("") & tmp["_pk"].ne("") & tmp["pos_can"].isin(POS_CANON)].copy()
            if not tmp.empty:
                # pick most frequent key for each (name,pos)
                for (pn, pos), g in tmp.groupby(["_pn", "pos_can"], sort=False):
                    vc = g["_pk"].value_counts()
                    if not vc.empty:
                        namepos_to_key[(str(pn), str(pos))] = str(vc.index[0])
    except Exception:
        namepos_to_key = {}

    # lookup (player,pos) -> vector
    lut: Dict[str, Tuple[List[float], int]] = {}
    for r in stats.itertuples(index=False):
        pl_name = str(r.player)
        pos = str(r.pos_can)
        gp = int(r.gp)
        vec = [float(getattr(r, f)) for f in feats]
        key = namepos_to_key.get((pl_name.strip().lower(), pos), pl_name)
        lut[_player_key(str(key), pos)] = (vec, gp)

    X: List[List[float]] = []
    Y: List[int] = []

    gids = set(games["gameid"].astype(str).tolist()) if "gameid" in games.columns else set()
    pr = player_rows[player_rows["gameid"].astype(str).isin(gids)].copy()
    if pr.empty:
        return None, {}

    # build lineups per game
    for g in games.itertuples(index=False):
        gid = str(g.gameid)
        y = int(g.blue_win)
        bl, rl = lineup_for_game(pr, gid)
        x = [0.0] * len(feats)
        used_any = False
        for pos in POS_CANON:
            pb = str(bl.get(pos, "") or "").strip()
            prd = str(rl.get(pos, "") or "").strip()
            if not pb or not prd:
                continue
            vb = lut.get(_player_key(pb, pos))
            vr = lut.get(_player_key(prd, pos))
            if vb is None or vr is None:
                continue
            used_any = True
            for i in range(len(feats)):
                x[i] += float(vb[0][i] - vr[0][i])
        if used_any:
            X.append(x)
            Y.append(y)

    if len(Y) < 100:
        # too little overlap between stats table and match_data
        return None, {}

    # Newton-Raphson for logistic regression beta
    beta = [0.0] * len(feats)
    l2 = float(l2)

    for _ in range(int(iters)):
        # grad/hess
        grad = [0.0] * len(feats)
        hess = [[0.0] * len(feats) for _ in range(len(feats))]
        for x, y in zip(X, Y):
            z = sum(beta[i] * x[i] for i in range(len(feats)))
            p = sigmoid(z)
            w = p * (1.0 - p)
            for i in range(len(feats)):
                grad[i] += (p - y) * x[i]
                for j in range(len(feats)):
                    hess[i][j] += w * x[i] * x[j]
        # L2
        for i in range(len(feats)):
            grad[i] += l2 * beta[i]
            hess[i][i] += l2

        # solve hess * step = grad (small 5x5 -> naive Gauss)
        step = _solve_linear_5x5(hess, grad)
        if step is None:
            break
        # update
        max_step = 0.0
        for i in range(len(feats)):
            beta[i] -= step[i]
            max_step = max(max_step, abs(step[i]))
        if max_step < 1e-6:
            break

    # build prior_scores
    prior_scores: Dict[str, Dict[str, Any]] = {}
    for k, (vec, gp) in lut.items():
        prior = float(sum(beta[i] * vec[i] for i in range(len(feats))))
        prior_scores[k] = {"prior": prior, "gp": int(gp)}
    return beta, prior_scores


def _solve_linear_5x5(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve Ax=b for small dense matrix via Gaussian elimination."""
    n = len(b)
    # build augmented
    m = [[*row[:], b[i]] for i, row in enumerate(a)]
    try:
        for col in range(n):
            # pivot
            piv = col
            for r in range(col + 1, n):
                if abs(m[r][col]) > abs(m[piv][col]):
                    piv = r
            if abs(m[piv][col]) < 1e-12:
                return None
            if piv != col:
                m[col], m[piv] = m[piv], m[col]
            # normalize
            div = m[col][col]
            for c in range(col, n + 1):
                m[col][c] /= div
            # eliminate
            for r in range(n):
                if r == col:
                    continue
                fac = m[r][col]
                if abs(fac) < 1e-12:
                    continue
                for c in range(col, n + 1):
                    m[r][c] -= fac * m[col][c]
        return [m[i][n] for i in range(n)]
    except Exception:
        return None


# -----------------------------
# w_lineup fit
# -----------------------------


def fit_w_lineup(
    *,
    y: List[int],
    logit_base: List[float],
    delta: List[float],
    l2: float = 0.0,
    iters: int = 50,
) -> float:
    """Fit scalar w by minimizing logloss on holdout."""
    if not y or not delta or len(y) != len(delta) or len(y) != len(logit_base):
        return 0.0

    w = 0.0
    l2 = float(l2)
    for _ in range(int(iters)):
        grad = 0.0
        hess = 0.0
        for yi, lb, di in zip(y, logit_base, delta):
            z = float(lb + w * di)
            p = sigmoid(z)
            grad += (p - float(yi)) * di
            hess += (p * (1.0 - p)) * (di * di)
        grad += l2 * w
        hess += l2
        if hess <= 1e-12:
            break
        step = grad / hess
        w -= step
        if abs(step) < 1e-6:
            break
    return float(w)
