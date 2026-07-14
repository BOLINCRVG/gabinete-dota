from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from dota_ml import predict_winner_matchup, winner_ml_bundle_for_path

CSV_NAME = "data.csv"
SAVED_GAMES_FILE = "saved_games_dota.json"
PERIOD_SPECS: List[Tuple[str, Optional[int]]] = [
    ("Ano todo", None),
    ("Últimos 15", 15),
    ("Últimos 10", 10),
    ("Últimos 5", 5),
]
TOTAL_MARKETS = {
    "Kills": "totalkills",
    "Torres": "totaltowers",
    "Roshans": "totalroshans",
    "Barracks": "totalbarracks",
    "Tempo": "time_min",
}
HANDICAP_MARKETS = {
    "Kills": "diff_kills",
    "Torres": "diff_towers",
    "Roshans": "diff_roshans",
    "Barracks": "diff_barracks",
}


def app_dir() -> Path:
    return Path(__file__).resolve().parent


def csv_path() -> Path:
    return app_dir() / CSV_NAME


def saved_games_path() -> Path:
    return app_dir() / SAVED_GAMES_FILE


def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )


def norm_team(text: str) -> str:
    base = _strip_accents(str(text or "")).lower().strip()
    for suffix in [" esports", " e-sports", " esport", " gaming", " team", " club"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    base = base.replace(".", " ").replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def parse_float_list(text: str) -> List[float]:
    raw = (text or "").strip()
    if not raw:
        return []
    out: List[float] = []
    for part in raw.replace(";", ",").split(","):
        token = part.strip().replace(" ", "")
        if not token:
            continue
        try:
            out.append(float(token.replace(",", ".")))
        except ValueError:
            continue
    return out


def parse_time_line(text: str) -> List[float]:
    out: List[float] = []
    for part in (text or "").replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            try:
                mm, ss = token.split(":", 1)
                out.append(int(mm) + int(ss) / 60.0)
            except Exception:
                continue
        else:
            try:
                out.append(float(token.replace(",", ".")))
            except ValueError:
                continue
    return out


def fmt_mmss_from_minutes(minutes: float) -> str:
    try:
        if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
            return "-"
        total_sec = int(round(float(minutes) * 60))
        mm = total_sec // 60
        ss = total_sec % 60
        return f"{mm:02d}:{ss:02d}"
    except Exception:
        return "-"


def normal_cdf(x: float, mean: float, sd: float) -> float:
    if not math.isfinite(mean):
        return 0.5
    if not math.isfinite(sd) or sd <= 1e-9:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def clamp_prob(prob: float) -> float:
    return max(0.001, min(0.999, float(prob)))


def fair_odds(prob: float) -> float:
    p = clamp_prob(prob)
    return 1.0 / p


def fmt_hist_prob(wins: int, total: int, pushes: int = 0, laplace: bool = True) -> str:
    valid = max(0, int(total) - int(pushes))
    if valid <= 0:
        return "-"
    if laplace:
        p = (int(wins) + 1) / (valid + 2)
    else:
        p = int(wins) / valid
    return f"{wins}/{valid} ({fair_odds(p):.2f}x)"


def _laplace_rate(wins: int, total: int) -> float:
    valid = max(0, int(total))
    if valid <= 0:
        return 0.5
    return clamp_prob((int(wins) + 1) / (valid + 2))


def _hist_prob_from_text(text: str) -> Optional[float]:
    raw = str(text or "").strip()
    if not raw or raw == "-":
        return None
    match = re.search(r"\(([\d.]+)x\)", raw)
    if not match:
        return None
    try:
        odd = float(match.group(1))
    except Exception:
        return None
    if not math.isfinite(odd) or odd <= 0:
        return None
    return clamp_prob(1.0 / odd)


def _blend_hist_prob(
    league_hist: str,
    h2h_hist: str,
    *,
    league_n: int = 0,
    h2h_n: int = 0,
) -> Optional[float]:
    weights: List[Tuple[float, float]] = []
    p_league = _hist_prob_from_text(league_hist)
    if p_league is not None:
        weights.append((float(max(1, int(league_n))), float(p_league)))
    p_h2h = _hist_prob_from_text(h2h_hist)
    if p_h2h is not None:
        weights.append((float(max(1, int(h2h_n))), float(p_h2h)))
    if not weights:
        return None
    total_w = sum(w for w, _ in weights)
    return clamp_prob(sum(w * p for w, p in weights) / total_w)


def _mix_model_hist_prob(model_prob: float, hist_prob: Optional[float]) -> float:
    model = clamp_prob(model_prob)
    if hist_prob is None or not math.isfinite(float(hist_prob)):
        return model
    return clamp_prob((0.65 * model) + (0.35 * float(hist_prob)))


def _confidence_score(league_n: int, h2h_n: int, model_prob: float, hist_prob: Optional[float]) -> int:
    score = 15.0
    score += min(40.0, max(0, int(league_n)) * 2.0)
    score += min(25.0, max(0, int(h2h_n)) * 4.0)
    if math.isfinite(float(model_prob)):
        score += 10.0
    if hist_prob is not None and math.isfinite(float(hist_prob)) and math.isfinite(float(model_prob)):
        diff = abs(float(model_prob) - float(hist_prob))
        score += max(0.0, 15.0 - (diff * 100.0))
    return int(max(0, min(100, round(score))))


def _fmt_odd(prob: Optional[float]) -> str:
    if prob is None or not math.isfinite(float(prob)):
        return "-"
    return f"{fair_odds(float(prob)):.2f}"


def _blend_probs(base_prob: float, extra_prob: float, weight: float) -> float:
    w = max(0.0, min(1.0, float(weight)))
    return clamp_prob((clamp_prob(base_prob) * (1.0 - w)) + (clamp_prob(extra_prob) * w))


def _winner_ml_blend_weight(ml_result: dict, team1_games: int, team2_games: int, h2h_games: int) -> float:
    if not bool(ml_result.get("ready")) or ml_result.get("t1_prob") is None:
        return 0.0
    if min(int(team1_games), int(team2_games)) < 2:
        return 0.0
    auc = ml_result.get("holdout_auc")
    brier = ml_result.get("holdout_brier")
    baseline_brier = ml_result.get("baseline_brier")
    quality = 0.25
    if auc is not None and math.isfinite(float(auc)):
        quality += max(0.0, min(0.5, (float(auc) - 0.5) / 0.10))
    if (
        brier is not None
        and baseline_brier is not None
        and math.isfinite(float(brier))
        and math.isfinite(float(baseline_brier))
    ):
        quality += max(0.0, min(0.25, (float(baseline_brier) - float(brier)) / 0.05))
    quality = max(0.0, min(1.0, quality))
    coverage = (
        min(1.0, max(0.0, float(team1_games)) / 18.0) * 0.40
        + min(1.0, max(0.0, float(team2_games)) / 18.0) * 0.40
        + min(1.0, max(0.0, float(h2h_games)) / 6.0) * 0.20
    )
    return max(0.0, min(0.35, 0.08 + (0.27 * quality * coverage)))


def _winner_ml_signal(
    ready: bool,
    *,
    holdout_auc: Optional[float],
    holdout_brier: Optional[float],
    baseline_brier: Optional[float],
    blend_weight: float,
    reason: str = "",
) -> tuple[str, str]:
    if not ready:
        detail = str(reason or "Sem treino/holdout utilizavel").strip()
        return "ML indisponível", detail
    if (
        holdout_auc is None
        or holdout_brier is None
        or baseline_brier is None
        or not math.isfinite(float(holdout_auc))
        or not math.isfinite(float(holdout_brier))
        or not math.isfinite(float(baseline_brier))
    ):
        return "Usar heurístico", "ML sem métricas estáveis"
    auc = float(holdout_auc)
    brier = float(holdout_brier)
    baseline = float(baseline_brier)
    gain = baseline - brier
    weight = max(0.0, float(blend_weight))
    if auc >= 0.60 and gain >= 0.008 and weight >= 0.25:
        label = "ML forte"
    elif auc >= 0.57 and gain >= 0.003 and weight >= 0.18:
        label = "ML moderado"
    elif auc >= 0.54 and gain > 0.0 and weight >= 0.08:
        label = "ML fraco"
    else:
        label = "Usar heurístico"
    detail = f"AUC {auc:.3f} | ΔBrier {gain:+.3f} | Blend {weight * 100:.0f}%"
    return label, detail


@dataclass
class LinesConfig:
    total_kills: List[float] = field(default_factory=list)
    total_towers: List[float] = field(default_factory=list)
    total_roshans: List[float] = field(default_factory=list)
    total_barracks: List[float] = field(default_factory=list)
    total_time_min: List[float] = field(default_factory=list)
    hc_kills_t1: List[float] = field(default_factory=list)
    hc_towers_t1: List[float] = field(default_factory=list)
    hc_roshans_t1: List[float] = field(default_factory=list)
    hc_barracks_t1: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_kills": list(self.total_kills),
            "total_towers": list(self.total_towers),
            "total_roshans": list(self.total_roshans),
            "total_barracks": list(self.total_barracks),
            "total_time_min": list(self.total_time_min),
            "hc_kills_t1": list(self.hc_kills_t1),
            "hc_towers_t1": list(self.hc_towers_t1),
            "hc_roshans_t1": list(self.hc_roshans_t1),
            "hc_barracks_t1": list(self.hc_barracks_t1),
        }

    @staticmethod
    def from_dict(data: Optional[dict]) -> "LinesConfig":
        raw = data or {}
        return LinesConfig(
            total_kills=[float(x) for x in raw.get("total_kills", [])],
            total_towers=[float(x) for x in raw.get("total_towers", [])],
            total_roshans=[float(x) for x in raw.get("total_roshans", [])],
            total_barracks=[float(x) for x in raw.get("total_barracks", [])],
            total_time_min=[float(x) for x in raw.get("total_time_min", [])],
            hc_kills_t1=[float(x) for x in raw.get("hc_kills_t1", [])],
            hc_towers_t1=[float(x) for x in raw.get("hc_towers_t1", [])],
            hc_roshans_t1=[float(x) for x in raw.get("hc_roshans_t1", [])],
            hc_barracks_t1=[float(x) for x in raw.get("hc_barracks_t1", [])],
        )

    def is_empty(self) -> bool:
        return not any(
            [
                self.total_kills,
                self.total_towers,
                self.total_roshans,
                self.total_barracks,
                self.total_time_min,
                self.hc_kills_t1,
                self.hc_towers_t1,
                self.hc_roshans_t1,
                self.hc_barracks_t1,
            ]
        )


class DotaDataset:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else csv_path()
        if not self.path.exists():
            raise FileNotFoundError(f"Arquivo CSV nao encontrado: {self.path}")
        df = pd.read_csv(self.path, low_memory=False)
        required = {
            "match_id", "date", "league", "teamname", "opp_teamname", "duration", "result",
            "teamkills", "oppkills", "towers", "opp_towers", "roshans", "opp_roshans", "barracks", "opp_barracks"
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"CSV sem colunas obrigatorias: {missing}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
        if "year" not in df.columns:
            df["year"] = df["date"].dt.year
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["date"].dt.year).astype(int)
        df["league"] = df["league"].fillna("Sem campeonato").astype(str).str.strip()
        df["teamname"] = df["teamname"].astype(str).str.strip()
        df["opp_teamname"] = df["opp_teamname"].astype(str).str.strip()
        df["teamname_norm"] = df["teamname"].apply(norm_team)
        df["opp_teamname_norm"] = df["opp_teamname"].apply(norm_team)
        df["match_uid"] = df["match_id"].astype(str)
        df["side"] = df.get("side", "").fillna("").astype(str).str.lower().str.strip()
        for col in [
            "duration", "result", "teamkills", "oppkills", "towers", "opp_towers",
            "roshans", "opp_roshans", "barracks", "opp_barracks",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["time_min"] = df["duration"] / 60.0
        df["totalkills"] = df["teamkills"] + df["oppkills"]
        df["totaltowers"] = df["towers"] + df["opp_towers"]
        df["totalroshans"] = df["roshans"] + df["opp_roshans"]
        df["totalbarracks"] = df["barracks"] + df["opp_barracks"]
        df["diff_kills"] = df["teamkills"] - df["oppkills"]
        df["diff_towers"] = df["towers"] - df["opp_towers"]
        df["diff_roshans"] = df["roshans"] - df["opp_roshans"]
        df["diff_barracks"] = df["barracks"] - df["opp_barracks"]
        df["result"] = df["result"].fillna(0).astype(int)
        self.df = df.sort_values("date", ascending=False).reset_index(drop=True)

    def years(self) -> List[int]:
        return sorted(self.df["year"].dropna().astype(int).unique().tolist(), reverse=True)

    def leagues(self, year: Optional[int] = None) -> List[str]:
        scope = self.filter_scope(year, "(Todas)")
        return ["(Todas)"] + sorted(scope["league"].dropna().astype(str).unique().tolist())

    def teams_for_scope(self, year: Optional[int], league: str) -> List[str]:
        scope = self.filter_scope(year, league)
        teams = pd.unique(pd.concat([scope["teamname"], scope["opp_teamname"]], ignore_index=True))
        return sorted([str(team) for team in teams if str(team).strip()])

    def filter_scope(self, year: Optional[int], league: str) -> pd.DataFrame:
        out = self.df.copy()
        if year is not None:
            out = out[out["year"] == int(year)].copy()
        if league and league != "(Todas)":
            out = out[out["league"] == league].copy()
        return out.sort_values("date", ascending=False)

    def team_games(self, year: Optional[int], league: str, team: str) -> pd.DataFrame:
        scope = self.filter_scope(year, league)
        team_norm = norm_team(team)
        out = scope[scope["teamname_norm"] == team_norm].copy()
        return out.sort_values("date", ascending=False).drop_duplicates(subset=["match_uid"])

    def team_games_all_leagues(self, year: Optional[int], team: str) -> pd.DataFrame:
        return self.team_games(year, "(Todas)", team)

    def pair_games(self, year: Optional[int], league: str, team1: str, team2: str) -> pd.DataFrame:
        scope = self.filter_scope(year, league)
        t1 = norm_team(team1)
        t2 = norm_team(team2)
        mask = scope["teamname_norm"].isin([t1, t2]) & scope["opp_teamname_norm"].isin([t1, t2])
        h2h = scope[mask].copy()
        if h2h.empty:
            return pd.DataFrame()
        pivot = h2h.pivot_table(
            index=["match_uid", "date", "league"],
            columns="teamname_norm",
            values=["teamkills", "towers", "roshans", "barracks", "duration", "result"],
            aggfunc="first",
        )
        if t1 not in pivot["teamkills"].columns or t2 not in pivot["teamkills"].columns:
            return pd.DataFrame()
        out = pd.DataFrame(index=pivot.index).reset_index()
        out["kills_t1"] = pivot["teamkills"][t1].values
        out["kills_t2"] = pivot["teamkills"][t2].values
        out["towers_t1"] = pivot["towers"][t1].values
        out["towers_t2"] = pivot["towers"][t2].values
        out["roshans_t1"] = pivot["roshans"][t1].values
        out["roshans_t2"] = pivot["roshans"][t2].values
        out["barracks_t1"] = pivot["barracks"][t1].values
        out["barracks_t2"] = pivot["barracks"][t2].values
        dur_t1 = pivot["duration"][t1].values
        dur_t2 = pivot["duration"][t2].values
        out["duration"] = np.where(~np.isnan(dur_t1), dur_t1, dur_t2)
        out["time_min"] = out["duration"] / 60.0
        out["result_t1"] = pivot["result"][t1].values
        out["totalkills"] = out["kills_t1"] + out["kills_t2"]
        out["totaltowers"] = out["towers_t1"] + out["towers_t2"]
        out["totalroshans"] = out["roshans_t1"] + out["roshans_t2"]
        out["totalbarracks"] = out["barracks_t1"] + out["barracks_t2"]
        out["diff_kills"] = out["kills_t1"] - out["kills_t2"]
        out["diff_towers"] = out["towers_t1"] - out["towers_t2"]
        out["diff_roshans"] = out["roshans_t1"] - out["roshans_t2"]
        out["diff_barracks"] = out["barracks_t1"] - out["barracks_t2"]
        return out.sort_values("date", ascending=False)

    def championships_for_matchup(self, year: Optional[int], team1: str, team2: str) -> List[str]:
        scope = self.filter_scope(year, "(Todas)")
        teams = {norm_team(team1), norm_team(team2)}
        mask = scope["teamname_norm"].isin(teams) | scope["opp_teamname_norm"].isin(teams)
        data = scope[mask].copy()
        if data.empty:
            return []
        latest = data.groupby("league", dropna=False)["date"].max().sort_values(ascending=False)
        return latest.index.astype(str).tolist()

    def team_id_candidates(self, team: str) -> pd.DataFrame:
        team_norm = norm_team(team)
        frames = []
        for side_col, id_col in [("teamname_norm", "team_id"), ("opp_teamname_norm", "opp_team_id")]:
            if side_col not in self.df.columns or id_col not in self.df.columns:
                continue
            frame = self.df[self.df[side_col] == team_norm].copy()
            if frame.empty:
                continue
            frame["resolved_team_id"] = pd.to_numeric(frame[id_col], errors="coerce")
            frame = frame[frame["resolved_team_id"].notna()].copy()
            if frame.empty:
                continue
            frames.append(
                frame[["date", "league", "resolved_team_id"]].rename(columns={"resolved_team_id": "team_id"})
            )
        if not frames:
            return pd.DataFrame(columns=["team_id", "games", "last_seen", "last_league"])
        sample = pd.concat(frames, ignore_index=True)
        sample["team_id"] = sample["team_id"].astype("int64")
        grouped = (
            sample.groupby("team_id", dropna=False)
            .agg(
                games=("team_id", "size"),
                last_seen=("date", "max"),
                last_league=("league", lambda s: str(s.dropna().iloc[0]) if not s.dropna().empty else ""),
            )
            .reset_index()
            .sort_values(["last_seen", "games", "team_id"], ascending=[False, False, True])
        )
        return grouped.reset_index(drop=True)

    def resolve_team_id(self, team: str) -> Optional[int]:
        candidates = self.team_id_candidates(team)
        if candidates.empty:
            return None
        try:
            return int(candidates.iloc[0]["team_id"])
        except Exception:
            return None

    def league_match_table(self, year: Optional[int], league: str) -> pd.DataFrame:
        scope = self.filter_scope(year, league)
        return scope.sort_values("date", ascending=False).drop_duplicates(subset=["match_uid"])


def load_saved_games(path: Optional[Path] = None) -> Dict[str, dict]:
    src = Path(path) if path else saved_games_path()
    if not src.exists():
        return {}
    raw = json.loads(src.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.values()
    else:
        items = raw
    out: Dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if key:
            out[str(key)] = item
    return out


def save_saved_games(records: Dict[str, dict], path: Optional[Path] = None) -> None:
    target = Path(path) if path else saved_games_path()
    payload = list(records.values())
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_saved_key(year: Optional[int], league: str, team1: str, team2: str) -> str:
    season = str(year or "all")
    pair = "::".join(sorted([norm_team(team1), norm_team(team2)]))
    return f"{season}::{norm_team(league or '(Todas)')}::{pair}"


def upsert_saved_game(
    records: Dict[str, dict],
    year: Optional[int],
    league: str,
    team1: str,
    team2: str,
    lines: LinesConfig,
    *,
    event_name: str = "",
    match_format: str = "",
) -> Dict[str, dict]:
    key = make_saved_key(year, league, team1, team2)
    now = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    previous = records.get(key, {})
    records[key] = {
        "key": key,
        "year": year,
        "league": league,
        "event_name": str(event_name or previous.get("event_name") or "").strip(),
        "match_format": str(match_format or previous.get("match_format") or "").strip(),
        "team1": team1,
        "team2": team2,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
        "lines": lines.to_dict(),
    }
    return records


def _team_stat_block(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "games": 0,
            "win_rate": np.nan,
            "record": "-",
            "blue": np.nan,
            "red": np.nan,
            "avg_kills": np.nan,
            "avg_towers": np.nan,
            "avg_roshans": np.nan,
            "avg_barracks": np.nan,
            "avg_time": np.nan,
        }
    wins = int(df["result"].sum())
    games = int(len(df))
    losses = games - wins
    blue = df[df["side"] == "radiant"]
    red = df[df["side"] == "dire"]
    return {
        "games": games,
        "win_rate": wins / games if games else np.nan,
        "record": f"{wins}W-{losses}L",
        "blue": float(blue["result"].mean()) if not blue.empty else np.nan,
        "red": float(red["result"].mean()) if not red.empty else np.nan,
        "avg_kills": float(df["totalkills"].mean()),
        "avg_towers": float(df["totaltowers"].mean()),
        "avg_roshans": float(df["totalroshans"].mean()),
        "avg_barracks": float(df["totalbarracks"].mean()),
        "avg_time": float(df["time_min"].mean()),
    }


def build_overview(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> dict:
    t1_games = dataset.team_games(year, league, team1)
    t2_games = dataset.team_games(year, league, team2)
    h2h_games = dataset.pair_games(year, league, team1, team2)
    return {
        "team1": _team_stat_block(t1_games),
        "team2": _team_stat_block(t2_games),
        "recent_team1": t1_games.head(10).copy(),
        "recent_team2": t2_games.head(10).copy(),
        "h2h": h2h_games.head(10).copy(),
    }


def _union_recent_unique(df1: pd.DataFrame, df2: pd.DataFrame, per_team: Optional[int]) -> pd.DataFrame:
    if per_team is None:
        combo = pd.concat([df1, df2], ignore_index=True)
        return combo.sort_values("date", ascending=False).drop_duplicates(subset=["match_uid"])

    a = df1.sort_values("date", ascending=False).reset_index(drop=True)
    b = df2.sort_values("date", ascending=False).reset_index(drop=True)
    seed = pd.concat([a.head(per_team), b.head(per_team)], ignore_index=True)
    unique = seed.sort_values("date", ascending=False).drop_duplicates(subset=["match_uid"])
    target = per_team * 2
    if len(unique) >= target:
        return unique.head(target).copy()

    seen = set(unique["match_uid"].astype(str).tolist())
    extra_rows = []
    extra_a = a.iloc[per_team:]
    extra_b = b.iloc[per_team:]
    extra = pd.concat([extra_a, extra_b], ignore_index=True).sort_values("date", ascending=False)
    for _, row in extra.iterrows():
        match_uid = str(row["match_uid"])
        if match_uid in seen:
            continue
        extra_rows.append(row)
        seen.add(match_uid)
        if len(unique) + len(extra_rows) >= target:
            break
    if extra_rows:
        unique = pd.concat([unique, pd.DataFrame(extra_rows)], ignore_index=True)
    return unique.sort_values("date", ascending=False).head(target).copy()


def _period_frames(t1_games: pd.DataFrame, t2_games: pd.DataFrame, h2h_games: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    league_periods: Dict[str, pd.DataFrame] = {}
    h2h_periods: Dict[str, pd.DataFrame] = {}
    for label, per_team in PERIOD_SPECS:
        league_periods[label] = _union_recent_unique(t1_games, t2_games, per_team)
        if per_team is None:
            h2h_periods[label] = h2h_games.copy()
        else:
            h2h_periods[label] = h2h_games.head(per_team).copy()
    return league_periods, h2h_periods


def _ou_counts(values: np.ndarray, line: float) -> Tuple[int, int, int]:
    if values.size == 0:
        return 0, 0, 0
    return int(np.sum(values > line)), int(np.sum(values < line)), int(np.sum(values == line))


def _values_from_df(df: pd.DataFrame, col: str) -> np.ndarray:
    if df.empty or col not in df.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def analyze_total_market(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, market_label: str, line: float, laplace: bool = True) -> dict:
    col = TOTAL_MARKETS[market_label]
    t1_games = dataset.team_games(year, league, team1)
    t2_games = dataset.team_games(year, league, team2)
    h2h = dataset.pair_games(year, league, team1, team2)
    league_periods, h2h_periods = _period_frames(t1_games, t2_games, h2h)

    sample = pd.concat([t1_games[[col]], t2_games[[col]]], ignore_index=True)
    vals_sample = _values_from_df(sample, col)
    mean = float(np.mean(vals_sample)) if vals_sample.size else np.nan
    sd = float(np.std(vals_sample, ddof=1)) if vals_sample.size > 1 else float(np.std(vals_sample)) if vals_sample.size else np.nan
    if vals_sample.size:
        p_over_model = clamp_prob(1.0 - normal_cdf(line, mean, sd))
    else:
        p_over_model = 0.5
    out = {
        "market": market_label,
        "line": line,
        "expected_mean": mean,
        "sd": sd,
        "model_over_prob": p_over_model,
        "model_under_prob": 1.0 - p_over_model,
        "periods": {},
    }
    for label, df_period in league_periods.items():
        vals = _values_from_df(df_period, col)
        over, under, push = _ou_counts(vals, line)
        out["periods"].setdefault(label, {})["league_over"] = fmt_hist_prob(over, len(vals), push, laplace)
        out["periods"].setdefault(label, {})["league_under"] = fmt_hist_prob(under, len(vals), push, laplace)
        out["periods"].setdefault(label, {})["league_n"] = max(0, int(len(vals) - push))
    for label, df_period in h2h_periods.items():
        vals = _values_from_df(df_period, col)
        over, under, push = _ou_counts(vals, line)
        out["periods"].setdefault(label, {})["h2h_over"] = fmt_hist_prob(over, len(vals), push, laplace)
        out["periods"].setdefault(label, {})["h2h_under"] = fmt_hist_prob(under, len(vals), push, laplace)
        out["periods"].setdefault(label, {})["h2h_n"] = max(0, int(len(vals) - push))
    return out


def analyze_handicap_market(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, market_label: str, line: float, laplace: bool = True) -> dict:
    col = HANDICAP_MARKETS[market_label]
    t1_games = dataset.team_games(year, league, team1)
    t2_games = dataset.team_games(year, league, team2)
    h2h = dataset.pair_games(year, league, team1, team2)
    sample_t1 = _values_from_df(t1_games, col)
    sample_t2 = _values_from_df(t2_games, col)
    combined = np.concatenate([sample_t1, -sample_t2]) if sample_t1.size or sample_t2.size else np.array([], dtype=float)
    mean = float(np.mean(combined)) if combined.size else np.nan
    sd = float(np.std(combined, ddof=1)) if combined.size > 1 else float(np.std(combined)) if combined.size else np.nan
    thr = -line
    if combined.size:
        p_t1_cover = clamp_prob(1.0 - normal_cdf(thr, mean, sd))
    else:
        p_t1_cover = 0.5
    out = {
        "market": market_label,
        "line": line,
        "expected_edge": mean,
        "sd": sd,
        "model_t1_prob": p_t1_cover,
        "model_t2_prob": 1.0 - p_t1_cover,
        "periods": {},
    }
    for label, per_team in PERIOD_SPECS:
        df_period = h2h.copy() if per_team is None else h2h.head(per_team).copy()
        vals = _values_from_df(df_period, col)
        wins_t1 = int(np.sum(vals > thr)) if vals.size else 0
        wins_t2 = int(np.sum(vals < thr)) if vals.size else 0
        pushes = int(np.sum(vals == thr)) if vals.size else 0
        out["periods"][label] = {
            "t1_cover": fmt_hist_prob(wins_t1, len(vals), pushes, laplace),
            "t2_cover": fmt_hist_prob(wins_t2, len(vals), pushes, laplace),
            "h2h_n": max(0, int(len(vals) - pushes)),
        }
    return out


def analyze_winner_market(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, laplace: bool = True) -> dict:
    t1_games = dataset.team_games(year, league, team1)
    t2_games = dataset.team_games(year, league, team2)
    h2h = dataset.pair_games(year, league, team1, team2)
    t1_games_all = dataset.team_games_all_leagues(year, team1)
    t2_games_all = dataset.team_games_all_leagues(year, team2)
    h2h_all = dataset.pair_games(year, "(Todas)", team1, team2)

    t1_res = pd.to_numeric(t1_games["result"] if "result" in t1_games.columns else pd.Series(dtype=float), errors="coerce")
    t2_res = pd.to_numeric(t2_games["result"] if "result" in t2_games.columns else pd.Series(dtype=float), errors="coerce")
    h2h_res = pd.to_numeric(h2h["result_t1"] if "result_t1" in h2h.columns else pd.Series(dtype=float), errors="coerce")

    n_t1 = int(t1_res.notna().sum())
    n_t2 = int(t2_res.notna().sum())
    n_h2h = int(h2h_res.notna().sum())
    w_t1 = int((t1_res > 0.5).sum())
    w_t2 = int((t2_res > 0.5).sum())
    w_h2h_t1 = int((h2h_res > 0.5).sum())

    p_t1_form = _laplace_rate(w_t1, n_t1)
    p_t2_form = _laplace_rate(w_t2, n_t2)
    p_form = clamp_prob((p_t1_form + (1.0 - p_t2_form)) / 2.0)
    p_h2h = _laplace_rate(w_h2h_t1, n_h2h)
    h2h_weight = min(0.30, (n_h2h / 15.0) * 0.30) if n_h2h > 0 else 0.0
    p_t1_heuristic = clamp_prob((p_form * (1.0 - h2h_weight)) + (p_h2h * h2h_weight))

    ml_bundle = winner_ml_bundle_for_path(str(dataset.path), dataset.path.stat().st_mtime)
    ml_result = predict_winner_matchup(
        ml_bundle,
        t1_games_all,
        t2_games_all,
        t1_games,
        t2_games,
        h2h_all,
    )
    ml_t1_prob = ml_result.get("t1_prob")
    ml_weight = _winner_ml_blend_weight(
        ml_result,
        int(len(t1_games_all)),
        int(len(t2_games_all)),
        max(int(n_h2h), int(ml_result.get("h2h_games_used", 0) or 0)),
    )
    p_t1_model = (
        _blend_probs(p_t1_heuristic, float(ml_t1_prob), ml_weight)
        if ml_t1_prob is not None and math.isfinite(float(ml_t1_prob)) and ml_weight > 0.0
        else p_t1_heuristic
    )
    if bool(ml_result.get("ready")) and ml_t1_prob is not None and math.isfinite(float(ml_t1_prob)):
        model_source = f"Blend Heuristico + ML ({ml_weight * 100:.0f}% ML)" if ml_weight > 0.0 else "Heuristico (ML sem peso)"
    else:
        model_source = "Heuristico"
    ml_signal_label, ml_signal_detail = _winner_ml_signal(
        bool(ml_result.get("ready")),
        holdout_auc=float(ml_result["holdout_auc"]) if ml_result.get("holdout_auc") is not None and math.isfinite(float(ml_result["holdout_auc"])) else None,
        holdout_brier=float(ml_result["holdout_brier"]) if ml_result.get("holdout_brier") is not None and math.isfinite(float(ml_result["holdout_brier"])) else None,
        baseline_brier=float(ml_result["baseline_brier"]) if ml_result.get("baseline_brier") is not None and math.isfinite(float(ml_result["baseline_brier"])) else None,
        blend_weight=ml_weight,
        reason=str(ml_result.get("reason", "")),
    )

    out = {
        "market": "ML",
        "team1_win_rate": p_t1_form,
        "team2_win_rate": p_t2_form,
        "h2h_win_rate_t1": p_h2h,
        "heuristic_t1_prob": p_t1_heuristic,
        "heuristic_t2_prob": 1.0 - p_t1_heuristic,
        "ml_t1_prob": clamp_prob(float(ml_t1_prob)) if ml_t1_prob is not None and math.isfinite(float(ml_t1_prob)) else None,
        "ml_t2_prob": clamp_prob(1.0 - float(ml_t1_prob)) if ml_t1_prob is not None and math.isfinite(float(ml_t1_prob)) else None,
        "ml_ready": bool(ml_result.get("ready")),
        "ml_reason": str(ml_result.get("reason", "")),
        "ml_model_label": str(ml_result.get("model_label", "")),
        "ml_signal_label": ml_signal_label,
        "ml_signal_detail": ml_signal_detail,
        "ml_blend_weight": float(ml_weight),
        "ml_train_rows": int(ml_result.get("train_rows", 0) or 0),
        "ml_holdout_rows": int(ml_result.get("holdout_rows", 0) or 0),
        "ml_train_matches": int(ml_result.get("train_matches", 0) or 0),
        "ml_holdout_matches": int(ml_result.get("holdout_matches", 0) or 0),
        "ml_holdout_auc": float(ml_result["holdout_auc"]) if ml_result.get("holdout_auc") is not None and math.isfinite(float(ml_result["holdout_auc"])) else None,
        "ml_holdout_brier": float(ml_result["holdout_brier"]) if ml_result.get("holdout_brier") is not None and math.isfinite(float(ml_result["holdout_brier"])) else None,
        "ml_baseline_brier": float(ml_result["baseline_brier"]) if ml_result.get("baseline_brier") is not None and math.isfinite(float(ml_result["baseline_brier"])) else None,
        "model_source": model_source,
        "model_t1_prob": p_t1_model,
        "model_t2_prob": 1.0 - p_t1_model,
        "team1_games": n_t1,
        "team2_games": n_t2,
        "h2h_games": n_h2h,
        "periods": {},
    }

    for label, per_team in PERIOD_SPECS:
        t1_period_df = t1_games.copy() if per_team is None else t1_games.head(per_team).copy()
        t2_period_df = t2_games.copy() if per_team is None else t2_games.head(per_team).copy()
        t1_period = pd.to_numeric(t1_period_df.get("result"), errors="coerce")
        t2_period = pd.to_numeric(t2_period_df.get("result"), errors="coerce")
        n1 = int(t1_period.notna().sum())
        n2 = int(t2_period.notna().sum())
        w1 = int((t1_period > 0.5).sum())
        w2 = int((t2_period > 0.5).sum())
        out["periods"].setdefault(label, {})["league_t1"] = fmt_hist_prob(w1, n1, 0, laplace)
        out["periods"].setdefault(label, {})["league_t2"] = fmt_hist_prob(w2, n2, 0, laplace)
        out["periods"].setdefault(label, {})["league_n_t1"] = n1
        out["periods"].setdefault(label, {})["league_n_t2"] = n2
    for label, per_team in PERIOD_SPECS:
        df_period = h2h.copy() if per_team is None else h2h.head(per_team).copy()
        period_res = pd.to_numeric(df_period["result_t1"] if "result_t1" in df_period.columns else pd.Series(dtype=float), errors="coerce")
        n = int(period_res.notna().sum())
        w1 = int((period_res > 0.5).sum())
        w2 = max(0, n - w1)
        out["periods"].setdefault(label, {})["h2h_t1"] = fmt_hist_prob(w1, n, 0, laplace)
        out["periods"].setdefault(label, {})["h2h_t2"] = fmt_hist_prob(w2, n, 0, laplace)
        out["periods"].setdefault(label, {})["h2h_n"] = n
    return out


def build_summary_rows(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str, lines: LinesConfig, laplace: bool = True) -> pd.DataFrame:
    rows: List[dict] = []
    winner = analyze_winner_market(dataset, year, league, team1, team2, laplace)
    winner_year = winner["periods"].get("Ano todo", {})
    for team_label, model_prob_key, league_key, h2h_key, league_n_key in [
        (team1, "model_t1_prob", "league_t1", "h2h_t1", "league_n_t1"),
        (team2, "model_t2_prob", "league_t2", "h2h_t2", "league_n_t2"),
    ]:
        model_prob = float(winner[model_prob_key])
        ml_prob_key = "ml_t1_prob" if team_label == team1 else "ml_t2_prob"
        ml_prob = winner.get(ml_prob_key)
        league_hist = str(winner_year.get(league_key, "-"))
        h2h_hist = str(winner_year.get(h2h_key, "-"))
        hist_prob = _blend_hist_prob(
            league_hist,
            h2h_hist,
            league_n=int(winner_year.get(league_n_key, 0)),
            h2h_n=int(winner_year.get("h2h_n", 0)),
        )
        rows.append(
            {
                "Mercado": "ML (Partida)",
                "Lado": team_label,
                "Linha": "-",
                "Confianca (Score)": _confidence_score(
                    int(winner_year.get(league_n_key, 0)),
                    int(winner_year.get("h2h_n", 0)),
                    model_prob,
                    hist_prob,
                ),
                "Laplace (Liga)": league_hist,
                "Laplace (H2H)": h2h_hist,
                "Odd (Modelo)": _fmt_odd(model_prob),
                "Odd (ML)": _fmt_odd(float(ml_prob)) if ml_prob is not None else "-",
                "Sinal ML": str(winner.get("ml_signal_label", "-")),
                "Fonte": str(winner.get("model_source", "Heuristico")),
                "Stats": (
                    f"Forma {team1}: {winner['team1_win_rate'] * 100:.1f}% | {team2}: {winner['team2_win_rate'] * 100:.1f}%"
                    + (
                        f" | ML {team_label}: {float(ml_prob) * 100:.1f}%"
                        if ml_prob is not None and math.isfinite(float(ml_prob))
                        else ""
                    )
                ),
                "Odd (Hist)": _fmt_odd(hist_prob),
                "Odd (Mix)": _fmt_odd(_mix_model_hist_prob(model_prob, hist_prob)),
                "Media": f"WR {team_label}: {model_prob * 100:.1f}%",
            }
        )
    for market_label, values in [
        ("Kills", lines.total_kills),
        ("Torres", lines.total_towers),
        ("Roshans", lines.total_roshans),
        ("Barracks", lines.total_barracks),
        ("Tempo", lines.total_time_min),
    ]:
        for line in values:
            result = analyze_total_market(dataset, year, league, team1, team2, market_label, line, laplace)
            year_period = result["periods"].get("Ano todo", {})
            mean_display = fmt_mmss_from_minutes(result["expected_mean"]) if market_label == "Tempo" else f"{result['expected_mean']:.2f}"
            hist_over = str(year_period.get("league_over", "-"))
            hist_under = str(year_period.get("league_under", "-"))
            h2h_over = str(year_period.get("h2h_over", "-"))
            h2h_under = str(year_period.get("h2h_under", "-"))
            hist_over_prob = _blend_hist_prob(
                hist_over,
                h2h_over,
                league_n=int(year_period.get("league_n", 0)),
                h2h_n=int(year_period.get("h2h_n", 0)),
            )
            hist_under_prob = _blend_hist_prob(
                hist_under,
                h2h_under,
                league_n=int(year_period.get("league_n", 0)),
                h2h_n=int(year_period.get("h2h_n", 0)),
            )
            rows.append({
                "Mercado": market_label,
                "Lado": "Acima",
                "Linha": fmt_mmss_from_minutes(line) if market_label == "Tempo" else f"{line:g}",
                "Confianca (Score)": _confidence_score(
                    int(year_period.get("league_n", 0)),
                    int(year_period.get("h2h_n", 0)),
                    float(result["model_over_prob"]),
                    hist_over_prob,
                ),
                "Laplace (Liga)": hist_over,
                "Laplace (H2H)": h2h_over,
                "Odd (Modelo)": _fmt_odd(float(result["model_over_prob"])),
                "Odd (ML)": "-",
                "Sinal ML": "-",
                "Fonte": "Motor Estatistico",
                "Stats": f"Media {mean_display}",
                "Odd (Hist)": _fmt_odd(hist_over_prob),
                "Odd (Mix)": _fmt_odd(_mix_model_hist_prob(float(result["model_over_prob"]), hist_over_prob)),
                "Media": mean_display,
            })
            rows.append({
                "Mercado": market_label,
                "Lado": "Abaixo",
                "Linha": fmt_mmss_from_minutes(line) if market_label == "Tempo" else f"{line:g}",
                "Confianca (Score)": _confidence_score(
                    int(year_period.get("league_n", 0)),
                    int(year_period.get("h2h_n", 0)),
                    float(result["model_under_prob"]),
                    hist_under_prob,
                ),
                "Laplace (Liga)": hist_under,
                "Laplace (H2H)": h2h_under,
                "Odd (Modelo)": _fmt_odd(float(result["model_under_prob"])),
                "Odd (ML)": "-",
                "Sinal ML": "-",
                "Fonte": "Motor Estatistico",
                "Stats": f"Media {mean_display}",
                "Odd (Hist)": _fmt_odd(hist_under_prob),
                "Odd (Mix)": _fmt_odd(_mix_model_hist_prob(float(result["model_under_prob"]), hist_under_prob)),
                "Media": mean_display,
            })
    for market_label, values in [
        ("Kills", lines.hc_kills_t1),
        ("Torres", lines.hc_towers_t1),
        ("Roshans", lines.hc_roshans_t1),
        ("Barracks", lines.hc_barracks_t1),
    ]:
        for line in values:
            result = analyze_handicap_market(dataset, year, league, team1, team2, market_label, line, laplace)
            year_period = result["periods"].get("Ano todo", {})
            hist_t1 = str(year_period.get("t1_cover", "-"))
            hist_t2 = str(year_period.get("t2_cover", "-"))
            hist_t1_prob = _hist_prob_from_text(hist_t1)
            hist_t2_prob = _hist_prob_from_text(hist_t2)
            rows.append({
                "Mercado": f"HC {market_label}",
                "Lado": team1,
                "Linha": f"{line:+g}",
                "Confianca (Score)": _confidence_score(
                    0,
                    int(year_period.get("h2h_n", 0)),
                    float(result["model_t1_prob"]),
                    hist_t1_prob,
                ),
                "Laplace (Liga)": "-",
                "Laplace (H2H)": hist_t1,
                "Odd (Modelo)": _fmt_odd(float(result["model_t1_prob"])),
                "Odd (ML)": "-",
                "Sinal ML": "-",
                "Fonte": "Motor Estatistico",
                "Stats": f"Edge {result['expected_edge']:.2f}",
                "Odd (Hist)": _fmt_odd(hist_t1_prob),
                "Odd (Mix)": _fmt_odd(_mix_model_hist_prob(float(result["model_t1_prob"]), hist_t1_prob)),
                "Media": f"{result['expected_edge']:.2f}",
            })
            rows.append({
                "Mercado": f"HC {market_label}",
                "Lado": team2,
                "Linha": f"{-line:+g}",
                "Confianca (Score)": _confidence_score(
                    0,
                    int(year_period.get("h2h_n", 0)),
                    float(result["model_t2_prob"]),
                    hist_t2_prob,
                ),
                "Laplace (Liga)": "-",
                "Laplace (H2H)": hist_t2,
                "Odd (Modelo)": _fmt_odd(float(result["model_t2_prob"])),
                "Odd (ML)": "-",
                "Sinal ML": "-",
                "Fonte": "Motor Estatistico",
                "Stats": f"Edge {result['expected_edge']:.2f}",
                "Odd (Hist)": _fmt_odd(hist_t2_prob),
                "Odd (Mix)": _fmt_odd(_mix_model_hist_prob(float(result["model_t2_prob"]), hist_t2_prob)),
                "Media": f"{result['expected_edge']:.2f}",
            })
    return pd.DataFrame(rows)


def build_stats_table(dataset: DotaDataset, year: Optional[int], championship: Optional[str], side_filter: str = "All", result_filter: str = "All", custom_market: str = "Kills", custom_line: Optional[float] = None) -> pd.DataFrame:
    scope = dataset.filter_scope(year, championship or "(Todas)")
    if side_filter != "All":
        scope = scope[scope["side"] == side_filter.lower()].copy()
    if result_filter == "Win":
        scope = scope[scope["result"] == 1].copy()
    elif result_filter == "Loss":
        scope = scope[scope["result"] == 0].copy()
    if scope.empty:
        return pd.DataFrame()

    custom_col = TOTAL_MARKETS.get(custom_market, "totalkills")
    line = float(custom_line) if custom_line is not None else np.nan

    rows = []
    for _, group in scope.groupby(["teamname_norm", "teamname"], dropna=False):
        team = str(group["teamname"].mode().iloc[0])
        games = int(len(group))
        kills = pd.to_numeric(group["teamkills"], errors="coerce")
        deaths = pd.to_numeric(group["oppkills"], errors="coerce")
        towers = pd.to_numeric(group["towers"], errors="coerce")
        roshans = pd.to_numeric(group["roshans"], errors="coerce")
        barracks = pd.to_numeric(group["barracks"], errors="coerce")
        duration = pd.to_numeric(group["duration"], errors="coerce")
        totals = {
            "Total Kills": kills + deaths,
            "Total Towers": towers + pd.to_numeric(group["opp_towers"], errors="coerce"),
            "Total Roshans": roshans + pd.to_numeric(group["opp_roshans"], errors="coerce"),
            "Total Barracks": barracks + pd.to_numeric(group["opp_barracks"], errors="coerce"),
        }
        row = {
            "Team": team,
            "# Games": games,
            "Win Rate": float(group["result"].mean() * 100.0),
            "Game Time": float(duration.mean()) if duration.notna().any() else np.nan,
            "Kills": float(kills.mean()) if kills.notna().any() else np.nan,
            "Deaths": float(deaths.mean()) if deaths.notna().any() else np.nan,
            "Towers": float(towers.mean()) if towers.notna().any() else np.nan,
            "Roshans": float(roshans.mean()) if roshans.notna().any() else np.nan,
            "Barracks": float(barracks.mean()) if barracks.notna().any() else np.nan,
            "C. Kills": float((kills - deaths).mean()) if kills.notna().any() else np.nan,
            "C. Towers": float((towers - pd.to_numeric(group["opp_towers"], errors="coerce")).mean()),
            "C. Roshans": float((roshans - pd.to_numeric(group["opp_roshans"], errors="coerce")).mean()),
            "C. Barracks": float((barracks - pd.to_numeric(group["opp_barracks"], errors="coerce")).mean()),
            "% TK > 45.5": float((totals["Total Kills"] > 45.5).mean() * 100.0),
            "% TK > 50.5": float((totals["Total Kills"] > 50.5).mean() * 100.0),
            "% TK > 55.5": float((totals["Total Kills"] > 55.5).mean() * 100.0),
            "% TT > 15.5": float((totals["Total Towers"] > 15.5).mean() * 100.0),
            "% TR > 1.5": float((totals["Total Roshans"] > 1.5).mean() * 100.0),
            "% TB > 3.5": float((totals["Total Barracks"] > 3.5).mean() * 100.0),
            "% Time > 40.5": float(((duration / 60.0) > 40.5).mean() * 100.0),
            "% Time > 45.5": float(((duration / 60.0) > 45.5).mean() * 100.0),
        }
        if math.isfinite(line):
            row[f"% {custom_market} > {line:g}"] = float((pd.to_numeric(group[custom_col], errors="coerce") > line).mean() * 100.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Win Rate", "# Games", "Team"], ascending=[False, False, True]).reset_index(drop=True)


def detail_tables(dataset: DotaDataset, year: Optional[int], league: str, team1: str, team2: str) -> dict:
    return {
        "team1": dataset.team_games_all_leagues(year, team1),
        "team2": dataset.team_games_all_leagues(year, team2),
        "h2h": dataset.pair_games(year, "(Todas)", team1, team2),
    }
