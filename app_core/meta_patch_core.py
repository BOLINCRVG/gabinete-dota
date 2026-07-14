"""Motor da aba META — valor dos campeões por patch, medido no comportamento
dos profissionais (pedido 2026-07-05).

Sinais, do mais rápido ao mais lento:
- PRESENÇA = pick% + ban% do patch (bans revelam valor em dias, sem esperar WR);
- PRIORIDADE = % dos picks na 1ª rotação (pick1-2) e dos bans na 1ª fase (ban1-3);
- WR shrinkado (k=20 pra 50%) — complemento fraco no pro play (time bom enviesa).

Tier por presença: S ≥60 · A ≥35 · B ≥15 · C ≥5 · D <5.
A aba é a fonte; o Radar CONSOME daqui (fator meta do veredito + badge no grid).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

_PICK_COLS = ["pick1", "pick2", "pick3", "pick4", "pick5"]
_BAN_COLS = ["ban1", "ban2", "ban3", "ban4", "ban5"]
_USECOLS = ["gameid", "date", "patch", "league", "year", "participantid",
            "teamname", "result", *_BAN_COLS, *_PICK_COLS]
WR_SHRINK_K = 20.0


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def load_draft_rows(csv_path: str | Path) -> pd.DataFrame:
    """Linhas de TIME (participantid 100/200) com picks/bans/patch."""
    preview = pd.read_csv(csv_path, nrows=1, low_memory=False)
    usecols = [c for c in _USECOLS if c in set(preview.columns)]
    df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    if "participantid" in df.columns:
        df = df[df["participantid"].isin([100, 200])].copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df["patch"] = df.get("patch").astype(str).str.strip()
    return df.drop_duplicates(subset=["gameid", "teamname"], keep="last")


def _patch_sort_key(p: str) -> tuple:
    try:
        parts = [int(x) for x in str(p).split(".")[:2]]
        return (parts[0], parts[1] if len(parts) > 1 else 0)
    except (TypeError, ValueError):
        return (-1, -1)


def patches_by_recency(df: pd.DataFrame, min_maps: int = 30) -> list[tuple[str, int]]:
    """[(patch, n_mapas)] do mais recente pro mais antigo (só patches com jogo)."""
    if df is None or df.empty or "patch" not in df.columns:
        return []
    counts = df.groupby("patch")["gameid"].nunique()
    out = [(str(p), int(n)) for p, n in counts.items()
           if str(p) not in ("", "nan", "None") and int(n) >= int(min_maps)]
    out.sort(key=lambda t: _patch_sort_key(t[0]), reverse=True)
    return out


def tier_for_presence(presence: float) -> str:
    if presence >= 60:
        return "S"
    if presence >= 35:
        return "A"
    if presence >= 15:
        return "B"
    if presence >= 5:
        return "C"
    return "D"


TIER_COLORS = {"S": "#f59e0b", "A": "#22c55e", "B": "#3b82f6", "C": "#9ca3af", "D": "#6b7280"}


def compute_meta_table(
    df: pd.DataFrame,
    *,
    patch: str,
    league: Optional[str] = None,
    prev_patch: Optional[str] = None,
) -> pd.DataFrame:
    """Tabela por campeão no patch: presença, pick/ban%, prioridade, WR, tier,
    delta de presença vs patch anterior. Nome do campeão = como está no CSV."""
    if df is None or df.empty:
        return pd.DataFrame()
    scope = df[df["patch"].astype(str) == str(patch)]
    if league and league != "Todas" and "league" in scope.columns:
        scope = scope[scope["league"].astype(str) == str(league)]
    n_maps = int(scope["gameid"].nunique())
    if n_maps == 0:
        return pd.DataFrame()

    rows: dict[str, dict[str, float]] = {}

    def _bump(champ: Any, field: str, amt: float = 1.0) -> None:
        name = str(champ or "").strip()
        if not name or name.lower() in ("nan", "none"):
            return
        rec = rows.setdefault(name, {"picks": 0, "bans": 0, "early_picks": 0,
                                     "early_bans": 0, "wins": 0})
        rec[field] += amt

    for r in scope.itertuples(index=False):
        win = 1.0 if float(getattr(r, "result", 0) or 0) == 1.0 else 0.0
        for i, c in enumerate(_PICK_COLS):
            champ = getattr(r, c, None)
            _bump(champ, "picks")
            if i < 2:
                _bump(champ, "early_picks")
            _bump(champ, "wins", win)
        for i, c in enumerate(_BAN_COLS):
            champ = getattr(r, c, None)
            _bump(champ, "bans")
            if i < 3:
                _bump(champ, "early_bans")

    prev_presence: dict[str, float] = {}
    if prev_patch:
        prev_tbl = compute_meta_table(df, patch=str(prev_patch), league=league, prev_patch=None)
        if not prev_tbl.empty:
            prev_presence = dict(zip(prev_tbl["champion"], prev_tbl["presence"]))

    out = []
    for name, rec in rows.items():
        picks, bans = rec["picks"], rec["bans"]
        pick_pct = 100.0 * picks / n_maps
        ban_pct = 100.0 * bans / n_maps
        presence = pick_pct + ban_pct
        wr = 100.0 * (rec["wins"] + 0.5 * WR_SHRINK_K) / (picks + WR_SHRINK_K) if picks else float("nan")
        out.append({
            "champion": name,
            "n_maps": n_maps,
            "picks": int(picks), "bans": int(bans),
            "pick_pct": pick_pct, "ban_pct": ban_pct,
            "presence": presence,
            "early_pick_share": (100.0 * rec["early_picks"] / picks) if picks else 0.0,
            "early_ban_share": (100.0 * rec["early_bans"] / bans) if bans else 0.0,
            "wr_shrunk": wr,
            "tier": tier_for_presence(presence),
            "presence_prev": prev_presence.get(name, float("nan")),
            "presence_delta": presence - prev_presence[name] if name in prev_presence else float("nan"),
        })
    tbl = pd.DataFrame(out)
    return tbl.sort_values(["presence", "pick_pct"], ascending=False).reset_index(drop=True)


def meta_lookup_for_radar(
    df: pd.DataFrame,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """{nome canônico do registro -> {tier, presence, delta}} do patch mais
    recente com amostra (fallback: soma dos 2 últimos). Consumido pelo Radar."""
    patches = patches_by_recency(df, min_maps=150)
    if not patches:
        patches = patches_by_recency(df, min_maps=30)
    if not patches:
        return {}
    cur = patches[0][0]
    prev = patches[1][0] if len(patches) > 1 else None
    tbl = compute_meta_table(df, patch=cur, prev_patch=prev)
    if tbl.empty:
        return {}
    aliases = (registry or {}).get("aliases") or {}
    champs = (registry or {}).get("champions") or {}
    out: dict[str, dict[str, Any]] = {}
    for r in tbl.itertuples(index=False):
        k = _norm_key(r.champion)
        entry = champs.get(str(aliases.get(k) or k))
        canon = str(entry.get("name")) if isinstance(entry, Mapping) and entry.get("name") else str(r.champion)
        out[canon] = {"tier": str(r.tier), "presence": float(r.presence),
                      "delta": float(r.presence_delta) if pd.notna(r.presence_delta) else None,
                      "patch": cur}
    return out
