"""Motor do Radar do Confronto (página "Radar", design fechado 2026-07-04).

Régua do sinal (B2, decidida com o usuário):
- Janelas: Liga (ano, liga própria), Forma (L10/L5) e H2H (ano).
- Sinal = recorte batendo >=70% (ou <=30%, que é 70% do outro lado) com n>=5.
- CONFLITO (destaque máximo) = H2H e Liga apontando >=70% para lados opostos
  (H2H com n>=3 entra em conflito; com n<5 nunca vira sinal SOZINHO).
- Máximo ~5 sinais por time, mais fortes primeiro; frase seca com os números.
- Mercados: ML (vencedor do mapa), kills, torres, dragões e tempo — barões e
  inibidores ficam FORA (usuário só aposta esses no pré, nunca com draft em mãos).

Sem veredito automático: o radar lista fatos, o usuário julga (padrão HLTV).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import pandas as pd

SIGNAL_PCT = 70.0       # >=70% cobre / <=30% (= 70% do lado oposto)
SIGNAL_MIN_N = 5        # n mínimo pra virar sinal sozinho
CONFLICT_H2H_MIN_N = 3  # H2H com n>=3 pode compor CONFLITO (nunca sinal sozinho com n<5)
MAX_SIGNALS_PER_TEAM = 5

# Mercados do radar (decisão B2). "col" é a coluna do team_games (pós-augment).
RADAR_MARKETS: list[dict[str, Any]] = [
    {"key": "ml",      "label": "Vencedor",  "col": "win",               "line": None,  "decimals": 0},
    {"key": "kills",   "label": "Kills",     "col": "mv2_total_kills",   "line": 27.5,  "decimals": 1},
    {"key": "towers",  "label": "Torres",    "col": "mv2_total_towers",  "line": 11.5,  "decimals": 1},
    {"key": "dragons", "label": "Dragões",   "col": "mv2_total_dragons", "line": 4.5,   "decimals": 1},
    {"key": "time",    "label": "Tempo",     "col": "game_time_min",     "line": 32.5,  "decimals": 1},
]

WINDOW_LABELS = {"liga": "na liga", "ano": "no ano (todas as ligas)",
                 "l10": "nos últimos 10", "l5": "nos últimos 5", "h2h": "no H2H"}


@dataclass
class RadarSignal:
    team: str
    market_key: str
    market_label: str
    window: str          # liga | l10 | l5 | h2h
    direction: str       # "over" | "under" | "win" | "loss"
    hits: int
    n: int
    pct: float           # % do lado reportado (sempre >= 50)
    text: str
    kind: str = "sinal"  # "sinal" | "conflito"
    conflict_with: Optional["RadarSignal"] = field(default=None, repr=False)

    @property
    def strength(self) -> float:
        """Ordenação: extremidade da taxa pesada pelo tamanho da amostra."""
        return abs(self.pct - 50.0) * math.log1p(self.n)


def _ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
    """Garante game_time_min (gamelength está em segundos no OracleElixir)."""
    if df is None or df.empty:
        return df
    if "game_time_min" not in df.columns and "gamelength" in df.columns:
        out = df.copy()
        out["game_time_min"] = pd.to_numeric(out["gamelength"], errors="coerce") / 60.0
        return out
    return df


def _sorted_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "date" not in df.columns:
        return df
    out = df.copy()
    out["_dt"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("_dt").drop(columns=["_dt"])


def build_team_windows(
    df_team_year: pd.DataFrame,
    opponent: str,
) -> dict[str, pd.DataFrame]:
    """Janelas por time a partir dos jogos do ANO (liga própria já filtrada
    pelo chamador quando fizer sentido). L10/L5 = recorte por data."""
    base = _ensure_time_col(_sorted_by_date(df_team_year))
    if base is None or base.empty:
        empty = pd.DataFrame()
        return {"liga": empty, "l10": empty, "l5": empty, "h2h": empty}
    h2h = base[base["opponent"].astype(str) == str(opponent)] if "opponent" in base.columns else pd.DataFrame()
    return {"liga": base, "l10": base.tail(10), "l5": base.tail(5), "h2h": h2h}


def window_rate(df: pd.DataFrame, col: str, line: Optional[float]) -> tuple[int, int, float]:
    """(hits, n, pct) — ML: win==1; totais: valor > linha (taxa de Over)."""
    if df is None or df.empty or col not in df.columns:
        return 0, 0, float("nan")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return 0, 0, float("nan")
    hits = int((s > 0.5).sum()) if line is None else int((s > float(line)).sum())
    return hits, n, 100.0 * hits / n


def _fmt_line(line: float, market_key: str) -> str:
    txt = f"{float(line):g}"
    return f"{txt} min" if market_key == "time" else txt


def _signal_text(
    *, team: str, market: Mapping[str, Any], window: str,
    direction: str, hits: int, n: int, line: Optional[float],
) -> str:
    wl = WINDOW_LABELS.get(window, window)
    if market["key"] == "ml":
        verb = "venceu" if direction == "win" else "perdeu"
        return f"{team} {verb} {hits}/{n} {wl}"
    side = "Over" if direction == "over" else "Under"
    shown = hits if direction == "over" else (n - hits)
    line_txt = _fmt_line(float(line), str(market["key"])) if line is not None else "?"
    return f"{side} {line_txt} {market['label'].lower()}: {shown}/{n} nos jogos de {team} {wl}"


def _classify(hits: int, n: int, pct: float) -> Optional[str]:
    """Retorna 'high' (>=70% pro lado do hit) ou 'low' (<=30%), senão None."""
    if n <= 0 or not math.isfinite(pct):
        return None
    if pct >= SIGNAL_PCT:
        return "high"
    if pct <= 100.0 - SIGNAL_PCT:
        return "low"
    return None


def build_radar_signals(
    *,
    team: str,
    windows: Mapping[str, pd.DataFrame],
    lines: Optional[Mapping[str, float]] = None,
    markets: Optional[list[dict[str, Any]]] = None,
) -> list[RadarSignal]:
    """Sinais de UM time pela régua B2. `lines` sobrepõe as linhas padrão
    (ex.: linha da casa digitada). Conflitos: par Liga×H2H em lados opostos."""
    out: list[RadarSignal] = []
    for market in (markets if markets is not None else RADAR_MARKETS):
        key = str(market["key"])
        col = str(market["col"])
        line = None if market["line"] is None else float((lines or {}).get(key, market["line"]))
        rates: dict[str, tuple[int, int, float]] = {
            w: window_rate(df, col, line) for w, df in windows.items()
        }

        # --- conflito Liga × H2H (a leitura mais valiosa pro usuário) ---
        # Em cross-league a janela "liga" chega como "ano" (regra do usuário:
        # ano inteiro, todas as ligas) — o conflito usa a que existir.
        _base_key = "liga" if "liga" in rates else ("ano" if "ano" in rates else "liga")
        liga_side = _classify(*rates.get(_base_key, (0, 0, float("nan"))))
        h2h_hits, h2h_n, h2h_pct = rates.get("h2h", (0, 0, float("nan")))
        h2h_side = _classify(h2h_hits, h2h_n, h2h_pct) if h2h_n >= CONFLICT_H2H_MIN_N else None
        conflict_windows: set[str] = set()
        if liga_side and h2h_side and liga_side != h2h_side and _base_key in rates:
            sig_liga = _mk_signal(team, market, _base_key, line, rates[_base_key])
            sig_h2h = _mk_signal(team, market, "h2h", line, rates["h2h"])
            if sig_liga and sig_h2h:
                sig_h2h.kind = "conflito"
                sig_h2h.conflict_with = sig_liga
                out.append(sig_h2h)
                conflict_windows.update({_base_key, "h2h"})

        # --- sinais simples por janela ---
        for w, (hits, n, pct) in rates.items():
            if w in conflict_windows:
                continue
            if n < SIGNAL_MIN_N:
                continue  # H2H n<5 nunca vira sinal sozinho (só em conflito, acima)
            if _classify(hits, n, pct) is None:
                continue
            sig = _mk_signal(team, market, w, line, (hits, n, pct))
            if sig is not None:
                out.append(sig)

    # conflitos primeiro, depois os mais fortes; corta no máximo da régua
    out.sort(key=lambda s: (s.kind != "conflito", -s.strength))
    return out[:MAX_SIGNALS_PER_TEAM]


def _mk_signal(
    team: str, market: Mapping[str, Any], window: str,
    line: Optional[float], rate: tuple[int, int, float],
) -> Optional[RadarSignal]:
    hits, n, pct = rate
    side = _classify(hits, n, pct)
    if side is None:
        return None
    if market["key"] == "ml":
        direction = "win" if side == "high" else "loss"
    else:
        direction = "over" if side == "high" else "under"
    pct_shown = pct if side == "high" else 100.0 - pct
    return RadarSignal(
        team=str(team),
        market_key=str(market["key"]),
        market_label=str(market["label"]),
        window=str(window),
        direction=direction,
        hits=int(hits),
        n=int(n),
        pct=float(pct_shown),
        text=_signal_text(team=str(team), market=market, window=str(window),
                          direction=direction, hits=int(hits), n=int(n), line=line),
    )


# ============================================================================
# Padrões por número do mapa (estudo tools/eval_fearless_map_totals.py,
# 2026-07-04, gate ACCEPT). Dragões mapa-2 REJECT em 2026 (z=1.6) — não nasce.
# Kills/tempo mapa-2 são efeito GERAL (replicam em 2024 pré-fearless);
# torres mapa-3+ é da era fearless 2026.
# ============================================================================

# Textos enxutos na UI (proveniência estatística fica aqui no código: kills
# mapa-2 z 5.4 e tempo z 2.1-3.2 replicam em 2024; torres 3+ z 3.2 é da era
# fearless 2026; dragões mapa-2 REJECT z 1.6).
MAP_PATTERN_SIGNALS: dict[str, list[str]] = {
    "2": [
        "Mapa 2 tende a MAIS kills: over 27.5 bate 71% no mapa 2 vs 63% no mapa 1",
        "Mapa 2 tende a jogo MAIS CURTO: over 32.5min cai de 41% para 37%",
    ],
    "3+": [
        "Mapa 3+ tende a MAIS torres: over 11.5 em 56% vs 50% no mapa 1",
    ],
}


def map_pattern_signals(map_no: int) -> list[str]:
    if int(map_no) == 2:
        return list(MAP_PATTERN_SIGNALS["2"])
    if int(map_no) >= 3:
        return list(MAP_PATTERN_SIGNALS["3+"])
    return []


# ============================================================================
# Ajuste de TORRES pelo draft — ÚNICO mercado que passou o gate cego
# (tools/eval_draft_totals_adjust.py, fit 2025 / eval 2026: Brier +0.0006,
# t=8.3; kills/dragões neutros e tempo REJECT — NÃO deployar esses).
# Interação time×draft testada e NÃO significativa: a tendência do time já
# está na média do matchup; o draft empurra por cima. delta = a + b·z(sinal).
# ============================================================================

DRAFT_TOWERS_ADJUST = {"a": -0.1699, "b": 0.1823, "sig_mu": 13.3144, "sig_sd": 1.3727}


def towers_draft_signal(macro_a: Mapping[str, Any], macro_b: Mapping[str, Any]) -> float:
    """Sinal de torres das duas comps (mesma fórmula do comp_compare)."""
    def s(k: str) -> float:
        try:
            return float(macro_a.get(k, 0) or 0) + float(macro_b.get(k, 0) or 0)
        except (TypeError, ValueError):
            return 0.0
    return 1.45 * s("side_pressure") + 0.25 * s("siege_pressure") + 0.15 * s("objective_setup")


def towers_draft_delta(signal: float) -> float:
    """Deslocamento esperado do TOTAL de torres dado o draft (torres a mais/menos)."""
    c = DRAFT_TOWERS_ADJUST
    z = (float(signal) - c["sig_mu"]) / (c["sig_sd"] or 1.0)
    return float(c["a"] + c["b"] * z)


# ============================================================================
# Arquétipos (taxonomia iTero, decidida no B3: Poke / Engage / Scaling / Split-push)
# ============================================================================

ARCHETYPES = ("Poke", "Engage", "Scaling", "Split-push")

# Normas (média, desvio) dos componentes medidas em 4000 drafts pro de 2026
# (scratch calib 2026-07-04). Sem o z-score as escalas são incomparáveis e
# ~tudo caía em Scaling/Engage; com elas a distribuição fica 30/27/21/21.
# Re-medir se o registro de campeões ou o meta mudarem muito.
_ARCH_NORMS: dict[str, tuple[float, float]] = {
    "poke": (3.40, 1.34), "siege": (2.94, 0.99), "engage": (5.05, 1.72),
    "dive": (3.98, 1.41), "pick": (5.11, 1.89), "splitpush": (0.80, 0.52),
    "skirmish": (3.49, 0.81), "early": (4.40, 1.67), "late": (9.30, 1.87),
    "carry_scaling": (15.45, 2.40), "side_pressure": (2.57, 0.62),
    "siege_pressure": (8.36, 1.91), "dive_threat": (10.46, 1.94),
}


def classify_comp_archetype(profile: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Mapeia o comp_profile (comp_archetype_core.build_comp_profile) para o
    arquétipo iTero dominante, com componentes normalizados (z-score) pelas
    normas de drafts pro. Retorna (label, scores) para a UI mostrar o quão
    claro é o enquadramento."""
    styles = profile.get("style_scores") or {}
    macro = profile.get("macro_scores") or {}
    windows = profile.get("window_scores") or {}

    def z(m: Mapping[str, Any], k: str) -> float:
        mean, std = _ARCH_NORMS.get(k, (0.0, 1.0))
        try:
            v = float(m.get(k, 0.0) or 0.0)
            if not math.isfinite(v):
                return 0.0
        except (TypeError, ValueError):
            return 0.0
        return (v - mean) / (std if std > 1e-9 else 1.0)

    scores = {
        "Poke": z(styles, "poke") + 0.5 * z(styles, "siege") + 0.3 * z(macro, "siege_pressure"),
        "Engage": z(styles, "engage") + 0.6 * z(styles, "dive") + 0.4 * z(styles, "pick")
                  + 0.3 * z(macro, "dive_threat"),
        "Scaling": z(windows, "late") + 0.6 * z(macro, "carry_scaling") - 0.5 * z(windows, "early"),
        "Split-push": z(styles, "splitpush") + 0.6 * z(macro, "side_pressure")
                      + 0.2 * z(styles, "skirmish"),
    }
    label = max(scores, key=lambda k: scores[k])
    return label, {k: round(v, 2) for k, v in scores.items()}


ROLE_ORDER = ("top", "jng", "mid", "bot", "sup")
_POS_ALIASES = {"top": "top", "jng": "jng", "jungle": "jng", "mid": "mid", "middle": "mid",
                "bot": "bot", "adc": "bot", "bottom": "bot", "sup": "sup", "support": "sup"}


def norm_position(value: Any) -> str:
    return _POS_ALIASES.get(str(value or "").strip().lower(), "")


def team_style_history(
    *,
    players_df: pd.DataFrame,
    team: str,
    registry: Mapping[str, Any] | None,
    year: Optional[int] = None,
) -> dict[str, dict[str, int]]:
    """Fit time×estilo: classifica cada jogo histórico do time pelo arquétipo
    da comp que ELE jogou e devolve {arquétipo: {"wins": w, "n": n}}.
    Sempre mostrar com o n na cara (régua B3: sinal só >=70% com n>=5)."""
    from app_core.comp_archetype_core import build_comp_profile

    out: dict[str, dict[str, int]] = {a: {"wins": 0, "n": 0} for a in ARCHETYPES}
    if players_df is None or players_df.empty:
        return out
    df = players_df
    if "teamname" not in df.columns or "champion" not in df.columns:
        return out
    df = df[df["teamname"].astype(str) == str(team)]
    if year is not None and "year" in df.columns:
        df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)]
    if df.empty:
        return out
    for gameid, grp in df.groupby("gameid"):
        pos_map: dict[str, str] = {}
        for row in grp.itertuples(index=False):
            pos = norm_position(getattr(row, "position", ""))
            champ = str(getattr(row, "champion", "") or "").strip()
            if pos and champ and pos not in pos_map:
                pos_map[pos] = champ
        if len(pos_map) < 5:
            continue
        picks = [pos_map[r] for r in ROLE_ORDER]
        profile = build_comp_profile(registry=registry, picks=picks, role_order=list(ROLE_ORDER))
        label, _ = classify_comp_archetype(profile)
        try:
            win = int(pd.to_numeric(grp["result"], errors="coerce").fillna(0).max())
        except (KeyError, TypeError, ValueError):
            win = 0
        out[label]["n"] += 1
        out[label]["wins"] += 1 if win == 1 else 0
    return out


def player_champion_familiarity(
    *,
    players_df: pd.DataFrame,
    player: str,
    champion: str,
) -> int:
    """Jogos de palco do player no campeão (todas as seasons carregadas).
    iTero: 0→14 jogos importa, depois satura."""
    if players_df is None or players_df.empty or not player or not champion:
        return 0
    df = players_df
    if "playername" not in df.columns or "champion" not in df.columns:
        return 0
    mask = (df["playername"].astype(str).str.strip().str.casefold() == str(player).strip().casefold()) & \
           (df["champion"].astype(str).str.strip().str.casefold() == str(champion).strip().casefold())
    return int(mask.sum())


def champion_meta_presence(
    *,
    players_df_scope: pd.DataFrame,
    champion: str,
) -> tuple[float, int]:
    """(presença %, jogos do recorte): % dos jogos do recorte (ex.: liga+ano)
    em que o campeão foi ESCOLHIDO. Proxy de força meta (sem dado de tier)."""
    if players_df_scope is None or players_df_scope.empty or not champion:
        return float("nan"), 0
    df = players_df_scope
    if "gameid" not in df.columns or "champion" not in df.columns:
        return float("nan"), 0
    total_games = int(df["gameid"].nunique())
    if total_games == 0:
        return float("nan"), 0
    picked = int(df[df["champion"].astype(str).str.strip().str.casefold()
                    == str(champion).strip().casefold()]["gameid"].nunique())
    return 100.0 * picked / total_games, total_games


def latest_roster(
    *,
    players_df: pd.DataFrame,
    team: str,
) -> dict[str, str]:
    """{posição: player} do jogo mais recente do time (pré-preenche o montador)."""
    out: dict[str, str] = {}
    if players_df is None or players_df.empty or "teamname" not in players_df.columns:
        return out
    df = players_df[players_df["teamname"].astype(str) == str(team)].copy()
    if df.empty or "date" not in df.columns:
        return out
    df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    last_game = df.sort_values("_dt")["gameid"].iloc[-1]
    for row in df[df["gameid"] == last_game].itertuples(index=False):
        pos = norm_position(getattr(row, "position", ""))
        name = str(getattr(row, "playername", "") or "").strip()
        if pos and name:
            out[pos] = name
    return out
