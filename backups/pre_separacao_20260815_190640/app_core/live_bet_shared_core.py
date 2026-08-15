from __future__ import annotations

import math
import re
from html import escape
from typing import Any, Optional

import pandas as pd

from mlcore.utils import safe_float_or_none as _safe_float


LIVE_MARKET_LABELS = ["Kills", "Torres", "DragÃµes", "BarÃµes", "Inibidores", "Tempo"]
ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]
VALUE_MARKET_META = {
    "Kills": {
        "kind": "kills",
        "column": "total_kills",
        "line_key": "lb_value_line_kills",
        "line_label": "Kills",
        "high_label": "Over",
        "low_label": "Under",
        "delta_high": 1.25,
        "profile_scale": 1.15,
        "draft_scale": 0.95,
    },
    "Torres": {
        "kind": "towers",
        "column": "total_towers",
        "line_key": "lb_value_line_towers",
        "line_label": "Torres",
        "high_label": "Mais torres",
        "low_label": "Menos torres",
        "delta_high": 0.55,
        "profile_scale": 0.55,
        "draft_scale": 0.45,
    },
    "DragÃµes": {
        "kind": "dragons",
        "column": "total_dragons",
        "line_key": "lb_value_line_dragons",
        "line_label": "DragÃµes",
        "high_label": "Mais dragÃµes",
        "low_label": "Menos dragÃµes",
        "delta_high": 0.3,
        "profile_scale": 0.3,
        "draft_scale": 0.22,
    },
    "BarÃµes": {
        "kind": "barons",
        "column": "total_nashors",
        "line_key": "lb_value_line_barons",
        "line_label": "BarÃµes",
        "high_label": "Mais barÃµes",
        "low_label": "Menos barÃµes",
        "delta_high": 0.18,
        "profile_scale": 0.14,
        "draft_scale": 0.12,
    },
    "Tempo": {
        "kind": "time",
        "column": "game_time_min",
        "line_key": "lb_value_line_time",
        "line_label": "Tempo",
        "high_label": "Mais longo",
        "low_label": "Mais curto",
        "delta_high": 0.75,
        "profile_scale": 0.7,
        "draft_scale": 0.55,
    },
}
PLAYER_PROP_META = {
    "Kills": {"columns": ("kills",), "line_key": "lb_prop_line_kills", "high_label": "Over", "low_label": "Under", "delta_high": 0.8},
    "Deaths": {"columns": ("deaths",), "line_key": "lb_prop_line_deaths", "high_label": "Over", "low_label": "Under", "delta_high": 0.8},
    "Assists": {"columns": ("assists",), "line_key": "lb_prop_line_assists", "high_label": "Over", "low_label": "Under", "delta_high": 1.1},
    "K+A": {"columns": ("kills", "assists"), "line_key": "lb_prop_line_ka", "high_label": "Over", "low_label": "Under", "delta_high": 1.4},
}

def _parse_line_value(raw: Any, *, is_time: bool = False) -> Optional[float]:
    text = str(raw or "").strip()
    if not text:
        return None
    if is_time and ":" in text:
        try:
            mins, secs = text.split(":", 1)
            mins_f = float(mins.strip())
            secs_f = float(secs.strip())
            return float(mins_f + (secs_f / 60.0))
        except Exception:
            return None
    return _safe_float(text.replace(",", "."))


def _parse_names(text: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\n\t,;]+", str(text or "").strip()) if token.strip()]


def _normalize_compact_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _build_compact_vocab(vocab: list[str]) -> tuple[dict[str, str], int]:
    mapping: dict[str, str] = {}
    max_parts = 1
    for raw in list(vocab or []):
        champ_name = str(raw or "").strip()
        if not champ_name:
            continue
        norm_key = _normalize_compact_key(champ_name)
        if norm_key:
            mapping.setdefault(norm_key, champ_name)
        part_count = len([token for token in re.split(r"[^A-Za-z0-9]+", champ_name) if token])
        max_parts = max(max_parts, int(part_count or 1))
    return mapping, max_parts


def parse_compact_draft_tokens(text: str, vocab: list[str], max_picks: int = 10) -> list[str]:
    raw_tokens = [token.strip() for token in re.split(r"[\s,;|/]+", str(text or "").strip()) if token.strip()]
    if not raw_tokens:
        return []
    vocab_map, max_parts = _build_compact_vocab(vocab)
    parsed: list[str] = []
    idx = 0
    while idx < len(raw_tokens) and len(parsed) < int(max_picks):
        matched_name: str | None = None
        max_span = min(int(max_parts), len(raw_tokens) - idx)
        for span in range(max_span, 0, -1):
            candidate = "".join(raw_tokens[idx : idx + span])
            norm_key = _normalize_compact_key(candidate)
            if norm_key in vocab_map:
                matched_name = str(vocab_map[norm_key])
                idx += span
                break
        if matched_name is None:
            matched_name = str(raw_tokens[idx])
            idx += 1
        parsed.append(matched_name)
    return parsed


def _profile_group(bucket: str) -> str:
    value = str(bucket or "").strip()
    if value in {"Muito Under", "Under"}:
        return "under"
    if value in {"Muito Over", "Over"}:
        return "over"
    return "neutral"


def _market_resolution(bucket_a: str, bucket_b: str) -> str:
    group_a = _profile_group(bucket_a)
    group_b = _profile_group(bucket_b)
    if group_a == group_b == "over":
        return "Over"
    if group_a == group_b == "under":
        return "Under"
    if group_a == group_b == "neutral":
        return "Neutro"
    if "neutral" in {group_a, group_b}:
        other = group_b if group_a == "neutral" else group_a
        if other == "over":
            return "Vies Over"
        if other == "under":
            return "Vies Under"
    return "Choque"


def _profile_strength_text(bucket: str) -> str:
    value = str(bucket or "").strip()
    if value == "Muito Under":
        return "Under forte"
    if value == "Under":
        return "Under moderado"
    if value == "Neutro":
        return "Neutro"
    if value == "Over":
        return "Over moderado"
    if value == "Muito Over":
        return "Over forte"
    return "-"


def _sample_band(games: int) -> str:
    n = int(games or 0)
    if n <= 0:
        return "sem base"
    if n <= 2:
        return "muito curta"
    if n <= 5:
        return "curta"
    if n <= 9:
        return "media"
    return "forte"


def _confidence_label(z_value: Optional[float], games: int) -> str:
    if int(games or 0) <= 0:
        return "Sem base"
    z_abs = abs(float(z_value or 0.0))
    if int(games or 0) >= 10 and z_abs >= 0.8:
        return "Alta"
    if int(games or 0) >= 6 and z_abs >= 0.35:
        return "Media"
    return "Baixa"


def _chip_html(text: str, tone: str) -> str:
    palette = {
        "over": ("#3b0f14", "#ff7a86", "#ffb3ba"),
        "under": ("#0e2437", "#6ec1ff", "#b9e1ff"),
        "neutral": ("#1d2430", "#8a94a6", "#d7dce5"),
        "high": ("#0f2e1b", "#4fd18b", "#bbf7d0"),
        "medium": ("#34240e", "#f0b45b", "#fde0a6"),
        "low": ("#31191b", "#df7d89", "#ffc8cf"),
        "sample_strong": ("#10243b", "#72aafc", "#d5e4ff"),
        "sample_mid": ("#273144", "#8da1bf", "#dce4f2"),
        "sample_short": ("#332618", "#d7a368", "#f7dfbf"),
        "sample_none": ("#2a2a2a", "#7f7f7f", "#d9d9d9"),
    }
    bg, border, fg = palette.get(tone, palette["neutral"])
    return (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:999px;'
        f'border:1px solid {border};background:{bg};color:{fg};font-size:12px;line-height:1.2;'
        f'margin:0 6px 6px 0;white-space:nowrap;">{escape(str(text or "-"))}</span>'
    )


def _profile_tone(profile: str) -> str:
    value = str(profile or "").casefold()
    if "over" in value:
        return "over"
    if "under" in value:
        return "under"
    return "neutral"


def _confidence_tone(confidence: str) -> str:
    value = str(confidence or "").casefold()
    if value == "alta":
        return "high"
    if value == "media":
        return "medium"
    return "low"


def _sample_tone(sample: str) -> str:
    value = str(sample or "").casefold()
    if value == "forte":
        return "sample_strong"
    if value == "media":
        return "sample_mid"
    if value in {"curta", "muito curta"}:
        return "sample_short"
    return "sample_none"


def _market_read_tone(reading: str) -> str:
    value = str(reading or "").casefold()
    if any(token in value for token in ("over", "mais ", "favorece over", "mais torres", "mais dragoes", "mais baroes")):
        return "over"
    if any(token in value for token in ("under", "menos ", "favorece under", "menos torres", "menos dragoes", "menos baroes")):
        return "under"
    return "neutral"


def _resolution_label(bucket_a: str, bucket_b: str) -> str:
    group_a = _profile_group(bucket_a)
    group_b = _profile_group(bucket_b)
    if group_a == "over" and group_b == "over":
        return "Over"
    if group_a == "under" and group_b == "under":
        return "Under"
    if group_a == "neutral" and group_b == "neutral":
        return "Neutro"
    if {group_a, group_b} == {"over", "neutral"}:
        return "Vies Over"
    if {group_a, group_b} == {"under", "neutral"}:
        return "Vies Under"
    return "Choque"


def _resolution_tone(resolution: str) -> str:
    value = str(resolution or "").casefold()
    if "over" in value:
        return "over"
    if "under" in value:
        return "under"
    if value == "choque":
        return "low"
    return "neutral"

def _value_confidence_label(*, abs_delta: float, evidence_count: int, strong_delta: float) -> str:
    if evidence_count >= 5 and abs_delta >= strong_delta:
        return "Alta"
    if evidence_count >= 4 and abs_delta >= (0.55 * strong_delta):
        return "Media"
    return "Baixa"

def _confidence_rank_from_score(score: float) -> str:
    value = abs(float(score or 0.0))
    if value >= 0.75:
        return "Alta"
    if value >= 0.4:
        return "Media"
    return "Baixa"



def _metric_team_mean(
    df_scope: pd.DataFrame | None,
    *,
    team_name: str,
    column: str,
    side: str | None = None,
    opponent: str | None = None,
) -> Optional[float]:
    if df_scope is None or df_scope.empty or not column or column not in df_scope.columns:
        return None
    subset = df_scope[df_scope["team"].astype(str) == str(team_name)].copy()
    if subset.empty:
        return None
    if side and "side" in subset.columns:
        subset = subset[subset["side"].astype(str).str.casefold() == str(side).casefold()].copy()
    if opponent and "opponent" in subset.columns:
        subset = subset[subset["opponent"].astype(str) == str(opponent)].copy()
    if subset.empty:
        return None
    return _safe_float(pd.to_numeric(subset.get(column), errors="coerce").mean())


def _weighted_value(parts: list[tuple[Optional[float], float]]) -> Optional[float]:
    valid = [(float(value), float(weight)) for value, weight in parts if value is not None and math.isfinite(float(value)) and float(weight) > 0.0]
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0.0:
        return None
    return sum(value * weight for value, weight in valid) / total_weight


__all__ = [
    "LIVE_MARKET_LABELS",
    "PLAYER_PROP_META",
    "ROLE_ORDER",
    "VALUE_MARKET_META",
    "_build_compact_vocab",
    "_chip_html",
    "_confidence_label",
    "_confidence_rank_from_score",
    "_confidence_tone",
    "_market_read_tone",
    "_market_resolution",
    "_metric_team_mean",
    "_normalize_compact_key",
    "_parse_line_value",
    "_parse_names",
    "_profile_group",
    "_profile_strength_text",
    "_profile_tone",
    "_resolution_label",
    "_resolution_tone",
    "_sample_band",
    "_sample_tone",
    "_value_confidence_label",
    "_weighted_value",
    "parse_compact_draft_tokens",
]
