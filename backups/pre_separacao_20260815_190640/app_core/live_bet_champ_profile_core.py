"""Champion meta-profile and tactical-profile helpers for live-bet draft analysis.

Extracted from live_bet_draft_core.py. All functions are stateless and depend only
on DataFrames, champion registry dicts, and helpers from live_bet_stats_core.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from app_core.champion_knowledge_core import champion_style_summary, get_champion_knowledge
from app_core.live_bet_shared_core import _weighted_value
from app_core.live_bet_stats_core import _laplace_win_pct, _scale_0_100
from mlcore.utils import safe_float_or_none as _safe_float


def _champ_trait(entry: Mapping[str, Any] | None, key: str) -> float:
    """Return a single trait score from a champion knowledge entry (0.0 if missing)."""
    if not isinstance(entry, Mapping):
        return 0.0
    traits = entry.get("traits") or {}
    return float(_safe_float((traits or {}).get(key)) or 0.0)


def _champ_meta_profile(
    *,
    df_scope_players: pd.DataFrame | None,
    champ_name: str,
    role: str,
    side_label: str,
    registry: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compute a champion's meta score and label for a given role/side context."""
    if df_scope_players is None or df_scope_players.empty or not str(champ_name or "").strip():
        return {"label": "Sem base", "score": 50.0, "global_label": "Sem base", "global_score": 50.0}
    role_key = str(role or "").strip().lower()
    role_df = df_scope_players.copy()
    if role_key and "pos" in role_df.columns:
        role_df = role_df[role_df["pos"].astype(str).str.lower() == role_key].copy()
    champ_df = role_df[role_df["champ"].astype(str) == str(champ_name)].copy() if "champ" in role_df.columns else pd.DataFrame([])
    if champ_df.empty:
        return {"label": "Sem base", "score": 50.0, "global_label": "Sem base", "global_score": 50.0}
    side_df = champ_df[champ_df["side"].astype(str).str.title() == str(side_label).title()].copy() if "side" in champ_df.columns else pd.DataFrame([])
    total_role_games = max(int(role_df.shape[0]), 1)
    sample = int(champ_df.shape[0])
    presence_pct = 100.0 * float(sample) / float(total_role_games)
    wr_role = _laplace_win_pct(champ_df)
    wr_side = _laplace_win_pct(side_df) if not side_df.empty else None
    role_counts = role_df.groupby("champ", dropna=False).size().astype(float).sort_values(ascending=False) if "champ" in role_df.columns else pd.Series(dtype=float)
    role_wr_values: list[float] = []
    if "champ" in role_df.columns and "result" in role_df.columns:
        for _, group in role_df.groupby("champ", dropna=False):
            wr_item = _laplace_win_pct(group)
            if wr_item is not None:
                role_wr_values.append(float(wr_item))
    count_rank_pct = 50.0
    if not role_counts.empty:
        sample_f = float(sample)
        count_rank_pct = 100.0 * float(sum(1 for value in role_counts.tolist() if float(value) <= sample_f)) / float(len(role_counts))
    wr_rank_pct = 50.0
    if role_wr_values and wr_role is not None:
        wr_rank_pct = 100.0 * float(sum(1 for value in role_wr_values if float(value) <= float(wr_role))) / float(len(role_wr_values))
    champ = get_champion_knowledge(registry, champ_name, role_hint=role_key)
    semantic_power = _scale_0_100(
        0.18 * _champ_trait(champ, "early")
        + 0.20 * _champ_trait(champ, "mid")
        + 0.22 * _champ_trait(champ, "late")
        + 0.16 * _champ_trait(champ, "teamfight")
        + 0.12 * _champ_trait(champ, "objective_control")
        + 0.12 * _champ_trait(champ, "pick")
        + 0.10 * _champ_trait(champ, "engage"),
        target=1.2,
    )
    presence_score = _scale_0_100(presence_pct, target=14.0)
    meta_score = _weighted_value(
        [
            (wr_role, 0.42),
            (wr_side, 0.16),
            (presence_score, 0.22),
            (semantic_power, 0.20),
        ]
    )
    label = "Meta neutro"
    if meta_score is not None and sample >= 4 and meta_score >= 66.0:
        label = "Meta forte"
    elif meta_score is not None and sample >= 3 and meta_score >= 57.0:
        label = "Meta ok"
    elif meta_score is not None and sample >= 4 and meta_score <= 45.0:
        label = "Meta fraco"
    global_score = _weighted_value(
        [
            (wr_rank_pct, 0.34),
            (count_rank_pct, 0.32),
            (semantic_power, 0.20),
            (presence_score, 0.14),
        ]
    )
    global_label = "Situacional"
    if sample < 3:
        global_label = "Sem base"
    elif global_score is not None and sample >= 6 and presence_pct >= 8.0 and global_score >= 78.0:
        global_label = "Premium"
    elif global_score is not None and sample >= 4 and presence_pct >= 5.0 and global_score >= 64.0:
        global_label = "Contestado"
    elif global_score is not None and sample >= 4 and global_score <= 38.0:
        global_label = "Abaixo do meta"
    return {
        "label": label,
        "score": float(_safe_float(meta_score) or 50.0),
        "wr_role": wr_role,
        "wr_side": wr_side,
        "presence_pct": presence_pct,
        "sample": sample,
        "global_label": global_label,
        "global_score": float(_safe_float(global_score) or 50.0),
        "count_rank_pct": float(count_rank_pct),
        "wr_rank_pct": float(wr_rank_pct),
    }


def _player_tactical_profile(role: str, champ_name: str, registry: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compute tactical role scores (carry/engage/frontline/setup/side) for a champion."""
    role_key = str(role or "").strip().lower()
    champ = get_champion_knowledge(registry, champ_name, role_hint=role)
    engage = _champ_trait(champ, "engage") + _champ_trait(champ, "dive") + 0.7 * _champ_trait(champ, "pick")
    carry = (
        _champ_trait(champ, "late")
        + 0.9 * _champ_trait(champ, "range")
        + 0.7 * _champ_trait(champ, "siege")
        + 0.55 * _champ_trait(champ, "teamfight")
        + 0.25 * _champ_trait(champ, "pick")
        - 0.55 * _champ_trait(champ, "frontline")
    )
    setup = _champ_trait(champ, "peel") + 0.7 * _champ_trait(champ, "objective_control") + 0.45 * _champ_trait(champ, "teamfight")
    frontline = _champ_trait(champ, "frontline") + 0.45 * _champ_trait(champ, "all_in") + 0.2 * _champ_trait(champ, "teamfight")
    side = (
        _champ_trait(champ, "splitpush")
        + 0.45 * _champ_trait(champ, "mobility")
        + 0.35 * _champ_trait(champ, "lane_priority")
        + 0.25 * _champ_trait(champ, "waveclear")
    )
    exposed = (
        0.75 * carry
        + 0.35 * _champ_trait(champ, "range")
        + 0.25 * _champ_trait(champ, "late")
        - 0.6 * _champ_trait(champ, "mobility")
        - 0.7 * frontline
        - 0.45 * _champ_trait(champ, "peel")
    )
    if role_key == "bot":
        carry += 0.25
        exposed += 0.85
    elif role_key == "mid":
        carry += 0.18
        exposed += 0.35
    elif role_key == "sup":
        setup += 0.35
        if engage >= 1.4:
            exposed += 0.45
    elif role_key == "jng":
        engage += 0.15
        if engage >= 1.4:
            exposed += 0.3
    elif role_key == "top":
        frontline += 0.18
        if side >= 1.0:
            side += 0.2

    labels = [
        ("carrega", carry),
        ("inicia", engage),
        ("frontline", frontline),
        ("setup", setup),
        ("side", side),
    ]
    primary_label, primary_score = max(labels, key=lambda item: item[1])
    if primary_score <= 0.75:
        primary_label = champion_style_summary(champ, limit=1)

    if exposed >= 1.8:
        risk_label = "Muito exposto"
    elif exposed >= 1.1:
        risk_label = "Exposto"
    else:
        risk_label = "Seguro"

    return {
        "primary_label": primary_label,
        "risk_label": risk_label,
        "carry_score": carry,
        "engage_score": engage,
        "setup_score": setup,
        "frontline_score": frontline,
        "side_score": side,
        "exposure_score": exposed,
    }
