# -*- coding: utf-8 -*-
"""Builds the ML strength rank table from an artifact."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app_core.achievements_resolve_core import ach_layer_for_artifact as _ach_layer_for_artifact
from app_core.ml_artifact_pricing_core import (
    strength_breakdown as _core_strength_breakdown,
    team_meta_from_artifact as _team_meta_from_artifact,
)
from app_core.ml_league_runtime_core import (
    norm_league as _norm_league,
    tier_from_league as _tier_from_league,
)
from app_core.ui_labels import fmt_num as _fmt_num
from mlcore.achievements import team_boost_points as _team_boost_points
from mlcore.tiers import tier_base_points as _tier_base_points, league_base_points as _league_base_points


def mlcore_rank_table_v2(
    artifact_path: str,
    as_of: date,
    top_n: int = 200,
    *,
    ctx: Mapping[str, Any],
) -> pd.DataFrame:
    _file_sig = ctx.get("_file_sig")
    _load_artifact_cached = ctx.get("_load_artifact_cached")

    cols = [
        "rank", "team", "league", "tier",
        "residual_elo", "league_strength", "tier_strength",
        "league_base", "tier_base", "ach_boost",
        "base_strength", "total_strength",
    ]
    p = Path(artifact_path)
    if not p.exists():
        return pd.DataFrame(columns=cols)

    art_sig = _file_sig(str(p))
    art = _load_artifact_cached(art_sig)
    team_meta = _team_meta_from_artifact(art)
    ach_layer = _ach_layer_for_artifact(art, str(p), ctx=ctx)

    teams: set = set()
    teams.update((getattr(art.model, "ratings", {}) or {}).keys())
    teams.update(team_meta.keys())

    rows = []
    for t in teams:
        bd = _core_strength_breakdown(
            art=art,
            team=str(t),
            as_of=as_of,
            team_meta=team_meta,
            ach_layer=ach_layer,
            norm_league_fn=_norm_league,
            tier_from_league_fn=_tier_from_league,
            tier_base_points_fn=_tier_base_points,
            league_base_points_fn=_league_base_points,
            team_boost_points_fn=_team_boost_points,
        )
        rows.append(bd)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)

    # ORDEM POR TIER (achado com ele 24/07): a tabela mostrava academies da LCKC
    # (residual altissimo por dominar pool fraco) ACIMA de Gen.G/T1. O
    # strength_breakdown monta o total com apply_rank_caps=False porque serve ao
    # PRICING, que precisa dos componentes crus — nao se mexe la. Aqui usamos a
    # MESMA regra do _top_global_v04 do modelo: tier manda, forca desempata.
    # Preferimos isso a capar o valor (o cap T1=909 achataria Gen.G/T1/BLG num
    # empate) — assim a forca exibida continua real e a ordem faz sentido.
    from mlcore.tiers import tier_priority as _tier_prio
    df["_tier_rank"] = df["tier"].map(lambda t: _tier_prio(str(t or "OTHER")))

    # tier ASC (T1 primeiro) + forca DESC dentro do tier; residual desempata.
    _tie = [c for c in ("residual_elo",) if c in df.columns]
    df = df.sort_values(
        ["_tier_rank", "total_strength", *_tie],
        ascending=[True, False] + [False] * len(_tie),
    ).reset_index(drop=True)
    df = df.drop(columns=["_tier_rank"])
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    if top_n is not None and int(top_n) > 0:
        df = df.head(int(top_n)).copy()

    for c in [
        "residual_elo", "league_strength", "tier_strength",
        "tier_base", "league_base", "ach_boost",
        "base_strength", "total_strength",
    ]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: _fmt_num(x, 1))

    return df
