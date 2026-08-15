from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app_core.flow_runtime_core import resolve_players_artifact_path as core_resolve_players_artifact_path
from mlcore.roster_signal_core import (  # noqa: F401 - re-exported for callers + tests
    lineup_continuity_from_details,
    transfer_signal_from_details,
)


def autofill_lineup_state(
    *,
    session_state: Any,
    lineup_key: str,
    do_auto: bool,
    df_pr: pd.DataFrame | None,
    team_a: str,
    team_b: str,
    filters: dict[str, Any] | None,
    league_fixed_value: Any,
    pos_canon: list[str],
    pos_canon_ml: list[str],
    safe_key_fn: Callable[[str], str],
    latest_lineup_for_team_fn: Callable[..., dict[str, str]],
    save_lineup_for_matchup_fn: Callable[[str], None],
) -> bool:
    if df_pr is None or df_pr.empty:
        return False
    if not (bool(do_auto) or session_state.get("_lineup_autofilled_key") != lineup_key):
        return False
    la = latest_lineup_for_team_fn(df_pr, team_a, filters, league_fixed_value, pos_col="pos", player_col="player", pos_list=pos_canon)
    lb = latest_lineup_for_team_fn(df_pr, team_b, filters, league_fixed_value, pos_col="pos", player_col="player", pos_list=pos_canon)
    la_ml = latest_lineup_for_team_fn(df_pr, team_a, filters, league_fixed_value, pos_col="pos_ml", player_col="player_key", pos_list=pos_canon_ml)
    lb_ml = latest_lineup_for_team_fn(df_pr, team_b, filters, league_fixed_value, pos_col="pos_ml", player_col="player_key", pos_list=pos_canon_ml)
    for pos in pos_canon:
        key_a = f"lineup_A_{safe_key_fn(pos)}"
        key_b = f"lineup_B_{safe_key_fn(pos)}"
        if do_auto or not str(session_state.get(key_a, "") or "").strip():
            session_state[key_a] = str(la.get(pos, "") or "")
        if do_auto or not str(session_state.get(key_b, "") or "").strip():
            session_state[key_b] = str(lb.get(pos, "") or "")
    for pos in pos_canon_ml:
        key_a = f"lineup_ml_A_{safe_key_fn(pos)}"
        key_b = f"lineup_ml_B_{safe_key_fn(pos)}"
        if do_auto or not str(session_state.get(key_a, "") or "").strip():
            session_state[key_a] = str(la_ml.get(pos, "") or "")
        if do_auto or not str(session_state.get(key_b, "") or "").strip():
            session_state[key_b] = str(lb_ml.get(pos, "") or "")
    session_state["_lineup_autofilled_key"] = lineup_key
    save_lineup_for_matchup_fn(lineup_key)
    return True


def resolve_players_layer_config(
    *,
    artifact_path: str,
    session_state: Any,
    load_artifact_cached_fn: Callable[[tuple[Any, ...]], Any],
    artifact_sig_fn: Callable[[str], tuple[Any, ...]],
    ui_weight_default: float = 1.15,
) -> dict[str, Any]:
    enabled = False
    players_path = ""
    weight = float(ui_weight_default)
    weight_source = "ui"
    art_obj: Any = None
    try:
        art_obj = load_artifact_cached_fn(artifact_sig_fn(str(artifact_path)))
        meta = dict(((art_obj.meta or {}).get("players_layer")) or {})
    except Exception:
        meta = {}
    try:
        artifact_parent = Path(str(artifact_path)).expanduser().resolve().parent if str(artifact_path or "").strip() else Path.cwd()
        candidate = core_resolve_players_artifact_path(
            str(artifact_path),
            artifact_parent,
            load_artifact_fn=lambda path: art_obj if art_obj is not None else load_artifact_cached_fn(artifact_sig_fn(str(path))),
        )
        if candidate and Path(candidate).exists():
            enabled = True
            players_path = str(candidate)
        meta_weight = float(meta.get("w_lineup", 0.0) or 0.0) if isinstance(meta, dict) else 0.0
        if meta_weight > 0:
            weight = float(meta_weight)
            weight_source = "meta"
        else:
            weight = float(session_state.get("ml_players_w_lineup", ui_weight_default) or ui_weight_default)
            weight_source = "ui"
    except Exception:
        enabled = False
        players_path = ""
        weight = float(session_state.get("ml_players_w_lineup", ui_weight_default) or ui_weight_default)
        weight_source = "ui"
    return {
        "enabled": bool(enabled),
        "players_path": str(players_path),
        "weight": float(weight),
        "weight_source": str(weight_source),
    }


def build_manual_lineup_choices(
    *,
    df_imp: pd.DataFrame | None,
    df_pr: pd.DataFrame | None,
    team_ui: str,
    team_key: str,
    pos_canon: list[str],
    norm_pos_fn: Callable[[str], str],
) -> dict[str, list[str]]:
    out = {pos: [] for pos in pos_canon}
    if df_imp is not None and not df_imp.empty and team_key:
        for pos in pos_canon:
            sub = df_imp[(df_imp["team"].astype(str) == str(team_key)) & (df_imp["pos"].astype(str) == str(pos))].copy()
            if not sub.empty:
                sub = sub.sort_values(["score", "gp"], ascending=[False, False])
                out[pos] += sub["player"].astype(str).tolist()
    if df_pr is not None and not df_pr.empty:
        try:
            d2 = df_pr[df_pr["team"].astype(str) == str(team_ui)].copy()
            d2["pos_can"] = d2["pos"].apply(norm_pos_fn)
            for pos in pos_canon:
                out[pos] += d2[d2["pos_can"] == pos]["player"].dropna().astype(str).unique().tolist()
        except Exception:
            pass
    for pos in pos_canon:
        vals = [v.strip() for v in out[pos] if str(v).strip()]
        seen = set()
        uniq: list[str] = []
        for value in vals:
            if value not in seen:
                uniq.append(value)
                seen.add(value)
        out[pos] = uniq
    return out


def build_ml_lineup_choices(
    *,
    df_pr: pd.DataFrame | None,
    team_ui: str,
    pos_canon_ml: list[str],
    norm_pos_ml_fn: Callable[[str], str],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    out = {pos: [] for pos in pos_canon_ml}
    disp: dict[str, str] = {}
    if df_pr is None or df_pr.empty:
        return out, disp
    d2 = df_pr[df_pr["team"].astype(str) == str(team_ui)].copy()
    if d2.empty:
        return out, disp
    if "player_key" not in d2.columns:
        d2["player_key"] = d2.get("playerid", "").astype(str).str.strip()
        d2.loc[d2["player_key"].astype(str).str.strip() == "", "player_key"] = d2.get("player", "").astype(str).str.strip()
    if "pos_ml" not in d2.columns:
        d2["pos_ml"] = d2.get("pos", "").apply(norm_pos_ml_fn)
    try:
        grp = d2.groupby("player_key")["player"].agg(
            lambda s: [x for i, x in enumerate([str(v).strip() for v in s.tolist() if str(v).strip()]) if x and x not in [str(v).strip() for v in s.tolist()[:i]]]
        )
        for pid, names in grp.items():
            if not pid:
                continue
            if isinstance(names, list) and names:
                main = names[0]
                alts = [n for n in names[1:] if n and n != main]
                disp[str(pid)] = f"{main} ({' / '.join(alts[:2])})" if alts else str(main)
            else:
                disp[str(pid)] = str(pid)
    except Exception:
        pass
    for pos in pos_canon_ml:
        sub = d2[d2["pos_ml"].astype(str) == str(pos)].copy()
        if sub.empty:
            continue
        sub["player_key"] = sub["player_key"].astype(str).str.strip()
        sub = sub[sub["player_key"] != ""]
        if sub.empty:
            continue
        try:
            freq = sub.groupby("player_key").agg(gp=("gameid", "nunique"), last=("date", "max")).reset_index()
            freq = freq.sort_values(["gp", "last"], ascending=[False, False])
            out[pos] = freq["player_key"].astype(str).tolist()
        except Exception:
            out[pos] = sorted(sub["player_key"].astype(str).unique().tolist())
    return out, disp


def score_manual_lineup_team(
    *,
    team_key: str,
    lineup: dict[str, Any],
    df_imp: pd.DataFrame | None,
    pos_canon: list[str],
    min_gp: int,
    best_match_fn: Callable[..., str],
) -> tuple[float, int, int, int]:
    if not team_key or df_imp is None or df_imp.empty:
        return 0.0, 0, 0, 0
    score = 0.0
    found = 0
    total = 0
    gp_sum = 0
    for pos in pos_canon:
        total += 1
        player = str(lineup.get(pos, "") or "").strip()
        if not player:
            continue
        sub = df_imp[(df_imp["team"].astype(str) == str(team_key)) & (df_imp["pos"].astype(str) == str(pos))]
        if sub.empty:
            continue
        player_key = best_match_fn(player, sub["player"].astype(str).tolist(), cutoff=0.68)
        if not player_key:
            continue
        row = sub[sub["player"].astype(str) == str(player_key)].head(1)
        if row.empty:
            continue
        gp = int(row.iloc[0].get("gp", 0) or 0)
        if gp < int(min_gp):
            continue
        score += float(row.iloc[0].get("score", 0.0) or 0.0)
        found += 1
        gp_sum += gp
    return float(score), int(found), int(total), int(gp_sum)
