from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd


def rank_decision(v: str) -> int:
    return {"promote": 0, "shadow_only": 1, "reject": 2}.get(str(v or "").strip().lower(), 1)


def fingerprint_inventory(records: Iterable[Dict[str, Any]]) -> str:
    rows: List[Dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "path": str(row.get("path") or ""),
                "size": int(row.get("size") or 0),
                "mtime_ns": int(row.get("mtime_ns") or 0),
            }
        )
    rows.sort(key=lambda r: str(r.get("path") or ""))
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def summarize_league_changes(
    raw_leagues: Iterable[str],
    *,
    known_raw_leagues: Iterable[str] | None = None,
    known_canonical_leagues: Iterable[str] | None = None,
    canonicalize_fn: Callable[[str], str] | None = None,
) -> Dict[str, Any]:
    known_raw = {str(x or "").strip().upper() for x in (known_raw_leagues or []) if str(x or "").strip()}
    known_canonical: set[str] = set()
    for item in (known_canonical_leagues or []):
        raw_item = str(item or "").strip().upper()
        if not raw_item:
            continue
        canon_item = str(canonicalize_fn(raw_item) if canonicalize_fn is not None else raw_item).strip().upper()
        known_canonical.add(canon_item or raw_item)
    raw_sorted = sorted({str(x or "").strip().upper() for x in raw_leagues if str(x or "").strip()})

    alias_pairs: List[Dict[str, str]] = []
    new_raw_mapped: List[Dict[str, str]] = []
    new_canonical: List[str] = []
    canonical_seen: set[str] = set()

    for raw in raw_sorted:
        canon = str(canonicalize_fn(raw) if canonicalize_fn is not None else raw).strip().upper()
        canon = canon or raw
        canonical_seen.add(canon)
        if canon != raw:
            alias_pairs.append({"raw": raw, "canonical": canon})
        if raw not in known_raw and canon in known_canonical:
            new_raw_mapped.append({"raw": raw, "canonical": canon})
        if canon not in known_canonical:
            new_canonical.append(canon)

    return {
        "raw_leagues": raw_sorted,
        "canonical_leagues": sorted(canonical_seen),
        "alias_pairs": alias_pairs,
        "new_raw_mapped_to_known_canonical": new_raw_mapped,
        "new_canonical_leagues": sorted(set(new_canonical)),
    }


def detect_series_gap_suspects(df: pd.DataFrame, *, source_name: str = "") -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "source": str(source_name or ""),
            "suspect_groups": 0,
            "suspect_games": 0,
            "examples": [],
        }

    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    required = {"gameid", "participantid", "teamname", "date", "league", "game"}
    if not required.issubset(cols):
        return {
            "source": str(source_name or ""),
            "suspect_groups": 0,
            "suspect_games": 0,
            "examples": [],
            "error": "missing_required_columns",
        }

    team_rows = df[pd.to_numeric(df[cols["participantid"]], errors="coerce").isin([100, 200])].copy()
    if team_rows.empty:
        return {
            "source": str(source_name or ""),
            "suspect_groups": 0,
            "suspect_games": 0,
            "examples": [],
        }

    team_rows["_csv_gate_date"] = pd.to_datetime(team_rows[cols["date"]], errors="coerce")
    team_rows["_csv_gate_team"] = team_rows[cols["teamname"]].fillna("").astype(str).str.strip()
    team_rows["_csv_gate_league"] = team_rows[cols["league"]].fillna("").astype(str).str.strip()
    team_rows["_csv_gate_split"] = (
        team_rows[cols["split"]].fillna("").astype(str).str.strip()
        if "split" in cols
        else ""
    )
    team_rows["_csv_gate_playoffs"] = (
        pd.to_numeric(team_rows[cols["playoffs"]], errors="coerce").fillna(0).astype(int)
        if "playoffs" in cols
        else 0
    )
    team_rows["_csv_gate_year"] = (
        pd.to_numeric(team_rows[cols["year"]], errors="coerce")
        if "year" in cols
        else pd.Series([pd.NA] * len(team_rows), index=team_rows.index, dtype="Int64")
    )
    team_rows["_csv_gate_game"] = pd.to_numeric(team_rows[cols["game"]], errors="coerce")
    team_rows = team_rows.dropna(subset=["_csv_gate_date"]).copy()
    if team_rows.empty:
        return {
            "source": str(source_name or ""),
            "suspect_groups": 0,
            "suspect_games": 0,
            "examples": [],
        }

    team_rows = team_rows.sort_values([cols["gameid"], "_csv_gate_date"], kind="mergesort")
    team_rows = team_rows.drop_duplicates(subset=[cols["gameid"], "_csv_gate_team"], keep="last")

    game_rows: list[Dict[str, Any]] = []
    for gid, group in team_rows.groupby(cols["gameid"], sort=False):
        teams = sorted({str(x).strip() for x in group["_csv_gate_team"].tolist() if str(x).strip()})
        if len(teams) != 2:
            continue
        game_vals = pd.to_numeric(group["_csv_gate_game"], errors="coerce").dropna().tolist()
        if not game_vals:
            continue
        game_num = int(game_vals[0])
        if game_num <= 0:
            continue
        dt = pd.to_datetime(group["_csv_gate_date"], errors="coerce").dropna()
        if dt.empty:
            continue
        row0 = group.iloc[0]
        year_val = pd.to_numeric(pd.Series([row0.get("_csv_gate_year")]), errors="coerce").iloc[0]
        playoffs_val = pd.to_numeric(pd.Series([row0.get("_csv_gate_playoffs")]), errors="coerce").iloc[0]
        game_rows.append(
            {
                "gameid": str(gid),
                "date": dt.iloc[0],
                "date_day": dt.iloc[0].date().isoformat(),
                "league": str(row0.get("_csv_gate_league") or ""),
                "year": (int(year_val) if pd.notna(year_val) else None),
                "split": str(row0.get("_csv_gate_split") or ""),
                "playoffs": (int(playoffs_val) if pd.notna(playoffs_val) else 0),
                "team_a": teams[0],
                "team_b": teams[1],
                "game_num": int(game_num),
            }
        )

    if not game_rows:
        return {
            "source": str(source_name or ""),
            "suspect_groups": 0,
            "suspect_games": 0,
            "examples": [],
        }

    grouped_df = pd.DataFrame(game_rows)
    suspect_groups = 0
    suspect_games = 0
    examples: list[Dict[str, Any]] = []
    group_cols = ["league", "year", "split", "playoffs", "date_day", "team_a", "team_b"]
    for _, group in grouped_df.groupby(group_cols, dropna=False, sort=False):
        nums = sorted({int(x) for x in pd.to_numeric(group["game_num"], errors="coerce").dropna().tolist() if int(x) > 0})
        if not nums or int(max(nums)) <= 1:
            continue
        expected = list(range(1, int(max(nums)) + 1))
        missing = [value for value in expected if value not in nums]
        if not missing:
            continue
        suspect_groups += 1
        suspect_games += len(group)
        if len(examples) >= 25:
            continue
        examples.append(
            {
                "source": str(source_name or ""),
                "date_day": str(group["date_day"].iloc[0]),
                "league": str(group["league"].iloc[0] or ""),
                "year": (
                    int(group["year"].iloc[0])
                    if pd.notna(group["year"].iloc[0])
                    else None
                ),
                "split": str(group["split"].iloc[0] or ""),
                "playoffs": int(group["playoffs"].iloc[0] or 0),
                "teams": f"{group['team_a'].iloc[0]} vs {group['team_b'].iloc[0]}",
                "present_game_numbers": nums,
                "missing_game_numbers": missing,
                "gameids": [str(x) for x in group["gameid"].astype(str).tolist()],
            }
        )

    return {
        "source": str(source_name or ""),
        "suspect_groups": int(suspect_groups),
        "suspect_games": int(suspect_games),
        "examples": examples,
    }


def evaluate_csv_gate_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    phase1_decision = str(metrics.get("phase1_decision") or "unknown").strip().lower()
    row_key_dup_rate = float(metrics.get("row_key_dup_rate") or 0.0)
    date_parse_fail_rate = float(metrics.get("date_parse_fail_rate") or 0.0)
    blank_team_rate = float(metrics.get("blank_team_rate") or 0.0)
    blank_league_rate = float(metrics.get("blank_league_rate") or 0.0)
    year_mismatch_rate = float(metrics.get("year_mismatch_rate") or 0.0)
    missing_required_cols_files = int(metrics.get("missing_required_cols_files") or 0)
    new_canonical_count = int(metrics.get("new_canonical_count") or 0)
    daily_series_gap_suspect_groups = int(metrics.get("daily_series_gap_suspect_groups") or 0)

    decision = "promote"
    reasons: List[str] = []

    if missing_required_cols_files > 0:
        decision = "reject"
        reasons.append("missing_required_columns")
    if date_parse_fail_rate > 0.001:
        decision = "reject"
        reasons.append("date_parse_fail_rate_high")
    if row_key_dup_rate > 0.0:
        decision = "reject"
        reasons.append("duplicate_row_keys")
    if blank_team_rate > 0.001:
        decision = "reject"
        reasons.append("blank_team_rate_high")
    if blank_league_rate > 0.001:
        decision = "reject"
        reasons.append("blank_league_rate_high")
    if year_mismatch_rate > 0.02:
        decision = "reject"
        reasons.append("year_mismatch_rate_high")

    if decision != "reject" and new_canonical_count > 0:
        decision = "shadow_only"
        reasons.append("new_canonical_leagues")
    if decision == "promote" and daily_series_gap_suspect_groups > 0:
        decision = "shadow_only"
        reasons.append("daily_series_gap_suspects")
    elif daily_series_gap_suspect_groups > 0:
        reasons.append("daily_series_gap_suspects")

    if rank_decision(phase1_decision) > rank_decision(decision):
        decision = phase1_decision if phase1_decision in {"promote", "shadow_only", "reject"} else "shadow_only"
        reasons.append(f"phase1:{phase1_decision or 'unknown'}")
    elif phase1_decision not in {"", "promote"}:
        reasons.append(f"phase1:{phase1_decision}")

    if not reasons:
        reasons.append("pass")

    return {
        "decision": decision,
        "reasons": reasons,
        "thresholds": {
            "date_parse_fail_rate_max": 0.001,
            "blank_team_rate_max": 0.001,
            "blank_league_rate_max": 0.001,
            "year_mismatch_rate_max": 0.02,
            "row_key_dup_rate_max": 0.0,
            "new_canonical_count_promote_max": 0,
            "daily_series_gap_suspect_groups_promote_max": 0,
        },
    }
