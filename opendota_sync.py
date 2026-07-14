from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

from dota_core import app_dir, csv_path
from opendota_client import get_league_matches, get_leagues, get_match, get_pro_matches


TEAM_RADIANT = 2
TEAM_DIRE = 3
SYNC_STATE_FILE = "opendota_sync_state.json"
SYNC_META_COLUMNS = {
    "league_id": pd.NA,
    "start_time": pd.NA,
    "has_objectives": pd.NA,
    "objective_events": pd.NA,
    "data_quality": "legacy",
    "quality_flags": "",
    "source_mode": "legacy_csv",
}


def sync_state_path() -> Path:
    return app_dir() / SYNC_STATE_FILE


def load_sync_state(path: Optional[Path] = None) -> dict:
    target = Path(path) if path else sync_state_path()
    if not target.exists():
        return {}
    try:
        return pd.read_json(target, typ="series").to_dict()
    except Exception:
        return {}


def save_sync_state(report: dict, path: Optional[Path] = None) -> None:
    target = Path(path) if path else sync_state_path()
    payload = dict(report)
    payload["saved_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pd.Series(payload).to_json(target, force_ascii=False, indent=2)


def _ensure_sync_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        for col, default in SYNC_META_COLUMNS.items():
            if col not in frame.columns:
                frame[col] = default
        return frame
    for col, default in SYNC_META_COLUMNS.items():
        if col not in frame.columns:
            frame[col] = default
    frame["data_quality"] = frame["data_quality"].fillna("legacy").astype(str)
    frame["quality_flags"] = frame["quality_flags"].fillna("").astype(str)
    frame["source_mode"] = frame["source_mode"].fillna("legacy_csv").astype(str)
    return frame


def _match_datetime_label(payload: dict) -> pd.Timestamp:
    start_time = int(payload.get("start_time") or 0)
    if start_time <= 0:
        return pd.Timestamp.utcnow()
    return pd.to_datetime(start_time, unit="s", utc=True)


def _league_name(payload: dict) -> str:
    league = payload.get("league")
    if isinstance(league, dict):
        name = str(league.get("name") or "").strip()
        if name:
            return name
    name = str(payload.get("league_name") or "").strip()
    return name or "Sem campeonato"


def _count_objectives(payload: dict) -> Dict[str, int]:
    radiant_towers = 0
    dire_towers = 0
    radiant_barracks = 0
    dire_barracks = 0
    radiant_roshans = 0
    dire_roshans = 0

    # "Primeiros" (mercados First X): 1.0 = radiant conseguiu, 0.0 = dire,
    # None = evento nao ocorreu/base antiga (NUNCA 0 falso).
    first_blood_radiant = None
    first_tower_radiant = None
    first_roshan_radiant = None

    for row in payload.get("objectives") or []:
        kind = str(row.get("type") or "")
        key = str(row.get("key") or "")
        if kind == "building_kill":
            lower = key.lower()
            if "_tower" in lower:
                if "badguys" in lower:
                    radiant_towers += 1
                    if first_tower_radiant is None:
                        first_tower_radiant = 1.0
                elif "goodguys" in lower:
                    dire_towers += 1
                    if first_tower_radiant is None:
                        first_tower_radiant = 0.0
            elif "rax" in lower:
                if "badguys" in lower:
                    radiant_barracks += 1
                elif "goodguys" in lower:
                    dire_barracks += 1
        elif kind == "CHAT_MESSAGE_ROSHAN_KILL":
            team = int(row.get("team") or 0)
            if team == TEAM_RADIANT:
                radiant_roshans += 1
                if first_roshan_radiant is None:
                    first_roshan_radiant = 1.0
            elif team == TEAM_DIRE:
                dire_roshans += 1
                if first_roshan_radiant is None:
                    first_roshan_radiant = 0.0
        elif kind == "CHAT_MESSAGE_FIRSTBLOOD" and first_blood_radiant is None:
            try:
                slot = row.get("player_slot")
                if slot is None:
                    slot = row.get("slot")
                if slot is not None:
                    first_blood_radiant = 1.0 if int(slot) < 128 else 0.0
            except (TypeError, ValueError):
                pass

    return {
        "radiant_towers": radiant_towers,
        "dire_towers": dire_towers,
        "radiant_barracks": radiant_barracks,
        "dire_barracks": dire_barracks,
        "radiant_roshans": radiant_roshans,
        "dire_roshans": dire_roshans,
        "first_blood_radiant": first_blood_radiant,
        "first_tower_radiant": first_tower_radiant,
        "first_roshan_radiant": first_roshan_radiant,
    }


def _quality_meta(payload: dict) -> dict:
    flags: List[str] = []
    objectives = payload.get("objectives") or []
    if not isinstance(objectives, list) or not objectives:
        flags.append("missing_objectives")
    duration = int(payload.get("duration") or 0)
    if duration <= 0:
        flags.append("missing_duration")
    if payload.get("radiant_score") is None or payload.get("dire_score") is None:
        flags.append("missing_score")
    radiant_team_id = int(payload.get("radiant_team_id") or 0)
    dire_team_id = int(payload.get("dire_team_id") or 0)
    if radiant_team_id <= 0 or dire_team_id <= 0:
        flags.append("missing_team_id")
    radiant_name = str(payload.get("radiant_name") or payload.get("radiant_team", {}).get("name") or "").strip()
    dire_name = str(payload.get("dire_name") or payload.get("dire_team", {}).get("name") or "").strip()
    if not radiant_name or radiant_name.lower() == "radiant":
        flags.append("placeholder_radiant_name")
    if not dire_name or dire_name.lower() == "dire":
        flags.append("placeholder_dire_name")
    league_id = int(payload.get("leagueid") or payload.get("league_id") or 0)
    if league_id <= 0:
        flags.append("missing_league_id")
    return {
        "league_id": league_id,
        "start_time": int(payload.get("start_time") or 0),
        "has_objectives": bool(isinstance(objectives, list) and objectives),
        "objective_events": int(len(objectives) if isinstance(objectives, list) else 0),
        "data_quality": "ok" if not flags else "partial",
        "quality_flags": ",".join(flags),
        "source_mode": "opendota_api",
    }


def _bool_series(series: pd.Series) -> pd.Series:
    lowered = series.astype("string").fillna("").str.strip().str.lower()
    return lowered.isin(["true", "1", "yes", "y"])


def csv_health_report(path: Optional[Path] = None, df: Optional[pd.DataFrame] = None) -> dict:
    target = Path(path) if path else csv_path()
    frame = df.copy() if df is not None else load_existing_csv(target)
    report = {
        "csv_path": str(target),
        "csv_exists": bool(target.exists()),
        "total_rows": 0,
        "total_matches": 0,
        "latest_match_date": "",
        "latest_match_age_hours": None,
        "partial_rows": 0,
        "partial_matches": 0,
        "legacy_rows": 0,
        "legacy_matches": 0,
        "matches_missing_objectives": 0,
        "rows_missing_objectives": 0,
    }
    if frame.empty:
        return report

    report["total_rows"] = int(len(frame))
    if "match_id" in frame.columns:
        report["total_matches"] = int(pd.to_numeric(frame["match_id"], errors="coerce").dropna().astype(int).nunique())
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        latest = dates.max()
        if pd.notna(latest):
            report["latest_match_date"] = latest.strftime("%Y-%m-%dT%H:%M:%SZ")
            age_hours = (pd.Timestamp.utcnow() - latest).total_seconds() / 3600.0
            report["latest_match_age_hours"] = round(float(age_hours), 2)
    if "data_quality" in frame.columns:
        quality = frame["data_quality"].fillna("legacy").astype(str).str.strip().str.lower()
        partial_mask = quality == "partial"
        legacy_mask = quality == "legacy"
        report["partial_rows"] = int(partial_mask.sum())
        report["legacy_rows"] = int(legacy_mask.sum())
        if "match_id" in frame.columns and partial_mask.any():
            report["partial_matches"] = int(
                pd.to_numeric(frame.loc[partial_mask, "match_id"], errors="coerce").dropna().astype(int).nunique()
            )
        if "match_id" in frame.columns and legacy_mask.any():
            report["legacy_matches"] = int(
                pd.to_numeric(frame.loc[legacy_mask, "match_id"], errors="coerce").dropna().astype(int).nunique()
            )
    if "has_objectives" in frame.columns:
        known_objectives_mask = frame["has_objectives"].notna()
        missing_objectives_mask = known_objectives_mask & (~_bool_series(frame["has_objectives"]))
        report["rows_missing_objectives"] = int(missing_objectives_mask.sum())
        if "match_id" in frame.columns and missing_objectives_mask.any():
            report["matches_missing_objectives"] = int(
                pd.to_numeric(frame.loc[missing_objectives_mask, "match_id"], errors="coerce").dropna().astype(int).nunique()
            )
    return report


RAW_SIDECAR_FILE = "raw_matches.jsonl"


def _append_raw_sidecar(payload: dict) -> None:
    """Guarda players/draft/patch de todo fetch completo (base do P3).

    Decisao 2026-07-09 ("mesma qualidade p/ ambos os modelos"): antes esses
    campos eram descartados; agora todo /matches/{id} acumula um registro
    enxuto em raw_matches.jsonl — as camadas de players e herois nascem com
    historico sem refetch. Consumidores dedupam por match_id.
    """
    try:
        import json as _json
        match_id = int(payload.get("match_id") or 0)
        if match_id <= 0 or not (payload.get("players") or payload.get("picks_bans")):
            return
        players = []
        for p in (payload.get("players") or []):
            players.append({
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "isRadiant": bool(p.get("isRadiant", (p.get("player_slot") or 0) < 128)),
                "kills": p.get("kills"), "deaths": p.get("deaths"),
                "assists": p.get("assists"), "gold_per_min": p.get("gold_per_min"),
                "xp_per_min": p.get("xp_per_min"), "lane_role": p.get("lane_role"),
            })
        rec = {
            "match_id": match_id,
            "start_time": int(payload.get("start_time") or 0),
            "patch": payload.get("patch"),
            "league_id": int(payload.get("leagueid") or payload.get("league_id") or 0),
            "radiant_team_id": int(payload.get("radiant_team_id") or 0),
            "dire_team_id": int(payload.get("dire_team_id") or 0),
            "picks_bans": payload.get("picks_bans") or [],
            "players": players,
        }
        target = app_dir() / RAW_SIDECAR_FILE
        with target.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # sidecar nunca pode derrubar o sync


def match_to_rows(payload: dict) -> List[dict]:
    match_id = int(payload.get("match_id") or 0)
    if match_id <= 0:
        return []
    _append_raw_sidecar(payload)

    dt = _match_datetime_label(payload)
    league = _league_name(payload)
    counts = _count_objectives(payload)
    quality = _quality_meta(payload)
    radiant_win = bool(payload.get("radiant_win"))
    duration = int(payload.get("duration") or 0)

    radiant_team_id = int(payload.get("radiant_team_id") or 0)
    dire_team_id = int(payload.get("dire_team_id") or 0)
    radiant_name = str(payload.get("radiant_name") or payload.get("radiant_team", {}).get("name") or "Radiant").strip()
    dire_name = str(payload.get("dire_name") or payload.get("dire_team", {}).get("name") or "Dire").strip()
    radiant_score = int(payload.get("radiant_score") or 0)
    dire_score = int(payload.get("dire_score") or 0)

    base = {
        "match_id": match_id,
        "date": dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "year": int(dt.year),
        "league": league,
        "duration": duration,
        **quality,
    }

    radiant_row = {
        **base,
        "team_id": radiant_team_id,
        "teamname": radiant_name,
        "opp_team_id": dire_team_id,
        "opp_teamname": dire_name,
        "side": "radiant",
        "result": 1 if radiant_win else 0,
        "teamkills": radiant_score,
        "oppkills": dire_score,
        "towers": counts["radiant_towers"],
        "opp_towers": counts["dire_towers"],
        "roshans": counts["radiant_roshans"],
        "opp_roshans": counts["dire_roshans"],
        "barracks": counts["radiant_barracks"],
        "opp_barracks": counts["dire_barracks"],
        "first_blood": counts.get("first_blood_radiant"),
        "first_tower": counts.get("first_tower_radiant"),
        "first_roshan": counts.get("first_roshan_radiant"),
        # marcador de schema: esta linha JA passou pela extracao dos
        # "primeiros" (mesmo que o evento nao exista no jogo) — evita a fila
        # de re-passada re-baixar o mesmo jogo para sempre.
        "firsts_pass": 1,
    }
    dire_row = {
        **base,
        "team_id": dire_team_id,
        "teamname": dire_name,
        "opp_team_id": radiant_team_id,
        "opp_teamname": radiant_name,
        "side": "dire",
        "result": 0 if radiant_win else 1,
        "teamkills": dire_score,
        "oppkills": radiant_score,
        "towers": counts["dire_towers"],
        "opp_towers": counts["radiant_towers"],
        "roshans": counts["dire_roshans"],
        "opp_roshans": counts["radiant_roshans"],
        "barracks": counts["dire_barracks"],
        "opp_barracks": counts["radiant_barracks"],
        "first_blood": (None if counts.get("first_blood_radiant") is None
                        else 1.0 - float(counts["first_blood_radiant"])),
        "first_tower": (None if counts.get("first_tower_radiant") is None
                        else 1.0 - float(counts["first_tower_radiant"])),
        "first_roshan": (None if counts.get("first_roshan_radiant") is None
                         else 1.0 - float(counts["first_roshan_radiant"])),
        "firsts_pass": 1,
    }
    return [radiant_row, dire_row]


def load_existing_csv(path: Optional[Path] = None) -> pd.DataFrame:
    target = Path(path) if path else csv_path()
    if not target.exists():
        return pd.DataFrame()
    return _ensure_sync_columns(pd.read_csv(target, low_memory=False))


def _merge_rows(existing: pd.DataFrame, new_rows: List[dict], target: Path) -> pd.DataFrame:
    combined = _ensure_sync_columns(existing) if not existing.empty else _ensure_sync_columns(pd.DataFrame())
    if new_rows:
        fresh = _ensure_sync_columns(pd.DataFrame(new_rows))
        combined = pd.concat([combined, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id", "team_id", "opp_team_id", "side"], keep="last")
        combined = combined.sort_values(["date", "match_id", "teamname"], ascending=[False, False, True]).reset_index(drop=True)
        # Escrita ATOMICA (tmp + os.replace): leitores concorrentes (adapter,
        # app) veem o arquivo velho ou o novo, NUNCA um truncado. Corrida real
        # vista em 2026-07-10: adapter leu data.csv no meio de um to_csv da
        # esteira e gerou CSV do app pela metade.
        import os as _os
        tmp = target.with_suffix(".csv.tmp")
        combined.to_csv(tmp, index=False)
        _os.replace(tmp, target)
    return combined


def _seen_match_ids(existing: pd.DataFrame) -> Set[int]:
    if existing.empty or "match_id" not in existing.columns:
        return set()
    return {int(x) for x in pd.to_numeric(existing["match_id"], errors="coerce").dropna().astype(int).tolist()}


def professional_league_ids() -> Set[int]:
    leagues = get_leagues("professional")
    out: Set[int] = set()
    for row in leagues:
        try:
            league_id = int((row or {}).get("leagueid") or 0)
        except Exception:
            league_id = 0
        if league_id > 0:
            out.add(league_id)
    return out


def _item_start_time(item: dict) -> int:
    try:
        return int((item or {}).get("start_time") or 0)
    except Exception:
        return 0


def _merge_report_health(report: dict, df: pd.DataFrame, target: Path) -> dict:
    health = csv_health_report(target, df=df)
    merged = dict(report)
    merged.update(health)
    return merged


def _legacy_match_ids(df: pd.DataFrame) -> List[int]:
    if df.empty or "match_id" not in df.columns or "data_quality" not in df.columns:
        return []
    quality = df["data_quality"].fillna("legacy").astype(str).str.strip().str.lower()
    frame = df.loc[quality == "legacy", ["match_id", "date"]].copy()
    if frame.empty:
        return []
    frame["match_id"] = pd.to_numeric(frame["match_id"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["match_id"]).copy()
    frame = frame.sort_values(["date", "match_id"], ascending=[False, False]).drop_duplicates(subset=["match_id"])
    return frame["match_id"].astype(int).tolist()


def sync_recent_pro_matches(path: Optional[Path] = None, limit: int = 100, backfill: bool = False, professional_only: bool = True, league_sweep: bool = True, max_seconds: int = 90) -> dict:
    _start = time.monotonic()
    target = Path(path) if path else csv_path()
    existing = load_existing_csv(target)
    seen_matches = _seen_match_ids(existing)
    existing_health = csv_health_report(target, df=existing)
    # buffer de 50% para compensar filtragem, sem explodir para 3x
    source_limit = min(max(int(limit) + int(limit // 2), 50), 300)
    pro_matches = get_pro_matches(source_limit)
    allowed_leagues = professional_league_ids() if professional_only else set()
    filtered_out_ids: List[int] = []
    fetched_ids: List[int] = []
    skipped_ids: List[int] = []
    new_rows: List[dict] = []
    failed_leagues: List[int] = []
    recent_candidates: List[dict] = []
    recent_candidate_ids: Set[int] = set()

    for item in pro_matches:
        match_id = int(item.get("match_id") or 0)
        if match_id <= 0:
            continue
        league_id = int(item.get("leagueid") or 0)
        if professional_only and league_id not in allowed_leagues:
            filtered_out_ids.append(match_id)
            continue
        recent_candidates.append(item)
        recent_candidate_ids.add(match_id)

    recent_candidates = sorted(recent_candidates, key=_item_start_time, reverse=True)
    recent_cutoff_ts = min([_item_start_time(item) for item in recent_candidates if _item_start_time(item) > 0], default=0)

    for item in recent_candidates:
        if time.monotonic() - _start > max_seconds:
            break
        match_id = int(item.get("match_id") or 0)
        if match_id <= 0:
            continue
        if not backfill and match_id in seen_matches:
            skipped_ids.append(match_id)
            continue
        try:
            payload = get_match(match_id)
        except Exception:
            continue
        rows = match_to_rows(payload)
        if not rows:
            skipped_ids.append(match_id)
            continue
        new_rows.extend(rows)
        fetched_ids.append(match_id)
        seen_matches.add(match_id)

    league_sweep_scanned = 0
    league_sweep_candidates = 0
    league_sweep_added = 0
    if professional_only and league_sweep and (time.monotonic() - _start) < max_seconds:
        leagues = get_leagues("professional")
        leagues = [row for row in leagues if int((row or {}).get("leagueid") or 0) > 0]
        leagues = sorted(leagues, key=lambda row: int((row or {}).get("leagueid") or 0), reverse=True)
        league_limit = 25 if int(limit) <= 50 else 50
        for row in leagues[:league_limit]:
            if time.monotonic() - _start > max_seconds:
                break
            league_id = int((row or {}).get("leagueid") or 0)
            if league_id <= 0:
                continue
            league_sweep_scanned += 1
            try:
                matches = get_league_matches(league_id)
            except Exception:
                failed_leagues.append(league_id)
                continue
            matches = sorted(matches, key=_item_start_time, reverse=True)
            for item in matches[:100]:
                if time.monotonic() - _start > max_seconds:
                    break
                match_id = int((item or {}).get("match_id") or 0)
                if match_id <= 0 or match_id in recent_candidate_ids:
                    continue
                start_ts = _item_start_time(item)
                if recent_cutoff_ts > 0 and start_ts > 0 and start_ts < recent_cutoff_ts:
                    continue
                league_sweep_candidates += 1
                if not backfill and match_id in seen_matches:
                    skipped_ids.append(match_id)
                    continue
                try:
                    payload = get_match(match_id)
                except Exception:
                    continue
                rows = match_to_rows(payload)
                if not rows:
                    skipped_ids.append(match_id)
                    continue
                new_rows.extend(rows)
                fetched_ids.append(match_id)
                seen_matches.add(match_id)
                league_sweep_added += 1

    combined = _merge_rows(existing, new_rows, target)

    report = _merge_report_health(
        {
        "path": str(target),
        "mode": "recent",
        "coverage_mode": "proMatches + league_sweep",
        "backfill": bool(backfill),
        "professional_only": bool(professional_only),
        "requested_limit": int(limit),
        "source_limit": int(source_limit),
        "recent_candidates": int(len(recent_candidates)),
        "recent_cutoff_utc": pd.to_datetime(recent_cutoff_ts, unit="s", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ") if recent_cutoff_ts > 0 else "",
        "league_sweep_scanned": int(league_sweep_scanned),
        "league_sweep_candidates": int(league_sweep_candidates),
        "league_sweep_added": int(league_sweep_added),
        "failed_leagues": failed_leagues[:25],
        "fetched_matches": len(fetched_ids),
        "skipped_matches": len(skipped_ids),
        "filtered_out_matches": len(filtered_out_ids),
        "new_rows": len(new_rows),
        "latest_match_ids": fetched_ids[:10],
        "elapsed_seconds": round(time.monotonic() - _start, 1),
        "latest_match_date_before": existing_health.get("latest_match_date", ""),
        "latest_match_age_hours_before": existing_health.get("latest_match_age_hours"),
    },
        combined,
        target,
    )
    save_sync_state(report)
    return report


def sync_professional_leagues(
    path: Optional[Path] = None,
    league_limit: Optional[int] = None,
    matches_per_league: int = 50,
    backfill: bool = False,
) -> dict:
    target = Path(path) if path else csv_path()
    existing = load_existing_csv(target)
    seen_matches = _seen_match_ids(existing)
    existing_health = csv_health_report(target, df=existing)
    leagues = get_leagues("professional")
    leagues = [row for row in leagues if int((row or {}).get("leagueid") or 0) > 0]
    leagues = sorted(leagues, key=lambda row: int((row or {}).get("leagueid") or 0), reverse=True)
    if league_limit:
        leagues = leagues[: max(1, int(league_limit))]

    fetched_ids: List[int] = []
    skipped_ids: List[int] = []
    failed_leagues: List[int] = []
    new_rows: List[dict] = []
    leagues_scanned = 0

    for row in leagues:
        league_id = int((row or {}).get("leagueid") or 0)
        if league_id <= 0:
            continue
        leagues_scanned += 1
        try:
            matches = get_league_matches(league_id)
        except Exception:
            failed_leagues.append(league_id)
            continue
        matches = sorted(matches, key=_item_start_time, reverse=True)
        if matches_per_league and matches_per_league > 0:
            matches = matches[: int(matches_per_league)]
        for item in matches:
            match_id = int((item or {}).get("match_id") or 0)
            if match_id <= 0:
                continue
            if not backfill and match_id in seen_matches:
                skipped_ids.append(match_id)
                continue
            payload = get_match(match_id)
            rows = match_to_rows(payload)
            if not rows:
                skipped_ids.append(match_id)
                continue
            new_rows.extend(rows)
            fetched_ids.append(match_id)
            seen_matches.add(match_id)

    combined = _merge_rows(existing, new_rows, target)
    report = _merge_report_health(
        {
            "path": str(target),
            "mode": "professional_leagues",
            "backfill": bool(backfill),
            "league_limit": int(league_limit or 0),
            "matches_per_league": int(matches_per_league),
            "leagues_scanned": leagues_scanned,
            "failed_leagues": failed_leagues[:25],
            "fetched_matches": len(fetched_ids),
            "skipped_matches": len(skipped_ids),
            "new_rows": len(new_rows),
            "latest_match_ids": fetched_ids[:10],
            "latest_match_date_before": existing_health.get("latest_match_date", ""),
            "latest_match_age_hours_before": existing_health.get("latest_match_age_hours"),
        },
        combined,
        target,
    )
    save_sync_state(report)
    return report


def import_match_ids(match_ids: List[int], path: Optional[Path] = None, overwrite: bool = False) -> dict:
    target = Path(path) if path else csv_path()
    existing = load_existing_csv(target)
    existing_health = csv_health_report(target, df=existing)
    seen_matches = set()
    if not existing.empty and "match_id" in existing.columns:
        seen_matches = {int(x) for x in pd.to_numeric(existing["match_id"], errors="coerce").dropna().astype(int).tolist()}

    fetched_ids: List[int] = []
    skipped_ids: List[int] = []
    invalid_ids: List[int] = []
    new_rows: List[dict] = []

    for raw in match_ids:
        try:
            match_id = int(raw)
        except Exception:
            continue
        if match_id <= 0:
            invalid_ids.append(match_id)
            continue
        if (not overwrite) and match_id in seen_matches:
            skipped_ids.append(match_id)
            continue
        payload = get_match(match_id)
        rows = match_to_rows(payload)
        if not rows:
            invalid_ids.append(match_id)
            continue
        new_rows.extend(rows)
        fetched_ids.append(match_id)

    combined = _merge_rows(existing, new_rows, target)
    report = _merge_report_health(
        {
            "path": str(target),
            "mode": "manual",
            "overwrite": bool(overwrite),
            "imported_matches": len(fetched_ids),
            "skipped_matches": len(skipped_ids),
            "invalid_matches": invalid_ids,
            "new_rows": len(new_rows),
            "match_ids": fetched_ids,
            "latest_match_date_before": existing_health.get("latest_match_date", ""),
            "latest_match_age_hours_before": existing_health.get("latest_match_age_hours"),
        },
        combined,
        target,
    )
    save_sync_state(report)
    return report


def backfill_legacy_matches(path: Optional[Path] = None, limit: Optional[int] = 100) -> dict:
    target = Path(path) if path else csv_path()
    existing = load_existing_csv(target)
    existing_health = csv_health_report(target, df=existing)
    candidates = _legacy_match_ids(existing)
    if limit is not None and int(limit) > 0:
        candidates = candidates[: int(limit)]

    fetched_ids: List[int] = []
    failed_ids: List[int] = []
    new_rows: List[dict] = []
    rate_limited = False

    for match_id in candidates:
        try:
            payload = get_match(int(match_id))
            rows = match_to_rows(payload)
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if int(status or 0) == 429:
                rate_limited = True
                break
            failed_ids.append(int(match_id))
            continue
        except Exception:
            failed_ids.append(int(match_id))
            continue
        if not rows:
            failed_ids.append(int(match_id))
            continue
        new_rows.extend(rows)
        fetched_ids.append(int(match_id))

    combined = _merge_rows(existing, new_rows, target)
    report = _merge_report_health(
        {
            "path": str(target),
            "mode": "backfill_legacy",
            "requested_matches": int(len(candidates)),
            "backfilled_matches": int(len(fetched_ids)),
            "rate_limited": bool(rate_limited),
            "failed_matches": failed_ids[:100],
            "new_rows": int(len(new_rows)),
            "latest_match_ids": fetched_ids[:10],
            "legacy_matches_before": existing_health.get("legacy_matches", 0),
            "legacy_rows_before": existing_health.get("legacy_rows", 0),
            "latest_match_date_before": existing_health.get("latest_match_date", ""),
            "latest_match_age_hours_before": existing_health.get("latest_match_age_hours"),
        },
        combined,
        target,
    )
    save_sync_state(report)
    return report
