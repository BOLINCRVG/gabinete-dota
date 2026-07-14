from __future__ import annotations

from typing import Any, Dict, Optional


def _as_dict(v: Any) -> Dict[str, Any]:
    return dict(v) if isinstance(v, dict) else {}


def resolve_odds_fixture_snapshot(payload: Dict[str, Any], artifact_sha256: str) -> Optional[Dict[str, Any]]:
    """Resolve an odds fixture snapshot for the current artifact hash.

    Supports legacy single-snapshot payloads and versioned payloads with
    `snapshots[artifact_sha256]`.
    """
    data = _as_dict(payload)
    sha = str(artifact_sha256 or "").strip()
    if not sha:
        return None

    snaps = _as_dict(data.get("snapshots"))
    snap = _as_dict(snaps.get(sha))
    if snap and isinstance(snap.get("matchups"), list):
        out = dict(snap)
        out.setdefault("artifact_sha256", sha)
        return out

    # Legacy format fallback.
    if str(data.get("artifact_sha256") or "").strip() == sha and isinstance(data.get("matchups"), list):
        return data
    return None


def upsert_odds_fixture_snapshot(payload: Dict[str, Any], artifact_sha256: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Store snapshot under artifact_sha256 key; also updates legacy top-level fields."""
    data = _as_dict(payload)
    sha = str(artifact_sha256 or "").strip()
    snap = _as_dict(snapshot)
    if not sha or not isinstance(snap.get("matchups"), list):
        return data

    snaps = _as_dict(data.get("snapshots"))
    snap["artifact_sha256"] = sha
    snaps[sha] = snap
    data["snapshots"] = snaps
    data["fixture_schema"] = "artifact_snapshot_v1"
    data["latest_artifact_sha256"] = sha

    # Keep legacy top-level keys for compatibility with old tooling.
    for k in ("artifact_file", "artifact_sha256", "generated_at", "method", "matchups"):
        if k in snap:
            data[k] = snap[k]
    data["artifact_sha256"] = sha
    return data


def build_players_fixture_key(
    artifact_sha256: str,
    players_artifact_sha256: str,
    csv_sha256: str,
) -> str:
    """Build a composite lookup key from the three SHA256 hashes that uniquely identify a snapshot."""
    a = str(artifact_sha256 or "").strip()
    p = str(players_artifact_sha256 or "").strip()
    c = str(csv_sha256 or "").strip()
    return f"a={a}|p={p}|c={c}"


def resolve_players_fixture_snapshot(
    payload: Dict[str, Any],
    artifact_sha256: str,
    players_artifact_sha256: str,
    csv_sha256: str,
) -> Optional[Dict[str, Any]]:
    """Resolve players+fusion snapshot for the triple (artifact, players, csv)."""
    data = _as_dict(payload)
    key = build_players_fixture_key(artifact_sha256, players_artifact_sha256, csv_sha256)

    snaps = _as_dict(data.get("snapshots"))
    snap = _as_dict(snaps.get(key))
    if snap and isinstance(snap.get("matchups"), list):
        out = dict(snap)
        out.setdefault("artifact_sha256", str(artifact_sha256 or "").strip())
        out.setdefault("players_artifact_sha256", str(players_artifact_sha256 or "").strip())
        out.setdefault("csv_sha256", str(csv_sha256 or "").strip())
        return out

    # Legacy format fallback.
    if (
        str(data.get("artifact_sha256") or "").strip() == str(artifact_sha256 or "").strip()
        and str(data.get("players_artifact_sha256") or "").strip() == str(players_artifact_sha256 or "").strip()
        and str(data.get("csv_sha256") or "").strip() == str(csv_sha256 or "").strip()
        and isinstance(data.get("matchups"), list)
    ):
        return data
    return None


def upsert_players_fixture_snapshot(
    payload: Dict[str, Any],
    artifact_sha256: str,
    players_artifact_sha256: str,
    csv_sha256: str,
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Store a players+fusion snapshot keyed by the composite triple (artifact, players, csv)."""
    data = _as_dict(payload)
    snap = _as_dict(snapshot)
    if not isinstance(snap.get("matchups"), list):
        return data

    key = build_players_fixture_key(artifact_sha256, players_artifact_sha256, csv_sha256)
    snaps = _as_dict(data.get("snapshots"))
    snap["artifact_sha256"] = str(artifact_sha256 or "").strip()
    snap["players_artifact_sha256"] = str(players_artifact_sha256 or "").strip()
    snap["csv_sha256"] = str(csv_sha256 or "").strip()
    snaps[key] = snap

    data["snapshots"] = snaps
    data["fixture_schema"] = "artifact_players_csv_snapshot_v1"
    data["latest_snapshot_key"] = key

    # Legacy top-level keys for compatibility.
    for k in (
        "artifact_sha256",
        "players_artifact_sha256",
        "csv_sha256",
        "generated_at",
        "method",
        "matchups",
    ):
        if k in snap:
            data[k] = snap[k]
    return data

