from __future__ import annotations

from typing import Any, Dict
import re


POS_CANON = ["Top", "Jungle", "Mid", "ADC", "Support"]
POS_CANON_ML = ["Top", "Jungle", "Mid", "Bot", "Support"]
LINEUP_SETTINGS_DEFAULTS = {
    "use_lineup_adjust": False,
    "players_csv_path": "data/oracleselixir_players.csv",
    "lineup_lambda_points": 90,
    "lineup_min_gp": 5,
    "lineup_shrink_m": 10,
    "ml_players_w_lineup": 1.15,
}


def safe_key(s: str) -> str:
    """Convert arbitrary string to a safe session-state key (alphanumeric + underscore, min 'x')."""
    s = str(s or "")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def norm_pos(pos: str) -> str:
    """Normalize a position string to canonical display form (Top/Jungle/Mid/ADC/Support)."""
    p = str(pos or "").strip().lower()
    if not p:
        return ""
    if p in ("top", "toplane"):
        return "Top"
    if p in ("jungle", "jg", "jng", "jgl", "jung", "jngl", "jngle"):
        return "Jungle"
    if p in ("mid", "middle", "midlane"):
        return "Mid"
    if p in ("bot", "bottom", "adc", "ad", "marksman"):
        return "ADC"
    if p in ("sup", "support"):
        return "Support"
    return str(pos or "").strip().title()


def norm_pos_ml(pos: str) -> str:
    """Normalize position to ML-pipeline form (Bot instead of ADC; others same as norm_pos)."""
    p = str(pos or "").strip().lower()
    if not p:
        return ""
    if p in ("top", "toplane"):
        return "Top"
    if p in ("jungle", "jg", "jng", "jgl", "jung", "jngl", "jngle"):
        return "Jungle"
    if p in ("mid", "middle", "midlane"):
        return "Mid"
    if p in ("bot", "bottom", "adc", "ad", "marksman"):
        return "Bot"
    if p in ("sup", "support"):
        return "Support"
    return str(pos or "").strip().title()


def parse_pct(x: Any) -> float:
    """Parse a percentage string ('55%' → 0.55) or decimal string ('0.6' → 0.6). Returns NaN on failure."""
    s = str(x or "").strip()
    if not s:
        return float("nan")
    try:
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        return float(s)
    except Exception:
        return float("nan")


def lineup_store(session_state: Dict[str, Any]) -> dict:
    store = session_state.get("_lineup_store")
    if not isinstance(store, dict):
        store = {}
        session_state["_lineup_store"] = store
    return store


def lineup_key(season_label: str, team_a: str, team_b: str) -> str:
    a = (team_a or "").strip()
    b = (team_b or "").strip()
    s = (season_label or "").strip()
    pair = "|".join(sorted([a, b]))
    return f"{s}::{pair}" if s else pair


def ensure_lineup_runtime_state(session_state: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in LINEUP_SETTINGS_DEFAULTS.items():
        if key not in session_state:
            session_state[key] = value
    return {
        "use_lineup_adjust": bool(session_state.get("use_lineup_adjust", LINEUP_SETTINGS_DEFAULTS["use_lineup_adjust"])),
        "players_csv_path": str(session_state.get("players_csv_path", LINEUP_SETTINGS_DEFAULTS["players_csv_path"]) or LINEUP_SETTINGS_DEFAULTS["players_csv_path"]),
        "lineup_lambda_points": int(session_state.get("lineup_lambda_points", LINEUP_SETTINGS_DEFAULTS["lineup_lambda_points"]) or LINEUP_SETTINGS_DEFAULTS["lineup_lambda_points"]),
        "lineup_min_gp": int(session_state.get("lineup_min_gp", LINEUP_SETTINGS_DEFAULTS["lineup_min_gp"]) or LINEUP_SETTINGS_DEFAULTS["lineup_min_gp"]),
        "lineup_shrink_m": int(session_state.get("lineup_shrink_m", LINEUP_SETTINGS_DEFAULTS["lineup_shrink_m"]) or LINEUP_SETTINGS_DEFAULTS["lineup_shrink_m"]),
        "ml_players_w_lineup": float(session_state.get("ml_players_w_lineup", LINEUP_SETTINGS_DEFAULTS["ml_players_w_lineup"]) or LINEUP_SETTINGS_DEFAULTS["ml_players_w_lineup"]),
    }


def get_lineup_runtime_settings(session_state: Dict[str, Any]) -> Dict[str, Any]:
    return ensure_lineup_runtime_state(session_state)


def get_lineup_loaded_key(session_state: Dict[str, Any]) -> str:
    return str(session_state.get("lineup_loaded_key") or "")


def set_lineup_loaded_key(session_state: Dict[str, Any], key: str) -> str:
    value = str(key or "")
    session_state["lineup_loaded_key"] = value
    return value


def load_lineup_for_matchup(session_state: Dict[str, Any], key: str) -> None:
    store = (lineup_store(session_state).get(key) or {})

    ensure_lineup_runtime_state(session_state)

    if "ml_players_w_lineup" in store:
        try:
            session_state["ml_players_w_lineup"] = float(store.get("ml_players_w_lineup", LINEUP_SETTINGS_DEFAULTS["ml_players_w_lineup"]) or LINEUP_SETTINGS_DEFAULTS["ml_players_w_lineup"])
        except Exception:
            session_state["ml_players_w_lineup"] = LINEUP_SETTINGS_DEFAULTS["ml_players_w_lineup"]

    for side in ("A", "B"):
        for pos in POS_CANON:
            k = f"lineup_{side}_{safe_key(pos)}"
            if k not in session_state:
                session_state[k] = ""
            if k in store:
                session_state[k] = str(store.get(k, "") or "")
        for pos in POS_CANON_ML:
            k = f"lineup_ml_{side}_{safe_key(pos)}"
            if k not in session_state:
                session_state[k] = ""
            if k in store:
                session_state[k] = str(store.get(k, "") or "")


def save_lineup_for_matchup(session_state: Dict[str, Any], key: str) -> None:
    store = lineup_store(session_state)
    payload: Dict[str, Any] = {}
    settings = get_lineup_runtime_settings(session_state)
    for side in ("A", "B"):
        for pos in POS_CANON:
            k = f"lineup_{side}_{safe_key(pos)}"
            payload[k] = session_state.get(k, "")
        for pos in POS_CANON_ML:
            k = f"lineup_ml_{side}_{safe_key(pos)}"
            payload[k] = session_state.get(k, "")

    payload.update(settings)
    store[key] = payload


def get_lineup_selected(session_state: Dict[str, Any], side: str, *, mode: str = "manual") -> dict:
    out: Dict[str, str] = {}
    if mode == "ml":
        prefix = "lineup_ml"
        pos_list = POS_CANON_ML
    else:
        prefix = "lineup"
        pos_list = POS_CANON
    for pos in pos_list:
        k = f"{prefix}_{side}_{safe_key(pos)}"
        v = str(session_state.get(k, "") or "").strip()
        if v:
            out[pos] = v
    return out
