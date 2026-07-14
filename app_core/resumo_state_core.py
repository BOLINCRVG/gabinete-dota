"""State helpers for Resumo tab filters, presets, and runtime view state."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping


_FILTER_DEFAULTS: dict[str, Any] = {
    "resumo_max_odd": 2.0,
    "resumo_req_liga": True,
    "resumo_req_h2h": True,
    "resumo_req_model": True,
    "resumo_min_sample_liga": 8,
    "resumo_min_sample_h2h": 3,
}

_VIEW_DEFAULTS: dict[str, Any] = {
    "resumo_show_laplace_cols": False,
    "resumo_sc_thr": 60,
    "resumo_sc_use": True,
    # Default = Média Liga+H2H 5 (a análise que o usuário considera mais certeira).
    # DEVE bater com o índice 0 das opções e o default-set do selectbox em
    # resumo_tab_core, senão a HIST "pula" no 1º render (reset do widget). Se o
    # H2H do recorte não existir, laplace_pick_odd cai na Liga sozinha.
    "resumo_laplace_pick": "Média Liga+H2H 5",
    # Pesos do Mix: usuário acerta mais seguindo Histórico → Hist tem peso dominante.
    "resumo_mix_w_model": 25.0,
    "resumo_mix_w_laplace": 55.0,
    "resumo_mix_w_sc": 20.0,
    "resumo_combo_preset": "Neutro",
    "resumo_combo_use_mismatch": False,
    "resumo_combo_hist_w": 0.45,
    "resumo_combo_style_w": 0.18,
    "resumo_combo_form_w": 0.16,
}

_COMBO_PRESET_MAP: dict[str, tuple[float, float]] = {
    "Conservador": (0.30, 0.10),
    "Neutro": (0.45, 0.18),
    "Agressivo": (0.65, 0.30),
}


def _filter_snapshot(session_state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "resumo_max_odd": float(session_state.get("resumo_max_odd", _FILTER_DEFAULTS["resumo_max_odd"]) or _FILTER_DEFAULTS["resumo_max_odd"]),
        "resumo_req_liga": bool(session_state.get("resumo_req_liga", _FILTER_DEFAULTS["resumo_req_liga"])),
        "resumo_req_h2h": bool(session_state.get("resumo_req_h2h", _FILTER_DEFAULTS["resumo_req_h2h"])),
        "resumo_req_model": bool(session_state.get("resumo_req_model", _FILTER_DEFAULTS["resumo_req_model"])),
        "resumo_min_sample_liga": int(session_state.get("resumo_min_sample_liga", _FILTER_DEFAULTS["resumo_min_sample_liga"]) or _FILTER_DEFAULTS["resumo_min_sample_liga"]),
        "resumo_min_sample_h2h": int(session_state.get("resumo_min_sample_h2h", _FILTER_DEFAULTS["resumo_min_sample_h2h"]) or _FILTER_DEFAULTS["resumo_min_sample_h2h"]),
    }


def get_resumo_view_state(session_state: Mapping[str, Any]) -> dict[str, Any]:
    """Coleta e retorna todos os parâmetros de filtro e visualização do Resumo da session_state."""
    return {
        "resumo_max_odd": float(session_state.get("resumo_max_odd", _FILTER_DEFAULTS["resumo_max_odd"]) or _FILTER_DEFAULTS["resumo_max_odd"]),
        "resumo_req_liga": bool(session_state.get("resumo_req_liga", _FILTER_DEFAULTS["resumo_req_liga"])),
        "resumo_req_h2h": bool(session_state.get("resumo_req_h2h", _FILTER_DEFAULTS["resumo_req_h2h"])),
        "resumo_req_model": bool(session_state.get("resumo_req_model", _FILTER_DEFAULTS["resumo_req_model"])),
        "resumo_min_sample_liga": int(session_state.get("resumo_min_sample_liga", _FILTER_DEFAULTS["resumo_min_sample_liga"]) or _FILTER_DEFAULTS["resumo_min_sample_liga"]),
        "resumo_min_sample_h2h": int(session_state.get("resumo_min_sample_h2h", _FILTER_DEFAULTS["resumo_min_sample_h2h"]) or _FILTER_DEFAULTS["resumo_min_sample_h2h"]),
        "resumo_show_laplace_cols": bool(session_state.get("resumo_show_laplace_cols", _VIEW_DEFAULTS["resumo_show_laplace_cols"])),
        "resumo_sc_thr": float(session_state.get("resumo_sc_thr", _VIEW_DEFAULTS["resumo_sc_thr"]) or _VIEW_DEFAULTS["resumo_sc_thr"]),
        "resumo_sc_use": bool(session_state.get("resumo_sc_use", _VIEW_DEFAULTS["resumo_sc_use"])),
        "resumo_laplace_pick": str(session_state.get("resumo_laplace_pick", _VIEW_DEFAULTS["resumo_laplace_pick"]) or _VIEW_DEFAULTS["resumo_laplace_pick"]),
        "resumo_mix_w_model": float(session_state.get("resumo_mix_w_model", _VIEW_DEFAULTS["resumo_mix_w_model"]) or _VIEW_DEFAULTS["resumo_mix_w_model"]),
        "resumo_mix_w_laplace": float(session_state.get("resumo_mix_w_laplace", _VIEW_DEFAULTS["resumo_mix_w_laplace"]) or _VIEW_DEFAULTS["resumo_mix_w_laplace"]),
        "resumo_mix_w_sc": float(session_state.get("resumo_mix_w_sc", _VIEW_DEFAULTS["resumo_mix_w_sc"]) or _VIEW_DEFAULTS["resumo_mix_w_sc"]),
        "resumo_combo_preset": str(session_state.get("resumo_combo_preset", _VIEW_DEFAULTS["resumo_combo_preset"]) or _VIEW_DEFAULTS["resumo_combo_preset"]),
        "resumo_combo_use_mismatch": bool(session_state.get("resumo_combo_use_mismatch", _VIEW_DEFAULTS["resumo_combo_use_mismatch"])),
        "resumo_combo_hist_w": float(session_state.get("resumo_combo_hist_w", _VIEW_DEFAULTS["resumo_combo_hist_w"]) or _VIEW_DEFAULTS["resumo_combo_hist_w"]),
        "resumo_combo_style_w": float(session_state.get("resumo_combo_style_w", _VIEW_DEFAULTS["resumo_combo_style_w"]) or _VIEW_DEFAULTS["resumo_combo_style_w"]),
        "resumo_combo_form_w": float(session_state.get("resumo_combo_form_w", _VIEW_DEFAULTS["resumo_combo_form_w"]) or _VIEW_DEFAULTS["resumo_combo_form_w"]),
        "combo_kills_text": str(session_state.get("combo_kills_text", "") or ""),
        "combo_time_text": str(session_state.get("combo_time_text", "") or ""),
        "p_ml_scope": float(session_state.get("_p_ml_scope", float("nan"))),
        "ml_trace_markets": list(session_state.get("_ml_trace_markets", []) or []),
    }


def set_resumo_market_trace(session_state: MutableMapping[str, Any], rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Persiste as linhas de trace de mercados de resumo na session_state."""
    trace_rows = list(rows or [])
    session_state["_ml_trace_markets"] = trace_rows
    return trace_rows


def get_resumo_market_trace(session_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Retorna as linhas de trace de mercados de resumo armazenadas na session_state."""
    return list(session_state.get("_ml_trace_markets", []) or [])


def list_resumo_filter_presets(session_state: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str], str, int]:
    """Lista perfis de filtro salvos, opções de preset e preset ativo da session_state."""
    profiles = {
        str(name): dict(values or {})
        for name, values in dict(session_state.get("resumo_filter_profiles") or {}).items()
        if str(name).strip()
    }
    preset_opts = ["Custom", *sorted(profiles.keys())]
    current = str(session_state.get("resumo_filter_preset", "Custom") or "Custom")
    preset_idx = preset_opts.index(current) if current in preset_opts else 0
    return profiles, preset_opts, current, preset_idx


def apply_resumo_filter_values(
    session_state: MutableMapping[str, Any],
    values: Mapping[str, Any],
    *,
    preset_name: str | None,
) -> bool:
    """Aplica um conjunto de valores de filtro na session_state e registra o preset ativo."""
    applied = False
    for key, value in dict(values or {}).items():
        session_state[str(key)] = value
        applied = True
    if applied:
        selected = str(preset_name or "").strip()
        session_state["resumo_filter_profile_active"] = selected
        session_state["resumo_filter_preset"] = selected or "Custom"
    return applied


def save_resumo_filter_custom(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    """Salva o snapshot atual dos filtros como preset Custom na session_state."""
    payload = _filter_snapshot(session_state)
    session_state["resumo_filter_custom"] = payload
    session_state["resumo_filter_preset"] = "Custom"
    session_state["resumo_filter_profile_active"] = ""
    return payload


def load_resumo_filter_custom(session_state: Mapping[str, Any]) -> dict[str, Any]:
    """Carrega e retorna o preset de filtros Custom salvo na session_state."""
    return dict(session_state.get("resumo_filter_custom") or {})


def save_resumo_filter_profile(session_state: MutableMapping[str, Any], profile_name: str) -> str:
    """Salva o snapshot atual dos filtros como perfil nomeado na session_state e o ativa."""
    name = str(profile_name or "").strip()
    if not name:
        return ""
    profiles = dict(session_state.get("resumo_filter_profiles") or {})
    profiles[name] = _filter_snapshot(session_state)
    session_state["resumo_filter_profiles"] = profiles
    session_state["resumo_filter_preset"] = name
    session_state["resumo_filter_profile_active"] = name
    return name


def delete_active_resumo_filter_profile(session_state: MutableMapping[str, Any]) -> str:
    """Remove o perfil de filtro atualmente ativo da session_state e reverte para Custom."""
    active = str(session_state.get("resumo_filter_preset", "") or "").strip()
    profiles = dict(session_state.get("resumo_filter_profiles") or {})
    if not active or active not in profiles:
        return ""
    profiles.pop(active, None)
    session_state["resumo_filter_profiles"] = profiles
    session_state["resumo_filter_preset"] = "Custom"
    session_state["resumo_filter_profile_active"] = ""
    return active


def ensure_resumo_combo_state(session_state: MutableMapping[str, Any]) -> None:
    """Garante que as chaves de estado combo do Resumo existam na session_state com valores padrão."""
    if "resumo_combo_preset" not in session_state:
        session_state["resumo_combo_preset"] = "Neutro"
    if "resumo_combo_use_mismatch" not in session_state:
        session_state["resumo_combo_use_mismatch"] = False


def apply_resumo_combo_preset(session_state: MutableMapping[str, Any]) -> tuple[float, float]:
    """Aplica os pesos do preset combo ativo na session_state e retorna (hist_w, style_w)."""
    preset = str(session_state.get("resumo_combo_preset", "Neutro") or "Neutro")
    hist_w, style_w = _COMBO_PRESET_MAP.get(preset, _COMBO_PRESET_MAP["Neutro"])
    session_state["resumo_combo_hist_w"] = float(hist_w)
    session_state["resumo_combo_style_w"] = float(style_w)
    return float(hist_w), float(style_w)
