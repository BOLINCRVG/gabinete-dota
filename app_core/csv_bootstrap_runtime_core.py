from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple


def list_csv_candidates(folder: Path) -> List[Dict[str, object]]:
    """Lista todos os arquivos CSV em uma pasta ordenados por data de modificação."""
    out: List[Dict[str, object]] = []
    if not folder.exists():
        return out
    for p in sorted(folder.glob("*.csv")):
        try:
            st_ = p.stat()
            out.append({"name": p.name, "path": str(p), "mtime": int(st_.st_mtime), "size": int(st_.st_size)})
        except Exception:
            out.append({"name": p.name, "path": str(p), "mtime": 0, "size": 0})
    out.sort(key=lambda x: (x.get("mtime", 0), x.get("size", 0)), reverse=True)
    return out


def season_from_year(year: int) -> str:
    """Converte um ano numérico para o identificador de temporada correspondente (ex: 2025 -> S15)."""
    return f"S{int(year) - 2010}"


def discover_season_csvs(root: Path) -> List[Dict[str, object]]:
    """Descobre arquivos CSV de temporadas Oracle's Elixir na árvore de diretórios a partir de root."""
    candidates: List[Path] = []
    for pat in [
        "*LoL_esports_match_data_from_OraclesElixir.csv",
        "CSV_*/*LoL_esports_match_data_from_OraclesElixir.csv",
        "seasons/*/*LoL_esports_match_data_from_OraclesElixir.csv",
        "seasons/*LoL_esports_match_data_from_OraclesElixir.csv",
        "data/*LoL_esports_match_data_from_OraclesElixir.csv",
    ]:
        candidates.extend(root.glob(pat))

    out: Dict[str, Dict[str, object]] = {}
    for p in candidates:
        m = re.match(r"(\d{4})_LoL_esports_match_data_from_OraclesElixir\.csv$", p.name)
        if not m:
            continue
        year = int(m.group(1))
        season = season_from_year(year)
        key = season
        try:
            st_ = p.stat()
            mtime = int(st_.st_mtime)
            size = int(st_.st_size)
        except Exception:
            mtime, size = 0, 0

        cur = out.get(key)
        if cur is None or int(cur.get("mtime", 0)) < mtime:
            out[key] = {"season": season, "year": year, "csv_path": str(p), "mtime": mtime, "size": size}

    return sorted(out.values(), key=lambda d: int(d.get("year", 0)))


def default_artifact_for(root: Path, season: str, year: int) -> str:
    """Resolve o caminho padrão do artefato ML para uma temporada/ano, buscando candidatos existentes."""
    cand = [
        root / "ml_artifact.json",
        root / f"ml_artifact_{year}.json",
        root / "artifacts" / "ml_artifact.json",
        root / "seasons" / season / "ml_artifact.json",
        root / "seasons" / f"{year}" / "ml_artifact.json",
    ]
    for p in cand:
        if p.exists():
            return str(p)
    return str(root / "ml_artifact.json")


def csv_bootstrap_fingerprint(
    csv_path: str,
    hist_csvs: Optional[List[str]],
    artifact_path: str,
    *,
    file_sig_fn: Callable[[str], Tuple[str, int, int]],
) -> Dict[str, Any]:
    """Gera um dicionário de fingerprint com assinaturas dos arquivos CSV e artefato para deduplicação de bootstrap."""
    hist_norm = [str(p) for p in (hist_csvs or []) if str(p or "").strip()]
    return {
        "csv_sig": list(file_sig_fn(str(csv_path or "").strip())),
        "artifact_sig": list(file_sig_fn(str(artifact_path or "").strip())),
        "hist_csvs": hist_norm,
    }


def apply_bootstrap_selection(
    session_state: MutableMapping[str, Any],
    daily_choice: str,
    hist_choice: List[str] | Tuple[str, ...],
    *,
    clear_selector_flag: bool = False,
) -> None:
    """Aplica a seleção de CSVs diário e histórico na session_state, marcando bootstrap como pendente."""
    daily_s = str(daily_choice or "").strip()
    hist_s = [str(x) for x in (hist_choice or []) if str(x or "").strip()]
    session_state["boot_daily_csv"] = daily_s
    session_state["boot_hist_csvs"] = hist_s
    session_state["cfg_auto_paths"] = False
    session_state["cfg_csv_override"] = daily_s
    session_state["boot_done"] = True
    if clear_selector_flag:
        session_state["boot_show_selector"] = False
    session_state["_csv_bootstrap_force_once"] = True
    session_state["_csv_bootstrap_hold_screen"] = True


def request_csv_runtime_refresh(
    session_state: MutableMapping[str, Any],
    *,
    prepare: bool = False,
    bootstrap: bool = True,
    hold_screen: bool = True,
    force_maint: bool = False,
    force_full: bool = False,
) -> Dict[str, bool]:
    """Sinaliza na session_state que um recarregamento de CSV/bootstrap deve ocorrer no próximo ciclo."""
    if bool(prepare):
        session_state["_csv_prepare_force_once"] = True
    if bool(bootstrap):
        session_state["_csv_bootstrap_force_once"] = True
    if bool(hold_screen):
        session_state["_csv_bootstrap_hold_screen"] = True
    if bool(force_maint):
        session_state["_csv_bootstrap_force_maint"] = True
    # force_full: manutenção COMPLETA (full retrain de modelo+players + refit de
    # params). Necessário p/ sincronizar quando a incremental rápida não dá conta
    # (ex.: players_artifact em drift). One-shot.
    if bool(force_full):
        session_state["_csv_bootstrap_force_full"] = True
    return {
        "prepare": bool(prepare),
        "bootstrap": bool(bootstrap),
        "hold_screen": bool(hold_screen),
        "force_maint": bool(force_maint),
        "force_full": bool(force_full),
    }


def consume_csv_prepare_refresh(session_state: MutableMapping[str, Any]) -> bool:
    """Lê e remove o flag de recarregamento de preparação CSV da session_state (one-shot)."""
    return bool(session_state.pop("_csv_prepare_force_once", False))


def consume_bootstrap_gate_state(session_state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Lê e consome os flags de controle do portão de bootstrap da session_state."""
    return {
        "probe_skip": bool(session_state.get("_probe_skip_csv_bootstrap_gate", False)),
        "force_once": bool(session_state.pop("_csv_bootstrap_force_once", False)),
        "force_maint": bool(session_state.pop("_csv_bootstrap_force_maint", False)),
        "force_full": bool(session_state.pop("_csv_bootstrap_force_full", False)),
        "last_done_key": str(session_state.get("_csv_bootstrap_done_fp", "") or ""),
        "pending_open": bool(session_state.get("_csv_bootstrap_hold_screen", False)),
    }


def mark_bootstrap_gate_completed(session_state: MutableMapping[str, Any], fingerprint_key: str) -> str:
    """Marca o portão de bootstrap como concluído, gravando o fingerprint e liberando a tela de espera."""
    text = str(fingerprint_key or "")
    session_state["_csv_bootstrap_done_fp"] = text
    session_state["_csv_bootstrap_hold_screen"] = False
    return text


def reset_bootstrap_selection(session_state: MutableMapping[str, Any], *, show_selector: bool = True) -> None:
    """Limpa a seleção de bootstrap na session_state e opcionalmente reexibe o seletor de CSV."""
    for key in ["boot_done", "boot_daily_csv", "boot_hist_csvs"]:
        session_state.pop(key, None)
    session_state["cfg_auto_paths"] = False
    session_state["boot_show_selector"] = bool(show_selector)
    session_state["_csv_bootstrap_done_fp"] = ""
    session_state["_csv_bootstrap_hold_screen"] = True


def resolve_ml_paths(session_state: Mapping[str, Any], root: Path) -> tuple[str, list[str], str]:
    """Resolve os caminhos ativos de CSV diário, CSVs históricos e artefato ML a partir da session_state."""
    daily = str(session_state.get("boot_daily_csv") or session_state.get("cfg_csv_override") or "").strip()
    hist = [str(x) for x in (session_state.get("boot_hist_csvs") or []) if str(x or "").strip()]
    art = str(session_state.get("cfg_artifact_override") or str(root / "ml_artifact.json")).strip()
    return daily, hist, art


def consume_auto_run_incremental(session_state: MutableMapping[str, Any]) -> bool:
    """Lê e desativa o flag de execução incremental automática da session_state."""
    enabled = bool(session_state.get("auto_run_incremental", False))
    session_state["auto_run_incremental"] = False
    return enabled


def request_auto_run_incremental(session_state: MutableMapping[str, Any], enabled: bool = True) -> bool:
    """Ativa ou desativa o flag de execução incremental automática na session_state."""
    flag = bool(enabled)
    session_state["auto_run_incremental"] = flag
    return flag


def consume_auto_run_maintenance(session_state: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Lê e remove o flag e status de manutenção automática da session_state (one-shot)."""
    enabled = bool(session_state.pop("auto_run_maintenance", False))
    status = session_state.pop("auto_run_maintenance_status", None)
    return {
        "enabled": bool(enabled),
        "status": (dict(status) if isinstance(status, Mapping) else status),
    }


def request_auto_run_maintenance(session_state: MutableMapping[str, Any], status: Any = None) -> Dict[str, Any]:
    """Solicita execução automática de manutenção na session_state, opcionalmente com payload de status."""
    session_state["auto_run_maintenance"] = True
    payload = dict(status) if isinstance(status, Mapping) else status
    session_state["auto_run_maintenance_status"] = payload
    return {
        "enabled": True,
        "status": payload,
    }


def build_runtime_config_defaults(
    session_state: Mapping[str, Any],
    *,
    app_root: Path,
    seasons: List[Dict[str, object]] | None,
) -> Dict[str, Any]:
    """Constrói o dicionário de configuração padrão de runtime com caminhos CSV e artefato resolvidos."""
    seasons_list = list(seasons or [])
    csv_default = str(
        session_state.get(
            "cfg_csv_override",
            str(app_root / "2025_LoL_esports_match_data_from_OraclesElixir.csv"),
        )
    )
    if seasons_list:
        last = seasons_list[-1]
        artifact_fallback = default_artifact_for(app_root, str(last.get("season") or ""), int(last.get("year") or 0))
    else:
        artifact_fallback = str(app_root / "ml_artifact.json")
    artifact_default = str(session_state.get("cfg_artifact_override", artifact_fallback))
    return {
        "auto_paths": bool(session_state.get("cfg_auto_paths", True)),
        "csv_override": csv_default,
        "artifact_override": artifact_default,
    }


def resolve_season_context(
    seasons: List[Dict[str, object]] | None,
    *,
    selected_label: str | None = None,
    fallback_label: str = "S15",
    fallback_year: int = 2025,
) -> Dict[str, Any]:
    """Resolve o contexto de temporada selecionada incluindo opções, índices e informações da temporada."""
    seasons_list = list(seasons or [])
    if seasons_list:
        options = [str(s.get("season") or "") for s in seasons_list]
        default_idx = max(0, len(options) - 1)
        selected = str(selected_label or "").strip()
        if selected in options:
            selected_idx = int(options.index(selected))
        else:
            selected_idx = int(default_idx)
            selected = str(options[selected_idx])
        info = next((s for s in seasons_list if str(s.get("season") or "") == selected), seasons_list[-1])
        year = int(info.get("year") or fallback_year)
        return {
            "has_seasons": True,
            "options": options,
            "default_idx": int(default_idx),
            "selected_idx": int(selected_idx),
            "selected_label": selected,
            "season_info": dict(info),
            "season_year": int(year),
        }

    return {
        "has_seasons": False,
        "options": [str(fallback_label)],
        "default_idx": 0,
        "selected_idx": 0,
        "selected_label": str(fallback_label),
        "season_info": {"season": str(fallback_label), "year": int(fallback_year), "csv_path": ""},
        "season_year": int(fallback_year),
    }


def resolve_active_csv_and_artifact(
    session_state: Mapping[str, Any],
    *,
    app_root: Path,
    season_label: str,
    season_year: int,
    season_csv_path: str,
    seasons_available: bool,
) -> tuple[str, str]:
    """Resolve os caminhos ativos de CSV e artefato considerando overrides e caminho gerenciado de temporada."""
    csv_override = str(
        session_state.get(
            "cfg_csv_override",
            str(app_root / f"{int(season_year)}_LoL_esports_match_data_from_OraclesElixir.csv"),
        )
    ).strip()
    override_norm = csv_override.replace("/", "\\").lower()
    override_is_repo_managed = any(
        token in override_norm
        for token in (
            "\\csv_diario\\",
            "\\csv_historico\\",
            "\\seasons\\",
        )
    )
    use_season_path = bool(seasons_available) and bool(str(season_csv_path or "").strip()) and (
        bool(session_state.get("cfg_auto_paths", True)) or bool(override_is_repo_managed)
    )
    if use_season_path:
        csv_path = str(season_csv_path or "").strip()
    else:
        csv_path = csv_override
    artifact_path = str(
        session_state.get(
            "cfg_artifact_override",
            default_artifact_for(app_root, str(season_label or ""), int(season_year or 0)),
        )
    ).strip()
    return csv_path, artifact_path


def reconcile_bootstrap_maintenance_status(
    sta: Optional[Dict[str, Any]],
    *,
    stage: str = "",
) -> tuple[Dict[str, Any], str]:
    """Reconcilia e normaliza o dicionário de status de manutenção de bootstrap, retornando o status e nota de diagnóstico."""
    src = dict(sta or {})
    need_flags = {
        "need_retrain_model": bool(src.get("need_retrain_model", False)),
        "need_retrain_params": bool(src.get("need_retrain_params", False)),
        "need_walkforward": bool(src.get("need_walkforward", False)),
        "need_patch_recalibration": bool(src.get("need_patch_recalibration", False)),
        "need_layer_validation": bool(src.get("need_layer_validation", False)),
        "need_calibration_diag": bool(src.get("need_calibration_diag", False)),
    }
    any_need = bool(any(need_flags.values()))
    alerts = [str(x) for x in (src.get("alerts") or []) if str(x).strip()]
    critical = [str(x) for x in (src.get("critical") or []) if str(x).strip()]
    has_issues_raw = bool(src.get("has_issues", False))
    has_issues = bool(has_issues_raw or any_need or alerts)
    severity = str(src.get("severity", "ok") or "ok").strip().lower()
    if not has_issues:
        severity = "ok"
    elif severity not in {"warning", "critical"}:
        severity = "critical" if critical else "warning"

    out = dict(src)
    out["alerts"] = alerts
    out["critical"] = critical
    out["has_issues"] = bool(has_issues)
    out["severity"] = str(severity)
    out["flags_any_need"] = bool(any_need)
    out["reconciled"] = True

    note = ""
    if (has_issues != has_issues_raw) or (src.get("flags_any_need") is None):
        prefix = f"[{stage}] " if str(stage or "").strip() else ""
        note = (
            f"{prefix}status reconciliado (has_issues_raw={str(has_issues_raw).lower()} -> "
            f"{str(has_issues).lower()}, flags_any_need={str(any_need).lower()})."
        )
    return out, note


def should_run_bootstrap_auto_maintenance(
    status_before: Mapping[str, Any],
    *,
    force_maint: bool = False,
) -> bool:
    """Decide se a manutenção automática pós-bootstrap deve ser executada com base nos flags de necessidade."""
    need_model = bool(status_before.get("need_retrain_model"))
    need_patch = bool(status_before.get("need_patch_recalibration"))
    need_params = bool(status_before.get("need_retrain_params"))
    need_wf = bool(status_before.get("need_walkforward"))
    is_critical = str(status_before.get("severity", "")).strip().lower() == "critical"
    return bool(force_maint or need_model or need_params or need_wf or (is_critical and need_patch))


def csv_bootstrap_fingerprint_key(
    csv_path: str,
    hist_csvs: Optional[List[str]],
    artifact_path: str,
    *,
    file_sig_fn: Callable[[str], Tuple[str, int, int]],
) -> str:
    """Gera uma chave string JSON estável representando o fingerprint de bootstrap dos arquivos CSV e artefato."""
    return json.dumps(
        csv_bootstrap_fingerprint(csv_path, hist_csvs, artifact_path, file_sig_fn=file_sig_fn),
        ensure_ascii=False,
        sort_keys=True,
    )
