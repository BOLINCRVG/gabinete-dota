from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple


def read_json_file_safe(path_like: str | Path) -> Dict[str, Any]:
    """Lê e retorna um arquivo JSON como dicionário, retornando {} em caso de erro ou ausência."""
    try:
        p = Path(path_like).expanduser()
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
    return {}


def write_json_file_atomic(path_like: str | Path, payload: Dict[str, Any]) -> bool:
    """Escreve atomicamente um dicionário como JSON em arquivo usando arquivo temporário intermediário."""
    try:
        p = Path(path_like).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception:
        return False


def pid_is_alive(pid_like: Any) -> bool:
    """Verifica se um processo com o PID fornecido ainda está em execução no sistema operacional."""
    try:
        pid = int(pid_like)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            process_query_limited_information = 0x1000
            hnd = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
            if hnd:
                ctypes.windll.kernel32.CloseHandle(hnd)
                return True
            try:
                err = int(ctypes.GetLastError())
            except Exception:
                err = 0
            return bool(err == 5)
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_bootstrap_maint_state(state_path: str | Path) -> Dict[str, Any]:
    """Lê o arquivo de estado de manutenção de bootstrap e retorna como dicionário."""
    obj = read_json_file_safe(state_path)
    return obj if isinstance(obj, dict) else {}


def write_bootstrap_maint_state(state_path: str | Path, payload: Dict[str, Any]) -> bool:
    """Persiste o dicionário de estado de manutenção de bootstrap no arquivo JSON indicado."""
    if not isinstance(payload, dict):
        return False
    return write_json_file_atomic(state_path, payload)


def bootstrap_maint_state_start(
    state_path: str | Path,
    *,
    fp_key: str,
    owner_token: str = "",
    total_steps: int = 8,
    pid_value: int | None = None,
    now_dt: datetime | None = None,
    now_ts: float | None = None,
    run_id: str | None = None,
) -> str:
    """Inicia e persiste um novo estado de execução de manutenção de bootstrap, retornando o run_id gerado."""
    dt = now_dt or datetime.now()
    rid = str(run_id or f"{dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}")
    ts = float(time.time() if now_ts is None else now_ts)
    payload = {
        "version": "csv_bootstrap_maint_state_v1",
        "run_id": rid,
        "fp_key": str(fp_key),
        "pid": int(os.getpid() if pid_value is None else pid_value),
        "owner_token": str(owner_token or ""),
        "status": "running",
        "started_at": dt.isoformat(timespec="seconds"),
        "updated_at": dt.isoformat(timespec="seconds"),
        "heartbeat_ts": ts,
        "current": {
            "step": "init",
            "idx": 0,
            "total": int(max(1, int(total_steps or 8))),
            "status": "iniciando",
        },
        "steps": [],
    }
    write_bootstrap_maint_state(state_path, payload)
    return rid


def bootstrap_maint_state_update(
    state_path: str | Path,
    *,
    run_id: str,
    fp_key: str,
    owner_token: str = "",
    event: Optional[Dict[str, Any]] = None,
    pid_value: int | None = None,
    now_dt: datetime | None = None,
    now_ts: float | None = None,
) -> None:
    """Atualiza o estado de execução de manutenção com o evento de progresso fornecido."""
    rid = str(run_id or "").strip()
    if not rid:
        return
    evt = dict(event or {})
    cur = read_bootstrap_maint_state(state_path)
    cur_rid = str(cur.get("run_id", "") or "").strip() if isinstance(cur, dict) else ""
    if cur_rid and cur_rid != rid:
        return
    out = dict(cur) if isinstance(cur, dict) else {}
    dt = now_dt or datetime.now()
    ts = float(time.time() if now_ts is None else now_ts)
    now_iso = dt.isoformat(timespec="seconds")
    out["version"] = "csv_bootstrap_maint_state_v1"
    out["run_id"] = rid
    out["fp_key"] = str(fp_key)
    out["pid"] = int(os.getpid() if pid_value is None else pid_value)
    out["owner_token"] = str(owner_token or out.get("owner_token") or "")
    out["updated_at"] = now_iso
    out["heartbeat_ts"] = ts
    if not str(out.get("started_at", "") or "").strip():
        out["started_at"] = now_iso

    step_name = str(evt.get("step", "") or "").strip() or "-"
    idx = int(max(0, evt.get("idx") or 0))
    total = int(max(1, evt.get("total") or 1))
    status_txt = str(evt.get("status", "") or "").strip()
    summary_txt = str(evt.get("summary", "") or "").strip()
    ev_name = str(evt.get("event", "") or "").strip().lower()

    out["current"] = {
        "step": step_name,
        "idx": idx,
        "total": total,
        "status": status_txt,
        "event": ev_name,
    }

    if ev_name in {"step_done", "running"}:
        rows = list(out.get("steps") or [])
        if not isinstance(rows, list):
            rows = []
        row = {
            "step": step_name,
            "status": status_txt,
            "summary": summary_txt,
            "idx": idx,
            "total": total,
            "updated_at": now_iso,
        }
        found = False
        for i in range(len(rows) - 1, -1, -1):
            rec = rows[i]
            if isinstance(rec, dict) and str(rec.get("step", "") or "").strip() == step_name:
                rows[i] = row
                found = True
                break
        if not found:
            rows.append(row)
        out["steps"] = rows[-20:]

    if ev_name == "done":
        st_txt = status_txt.lower()
        out["status"] = "done" if st_txt in {"ok", "done", "sucesso"} else "error"
        out["finished_at"] = now_iso
    elif ev_name == "error":
        out["status"] = "error"
        out["finished_at"] = now_iso
    else:
        out["status"] = str(out.get("status", "running") or "running")

    write_bootstrap_maint_state(state_path, out)


def bootstrap_maint_state_finish(
    state_path: str | Path,
    *,
    run_id: str,
    fp_key: str,
    owner_token: str = "",
    ok: bool,
    report: Optional[Dict[str, Any]] = None,
    pid_value: int | None = None,
    now_dt: datetime | None = None,
    now_ts: float | None = None,
) -> None:
    """Finaliza e persiste o estado de manutenção com resultado de sucesso ou erro e relatório."""
    rid = str(run_id or "").strip()
    if not rid:
        return
    rep = dict(report or {})
    bootstrap_maint_state_update(
        state_path,
        run_id=rid,
        fp_key=str(fp_key),
        owner_token=str(owner_token or ""),
        event={
            "event": "done",
            "step": "finish",
            "idx": int(max(0, len(rep.get("steps") or []) if isinstance(rep.get("steps"), list) else 0)),
            "total": 8,
            "status": ("ok" if bool(ok) else "erro"),
            "summary": str(rep.get("error") or ""),
        },
        pid_value=pid_value,
        now_dt=now_dt,
        now_ts=now_ts,
    )
    cur = read_bootstrap_maint_state(state_path)
    cur_rid = str(cur.get("run_id", "") or "").strip() if isinstance(cur, dict) else ""
    if cur_rid != rid:
        return
    out = dict(cur)
    out["result"] = {
        "ok": bool(ok),
        "error": str(rep.get("error") or ""),
        "duration_s": float(rep.get("duration_s") or 0.0),
        "policy_profile": str(rep.get("policy_profile") or ""),
        "policy_reason": str(rep.get("policy_reason") or ""),
    }
    write_bootstrap_maint_state(state_path, out)


def _stale_clear_decision(
    *,
    pid_alive: bool,
    has_live_state: bool,
    lock_age: float,
    hb_age: float,
    stale_limit: float,
) -> Tuple[bool, str]:
    """Decisão PURA: o lock obsoleto pode ser limpo? Retorna (pode, motivo).

    Extraída de cleanup_stale_bootstrap_maint_lock. Ordem de prioridade dos
    motivos preservada (pid morto > sem state vivo > running sem heartbeat).
    Pura para poder testar cada ramo sem I/O; golden end-to-end prova igual."""
    if not pid_alive and lock_age >= 5.0:
        return True, f"pid_dead(age_s={lock_age:.1f})"
    if (not has_live_state) and lock_age >= stale_limit:
        return True, f"no_live_state(age_s={lock_age:.1f})"
    if has_live_state and hb_age > stale_limit and lock_age > max(10.0, stale_limit / 3.0):
        return True, f"running_state_no_heartbeat(hb_age_s={hb_age:.1f})"
    return False, "not_stale_enough"


def cleanup_stale_bootstrap_maint_lock(
    *,
    lock_path: str | Path,
    state_path: str | Path,
    fp_key: str,
    stale_s: int = 180,
    now_ts: float | None = None,
) -> Tuple[bool, str]:
    """Remove o lock de manutenção de bootstrap se estiver obsoleto (processo morto ou heartbeat expirado)."""
    lock = read_json_file_safe(lock_path)
    if not isinstance(lock, dict) or not lock:
        return False, "no_lock"
    ts_now = float(time.time() if now_ts is None else now_ts)
    lock_fp = str(lock.get("fp_key", "") or "")
    lock_pid = str(lock.get("pid", "") or "")
    lock_ts = float(lock.get("ts", 0.0) or 0.0)
    lock_age = float(max(0.0, ts_now - lock_ts)) if lock_ts > 0.0 else float("inf")
    if lock_fp and lock_fp != str(fp_key):
        return False, "lock_other_fp"

    state = read_bootstrap_maint_state(state_path)
    state_fp = str(state.get("fp_key", "") or "") if isinstance(state, dict) else ""
    state_status = str(state.get("status", "") or "").strip().lower() if isinstance(state, dict) else ""
    hb_ts = float(state.get("heartbeat_ts", 0.0) or 0.0) if isinstance(state, dict) else 0.0
    hb_age = float(max(0.0, ts_now - hb_ts)) if hb_ts > 0.0 else float("inf")

    pid_alive = pid_is_alive(lock_pid) if str(lock_pid).strip() else False
    stale_limit = float(max(30, int(stale_s)))

    has_live_state = bool(state_fp == lock_fp and state_status == "running")
    if has_live_state and pid_alive and hb_age <= stale_limit:
        step_name = str((state.get("current") or {}).get("step") or "-")
        return False, f"active_run(step={step_name}, hb_age_s={hb_age:.1f})"

    can_clear, reason = _stale_clear_decision(
        pid_alive=pid_alive, has_live_state=has_live_state,
        lock_age=lock_age, hb_age=hb_age, stale_limit=stale_limit,
    )
    if not can_clear:
        return False, reason

    try:
        if has_live_state and isinstance(state, dict):
            rid = str(state.get("run_id") or "").strip()
            if rid:
                bootstrap_maint_state_update(
                    state_path,
                    run_id=rid,
                    fp_key=str(lock_fp or fp_key),
                    owner_token=str(state.get("owner_token") or ""),
                    event={
                        "event": "error",
                        "step": "finish",
                        "idx": 0,
                        "total": 8,
                        "status": "erro",
                        "summary": f"stale_lock_recovered:{reason}",
                    },
                    now_ts=ts_now,
                )
        p = Path(lock_path).expanduser()
        if p.exists():
            p.unlink()
        return True, f"cleared:{reason}"
    except Exception as exc:
        return False, f"clear_failed:{type(exc).__name__}:{exc}"


def load_bootstrap_maint_report_for_fp(report_path: str | Path, fp_key: str) -> Optional[Dict[str, Any]]:
    """Carrega o relatório de manutenção do arquivo somente se o fingerprint corresponder ao fp_key informado."""
    rep = read_json_file_safe(report_path)
    if not isinstance(rep, dict) or not rep:
        return None
    try:
        rep_fp = rep.get("csv_fingerprint")
        rep_key = json.dumps(rep_fp, ensure_ascii=False, sort_keys=True) if isinstance(rep_fp, dict) else ""
    except Exception:
        rep_key = ""
    if rep_key != str(fp_key):
        return None
    return rep


def get_bootstrap_runtime_state(session_state: Mapping[str, Any]) -> Dict[str, Any]:
    """Coleta e retorna o estado de runtime do bootstrap (poll, run_id, status, relatórios) da session_state."""
    return {
        "lock_poll_n": int(session_state.get("_csv_bootstrap_lock_poll_n", 0) or 0),
        "auto_maint_ran_fp": str(session_state.get("_csv_bootstrap_auto_maint_ran_fp", "") or ""),
        "active_run_id": str(session_state.get("_csv_bootstrap_active_run_id", "") or ""),
        "status_before": (
            dict(session_state.get("_csv_bootstrap_status_before", {}))
            if isinstance(session_state.get("_csv_bootstrap_status_before"), Mapping)
            else {}
        ),
        "last_report": (
            dict(session_state.get("_csv_bootstrap_last_report", {}))
            if isinstance(session_state.get("_csv_bootstrap_last_report"), Mapping)
            else {}
        ),
        "auto_maint_last_report": (
            dict(session_state.get("_csv_bootstrap_auto_maint_last_report", {}))
            if isinstance(session_state.get("_csv_bootstrap_auto_maint_last_report"), Mapping)
            else {}
        ),
        "lock_stale_s": int(session_state.get("auto_maint_lock_stale_s", 180) or 180),
        "lock_poll_max": int(session_state.get("auto_maint_lock_poll_max", 240) or 240),
        "lock_poll_sleep_s": float(session_state.get("auto_maint_lock_poll_sleep_s", 3.0) or 3.0),
    }


def set_bootstrap_lock_poll_n(session_state: MutableMapping[str, Any], value: Any) -> int:
    """Atualiza o contador de polls de lock de bootstrap na session_state."""
    count = int(value or 0)
    session_state["_csv_bootstrap_lock_poll_n"] = count
    return count


def set_bootstrap_auto_maint_run_fp(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Grava o fingerprint da última execução de manutenção automática de bootstrap na session_state."""
    text = str(value or "")
    session_state["_csv_bootstrap_auto_maint_ran_fp"] = text
    return text


def set_bootstrap_active_run_id(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Grava o run_id da execução de manutenção ativa na session_state."""
    text = str(value or "")
    session_state["_csv_bootstrap_active_run_id"] = text
    return text


def set_bootstrap_status_before(session_state: MutableMapping[str, Any], status: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Persiste o status pré-manutenção do bootstrap na session_state para referência posterior."""
    payload = dict(status or {})
    session_state["_csv_bootstrap_status_before"] = payload
    return payload


def set_bootstrap_last_report(session_state: MutableMapping[str, Any], report: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Persiste o último relatório de bootstrap na session_state."""
    payload = dict(report or {})
    session_state["_csv_bootstrap_last_report"] = payload
    return payload


def set_bootstrap_auto_maint_last_report(session_state: MutableMapping[str, Any], report: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Persiste o último relatório de manutenção automática de bootstrap na session_state."""
    payload = dict(report or {})
    session_state["_csv_bootstrap_auto_maint_last_report"] = payload
    return payload


def clear_bootstrap_auto_maint_run_state(session_state: MutableMapping[str, Any]) -> None:
    """Limpa o fingerprint e run_id de manutenção automática de bootstrap na session_state."""
    session_state["_csv_bootstrap_auto_maint_ran_fp"] = ""
    session_state["_csv_bootstrap_active_run_id"] = ""


def _lock_busy_reason(
    *,
    fp_key: str,
    lock_ts: float,
    lock_fp: str,
    lock_pid: str,
    lock_age: float,
    lock_pid_alive: bool,
    same_owner: bool,
    state_fp: str,
    state_status: str,
    hb_age: float,
    ttl: float,
    same_fp_busy_s: float,
    hb_busy_s: float,
    other_fp_grace_s: float,
) -> str:
    """Decisão PURA: o lock está ocupado? Retorna a mensagem de 'busy' ou "".

    Extraída de acquire_bootstrap_maint_lock (era inline, CC alto). É a
    SEMÂNTICA de concorrência do lock — separada aqui só para poder testá-la
    exaustivamente sem I/O. Nenhuma regra mudou; o golden end-to-end prova.
    """
    if lock_ts > 0.0 and lock_fp and (not same_owner) and lock_fp != str(fp_key) and lock_age < other_fp_grace_s:
        return f"lock_busy(fp={lock_fp}, pid={lock_pid}, age_s={lock_age:.1f})"
    if lock_ts > 0.0 and lock_fp and bool(lock_pid_alive) and (not same_owner):
        if (state_fp == lock_fp) and (state_status == "running") and (hb_age < hb_busy_s):
            return f"lock_busy_same_fp(pid={lock_pid}, age_s={lock_age:.1f}, hb_s={hb_age:.1f})"
        if lock_fp == str(fp_key):
            hb_stale_running = bool((state_fp == lock_fp) and (state_status == "running") and (hb_age >= hb_busy_s))
            if (not hb_stale_running) and (lock_age < same_fp_busy_s):
                return f"lock_busy_same_fp(pid={lock_pid}, age_s={lock_age:.1f})"
        elif lock_age < ttl:
            return f"lock_busy(fp={lock_fp}, pid={lock_pid}, age_s={lock_age:.1f})"
    return ""


def acquire_bootstrap_maint_lock(
    *,
    lock_path: str | Path,
    state_path: str | Path,
    fp_key: str,
    ttl_s: int = 1800,
    owner_token: str = "",
    stale_s: int = 180,
    now_ts: float | None = None,
    pid_value: int | None = None,
) -> Tuple[bool, str]:
    """Tenta adquirir o lock de manutenção de bootstrap, retornando (sucesso, mensagem)."""
    ts_now = float(time.time() if now_ts is None else now_ts)
    self_pid = int(os.getpid() if pid_value is None else pid_value)
    owner = str(owner_token or "").strip()
    stale_cleaned, stale_msg = cleanup_stale_bootstrap_maint_lock(
        lock_path=lock_path,
        state_path=state_path,
        fp_key=str(fp_key),
        stale_s=int(stale_s),
        now_ts=ts_now,
    )
    lock = read_json_file_safe(lock_path)
    try:
        lock_ts = float(lock.get("ts", 0.0) or 0.0) if isinstance(lock, dict) else 0.0
    except Exception:
        lock_ts = 0.0
    lock_fp = str(lock.get("fp_key", "") or "") if isinstance(lock, dict) else ""
    lock_pid = str(lock.get("pid", "") or "") if isinstance(lock, dict) else ""
    lock_owner = str(lock.get("owner_token", "") or "") if isinstance(lock, dict) else ""
    lock_age = (ts_now - lock_ts) if lock_ts > 0.0 else float("inf")
    lock_pid_alive = pid_is_alive(lock_pid) if str(lock_pid).strip() else False
    same_owner = bool(owner and lock_owner and (owner == lock_owner))
    state = read_bootstrap_maint_state(state_path)
    state_fp = str(state.get("fp_key", "") or "") if isinstance(state, dict) else ""
    state_status = str(state.get("status", "") or "").strip().lower() if isinstance(state, dict) else ""
    hb_ts = float(state.get("heartbeat_ts", 0.0) or 0.0) if isinstance(state, dict) else 0.0
    hb_age = (ts_now - hb_ts) if hb_ts > 0.0 else float("inf")

    ttl = float(max(30, int(ttl_s)))
    same_fp_busy_s = float(min(ttl, max(30.0, float(stale_s))))
    hb_busy_s = float(max(15.0, min(same_fp_busy_s, float(stale_s))))
    other_fp_grace_s = float(min(ttl, 45.0))
    busy = _lock_busy_reason(
        fp_key=str(fp_key), lock_ts=lock_ts, lock_fp=lock_fp, lock_pid=lock_pid,
        lock_age=lock_age, lock_pid_alive=lock_pid_alive, same_owner=same_owner,
        state_fp=state_fp, state_status=state_status, hb_age=hb_age, ttl=ttl,
        same_fp_busy_s=same_fp_busy_s, hb_busy_s=hb_busy_s, other_fp_grace_s=other_fp_grace_s,
    )
    if busy:
        return False, busy

    payload = {
        "fp_key": str(fp_key),
        "pid": self_pid,
        "owner_token": (owner if owner else f"pid:{self_pid}"),
        "ts": ts_now,
        "acquired_at": datetime.now().isoformat(timespec="seconds"),
    }
    if not write_json_file_atomic(lock_path, payload):
        return False, "lock_write_failed"
    if stale_cleaned:
        return True, f"ok_after_{stale_msg}"
    return True, "ok"


def release_bootstrap_maint_lock(
    *,
    lock_path: str | Path,
    state_path: str | Path,
    fp_key: str,
    owner_token: str = "",
) -> None:
    """Libera o lock de manutenção de bootstrap, marcando estado como erro se ainda em execução."""
    try:
        lock = read_json_file_safe(lock_path)
        lock_fp = str(lock.get("fp_key", "") or "") if isinstance(lock, dict) else ""
        lock_owner = str(lock.get("owner_token", "") or "") if isinstance(lock, dict) else ""
        owner = str(owner_token or "").strip()
        if lock_fp and lock_fp != str(fp_key):
            return
        if owner and lock_owner and lock_owner != owner:
            return
        p = Path(lock_path).expanduser()
        if p.exists():
            p.unlink()
        stt = read_bootstrap_maint_state(state_path)
        stt_fp = str(stt.get("fp_key", "") or "") if isinstance(stt, dict) else ""
        stt_owner = str(stt.get("owner_token", "") or "") if isinstance(stt, dict) else ""
        stt_status = str(stt.get("status", "") or "").strip().lower() if isinstance(stt, dict) else ""
        if stt_fp == str(fp_key) and ((owner and stt_owner == owner) or (not owner)):
            if stt_status == "running":
                bootstrap_maint_state_update(
                    state_path,
                    run_id=str(stt.get("run_id") or ""),
                    fp_key=str(fp_key),
                    owner_token=str(stt_owner or owner),
                    event={
                        "event": "error",
                        "step": "finish",
                        "idx": 0,
                        "total": 8,
                        "status": "erro",
                        "summary": "lock_released_without_finish",
                    },
                )
    except Exception:
        pass
