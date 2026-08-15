from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def run_governance_commands(
    commands: list[list[str]],
    *,
    cwd: str | Path,
    prod_refresh_marker: str = "tools/prod_refresh.py",
    default_timeout_s: int = 240,
    prod_refresh_timeout_s: int = 3600,
) -> dict[str, Any]:
    logs: list[str] = []
    for cmd in commands:
        cmd_list = [str(x) for x in (cmd or [])]
        if not cmd_list:
            continue
        cmd_str = " ".join(cmd_list)
        try:
            timeout_s = int(prod_refresh_timeout_s if prod_refresh_marker in cmd_str else default_timeout_s)
            cp = subprocess.run(
                cmd_list,
                cwd=str(Path(str(cwd)).expanduser()),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = ((str(cp.stdout or "")) + ("\n" + str(cp.stderr or "") if cp.stderr else "")).strip() or "-"
            logs.append(f"$ {cmd_str}\nrc={int(cp.returncode)}\n{out}")
        except Exception as exc:
            logs.append(f"$ {cmd_str}\nERRO: {exc}")
    return {
        "logs": logs,
        "last_run": datetime.now().isoformat(timespec="seconds"),
    }


def normalize_governance_decision(value: Any) -> str:
    parsed = str(value or "").strip().lower()
    return parsed if parsed in {"promote", "shadow_only", "reject"} else "unknown"


def combine_governance_decisions(parts: dict[str, Any]) -> dict[str, Any]:
    rank = {"promote": 0, "shadow_only": 1, "reject": 2, "unknown": 1}
    normalized = {str(k): normalize_governance_decision(v) for k, v in dict(parts or {}).items()}
    present = [v for v in normalized.values() if v in rank]
    final = max(present, key=lambda x: rank.get(x, 1)) if present else "unknown"
    final_label = {
        "promote": "PROMOVER",
        "shadow_only": "SHADOW ONLY",
        "reject": "REJEITAR",
        "unknown": "DESCONHECIDO",
    }.get(final, "DESCONHECIDO")
    message = f"Decisão combinada: {final_label} | " + " | ".join([f"{k}={v}" for k, v in normalized.items()])
    return {
        "final": final,
        "final_label": final_label,
        "parts": normalized,
        "message": message,
    }


def run_governance_operation(
    *,
    promote: bool,
    cwd: str | Path,
    tag: str = "app_governance",
    timeout_s: int = 1200,
) -> dict[str, Any]:
    cmd = ["python", "tools/promote_baseline.py", "--tag", str(tag)] if bool(promote) else ["python", "tools/rollback_baseline.py"]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(Path(str(cwd)).expanduser()),
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
        )
        output = (str(cp.stdout or "") + ("\n" + str(cp.stderr or "") if cp.stderr else "")).strip() or "-"
        return {
            "ok": int(cp.returncode) == 0,
            "cmd": cmd,
            "returncode": int(cp.returncode),
            "log_line": f"$ {' '.join(cmd)}\nrc={int(cp.returncode)}\n{output}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "cmd": cmd,
            "returncode": None,
            "log_line": f"$ {' '.join(cmd)}\nERRO: {exc}",
            "error": str(exc),
        }


def append_governance_log(previous_log: str | None, line: str | None, *, now_iso: str | None) -> dict[str, str]:
    prev = str(previous_log or "").strip()
    log_line = str(line or "").strip()
    merged = (prev + "\n\n" + log_line).strip() if prev and log_line else (log_line or prev)
    return {
        "last_log": merged,
        "last_run": str(now_iso or ""),
    }
