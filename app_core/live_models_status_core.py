"""Status, freshness detection and auto-retrain helpers for live ML models.

Tracks when each saved model in `live_models/` was last trained and against
which CSV state. Lets the app:

  - Display "model freshness" (up-to-date / stale / very stale).
  - Decide automatically when to retrain.
  - Run retraining in a background subprocess so the UI stays responsive.

State persisted in `live_models/manifest.json`:

    {
      "version": "live_models_manifest_v1",
      "last_trained_at": "2026-05-26T14:00:00Z",
      "csv_signatures": [{"path": ..., "size": ..., "mtime_ns": ..., "n_games": ...}],
      "models_present": ["state/minute_10", "state/minute_15", ...],
      "n_games_at_train": 25506
    }
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd


_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# CSV signature
# ---------------------------------------------------------------------------

def _file_signature(path: str | Path) -> Dict[str, Any]:
    p = Path(str(path))
    out: Dict[str, Any] = {
        "path": str(p.resolve(strict=False) if p.exists() else p),
        "size": None, "mtime_ns": None, "n_games": None,
    }
    try:
        st = p.stat()
        out["size"] = int(st.st_size)
        out["mtime_ns"] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
    except OSError:
        pass
    # n_games is computed lazily by the caller if needed.
    return out


def csv_signatures(csv_paths: List[str | Path], *, count_games: bool = False) -> List[Dict[str, Any]]:
    """Build a list of CSV signatures, optionally counting unique games per file."""
    sigs = []
    for p in csv_paths:
        sig = _file_signature(p)
        if count_games and Path(p).exists():
            try:
                df = pd.read_csv(p, usecols=["gameid"], low_memory=False)
                sig["n_games"] = int(df["gameid"].nunique())
            except Exception:
                sig["n_games"] = None
        sigs.append(sig)
    return sigs


def _sigs_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return (
        str(a.get("path") or "") == str(b.get("path") or "")
        and int(a.get("size") or -1) == int(b.get("size") or -1)
        and int(a.get("mtime_ns") or -1) == int(b.get("mtime_ns") or -1)
    )


def _sig_diff_n_games(current: List[Dict[str, Any]], at_train: List[Dict[str, Any]]) -> int:
    """How many MORE games exist in current vs at train time (cumulative across CSVs)."""
    cur_total = sum(int(s.get("n_games") or 0) for s in current)
    train_total = sum(int(s.get("n_games") or 0) for s in at_train)
    return int(max(0, cur_total - train_total))


# ---------------------------------------------------------------------------
# Manifest read / write
# ---------------------------------------------------------------------------

def manifest_path(models_dir: str | Path) -> Path:
    return Path(str(models_dir)) / _MANIFEST_NAME


def read_manifest(models_dir: str | Path) -> Dict[str, Any]:
    p = manifest_path(models_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_manifest(models_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    p = manifest_path(models_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def update_manifest_after_training(
    models_dir: str | Path,
    csv_paths: List[str | Path],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record the current CSV signatures + timestamp into the manifest."""
    sigs = csv_signatures(csv_paths, count_games=True)
    total_games = sum(int(s.get("n_games") or 0) for s in sigs)
    state_dir = Path(str(models_dir))
    models_present = []
    if state_dir.exists():
        for sub in sorted(state_dir.iterdir()):
            if sub.is_dir() and (sub / "metadata.json").exists():
                models_present.append(sub.name)
    manifest = {
        "version": "live_models_manifest_v1",
        "last_trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "csv_signatures": sigs,
        "models_present": models_present,
        "n_games_at_train": total_games,
    }
    if extra:
        manifest.update(extra)
    write_manifest(models_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Freshness assessment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreshnessStatus:
    """Outcome of comparing the manifest to the current CSV state."""

    state: str  # 'fresh' | 'stale' | 'very_stale' | 'no_manifest' | 'no_models'
    n_new_games: int
    last_trained_at: Optional[str]
    days_since_train: Optional[float]
    csv_changed: bool
    message: str

    def as_dict(self) -> dict:
        return {
            "state": str(self.state),
            "n_new_games": int(self.n_new_games),
            "last_trained_at": str(self.last_trained_at or ""),
            "days_since_train": float(self.days_since_train) if self.days_since_train is not None else None,
            "csv_changed": bool(self.csv_changed),
            "message": str(self.message),
        }


def assess_freshness(
    models_dir: str | Path,
    csv_paths: List[str | Path],
    *,
    stale_n_games: int = 200,
    very_stale_n_games: int = 1000,
    stale_days: float = 14.0,
    very_stale_days: float = 45.0,
) -> FreshnessStatus:
    """Determine whether the saved models are still considered current.

    Thresholds:
      - "stale"      = more than `stale_n_games` new games, OR `stale_days` old
      - "very_stale" = more than `very_stale_n_games` new games, OR `very_stale_days` old
      - "no_manifest" = there's no manifest yet
      - "no_models" = manifest exists but model files are missing
    """
    state_dir = Path(str(models_dir))
    if not state_dir.exists():
        return FreshnessStatus(
            state="no_models", n_new_games=0, last_trained_at=None,
            days_since_train=None, csv_changed=False,
            message="Diretorio live_models/ nao existe. Treinar modelos pela primeira vez.",
        )

    manifest = read_manifest(state_dir)
    if not manifest:
        return FreshnessStatus(
            state="no_manifest", n_new_games=0, last_trained_at=None,
            days_since_train=None, csv_changed=False,
            message="Manifest ausente. Treinar para registrar status atual.",
        )

    # Days since last train
    last_iso = str(manifest.get("last_trained_at") or "")
    days = None
    try:
        if last_iso:
            dt = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
            days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        days = None

    cur_sigs = csv_signatures(csv_paths, count_games=True)
    train_sigs = list(manifest.get("csv_signatures") or [])

    # CSV identity change (different path or new file added)
    csv_changed = False
    if len(cur_sigs) != len(train_sigs):
        csv_changed = True
    else:
        for cur, tr in zip(cur_sigs, train_sigs):
            if str(cur.get("path") or "") != str(tr.get("path") or ""):
                csv_changed = True
                break

    n_new = _sig_diff_n_games(cur_sigs, train_sigs)

    # Decide state
    state = "fresh"
    if n_new >= very_stale_n_games or (days is not None and days >= very_stale_days):
        state = "very_stale"
    elif n_new >= stale_n_games or (days is not None and days >= stale_days):
        state = "stale"

    if state == "fresh":
        msg = f"Modelos atualizados ({n_new} jogos novos desde o ultimo treino"
        if days is not None:
            msg += f", {days:.1f} dias atras"
        msg += ")."
    elif state == "stale":
        msg = (
            f"Modelos comecando a envelhecer: {n_new} jogos novos no CSV. "
            "Recomendado retreinar nos proximos dias."
        )
    else:  # very_stale
        msg = (
            f"Modelos desatualizados ({n_new} jogos novos, {days:.1f} dias se aplicavel). "
            "Retreinar agora para evitar degradacao."
        )

    return FreshnessStatus(
        state=state, n_new_games=n_new,
        last_trained_at=last_iso or None,
        days_since_train=days, csv_changed=csv_changed,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------

def build_retrain_command(
    *,
    val_cutoff: str = "2026-03-31",
    output: str = "FASE6_LIVE.json",
    save_models_dir: str = "live_models",
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build the CLI command to retrain ALL live models (fases 6/7/8/10).

    2026-07-05: antes chamava só o run_phase6_live_gate (estado→vitória) e os
    modelos de MERCADO live (kills/torres/... , first-X, player props) ficavam
    envelhecendo pra sempre. Agora usa o orquestrador, que também atualiza o
    manifest de frescor no fim.
    """
    cmd = [
        sys.executable, "tools/retrain_live_models_all.py",
        "--val-cutoff", str(val_cutoff),
        "--models-root", str(save_models_dir),
        "--n-bootstrap", "500",  # smaller bootstrap for faster turnaround
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def launch_retrain_background(
    *,
    project_root: str | Path = ".",
    val_cutoff: str = "2026-03-31",
    save_models_dir: str = "live_models",
    log_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Start the retrain process in the background. Returns process info.

    The caller can poll `is_retrain_running()` or check the log file to monitor
    progress. The process automatically updates the manifest on success.
    """
    cmd = build_retrain_command(val_cutoff=val_cutoff, save_models_dir=save_models_dir)
    root = Path(str(project_root)).resolve()

    # PRE-VOO (achado 16/08/2026): o Popen SUCEDE mesmo com o script inexistente,
    # porque quem existe e o python.exe — quem falha e o Python depois, ao abrir o
    # arquivo. Resultado: "ok: True", carimbo de tentativa gravado, e a unica
    # pista era uma linha num retrain.log que ninguem abre.
    # Foi assim que o Dota ficou SEM NENHUM modelo de estado ao vivo: o clone nao
    # tem tools/retrain_live_models_all.py, o app tentava todo dia, "dava certo",
    # e a aba mais validada do projeto simplesmente nao existia la.
    _script = next((Path(str(c)) for c in cmd[1:2]), None)
    if _script is not None and not (root / _script).exists() and not _script.is_absolute():
        return {"ok": False, "command": cmd,
                "error": f"script de retreino nao existe neste repositorio: {_script}"}
    log = Path(str(log_path)) if log_path else root / "live_models" / "retrain.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Use Popen so it runs detached; redirect output to log.
        log_f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=str(root),
            stdout=log_f, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        # Save PID for status checks
        pid_file = log.parent / "retrain.pid"
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        return {
            "ok": True, "pid": int(proc.pid),
            "log_path": str(log), "command": cmd,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "command": cmd}


# ---------------------------------------------------------------------------
# AUTO-RETREINO (22/07, pedido dele "não quero mais saber do ao vivo"): os
# modelos ao vivo não estavam na cadência do robô e ficavam velhos calados
# (manifesto 05/07). Agora o app dispara o retreino SOZINHO quando detecta que
# estão desatualizados — com trava de tempo para não relançar a cada F5.
# ---------------------------------------------------------------------------

# Estados de frescor que justificam um retreino automático.
_STALE_STATES = ("stale", "very_stale", "no_manifest", "no_models")


def _auto_retrain_marker(models_dir: str | Path) -> Path:
    return Path(str(models_dir)) / "last_auto_retrain.txt"


def hours_since_auto_retrain(models_dir: str | Path = "live_models") -> float:
    """Horas desde a última tentativa AUTOMÁTICA (inf se nunca / ilegível)."""
    p = _auto_retrain_marker(models_dir)
    try:
        if not p.exists():
            return float("inf")
        ts = datetime.fromisoformat(p.read_text(encoding="utf-8").strip())
        return float((datetime.now(ts.tzinfo) - ts).total_seconds() / 3600.0)
    except (OSError, ValueError, TypeError):
        return float("inf")


def mark_auto_retrain_attempt(models_dir: str | Path = "live_models") -> None:
    """Carimba agora — a trava de cooldown olha este arquivo."""
    try:
        p = _auto_retrain_marker(models_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(datetime.now().astimezone().isoformat(), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def should_auto_retrain(
    *,
    state: str,
    running: bool,
    hours_since_attempt: float,
    min_cooldown_hours: float = 6.0,
) -> bool:
    """Decide (puro) se o app deve disparar o retreino automático.

    Só dispara quando: os modelos estão desatualizados E não há retreino rodando
    E a última tentativa automática foi há tempo suficiente (a trava que evita
    relançar a cada abertura da página / a cada F5, ou repetir um retreino que
    falha rápido)."""
    if bool(running):
        return False
    if str(state) not in _STALE_STATES:
        return False
    try:
        h = float(hours_since_attempt)
    except (TypeError, ValueError):
        h = float("inf")
    return h >= float(min_cooldown_hours)


def _pid_alive(pid: int) -> bool:
    """True se o PID está vivo AGORA. Match EXATO — nada de substring.

    Bug achado com ele (23/07): a versão antiga fazia `str(pid) in stdout`, então
    um processo `18440` ou a coluna de memória `"1,844 K"` faziam o 1844 morto
    parecer vivo -> "em retreino" fantasma. Aqui o tasklist sai em CSV e a coluna
    PID (índice 1) é comparada exatamente."""
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            out = (r.stdout or "").strip()
            if not out or out.upper().startswith("INFO:"):
                return False
            for row in csv.reader(io.StringIO(out)):
                if len(row) >= 2 and row[1].strip() == str(int(pid)):
                    return True
            return False
        os.kill(int(pid), 0)
        return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def is_retrain_running(models_dir: str | Path = "live_models") -> bool:
    """True se um retreino está rodando de verdade (checa liveness do PID).

    Auto-cura: se o pid_file existe mas o processo está MORTO (ou o arquivo está
    ilegível), o arquivo é APAGADO aqui mesmo. O processo de retreino é detached
    e não limpa o próprio pid ao morrer, então sem isto o fantasma ficava pra
    sempre e o app dizia "em retreino" eternamente. Nunca mais limpar na mão."""
    pid_file = Path(str(models_dir)) / "retrain.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        _remove_stale_pid(pid_file)
        return False
    if _pid_alive(pid):
        return True
    _remove_stale_pid(pid_file)
    return False


def _remove_stale_pid(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except OSError:
        pass


def is_streamlit_cloud() -> bool:
    """True se estamos rodando na Streamlit Community Cloud.

    A Cloud roda como usuário 'appuser' (HOME=/home/appuser) e monta o repo em
    /mount/src/<repo>. Qualquer um dos dois sinais basta. No Windows nunca é."""
    if os.name == "nt":
        return False
    try:
        if os.environ.get("HOME", "") == "/home/appuser":
            return True
        if "/mount/src" in os.getcwd():
            return True
    except OSError:
        pass
    return False


def is_training_capable_env() -> bool:
    """Onde DÁ pra treinar os modelos ao vivo (LightGBM em ~26k jogos = pesado).

    A Streamlit Cloud tem ~1GB de RAM: o treino lá trava ou é morto (era o
    'em retreino em background' que nunca terminava). Quem treina é o robô do
    GitHub Actions ou o PC local; a nuvem só CARREGA os modelos versionados no
    git. Por isso o auto-retreino e o botão 'Retreinar agora' são escondidos lá."""
    return not is_streamlit_cloud()


def retrain_status(models_dir: str | Path = "live_models") -> Dict[str, Any]:
    """Return a human-readable status dict for the retrain process."""
    md = Path(str(models_dir))
    out = {
        "running": is_retrain_running(md),
        "log_path": str(md / "retrain.log"),
        "log_exists": (md / "retrain.log").exists(),
        "log_tail": "",
    }
    log = md / "retrain.log"
    if log.exists():
        try:
            txt = log.read_text(encoding="utf-8", errors="replace")
            out["log_tail"] = "\n".join(txt.splitlines()[-15:])
        except OSError:
            pass
    return out
