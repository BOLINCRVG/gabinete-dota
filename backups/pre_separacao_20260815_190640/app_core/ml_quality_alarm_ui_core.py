"""Banner do alarme de qualidade do ML (lê ops/monitoring/ml_quality_last_check.json).

O check é escrito por tools/ml_quality_check.py (rodado pela manutenção após qualquer
mudança de modelo). Aqui só renderizamos: fail → banner vermelho; warn → amarelo.
Verde/ok não renderiza nada (não poluir a UI).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_last_quality_check(root: Path) -> dict[str, Any]:
    p = Path(root) / "ops" / "monitoring" / "ml_quality_last_check.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def render_ml_quality_alarm(*, st: Any, root: Path) -> None:
    data = read_last_quality_check(root)
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    verdict = str(result.get("verdict") or "").strip().lower()
    if verdict not in ("fail", "warn", "error"):
        return
    reason = str(result.get("reason") or "").strip()
    checked = str(data.get("checked_at") or "")[:16].replace("T", " ")
    if verdict == "fail":
        st.error(
            f"🚨 **ALARME DE QUALIDADE DO ML** — {reason} "
            f"(check {checked} UTC). As probabilidades do modelo podem estar degradadas; "
            "NÃO confie nas odds do ML até resolver (rode `tools/atualizar_modelo.py` ou restaure backup)."
        )
    elif verdict == "error":
        st.warning(
            f"⚠️ Checagem de qualidade do ML falhou ao rodar ({reason or 'erro desconhecido'}; check {checked} UTC). "
            "Sem veredito sobre o modelo atual."
        )
    else:
        st.warning(
            f"⚠️ Qualidade do ML levemente abaixo do baseline — {reason} (check {checked} UTC)."
        )
