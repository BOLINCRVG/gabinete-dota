# -*- coding: utf-8 -*-
"""Movimento do ranking (↑/↓) — o "quem subiu?" que toda ranking de esports tem.

Pesquisa (regra dele, 22/07): a coluna de MOVIMENTO é o elemento nº 1 que
separa uma ranking viva de uma tabela estática — Liquipedia/Riot GPR/GosuGamers
todas mostram "subiu/caiu desde o último snapshot", e os guias de UI insistem em
destacar os *latest movers*, não só a ordem estável.

Como funciona: guardamos um SNAPSHOT (time -> posição) com data. A cada abertura,
comparamos o ranking atual com esse snapshot e mostramos a variação. O snapshot
só é reescrito quando fica velho (min_age_days) — senão o movimento seria sempre
zero (comparar com si mesmo).

Puro/testável: nada de Streamlit ou arquivo aqui além dos helpers de I/O
explícitos, para o cálculo do movimento ser coberto por teste.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from mlcore.team_artifact_normalize import _norm_team_key


def rank_map_from_df(df: Any, *, team_col: str = "team", rank_col: str = "rank") -> Dict[str, int]:
    """{chave_normalizada_do_time -> posição} a partir da tabela de ranking."""
    out: Dict[str, int] = {}
    if df is None or getattr(df, "empty", True):
        return out
    cols = getattr(df, "columns", [])
    if team_col not in cols:
        return out
    has_rank = rank_col in cols
    for i, (_, r) in enumerate(df.iterrows()):
        k = _norm_team_key(r.get(team_col))
        if not k or k in out:
            continue
        try:
            pos = int(r.get(rank_col)) if has_rank else (i + 1)
        except (TypeError, ValueError):
            pos = i + 1
        out[k] = pos
    return out


def compute_movement(current: Mapping[str, int], snapshot: Mapping[str, int]) -> Dict[str, Any]:
    """Variação de posição por time: {chave -> {mov, novo}}.

    mov > 0  => SUBIU (posição menor agora); mov < 0 => caiu; 0 = parado.
    novo = True quando o time não estava no snapshot anterior."""
    prev = (snapshot or {}).get("ranks") if isinstance(snapshot, dict) and "ranks" in snapshot else snapshot
    prev = prev if isinstance(prev, Mapping) else {}
    out: Dict[str, Any] = {}
    for k, pos_now in (current or {}).items():
        pos_old = prev.get(k)
        if pos_old is None:
            out[k] = {"mov": 0, "novo": True}
        else:
            try:
                out[k] = {"mov": int(pos_old) - int(pos_now), "novo": False}
            except (TypeError, ValueError):
                out[k] = {"mov": 0, "novo": False}
    return out


def movement_badge_html(info: Mapping[str, Any] | None) -> str:
    """Seta colorida para a célula: ▲n verde / ▼n vermelho / — cinza / ✦ novo."""
    if not isinstance(info, Mapping):
        return "<span style='opacity:0.35;'>—</span>"
    if info.get("novo"):
        return "<span style='color:#2daac4;font-weight:700;' title='novo no ranking'>✦</span>"
    mov = int(info.get("mov", 0) or 0)
    if mov > 0:
        return f"<span style='color:#22c55e;font-weight:700;'>▲{mov}</span>"
    if mov < 0:
        return f"<span style='color:#ef4444;font-weight:700;'>▼{abs(mov)}</span>"
    return "<span style='opacity:0.4;'>—</span>"


def _days_since(iso: Any) -> float:
    try:
        d = datetime.fromisoformat(str(iso)).date()
    except (TypeError, ValueError):
        return float("inf")
    return float((date.today() - d).days)


def load_snapshot(path: str | Path) -> Dict[str, Any]:
    """Carrega o snapshot salvo; {} se ausente/corrompido (nunca derruba)."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return {}
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def snapshot_is_stale(snapshot: Mapping[str, Any], *, min_age_days: float) -> bool:
    """O snapshot precisa ser reescrito? (ausente, sem data, ou velho o bastante)."""
    if not isinstance(snapshot, Mapping) or not snapshot.get("ranks"):
        return True
    return _days_since(snapshot.get("date")) >= float(min_age_days)


def save_snapshot(path: str | Path, current: Mapping[str, int]) -> bool:
    """Grava o snapshot atual com a data de hoje. False em falha (sem derrubar)."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"date": date.today().isoformat(),
                   "ranks": {str(k): int(v) for k, v in (current or {}).items()}}
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except (OSError, TypeError, ValueError):
        return False
