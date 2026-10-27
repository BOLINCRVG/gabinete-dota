# -*- coding: utf-8 -*-
"""Mantem o arquivo VERSIONADO de elencos por team_id (27/10/2026).

A uniao de ids pelo elenco (eval_whr_dota.roster_identity_map) lia so o
raw_matches.jsonl -- que nao vai para o git e, no robo, so tem as partidas
baixadas no dia. Em producao a uniao quase nao rodava, e um time com dois ids
(PARIVISION) ficava partido. Regra dele: historico partido, nunca.

Este script junta o que ja esta em elencos_times_dota.json com o raw_matches.jsonl
(quando existe) e guarda os ultimos MANTER jogos de cada team_id:
  {"team_id": [[start_time, [account_id, ...]], ...]}
Roda no robo depois do daily_sync; no PC dele, uma vez, com o historico todo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval_whr_dota import ELENCOS_FILE, _rosters_por_team_id  # noqa: E402

MANTER = 30   # a uniao olha os ultimos 20 (ROSTER_ULTIMOS_N); folga para o futuro


def compactar(por_id: dict) -> dict:
    """Pura: team_id -> ultimos MANTER [instante, contas ordenadas]."""
    return {str(i): [[int(st), sorted(int(a) for a in accs)] for st, accs in lst[-MANTER:]]
            for i, lst in sorted(por_id.items()) if lst}


def main() -> int:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "raw_matches.jsonl"
    destino = RAIZ / ELENCOS_FILE
    por_id = _rosters_por_team_id(raw)            # cru + o compacto da pasta do cru
    if raw.parent.resolve() != RAIZ.resolve():    # cru de outra pasta: soma o compacto daqui
        for i, lst in _rosters_por_team_id(RAIZ / "_nao_existe.jsonl").items():
            por_id[i] = sorted(set(por_id.get(i, [])) | set(lst), key=lambda x: x[0])
    novo = compactar(por_id)
    destino.write_text(json.dumps(novo, separators=(",", ":")), encoding="utf-8")
    print(f"[elencos] {len(novo)} team_id com elenco; raw {'lido' if raw.exists() else 'ausente'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
