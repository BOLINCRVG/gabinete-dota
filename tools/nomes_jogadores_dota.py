# -*- coding: utf-8 -*-
"""NOMES PROFISSIONAIS dos jogadores do Dota (23/11/2026) -> nomes_jogadores_dota.json.

Para a tela de draft colado do SUPER-LIVE: a base guarda o jogador pela conta (account_id); a tela
precisa mostrar "Yatoro", nao "321580662". Fonte: OpenDota /api/proPlayers (1 chamada por rodada do
robo), campo `name` = nome profissional (publico, ligado a carreira, como o playername do LoL).
NAO guarda apelido/perfil Steam, nome real, pais (tools/vigia_privacidade.py).
So entram as contas dos elencos (camada_draft_dota.json). Falhou: fica o arquivo velho.
Uso: python tools/nomes_jogadores_dota.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1]
ARQ = RAIZ / "nomes_jogadores_dota.json"
URL = "https://api.opendota.com/api/proPlayers"
UA = "GabineteDota/0.1 (" + ("https://github.com/" + os.environ["GITHUB_REPOSITORY"]
                             if os.environ.get("GITHUB_REPOSITORY") else "modelo pessoal") + ")"


def contas_dos_elencos(raiz: Path = RAIZ) -> set:
    """Contas dos elencos da camada de draft (o ultimo jogo de cada time; tools/camada_draft.py)."""
    try:
        el = json.loads((raiz / "camada_draft_dota.json").read_text(encoding="utf-8")).get("elencos") or {}
    except (OSError, ValueError):
        return set()
    return {str(a) for t in el.values() for a in (t.get("jogadores") or [])}


def montar(pro: list, contas: set) -> dict:
    """Pura: lista do /proPlayers -> {conta: nome profissional} so das contas pedidas."""
    out = {}
    for p in pro or []:
        a, n = str(p.get("account_id") or ""), str(p.get("name") or "").strip()
        if a in contas and n:
            out[a] = n
    return dict(sorted(out.items()))


def main() -> int:
    contas = contas_dos_elencos()
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=90) as r:
            b = r.read()
        if b[:2] == b"\x1f\x8b":
            b = gzip.decompress(b)
        pro = json.loads(b.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[nomes_jogadores] OpenDota nao respondeu ({type(e).__name__}: {e}); fica o arquivo anterior")
        return 0
    nomes = montar(pro, contas)
    doc = {"gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "fonte": "OpenDota /api/proPlayers (nome profissional)", "nomes": nomes}
    ARQ.write_text(json.dumps(doc, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[nomes_jogadores] {len(nomes)} de {len(contas)} contas dos elencos com nome -> {ARQ.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
