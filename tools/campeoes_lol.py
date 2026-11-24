# -*- coding: utf-8 -*-
"""LISTA DE CAMPEOES DO LOL (24/11/2026) -> campeoes_lol.json, para ler o draft colado e mostrar a foto.
Fonte: Data Dragon da Riot (versao mais nova). Rodar quando sair campeao. Arquivo identico nos repos.
Uso: python tools/campeoes_lol.py"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Gabinete/0.1"}


def main() -> int:
    v = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://ddragon.leagueoflegends.com/api/versions.json", headers=UA), timeout=60).read())[0]
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json", headers=UA), timeout=60).read())
    camp = {k: c["name"] for k, c in sorted(d["data"].items())}
    (RAIZ / "campeoes_lol.json").write_text(json.dumps({"versao": v, "campeoes": camp}, ensure_ascii=False, indent=0),
                                            encoding="utf-8")
    print(f"[campeoes_lol] {len(camp)} campeoes (versao {v})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
