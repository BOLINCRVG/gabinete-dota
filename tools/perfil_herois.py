# -*- coding: utf-8 -*-
"""PERFIL DE CADA HEROI/CAMPEAO (24/11/2026) -> perfil_herois_<esporte>.json, para a linha DESCRITIVA da comp
na tela do draft ("Radiant: 3 iniciadores, 4 de alcance..."). NAO entra no preco: medido em 24/11/2026,
as interacoes de estilo (alcance x engage, dive x protecao...) nao melhoraram a previsao.
Dota: dotaconstants (papeis da OpenDota, ranged, alcance). LoL: wiki oficial (Module:ChampionData: classe,
ranged, alcance, dano magico/fisico). Rodar quando sair heroi/campeao. Arquivo identico nos repos.
Uso: python tools/perfil_herois.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from app_core.camada_draft_core import chave_campeao  # noqa: E402

E_DOTA = (RAIZ / "run_phase6_live_dota.py").exists()
UA = {"User-Agent": "Gabinete/0.1 (modelo pessoal)"}


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def perfil_dota() -> dict:
    d = json.loads(_get("https://cdn.jsdelivr.net/npm/dotaconstants/build/heroes.json"))
    return {str(k): {"papeis": list(v.get("roles") or []), "ranged": v.get("attack_type") == "Ranged",
                     "alcance": int(v.get("attack_range") or 0)} for k, v in d.items()}


def ler_wiki_lol(txt: str) -> dict:
    """Pura: Module:ChampionData (Lua) -> {chave normalizada: {classes, ranged, alcance, magico}}."""
    out = {}
    inicios = list(re.finditer(r'\n  \["([^"]+)"\] = \{', txt))   # cada entrada vai ate o comeco da proxima
    for k, m in enumerate(inicios):
        nome = m.group(1)
        b = txt[m.end():inicios[k + 1].start() if k + 1 < len(inicios) else len(txt)]
        roles = re.findall(r'"([A-Za-z]+)"', (re.search(r'\["role"\]\s*=\s*\{([^}]*)\}', b) or [None, ""])[1])
        tags = (re.search(r'\["client_tags"\]\s*=\s*\{([^}]*)\}', b) or [None, ""])[1]
        rng = re.search(r'\["range"\]\s*=\s*([0-9.]+)', b)
        rt = re.search(r'\["rangetype"\]\s*=\s*"([A-Za-z]+)"', b)
        ad = re.search(r'\["adaptivetype"\]\s*=\s*"([A-Za-z]+)"', b)
        out[chave_campeao(nome)] = {"classes": roles, "longo_alcance": "Long Range" in tags,
                                    "ranged": bool(rt and rt.group(1) == "Ranged"),
                                    "alcance": int(float(rng.group(1))) if rng else 0,
                                    "magico": bool(ad and ad.group(1) == "Magic")}
    return out


def main() -> int:
    if E_DOTA:
        perfil, esporte = perfil_dota(), "dota"
    else:
        txt = _get("https://wiki.leagueoflegends.com/en-us/Module:ChampionData/data?action=raw").decode("utf-8")
        perfil, esporte = ler_wiki_lol(txt), "lol"
    destino = RAIZ / f"perfil_herois_{esporte}.json"
    destino.write_text(json.dumps(perfil, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[perfil_herois] {len(perfil)} -> {destino.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
