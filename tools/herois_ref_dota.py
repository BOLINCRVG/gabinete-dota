# -*- coding: utf-8 -*-
"""REFERENCIAS DOS HEROIS para reconhecer o draft numa captura de tela (24/11/2026).

Baixa a arte oficial de cada heroi (CDN da Valve, a mesma que a cyberscore e o proprio Dota usam),
reduz para 48x27 e grava herois_dota_ref.npz (~0,5 MB) + herois_dota_img.json ({id: nome interno},
para a tela mostrar a foto). Nomes internos: pacote dotaconstants (jsdelivr). Rodar quando sair heroi.
Uso: python tools/herois_ref_dota.py
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from app_core.draft_imagem_core import REF_H, REF_W  # noqa: E402

CONST = "https://cdn.jsdelivr.net/npm/dotaconstants/build/heroes.json"
IMG = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{}.png"
UA = {"User-Agent": "GabineteDota/0.1"}


def main() -> int:
    d = json.loads(urllib.request.urlopen(urllib.request.Request(CONST, headers=UA), timeout=60).read())
    nomes = {str(k): v["name"].replace("npc_dota_hero_", "") for k, v in d.items()}
    ids, arr = [], []
    for hid, nome in sorted(nomes.items(), key=lambda x: int(x[0])):
        b = urllib.request.urlopen(urllib.request.Request(IMG.format(nome), headers=UA), timeout=30).read()
        im = Image.open(io.BytesIO(b)).convert("RGB").resize((REF_W, REF_H), Image.BILINEAR)
        ids.append(int(hid))
        arr.append(np.asarray(im, dtype=np.uint8))
        time.sleep(0.15)
    np.savez_compressed(RAIZ / "herois_dota_ref.npz", ids=np.array(ids), img=np.stack(arr))
    (RAIZ / "herois_dota_img.json").write_text(json.dumps(nomes, indent=0), encoding="utf-8")
    print(f"[herois_ref] {len(ids)} herois -> herois_dota_ref.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
