# -*- coding: utf-8 -*-
"""DRAFT PELA CAPTURA DE TELA (24/11/2026). Ele: "estou pegando do cyberscore, e para isso, para ser
mais rapido, eu vou pegar na funcao de screenshot".

Como funciona (so numpy/Pillow/scipy -- nada de dependencia nova na nuvem):
  1. acha os retratos: regioes retangulares (16:9 a ~4:3) que destoam do fundo;
  2. separa pelo TAMANHO: os 10 maiores e parecidos = picks; os menores parecidos = bans;
  3. lado esquerdo/direito pelo centro; ordem da esquerda para a direita (na cyberscore = pos 1..5);
  4. cada retrato e comparado (correlacao normalizada, 48x27 RGB) com a arte oficial dos herois
     (herois_dota_ref.npz). Pick: o canto do icone de posicao fica de fora da comparacao.
Medido em 24/11/2026: 127/127 herois certos em tamanho de pick e de ban (com JPEG, icone e risco de
ban simulados) e a captura real dele (Synapse x Angry Birds) lida inteira.
Duvida (1o e 2o colocados perto) NUNCA e chutada calada: volta marcada para ele confirmar.
Arquivo identico nos repositorios.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

REF_W, REF_H = 48, 27
MARGEM_DUVIDA = 0.06       # 1o - 2o abaixo disto: marcado para confirmar
SCORE_MIN = 0.35           # abaixo disto: nao reconhecido


def carregar_refs(raiz: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        z = np.load(Path(raiz) / "herois_dota_ref.npz")
        return z["ids"], z["img"].astype(np.float32)
    except (OSError, KeyError, ValueError):
        return None


def caixas(im: Image.Image) -> List[Tuple[int, int, int, int]]:
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    borda = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fundo = np.median(borda, axis=0)
    dif = np.abs(a - fundo).sum(axis=2) > 60
    dif = ndimage.binary_opening(dif, iterations=1)
    lab, _ = ndimage.label(dif)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if w < 25 or h < 14:
            continue
        if 1.05 <= w / h <= 2.1 and (lab[sl] == i).mean() > 0.6:
            out.append((sl[1].start, sl[0].start, w, h))
    return out


def _vetor(crop: Image.Image, icone: bool) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(crop.resize((REF_W, REF_H), Image.BILINEAR), dtype=np.float32)
    m = np.ones((REF_H, REF_W), bool)
    if icone:
        m[int(REF_H * 0.5):, int(REF_W * 0.66):] = False
    return v, m


def _identificar(v: np.ndarray, m: np.ndarray, ids: np.ndarray, refs: np.ndarray) -> List[Tuple[float, int]]:
    x = v[m].ravel()
    x = x - x.mean()
    R = refs[:, m].reshape(len(refs), -1)
    R = R - R.mean(axis=1, keepdims=True)
    s = (R @ x) / (np.linalg.norm(R, axis=1) * np.linalg.norm(x) + 1e-9)
    o = np.argsort(-s)[:3]
    return [(float(s[i]), int(ids[i])) for i in o]


def _grupo(bx: List[Tuple[int, int, int, int]], area_ref: float, tol: float) -> List[Tuple[int, int, int, int]]:
    return [b for b in bx if abs(b[2] * b[3] - area_ref) <= tol * area_ref]


def reconhecer(img_bytes: bytes, ids: np.ndarray, refs: np.ndarray) -> Dict[str, Any]:
    """-> {'picks': (esq[5 ids], dir[5]), 'bans': (esq[], dir[]), 'duvidas': [...], 'aviso': str}
    Cada item de picks/bans e {'id', 'score', 'alternativas': [(id, score)...], 'duvida': bool}."""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bx = sorted(caixas(im), key=lambda b: -b[2] * b[3])
    if len(bx) < 10:
        return {"picks": None, "bans": ([], []), "duvidas": [],
                "aviso": f"Achei só {len(bx)} retrato(s) na imagem; preciso dos 10 picks. Confira o recorte."}
    picks = _grupo(bx, float(np.median([b[2] * b[3] for b in bx[:10]])), 0.15)[:10]
    resto = [b for b in bx if b not in picks]
    bans = []
    if resto:
        area_ban = float(np.median([b[2] * b[3] for b in resto[:14]]))
        bans = [b for b in _grupo(resto, area_ban, 0.2) if b[2] * b[3] < 0.85 * min(p[2] * p[3] for p in picks)]
    xs = [b[0] + b[2] / 2 for b in picks]
    meio = (min(xs) + max(xs)) / 2

    def ler(b, icone):
        x, y, w, h = b
        mg = int(round(min(w, h) * 0.06))
        v, m = _vetor(im.crop((x + mg, y + mg, x + w - mg, y + h - mg)), icone)
        top = _identificar(v, m, ids, refs)
        return {"id": top[0][1], "score": top[0][0], "alternativas": top[1:],
                "duvida": top[0][0] < SCORE_MIN or top[0][0] - top[1][0] < MARGEM_DUVIDA}

    def lados(caixas_, icone):
        esq = sorted([b for b in caixas_ if b[0] + b[2] / 2 < meio], key=lambda b: (b[1] // 20, b[0]))
        dir_ = sorted([b for b in caixas_ if b[0] + b[2] / 2 >= meio], key=lambda b: (b[1] // 20, b[0]))
        return [ler(b, icone) for b in esq], [ler(b, icone) for b in dir_]

    pe, pd_ = lados(picks, True)
    be, bd = lados(bans, False)
    aviso = ""
    if len(pe) != 5 or len(pd_) != 5:
        aviso = f"Li {len(pe)} + {len(pd_)} picks (esperava 5 + 5). Confira o recorte da imagem."
    duvidas = [(lado, tipo, i) for tipo, (a, b) in (("pick", (pe, pd_)), ("ban", (be, bd)))
               for lado, lst in (("esquerda", a), ("direita", b)) for i, x in enumerate(lst) if x["duvida"]]
    return {"picks": (pe, pd_), "bans": (be, bd), "duvidas": duvidas, "aviso": aviso}
