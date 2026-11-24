# -*- coding: utf-8 -*-
"""Draft pela captura de tela (24/11/2026): monta uma captura no formato da cyberscore e confere."""
from __future__ import annotations

import io
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app_core import draft_imagem_core as R

RAIZ = Path(__file__).resolve().parents[2]


@unittest.skipUnless((RAIZ / "herois_dota_ref.npz").exists(), "sem herois_dota_ref.npz (repo do LoL)")
class TestDraftImagem(unittest.TestCase):

    def test_captura_estilo_cyberscore(self):
        ids, refs = R.carregar_refs(RAIZ)
        rng = np.random.default_rng(3)
        esc = [int(i) for i in rng.choice(ids, 24, replace=False)]
        tela = Image.new("RGB", (1250, 470), (255, 255, 255))
        d = ImageDraw.Draw(tela)

        def tile(hid, x, y, w, h, ban):
            k = int(np.where(ids == hid)[0][0])
            im = Image.fromarray(refs[k].astype(np.uint8)).resize((w, h), Image.BILINEAR)
            if ban:
                ImageDraw.Draw(im).line((0, h, w, 0), fill=(220, 40, 40), width=2)
            else:
                ImageDraw.Draw(im).rectangle((int(w * .74), int(h * .58), w - 4, h - 4), fill=(90, 110, 200))
            tela.paste(im, (x, y))
            d.rectangle((x - 1, y - 1, x + w, y + h), outline=(230, 120, 120))

        for i in range(5):
            tile(esc[i], 30 + 110 * i, 60, 100, 70, False)
            tile(esc[5 + i], 688 + 110 * i, 60, 100, 70, False)
        for i in range(7):
            tile(esc[10 + i], 30 + 78 * i, 300, 68, 52, True)
            tile(esc[17 + i], 688 + 78 * i, 300, 68, 52, True)
        b = io.BytesIO()
        tela.save(b, "PNG")
        r = R.reconhecer(b.getvalue(), ids, refs)
        self.assertEqual(r["aviso"], "")
        pe, pd_ = r["picks"]
        be, bd = r["bans"]
        self.assertEqual([x["id"] for x in pe], esc[0:5])
        self.assertEqual([x["id"] for x in pd_], esc[5:10])
        self.assertEqual([x["id"] for x in be], esc[10:17])
        self.assertEqual([x["id"] for x in bd], esc[17:24])

    def test_recorte_ruim_avisa(self):
        ids, refs = R.carregar_refs(RAIZ)
        b = io.BytesIO()
        Image.new("RGB", (400, 200), (255, 255, 255)).save(b, "PNG")
        r = R.reconhecer(b.getvalue(), ids, refs)
        self.assertIsNone(r["picks"])
        self.assertIn("preciso dos 10 picks", r["aviso"])


if __name__ == "__main__":
    unittest.main()
