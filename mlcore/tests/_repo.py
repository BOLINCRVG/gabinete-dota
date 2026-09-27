# -*- coding: utf-8 -*-
"""Em qual repositorio o teste esta rodando -- LoL ou Dota.

POR QUE (27/09/2026). O Dota recebe o motor do LoL inteiro, TESTES juntos
(sync_vendor.py copia mlcore/). A suite do Dota ficou meses com 15 falhas e 30
erros "conhecidos": nenhum era defeito do Dota -- eram testes do LoL
procurando plays_app.py e ferramentas de tools/ que so existem no LoL,
fotografias do modelo do LoL, campeoes do LoL. Esse vermelho permanente
escondia qualquer erro de verdade que aparecesse.

Um teste que so vale para o LoL leva `@so_no_lol("motivo")`: no LoL roda
normalmente; no Dota aparece como PULADO, com o motivo. Nunca se pula por
"o teste falhou" -- so por "este teste e de outro produto".
"""
from __future__ import annotations

import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
E_O_DOTA = (RAIZ / "dota_paths.py").exists()


def so_no_lol(motivo: str):
    return unittest.skipIf(E_O_DOTA, "so no LoL: " + motivo)
