# -*- coding: utf-8 -*-
"""Os arquivos dos robos do GitHub -- o erro que parou as odds do LoL.

O QUE ACONTECEU (24/09/2026). O robo de odds do LoL parou de salvar por ~18
horas. A leitura da Pinnacle funcionava; quebrava na hora de gravar:

    git add odds_pinnacle.json odds_pinnacle_historico.jsonl \\n   odds_pinnacle_partidas.jsonl

O "\\n" entrou LITERAL numa edicao minha de 21/09 (a quebra de linha virou
dois caracteres). No terminal do robo ele vira a letra "n", e o git recusa:
"pathspec 'n' did not match any files". O commit so chegou ao GitHub em
23/09 as 20h51 -- e a ultima leitura salva foi as 18h46. Nenhum teste lia os
arquivos dos robos.

A trava: "\\n" FORA de aspas numa linha de comando e sempre engano. Dentro
de aspas (tr -s " \\n", sep="\\n") e legitimo.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
_ASPAS = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')


def _barra_n_solta(linha: str) -> bool:
    sem_comentario = linha.split(" #", 1)[0] if not linha.lstrip().startswith("#") else ""
    return "\\n" in _ASPAS.sub("", sem_comentario)


def _ymls():
    fora = list((RAIZ / ".github" / "workflows").glob("*.yml"))
    dota = RAIZ.parent / "GabineteDOTA" / ".github" / "workflows"
    if dota.exists():
        fora += list(dota.glob("*.yml"))
    return fora


class TestNenhumaQuebraDeLinhaLiteral(unittest.TestCase):

    def test_a_trava_pega_o_caso_real(self):
        self.assertTrue(_barra_n_solta(
            "git add odds_pinnacle.json odds_pinnacle_historico.jsonl \\n     x.jsonl"))
        self.assertFalse(_barra_n_solta('quais=$(tr -s " \\n" \' \')'))
        self.assertFalse(_barra_n_solta('print("x", sep="\\n  ")'))

    def test_nenhum_robo_tem_barra_n_solta(self):
        for f in _ymls():
            for i, linha in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                self.assertFalse(_barra_n_solta(linha), "%s:%d  %s" % (f, i, linha.strip()))

    def test_o_robo_de_odds_salva_os_TRES_arquivos(self):
        for base in (RAIZ, RAIZ.parent / "GabineteDOTA"):
            yml = base / ".github" / "workflows" / "odds.yml"
            if not yml.exists():
                continue
            texto = yml.read_text(encoding="utf-8")
            linha = [l for l in texto.splitlines() if "git add odds_pinnacle.json" in l]
            self.assertEqual(len(linha), 1, yml)
            for arq in ("odds_pinnacle.json", "odds_pinnacle_historico.jsonl",
                        "odds_pinnacle_partidas.jsonl"):
                self.assertIn(arq, linha[0], yml)


if __name__ == "__main__":
    unittest.main()
