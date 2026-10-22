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


# ----------------------------------------------------------------------------
# O VIGIA QUE NUNCA PUBLICOU (achado em 16/10/2026). O nota_semanal.yml tinha
# `contents: read` desde 17/07: o push era recusado, e o laco de retry
#     for i in 1 2 3; do if git push; then break; fi; git pull --rebase; done
# terminava no `git pull --rebase`, que da certo -- o passo ficava VERDE sem
# gravar nada. Ninguem viu por 3 meses: a nota do produto e o frescor ficaram
# parados em 07/09. Duas travas: (1) quem faz `git push` declara escrita;
# (2) laco de push termina num `git push` que falha alto, nunca num break.
# FONTES (aberta em 16/10/2026):
# https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax
#   "If you specify the access for any of these permissions, all of those that
#   are not specified are set to `none`." e "`contents: read` permits an action
#   to list the commits". Que push exige `write` e prova empirica deste repo:
#   os 6 robos com write publicam todo dia, o unico com read nunca publicou.
_PERM_CONTENTS = re.compile(r"^permissions:[ \t]*\n((?:[ \t]+.*\n?)+)", re.M)


def _contents_permitido(texto: str):
    m = _PERM_CONTENTS.search(texto)
    if not m:
        return None          # sem bloco: vale o padrao do repositorio
    achou = re.search(r"^\s*contents:\s*(\w+)", m.group(1), re.M)
    return achou.group(1) if achou else "none"


def _push_engolido(texto: str) -> bool:
    return bool(re.search(r"if git push; then break; fi|git push && break", texto))


class TestQuemPublicaPodeEscrever(unittest.TestCase):

    def test_a_trava_pega_o_caso_real(self):
        velho = ("permissions:\n  contents: read\n  issues: write\n\njobs:\n"
                 "            for i in 1 2 3; do\n              if git push; then break; fi\n")
        self.assertEqual(_contents_permitido(velho), "read")
        self.assertTrue(_push_engolido(velho))
        bom = "permissions:\n  contents: write\n\njobs:\n  if git push; then exit 0; fi\n  git push\n"
        self.assertEqual(_contents_permitido(bom), "write")
        self.assertFalse(_push_engolido(bom))

    def test_todo_robo_que_faz_push_declara_escrita(self):
        for f in _ymls():
            texto = f.read_text(encoding="utf-8")
            if "git push" not in texto:
                continue
            self.assertEqual(_contents_permitido(texto), "write",
                             "%s faz git push mas nao declara `contents: write`" % f)

    def test_nenhum_laco_de_push_engole_a_recusa(self):
        for f in _ymls():
            self.assertFalse(_push_engolido(f.read_text(encoding="utf-8")),
                             "%s: laco de push que termina em break esconde a recusa" % f)


if __name__ == "__main__":
    unittest.main()
