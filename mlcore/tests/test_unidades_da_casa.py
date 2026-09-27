# -*- coding: utf-8 -*-
"""A unidade como a CASA escreve, e nao como o nosso motor escreve.

ELE PERGUNTOU (22/09/2026): "onde eu consigo ver as outras linhas, kill, torre
etc". A resposta medida foi desconfortavel: abates, dragoes e baroes ja
apareciam, mas TORRES nao -- e sumiam caladas.

A causa era so o nome. A Pinnacle manda a unidade assim:

    Kills    -> casava com "kills"     -> 3.674 linhas precificadas
    Dragons  -> casava com "dragons"   ->   242 linhas
    Barons   -> casava com "barons"    ->   239 linhas
    Turrets  -> NAO casava com nada    ->   225 linhas PERDIDAS

"towers" e o nome do motor; "turrets" e o nome dela. A tabela e indexada pela
grafia DELA, entao os dois precisam estar la.

O RISCO PERMANENTE, e o motivo do ultimo teste deste arquivo: quando a casa
abrir um mercado novo, ou mudar a grafia de um, a linha volta a sumir em
silencio. O teste le o historico de verdade e reprova se aparecer unidade de
TOTAL que ninguem sabe traduzir. Ele falha sozinho no dia em que isso
acontecer, sem depender de alguem reparar.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from app_core.oportunidades_core import UNIDADE_PARA_METRICA, metrica_da_linha

RAIZ = Path(__file__).resolve().parents[2]


def _linha(unidade, tipo="total", lado="over", pontos=27.5, periodo=1):
    return {"unidade": unidade, "tipo": tipo, "lado": lado,
            "pontos": pontos, "periodo": periodo}


class TestAsUnidadesQueACasaMANDA(unittest.TestCase):

    def test_TORRES_o_caso_que_estava_quebrado(self):
        self.assertEqual(metrica_da_linha(_linha("Turrets")), ("towers", "m1"))

    def test_abates_dragoes_e_baroes(self):
        self.assertEqual(metrica_da_linha(_linha("Kills"))[0], "kills")
        self.assertEqual(metrica_da_linha(_linha("Dragons"))[0], "dragons")
        self.assertEqual(metrica_da_linha(_linha("Barons"))[0], "barons")

    def test_a_grafia_do_MOTOR_continua_valendo(self):
        """Se um dia a casa passar a escrever "Towers", nada quebra."""
        self.assertEqual(metrica_da_linha(_linha("Towers"))[0], "towers")

    def test_maiuscula_minuscula_nao_importa(self):
        for g in ("TURRETS", "turrets", "TuRrEtS"):
            self.assertEqual(metrica_da_linha(_linha(g))[0], "towers")

    def test_o_mapa_certo_vai_junto(self):
        for p in (1, 2, 3, 4, 5):
            self.assertEqual(metrica_da_linha(_linha("Turrets", periodo=p))[1],
                             "m%d" % p)

    def test_a_SERIE_nao_entra_porque_o_motor_fala_de_MAPA(self):
        self.assertIsNone(metrica_da_linha(_linha("Kills", periodo=0)))

    def test_unidade_desconhecida_devolve_NONE_e_nao_chuta(self):
        """Chutar aqui poria uma aposta de dragao na conta de abates."""
        self.assertIsNone(metrica_da_linha(_linha("Coisa Nova")))

    def test_handicap_e_vencedor_nao_passam_por_aqui(self):
        """Eles pedem a DIFERENCA entre os times, nao o total."""
        self.assertIsNone(metrica_da_linha(_linha("Kills", tipo="spread")))
        self.assertIsNone(metrica_da_linha(_linha("Kills", tipo="moneyline")))


class TestOHistoricoDeVerdadeNaoTemOrfao(unittest.TestCase):
    """O teste que falha sozinho quando a casa muda alguma coisa."""

    def _totais(self):
        alvo = RAIZ / "odds_pinnacle_historico.jsonl"
        if not alvo.exists():
            self.skipTest("o historico nao esta nesta maquina")
        fora = []
        for linha in alvo.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha:
                continue
            try:
                x = json.loads(linha)
            except ValueError:
                continue
            if x.get("tipo") == "total" and x.get("unidade"):
                fora.append(x)
        return fora

    def test_TODA_unidade_de_total_ja_coletada_sabe_virar_metrica(self):
        orfas = {}
        for x in self._totais():
            if metrica_da_linha(x) is None and 1 <= int(x.get("periodo") or 0) <= 5:
                u = str(x.get("unidade"))
                orfas[u] = orfas.get(u, 0) + 1
        self.assertEqual(
            orfas, {},
            "a casa manda unidade(s) que ninguem traduz: %s -- acrescente em "
            "UNIDADE_PARA_METRICA" % orfas)

    def test_as_torres_do_historico_realmente_precificam(self):
        t = [x for x in self._totais() if str(x.get("unidade")).lower() == "turrets"]
        if not t:
            self.skipTest("ainda nao ha linha de torres coletada")
        self.assertTrue(all(metrica_da_linha(x) is not None for x in t))
        self.assertGreater(len(t), 100, "sobraram poucas torres no historico")


class TestATabelaEhIndexadaPelaGrafiaDaCASA(unittest.TestCase):

    def test_turrets_esta_na_tabela(self):
        self.assertIn("turrets", UNIDADE_PARA_METRICA)

    def test_turrets_e_towers_apontam_para_o_MESMO_lugar(self):
        self.assertEqual(UNIDADE_PARA_METRICA["turrets"],
                         UNIDADE_PARA_METRICA["towers"])

    def test_todas_as_chaves_sao_minusculas(self):
        """A busca e feita em minusculas; chave maiuscula nunca casaria."""
        for k in UNIDADE_PARA_METRICA:
            self.assertEqual(k, k.lower(), "chave %r nunca vai casar" % k)


if __name__ == "__main__":
    unittest.main()
