# -*- coding: utf-8 -*-
"""Testes da escolha de temporada — mlcore/temporada.py.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ O caso real que originou tudo (06/09/2026): 60 linhas da liga EBL vieram    ║
║ da fonte com `year = 2027` e data 2026-09-02. As réguas escolhiam a         ║
║ temporada por `max(year)`, iam para 2027, achavam 5 jogos, e a NOTA         ║
║ OFICIAL virava `nan` enquanto a NOTA POR LIGA estourava com KeyError.       ║
║ O vigia semanal não disparou: `nan` não é maior nem menor que o piso.       ║
║                                                                             ║
║ O que estes testes travam, e que é a regra da fonte 1 do módulo:            ║
║ ausência de amostra NÃO pode virar número — ou se escolhe uma temporada     ║
║ com dado, ou se levanta erro alto. Nunca um palpite silencioso.             ║
║ https://thicket.sh/blog/when-your-metric-is-broken-not-your-business        ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import unittest

from mlcore.temporada import (
    MIN_JOGOS_TEMPORADA,
    TemporadaInsuficiente,
    aviso,
    escolher_temporada,
)


class TestEscolhaDeTemporada(unittest.TestCase):

    def test_o_caso_real_de_2027_nao_sequestra_a_nota(self):
        """60 linhas de rotulo novo nao podem apagar 8.586 jogos de 2026."""
        anos = [2024] * 5000 + [2025] * 4000 + [2026] * 8586 + [2027] * 60
        temporada, diag = escolher_temporada(anos)
        self.assertEqual(temporada, 2026, "voltou a cair no rotulo esquisito")
        self.assertEqual(diag["n"], 8586)

    def test_a_temporada_descartada_APARECE(self):
        """Descartar em silencio seria trocar um problema por outro."""
        anos = [2026] * 8586 + [2027] * 60
        _, diag = escolher_temporada(anos)
        self.assertIn((2027, 60), diag["descartadas"])
        self.assertIn("2027", aviso(diag))
        self.assertIn("descartada", aviso(diag))

    def test_escolhe_a_MAIS_RECENTE_entre_as_que_tem_amostra(self):
        anos = [2024] * 9000 + [2025] * 9000 + [2026] * 9000
        temporada, _ = escolher_temporada(anos)
        self.assertEqual(temporada, 2026)

    def test_temporada_nova_so_entra_quando_tiver_amostra(self):
        """Vira do ano: 2027 assume sozinha assim que passar do minimo."""
        anos = [2026] * 9000 + [2027] * (MIN_JOGOS_TEMPORADA - 1)
        self.assertEqual(escolher_temporada(anos)[0], 2026)
        anos = [2026] * 9000 + [2027] * MIN_JOGOS_TEMPORADA
        self.assertEqual(escolher_temporada(anos)[0], 2027)

    def test_sem_amostra_nenhuma_LEVANTA_em_vez_de_chutar(self):
        """A regra da fonte 1: falhar alto, nunca devolver palpite."""
        with self.assertRaises(TemporadaInsuficiente):
            escolher_temporada([2026] * 10 + [2027] * 5)

    def test_coluna_vazia_ou_ilegivel_LEVANTA(self):
        with self.assertRaises(TemporadaInsuficiente):
            escolher_temporada([])
        with self.assertRaises(TemporadaInsuficiente):
            escolher_temporada(["", None, "abacaxi"])

    def test_valores_ilegiveis_nao_derrubam_os_legiveis(self):
        anos = [2026] * 9000 + [None, "", "xx"] * 10
        self.assertEqual(escolher_temporada(anos)[0], 2026)

    def test_season_explicito_vence(self):
        anos = [2024] * 9000 + [2026] * 9000
        temporada, diag = escolher_temporada(anos, explicito=2024)
        self.assertEqual(temporada, 2024)
        self.assertTrue(diag["forcada"])

    def test_season_explicito_vazio_AVISA_que_a_nota_nao_vale(self):
        """Forcar uma temporada sem jogos e permitido, calar sobre isso nao."""
        anos = [2026] * 9000 + [2027] * 5
        _, diag = escolher_temporada(anos, explicito=2027)
        self.assertEqual(diag["n"], 5)
        self.assertIn("ATENCAO", aviso(diag))

    def test_o_aviso_sai_mesmo_quando_esta_tudo_bem(self):
        """Regua que so fala quando quebra nao ensina a ler o normal."""
        texto = aviso(escolher_temporada([2026] * 9000)[1])
        self.assertIn("temporada medida: 2026", texto)
        self.assertIn("9000", texto)


if __name__ == "__main__":
    unittest.main()
