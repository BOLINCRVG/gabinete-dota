# -*- coding: utf-8 -*-
"""Testes da leitura das odds coletadas.

O ponto todo destes testes: a tela vazia tem de saber DIZER por que esta
vazia. "Nao ha aposta com vantagem" e "nao consegui ler o arquivo" parecem a
mesma coisa em tela, e sao coisas muito diferentes para quem vai apostar.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app_core.odds_coletadas_core import (
    HORAS_ATE_ENVELHECER,
    carregar_odds_coletadas,
    times_da_coleta,
)


def _arquivo(tmp: Path, conteudo, nome="odds_pinnacle.json"):
    alvo = tmp / nome
    if isinstance(conteudo, str):
        alvo.write_text(conteudo, encoding="utf-8")
    else:
        alvo.write_text(json.dumps(conteudo, ensure_ascii=False), encoding="utf-8")
    return tmp


def _agora(menos_horas=0.0):
    return (datetime.now(timezone.utc) - timedelta(hours=menos_horas)).isoformat()


def _bom(idade_horas=0.0):
    return {
        "lido_em": _agora(idade_horas),
        "jogos": [{"id": 1, "liga": "LPL", "casa": "Top Esports", "fora": "Invictus"}],
        "linhas": [{"jogo": 1, "tipo": "total", "periodo": 1, "lado": "over",
                    "pontos": 31.5, "odd": 2.09, "unidade": "Kills"}],
    }


class TestLeituraNormal(unittest.TestCase):

    def test_le_jogos_e_linhas(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), _bom()))
            self.assertEqual(len(r["jogos"]), 1)
            self.assertEqual(len(r["linhas"]), 1)
            self.assertEqual(r["motivo"], "")

    def test_a_idade_vem_junto(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), _bom(2.0)))
            self.assertAlmostEqual(r["idade_horas"], 2.0, delta=0.05)


class TestOddVelhaEMarcada(unittest.TestCase):
    """Odd e preco vivo: a de horas atras pode ja nao existir.

    Mostrar vantagem calculada sobre preco velho e mandar ele apostar num
    numero que sumiu da tela da casa.
    """

    def test_leitura_recente_nao_e_velha(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(carregar_odds_coletadas(_arquivo(Path(d), _bom(1.0)))["velha"])

    def test_passou_da_janela_do_robo_fica_marcada(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), _bom(HORAS_ATE_ENVELHECER + 1)))
            self.assertTrue(r["velha"])

    def test_sem_carimbo_nao_INVENTA_idade(self):
        """PRINCIPIO IV: sem hora, "velha" nao vira False por conveniencia.

        False diria "esta fresca", que e uma afirmacao que ninguem mediu.
        """
        with tempfile.TemporaryDirectory() as d:
            dados = _bom()
            dados.pop("lido_em")
            r = carregar_odds_coletadas(_arquivo(Path(d), dados))
            self.assertIsNone(r["idade_horas"])

    def test_carimbo_ilegivel_nao_estoura(self):
        with tempfile.TemporaryDirectory() as d:
            dados = _bom()
            dados["lido_em"] = "ontem de tarde"
            r = carregar_odds_coletadas(_arquivo(Path(d), dados))
            self.assertIsNone(r["idade_horas"])
            self.assertEqual(len(r["linhas"]), 1)


class TestQuandoNaoDaParaLer(unittest.TestCase):
    """Cada jeito de falhar tem de sair com um motivo DIFERENTE e util."""

    def test_arquivo_ausente_diz_que_o_robo_nao_coletou(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(Path(d))
            self.assertEqual(r["linhas"], [])
            self.assertIn("ainda nao coletou", r["motivo"])

    def test_arquivo_corrompido_diz_que_esta_ilegivel(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), "{isto nao e json"))
            self.assertIn("ilegivel", r["motivo"])

    def test_formato_inesperado_e_dito(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), [1, 2, 3]))
            self.assertIn("formato inesperado", r["motivo"])

    def test_sem_a_lista_de_linhas_e_dito(self):
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), {"jogos": [], "linhas": "nao e lista"}))
            self.assertIn("sem a lista", r["motivo"])

    def test_coleta_vazia_e_DIFERENTE_de_erro(self):
        """O robo rodou e a casa nao tinha jogo. Nao e falha."""
        with tempfile.TemporaryDirectory() as d:
            r = carregar_odds_coletadas(_arquivo(Path(d), {"lido_em": _agora(),
                                                           "jogos": [], "linhas": []}))
            self.assertIn("nao trouxe linha", r["motivo"])
            self.assertNotIn("ilegivel", r["motivo"])

    def test_todo_motivo_de_falha_e_diferente_dos_outros(self):
        """Motivos iguais nao ajudam a achar a causa."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ausente = carregar_odds_coletadas(tmp)["motivo"]
            ilegivel = carregar_odds_coletadas(_arquivo(tmp, "{"))["motivo"]
            formato = carregar_odds_coletadas(_arquivo(tmp, [1]))["motivo"]
            vazio = carregar_odds_coletadas(
                _arquivo(tmp, {"lido_em": _agora(), "jogos": [], "linhas": []}))["motivo"]
        self.assertEqual(len({ausente, ilegivel, formato, vazio}), 4)

    def test_lixo_no_meio_das_linhas_nao_derruba_o_resto(self):
        with tempfile.TemporaryDirectory() as d:
            dados = _bom()
            dados["linhas"] = ["lixo", None, dados["linhas"][0]]
            r = carregar_odds_coletadas(_arquivo(Path(d), dados))
            self.assertEqual(len(r["linhas"]), 1)


class TestOsNomesQuePrecisamCasar(unittest.TestCase):

    def test_lista_os_times_como_a_CASA_escreve(self):
        odds = {"jogos": [{"casa": "Top Esports", "fora": "Invictus"},
                          {"casa": "Fuego", "fora": "9z"}]}
        self.assertEqual(times_da_coleta(odds),
                         ["Top Esports", "Invictus", "Fuego", "9z"])

    def test_time_que_joga_duas_vezes_aparece_uma(self):
        odds = {"jogos": [{"casa": "T1", "fora": "Gen.G"},
                          {"casa": "T1", "fora": "DK"}]}
        self.assertEqual(times_da_coleta(odds), ["T1", "Gen.G", "DK"])

    def test_sem_jogos_devolve_lista_vazia(self):
        self.assertEqual(times_da_coleta({}), [])
        self.assertEqual(times_da_coleta({"jogos": None}), [])

    def test_lixo_na_lista_nao_derruba(self):
        self.assertEqual(times_da_coleta({"jogos": ["x", None, {"casa": "T1", "fora": ""}]}),
                         ["T1"])


class TestSemRede(unittest.TestCase):
    """O app nunca vai a casa de apostas. Quem vai e o robo."""

    def test_o_modulo_nao_importa_requests(self):
        """PROCURA REDE, NAO A PALAVRA "http".

        A primeira versao deste teste procurava a substring "http" em
        qualquer lugar do arquivo. Reprovou sozinha em 16/09/2026, quando o
        modulo ganhou o rotulo `http_inesperado` -- um NOME DE MOTIVO, nada
        a ver com ir a rede.

        Teste grosseiro demais tem dois custos: reprova o que esta certo, e
        quem se acostuma a ver esse tipo de falha passa a ignorar a proxima,
        que pode ser de verdade. Agora a checagem e do que de fato leva a
        rede: os imports e os enderecos.
        """
        import inspect

        from app_core import odds_coletadas_core as m
        src = inspect.getsource(m)
        for proibido in ("import requests", "import urllib", "import socket",
                         "import http.client", "urlopen", "http://", "https://"):
            self.assertNotIn(proibido, src,
                             "%s levaria o app a rede -- quem coleta e o robo" % proibido)


if __name__ == "__main__":
    unittest.main()
