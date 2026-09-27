# -*- coding: utf-8 -*-
"""TODA corrida do robô de odds deixa rastro — inclusive as que param.

O PROBLEMA (16/09/2026). Ele perguntou se tudo tinha funcionado durante a
noite. As odds deviam ser lidas as 21:25, 00:25, 03:25 e 06:25; o historico
tinha UMA leitura. E eu NAO CONSEGUI DIZER POR QUE.

A culpa era de um desenho meu. Quando a casa responde 403/429 o coletor para
em silencio e sai com rc=75 -- de proposito: nao insistir num bloqueio, nao
encher ele de e-mail por problema passageiro. So que o efeito colateral e
que UMA JANELA BLOQUEADA FICAVA IDENTICA A UMA JANELA QUE NEM RODOU.

E e justamente essa distincao que importa. Se a casa comecou a barrar, e
grave e muda o que fazer (parar de varrer derivados, espacar mais). Se o
GitHub so atrasou a janela, nao e nada. Sem rastro, as duas viram a mesma
ausencia -- e eu fiquei adivinhando na frente dele.

PARENTE do silencio que segurou kills por horas: o que nao deixa marca nao
existe para quem le depois.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.coletar_odds import DIARIO, anotar_corrida


def _linhas(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class TestOQueODiarioGuarda(unittest.TestCase):

    def test_grava_uma_linha_por_corrida(self):
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "diario.jsonl"
            anotar_corrida(alvo, resultado="ok", rc=0)
            anotar_corrida(alvo, resultado="bloqueado", rc=75)
            self.assertEqual(len(_linhas(alvo)), 2)

    def test_toda_linha_tem_QUANDO(self):
        """Corrida sem hora nao serve para saber qual janela falhou."""
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "diario.jsonl"
            anotar_corrida(alvo, resultado="ok", rc=0)
            self.assertTrue(str(_linhas(alvo)[0].get("quando") or "").strip())

    def test_o_bloqueio_fica_DISTINGUIVEL_da_rede_fora(self):
        """Eram as duas coisas que eu nao conseguia separar."""
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "diario.jsonl"
            anotar_corrida(alvo, resultado="bloqueado", rc=75, detalhe="HTTP 429")
            anotar_corrida(alvo, resultado="rede", rc=75, detalhe="ConnectionError")
            rs = [x["resultado"] for x in _linhas(alvo)]
            self.assertEqual(rs, ["bloqueado", "rede"])

    def test_a_corrida_boa_guarda_o_TAMANHO_da_colheita(self):
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "diario.jsonl"
            anotar_corrida(alvo, resultado="ok", rc=0, jogos=9, linhas=298,
                           precos_novos=151, kills=78)
            r = _linhas(alvo)[0]
            self.assertEqual((r["jogos"], r["linhas"], r["kills"]), (9, 298, 78))


class TestODiarioNUNCADerrubaAColeta(unittest.TestCase):
    """Um diario que quebra o robô e pior que nao ter diario."""

    def test_caminho_impossivel_nao_estoura(self):
        anotar_corrida(Path("/nao/existe/mesmo/diario.jsonl"), resultado="ok", rc=0)

    def test_campo_que_nao_vira_json_nao_estoura(self):
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "diario.jsonl"
            anotar_corrida(alvo, resultado="ok", rc=0, bicho=object())


class TestOColetorANOTAEmTodaSaida(unittest.TestCase):
    """Uma saida sem anotacao devolve a cegueira por aquele caminho."""

    FONTE = ""

    @classmethod
    def setUpClass(cls):
        import inspect

        from tools import coletar_odds as m
        cls.FONTE = inspect.getsource(m.main)

    def test_bloqueio_na_lista_de_confrontos(self):
        self.assertIn('resultado="bloqueado"', self.FONTE)

    def test_rede_fora(self):
        self.assertIn('resultado="rede"', self.FONTE)

    def test_formato_da_api_mudou(self):
        self.assertIn('resultado="formato_mudou"', self.FONTE)

    def test_http_inesperado(self):
        self.assertIn('resultado="http_inesperado"', self.FONTE)

    def test_corrida_boa(self):
        self.assertIn('resultado="ok"', self.FONTE)

    def test_CADA_return_tem_uma_anotacao_antes(self):
        """Conta bruta: nao pode haver saida sem rastro.

        Os `return` de main() sao: bloqueio na lista, bloqueio nos mercados,
        rede, http inesperado, formato, e o ok. Seis saidas, seis anotacoes.
        """
        self.assertEqual(self.FONTE.count("anotar_corrida("), 6)
        self.assertEqual(self.FONTE.count("return RC_"), 6)


class TestOnDeODiarioMora(unittest.TestCase):

    def test_fica_na_raiz_do_repositorio(self):
        """Tem de ser versionado: diario que fica so no servidor some com ele."""
        self.assertEqual(DIARIO.name, "odds_coletas.jsonl")

    def test_o_robo_commita_o_diario_MESMO_quando_para(self):
        """Era a metade que faltava: anotar no servidor e nao subir seria a
        mesma cegueira com mais codigo."""
        from pathlib import Path as _P
        raiz = _P(__file__).resolve().parents[2]
        wf = (raiz / ".github" / "workflows" / "odds.yml").read_text(encoding="utf-8")
        self.assertIn("odds_coletas.jsonl", wf)
        self.assertIn("always()", wf)

    def test_dado_RUIM_continua_sem_subir(self):
        """So o diario sobe quando a corrida para. O retrato de odds e o
        historico so entram com rc=0 -- leitura pela metade nao vira preco."""
        from pathlib import Path as _P
        raiz = _P(__file__).resolve().parents[2]
        wf = (raiz / ".github" / "workflows" / "odds.yml").read_text(encoding="utf-8")
        corte = wf.split("git add odds_coletas.jsonl", 1)[1]
        self.assertIn('= "0" ]', corte)
        self.assertIn("git add odds_pinnacle.json", corte)


if __name__ == "__main__":
    unittest.main()


class TestODiarioAPARECENaTela(unittest.TestCase):
    """Diario que exige abrir o GitHub para ler devolve o problema pela metade.

    Ele perguntou "funcionou durante a noite?" -- a resposta tem de estar na
    tela que ele ja abre, nao num arquivo do repositorio.
    """

    def test_a_aba_le_o_diario(self):
        import inspect

        from app_core import oportunidades_tab_core as t
        self.assertIn("carregar_diario", inspect.getsource(t))

    def test_a_aba_AVISA_quando_houve_bloqueio(self):
        """Bloqueio e grave e pede acao; janela atrasada nao e nada. As duas
        nao podem virar a mesma ausencia de novo."""
        import inspect

        from app_core import oportunidades_tab_core as t
        src = inspect.getsource(t)
        self.assertIn("houve_bloqueio", src)
        self.assertIn("A casa barrou", src)


class TestALeituraDoDiario(unittest.TestCase):

    def _com(self, linhas):
        import json as _j
        import tempfile as _t
        d = _t.mkdtemp()
        (Path(d) / "odds_coletas.jsonl").write_text(
            "\n".join(_j.dumps(x, ensure_ascii=False) for x in linhas), encoding="utf-8")
        return Path(d)

    def test_a_mais_NOVA_vem_primeiro(self):
        from app_core.odds_coletadas_core import carregar_diario
        raiz = self._com([{"quando": "2026-09-16T01:00:00-03:00", "resultado": "ok"},
                          {"quando": "2026-09-16T04:00:00-03:00", "resultado": "bloqueado"}])
        c = carregar_diario(raiz)
        self.assertEqual(c[0]["resultado"], "bloqueado")

    def test_linha_torta_nao_apaga_o_diario_inteiro(self):
        import tempfile as _t
        d = _t.mkdtemp()
        (Path(d) / "odds_coletas.jsonl").write_text(
            '{"quando":"a","resultado":"ok"}\nisto nao e json\n'
            '{"quando":"b","resultado":"rede"}\n', encoding="utf-8")
        from app_core.odds_coletadas_core import carregar_diario
        self.assertEqual(len(carregar_diario(Path(d))), 2)

    def test_sem_arquivo_devolve_lista_vazia_e_nao_estoura(self):
        import tempfile as _t
        from app_core.odds_coletadas_core import carregar_diario
        self.assertEqual(carregar_diario(Path(_t.mkdtemp())), [])

    def test_o_resumo_e_em_PORTUGUES(self):
        from app_core.odds_coletadas_core import resumo_da_corrida
        self.assertIn("a casa barrou",
                      resumo_da_corrida({"quando": "2026-09-16T04:00:00-03:00",
                                         "resultado": "bloqueado"}))
        self.assertIn("leu", resumo_da_corrida(
            {"quando": "2026-09-16T04:00:00-03:00", "resultado": "ok",
             "jogos": 9, "precos_novos": 12}))

    def test_o_resumo_da_corrida_boa_diz_o_TAMANHO(self):
        from app_core.odds_coletadas_core import resumo_da_corrida
        t = resumo_da_corrida({"quando": "2026-09-16T04:00:00-03:00", "resultado": "ok",
                               "jogos": 9, "precos_novos": 12})
        self.assertIn("9", t)
        self.assertIn("12", t)

    def test_bloqueio_e_detectado_entre_varias_corridas(self):
        from app_core.odds_coletadas_core import houve_bloqueio
        self.assertTrue(houve_bloqueio([{"resultado": "ok"}, {"resultado": "bloqueado"}]))
        self.assertFalse(houve_bloqueio([{"resultado": "ok"}, {"resultado": "rede"}]))
        self.assertFalse(houve_bloqueio([]))
