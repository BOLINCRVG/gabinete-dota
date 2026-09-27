# -*- coding: utf-8 -*-
"""O modelo de handicap de abates contra o preco da Pinnacle.

A PERGUNTA. O modelo de handicap esta bem calibrado (0,31pp de erro medio,
fora da amostra, em 10.994 jogos). Calibrado NAO e lucrativo: a casa tambem e
calibrada e cobra margem. Vantagem exige acertar MAIS que ela.

POR QUE NESTE MERCADO. No vencedor uma partida da UMA aposta (20 medidas, ~100
necessarias). No handicap, uma partida da varias linhas e varios mapas.

TRES ARMADILHAS QUE ESTES TESTES IMPEDEM:

1. CONTAR APOSTA COMO SE FOSSE PARTIDA. Sao 100 apostas em 18 partidas. No
   mesmo mapa, todas as linhas acertam ou erram JUNTAS -- quando o time
   atropela, o handicap cobre em todas elas. Tratar 100 como independentes
   daria uma certeza que nao existe. A incerteza sai agrupada por partida.

2. MEDIR NA LINHA ALTERNATIVA. Sao 3.182 das 3.700 leituras, entao entram
   sozinhas se ninguem filtrar -- e cobram 8,49% contra 6,98% da principal.
   Mediriam uma aposta que ele nao deveria fazer.

3. ESQUECER A MARGEM. 1/odd dos dois lados soma mais que 1. Sem tirar, a casa
   pareceria pior do que e e o modelo ganharia de mentira.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "tools" / "handicap_contra_a_casa.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("handicap_contra_a_casa", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _linha(jogo, mapa, lado, pontos, odd, lido_em="t", alt=False):
    return {"jogo": jogo, "mercado": "Handicap - Kills - mapa %d" % mapa,
            "lado": lado, "pontos": pontos, "odd": odd, "lido_em": lido_em,
            "alternativa": alt, "limite": 500}


class TestSoALinhaPrincipal(unittest.TestCase):

    def _ler(self, linhas):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "h.jsonl"
            alvo.write_text("\n".join(json.dumps(x) for x in linhas), encoding="utf-8")
            m.PRECOS = alvo
            return m.apostas_principais()

    @so_no_lol("so vale para o LoL")
    def test_a_alternativa_fica_de_fora(self):
        """Ela e a maioria das leituras e cobra mais caro."""
        self.assertEqual(self._ler([
            _linha(1, 1, "home", -9.5, 1.9, alt=True),
            _linha(1, 1, "away", 9.5, 1.9, alt=True)]), [])

    @so_no_lol("so vale para o LoL")
    def test_a_principal_entra(self):
        r = self._ler([_linha(1, 1, "home", -9.5, 1.9),
                       _linha(1, 1, "away", 9.5, 1.9)])
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["mapa"], 1)
        self.assertAlmostEqual(r[0]["pontos_casa"], -9.5)

    @so_no_lol("so vale para o LoL")
    def test_mapas_diferentes_sao_apostas_diferentes(self):
        r = self._ler([_linha(1, 1, "home", -9.5, 1.9), _linha(1, 1, "away", 9.5, 1.9),
                       _linha(1, 2, "home", -7.5, 1.9), _linha(1, 2, "away", 7.5, 1.9)])
        self.assertEqual(sorted(x["mapa"] for x in r), [1, 2])

    @so_no_lol("so vale para o LoL")
    def test_um_lado_so_NAO_vira_aposta(self):
        """Sem os dois lados nao da para tirar a margem."""
        self.assertEqual(self._ler([_linha(1, 1, "home", -9.5, 1.9)]), [])

    @so_no_lol("so vale para o LoL")
    def test_usa_a_leitura_mais_RECENTE_de_cada_lado(self):
        r = self._ler([_linha(1, 1, "home", -9.5, 3.0, "2026-09-20T08:00"),
                       _linha(1, 1, "home", -9.5, 1.8, "2026-09-20T19:00"),
                       _linha(1, 1, "away", 9.5, 1.4, "2026-09-20T08:00"),
                       _linha(1, 1, "away", 9.5, 2.0, "2026-09-20T19:00")])
        self.assertAlmostEqual(r[0]["odd_casa"], 1.8)
        self.assertAlmostEqual(r[0]["odd_fora"], 2.0)


class TestAMargemSaiAntes(unittest.TestCase):

    def _uma(self, odd_casa, odd_fora):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "h.jsonl"
            alvo.write_text("\n".join(json.dumps(x) for x in [
                _linha(1, 1, "home", -9.5, odd_casa),
                _linha(1, 1, "away", 9.5, odd_fora)]), encoding="utf-8")
            m.PRECOS = alvo
            return m.apostas_principais()[0]

    @so_no_lol("so vale para o LoL")
    def test_as_duas_probabilidades_somam_um(self):
        a = self._uma(1.7463, 1.8547)
        self.assertAlmostEqual(a["p_casa"] + a["p_fora"], 1.0, places=9)

    @so_no_lol("so vale para o LoL")
    def test_a_margem_e_a_que_a_casa_cobra(self):
        a = self._uma(2.00, 1.80)
        self.assertAlmostEqual(a["margem_pct"], 5.5556, places=3)

    @so_no_lol("so vale para o LoL")
    def test_favorito_fica_com_a_maior_probabilidade(self):
        self.assertGreater(self._uma(1.50, 2.60)["p_casa"],
                           self._uma(1.50, 2.60)["p_fora"])


class TestAIncertezaContaPARTIDA(unittest.TestCase):
    """A armadilha que daria a certeza mais convincente e mais falsa."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_agrupa_por_partida_antes_de_calcular_a_margem_de_erro(self):
        self.assertIn("por_partida", self.fonte)
        i_grupo = self.fonte.index("por_partida.setdefault")
        i_erro = self.fonte.index("margem de erro")
        self.assertLess(i_grupo, i_erro,
                        "a margem de erro foi calculada antes de agrupar")

    @so_no_lol("so vale para o LoL")
    def test_o_piso_conta_PARTIDAS_e_nao_apostas(self):
        m = _carregar()
        self.assertGreaterEqual(m.PARTIDAS_MINIMAS, 30)
        self.assertIn("len(medias) < PARTIDAS_MINIMAS", self.fonte)
        self.assertIn("AMOSTRA PEQUENA DEMAIS", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_mostra_em_quantas_PARTIDAS_o_modelo_bateu_a_casa(self):
        """Uma contagem de partidas diz mais que uma media de apostas."""
        self.assertIn("partidas em que o modelo bateu a casa", self.fonte)


class TestONaoAdivinhaNemVazaOFuturo(unittest.TestCase):

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_usa_o_resolvedor_de_producao(self):
        self.assertIn("from app_core.schedule_import_core import resolve_schedule_team",
                      self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_modelo_usa_so_o_que_se_sabia_ANTES_do_jogo(self):
        """Desde 25/09/2026 a regua mede a conta da TELA, com os kills
        esperados de cada jogo guardados ANTES de ele entrar no historico."""
        self.assertIn('prob_cobre_handicap(est["ea"], est["eb"], linha_azul)', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_estado_de_cada_jogo_e_gravado_ANTES_de_ele_entrar(self):
        i_estado = self.fonte.index("estado[str(r.gameid)] =")
        i_aprende = self.fonte.index("hist[t] = nova if d is None")
        self.assertLess(i_estado, i_aprende)

    @so_no_lol("so vale para o LoL")
    def test_reusa_a_conta_de_PRODUCAO_em_vez_de_copiar(self):
        """PRINCIPIO V: duas implementacoes do mesmo preco divergem. Ate
        25/09/2026 esta regua reusava a conta de 21/09 (tools/
        handicap_de_abates); agora reusa a que a tela serve."""
        self.assertIn("from mlcore.abates_por_time import prob_cobre_handicap", self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_NAO_escreve_no_arquivo_de_apelidos(self):
        for proibido in ("write_text", "json.dump"):
            self.assertNotIn(proibido, self.fonte)


class TestOLadoCertoDaAposta(unittest.TestCase):
    """Trocar casa por fora inverteria a comparacao inteira em silencio."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_tudo_e_visto_do_lado_AZUL(self):
        self.assertIn("azul_e_casa = str(r.team) == ca", self.fonte)
        self.assertIn('a["pontos_casa"] if azul_e_casa else -a["pontos_casa"]',
                      self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_resultado_usa_a_linha_do_lado_azul(self):
        self.assertIn('1.0 if dif + linha_azul > 0 else 0.0', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_mapa_da_aposta_casa_com_o_mapa_do_jogo(self):
        """Handicap do mapa 2 comparado ao mapa 1 mediria outra partida."""
        self.assertIn('indice.get((dd, par, a["mapa"]), [])', self.fonte)


if __name__ == "__main__":
    unittest.main()
