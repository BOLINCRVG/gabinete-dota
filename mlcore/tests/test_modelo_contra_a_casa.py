# -*- coding: utf-8 -*-
"""A regua que compara o modelo com o preco da Pinnacle.

POR QUE ELA EXISTE (21/09/2026). Depois de reprovar forma recente, confronto
direto, descanso, experiencia e o afrouxamento da trava de concordancia, ficou
claro que o erro que sobra no modelo de vencedor nao e alcancavel com sinal
simples. A pergunta virou outra: o modelo vale a aposta? A Pinnacle e a
referencia mais afiada de graca que existe.

TRES ERROS QUE ESTES TESTES IMPEDEM -- um deles ja aconteceu:

  1. AFIRMAR COM 10 PARTIDAS. A primeira rodada imprimiu "ja da para
     afirmar" sobre uma diferenca de logloss medida em DEZ jogos, um deles
     respondendo por 34% dela. A formula do intervalo supoe muitos jogos
     parecidos; nessa quantidade ela mente com cara de estatistica. Entrou um
     piso: abaixo de PARTIDAS_MINIMAS nenhuma conclusao sai.

  2. ESQUECER A MARGEM DA CASA. 1/odd dos dois lados soma mais que 1. Comparar
     sem tirar isso faria o modelo parecer sistematicamente mais timido que o
     mercado -- uma conclusao inteiramente falsa.

  3. ADIVINHAR O TIME. A casa escreve "Liquid", o CSV tem "Team Liquid". Um
     palpite errado compara o modelo de UM time com o preco de OUTRO, com
     toda a confianca do mundo. O resolvedor de producao nao adivinha, e a
     partida some da medicao ate o apelido ser ensinado -- some com o nome na
     tela, para ele saber o que ensinar.
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
ARQ = RAIZ / "tools" / "modelo_contra_a_casa.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("modelo_contra_a_casa", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _historico(tmp: Path, linhas):
    alvo = tmp / "odds_pinnacle_historico.jsonl"
    alvo.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in linhas),
                    encoding="utf-8")
    return alvo


def _preco(jogo, lado, odd, lido_em, mercado="Vencedor - mapa 1", alternativa=False):
    return {"jogo": jogo, "lado": lado, "odd": odd, "lido_em": lido_em,
            "mercado": mercado, "alternativa": alternativa}


class TestAMargemDaCasaSaiAntes(unittest.TestCase):

    def _ler(self, linhas):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            m.PRECOS = _historico(Path(d), linhas)
            return m.precos_do_vencedor()

    @so_no_lol("so vale para o LoL")
    def test_as_duas_probabilidades_somam_um(self):
        r = self._ler([_preco(1, "home", 2.00, "2026-09-20T10:00"),
                       _preco(1, "away", 1.80, "2026-09-20T10:00")])
        self.assertAlmostEqual(r[1]["casa"] + r[1]["fora"], 1.0, places=9)

    @so_no_lol("so vale para o LoL")
    def test_a_margem_e_a_que_a_casa_cobra(self):
        """1/2.00 + 1/1.80 = 1,0556 -> 5,56%."""
        r = self._ler([_preco(1, "home", 2.00, "2026-09-20T10:00"),
                       _preco(1, "away", 1.80, "2026-09-20T10:00")])
        self.assertAlmostEqual(r[1]["margem_pct"], 5.5556, places=3)

    @so_no_lol("so vale para o LoL")
    def test_odds_iguais_dao_CINQUENTA_por_cento_a_cada_lado(self):
        r = self._ler([_preco(1, "home", 1.90, "2026-09-20T10:00"),
                       _preco(1, "away", 1.90, "2026-09-20T10:00")])
        self.assertAlmostEqual(r[1]["casa"], 0.5, places=9)

    @so_no_lol("so vale para o LoL")
    def test_favorito_fica_com_a_maior_probabilidade(self):
        r = self._ler([_preco(1, "home", 1.20, "2026-09-20T10:00"),
                       _preco(1, "away", 4.50, "2026-09-20T10:00")])
        self.assertGreater(r[1]["casa"], r[1]["fora"])


class TestPegaAULTIMALeitura(unittest.TestCase):
    """O preco que interessa e o mais proximo do apito, nao o primeiro visto.

    (A pesquisa de 16/09/2026 ja tinha estabelecido isso: o fechamento e mais
    afiado que a abertura, e o CLV so usa o preco pego e o fechamento.)
    """

    def _ler(self, linhas):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            m.PRECOS = _historico(Path(d), linhas)
            return m.precos_do_vencedor()

    @so_no_lol("so vale para o LoL")
    def test_usa_a_leitura_mais_recente_de_cada_lado(self):
        r = self._ler([_preco(1, "home", 3.00, "2026-09-20T08:00"),
                       _preco(1, "home", 2.00, "2026-09-20T19:00"),
                       _preco(1, "away", 1.40, "2026-09-20T08:00"),
                       _preco(1, "away", 1.80, "2026-09-20T19:00")])
        self.assertAlmostEqual(r[1]["odd_casa"], 2.00)
        self.assertAlmostEqual(r[1]["odd_fora"], 1.80)

    @so_no_lol("so vale para o LoL")
    def test_um_lado_so_NAO_vira_probabilidade(self):
        """Sem os dois lados nao da para tirar a margem. Inventar o outro lado
        seria transformar ausencia em numero."""
        self.assertEqual(self._ler([_preco(1, "home", 2.00, "2026-09-20T10:00")]), {})

    @so_no_lol("so vale para o LoL")
    def test_linha_alternativa_fica_de_fora(self):
        self.assertEqual(
            self._ler([_preco(1, "home", 2.0, "t", alternativa=True),
                       _preco(1, "away", 1.8, "t", alternativa=True)]), {})

    @so_no_lol("so vale para o LoL")
    def test_outro_mercado_nao_se_mistura(self):
        self.assertEqual(
            self._ler([_preco(1, "home", 2.0, "t", mercado="Total - Kills - mapa 1"),
                       _preco(1, "away", 1.8, "t", mercado="Total - Kills - mapa 1")]), {})

    @so_no_lol("so vale para o LoL")
    def test_odd_impossivel_e_descartada(self):
        for ruim in (0.0, 1.0, -2.0, None, "x"):
            self.assertEqual(self._ler([_preco(1, "home", ruim, "t"),
                                        _preco(1, "away", 1.8, "t")]), {})

    @so_no_lol("so vale para o LoL")
    def test_linha_torta_no_arquivo_nao_apaga_o_resto(self):
        m = _carregar()
        with tempfile.TemporaryDirectory() as d:
            alvo = Path(d) / "odds_pinnacle_historico.jsonl"
            alvo.write_text(
                json.dumps(_preco(1, "home", 2.0, "t")) + "\nnao e json\n"
                + json.dumps(_preco(1, "away", 1.8, "t")) + "\n", encoding="utf-8")
            m.PRECOS = alvo
            self.assertIn(1, m.precos_do_vencedor())


class TestNaoAfirmaComPoucasPartidas(unittest.TestCase):
    """O erro que ja aconteceu: "ja da para afirmar" com 10 jogos."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_existe_um_piso_de_partidas(self):
        m = _carregar()
        self.assertGreaterEqual(m.PARTIDAS_MINIMAS, 100)

    @so_no_lol("so vale para o LoL")
    def test_abaixo_do_piso_a_conclusao_e_BLOQUEADA(self):
        self.assertIn("if n < PARTIDAS_MINIMAS:", self.fonte)
        i_piso = self.fonte.index("if n < PARTIDAS_MINIMAS:")
        i_afirma = self.fonte.index('print("  -> ja da para afirmar")')
        self.assertLess(i_piso, i_afirma,
                        "o piso tem de ser conferido ANTES de qualquer afirmacao")

    @so_no_lol("so vale para o LoL")
    def test_mostra_o_peso_da_partida_mais_influente(self):
        """Com poucos jogos, uma partida costuma decidir tudo. Esconder isso
        deixaria o numero parecer mais solido do que e."""
        self.assertIn("a partida mais influente sozinha responde por", self.fonte)


class TestNaoAdivinhaOTime(unittest.TestCase):

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_usa_o_resolvedor_de_PRODUCAO(self):
        self.assertIn("from app_core.schedule_import_core import resolve_schedule_team",
                      self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_quem_nao_resolve_vai_para_SEM_APELIDO_e_nao_entra_na_conta(self):
        self.assertIn("sem_apelido.append", self.fonte)
        i_falha = self.fonte.index("sem_apelido.append")
        i_continua = self.fonte.index("continue", i_falha)
        i_caso = self.fonte.index("casos.append")
        self.assertLess(i_continua, i_caso,
                        "partida sem apelido nao pode chegar na conta")

    @so_no_lol("so vale para o LoL")
    def test_a_tela_DIZ_qual_nome_falta(self):
        """Sumir em silencio deixaria ele sem saber o que ensinar."""
        self.assertIn("SEM APELIDO ENSINADO", self.fonte)
        self.assertIn('x["falta"]', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_NUNCA_escreve_no_arquivo_de_apelidos(self):
        """Uma vez eu gravei um apelido adivinhado e tive de desfazer."""
        i = self.fonte.index("user_team_aliases.json")
        trecho = self.fonte[max(0, i - 900):i + 900]
        for proibido in ("write_text", "json.dump", 'open(', "save_schedule_alias"):
            self.assertNotIn(proibido, trecho,
                             "ferramenta de medicao nao escreve apelido")


class TestOLadoCertoDaAposta(unittest.TestCase):
    """Trocar casa por fora inverteria a comparacao inteira em silencio."""

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    @so_no_lol("so vale para o LoL")
    def test_tudo_e_visto_do_lado_AZUL(self):
        self.assertIn('azul_e_casa = str(r.team) == ca', self.fonte)
        self.assertIn('pr["casa"] if azul_e_casa else pr["fora"]', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_so_o_MAPA_1_entra(self):
        """O mercado lido e 'Vencedor - mapa 1'; casar com outro mapa
        compararia precos de partidas diferentes."""
        self.assertIn('map_number"].fillna(1).astype(int) == 1', self.fonte)
        m = _carregar()
        self.assertEqual(m.MERCADO, "Vencedor - mapa 1")


if __name__ == "__main__":
    unittest.main()


class TestAReguaJuntaAsMedicoes(unittest.TestCase):
    """A regua do vencedor PARA DE CRESCER sem reforco -- e o reforco tem de
    ser barato, senao ninguem faz.

    DESCOBERTO EM 22/09/2026: ela travou em 20 partidas. Cinco jogos novos
    estavam no CSV e apareciam como "sem nota do modelo", porque a nota
    honesta so pontua os jogos que existiam quando ela rodou. Refazer as 8
    dobras leva 1h40; uma dobra so cobre os ultimos 30 dias e leva ~12
    minutos, que e tudo de que o reforco precisa.

    Juntar e valido porque, em qualquer das rodadas, o jogo foi pontuado por
    um modelo que nao o tinha visto. O risco e juntar rodadas de VARIANTES
    diferentes do modelo -- "margem", "elo", "prod" sao modelos distintos, e
    mistura-los mediria um Frankenstein. Por isso a familia e um prefixo fixo.
    """

    def _pasta(self, d, arquivos):
        m = _carregar()
        for nome, por_jogo in arquivos.items():
            (Path(d) / nome).write_text(
                json.dumps({"por_jogo": por_jogo}), encoding="utf-8")
        m.PASTA_REGUA = Path(d)
        return m

    @so_no_lol("so vale para o LoL")
    def test_junta_dois_arquivos_da_mesma_familia(self):
        with tempfile.TemporaryDirectory() as d:
            m = self._pasta(d, {
                "nota_honesta_base_agree_diaria_hybrid_form_platt.json":
                    [["G1", 0.6, 1], ["G2", 0.4, 0]],
                "nota_honesta_base_agree_reforco_diaria_hybrid_form_platt.json":
                    [["G3", 0.7, 1]]})
            modelo, resumo = m.carregar_regua()
            self.assertEqual(sorted(modelo), ["G1", "G2", "G3"])
            self.assertEqual(resumo["jogos"], 3)

    @so_no_lol("so vale para o LoL")
    def test_NAO_mistura_outra_variante_do_modelo(self):
        """"margem" e "elo" sao modelos diferentes. Junta-los mediria um
        Frankenstein que nunca existiu."""
        with tempfile.TemporaryDirectory() as d:
            m = self._pasta(d, {
                "nota_honesta_base_agree_diaria_hybrid_form_platt.json":
                    [["G1", 0.6, 1]],
                "nota_honesta_margem_diaria_hybrid_form_platt.json":
                    [["G9", 0.9, 1]]})
            modelo, _ = m.carregar_regua()
            self.assertIn("G1", modelo)
            self.assertNotIn("G9", modelo, "entrou jogo de outra variante")

    @so_no_lol("so vale para o LoL")
    def test_jogo_repetido_fica_com_UM_valor_so_e_deterministico(self):
        with tempfile.TemporaryDirectory() as d:
            m = self._pasta(d, {
                "nota_honesta_base_agree_diaria_hybrid_form_platt.json":
                    [["G1", 0.60, 1]],
                "nota_honesta_base_agree_zz_diaria_hybrid_form_platt.json":
                    [["G1", 0.64, 1]]})
            a, resumo = m.carregar_regua()
            b, _ = m.carregar_regua()
            self.assertEqual(a["G1"], b["G1"], "o resultado mudou entre chamadas")
            self.assertEqual(resumo["repetidos"], 1)

    @so_no_lol("so vale para o LoL")
    def test_DIZ_o_quanto_as_rodadas_divergem(self):
        """Se duas rodadas discordarem muito sobre o mesmo jogo, alguma delas
        deixou de ser comparavel -- e isso tem de aparecer, nao ser engolido."""
        with tempfile.TemporaryDirectory() as d:
            m = self._pasta(d, {
                "nota_honesta_base_agree_diaria_hybrid_form_platt.json":
                    [["G1", 0.60, 1]],
                "nota_honesta_base_agree_zz_diaria_hybrid_form_platt.json":
                    [["G1", 0.70, 1]]})
            _, resumo = m.carregar_regua()
            self.assertAlmostEqual(resumo["divergencia_media"], 0.10, places=6)

    @so_no_lol("so vale para o LoL")
    def test_arquivo_torto_nao_derruba_os_outros(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "nota_honesta_base_agree_zz_diaria_hybrid_form_platt.json"
             ).write_text("nao e json", encoding="utf-8")
            m = self._pasta(d, {
                "nota_honesta_base_agree_diaria_hybrid_form_platt.json":
                    [["G1", 0.6, 1]]})
            modelo, _ = m.carregar_regua()
            self.assertEqual(sorted(modelo), ["G1"])

    @so_no_lol("so vale para o LoL")
    def test_pasta_vazia_devolve_vazio_e_diz_o_comando(self):
        with tempfile.TemporaryDirectory() as d:
            m = self._pasta(d, {})
            modelo, resumo = m.carregar_regua()
            self.assertEqual(modelo, {})
            self.assertEqual(resumo["jogos"], 0)
        fonte = ARQ.read_text(encoding="utf-8")
        self.assertIn("python tools/nota_honesta.py --marca base_agree", fonte)

    @so_no_lol("so vale para o LoL")
    def test_o_comando_do_reforco_BARATO_esta_no_cabecalho(self):
        """Se o reforco nao estiver escrito, ninguem lembra de fazer."""
        fonte = ARQ.read_text(encoding="utf-8")
        self.assertIn("REFORCO SEMANAL", fonte)
        self.assertIn("--dobras 1", fonte)
