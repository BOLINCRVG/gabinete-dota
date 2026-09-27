# -*- coding: utf-8 -*-
"""O teste que pergunta se sobrou sinal -- e as armadilhas que ele tem de evitar.

CONTEXTO (21/09/2026). O diagnostico anterior mostrou que o erro do modelo de
vencedor e de ORDEM, nao de escala: recalibrar ganha 0,016 e sobram 0,080. Ele
mandou atacar. `tools/sobrou_sinal.py` e o passo que vem ANTES de construir
qualquer coisa: provar que existe sinal sobrando, e onde.

Dois erros ja aconteceram de verdade nesta ferramenta, e sao eles que estes
testes travam:

  1. O OTIMIZADOR INVENTOU UM SINAL. A primeira versao ajustava por gradiente
     com passo fixo. O sinal "descanso" (diferenca de dias de folga) saiu com
     ganho de -0,023 -- grande demais para acaso. Nao era o sinal: a coluna
     vale +-100 dias e o ajuste nem chegava perto do minimo. Com IRLS o mesmo
     sinal deu -0,0001. Numero de ajuste nao convergido mede o otimizador, nao
     o mundo.

  2. A PERGUNTA FOI FEITA AO MODELO ERRADO. A primeira rodada usou
     `nota_honesta_prod_diaria.json`, gerado com com_hybrid=False e
     com_form=False -- sem as duas camadas de forma que a producao aplica.
     Ele "descobriu" que forma ajuda um modelo sem forma. Verdadeiro e inutil.

E a armadilha classica, que nunca chegou a acontecer porque o desenho a
impede: olhar o futuro. Um sinal montado com jogos POSTERIORES preveria
qualquer coisa.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

RAIZ = Path(__file__).resolve().parents[2]
ARQ = RAIZ / "tools" / "sobrou_sinal.py"


def _carregar():
    spec = importlib.util.spec_from_file_location("sobrou_sinal", ARQ)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOAjusteConverge(unittest.TestCase):
    """Se o ajuste nao converge, todo o resto e ruido com cara de resultado."""

    def setUp(self):
        self.m = _carregar()

    @so_no_lol("so vale para o LoL")
    def test_acha_o_coeficiente_certo_de_um_caso_conhecido(self):
        import math
        import random
        rng = random.Random(7)
        xs, ys = [], []
        for _ in range(4000):
            x = rng.gauss(0, 1)
            p = 1.0 / (1.0 + math.exp(-(0.5 + 1.5 * x)))
            xs.append([x])
            ys.append(1 if rng.random() < p else 0)
        w, b = self.m._ajustar(xs, ys)
        self.assertAlmostEqual(w[0], 1.5, delta=0.15)
        self.assertAlmostEqual(b, 0.5, delta=0.15)

    @so_no_lol("so vale para o LoL")
    def test_a_ESCALA_da_coluna_nao_muda_o_resultado(self):
        """O bug de verdade: com passo fixo, uma coluna em "dias" (+-100)
        ficava longe do minimo e o ganho saia negativo e grande."""
        import math
        import random
        rng = random.Random(11)
        xs_p, xs_g, ys = [], [], []
        for _ in range(3000):
            x = rng.gauss(0, 1)
            p = 1.0 / (1.0 + math.exp(-(0.8 * x)))
            xs_p.append([x])
            xs_g.append([x * 100.0])          # mesma informacao, outra escala
            ys.append(1 if rng.random() < p else 0)
        w_p, _ = self.m._ajustar(xs_p, ys)
        w_g, _ = self.m._ajustar(xs_g, ys)
        self.assertAlmostEqual(w_p[0], w_g[0] * 100.0, delta=0.05)

    @so_no_lol("so vale para o LoL")
    def test_coluna_sem_informacao_devolve_peso_perto_de_zero(self):
        import random
        rng = random.Random(3)
        xs = [[rng.gauss(0, 50)] for _ in range(3000)]
        ys = [1 if rng.random() < 0.5 else 0 for _ in range(3000)]
        w, _ = self.m._ajustar(xs, ys)
        self.assertLess(abs(w[0]), 0.02)


class TestNaoOlhaOFuturo(unittest.TestCase):
    """O historico de cada jogo so pode ter o que veio ANTES dele.

    ESTE TESTE NASCEU DE UM VAZAMENTO QUE PASSOU (21/09/2026).

    A versao anterior aqui LIA O CODIGO: conferia que `dados.append` vinha
    antes de `historico.setdefault`. Passou -- e o vazamento estava la do
    mesmo jeito.

    Por que a leitura nao bastou: cada jogo ocupa DUAS linhas no CSV, uma por
    time, com a MESMA data. O laco percorria linha a linha e so punha no
    historico a linha ja usada. Correto para o time da linha; errado para o
    ADVERSARIO, cuja linha do mesmo jogo podia ter sido processada antes --
    a ordenacao do pandas nao desempata datas iguais. MEDIDO: a linha
    vermelha vinha primeiro em 46,6% dos jogos.

    Nesses, a "forma recente do adversario" ja continha o resultado DESTE
    jogo. Numa janela de 10 mapas, isso e sinal quase perfeito, e foi o que
    apareceu como descoberta:

        antes (com vazamento) .... forma +0,018   h2h +0,009
        depois (corrigido) ....... forma +0,004   h2h +0,001

    Quatro vezes menor, e o h2h sumiu. No mapa 1 o ganho foi de +0,013 para
    +0,0009, com o intervalo cruzando zero -- ou seja, nada.

    Agora o teste ALIMENTA dados onde o vazamento seria gritante e olha o
    numero que sai. Teste que le codigo so pega o erro que voce ja imaginou.
    """

    def setUp(self):
        self.fonte = ARQ.read_text(encoding="utf-8")

    def _caso_montado(self, vermelho_primeiro=True):
        """Um caso onde a resposta certa e UM numero so.

        A venceu 3 jogos. B perdeu 3. Entao A (azul) enfrenta B, e B VENCE.

            sem vazamento:  forma(A)=1.0, forma(B)=0.0  ->  forma = +1.000
            com vazamento:  B ja carrega a vitoria DESTE jogo,
                            forma(B)=1/4=0.25           ->  forma = +0.750

        A primeira tentativa deste teste era estatistica -- 400 jogos de
        cara-ou-coroa, reprovando se a forma acertasse demais. Nao pegou o
        bug: com janela de 10 mapas, um jogo contaminado move pouco, e o
        ruido cobriu (51,9% com o bug contra 44,5% sem). Teste frouxo da
        sensacao de seguranca sem a seguranca. Este aqui da 0,750 ou 1,000.
        """
        import pandas as pd
        L, t0 = [], pd.Timestamp("2026-01-01")

        def jogo(gid, azul, verm, azul_venceu, h):
            q = t0 + pd.Timedelta(hours=h)
            a = {"gameid": gid, "date": q, "map_number": 1, "league": "L",
                 "side": "Blue", "team": azul, "opponent": verm,
                 "win": bool(azul_venceu)}
            v = {"gameid": gid, "date": q, "map_number": 1, "league": "L",
                 "side": "Red", "team": verm, "opponent": azul,
                 "win": bool(not azul_venceu)}
            return [v, a] if vermelho_primeiro else [a, v]

        for i in range(3):
            L += jogo("A%d" % i, "A", "D", True, i)
            L += jogo("B%d" % i, "C", "B", True, i)
        L += jogo("ALVO", "A", "B", False, 10)
        return pd.DataFrame(L)

    @so_no_lol("so vale para o LoL")
    def test_a_forma_do_ADVERSARIO_nao_contem_o_jogo_atual(self):
        """O teste que teria pego o vazamento no dia em que ele entrou."""
        m = _carregar()
        linhas = m.montar_linhas(self._caso_montado(vermelho_primeiro=True),
                                 {"ALVO": (0.5, 0)})
        self.assertEqual(len(linhas), 1)
        self.assertAlmostEqual(
            linhas[0]["forma"], 1.0, places=9,
            msg="deu %.3f: o resultado do proprio jogo entrou na forma do "
                "adversario (0.750 e a assinatura exata desse vazamento)"
                % linhas[0]["forma"])

    @so_no_lol("so vale para o LoL")
    def test_a_ordem_das_DUAS_LINHAS_do_jogo_nao_muda_o_resultado(self):
        """Se mudar, o laco depende de um desempate que ninguem garante --
        e o pandas nao garante: medido, a linha vermelha vinha primeiro em
        46,6% dos jogos de 2026."""
        m = _carregar()
        a = m.montar_linhas(self._caso_montado(True), {"ALVO": (0.5, 0)})
        b = m.montar_linhas(self._caso_montado(False), {"ALVO": (0.5, 0)})
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual(x["forma"], y["forma"])
            self.assertEqual(x["h2h"], y["h2h"])

    @so_no_lol("so vale para o LoL")
    def test_percorre_JOGO_a_jogo_e_nao_linha_a_linha(self):
        """A causa-raiz, travada tambem na forma."""
        self.assertIn('side"].astype(str).str.lower() == "blue"', self.fonte)
        self.assertIn('kind="mergesort"', self.fonte)

    @so_no_lol("so vale para o LoL")
    def test_os_DOIS_times_aprendem_depois_de_a_linha_ficar_pronta(self):
        i_monta = self.fonte.index('dados.append({')
        self.assertLess(i_monta, self.fonte.index('historico.setdefault(blue, []).append'))
        self.assertLess(i_monta, self.fonte.index('historico.setdefault(red, []).append'))

    @so_no_lol("so vale para o LoL")
    def test_o_confronto_direto_tambem_so_aprende_depois(self):
        i_monta = self.fonte.index('dados.append({')
        i_aprende = self.fonte.index('confrontos.setdefault(ch, []).append')
        self.assertLess(i_monta, i_aprende)

    @so_no_lol("so vale para o LoL")
    def test_as_datas_do_CSV_tem_HORA(self):
        """Sem hora, dois jogos do mesmo dia entrariam em ordem arbitraria e
        a "forma" poderia incluir um jogo que ainda nao tinha acontecido."""
        import pandas as pd

        from mlcore.stats_teams import build_team_games, load_team_rows
        csv = RAIZ / "CSV_DIARIO" / "2026_LoL_esports_match_data_from_OraclesElixir.csv"
        if not csv.exists():
            self.skipTest("CSV do ano nao esta aqui")
        tg = build_team_games(load_team_rows(str(csv)))
        d = pd.to_datetime(tg["date"], errors="coerce").dropna()
        com_hora = (d.dt.time.astype(str) != "00:00:00").mean()
        self.assertGreater(com_hora, 0.99, "datas sem hora: a ordem do dia vira chute")


class TestPerguntaAoModeloCERTO(unittest.TestCase):
    """Medir contra um modelo que nao tem forma responde outra pergunta."""

    @so_no_lol("so vale para o LoL")
    def test_o_padrao_e_a_medicao_COM_as_camadas_de_producao(self):
        m = _carregar()
        self.assertIn("hybrid_form_platt", m.PADRAO,
                      "o padrao voltou a ser a medicao sem as camadas de forma")

    @so_no_lol("so vale para o LoL")
    def test_o_arquivo_padrao_realmente_tem_form_e_hybrid_ligados(self):
        import json
        alvo = RAIZ / "scratchpad" / "nota_honesta" / _carregar().PADRAO
        if not alvo.exists():
            self.skipTest("a medicao honesta nao esta nesta maquina")
        d = json.loads(alvo.read_text(encoding="utf-8"))
        self.assertTrue(d.get("com_hybrid"))
        self.assertTrue(d.get("com_form"))


class TestSeparaOQueServeDoQueNaoServe(unittest.TestCase):
    """MAPA 1 x MAPA 2+.

    Num BO3 o mapa 2 sabe quem ganhou o mapa 1 -- informacao legitima, mas que
    so existe com a serie em curso. A aba de Oportunidades precifica partida
    que ainda NAO comecou. Medido em 21/09/2026: o ganho de forma e +0,026 nos
    mapas 2+ e +0,013 no mapa 1. Sem esta separacao, o numero grande seria
    vendido como se valesse para o pre-jogo.
    """

    @so_no_lol("so vale para o LoL")
    def test_a_ferramenta_separa_os_dois_recortes(self):
        fonte = ARQ.read_text(encoding="utf-8")
        self.assertIn('x["mapa"] <= 1', fonte)
        self.assertIn('x["mapa"] >= 2', fonte)

    @so_no_lol("so vale para o LoL")
    def test_a_linha_guarda_o_numero_do_mapa(self):
        fonte = ARQ.read_text(encoding="utf-8")
        self.assertIn('"mapa":', fonte)


if __name__ == "__main__":
    unittest.main()
