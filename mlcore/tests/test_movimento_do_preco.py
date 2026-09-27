# -*- coding: utf-8 -*-
"""De onde o preco veio -- e por que nao se chama "abertura".

ELE OBSERVOU (16/09/2026): "voce pode estar pegando algo que na realidade ja
mudou de odd ja tem um tempo, ele nao necessariamente abriu nessa odd ai que
voce esta pegando, ate porque voce nao pega a abertura".

Certissimo, e sem conserto possivel: o robo le de 3 em 3 horas, entao o
primeiro preco que vemos e o que estava la quando ele passou.

O QUE A PESQUISA DISSE (pedida por ele, fontes abertas em 16/09/2026):
  - o CLV so precisa do preco que ele pegou e do FECHAMENTO. "Opening odds
    are irrelevant to calculating CLV" (pinnacleoddsdropper.com);
  - o fechamento e mais afiado que a abertura, nao o contrario
    (boydsbets.com/opening-vs-closing-line);
  - e a abertura da Pinnacle ja e quase tao eficiente quanto o fechamento
    dela -- "almost no difference in market efficiency between Pinnacle's
    opening and closing prices", medido em 147.629 jogos
    (football-data.co.uk/blog/opening_price_wisdom.php).

Ou seja: nao pegar a abertura NAO quebra a regua. O buraco de verdade e o
fechamento, e esse da para atacar.

O que a tela passa a mostrar e o MOVIMENTO desde a primeira leitura nossa --
chamado de "visto desde", nunca de "abriu em". O nome errado transformaria
uma limitacao conhecida numa afirmacao falsa.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app_core.odds_coletadas_core import carregar_movimento, movimento_da_linha
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


def _hist(tmp: Path, linhas):
    (tmp / "odds_pinnacle_historico.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in linhas), encoding="utf-8")
    return tmp


class TestOMovimentoEHLido(unittest.TestCase):

    def test_guarda_o_PRIMEIRO_preco_visto_e_quando(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = _hist(Path(d), [
                {"chave": "k", "odd": 2.84, "lido_em": "2026-09-15T20:21:00-03:00"},
                {"chave": "k", "odd": 2.55, "lido_em": "2026-09-16T08:40:00-03:00"},
            ])
            m = carregar_movimento(raiz)
            self.assertEqual(m["k"]["primeiro"], 2.84)
            self.assertEqual(m["k"]["quando"], "2026-09-15T20:21:00-03:00")
            self.assertEqual(m["k"]["leituras"], 2)

    def test_calcula_a_variacao_do_primeiro_ate_agora(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = _hist(Path(d), [
                {"chave": "k", "odd": 2.84, "lido_em": "2026-09-15T20:21:00-03:00"},
                {"chave": "k", "odd": 2.55, "lido_em": "2026-09-16T08:40:00-03:00"},
            ])
            r = movimento_da_linha({"chave": "k", "odd": 2.55}, carregar_movimento(raiz))
            self.assertTrue(r["sabido"])
            self.assertAlmostEqual(r["variacao_pct"], 100 * (2.55 - 2.84) / 2.84, places=6)

    def test_linha_torta_no_historico_nao_apaga_o_resto(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "odds_pinnacle_historico.jsonl").write_text(
                '{"chave":"k","odd":2.0,"lido_em":"a"}\nnao e json\n'
                '{"chave":"k","odd":2.2,"lido_em":"b"}\n', encoding="utf-8")
            self.assertEqual(carregar_movimento(Path(d))["k"]["leituras"], 2)

    def test_sem_arquivo_devolve_vazio_e_nao_estoura(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(carregar_movimento(Path(d)), {})


class TestUmaLeituraNAOEhMovimento(unittest.TestCase):
    """PRINCIPIO IV: "0% de variacao" e uma afirmacao.

    Dizer que a linha nao andou, tendo visto ela UMA vez, inventa uma
    estabilidade que ninguem observou -- e estabilidade e justamente o que
    daria conforto para apostar.
    """

    def test_vista_uma_vez_NAO_vira_zero_por_cento(self):
        with tempfile.TemporaryDirectory() as d:
            raiz = _hist(Path(d), [{"chave": "k", "odd": 2.84, "lido_em": "x"}])
            r = movimento_da_linha({"chave": "k", "odd": 2.84}, carregar_movimento(raiz))
            self.assertFalse(r["sabido"])
            self.assertIsNone(r["variacao_pct"])

    def test_mas_DIZ_que_viu_uma_vez(self):
        """Entre "nao sei" e "nao ha dado", o primeiro ainda informa."""
        with tempfile.TemporaryDirectory() as d:
            raiz = _hist(Path(d), [{"chave": "k", "odd": 2.84, "lido_em": "x"}])
            r = movimento_da_linha({"chave": "k", "odd": 2.84}, carregar_movimento(raiz))
            self.assertEqual(r["leituras"], 1)

    def test_linha_sem_historico_nenhum_devolve_nao_sei(self):
        r = movimento_da_linha({"chave": "nova", "odd": 2.0}, {})
        self.assertFalse(r["sabido"])
        self.assertEqual(r["leituras"], 0)

    def test_linha_sem_chave_nao_estoura(self):
        self.assertFalse(movimento_da_linha({}, {"k": {"primeiro": 2.0}})["sabido"])


class TestALeituraPertoDoApito(unittest.TestCase):
    """O buraco que a pesquisa apontou -- e que tinha conserto.

    Nao pegar a ABERTURA nao quebra o CLV: a formula so usa o preco que ele
    pegou e o FECHAMENTO. Nao pegar o FECHAMENTO quebra.

    E com o agendamento antigo (de 3 em 3 horas, aos :25) a ultima leitura
    antes de um jogo podia ficar 2h35 atras dele. Medido na agenda, os jogos
    de LoL comecam na HORA CHEIA -- 09, 10, 12, 14, 15, 16 e 20 UTC --, entao
    ler aos :50 poe a ultima leitura a ~10 minutos do apito.
    """

    def _crons(self):
        import yaml
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        d = yaml.safe_load((raiz / ".github" / "workflows" / "odds.yml").read_text(
            encoding="utf-8"))
        return [c["cron"] for c in d[True]["schedule"]]

    def _minutos_das_leituras(self):
        """Todos os instantes de leitura do dia, em minutos desde 00:00 UTC.

        O campo de MINUTO tambem aceita lista ("43,53") desde 17/09/2026 --
        sao duas tentativas por hora, porque o GitHub derrubou 67% das
        janelas com uma so.
        """
        minutos = []
        for c in self._crons():
            partes = c.split()
            campo_m, campo_h = partes[0], partes[1]
            if "-" in campo_h:
                a, b = campo_h.split("-")
                horas = list(range(int(a), int(b) + 1))
            else:
                horas = [int(x) for x in campo_h.split(",")]
            for m in (int(x) for x in campo_m.split(",")):
                minutos.extend(x * 60 + m for x in horas)
        return sorted(minutos)

    @so_no_lol("os horarios e o arquivo de partidas do robo de odds do LoL")
    def test_existe_a_janela_perto_do_apito(self):
        """A REGRA, nao o minuto: existe leitura na faixa de horario de jogo.

        A primeira versao deste teste travava o literal ":50". Quando o
        minuto mudou para :43/:53 -- porque o :50 caia na hora cheia, que a
        propria documentacao do GitHub chama de "high load times" -- o teste
        reprovou um conserto certo. Teste que guarda o literal em vez da
        regra atrapalha quem vem depois.
        """
        faixa = [c for c in self._crons() if "8-20" in c]
        self.assertTrue(faixa, "sumiu a leitura da faixa de jogos")

    @so_no_lol("os horarios e o arquivo de partidas do robo de odds do LoL")
    def test_a_base_de_3_em_3_horas_continua(self):
        """Ela e quem constroi o historico de MOVIMENTO do preco."""
        self.assertTrue(any(c.startswith("25 ") for c in self._crons()))

    def test_todo_jogo_na_hora_cheia_tem_leitura_ATE_15_MIN_antes(self):
        leituras = self._minutos_das_leituras()
        for hora in (9, 10, 12, 14, 15, 16, 20):
            alvo = hora * 60
            antes = [l for l in leituras if l <= alvo]
            self.assertTrue(antes, "jogo as %02d:00 sem leitura antes" % hora)
            gap = alvo - max(antes)
            self.assertLessEqual(gap, 15,
                                 "jogo as %02d:00 UTC fica %d min sem leitura" % (hora, gap))

    def test_a_leitura_NAO_e_depois_do_apito(self):
        """Toda leitura da faixa de jogos cai ANTES da hora cheia.

        O agendador do GitHub atrasa; uma leitura marcada perto demais do
        apito chega depois dele, e ai o mercado pre-jogo ja sumiu -- o
        coletor descarta o que esta ao vivo.
        """
        for c in self._crons():
            if "8-20" not in c:
                continue
            for m in (int(x) for x in c.split()[0].split(",")):
                self.assertLessEqual(m, 55, "leitura marcada quase na hora cheia")
                self.assertGreaterEqual(60 - m, 5, "menos de 5 min de folga")

    @so_no_lol("os horarios e o arquivo de partidas do robo de odds do LoL")
    def test_ha_MAIS_DE_UMA_tentativa_por_hora(self):
        """MEDIDO em 17/09/2026: com uma tentativa so, 14 de 21 janelas nao
        aconteceram (67%). Nenhum bloqueio da casa, nenhum erro -- elas
        simplesmente nao dispararam.

        A documentacao do GitHub diz que sob carga "some queued jobs may be
        dropped", e que a carga alta inclui o comeco de cada hora. Duas
        tentativas em minutos diferentes dao a segunda chance.
        """
        for c in self._crons():
            if "8-20" in c:
                minutos = c.split()[0].split(",")
                self.assertGreaterEqual(len(minutos), 2,
                                        "uma tentativa so ja perdeu 67% das janelas")
                return
        self.fail("nao ha janela na faixa de jogos")

    def test_os_minutos_fogem_dos_MAIS_DISPUTADOS(self):
        """:00, :15, :30 e :45 sao os minutos que todo mundo usa."""
        for c in self._crons():
            if "8-20" not in c:
                continue
            for m in (int(x) for x in c.split()[0].split(",")):
                self.assertNotIn(m, (0, 15, 30, 45),
                                 "minuto %d e dos mais disputados" % m)


class TestQuemJogavaFicaGravado(unittest.TestCase):
    """O historico de precos guarda o ID da partida, nao os nomes.

    EM 21/09/2026 isso me custou caro: para comparar o modelo com o preco da
    casa, precisei dos nomes das partidas ja coletadas, e `odds_pinnacle.json`
    (o unico arquivo com nomes) e SOBRESCRITO a cada corrida. So consegui
    escavando o historico do git, commit a commit. Funcionou por sorte -- o
    robo comita aquele arquivo. Medicao que depende de arqueologia nao e
    medicao.

    Entrou `odds_pinnacle_partidas.jsonl`: uma linha por partida, so na
    primeira vez que ela e vista.
    """

    def test_o_coletor_grava_as_partidas_num_arquivo_proprio(self):
        import inspect

        from tools import coletar_odds as c
        fonte = inspect.getsource(c)
        self.assertIn("odds_pinnacle_partidas.jsonl", fonte)
        self.assertIn('"visto_em": lido_em', fonte)

    def test_nao_repete_partida_ja_registrada(self):
        import inspect

        from tools import coletar_odds as c
        fonte = inspect.getsource(c)
        self.assertIn('if str(j.get("id")) in ja:', fonte)

    def test_falhar_ao_gravar_NAO_derruba_a_coleta_de_preco(self):
        """O preco e o que nao da para recuperar depois; o nome da para."""
        import inspect

        from tools import coletar_odds as c
        fonte = inspect.getsource(c)
        i = fonte.index("partidas_novas = 0")
        trecho = fonte[i:i + 1200]
        self.assertIn("except OSError", trecho)
        self.assertIn("nao consegui gravar as partidas", trecho)

    def test_o_robo_COMITA_esse_arquivo(self):
        """Gravar e nao subir daria no mesmo: a maquina dele fica desligada."""
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        yml = (raiz / ".github" / "workflows" / "odds.yml").read_text(encoding="utf-8")
        self.assertIn("odds_pinnacle_partidas.jsonl", yml)

    @so_no_lol("os horarios e o arquivo de partidas do robo de odds do LoL")
    def test_o_arquivo_ja_tem_as_partidas_recuperadas(self):
        """O passado foi preenchido a partir do git, uma vez so."""
        import json
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        alvo = raiz / "odds_pinnacle_partidas.jsonl"
        if not alvo.exists():
            self.skipTest("o arquivo nao esta nesta maquina")
        linhas = [json.loads(x) for x in alvo.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.assertGreater(len(linhas), 20)
        for x in linhas:
            self.assertIn("casa", x)
            self.assertIn("fora", x)
            self.assertIn("id", x)
        ids = [x["id"] for x in linhas]
        self.assertEqual(len(ids), len(set(ids)), "partida repetida no arquivo")
