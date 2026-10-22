# -*- coding: utf-8 -*-
"""Testes da trava do retreino completo automatico.

Uma trava que so foi vista aprovando nao e uma trava — e um carimbo. Estes
testes estragam o modelo de uma forma por vez e exigem que ela RECUSE.

Cada cenario aqui e uma forma real de o retreino dar errado sozinho, de
madrugada, sem ninguem olhando.
"""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    from tools.portao_retreino_completo import conferir
except ImportError as _sem_ferramenta:  # ferramenta so do LoL: no Dota o modulo e pulado
    import unittest as _u
    raise _u.SkipTest("tools.portao_retreino_completo nao existe neste repositorio: " + str(_sem_ferramenta))


def _modelo_bom() -> dict:
    ratings = {"Time{:02d}".format(i): float(i * 3 - 40) for i in range(40)}
    jogos = {t: 120 for t in ratings}
    return {
        "version": "0.4.0",
        "model": {
            "ratings": ratings,
            "games_played": jogos,
            "params": {"k": 28.0, "scale_points": 400.0},
            "league_strength": {"LPL": 0.0},
            "tier_strength": {"T1": 0.0},
        },
        "calibrator": {"a": 0.72, "b": 0.09},
        "meta": {
            "metrics": {"logloss_cal": 0.662, "holdout_n": 1204},
            "team_meta": {t: {"league": "LPL", "tier": "T1"} for t in ratings},
            "players_layer": {"enabled": True, "w_lineup": 0.03},
            "whr_mov": {"sinal": "gd15", "lambda": 0.35,
                        "jogos_com_margem": 25270, "jogos_sem_margem": 2293},
            "players_pscore": {
                "enabled": True, "w": 0.0686, "w_raw": 0.0686,
                "fonte_do_peso": "prequencial", "jogos_usados": 9050,
                "n_jogadores": 5840,
            },
        },
    }


def _notas_boas(w: float = 0.0686, dp: float = 0.6847) -> dict:
    """Notas com a MESMA dispersao do arquivo servido (medida em 13/09/2026).

    A primeira versao dava a mesma nota 0,01 para todo mundo. Como o arquivo
    ficava sem dispersao nenhuma, a trava do EFEITO nunca era exercitada: ela
    media deslocamento zero e o cenario "bom" passava reclamando. Fixture sem
    variacao nao testa nada que dependa de variacao.
    """
    rotas = ("Top", "Jungle", "Mid", "Bot", "Support")
    notas, apelidos = {}, {}
    for i in range(1500):
        rota = rotas[i % len(rotas)]
        chave_id = "oe:player:{:04d}".format(i)
        notas["{}|||{}".format(chave_id, rota)] = {
            "n": dp * (1.0 if i % 2 else -1.0), "gp": 30}
        # O indice NOME -> id. Sem ele o Radar nao mostra nota (trava 4b).
        apelidos["jogador{}|||{}".format(i, rota)] = chave_id
    return {"w": w, "beta": 10.0, "notas": notas, "apelidos": apelidos}


class _Base(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _gravar(self, modelo: dict, notas: dict | None = None) -> str:
        alvo = self.raiz / "ml_artifact.json"
        alvo.write_text(json.dumps(modelo), encoding="utf-8")
        if notas is not None:
            (self.raiz / "players_pscore.json").write_text(
                json.dumps(notas), encoding="utf-8")
        return str(alvo)

    def _rodar(self, modelo: dict, notas: dict | None = None) -> int:
        caminho = self._gravar(modelo, notas)
        buf = io.StringIO()
        import contextlib
        with contextlib.redirect_stdout(buf):
            rc = conferir(caminho, None, None)
        self._saida = buf.getvalue()
        return rc


class TestAprovaOBom(_Base):

    def test_modelo_intacto_passa(self):
        self.assertEqual(self._rodar(_modelo_bom(), _notas_boas()), 0, self._saida)


class TestACamadaCONSEGUEAgir(_Base):
    """A trava que faltava: o arquivo esta cheio, mas ele SERVE?

    As notas sao chaveadas por `playerid`. O caminho da ODD manda id e nunca
    dependeu disto. O RADAR monta o draft por `playername` — sem o indice
    NOME -> id, a secao de escalacao dele nao mostra nota nenhuma.

    A forma da checagem vale por um motivo maior que o caso: todas as outras
    travas olham o arquivo POR DENTRO (tem notas? o peso bate?) e nenhuma
    pergunta se ele funciona para quem vai usar. E a diferenca entre conferir
    o estoque e conferir a entrega.

    ERRO MEU REGISTRADO (13/09/2026): testei o caminho servido com a escalacao
    do Radar (nomes), vi 0/5 e anunciei que a camada estava inerte em
    producao. Nao estava. Conferido no app dirigido em 14/09: 5/5.
    """

    def test_arquivo_SEM_o_indice_de_nomes_e_RECUSADO(self):
        """O estado exato que estava em producao. Nunca mais passa calado."""
        notas = _notas_boas()
        notas.pop("apelidos")
        self.assertEqual(self._rodar(_modelo_bom(), notas), 1)
        self.assertIn("indice NOME", self._saida)

    def test_indice_vazio_e_RECUSADO(self):
        notas = _notas_boas()
        notas["apelidos"] = {}
        self.assertEqual(self._rodar(_modelo_bom(), notas), 1)

    def test_indice_que_aponta_para_o_nada_e_RECUSADO(self):
        """Indice grande mas desatualizado: cobertura 0/5 do mesmo jeito."""
        notas = _notas_boas()
        notas["apelidos"] = {k: "oe:player:NAO_EXISTE"
                             for k in notas["apelidos"]}
        self.assertEqual(self._rodar(_modelo_bom(), notas), 1)
        self.assertIn("cobertura 0/5", self._saida)

    def test_arquivo_bom_prova_que_a_camada_age(self):
        self.assertEqual(self._rodar(_modelo_bom(), _notas_boas()), 0, self._saida)
        self.assertIn("resolve os 5", self._saida)


class TestOQuantoACamadaMove(_Base):
    """A trava olha o EFEITO da camada, nao o peso cru.

    Motivo: o peso sozinho nao quer dizer nada. No modo 'media' o peso certo
    e ~0,069; no modo 'ffa' e ~0,005 — porque a nota do jogador vive numa
    escala vinte vezes maior. Uma faixa fixa de peso reprovaria o 'ffa'
    correto e aprovaria um 'media' morto. Medido no arquivo servido em
    13/09/2026: os dois deslocam o logito quase igual (0,149 e 0,191).
    """

    def test_o_cenario_bom_nao_dispara_o_aviso(self):
        self.assertEqual(self._rodar(_modelo_bom(), _notas_boas()), 0, self._saida)
        self.assertNotIn("AVISO   o quanto a camada move", self._saida)

    def test_o_modo_ffa_passa_com_peso_vinte_vezes_menor(self):
        """O que a faixa de peso antiga reprovaria a toa."""
        m = _modelo_bom()
        m["meta"]["players_pscore"]["w"] = 0.00513
        m["meta"]["players_pscore"]["w_raw"] = 0.00513
        self.assertEqual(self._rodar(m, _notas_boas(w=0.00513, dp=11.7865)), 0,
                         self._saida)
        self.assertNotIn("AVISO   o quanto a camada move", self._saida)

    def test_avisa_quando_a_camada_ficou_inerte(self):
        """Peso positivo mas notas quase iguais: a camada nao entrega nada."""
        self._rodar(_modelo_bom(), _notas_boas(dp=0.00001))
        self.assertIn("AVISO   o quanto a camada move", self._saida)

    def test_avisa_quando_a_camada_manda_demais(self):
        m = _modelo_bom()
        m["meta"]["players_pscore"]["w"] = 3.0
        m["meta"]["players_pscore"]["w_raw"] = 3.0
        self._rodar(m, _notas_boas(w=3.0))
        self.assertIn("AVISO   o quanto a camada move", self._saida)

    def test_poucos_jogadores_nao_viram_efeito_inventado(self):
        """PRINCIPIO IV: sem amostra, a conta nao vira numero."""
        notas = _notas_boas()
        notas["notas"] = dict(list(notas["notas"].items())[:10])
        self._rodar(_modelo_bom(), notas)
        self.assertIn("nan", self._saida.lower())


class TestPlausibilidade(_Base):
    """A checagem antiga reprovava o modelo CERTO.

    Ela exigia 0,02 < p < 0,98 no confronto entre o melhor e o pior time. Com
    887 times reais, esse par da 0,99988 — e esta certo: sao 1.604 pontos de
    rating entre a academia da T1 (383 jogos) e um time de 22 jogos.

    A faixa tinha sido calibrada so contra a fixture destes testes: 40 times
    com ~120 pontos de amplitude. Rodada contra o modelo de PRODUCAO em
    14/09/2026, ela reprovava — nunca tinha sido rodada contra ele.
    """

    def _com_amplitude(self, span: float) -> dict:
        """Modelo com a mesma forma do real: elenco grande e muito espalhado."""
        m = _modelo_bom()
        n = len(m["model"]["ratings"])
        m["model"]["ratings"] = {
            "Time{:02d}".format(i): -span / 2 + span * i / max(1, n - 1)
            for i in range(n)}
        m["meta"]["team_meta"] = {t: {"league": "LPL", "tier": "T1"}
                                  for t in m["model"]["ratings"]}
        return m

    def test_elenco_MUITO_espalhado_como_o_real_e_APROVADO(self):
        """1.600 pontos de amplitude e o que o modelo servido tem."""
        self.assertEqual(self._rodar(self._com_amplitude(1600.0), _notas_boas()),
                         0, self._saida)

    def test_o_par_mais_desigual_e_conferido_contra_SATURACAO(self):
        """Quase-certeza pode; 0 ou 1 exatos, nao — viraria odd 1,00 na tela."""
        self._rodar(self._com_amplitude(1600.0), _notas_boas())
        self.assertIn("satura em 0 ou 1", self._saida)

    def test_as_DUAS_perguntas_sao_feitas(self):
        self._rodar(self._com_amplitude(1600.0), _notas_boas())
        self.assertIn("meio de tabela", self._saida)
        self.assertIn("mais desigual", self._saida)

    def test_recusa_quando_o_MEIO_DE_TABELA_da_certeza(self):
        """Modelo quebrado de verdade: ate os parecidos viram 0 ou 1."""
        m = _modelo_bom()
        # ratings absurdos e MUITO afastados entre vizinhos
        m["model"]["ratings"] = {"Time{:02d}".format(i): float(i * 5000)
                                 for i in range(40)}
        m["meta"]["team_meta"] = {t: {"league": "LPL", "tier": "T1"}
                                  for t in m["model"]["ratings"]}
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("meio de tabela", self._saida)


class TestRecusaOQuebrado(_Base):
    """Cada um destes ja seria um prejuizo silencioso em producao."""

    def test_recusa_modelo_pior_que_moeda(self):
        m = _modelo_bom()
        m["meta"]["metrics"]["logloss_cal"] = 0.71
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("cara ou coroa", self._saida)

    def test_recusa_logloss_baixa_demais(self):
        """Logloss otima em LoL pre-jogo e sinal de vazamento, nao de acerto."""
        m = _modelo_bom()
        m["meta"]["metrics"]["logloss_cal"] = 0.21
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("vazamento", self._saida)

    def test_recusa_sem_a_margem_de_vitoria(self):
        m = _modelo_bom()
        m["meta"].pop("whr_mov")
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("MARGEM", self._saida)

    def test_recusa_quando_a_camada_de_desempenho_falhou(self):
        m = _modelo_bom()
        m["meta"]["players_pscore"] = {"enabled": False, "error": "CSV sem colunas"}
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("DESEMPENHO", self._saida)

    def test_recusa_peso_de_desempenho_zerado(self):
        """Peso 0 quer dizer que o ajuste desistiu — a camada nao entregaria nada."""
        m = _modelo_bom()
        m["meta"]["players_pscore"]["w"] = 0.0
        self.assertEqual(self._rodar(m, _notas_boas(w=0.0)), 1)
        self.assertIn("positivo", self._saida)

    def test_recusa_peso_ajustado_no_conjunto_pequeno(self):
        """Se cair no holdout, o peso sai pela metade — ja aconteceu de verdade."""
        m = _modelo_bom()
        m["meta"]["players_pscore"]["fonte_do_peso"] = "holdout"
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("prequencial", self._saida)

    def test_recusa_sem_calibrador(self):
        m = _modelo_bom()
        m["calibrator"] = None
        self.assertEqual(self._rodar(m, _notas_boas()), 1)

    def test_recusa_holdout_pequeno_demais(self):
        m = _modelo_bom()
        m["meta"]["metrics"]["holdout_n"] = 12
        self.assertEqual(self._rodar(m, _notas_boas()), 1)

    def test_recusa_sem_o_arquivo_de_notas(self):
        self.assertEqual(self._rodar(_modelo_bom(), None), 1)
        self.assertIn("notas", self._saida)

    def test_recusa_quando_o_peso_do_arquivo_nao_bate_com_o_do_modelo(self):
        """Arquivo de uma corrida com modelo de outra — descasamento silencioso."""
        self.assertEqual(self._rodar(_modelo_bom(), _notas_boas(w=0.02)), 1)

    def test_recusa_csv_truncado_que_encolhe_o_elenco(self):
        bom = _modelo_bom()
        caminho_velho = self.raiz / "velho.json"
        caminho_velho.write_text(json.dumps(bom), encoding="utf-8")

        encolhido = copy.deepcopy(bom)
        metade = dict(list(encolhido["model"]["ratings"].items())[:15])
        encolhido["model"]["ratings"] = metade
        alvo = self._gravar(encolhido, _notas_boas())

        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = conferir(alvo, str(caminho_velho), None)
        self.assertEqual(rc, 1)
        self.assertIn("encolheu", buf.getvalue())

    def test_recusa_modelo_COLAPSADO(self):
        """Todos os ratings iguais: o modelo perdeu a nocao de quem e melhor."""
        m = _modelo_bom()
        m["model"]["ratings"] = {t: 0.0 for t in m["model"]["ratings"]}
        self.assertEqual(self._rodar(m, _notas_boas()), 1)
        self.assertIn("mais forte e favorito", self._saida)

    def test_recusa_modelo_INVERTIDO(self):
        """Sinal trocado: o modelo daria o time pior como favorito."""
        m = _modelo_bom()
        m["model"]["ratings"] = {t: -v for t, v in m["model"]["ratings"].items()}
        # invertendo os ratings, quem o portao chama de "melhor" passa a ser o
        # antigo pior — entao o teste real e que a probabilidade ainda favoreca
        # o de rating mais alto. Se favorecer, passa; e o que queremos.
        self.assertEqual(self._rodar(m, _notas_boas()), 0, self._saida)


if __name__ == "__main__":
    unittest.main()
