# -*- coding: utf-8 -*-
"""A aba Oportunidades tem de dizer POR QUE esta vazia.

ELE PERGUNTOU DUAS VEZES (16/09/2026):
    "essa parte ai das Oportunidades, eu ainda nao entendi, ate agora porque
     nao esta mostrando nada, voce nao disse que ja estava conseguindo pegar
     pelo menos as odds dos ml's?"

E a resposta era simples -- a tela e que nao dava. O confronto aberto era
eSuba x Meavedron, do Hitpoint Masters, e a PINNACLE NAO OFERECE Hitpoint
Masters (ela cobre LPL, LEC, LCS, CBLOL, Rift Legends e WSCI). Nao havia o
que comparar.

Mas a tela dizia "Nenhuma aposta com vantagem", que soa como "procurei e nao
achei nada bom". Para quem vai apostar, "nao achei valor" e "nao tinha o que
olhar" sao coisas MUITO diferentes -- e viraram a mesma frase.

Mesma familia do defeito que segurou kills por horas: o silencio com cara de
resposta completa.
"""
from __future__ import annotations

import inspect
import unittest

from app_core import oportunidades_tab_core as m
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol

FONTE = inspect.getsource(m)


class _StFalso:
    """Guarda o que a tela escreveu, para os testes lerem."""

    def __init__(self):
        self.infos = []
        self.avisos = []
        self.notas = []

    def info(self, txt):
        self.infos.append(str(txt))

    def warning(self, txt):
        self.avisos.append(str(txt))

    def caption(self, txt):
        self.notas.append(str(txt))

    def tudo(self):
        return " || ".join(self.infos + self.avisos + self.notas)


def _odds():
    return {"jogos": [
        {"id": 1, "liga": "LEC", "casa": "Natus Vincere", "fora": "Movistar KOI", "bo": 5},
        {"id": 2, "liga": "LPL", "casa": "Top Esports", "fora": "Invictus", "bo": 5},
    ]}


def _eu_mesmo(n):
    return n


class TestOConfrontoAbertoNaoEstaNaCasa(unittest.TestCase):
    """O caso real dele."""

    def test_diz_o_NOME_do_confronto_que_ficou_de_fora(self):
        st = _StFalso()
        ok = m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "eSuba", "teamB": "Meavedron"}, _eu_mesmo)
        self.assertFalse(ok)
        self.assertIn("eSuba", st.tudo())
        self.assertIn("Meavedron", st.tudo())
        self.assertIn("nao esta na Pinnacle", st.tudo())

    def test_LISTA_os_jogos_que_a_casa_tem(self):
        """E a informacao que resolve: basta abrir um deles."""
        st = _StFalso()
        m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "eSuba", "teamB": "Meavedron"}, _eu_mesmo)
        self.assertIn("Natus Vincere", st.tudo())
        self.assertIn("Top Esports", st.tudo())

    def test_NAO_diz_que_procurou_e_nao_achou(self):
        """"Nao achei valor" e "nao tinha o que olhar" nao podem virar a
        mesma frase."""
        st = _StFalso()
        m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "eSuba", "teamB": "Meavedron"}, _eu_mesmo)
        self.assertNotIn("Nenhuma aposta com vantagem", st.tudo())


class TestOConfrontoAbertoESTANaCasa(unittest.TestCase):

    def test_diz_que_esta_comparando_e_qual_BO(self):
        st = _StFalso()
        ok = m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "Natus Vincere", "teamB": "Movistar KOI"}, _eu_mesmo)
        self.assertTrue(ok)
        self.assertIn("Comparando", st.tudo())
        self.assertIn("BO5", st.tudo())

    def test_ordem_INVERTIDA_tambem_e_reconhecida(self):
        """A casa lista "A x B" e a tela pode ter "B x A". E o mesmo jogo."""
        st = _StFalso()
        ok = m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "Movistar KOI", "teamB": "Natus Vincere"}, _eu_mesmo)
        self.assertTrue(ok)


class TestSemConfrontoEscolhido(unittest.TestCase):

    def test_pede_para_escolher_em_vez_de_ficar_calada(self):
        st = _StFalso()
        ok = m._diagnostico_do_confronto(st, _odds(), {}, _eu_mesmo)
        self.assertFalse(ok)
        self.assertIn("Escolha um confronto", st.tudo())

    def test_ja_mostra_o_que_a_casa_tem(self):
        """Assim ele escolhe um que da certo de primeira."""
        st = _StFalso()
        m._diagnostico_do_confronto(st, _odds(), {}, _eu_mesmo)
        self.assertIn("Natus Vincere", st.tudo())


class TestQuandoNaoHaNADADaCasa(unittest.TestCase):

    def test_diz_que_a_coleta_veio_vazia(self):
        st = _StFalso()
        ok = m._diagnostico_do_confronto(
            st, {"jogos": []}, {"teamA": "eSuba", "teamB": "Meavedron"}, _eu_mesmo)
        self.assertFalse(ok)
        self.assertIn("nao trouxe jogo nenhum", st.tudo())

    def test_sem_resolvedor_nao_estoura(self):
        st = _StFalso()
        self.assertFalse(m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "a", "teamB": "b"}, None))

    def test_resolvedor_que_estoura_nao_derruba(self):
        def _explode(n):
            raise RuntimeError("sem CSV")
        st = _StFalso()
        self.assertFalse(m._diagnostico_do_confronto(
            st, _odds(), {"teamA": "a", "teamB": "b"}, _explode))


class TestATelaCHAMAODiagnostico(unittest.TestCase):

    def test_a_aba_usa_o_diagnostico(self):
        src = inspect.getsource(m.render_oportunidades_tab)
        self.assertIn("_diagnostico_do_confronto", src)

    def test_a_frase_de_nada_encontrado_so_sai_quando_HOUVE_o_que_olhar(self):
        """Senao volta o problema: vazio por falta de jogo parecendo vazio
        por falta de valor."""
        src = inspect.getsource(m.render_oportunidades_tab)
        self.assertIn("elif _jogos_que_dao:", src)


if __name__ == "__main__":
    unittest.main()


class TestAAbaLeOMESMONumeroQueOCard(unittest.TestCase):
    """DOIS NUMEROS PARA A MESMA PERGUNTA -- relatado por ele em 16/09/2026.

    Ele mandou dois prints do MESMO confronto:
        aba Oportunidades:  eSuba 1.85   Meavedron 2.18
        card Visao Geral:   eSuba 1.71   Meavedron 2.40
    e escreveu: "esta pegando uma odd do modelo que nao existe".

    Tinha razao. O card (e a Analise de Linha, e o entre-mapas) leem
    `_p_map_used`/`_p_series_used` -- o FIM do caminho, depois das camadas de
    escalacao e do ajuste de serie. Eu estava passando `p_map_cal`/`p_series`,
    que sao o MEIO. Mesma pergunta, duas respostas.

    E a doenca que este projeto passou a sessao inteira combatendo: existe um
    modulo (mlcore/replay_pre_jogo) escrito porque ja houve TRES
    implementacoes da probabilidade pre-jogo.
    """

    @so_no_lol("usa o plays_app.py, o app do LoL")
    def test_a_aba_recebe_o_numero_do_FIM_do_caminho(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        src = (raiz / "plays_app.py").read_text(encoding="utf-8")
        # desde 25/10/2026 o contexto das abas mora em _ctx_oportunidades()
        self.assertIn("ctx=_ctx_oportunidades()", src.split("TAB_OPORTUNIDADES: lambda:", 1)[1][:200])
        corte = src.split("def _ctx_oportunidades()", 1)[1][:2500]
        self.assertIn('_p_map_used', corte)
        self.assertIn('_p_series_used', corte)

    @so_no_lol("usa o plays_app.py, o app do LoL")
    def test_a_expressao_e_a_MESMA_que_o_card_usa(self):
        """Letra por letra: se as duas divergirem de novo, os numeros
        divergem junto, e o sintoma volta a ser um print dele."""
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        app = (raiz / "plays_app.py").read_text(encoding="utf-8")
        card = (raiz / "app_core" / "visao_tab_core.py").read_text(encoding="utf-8")
        for expr in ('get("_p_map_used", p_map_cal)', 'get("_p_series_used", p_series)'):
            alvo = expr.replace('"', "'")
            self.assertTrue(expr in app or alvo in app, "a aba nao usa %s" % expr)
            self.assertTrue(expr in card or alvo in card, "o card nao usa %s" % expr)


class TestACasaTEMOJogoMasONomeNaoCasou(unittest.TestCase):
    """ELE MANDOU PRINT (16/09/2026) com a tela se contradizendo.

    Abriu FlyQuest x Team Liquid e leu:

        "O confronto aberto (FlyQuest x Team Liquid) nao esta na Pinnacle"
        · LCS — Liquid x FlyQuest (BO5)

    O MESMO jogo, tres linhas abaixo da frase que dizia que ele nao existia.
    A casa escreve "Liquid"; o CSV tem "Team Liquid". Um lado casou, o outro
    nao, e eu tratei isso como "jogo inexistente".

    "A casa nao tem esse jogo" e "a casa tem, mas o nome nao casou" pedem
    ACOES DIFERENTES: a primeira e esperar, a segunda e ensinar um apelido
    uma vez. Dar a noticia errada custa a acao certa.
    """

    def _jogo(self):
        return {"id": 1, "liga": "LCS", "casa": "Liquid", "fora": "FlyQuest", "bo": 5}

    def _resolver(self, n):
        return {"FlyQuest": "FlyQuest", "Liquid": ""}.get(str(n), "")

    def test_devolve_o_nome_CRU_que_falta_casar(self):
        from app_core.oportunidades_core import nome_que_falta_casar
        self.assertEqual(
            nome_que_falta_casar(self._jogo(), "FlyQuest", "Team Liquid", self._resolver),
            "Liquid")

    def test_a_ORDEM_invertida_na_casa_nao_atrapalha(self):
        """A tela tinha FlyQuest x Team Liquid; a casa, Liquid x FlyQuest."""
        from app_core.oportunidades_core import nome_que_falta_casar
        self.assertEqual(
            nome_que_falta_casar(self._jogo(), "Team Liquid", "FlyQuest", self._resolver),
            "Liquid")

    def test_jogo_de_OUTRO_confronto_devolve_NADA(self):
        from app_core.oportunidades_core import nome_que_falta_casar
        self.assertIsNone(nome_que_falta_casar(
            {"casa": "T1", "fora": "Gen.G"}, "FlyQuest", "Team Liquid", self._resolver))

    def test_com_os_DOIS_lados_sem_casar_NAO_chuta(self):
        """PRINCIPIO IV: com dois nomes soltos, qualquer jogo "poderia" ser o
        confronto aberto. Adivinhar analisaria o jogo errado com toda a
        confianca."""
        from app_core.oportunidades_core import nome_que_falta_casar
        self.assertIsNone(nome_que_falta_casar(
            {"casa": "Liquid", "fora": "FLY"}, "FlyQuest", "Team Liquid",
            lambda n: ""))

    def test_com_os_dois_lados_casando_nao_e_caso_de_apelido(self):
        """Ai e o confronto de verdade, tratado pelo caminho normal."""
        from app_core.oportunidades_core import nome_que_falta_casar
        self.assertIsNone(nome_que_falta_casar(
            self._jogo(), "FlyQuest", "Team Liquid",
            lambda n: {"Liquid": "Team Liquid", "FlyQuest": "FlyQuest"}.get(str(n), "")))

    def test_a_tela_DIZ_que_a_casa_tem_o_jogo(self):
        st = _StFalso()
        ok = m._diagnostico_do_confronto(
            st, {"jogos": [self._jogo()]},
            {"teamA": "FlyQuest", "teamB": "Team Liquid"}, self._resolver)
        self.assertFalse(ok)
        self.assertIn("TEM este confronto", st.tudo())
        self.assertIn("Liquid", st.tudo())

    def test_a_tela_NAO_diz_mais_que_o_jogo_nao_existe(self):
        """Era a frase que contradizia a lista logo abaixo dela."""
        st = _StFalso()
        m._diagnostico_do_confronto(
            st, {"jogos": [self._jogo()]},
            {"teamA": "FlyQuest", "teamB": "Team Liquid"}, self._resolver)
        self.assertNotIn("nao esta na Pinnacle", st.tudo())

    def test_quando_a_casa_REALMENTE_nao_tem_a_frase_antiga_volta(self):
        st = _StFalso()
        m._diagnostico_do_confronto(
            st, {"jogos": [{"casa": "T1", "fora": "Gen.G", "bo": 5, "liga": "LCK"}]},
            {"teamA": "FlyQuest", "teamB": "Team Liquid"}, self._resolver)
        self.assertIn("nao esta na Pinnacle", st.tudo())


class TestOConsertoDeUmClique(unittest.TestCase):
    """Dar a noticia sem o conserto deixa ele sem saida."""

    def test_oferece_os_palpites_e_o_botao(self):
        vistos = {}

        class _St(_StFalso):
            def selectbox(self, rotulo, options=None, key=None, **k):
                vistos["rotulo"] = rotulo
                vistos["opcoes"] = list(options or [])
                # a tela devolve o ROTULO escolhido, nao o nome cru
                return next(o for o in vistos["opcoes"] if "Team Liquid" in o)

            def button(self, rotulo, key=None, **k):
                vistos["botao"] = rotulo
                return False

        st = _St()
        m._ensinar_apelido(st, lambda cru: ["Team Liquid", "Disguised"],
                           lambda cru, esc: None, "Liquid")
        self.assertIn("Liquid", vistos["rotulo"])
        # Desde 21/09/2026 a opcao carrega o aviso de quao velho e o dado
        # daquele time, a pedido dele. O NOME continua la dentro.
        self.assertTrue(any("Team Liquid" in o for o in vistos["opcoes"]),
                        vistos["opcoes"])
        self.assertTrue(vistos["botao"])

    def test_o_que_se_SALVA_e_o_NOME_e_nao_o_rotulo_com_a_data(self):
        """O rotulo mostra "Team Liquid - dado mais recente: 20/09". Se ele
        fosse gravado como apelido, nada casaria depois."""
        salvos = []
        rotulo_visto = {}

        class _St(_StFalso):
            def selectbox(self, rotulo, options=None, key=None, **k):
                opcoes = list(options or [])
                # escolhe a opcao do Team Liquid, com data e tudo
                escolhida = next(o for o in opcoes if "Team Liquid" in o)
                rotulo_visto["escolhida"] = escolhida
                return escolhida

            def button(self, rotulo, key=None, **k):
                return True

            def success(self, *a, **k):
                pass

            def rerun(self):
                pass

        st = _St()
        m._ensinar_apelido(st, lambda cru: ["Team Liquid", "Disguised"],
                           lambda cru, esc: salvos.append((cru, esc)), "Liquid")
        self.assertNotEqual(rotulo_visto["escolhida"], "Team Liquid",
                            "o rotulo devia trazer o aviso junto")
        self.assertEqual(salvos, [("Liquid", "Team Liquid")])

    def test_sem_palpite_parecido_DIZ_isso(self):
        st = _StFalso()
        m._ensinar_apelido(st, lambda cru: [], lambda cru, esc: None, "Liquid")
        self.assertIn("Nao achei palpite", st.tudo())

    def test_sem_a_maquina_de_apelidos_aponta_a_aba_Jogos(self):
        st = _StFalso()
        m._ensinar_apelido(st, None, None, "Liquid")
        self.assertIn("aba Jogos", st.tudo())
