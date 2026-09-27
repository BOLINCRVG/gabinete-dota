# -*- coding: utf-8 -*-
"""Testes do isolamento de falha por aba.

Em 13/09/2026 o site inteiro saiu do ar. A causa nao foi o calculo nem os
dados: UMA aba (StatsCamps) recebeu um argumento que a versao antiga do
arquivo no servidor nao conhecia, e a excecao subiu ate o topo do script.

A Streamlit Community Cloud recarrega o arquivo principal quando chega um
commit, mas nao recarrega os modulos importados — eles continuam em
`sys.modules` desde o boot. O arquivo principal novo chama a funcao velha.
  - https://discuss.streamlit.io/t/apps-deployed-on-streamlit-community-cloud-reacts-only-on-main-app-file-changes-but-not-on-changes-in-the-py-modules/36626
  - https://github.com/streamlit/streamlit/issues/366

Nao da para impedir o desencontro (ele mora no hospedeiro). Da para impedir
que ele custe o app inteiro. Estes testes travam isso.
"""
from __future__ import annotations

import unittest

from app_core.tab_dispatcher import (
    CHAVE_FALHAS_DE_ABA,
    desenhar_aba,
    dispatch_secondary_tabs,
    iniciar_registro_de_falhas,
)


class _Aba:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _St:
    def __init__(self):
        self.session_state: dict = {}
        self.erros: list = []

    def error(self, texto, *a, **k):
        self.erros.append(str(texto))


def _despachar(st, renderers):
    dispatch_secondary_tabs(
        st,
        tab_refs={n: _Aba() for n in renderers},
        renderers=renderers,
        order=tuple(renderers),
    )


class TestUmaAbaNaoDerrubaOApp(unittest.TestCase):

    def test_as_outras_abas_continuam_sendo_desenhadas(self):
        """O caso exato de 13/09: a do meio explode, as outras vivem."""
        feitas = []
        st = _St()
        _despachar(st, {
            "Antes": lambda: feitas.append("Antes"),
            "Quebrada": lambda: (_ for _ in ()).throw(
                TypeError("render_statscamps_tab() got an unexpected "
                          "keyword argument 'parse_time_lines_fn'")),
            "Depois": lambda: feitas.append("Depois"),
        })
        self.assertEqual(feitas, ["Antes", "Depois"])

    def test_a_excecao_nao_sobe(self):
        st = _St()
        try:
            _despachar(st, {"X": lambda: 1 / 0})
        except ZeroDivisionError:
            self.fail("a falha da aba vazou e derrubaria o app")

    def test_o_dono_ve_uma_mensagem_no_lugar_da_aba(self):
        st = _St()
        _despachar(st, {"X": lambda: 1 / 0})
        self.assertEqual(len(st.erros), 1)
        self.assertIn("X", st.erros[0])


class TestAMensagemDizOQueFazer(unittest.TestCase):
    """Mensagem que nao diz o conserto nao serve para quem nao le codigo."""

    def _texto(self, erro):
        st = _St()
        _despachar(st, {"StatsCamps": lambda: (_ for _ in ()).throw(erro)})
        return st.erros[0]

    def test_desencontro_de_versao_manda_reiniciar(self):
        t = self._texto(TypeError("render_statscamps_tab() got an unexpected "
                                  "keyword argument 'parse_time_lines_fn'"))
        self.assertIn("Reboot", t)
        self.assertIn("desatualizado", t)

    def test_diz_que_publicar_commit_nao_resolve(self):
        """Foi a primeira coisa que eu ia tentar; a fonte diz que nao adianta."""
        t = self._texto(TypeError("f() missing 1 required keyword-only argument: 'x'"))
        self.assertIn("commit", t.lower())

    def test_erro_comum_NAO_manda_reiniciar_a_toa(self):
        t = self._texto(ValueError("coluna 'kills' ausente"))
        self.assertNotIn("Reboot", t)
        self.assertIn("kills", t)

    def test_a_mensagem_diz_que_o_resto_esta_de_pe(self):
        t = self._texto(ValueError("qualquer"))
        self.assertIn("resto", t.lower())


class TestNaoVirouCobertor(unittest.TestCase):
    """Isolar a falha nao pode virar esconder a falha."""

    def test_a_falha_fica_registrada_para_os_testes_de_tela(self):
        st = _St()
        _despachar(st, {"X": lambda: 1 / 0})
        falhas = st.session_state.get(CHAVE_FALHAS_DE_ABA)
        self.assertEqual(len(falhas), 1)
        self.assertEqual(falhas[0][0], "X")
        self.assertIn("ZeroDivisionError", falhas[0][1])

    def test_sem_falha_o_registro_fica_vazio(self):
        st = _St()
        _despachar(st, {"X": lambda: None})
        self.assertEqual(st.session_state.get(CHAVE_FALHAS_DE_ABA), [])

    def test_quem_zera_o_registro_e_o_inicio_da_pagina(self):
        """Sobra de um desenho anterior faria a suite reprovar sozinha.

        O despacho das secundarias NAO pode zerar: as tabs principais sao
        desenhadas antes dele, e a falha delas sumiria antes de alguem ver.
        """
        st = _St()
        _despachar(st, {"X": lambda: 1 / 0})
        self.assertEqual(len(st.session_state[CHAVE_FALHAS_DE_ABA]), 1)

        iniciar_registro_de_falhas(st)      # novo desenho da pagina
        _despachar(st, {"X": lambda: None})
        self.assertEqual(st.session_state[CHAVE_FALHAS_DE_ABA], [])

    def test_falha_de_tab_principal_sobrevive_ao_despacho_das_secundarias(self):
        """O buraco que o helper fechou: 8 abas blindadas, 3 nao."""
        st = _St()
        iniciar_registro_de_falhas(st)
        desenhar_aba(st, _Aba(), "Visão Geral", lambda: 1 / 0)
        _despachar(st, {"Rankings": lambda: None})
        nomes = [n for n, _ in st.session_state[CHAVE_FALHAS_DE_ABA]]
        self.assertIn("Visão Geral", nomes)

    def test_duas_abas_quebradas_aparecem_as_duas(self):
        st = _St()
        _despachar(st, {"A": lambda: 1 / 0, "B": lambda: [][1]})
        self.assertEqual(len(st.session_state.get(CHAVE_FALHAS_DE_ABA)), 2)


class TestRerunEStopPassamReto(unittest.TestCase):
    """`st.rerun()` dentro de uma aba precisa continuar funcionando.

    As duas herdam de BaseException (conferido no streamlit 1.51.0), entao
    `except Exception` deixa passar. Se um dia alguem trocar por
    `except BaseException` para "ser mais seguro", qualquer botao que
    recarrega a pagina para de funcionar em silencio — e este teste avisa.
    """

    def test_o_pedido_de_recarregar_pagina_sobe(self):
        from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

        st = _St()
        with self.assertRaises(BaseException) as ctx:
            _despachar(st, {"X": lambda: (_ for _ in ()).throw(
                RerunException(None))})
        self.assertIsInstance(ctx.exception, RerunException)

    def test_o_pedido_de_parar_a_pagina_sobe(self):
        from streamlit.runtime.scriptrunner_utils.exceptions import StopException

        st = _St()
        with self.assertRaises(StopException):
            _despachar(st, {"X": lambda: (_ for _ in ()).throw(StopException())})

    def test_essas_duas_nao_herdam_de_Exception(self):
        from streamlit.runtime.scriptrunner_utils.exceptions import (
            RerunException, StopException,
        )
        for c in (RerunException, StopException):
            self.assertFalse(issubclass(c, Exception),
                             "{} virou Exception: o isolamento vai engoli-la "
                             "e o app trava".format(c.__name__))


class TestSemSessionState(unittest.TestCase):
    """O despacho e chamado tambem por testes com um `st` minimo."""

    def test_st_sem_session_state_nao_estoura(self):
        class _Cru:
            pass

        feitas = []
        dispatch_secondary_tabs(_Cru(), tab_refs={"X": _Aba()},
                                renderers={"X": lambda: feitas.append("X")},
                                order=("X",))
        self.assertEqual(feitas, ["X"])

    def test_st_sem_error_nao_estoura_quando_a_aba_quebra(self):
        class _Cru:
            pass

        dispatch_secondary_tabs(_Cru(), tab_refs={"X": _Aba()},
                                renderers={"X": lambda: 1 / 0},
                                order=("X",))


class TestAbaPrincipalIsolada(unittest.TestCase):
    """As tabs principais sao desenhadas soltas no arquivo do app.

    Antes de 13/09/2026 nenhuma delas tinha protecao: um erro em Parametros,
    Visao Geral ou Consistencia derrubava a pagina inteira do mesmo jeito.
    """

    def test_desenha_e_avisa_que_deu_certo(self):
        st = _St()
        feitas = []
        ok = desenhar_aba(st, _Aba(), "Parâmetros", lambda: feitas.append(1))
        self.assertTrue(ok)
        self.assertEqual(feitas, [1])

    def test_erro_nao_sobe_e_avisa_que_deu_errado(self):
        st = _St()
        iniciar_registro_de_falhas(st)
        ok = desenhar_aba(st, _Aba(), "Parâmetros", lambda: 1 / 0)
        self.assertFalse(ok)
        self.assertEqual(len(st.erros), 1)
        self.assertIn("Parâmetros", st.erros[0])

    def test_aba_ausente_nao_e_erro(self):
        """Nem toda tab existe em toda pagina; ausencia nao vira alarme."""
        st = _St()
        iniciar_registro_de_falhas(st)
        self.assertFalse(desenhar_aba(st, None, "X", lambda: None))
        self.assertFalse(desenhar_aba(st, _Aba(), "X", None))
        self.assertEqual(st.session_state[CHAVE_FALHAS_DE_ABA], [])
        self.assertEqual(st.erros, [])


if __name__ == "__main__":
    unittest.main()
