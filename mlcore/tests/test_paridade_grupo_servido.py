# -*- coding: utf-8 -*-
"""PARIDADE: o caminho SERVIDO aplica o mesmo calibrador que o gate mediu.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ Por que este arquivo existe, em uma frase: porque a divida D8 mostrou que   ║
║ ninguem estava vigiando isto, e o erro passou meses.                        ║
║                                                                             ║
║ 1. O NOME DO PERIGO: divergencia treino/servico (training-serving skew) —   ║
║    "o descasamento entre os dados ou features usados no treino e os vistos  ║
║    no momento de servir". A regra que as fontes dao e a REGRA DE OURO:      ║
║    "garanta que treino e producao se imitem o mais perto possivel", e "use  ║
║    o MESMO esquema para validar os dados de treino e os de servico".        ║
║    https://www.giskard.ai/glossary/training-serving-skew                    ║
║    https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring ║
║    (as duas conferidas abrindo as paginas em 06/09/2026)                    ║
║                                                                             ║
║ 1b. O QUE E INFERENCIA MINHA, e estava aqui como se fosse das fontes        ║
║    (corrigido em 06/09/2026): que o TESTE CEGO nao revela isso, porque o    ║
║    conjunto de teste e construido pelo mesmo pipeline e herda a             ║
║    divergencia. NENHUMA das duas paginas diz isso. Continua valendo como    ║
║    raciocinio — e foi o que aconteceu aqui, o erro so apareceu LENDO o      ║
║    caminho servido — mas e conclusao minha, sustentada pelo caso.           ║
║                                                                             ║
║ 2. A FORMA DO TESTE: rodar as MESMAS entradas pelos dois caminhos e exigir  ║
║    igualdade dentro de tolerancia numerica apertada, falhando em qualquer   ║
║    divergencia. NAO CONFERIDA: a pagina que eu citava aqui devolve 403 e    ║
║    nao pude abrir, entao a forma fica como pratica que eu apliquei, sem     ║
║    fonte verificavel. O que a sustenta e a REGRA DE OURO da fonte 1 e o     ║
║    proprio caso da D8.                                                      ║
║                                                                             ║
║ 3. ISTO E O ITEM MONITORAMENTO 3 DA REGUA DAS 28 ("treino e servico nao     ║
║    divergem"), que esta VERMELHO nos dois repositorios desde 26/08. Este    ║
║    arquivo nao pinta a regua de verde: cobre UM estagio, o do calibrador    ║
║    com grupo. Os outros continuam sem vigia.                                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import math
import unittest

from app_core.postfusion_calibration_core import (
    _calibrador_com_grupo,
    apply_postfusion_calibration,
)

# Bloco de coeficientes com a MESMA forma do que ajustar_calibrador_grupo.py
# grava: um nivel de referencia (que nao aparece) e os demais com dois pesos.
_COEF = {
    "niveis": ["1|regular", "3|playoffs", "3|regular"],
    "w": [1.296, 0.361, 0.0037, 0.0065, 0.178, -0.3147, -0.8308, -0.4131],
}
_TOL = 1e-12


def _servido(p, *, bo, phase, coef=_COEF):
    """Exatamente como o motor chama, no caminho servido: sem microssegmento
    da fusao e sem Platt da fusao (o app nao roda esses dois), so o grupo."""
    saida, _ = apply_postfusion_calibration(
        p, enabled=True, league_hint="LCK", bo=bo, split_hint="",
        phase_hint=phase, side="blue",
        microseg_table=None, apply_microseg_fn=None, apply_fused_postcal_fn=None,
        grupo_coef=coef,
    )
    return saida


class TestParidadeServidoVsMedido(unittest.TestCase):

    def test_servido_bate_com_o_calibrador_medido(self):
        """O numero servido tem que ser o MESMO que o gate mediu."""
        for bo, phase in ((1, "regular"), (3, "regular"), (3, "playoffs"),
                          (5, "regular"), (5, "playoffs")):
            medido = _calibrador_com_grupo(_COEF, str(bo), phase)
            for p in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
                with self.subTest(bo=bo, phase=phase, p=p):
                    self.assertAlmostEqual(
                        _servido(p, bo=bo, phase=phase), medido(p), delta=_TOL,
                        msg="divergencia treino/servico no calibrador com grupo")

    def test_grupo_desconhecido_cai_no_global_nos_dois_caminhos(self):
        """Fase que nao esta nos niveis usa so os pesos globais — e os dois
        caminhos tem que concordar nisso tambem."""
        medido = _calibrador_com_grupo(_COEF, "1", "fase_que_nao_existe")
        for p in (0.2, 0.5, 0.8):
            self.assertAlmostEqual(_servido(p, bo=1, phase="fase_que_nao_existe"),
                                   medido(p), delta=_TOL)

    def test_sem_bloco_o_app_nao_muda_nada(self):
        """O interruptor: sem coeficientes, o numero sai como entrou."""
        for p in (0.05, 0.5, 0.95):
            self.assertEqual(_servido(p, bo=3, phase="regular", coef=None), p)

    def test_bloco_malformado_nao_muda_nada_e_nao_estoura(self):
        """Bloco com tamanho errado de `w` e ignorado, nao aplicado pela metade."""
        ruim = {"niveis": ["1|regular"], "w": [1.0, 0.0]}   # faltam 2 pesos
        for p in (0.3, 0.7):
            self.assertEqual(_servido(p, bo=1, phase="regular", coef=ruim), p)

    def test_o_estagio_muda_de_verdade_quando_ligado(self):
        """Guarda contra o teste virar vacuo: com bloco, o numero TEM que mudar.

        Sem isto, um bug que fizesse o grupo nunca aplicar deixaria todos os
        testes acima passando — foi assim que a etapa pos-fusao ficou morta
        com a bandeira ligada (divida D8)."""
        p = 0.65
        self.assertGreater(abs(_servido(p, bo=3, phase="regular") - p), 1e-6,
                           "o calibrador ligado nao mexeu no numero: estagio inerte?")

    def test_a_inclinacao_afia(self):
        """w[0] = 1,296 > 1 AFIA: as probabilidades se afastam UMA DA OUTRA.

        Cuidado que me pegou ao escrever isto: afiar NAO e "empurrar para longe
        de 50%". O intercepto (0,361) desloca o centro — o ponto fixo fica em
        ~22,8%, nao em 50%, entao p=0,35 SOBE em vez de descer. A propriedade
        que a inclinacao garante e o espalhamento em logito, e e essa que o
        teste tem que travar."""
        def _logito(p):
            return math.log(p / (1.0 - p))

        p1, p2 = 0.35, 0.65
        antes = _logito(p2) - _logito(p1)
        depois = (_logito(_servido(p2, bo=1, phase="qualquer"))
                  - _logito(_servido(p1, bo=1, phase="qualquer")))
        self.assertGreater(depois, antes,
                           "inclinacao <= 1: o calibrador esta achatando, nao afiando")
        self.assertAlmostEqual(depois / antes, _COEF["w"][0], delta=1e-9)

    def test_saida_continua_sendo_probabilidade(self):
        for p in (1e-6, 0.001, 0.5, 0.999, 1 - 1e-6):
            q = _servido(p, bo=3, phase="regular")
            self.assertTrue(0.0 <= q <= 1.0 and math.isfinite(q), f"p={p} -> {q}")



class TestParidadeEmTodosOsAgrupamentos(unittest.TestCase):
    """A D10: paridade tem que valer para os QUATRO, nao so para bo|phase."""

    def _bloco(self, agrupamento, niveis):
        n = len(niveis)
        return {"agrupamento": agrupamento, "niveis": niveis,
                "w": [1.29, 0.36] + [0.1 * (i + 1) for i in range(2 * n)]}

    def test_paridade_em_cada_agrupamento(self):
        from mlcore.agrupamentos import AGRUPAMENTOS, chave as _ch
        casos = {
            "nenhum (so global)": [],
            "formato BO": ["3", "5"],
            "fase": ["regular"],
            "formato + fase": ["3|regular", "5|playoffs"],
        }
        self.assertEqual(set(casos), set(AGRUPAMENTOS))
        for agrupamento, niveis in casos.items():
            bloco = self._bloco(agrupamento, niveis)
            for bo, ph in ((1, "regular"), (3, "regular"), (5, "playoffs")):
                for p in (0.2, 0.5, 0.8):
                    with self.subTest(a=agrupamento, bo=bo, ph=ph, p=p):
                        medido = _calibrador_com_grupo(bloco, str(bo), ph)
                        self.assertAlmostEqual(
                            _servido(p, bo=bo, phase=ph, coef=bloco),
                            medido(p), delta=_TOL,
                            msg=f"divergencia em '{agrupamento}'")

    def test_agrupamento_que_o_app_nao_conhece_DESLIGA(self):
        """Nao servir o global fingindo que aplicou o grupo."""
        bloco = {"agrupamento": "inventado", "niveis": ["x"],
                 "w": [1.29, 0.36, 0.1, 0.1]}
        for p in (0.3, 0.7):
            self.assertEqual(_servido(p, bo=3, phase="regular", coef=bloco), p)

    def test_bloco_legado_sem_agrupamento_continua_valendo(self):
        """O que esta em producao hoje nao pode quebrar com esta mudanca."""
        legado = {"niveis": ["3|regular"], "w": [1.29, 0.36, -0.8, -0.4]}
        medido = _calibrador_com_grupo(legado, "3", "regular")
        for p in (0.2, 0.5, 0.8):
            self.assertAlmostEqual(_servido(p, bo=3, phase="regular", coef=legado),
                                   medido(p), delta=_TOL)

if __name__ == "__main__":
    unittest.main()


class TestOEstagioEAlcancadoNoApp(unittest.TestCase):
    """A reserva: sem ela, o calibrador NUNCA era alcancado (08/09/2026).

    Os dois pontos de aplicacao no motor vivem dentro de ramos `use_lineup`,
    e `use_lineup` vem DESLIGADO por padrao. Medido abrindo o app: com e sem o
    bloco em params_fitted.json, o numero servido era IDENTICO ate o ultimo
    digito. O ganho medido nao chegava na tela.

    Estes testes travam o CONTRATO da aplicacao de reserva. Eles nao substituem
    o A/B no app -- nenhum teste desta suite instancia o Streamlit --, mas
    pegam quem mexer no motor e quebrar a chamada.
    """

    def test_o_motor_tem_a_aplicacao_de_reserva(self):
        import inspect
        from app_core.ml_pre_engine_core import run_ml_pre_engine_layers
        src = inspect.getsource(run_ml_pre_engine_layers)
        self.assertIn("_pfc_grupo_aplicado", src,
                      "o marcador da reserva sumiu do motor")
        self.assertIn("postfusion_calibration_reserva", src,
                      "a aplicacao de reserva sumiu do motor")

    def test_a_reserva_fica_FORA_dos_ramos_de_escalacao(self):
        """Se ela voltar para dentro de `use_lineup`, o bug volta inteiro."""
        import inspect, re
        from app_core.ml_pre_engine_core import run_ml_pre_engine_layers
        linhas = inspect.getsource(run_ml_pre_engine_layers).split("\n")
        alvo = next(i for i, l in enumerate(linhas)
                    if "not _pfc_grupo_aplicado" in l)
        recuo = len(linhas[alvo]) - len(linhas[alvo].lstrip())
        # o guarda tem que estar no nivel do corpo da funcao (recuo 4), nao
        # aninhado dentro de um `if use_lineup` (recuo 8+)
        self.assertLessEqual(recuo, 4,
                             f"a reserva esta aninhada (recuo {recuo}): "
                             "voltou para dentro de um ramo condicional")

    def test_sem_bloco_a_reserva_nao_faz_nada(self):
        """O interruptor continua sendo o bloco no params_fitted.json."""
        for p in (0.2, 0.5, 0.8):
            self.assertEqual(_servido(p, bo=3, phase="regular", coef=None), p)
