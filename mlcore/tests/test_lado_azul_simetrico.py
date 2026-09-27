# -*- coding: utf-8 -*-
"""O numero da tela nao pode mudar conforme a ordem em que ele digita os times.

O ACHADO (16/09/2026). Ele mandou dois prints do MESMO confronto -- Natus
Vincere x Movistar KOI -- com numeros diferentes, e atribuiu ao CSV que tinha
acabado de subir. Medindo, o CSV nao tinha mexido NADA ali: 0.567 antes,
0.567 depois, e nenhum dos dois times jogou nos 54 jogos que entraram.

O que mudou foi a ORDEM. Num print a NAVI estava no Time 1; no outro, o KOI.

A CAUSA: o calibrador carrega o INTERCEPTO DO LADO AZUL. Ele manda p=0.50
para 0.5253, e a vitoria do lado azul nos dados e 53,62% em 8.758 mapas -- o
calibrador esta CERTO, aprendeu uma vantagem que existe. So que o caminho
servido passa o Time 1 como lado azul, e na tela o "Time 1" e apenas a caixa
onde ele digitou primeiro. Num BO5 os lados ainda se alternam a cada mapa.

E O PROJETO JA SIMETRIZAVA EM TRES LUGARES, menos no caminho servido:
    mlcore/fit_samples_core.py          (as amostras de TREINO)
    app_core/live_between_maps_core.py
    tools/calibrate_key_expectations.py
Ou seja: o modelo era TREINADO com a probabilidade simetrizada e SERVIDO sem
ela -- divergencia treino/servico.

MEDIDO (tools/medir_lado_azul.py, 6.331 mapas, ordem arbitraria):
    hoje         logloss 0.58871  nota 8.28
    simetrizado  logloss 0.58760  nota 8.32
e a inconsistencia por confronto caiu de 4,8 para 1,54 pontos (mediana).
No app, no confronto dele: de 5,7 pontos para 0,13.
"""
from __future__ import annotations

import inspect
import math
import unittest

from app_core import ml_artifact_pricing_core as m
try:
    from mlcore.tests._repo import so_no_lol
except ImportError:  # rodado de dentro da pasta de testes
    from _repo import so_no_lol


class _CalibradorComVies:
    """Um Platt com intercepto de lado, como o de producao.

    cal(0.50) = 0.5253 e o valor REAL medido no artefato em 16/09/2026.
    """

    def predict_proba(self, p):
        # sigmoide deslocada: reproduz cal(p)+cal(1-p) ~ 1.05
        z = math.log(float(p) / (1.0 - float(p))) if 0 < float(p) < 1 else 0.0
        return 1.0 / (1.0 + math.exp(-(0.9 * z + 0.101)))


def _simetrico(cal, p):
    return 0.5 * (cal.predict_proba(p) + (1.0 - cal.predict_proba(1.0 - p)))


class TestOCalibradorDePRODUCAOTemViesDeLado(unittest.TestCase):
    """Guarda o diagnostico: sem isto, ninguem sabe por que o conserto existe."""

    def test_o_calibrador_cru_NAO_e_simetrico(self):
        cal = _CalibradorComVies()
        soma = cal.predict_proba(0.5) + cal.predict_proba(0.5)
        self.assertGreater(soma, 1.01, "este dublê deveria ter vies de lado")

    def test_simetrizar_FAZ_somar_um(self):
        cal = _CalibradorComVies()
        for p in (0.30, 0.45, 0.50, 0.55, 0.70):
            a = _simetrico(cal, p)
            b = _simetrico(cal, 1.0 - p)
            self.assertAlmostEqual(a + b, 1.0, places=9,
                                   msg="a ordem ainda mudaria o numero em p=%s" % p)

    def test_confronto_equilibrado_da_MEIO_a_meio(self):
        """Dois times iguais nao podem dar 52,5% para quem foi digitado
        primeiro."""
        self.assertAlmostEqual(_simetrico(_CalibradorComVies(), 0.5), 0.5, places=9)

    def test_simetrizar_NAO_inverte_o_favorito(self):
        """O conserto tira o vies de lado, nao troca quem e melhor."""
        cal = _CalibradorComVies()
        self.assertGreater(_simetrico(cal, 0.70), 0.5)
        self.assertLess(_simetrico(cal, 0.30), 0.5)

    def test_simetrizar_preserva_a_ordem_entre_confrontos(self):
        cal = _CalibradorComVies()
        vals = [_simetrico(cal, p) for p in (0.20, 0.35, 0.50, 0.65, 0.80)]
        self.assertEqual(vals, sorted(vals))


class TestOCaminhoSERVIDOSimetriza(unittest.TestCase):
    """O caminho servido era o unico do projeto que nao fazia isso."""

    FONTE = inspect.getsource(m)

    def test_o_sentido_inverso_e_calculado(self):
        self.assertIn("p_raw_rev", self.FONTE)
        self.assertIn("team_red,", self.FONTE)

    def test_a_media_dos_dois_sentidos_e_a_conta_usada(self):
        self.assertIn("0.5 * (cal_f + (1.0 - cal_r))", self.FONTE)

    def test_a_liga_e_o_tier_tambem_trocam_de_lado(self):
        """Passar o inverso com a liga do lado errado mediria outra coisa --
        e o erro nao apareceria, so mudaria o numero um pouco."""
        self.assertIn("blue_league=red.get(\"league\")", self.FONTE)
        self.assertIn("red_league=blue.get(\"league\")", self.FONTE)
        self.assertIn("blue_tier=red.get(\"tier\")", self.FONTE)

    def test_sem_o_sentido_inverso_a_falha_e_DITA(self):
        """PRINCIPIO IV: cair no calculo antigo em silencio faria o numero
        mudar de natureza sem ninguem saber."""
        self.assertIn("sem_sentido_inverso_nao_simetrizado", self.FONTE)


class TestOsOUTROSLugaresContinuamSimetrizando(unittest.TestCase):
    """Se um deles voltar atras, dois numeros do app discordam de novo."""

    def test_as_amostras_de_treino_simetrizam(self):
        from mlcore import fit_samples_core as f
        self.assertIn("0.5 * (p_cal_f + (1.0 - p_cal_r))", inspect.getsource(f))

    def test_o_entre_mapas_simetriza(self):
        from app_core import live_between_maps_core as l
        self.assertIn("0.5 * (p_fwd + (1.0 - p_rev))", inspect.getsource(l))


class TestAMedicaoExiste(unittest.TestCase):
    """Numero central do app nao se muda no palpite."""

    @so_no_lol("a medicao do lado azul foi feita com dados do LoL")
    def test_a_ferramenta_de_medicao_esta_no_repositorio(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        alvo = raiz / "tools" / "medir_lado_azul.py"
        self.assertTrue(alvo.exists(), "a medicao que justificou o conserto sumiu")

    @so_no_lol("a medicao do lado azul foi feita com dados do LoL")
    def test_a_medicao_avalia_as_DUAS_ordens(self):
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        src = (raiz / "tools" / "medir_lado_azul.py").read_text(encoding="utf-8")
        self.assertIn("_inverter", src)
        self.assertIn("ordem arbitraria", src)

    @so_no_lol("a medicao do lado azul foi feita com dados do LoL")
    def test_a_medicao_NAO_reimplementa_o_caminho(self):
        """PRINCIPIO V: usa replay_pre_jogo, que existe para isso."""
        from pathlib import Path
        raiz = Path(__file__).resolve().parents[2]
        src = (raiz / "tools" / "medir_lado_azul.py").read_text(encoding="utf-8")
        self.assertIn("from mlcore.replay_pre_jogo import", src)


if __name__ == "__main__":
    unittest.main()
