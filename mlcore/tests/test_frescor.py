# -*- coding: utf-8 -*-
"""Testes do vigia de frescor — tools/frescor.py.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ Este arquivo existe por causa da dívida D6: `frescor.py` imprimia 8 réguas  ║
║ no estado `atrasada` e, logo abaixo, "todas as reguas declaradas estao      ║
║ dentro do prazo". O vigia do envelhecimento dava sinal verde com régua      ║
║ vencida, e não havia UM teste cobrindo o arquivo.                           ║
║                                                                             ║
║ O que estes testes travam é a regra da fonte 5 do próprio `frescor.py`:     ║
║ as duas faixas (aviso e erro) emitem sinais DIFERENTES, e o resumo não      ║
║ pode calar sobre a de aviso.                                                ║
║                                                                             ║
║ Nota de honestidade: são testes do RELATO (o que o resumo diz), não do      ║
║ cálculo de idade — o cálculo depende de carimbo em arquivo real e fica de   ║
║ fora. Se alguém mexer na fórmula da idade, estes testes não pegam.          ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import importlib.util as _iu
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path


def _raiz() -> Path:
    saida = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(Path(__file__).resolve().parent),
                           capture_output=True, text=True, check=True)
    return Path(saida.stdout.strip())


ROOT = _raiz()
_CAMINHO = ROOT / "tools" / "frescor.py"
if not _CAMINHO.exists():          # no Dota a ferramenta mora na raiz
    _CAMINHO = ROOT / "frescor.py"

_spec = _iu.spec_from_file_location("_frescor", _CAMINHO)
_frescor = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_frescor)


def _escreve(pasta: Path, nome: str, dias_atras: float) -> None:
    """Grava um artefato de evidência com carimbo de N dias atrás.

    Carimbo INGÊNUO (sem fuso), que é como as ferramentas de verdade gravam —
    `datetime.now().isoformat()`. Escrever com fuso aqui faria o teste medir
    um formato que não existe em produção."""
    quando = datetime.now() - timedelta(days=dias_atras)
    (pasta / nome).write_text(
        json.dumps({"gerado_em": quando.isoformat(timespec="seconds")}),
        encoding="utf-8")


class TestResumoDoFrescor(unittest.TestCase):
    """O resumo tem que contar a verdade que a tabela acima ja mostrou."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.raiz = Path(self._tmp.name)
        (self.raiz / "evidence").mkdir(parents=True, exist_ok=True)
        # Escolhe uma regua qualquer que TENHA cadencia declarada.
        self.arquivo, (_, self.cadencia) = next(
            (k, v) for k, v in _frescor.CADENCIA.items() if v[1])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _roda(self, dias: float) -> str:
        """Aciona pelo argv, que e como a ferramenta roda de verdade."""
        _escreve(self.raiz / "evidence", self.arquivo, dias)
        argv = sys.argv
        sys.argv = ["frescor.py", "--root", str(self.raiz),
                    "--out", "evidence/_frescor_teste.json"]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                _frescor.main()
        finally:
            sys.argv = argv
        return buf.getvalue()

    def test_dentro_do_prazo_diz_que_esta_tudo_no_prazo(self):
        saida = self._roda(self.cadencia * 0.5)
        self.assertIn("dentro do prazo", saida)

    def test_atrasada_NAO_pode_dizer_que_esta_tudo_no_prazo(self):
        """O bug da D6, travado: 2x a cadencia e atraso, e atraso nao e 'ok'."""
        saida = self._roda(self.cadencia * 2.5)
        self.assertIn("atrasada", saida, "a tabela tem que marcar como atrasada")
        self.assertNotIn(
            "todas as reguas declaradas estao dentro do prazo", saida,
            "REGRESSAO DA D6: o resumo deu sinal verde com regua atrasada")

    def test_atrasada_sai_no_resumo_como_AVISO(self):
        saida = self._roda(self.cadencia * 2.5)
        self.assertIn("AVISO", saida)
        self.assertIn("ATRASADA", saida)

    def test_velha_continua_saindo_como_ATENCAO(self):
        """A faixa de erro nao pode ter sido perdida no conserto da de aviso."""
        saida = self._roda(self.cadencia * 4.0)
        self.assertIn("VELHA", saida)
        self.assertIn("ATENCAO", saida)
        self.assertNotIn("todas as reguas declaradas estao dentro do prazo", saida)

    def test_aviso_e_erro_sao_niveis_distintos(self):
        """Atrasada nao pode disparar o alarme de VELHA, nem o contrario."""
        atrasada = self._roda(self.cadencia * 2.5)
        self.assertNotIn("regua(s) VELHA(s)", atrasada)
        velha = self._roda(self.cadencia * 4.0)
        self.assertNotIn("regua(s) ATRASADA(s)", velha)


if __name__ == "__main__":
    unittest.main()
