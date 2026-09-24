# -*- coding: utf-8 -*-
"""Handicap de kills e total de kills POR TIME -- os dois mercados que faltavam.

O PEDIDO (24/09/2026). Ele aposta kills ("sao mais logicas") e a aba de
Oportunidades mostrava o handicap de kills e o total de cada time so com o
preco da Pinnacle: o motor calculava a SOMA dos dois times, nunca cada um.

A CONTA
  - Kills esperados de cada time pelo MESMO Dixon-Coles que o motor ja usa no
    total (`mlcore.lines_engine.apply_dc_blend`): ataque de A x defesa de B,
    sobre o nivel geral. O motor calculava as duas metades e somava; aqui
    elas saem separadas. Mesmas constantes (meia-vida, amostra minima).
  - A FORMA do erro NAO e uma curva suposta: sao os quantis do residuo
    medidos no historico, como em `totais_forma_empirica.json`. Medido em
    21/09 (tools/handicap_de_abates.py): a diferenca de kills tem curtose
    -1,14 (achatada) e os kills dos dois times tem correlacao -0,229 --
    supor normal ou independencia erra os dois.
  - SEM LADO. A Pinnacle diz "mandante" e "visitante", que nao e azul e
    vermelho. A forma do handicap e medida com cada jogo contado nas DUAS
    direcoes, entao e simetrica por construcao e nao carrega a vantagem do
    lado azul (+0,75 kill) que a primeira regua tinha.

O LIMITE QUE ELE PRECISA SABER (visto na tela do Dota, 24/09/2026). A conta
enxerga so kills feitos e sofridos, NAO a forca de quem estava do outro lado.
Aurora x GamerLegion: a Pinnacle deu Aurora -12,5; a conta deu os dois quase
iguais (28,5/24,5 contra 27,7/23,8 de kills feitos/sofridos) e uma
"vantagem" de +40%. Na media ela passa na prova cega; num favorito claro
contra time medio, pode errar feio -- e a tela marca "grande demais". O
caminho seguinte e juntar a chance de vitoria do modelo de vencedor, e
MEDIR antes de ligar.

COM A CHANCE DE VITORIA (24/09/2026, tools/abates_com_vitoria.py). Resolve
o limite acima no CONFRONTO ABERTO, o unico com a chance do cartao. Medida
com a regua da nota honesta (chance do lado AZUL, feita no dia); servida com
a do cartao, que e a MEDIA dos dois lados (ml_prediction_core: 0,5 x (ida +
volta)). A diferenca e so o empurrao de lado, que a tela nao teria como
saber antes do jogo: a casa nao diz quem e azul.

O artefato (`abates_por_time_forma.json`) e gerado e MEDIDO por
`tools/gerar_abates_por_time.py`: ajuste em 2024-2025, prova cega em 2026.
Sem ele, as funcoes devolvem None e a tela diz "nao calcula" -- nunca um
numero de curva suposta (PRINCIPIO IV).
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ARTEFATO = Path(__file__).resolve().parents[1] / "abates_por_time_forma.json"
_CACHE: Dict[str, Any] = {}


def carregar(caminho: Optional[Path] = None) -> Dict[str, Any]:
    """O artefato, lido uma vez. Vazio quando falta ou esta quebrado."""
    chave = str(caminho or ARTEFATO)
    if chave in _CACHE:
        return _CACHE[chave]
    d: Dict[str, Any] = {}
    try:
        p = Path(chave)
        if p.exists():
            bruto = json.loads(p.read_text(encoding="utf-8"))
            # MERCADO POR MERCADO: so entra o bloco que passou na prova cega
            # e esta inteiro. No Dota o total por time NAO passou (erro de
            # calibracao 2,3 pp em 2026) e fica sem numero; o handicap passou.
            if isinstance(bruto, dict):
                d = {k: v for k, v in bruto.items()
                     if not k.startswith("_") and _bloco_valido(v)}
    except Exception:
        d = {}
    _CACHE[chave] = d
    return d


def _bloco_valido(reg: Any) -> bool:
    if not isinstance(reg, dict):
        return False
    q = reg.get("quantis")
    return isinstance(q, list) and len(q) >= 50 and q == sorted(q)


def medias_dc(team_games: Any, time_a: str, time_b: str) -> Optional[Tuple[float, float]]:
    """(kills esperados de A, kills esperados de B), ou None.

    LETRA POR LETRA as pecas de `apply_dc_blend` para kills: mesma data de
    referencia (a mais recente do quadro), mesma meia-vida, mesmo nivel
    geral, mesma amostra minima. Se divergisse, o total da tela e a soma
    destes dois numeros seriam contas diferentes para a mesma partida.
    """
    import pandas as pd

    from mlcore.lines_engine import DC_BLEND_HL_DAYS, DC_BLEND_MIN_EFF, _dc_decayed
    try:
        if team_games is None or team_games.empty:
            return None
        df = team_games
        for c in ("date", "team", "kills_for", "kills_against"):
            if c not in df.columns:
                return None
        as_of = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(as_of):
            return None
        hl, minimo = float(DC_BLEND_HL_DAYS), float(DC_BLEND_MIN_EFF)
        lm, ln = _dc_decayed(df["date"], df["kills_for"], as_of, hl)
        if not math.isfinite(lm) or lm <= 1e-9 or ln < minimo:
            return None
        ra = df[df["team"].astype(str) == str(time_a)]
        rb = df[df["team"].astype(str) == str(time_b)]
        if ra.empty or rb.empty:
            return None
        att_a, n1 = _dc_decayed(ra["date"], ra["kills_for"], as_of, hl)
        def_a, n2 = _dc_decayed(ra["date"], ra["kills_against"], as_of, hl)
        att_b, n3 = _dc_decayed(rb["date"], rb["kills_for"], as_of, hl)
        def_b, n4 = _dc_decayed(rb["date"], rb["kills_against"], as_of, hl)
        if min(n1, n2, n3, n4) < minimo:
            return None
        if not all(math.isfinite(x) for x in (att_a, def_a, att_b, def_b)):
            return None
        return att_a * def_b / lm, att_b * def_a / lm
    except Exception:
        return None


def _acima(quantis, folga: float) -> float:
    """P(residuo > folga) pelos quantis."""
    i = bisect.bisect_right(quantis, folga)
    return 1.0 - i / len(quantis)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def prob_cobre_handicap(e_lado: float, e_outro: float, pontos: float,
                        artefato: Optional[Dict[str, Any]] = None,
                        p_lado: Optional[float] = None) -> Optional[float]:
    """P(kills do lado + pontos > kills do outro), ou None.

    `pontos` como a casa escreve: -7,5 para quem da o handicap, +7,5 para
    quem recebe. Linha inteira (empate devolvido) nao e tratada: a casa so
    oferece meia linha neste mercado.

    `p_lado` = chance de o lado VENCER o mapa, do modelo de vencedor. Com
    ela, usa o bloco `handicap_com_vitoria` (tools/abates_com_vitoria.py,
    cego 23/06-21/09/2026: +0,0036 de Brier sobre a mesma conta sem ela).
    A conta sem vitoria nao enxerga a forca do adversario.
    """
    art = artefato if artefato is not None else carregar()
    # Sem o bloco com vitoria (Dota: nao ha regua do vencedor), volta a conta
    # de sempre -- nunca "sem numero" so porque a chance veio junto.
    if p_lado is not None and (art or {}).get("handicap_com_vitoria"):
        reg = (art or {}).get("handicap_com_vitoria") or {}
        q, coef = reg.get("quantis"), reg.get("coef")
        try:
            media = (float(coef[0]) * (float(e_lado) - float(e_outro))
                     + float(coef[1]) * _logit(p_lado))
            escala = max(float(reg["a"]) + float(reg["b"]) * (float(e_lado) + float(e_outro)), 1.0)
            return _acima(q, (-float(pontos) - media) / escala) if q else None
        except (TypeError, ValueError, KeyError, IndexError):
            return None
    reg = (art or {}).get("handicap") or {}
    q = reg.get("quantis")
    try:
        # ENCOLHIDA pela inclinacao medida: o Dixon-Coles EXAGERA a diferenca
        # (medido: quando ele preve +10, sai ~+8; inclinacao 0,75 a 0,84 em
        # 2024-2026). Sem centro: simetrico, nao ha lado.
        media = float(reg.get("inclinacao", 1.0)) * (float(e_lado) - float(e_outro))
        p = float(pontos)
    except (TypeError, ValueError):
        return None
    if not q or not (math.isfinite(media) and math.isfinite(p)):
        return None
    escala = _escala(reg, float(e_lado) + float(e_outro))
    if escala is None:
        return None
    # diferenca > -pontos  <=>  residuo > (-pontos - media)
    return _acima(q, (-p - media) / escala)


def prob_total_do_time_acima(e_time: float, linha: float,
                             artefato: Optional[Dict[str, Any]] = None,
                             p_time: Optional[float] = None) -> Optional[float]:
    """P(kills do time > linha), ou None.

    `p_time` = chance de ESTE time vencer o mapa; com ela, o bloco
    `total_time_com_vitoria` (cego: +0,0039 sobre a conta no ar).
    """
    art = artefato if artefato is not None else carregar()
    if p_time is not None and (art or {}).get("total_time_com_vitoria"):
        reg = (art or {}).get("total_time_com_vitoria") or {}
        q, coef = reg.get("quantis"), reg.get("coef")
        try:
            media = (float(coef[0]) + float(coef[1]) * float(e_time)
                     + float(coef[2]) * _logit(p_time))
            if not q or media <= 0:
                return None
            return _acima(q, (float(linha) - media) / math.sqrt(media))
        except (TypeError, ValueError, KeyError, IndexError):
            return None
    reg = (art or {}).get("total_time") or {}
    q = reg.get("quantis")
    try:
        # Reta medida: kills reais ~ centro + inclinacao x esperado. O
        # Dixon-Coles previa kills DE MENOS por time em 2024-2025 (+0,5 e
        # +0,4) e quase certo em 2026 (+0,1): o centro deixa de ir escondido
        # dentro da forma.
        media = float(reg.get("centro", 0.0)) + float(reg.get("inclinacao", 1.0)) * float(e_time)
        l = float(linha)
    except (TypeError, ValueError):
        return None
    if not q or not (math.isfinite(media) and math.isfinite(l)) or media <= 0:
        return None
    escala = _escala(reg, media)
    if escala is None:
        return None
    return _acima(q, (l - media) / escala)


def _escala(reg: Dict[str, Any], x: float) -> Optional[float]:
    """Por quanto o residuo foi dividido quando a forma foi medida.

    "bruto": 1 (o residuo em kills). "raiz": raiz da media (kills de um time
    espalham como contagem). "reta": a + b x total esperado (a diferenca
    espalha mais em jogo sangrento). Qual vale e decidido pela medicao.
    """
    modo = str(reg.get("modo") or "bruto")
    if modo == "bruto":
        return 1.0
    if modo == "raiz":
        return math.sqrt(x) if x > 0 else None
    if modo == "reta":
        a, b = reg.get("a"), reg.get("b")
        try:
            v = float(a) + float(b) * x
        except (TypeError, ValueError):
            return None
        return max(v, 1.0)
    return None
