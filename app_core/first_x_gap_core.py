# -*- coding: utf-8 -*-
"""Ajuste de FORÇA (gap do WHR) na probabilidade dos mercados "First X".

MEDIDO, não chutado (22/07, 10.638 jogos fora-da-amostra, gap = mu do WHR
construído só com o treino):

    evento          ganho Brier OOS     veredito
    first_blood        +0.11%           ~neutro  -> NÃO ajusta
    first_dragon       +0.11%           ~neutro  -> NÃO ajusta
    first_tower        +0.78%           ACCEPT   -> ajusta
    first_baron        +1.16%           ACCEPT   -> ajusta

A leitura: eventos CEDO e aleatórios (first blood, first dragon) são quase
moeda — a força do time não prevê quem pega. Eventos DECIDIDOS (torre, baron)
acontecem quando o jogo já pendeu, então quem é mais forte tende a levar.

Os totais (kills/torre/dragão/baron/inibidor/tempo) NÃO recebem este ajuste:
medido ~neutro em todos, porque o combine for/against JÁ carrega o contexto do
oponente (o total vs HLE já difere do total vs Vitality). Aplicar gap por cima
seria redundante — e no baron chegou a piorar. Ver ARQUITETURA_ALVO.md.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

# Coeficientes MEDIDOS (beta no espaço logit, sobre o gap normalizado por gap_sd
# — o desvio-padrão do gap de mu no treino). Só entram os eventos que ACEITARAM
# no gate OOS. Reproduzível: scratchpad/gap_whr_por_mercado.py (determinístico,
# rodou 2x com números idênticos); evidência em first_x_gap_calib.json.
FIRST_X_GAP_CALIB: Dict[str, Dict[str, float]] = {
    # ganho Brier OOS +0.78% | beta/sd = 0.450 de logit por unidade de mu
    "firsttower": {"beta": 0.2359, "gap_sd": 0.5240},
    # ganho Brier OOS +1.16% | beta/sd = 0.408 de logit por unidade de mu
    "firstbaron": {"beta": 0.2171, "gap_sd": 0.5323},
}


def _norm_name(v: Any) -> str:
    s = str(v or "").strip().lower()
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return "".join(ch for ch in s if ch.isalnum())


def mu_gap_from_trace(trace: Any, *, team: Any, opponent: Any) -> Any:
    """mu(time) - mu(oponente) a partir do trace do ML, ou None se indisponível.

    Lê `shadow_state_areal.state.{blue,red}` do trace que o pipeline já grava em
    session_state["_ml_trace_last"]. Casa pelo NOME do time (o trace guarda
    blue/red, a aba pensa em time/oponente). Devolve None — nunca um número
    inventado — quando o trace não existe, a camada está desligada, ou os nomes
    não casam; nesse caso o First X fica exatamente como é hoje.
    """
    try:
        state = ((trace or {}).get("shadow_state_areal") or {}).get("state") or {}
        blue = state.get("blue") or {}
        red = state.get("red") or {}
    except (AttributeError, TypeError):
        return None
    if not isinstance(blue, dict) or not isinstance(red, dict):
        return None
    t, o = _norm_name(team), _norm_name(opponent)
    if not t or not o:
        return None
    nb, nr = _norm_name(blue.get("team")), _norm_name(red.get("team"))
    if t == nb and o == nr:
        mu_t, mu_o = blue.get("mu"), red.get("mu")
    elif t == nr and o == nb:
        mu_t, mu_o = red.get("mu"), blue.get("mu")
    else:
        return None  # trace é de outro confronto
    try:
        gap = float(mu_t) - float(mu_o)
    except (TypeError, ValueError):
        return None
    return gap if math.isfinite(gap) else None


def _clip01(p: float) -> float:
    return float(max(1e-6, min(1.0 - 1e-6, float(p))))


def _logit(p: float) -> float:
    q = _clip01(p)
    return float(math.log(q / (1.0 - q)))


def _sigmoid(z: float) -> float:
    zz = float(max(-35.0, min(35.0, float(z))))
    return float(1.0 / (1.0 + math.exp(-zz)))


def first_x_gap_supported(event_col: Any) -> bool:
    """Este First X recebe ajuste de força? (só os que aceitaram no gate)."""
    return str(event_col or "").strip().lower() in FIRST_X_GAP_CALIB


def adjust_first_x_prob(
    p_base: Any,
    *,
    event_col: Any,
    mu_gap: Any,
) -> Tuple[float, Dict[str, Any]]:
    """P(o time pega o objetivo) ajustada pela diferença de força.

    p_base : taxa histórica do time (o que a tela já mostra hoje).
    event_col : 'firsttower' | 'firstbaron' | ... (os demais passam intactos).
    mu_gap : mu_do_time - mu_do_oponente (WHR). Positivo = time é mais forte.

    Retorna (p_ajustada, info). Entrada inválida ou evento não-suportado
    devolve p_base inalterada — nunca inventa número.
    """
    info: Dict[str, Any] = {"applied": False, "event": str(event_col or ""),
                            "reason": "", "p_before": None, "p_after": None,
                            "mu_gap": None, "beta": None}
    try:
        p0 = float(p_base)
    except (TypeError, ValueError):
        info["reason"] = "p_base_invalida"
        return float("nan"), info
    if not math.isfinite(p0):
        info["reason"] = "p_base_nao_finita"
        return p0, info
    info["p_before"] = float(p0)

    key = str(event_col or "").strip().lower()
    cal = FIRST_X_GAP_CALIB.get(key)
    if not cal:
        info["reason"] = "evento_sem_ajuste_medido"
        info["p_after"] = float(p0)
        return float(p0), info

    try:
        gap = float(mu_gap)
    except (TypeError, ValueError):
        gap = float("nan")
    if not math.isfinite(gap):
        info["reason"] = "gap_indisponivel"
        info["p_after"] = float(p0)
        return float(p0), info

    sd = float(cal.get("gap_sd") or 1.0)
    if not math.isfinite(sd) or sd <= 0.0:
        sd = 1.0
    beta = float(cal.get("beta") or 0.0)
    p1 = _sigmoid(_logit(p0) + beta * (gap / sd))
    info.update({"applied": True, "reason": "ok", "p_after": float(p1),
                 "mu_gap": float(gap), "beta": float(beta),
                 "diff_pp": float(100.0 * (p1 - p0))})
    return float(p1), info
