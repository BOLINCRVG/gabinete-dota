# -*- coding: utf-8 -*-
"""Motor de props de jogador (K/D/A): projeção contextual + filtro de atropelo.

Pedido dele (22/07): props dos 10 jogadores com linha da casa, edge, e um
resumo que ESCONDA as "linhas ridículas" — as que só batem porque o confronto
é um massacre (ex.: Gen.G × Pain), não porque o jogador tem valor.

PESQUISA (regra dele, feita antes de codar): esse problema tem nome no mercado —
*blowout risk / game script*. A recomendação da literatura de props é literal:
"monte seus props EM TORNO da sua visão do jogo, não independente dela;
incorpore a probabilidade de atropelo DENTRO do modelo". No basquete o mecanismo
é o astro sentar no 4º quarto; no LoL é o jogo acabar cedo (menos K/D/A p/ todos)
ou o time forte farmar abates. Efeito igual: o prop vira função do CONFRONTO.

A resposta deste módulo: projetar em COMPONENTES SEPARADOS, para dar p/ medir
quanto da projeção veio do jogador e quanto veio do massacre (stomp_share).
Assim o filtro é um número, não uma opinião.

    projeção = taxa_do_jogador × concessão_do_oponente × duração × efeito_do_gap

Fórmula vinda da pesquisa dos mercados (taxa esperada → distribuição → linha).
"""
from __future__ import annotations

import math
from typing import Any, Dict

# Sensibilidade de cada stat ao gap de força. Sinal:
#   kills/assists: time mais forte produz MAIS -> sobe com gap positivo
#   deaths:        time mais forte morre MENOS -> desce com gap positivo
# Magnitudes são PROVISÓRIAS até a medição OOS (mesmo rito do First X):
# ver tools/medir_gap_por_mercado.py. Enquanto não medidas, ficam conservadoras.
GAP_SENSITIVITY: Dict[str, float] = {
    "kills": +0.18,
    "assists": +0.12,
    "deaths": -0.18,
}
_STATS = tuple(GAP_SENSITIVITY)

# Acima disso, a projeção é considerada "movida pelo atropelo" e o resumo a
# esconde por padrão (escolha dele: resumo limpo + botão para revelar).
STOMP_SHARE_LIMIAR = 0.20


def _f(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def project_player_stat(
    *,
    base_rate: Any,
    stat: str,
    opp_concede_ratio: Any = 1.0,
    duration_ratio: Any = 1.0,
    mu_gap: Any = 0.0,
) -> Dict[str, Any]:
    """Projeta K/D/A de um jogador, devolvendo os COMPONENTES separados.

    base_rate        : taxa histórica do jogador por jogo (ex.: 3.1 kills).
    opp_concede_ratio: quanto o oponente concede vs a média da liga (1.0 = média;
                       1.2 = concede 20% a mais).
    duration_ratio   : duração esperada vs a média (1.0 = média; 1.1 = 10% mais
                       longo -> mais de tudo).
    mu_gap           : força do time do jogador MENOS a do oponente (WHR).

    Retorna projeção final, a projeção SEM o gap, e o stomp_share — a fração da
    projeção que veio do desequilíbrio e não do jogador.
    """
    out: Dict[str, Any] = {"stat": str(stat), "ok": False, "reason": ""}
    key = str(stat or "").strip().lower()
    if key not in GAP_SENSITIVITY:
        out["reason"] = "stat_desconhecido"
        return out
    rate = _f(base_rate)
    if not math.isfinite(rate) or rate < 0.0:
        out["reason"] = "taxa_invalida"
        return out

    occ = _f(opp_concede_ratio, 1.0)
    dur = _f(duration_ratio, 1.0)
    gap = _f(mu_gap, 0.0)
    occ = occ if occ > 0.0 else 1.0
    dur = dur if dur > 0.0 else 1.0

    # sem o gap: o que o jogador faria neste confronto ignorando o desequilíbrio
    proj_sem_gap = float(rate * occ * dur)
    # com o gap: o efeito do massacre entra aqui, e SÓ aqui
    mult_gap = 1.0 + float(GAP_SENSITIVITY[key]) * gap
    mult_gap = float(max(0.25, min(2.5, mult_gap)))  # nunca inverte nem explode
    proj = float(max(0.0, proj_sem_gap * mult_gap))

    denom = max(1e-9, abs(proj))
    stomp_share = float(min(1.0, abs(proj - proj_sem_gap) / denom))
    out.update({
        "ok": True,
        "reason": "ok",
        "proj": proj,
        "proj_sem_gap": float(proj_sem_gap),
        "base_rate": float(rate),
        "opp_concede_ratio": float(occ),
        "duration_ratio": float(dur),
        "mu_gap": float(gap),
        "gap_mult": float(mult_gap),
        "stomp_share": stomp_share,
        "movido_por_atropelo": bool(stomp_share >= STOMP_SHARE_LIMIAR),
    })
    return out


def _poisson_sf(k: int, lam: float) -> float:
    """P(X > k) para Poisson(lam) — a probabilidade de passar de uma linha .5."""
    lam = float(max(0.0, lam))
    if k < 0:
        return 1.0
    p, acc = math.exp(-lam), math.exp(-lam)
    for i in range(1, int(k) + 1):
        p *= lam / i
        acc += p
    return float(min(1.0, max(0.0, 1.0 - acc)))


def prob_over(proj: Any, line: Any) -> float:
    """P(estatística > linha). Contagens (K/D/A) seguem Poisson."""
    lam = _f(proj)
    ln = _f(line)
    if not (math.isfinite(lam) and math.isfinite(ln)):
        return float("nan")
    return _poisson_sf(int(math.floor(ln)), lam)


def edge_vs_odd(p: Any, odd_casa: Any) -> float:
    """Edge % = p × odd − 1. Positivo = a casa está pagando acima do justo."""
    pp, od = _f(p), _f(odd_casa)
    if not (math.isfinite(pp) and math.isfinite(od)) or od <= 1.0:
        return float("nan")
    return float(100.0 * (pp * od - 1.0))


def fair_odd(p: Any) -> float:
    pp = _f(p)
    if not math.isfinite(pp) or pp <= 0.0 or pp >= 1.0:
        return float("nan")
    return float(1.0 / pp)


def remover_vig(odd_over: Any, odd_under: Any) -> Dict[str, float]:
    """Tira a margem da casa e devolve a probabilidade que ELA está precificando.

    Pesquisa (regra dele, 22/07): o padrão do mercado é o *no-vig* — as duas
    probabilidades implícitas de um mercado somam MAIS que 1.00 (o excesso é a
    comissão); normalizando para somar 1.00 sobra a estimativa "verdadeira" da
    casa. Comparar a NOSSA probabilidade com essa (e não com a odd bruta) é o
    que diz se estamos DISCORDANDO do mercado, em vez de só medir a comissão.

    Devolve {} se faltar um dos lados — sem os dois não dá para tirar o vig.
    """
    a, b = _f(odd_over), _f(odd_under)
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 1.0 or b <= 1.0:
        return {}
    ia, ib = 1.0 / a, 1.0 / b
    soma = ia + ib
    if soma <= 0.0:
        return {}
    return {
        "p_casa_over": float(ia / soma),
        "p_casa_under": float(ib / soma),
        "vig_pct": float(100.0 * (soma - 1.0)),
        "odd_justa_casa_over": float(soma / ia),
        "odd_justa_casa_under": float(soma / ib),
    }


def divergencia_do_mercado(p_modelo: Any, p_casa_sem_vig: Any) -> float:
    """Quantos pontos percentuais o modelo discorda da casa (já sem vig).

    Positivo = o modelo acha MAIS provável que a casa. É o sinal de valor
    limpo — o edge contra a odd bruta ainda embute a comissão."""
    pm, pc = _f(p_modelo), _f(p_casa_sem_vig)
    if not (math.isfinite(pm) and math.isfinite(pc)):
        return float("nan")
    return float(100.0 * (pm - pc))


def build_prop_row(
    *,
    player: str,
    role: str,
    team: str,
    stat: str,
    line: Any,
    base_rate: Any,
    opp_concede_ratio: Any = 1.0,
    duration_ratio: Any = 1.0,
    mu_gap: Any = 0.0,
    odd_casa: Any = None,
) -> Dict[str, Any]:
    """Uma linha da tabela de props: projeção + probabilidade + edge + selo."""
    pr = project_player_stat(
        base_rate=base_rate, stat=stat, opp_concede_ratio=opp_concede_ratio,
        duration_ratio=duration_ratio, mu_gap=mu_gap,
    )
    row: Dict[str, Any] = {
        "player": str(player), "role": str(role), "team": str(team),
        "stat": str(stat), "line": _f(line), "ok": bool(pr.get("ok")),
        "reason": str(pr.get("reason") or ""),
    }
    if not pr.get("ok"):
        return row
    p_over = prob_over(pr["proj"], line)
    row.update({
        "proj": pr["proj"],
        "proj_sem_gap": pr["proj_sem_gap"],
        "p_over": p_over,
        "p_under": (1.0 - p_over) if math.isfinite(p_over) else float("nan"),
        "odd_justa_over": fair_odd(p_over),
        "odd_justa_under": fair_odd(1.0 - p_over) if math.isfinite(p_over) else float("nan"),
        "stomp_share": pr["stomp_share"],
        "movido_por_atropelo": pr["movido_por_atropelo"],
    })
    if odd_casa is not None:
        row["edge_pct"] = edge_vs_odd(p_over, odd_casa)
        row["odd_casa"] = _f(odd_casa)
    return row


def separar_por_atropelo(rows: list) -> tuple[list, list]:
    """Divide as linhas em (valor real, movidas pelo atropelo).

    O resumo mostra a primeira lista; a segunda fica atrás do botão (escolha
    dele: 'esconder por padrão, com botão pra ver')."""
    limpas, atropelo = [], []
    for r in (rows or []):
        if not isinstance(r, dict) or not r.get("ok"):
            continue
        (atropelo if r.get("movido_por_atropelo") else limpas).append(r)
    return limpas, atropelo
