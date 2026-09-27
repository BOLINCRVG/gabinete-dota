"""Post-fusion calibration helper for the live ML pipeline.

Applies Microseg + Platt re-calibration on the fused probability (p_fused), AFTER
team+players fusion. The ablation study (tools/run_ablation_cal_position.py) showed
that calibrating both BEFORE and AFTER fusion yields:
  - +0.97pp logloss vs calibrating only before fusion (current default behavior)
  - ECE reduced by 48% (0.085 -> 0.044 on 1313-sample holdout)

Bandeira: session_state["ml_postfusion_cal_enabled"]. O texto antigo aqui dizia
"default False", o que estava VENCIDO: o default de runtime e True
(app_core/ml_settings_runtime_core.py, com teste garantindo).

MAS A BANDEIRA LIGADA NAO SIGNIFICA A ETAPA ATIVA (achado de 04/09/2026).
O texto anterior afirmava "a etapa ESTA ativa em producao" e isso e FALSO.
Este modulo so faz alguma coisa se receber, por injecao, `apply_microseg_fn`,
`apply_fused_postcal_fn` ou `postfusion_microseg_table` — e no caminho servido
NENHUM dos tres e passado: o contrato `MlPreEngineInput`
(app_core/ml_pre_pipeline_models.py) nao tem esses campos, e
`build_pre_engine_input` nao os produz. Conferido rodando com os argumentos
exatos do app: p_out == p_in, applied == False.
Detalhe e perguntas em aberto na divida D8 do RETOMAR.md. NAO ligar os
callbacks sem medir o efeito isolado disso.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║ (Preenchido em 04/09/2026 — este arquivo estava na lista da divida D2, sem  ║
║ bloco FONTES, e a frente 0 o toca. Regra 5.)                                ║
║                                                                             ║
║ 1. PLATT: por que uma sigmoide ajustada sobre a saida do modelo, e nao a    ║
║    saida crua. Platt (1999), "Probabilistic outputs for support vector      ║
║    machines...". A implementacao esta em mlcore/calibration.py.             ║
║                                                                             ║
║ 2. O GRUPO ENTRA COMO FEATURE DE UM CALIBRADOR SO, nao um calibrador por    ║
║    grupo — assim os grupos compartilham a inclinacao global e o grupo       ║
║    pequeno encolhe sozinho, sem corte arbitrario de "minimo N jogos".       ║
║    O desenho, a escolha de L2/agrupamento e a medicao estao no bloco        ║
║    FONTES de tools/gate_calibrador_final.py — que carrega tambem            ║
║    Cawley & Talbot, JMLR 11:2079-2107 (2010),                               ║
║    https://jmlr.org/papers/v11/cawley10a.html (conferido 04/09/2026).       ║
║                                                                             ║
║ 3. RECALIBRAR TEM LIMITE: opera apenas sobre a ordenacao que o modelo ja    ║
║    produz. Um modelo que chuta sempre a taxa-base e perfeitamente           ║
║    calibrado e inutil. Logo, este arquivo NAO conserta perda de ordenacao;  ║
║    para isso e preciso refazer o modelo. Ver a discussao do ponto 3 no      ║
║    RETOMAR.md, com Schnellinger EM, Yang W, Kimmel SE, Diagn Progn Res      ║
║    2021;5:20, https://pmc.ncbi.nlm.nih.gov/articles/PMC8647501/             ║
║    (conferido 04/09/2026).                                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝

CALIBRADOR COM GRUPO (27/08/2026) — desligado ate os coeficientes existirem
em params_fitted.json["postfusion_group_calibration"]. Sem o bloco, o
comportamento e exatamente o de antes; remover o bloco desfaz.

Medido em teste cego de 5.525 jogos: o numero servido e SUBCONFIANTE (dizia
68.8% onde acontecia 75.7%), porque o calibrador final e ajustado em
temporadas ANTIGAS e a distorcao a corrigir e da atual. Ajustar em dado
recente e dar ao calibrador o GRUPO (formato de serie + fase) como entrada:
  logloss 0.58577 -> 0.56209   ganho +0.02368  IC 95% [+0.01847, +0.02866]
  Z de Spiegelhalter -8.97 -> -2.94
O grupo entra como FEATURE de um calibrador so (nao um calibrador por grupo):
assim os grupos compartilham a inclinacao global e o grupo pequeno encolhe
sozinho, sem corte arbitrario de "minimo N jogos".
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from app_core.prob_math_core import safe_float as _f
from mlcore.agrupamentos import chave as _chave_do_grupo
from mlcore.agrupamentos import nome_do_bloco as _nome_do_agrupamento


def _calibrador_com_grupo(coef, bo: str, phase: str):
    """Devolve a funcao de calibracao, ou None quando nao ha coeficientes.

    Formato do bloco em params_fitted.json:
      {"agrupamento": "formato + fase",
       "niveis": ["3|regular", ...], "w": [w_z, w_1, w_z_g1, w_1_g1, ...]}
    A ordem das features e [logit(p), 1] seguida de [ind*logit(p), ind] para
    cada nivel — o primeiro nivel do ajuste e a REFERENCIA e nao aparece.

    A CHAVE DO GRUPO vem do campo `agrupamento` do proprio bloco (divida D10,
    06/09/2026). Antes era cravada como f"{bo}|{phase}", entao dos quatro
    agrupamentos que a validacao cruzada pode escolher so um era
    representavel: quando ela escolhia outro, a paridade divergia e o
    reajuste ficava bloqueado. Bloco sem o campo cai no legado, que era o
    unico que podia estar em producao. Ver mlcore/agrupamentos.py."""
    if not coef:
        return None
    niveis = list(coef.get("niveis") or [])
    w = [float(x) for x in (coef.get("w") or [])]
    if len(w) != 2 + 2 * len(niveis):
        return None
    try:
        chave = _chave_do_grupo(_nome_do_agrupamento(coef), bo, phase)
    except KeyError:
        # Agrupamento que este app nao sabe montar: NAO servir o global
        # calado fingindo que aplicou o grupo — desligar e o comportamento
        # honesto, e o de antes.
        return None

    def aplicar(p: float) -> float:
        pc = min(max(float(p), 1e-12), 1.0 - 1e-12)
        z = math.log(pc / (1.0 - pc))
        s = w[0] * z + w[1]
        for i, nv in enumerate(niveis):
            if chave == nv:
                s += w[2 + 2 * i] * z + w[3 + 2 * i]
                break
        return 1.0 / (1.0 + math.exp(-s)) if s > -700 else 0.0

    return aplicar


def apply_postfusion_calibration(
    p_fused: float,
    *,
    enabled: bool,
    league_hint: str,
    bo: int,
    split_hint: str = "",
    phase_hint: str = "regular",
    side: str = "blue",
    microseg_table: Optional[Mapping[str, Any]] = None,
    microseg_min_n: int = 30,
    microseg_strength: float = 0.60,
    microseg_cap_pp: float = 8.0,
    microseg_hierarchical: bool = True,
    microseg_hier_decay: float = 0.80,
    apply_microseg_fn: Optional[Callable[..., Tuple[float, Dict[str, Any]]]] = None,
    apply_fused_postcal_fn: Optional[Callable[..., Tuple[float, Dict[str, Any]]]] = None,
    use_fitted_postcal: bool = True,
    grupo_coef: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Apply Microseg + Platt to a fused probability and return (p_out, trace_info).

    If enabled is False, returns (p_fused, info_with_enabled_false) unchanged.
    """
    info: Dict[str, Any] = {
        "enabled": bool(enabled),
        "applied": False,
        "p_in": float(p_fused) if isinstance(p_fused, (int, float)) and math.isfinite(float(p_fused)) else None,
        "p_out": None,
        "microseg": {"applied": False},
        "postcal": {"applied": False},
    }
    if not bool(enabled):
        info["reason"] = "disabled"
        info["p_out"] = info["p_in"]
        return float(p_fused), info

    p_in = _f(p_fused, float("nan"))
    if not math.isfinite(p_in) or p_in <= 0.0 or p_in >= 1.0:
        info["reason"] = "invalid_p_in"
        info["p_out"] = info["p_in"]
        return float(p_fused), info

    p_out = float(p_in)
    bo_hint = str(int(bo)) if int(bo) in (1, 3, 5) else "1"

    # Step 1: Microseg on p_fused
    if callable(apply_microseg_fn) and microseg_table:
        try:
            p_after_ms, ms_info = apply_microseg_fn(
                float(p_out),
                league=str(league_hint or ""),
                split=str(split_hint or ""),
                side=str(side or "blue"),
                bo=bo_hint,
                phase=str(phase_hint or "regular"),
                table=microseg_table,
                min_n=int(microseg_min_n),
                strength=float(microseg_strength),
                cap_pp=float(microseg_cap_pp),
                hierarchical=bool(microseg_hierarchical),
                hierarchy_decay=float(microseg_hier_decay),
            )
            if isinstance(ms_info, dict):
                info["microseg"] = dict(ms_info)
                info["microseg"]["applied"] = bool(ms_info.get("applied", abs(float(p_after_ms) - p_out) > 1e-9))
            if math.isfinite(float(p_after_ms)):
                p_out = float(p_after_ms)
        except Exception as exc:
            info["microseg"]["error"] = f"{type(exc).__name__}: {exc}"

    # Step 2b: calibrador COM GRUPO. Roda no lugar do Platt simples quando os
    # coeficientes existem; sem eles, cai no caminho de sempre.
    grupo_cal = _calibrador_com_grupo(grupo_coef, bo_hint, str(phase_hint or "regular"))
    if grupo_cal is not None:
        try:
            p_novo = grupo_cal(float(p_out))
            if math.isfinite(p_novo) and 0.0 < p_novo < 1.0:
                info["postcal"] = {"applied": True, "modo": "grupo",
                                   "grupo": f"{bo_hint}|{phase_hint or 'regular'}",
                                   "p_in": float(p_out), "p_out": float(p_novo)}
                p_out = float(p_novo)
                info["p_out"] = float(p_out)
                info["applied"] = bool(abs(float(p_out) - p_in) > 1e-9)
                info["diff_pp"] = float(100.0 * (p_out - p_in))
                return float(p_out), info
        except (TypeError, ValueError, KeyError) as exc:
            info["postcal"]["error_grupo"] = f"{type(exc).__name__}: {exc}"

    # Step 2: Fused post-calibration (Platt) on p_fused (or microseg'd p_fused)
    if callable(apply_fused_postcal_fn):
        try:
            p_after_pc, pc_info = apply_fused_postcal_fn(
                float(p_out),
                league_hint=str(league_hint or ""),
                use_fitted=bool(use_fitted_postcal),
            )
            if isinstance(pc_info, dict):
                info["postcal"] = dict(pc_info)
            if math.isfinite(float(p_after_pc)):
                p_out = float(p_after_pc)
        except Exception as exc:
            info["postcal"]["error"] = f"{type(exc).__name__}: {exc}"

    info["p_out"] = float(p_out)
    info["applied"] = bool(abs(float(p_out) - p_in) > 1e-9)
    info["diff_pp"] = float(100.0 * (p_out - p_in))
    return float(p_out), info
