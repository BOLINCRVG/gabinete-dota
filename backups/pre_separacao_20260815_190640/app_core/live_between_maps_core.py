"""Entre-mapas (série ao vivo): odd justa do PRÓXIMO mapa e da série restante,
com o momentum validado.

Base empírica (2026-07-03, tools/eval_series_momentum*.py, gate ACCEPT OOS):
- Quem LIDERA o placar joga o próximo mapa +d no logit (d=0.17 ≈ ±4pp); quem
  está atrás, -d. Placar EMPATADO (decider/1-1/2-2) = SEM ajuste (medido ~0).
- Ganho OOS no mapa-2: logloss 0.5754→0.5671 (+0.0084). A série PRÉ-jogo é
  neutra ao momentum (cancela no agregado) — o valor é ENTRE os mapas.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, MutableMapping, Optional

from app_core.resumo_core import edge_pct, odd_to_prob
from app_core.resumo_paste_core import discrete_stake_units, kelly_stake_fraction

MOMENTUM_D = 0.17  # fitado em 2025 (MLE), validado cego em 2026 — ver docstring.


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def p_next_map(p_base: float, score_a: int, score_b: int) -> float:
    """P(A vence o PRÓXIMO mapa) dado o placar — momentum só com líder definido."""
    if score_a == score_b:
        return float(p_base)
    shift = MOMENTUM_D if score_a > score_b else -MOMENTUM_D
    return float(_sig(_logit(p_base) + shift))


def p_series_rest(p_base: float, score_a: int, score_b: int, bo: int) -> float:
    """P(A fecha a série) dado o placar, com momentum estado-a-estado (DP)."""
    need = bo // 2 + 1
    memo: dict[tuple[int, int], float] = {}

    def rec(a: int, b: int) -> float:
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        key = (a, b)
        if key not in memo:
            pm = p_next_map(p_base, a, b)
            memo[key] = pm * rec(a + 1, b) + (1.0 - pm) * rec(a, b + 1)
        return memo[key]

    return float(rec(int(score_a), int(score_b)))


def _fair_odd(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return 1.0 / p


def resolve_p_base(
    *,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    artifact_path: str,
) -> tuple[Optional[float], str]:
    """P(A vence um mapa) pré-jogo. Preferência: análise principal (com todas as
    camadas) quando o confronto é o mesmo; senão, artefato simetrizado."""
    main_a = str(session_state.get("teamA", "") or "")
    main_b = str(session_state.get("teamB", "") or "")
    try:
        p_used = float(session_state.get("_p_map_used"))
    except (TypeError, ValueError):
        p_used = float("nan")
    if math.isfinite(p_used) and 0.0 < p_used < 1.0:
        if (team_a, team_b) == (main_a, main_b):
            return p_used, "análise principal"
        if (team_a, team_b) == (main_b, main_a):
            return 1.0 - p_used, "análise principal (invertida)"

    # Fallback: artefato ao vivo, simetrizado (cancela o intercepto de lado azul
    # do calibrador — mesma lição do estudo de momentum).
    try:
        from mlcore.artifact import MLArtifact
        p = Path(str(artifact_path or ""))
        if not p.exists():
            return None, "sem artefato"
        sig = (str(p), int(p.stat().st_mtime))
        cache = session_state.get("_lbm_art_cache")
        if not (isinstance(cache, tuple) and len(cache) == 2 and cache[0] == sig):
            cache = (sig, MLArtifact.load_json(str(p)))
            session_state["_lbm_art_cache"] = cache
        art = cache[1]
        p_fwd = float(art.model.p_blue(str(team_a), str(team_b)))
        p_rev = float(art.model.p_blue(str(team_b), str(team_a)))
        if art.calibrator is not None:
            p_fwd = float(art.calibrator.predict_proba(p_fwd))
            p_rev = float(art.calibrator.predict_proba(p_rev))
        p_sym = 0.5 * (p_fwd + (1.0 - p_rev))
        if 0.0 < p_sym < 1.0:
            return float(p_sym), "artefato (Elo+players calibrado)"
    except Exception:
        pass
    return None, "indisponível"


def render_between_maps_block(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    artifact_path: str,
    format_num_fn: Callable[[Any, int], str],
) -> None:
    if not team_a or not team_b or team_a == team_b:
        st.info("Escolha os dois times acima para abrir o entre-mapas.")
        return

    p_base, fonte = resolve_p_base(
        session_state=session_state, team_a=team_a, team_b=team_b, artifact_path=artifact_path,
    )
    if p_base is None:
        st.warning("Sem probabilidade pré-jogo disponível para este confronto (artefato ausente?).")
        return

    key_sfx = f"{team_a}|{team_b}"
    c_bo, c_sa, c_sb = st.columns([1, 1, 1])
    with c_bo:
        bo = st.radio("Formato", [3, 5], horizontal=True, key=f"lbm_bo_{key_sfx}")
    need = int(bo) // 2 + 1
    with c_sa:
        score_a = st.number_input(f"Mapas {team_a}", 0, need, step=1, key=f"lbm_sa_{key_sfx}")
    with c_sb:
        score_b = st.number_input(f"Mapas {team_b}", 0, need, step=1, key=f"lbm_sb_{key_sfx}")
    score_a, score_b = int(score_a), int(score_b)

    if score_a >= need or score_b >= need:
        st.success(f"Série encerrada: {team_a} {score_a} × {score_b} {team_b}.")
        return

    st.caption(f"Base pré-jogo (por mapa): **{team_a} {100.0 * p_base:.1f}%** · fonte: {fonte}")

    pa_next = p_next_map(p_base, score_a, score_b)
    pa_series = p_series_rest(p_base, score_a, score_b, int(bo))

    if score_a == score_b:
        mom_txt = "Placar empatado — **sem ajuste** (momentum zera em decider, medido nos dados)."
    else:
        lider = team_a if score_a > score_b else team_b
        dp = abs(pa_next - p_base) * 100.0
        mom_txt = f"Momentum aplicado: **{lider}** (líder {max(score_a, score_b)}–{min(score_a, score_b)}) ganha **{dp:+.1f}pp** no próximo mapa."
    st.markdown(mom_txt)

    n1, n2 = st.columns(2)
    with n1:
        st.markdown(
            f"**Próximo mapa**  \n"
            f"{team_a}: **{format_num_fn(_fair_odd(pa_next), 2)}** ({100.0 * pa_next:.1f}%)  \n"
            f"{team_b}: **{format_num_fn(_fair_odd(1.0 - pa_next), 2)}** ({100.0 * (1.0 - pa_next):.1f}%)"
        )
    with n2:
        st.markdown(
            f"**Fechar a série** (de {score_a}–{score_b}, MD{bo})  \n"
            f"{team_a}: **{format_num_fn(_fair_odd(pa_series), 2)}** ({100.0 * pa_series:.1f}%)  \n"
            f"{team_b}: **{format_num_fn(_fair_odd(1.0 - pa_series), 2)}** ({100.0 * (1.0 - pa_series):.1f}%)"
        )

    # Edge/stake do próximo mapa contra a odd da casa (mesma régua do Resumo:
    # ¼ Kelly → unidades discretas 0.5/0.75/1/1.25/1.5/2).
    e1, e2 = st.columns(2)
    for col, side_team, p_side in ((e1, team_a, pa_next), (e2, team_b, 1.0 - pa_next)):
        with col:
            oc_txt = st.text_input(
                f"Odd da casa — {side_team} (próx. mapa)",
                key=f"lbm_oc_{side_team}_{key_sfx}",
                placeholder="ex: 1.85",
            )
            oc = odd_to_prob(None) if not oc_txt else None
            try:
                oc = float(str(oc_txt).replace(",", ".")) if str(oc_txt).strip() else None
            except (TypeError, ValueError):
                oc = None
            if oc and oc > 1.0:
                edge = edge_pct(p_side, oc)
                k = kelly_stake_fraction(p_side, oc, fraction=0.25) if (edge or 0) > 0 else None
                stake = discrete_stake_units(k)
                cor = "#22c55e" if (edge or 0) > 0 else "#ef4444"
                stake_txt = f" · stake **{stake:g}u**" if stake else (" · sem valor" if (edge or 0) <= 0 else " · edge pequeno")
                st.markdown(
                    f"<span style='color:{cor};font-weight:700;'>edge {edge:+.1f}%</span>{stake_txt}",
                    unsafe_allow_html=True,
                )
