# -*- coding: utf-8 -*-
"""Pagina SUPER-LIVE (LoL e Dota): quem esta na vantagem no mapa -- pre-jogo, depois do draft e ao
vivo. Desde 24/11/2026 a pagina e a MESA (app_core/super_live_mesa_core.py): uma aba por jogo, o draft
colado por ele. Aqui ficam a carga das camadas e o preco da Polymarket ao vivo.
Regras: app_core/super_live_core.py e app_core/camada_draft_core.py. Arquivo identico nos dois repos.

REGRA ABSOLUTA (16/11/2026): nada some calado. Time fora da base, jogador fora da base, fonte fora
do ar, Polymarket sem mercado: cada um aparece na tela com o motivo.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from app_core import camada_draft_core as CD
from app_core import super_live_core as SL


def _pct(p: Optional[float]) -> str:
    return "—" if p is None else f"{100 * p:.1f}%"


def _odd(p: Optional[float]) -> str:
    o = SL.odd_justa(p)
    return "—" if o is None else f"{o:.2f}"


def render_super_live(st: Any, *, raiz: Any, esporte: str, times: List[str],
                      resolver: Callable[[str], str]) -> None:
    raiz = Path(raiz)
    st.subheader("SUPER-LIVE — quem está na vantagem no mapa")
    st.caption("Três leituras do mesmo mapa: **pré-jogo** (o modelo), **pós-draft** (o modelo + os "
               "campeões/heróis escolhidos e a prática de cada jogador neles) e **ao vivo** (o estado do "
               "jogo: ouro, abates, farm). Pós-draft medido às cegas em 16/11/2026: melhora a previsão do "
               "mapa (logloss −0,014 no LoL, −0,012 no Dota).")
    camada = CD.carregar(raiz, esporte)
    modelos = st.session_state.get(f"_sl_modelos_{esporte}")
    if modelos is None:
        modelos = SL.carregar_modelos(raiz)
        st.session_state[f"_sl_modelos_{esporte}"] = modelos
    if camada is None:
        st.warning("A camada pós-draft ainda não foi gerada (camada_draft_%s.json): o pós-draft fica igual ao pré-jogo." % esporte)
    if not modelos:
        st.warning("Os modelos ao vivo (live_models_feed/) não estão no app: a leitura ao vivo fica de fora.")
    # MESA (24/11/2026): uma aba por jogo, draft colado por ELE (captura no Dota, texto no LoL), a tela
    # desenhada do draft e o ao vivo digitado. A lista antiga que puxava o draft sozinho da fonte ao vivo
    # saiu (ele: "essa parte de tentar o draft do ao vivo e um pouco mal trabalhado, deixa que eu digito").
    from app_core.super_live_mesa_core import render_mesa
    render_mesa(st, raiz=raiz, esporte=esporte, times=times, resolver=resolver, camada=camada, modelos=modelos)


def _tier(esporte: str, liga: str) -> int:
    if esporte == "dota":
        return 3  # no treino do Dota toda liga cai no mesmo grupo (medido em 16/11/2026)
    try:
        from mlcore.live_ml.features import _encode_tier
        return int(_encode_tier(liga))
    except Exception:  # noqa: BLE001
        return 5


def render_polymarket(st: Any, *, raiz: Path, esporte: str, jogo, nome_a, nome_b, p_mapa, resolver) -> None:
    st.markdown("**Polymarket (agora)**")
    ev = SL.evento_polymarket(raiz, esporte, jogo["azul"], jogo["vermelho"], resolver=resolver)
    if not ev:
        st.caption("Este confronto não está na Polymarket (o robô não viu evento com esses dois times nas últimas 18 h).")
        return
    slug, ta, tb = ev
    k = f"_sl_poly_{slug}"
    c = st.session_state.get(k)
    if not (isinstance(c, tuple) and time.time() - c[0] < 20):
        c = (time.time(), SL.polymarket_ao_vivo(raiz, slug))
        st.session_state[k] = c
    pm = c[1]
    if pm.get("motivo"):
        st.caption(f"Polymarket ({slug}): {pm['motivo']}.")
    mapas = pm.get("mapas") or {}
    n = jogo.get("mapa") or (min(mapas) if mapas else None)
    if n and n in mapas and p_mapa is not None:
        lados, p1, spread = mapas[n]
        r = resolver or (lambda x: x)
        primeiro_e_azul = (r(lados[0]) or lados[0]) == nome_a
        p_az = p1 if primeiro_e_azul else 1 - p1
        meia = (spread or 0) / 2
        compra_a = 1 / min(0.999, p_az + meia)
        compra_b = 1 / min(0.999, (1 - p_az) + meia)
        st.write(f"Mapa {n}: {nome_a} {100 * p_az:.1f}% (compra {compra_a:.2f}) · {nome_b} {100 * (1 - p_az):.1f}% "
                 f"(compra {compra_b:.2f}) · spread {spread if spread is not None else '—'}")
        for nome, p_mod, compra in ((nome_a, p_mapa, compra_a), (nome_b, 1 - p_mapa, compra_b)):
            if p_mod * compra > 1.0:
                st.success(f"Valor em **{nome}** no mapa {n}: modelo {100 * p_mod:.1f}% x odd de compra {compra:.2f} "
                           f"(retorno esperado {100 * (p_mod * compra - 1):+.1f}%).")
    elif mapas:
        st.caption(f"Mercados de mapa abertos: {sorted(mapas)}; o mapa deste jogo não está entre eles.")


# ---------------------------------------------------------------------------------------------
# DRAFT COLADO (23/11/2026): cola os herois -> quanto a odd mexe e POR QUE (app_core/draft_colado_core)
# ---------------------------------------------------------------------------------------------