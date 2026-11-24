# -*- coding: utf-8 -*-
"""Pagina SUPER-LIVE (LoL e Dota): quem esta na vantagem no mapa -- pre-jogo, depois do draft e ao
vivo. Desde 24/11/2026 a pagina e a MESA (app_core/super_live_mesa_core.py): uma aba por jogo, o draft
colado por ele. Aqui fica a carga das camadas (sem Polymarket/valor: a decisao de entrar e dele).
Regras: app_core/super_live_core.py e app_core/camada_draft_core.py. Arquivo identico nos dois repos.

REGRA ABSOLUTA (16/11/2026): nada some calado. Time fora da base, jogador fora da base, fonte fora
do ar, cada um aparece na tela com o motivo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List

from app_core import camada_draft_core as CD
from app_core import super_live_core as SL


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
