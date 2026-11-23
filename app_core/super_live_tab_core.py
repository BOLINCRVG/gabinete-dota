# -*- coding: utf-8 -*-
"""Pagina SUPER-LIVE (LoL e Dota): quem esta na vantagem no mapa -- pre-jogo, depois do draft e ao
vivo -- puxando o jogo sozinho (lolesports / OpenDota) e com digitacao como reserva.
Regras: app_core/super_live_core.py e app_core/camada_draft_core.py. Arquivo identico nos dois repos.

REGRA ABSOLUTA (16/11/2026): nada some calado. Time fora da base, jogador fora da base, fonte fora
do ar, Polymarket sem mercado: cada um aparece na tela com o motivo.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app_core import camada_draft_core as CD
from app_core import super_live_core as SL


def _pct(p: Optional[float]) -> str:
    return "—" if p is None else f"{100 * p:.1f}%"


def _odd(p: Optional[float]) -> str:
    o = SL.odd_justa(p)
    return "—" if o is None else f"{o:.2f}"


def _nome_heroi(raiz: Path) -> Callable[[str], str]:
    import json
    try:
        d = json.loads((Path(raiz) / "herois_dota.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    return lambda h: d.get(str(h), str(h))


def _lista_ao_vivo(st: Any, esporte: str) -> Tuple[List[Dict[str, Any]], str]:
    """Cache de 20 s na sessao (a fonte atualiza a cada ~10 s; ninguem precisa martelar)."""
    k = f"_sl_lista_{esporte}"
    c = st.session_state.get(k)
    if isinstance(c, tuple) and time.time() - c[0] < 20:
        return c[1], c[2]
    try:
        js = SL.dota_ao_vivo() if esporte == "dota" else SL.lol_ao_vivo()
        motivo = ""
    except Exception as e:  # noqa: BLE001
        js, motivo = [], f"a fonte ao vivo nao respondeu ({type(e).__name__})"
    st.session_state[k] = (time.time(), js, motivo)
    return js, motivo


def _draft_chaves(esporte: str, lado: Sequence[Tuple[str, ...]], siglas: Sequence[str]) -> List[Tuple[str, str]]:
    if esporte == "dota":  # (conta, heroi, nome)
        return [(CD.chave_jogador(x[0]), CD.chave_campeao(x[1])) for x in lado]
    return [(CD.chave_jogador(CD.nome_sem_tag(x[0], siglas)), CD.chave_campeao(x[1])) for x in lado]


def _rotulos(esporte: str, lado, siglas, heroi) -> List[Tuple[str, str]]:
    if esporte == "dota":
        return [(x[2] or x[0], heroi(x[1])) for x in lado]
    return [(CD.nome_sem_tag(x[0], siglas), x[1]) for x in lado]


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
    heroi = _nome_heroi(raiz)

    if esporte == "dota":
        try:
            render_draft_colado(st, raiz=raiz, esporte=esporte, times=times, camada=camada, heroi=heroi)
        except Exception as e:  # noqa: BLE001
            st.error(f"O draft colado não abriu: {type(e).__name__}: {e}")
        st.divider()
        st.markdown("#### Ao vivo")

    jogos, motivo = _lista_ao_vivo(st, esporte)
    fonte = "OpenDota (ao vivo)" if esporte == "dota" else "lolesports (ao vivo)"
    c1, c2 = st.columns([4, 1])
    if c2.button("Atualizar agora", key=f"sl_atualizar_{esporte}"):
        st.session_state.pop(f"_sl_lista_{esporte}", None)
        st.rerun()
    if motivo:
        c1.warning(f"{fonte}: {motivo}. Use a digitação abaixo.")
    elif not jogos:
        c1.info(f"Nenhum mapa de liga rolando agora segundo {fonte}. Para um jogo que não apareceu, use a digitação abaixo.")

    opcoes = [f"{j['azul']} x {j['vermelho']}" + (f" — {j['liga']}" if esporte == "lol" and j.get("liga") else "")
              + (f" — mapa {j['mapa']}" if j.get("mapa") else "")
              + (f" — min {j['estado']['minuto']:.0f}" if j.get("estado") else "") for j in jogos]
    escolha = None
    if jogos:
        i = c1.selectbox("Jogo ao vivo", list(range(len(jogos))), format_func=lambda k: opcoes[k],
                         key=f"sl_jogo_{esporte}")
        escolha = jogos[int(i)]

    with st.expander("Digitar o jogo (se a fonte falhar ou o jogo não aparecer)", expanded=not jogos):
        manual = _render_manual(st, esporte, times, raiz)
    if manual is not None:
        escolha = manual

    if escolha is None:
        return
    estado = escolha.get("estado")
    if estado is None and escolha.get("fonte") == "lolesports":
        try:
            cache = st.session_state.setdefault("_sl_inicio_lol", {})
            estado = SL.lol_estado(escolha["id_jogo"], inicio_cache=cache)
        except Exception as e:  # noqa: BLE001
            st.warning(f"O estado do mapa não abriu na lolesports ({type(e).__name__}); digite abaixo se precisar.")
            estado = None
    _render_analise(st, raiz=raiz, esporte=esporte, jogo=escolha, estado=estado, camada=camada,
                    modelos=modelos, resolver=resolver, heroi=heroi)


def _id_heroi(raiz: Path) -> Callable[[str], str]:
    """'Dragon Knight' / 'dragon knight' / 'DK'?? -> '49' (pelo nome; numero passa direto)."""
    import json
    try:
        d = json.loads((Path(raiz) / "herois_dota.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    por_nome = {CD.chave_campeao(v): k for k, v in d.items()}
    return lambda h: str(h).strip() if str(h).strip().isdigit() else por_nome.get(CD.chave_campeao(h), str(h))


def _render_manual(st: Any, esporte: str, times: List[str], raiz: Path) -> Optional[Dict[str, Any]]:
    k = f"sl_man_{esporte}"
    ligar = st.checkbox("Usar o jogo digitado", key=f"{k}_on")
    lista = [""] + sorted(times)
    a = st.selectbox("Time azul" + (" (Radiant)" if esporte == "dota" else ""), lista, key=f"{k}_a")
    b = st.selectbox("Time vermelho" + (" (Dire)" if esporte == "dota" else ""), lista, key=f"{k}_b")
    rot = "heróis (nome ou número)" if esporte == "dota" else "campeões"
    st.caption(f"Draft: os 5 {rot} de cada lado, separados por vírgula, na ordem top/jungle/mid/adc/sup "
               "(LoL) ou pos 1–5 (Dota). Jogadores opcionais, na mesma ordem.")
    da = st.text_input(f"Azul: {rot}", key=f"{k}_da")
    db = st.text_input(f"Vermelho: {rot}", key=f"{k}_db")
    ja = st.text_input("Azul: jogadores (opcional)", key=f"{k}_ja")
    jb = st.text_input("Vermelho: jogadores (opcional)", key=f"{k}_jb")
    cols = st.columns(4)
    minuto = cols[0].number_input("Minuto", 0.0, 90.0, 0.0, 1.0, key=f"{k}_min")
    ouro = cols[1].number_input("Ouro azul − vermelho", -60000.0, 60000.0, 0.0, 500.0, key=f"{k}_ouro")
    ka = cols[2].number_input("Abates azul", 0, 150, 0, key=f"{k}_ka")
    kb = cols[3].number_input("Abates vermelho", 0, 150, 0, key=f"{k}_kb")
    if not ligar or not a or not b:
        return None
    sep = lambda s: [x.strip() for x in str(s or "").split(",") if x.strip()]  # noqa: E731
    ca, cb, pa, pb = sep(da), sep(db), sep(ja), sep(jb)
    if esporte == "dota":
        idh = _id_heroi(raiz)
        ca, cb = [idh(x) for x in ca], [idh(x) for x in cb]
    lado = lambda cs, ps: [((ps[i] if i < len(ps) else ""), c, "") for i, c in enumerate(cs)]  # noqa: E731
    if esporte == "dota":
        lado = lambda cs, ps: [((ps[i] if i < len(ps) else ""), c, (ps[i] if i < len(ps) else "")) for i, c in enumerate(cs)]  # noqa: E731
    est = {"fonte": "digitado", "minuto": float(minuto), "situacao": "",
           "azul": {"draft": lado(ca, pa), "ouro": max(ouro, 0.0), "abates": float(ka), "mortes": float(kb)},
           "vermelho": {"draft": lado(cb, pb), "ouro": max(-ouro, 0.0), "abates": float(kb), "mortes": float(ka)}}
    return {"fonte": "digitado", "azul": a, "vermelho": b, "siglas": ("", ""), "mapa": None, "liga": "",
            "estado": est, "ja_resolvido": True}


def _tier(esporte: str, liga: str) -> int:
    if esporte == "dota":
        return 3  # no treino do Dota toda liga cai no mesmo grupo (medido em 16/11/2026)
    try:
        from mlcore.live_ml.features import _encode_tier
        return int(_encode_tier(liga))
    except Exception:  # noqa: BLE001
        return 5


def _render_analise(st: Any, *, raiz: Path, esporte: str, jogo: Dict[str, Any], estado: Optional[Dict[str, Any]],
                    camada: Optional[Dict[str, Any]], modelos, resolver, heroi) -> None:
    from app_core.live_between_maps_core import resolve_p_base
    az_raw, vm_raw = jogo["azul"], jogo["vermelho"]
    az = az_raw if jogo.get("ja_resolvido") else (resolver(az_raw) or "")
    vm = vm_raw if jogo.get("ja_resolvido") else (resolver(vm_raw) or "")
    avisos: List[str] = []
    p_pre, origem = None, ""
    if az and vm:
        p_pre, origem = resolve_p_base(session_state=st.session_state, team_a=az, team_b=vm,
                                       artifact_path=str(raiz / "ml_artifact.json"))
    for bruto, res in ((az_raw, az), (vm_raw, vm)):
        if not res:
            avisos.append(f"**{bruto}** não está na base do modelo (time novo ou nome sem apelido salvo): "
                          "sem p pré-jogo, a análise parte de 50%.")
    if p_pre is None:
        if az and vm:
            avisos.append(f"p pré-jogo indisponível ({origem}); a análise parte de 50%.")
        p_base = 0.5
    else:
        p_base = float(p_pre)

    # pos-draft
    p_pos, faltam, linhas = p_base, [], ([], [])
    draft_ok = bool(estado) and len(estado["azul"].get("draft") or []) == 5 and len(estado["vermelho"].get("draft") or []) == 5
    if estado and not draft_ok:
        n1, n2 = len(estado["azul"].get("draft") or []), len(estado["vermelho"].get("draft") or [])
        avisos.append(f"Draft incompleto na fonte ({n1}+{n2} de 5+5): o pós-draft espera o draft fechar.")
    if camada and draft_ok:
        sig = jogo.get("siglas") or ("", "")
        ka = _draft_chaves(esporte, estado["azul"]["draft"], sig)
        kv = _draft_chaves(esporte, estado["vermelho"]["draft"], sig)
        agora = datetime.now(timezone.utc).replace(tzinfo=None)
        s, faltam = camada["estado"].sinais_no_app(ka, kv, agora)
        faltam = [f or "(jogador nao digitado)" for f in faltam]
        p_pos = CD.p_pos_draft(p_base, s, camada["coef"])
        linhas = (CD.detalhe_do_lado(camada["estado"], ka, agora, _rotulos(esporte, estado["azul"]["draft"], sig, heroi)),
                  CD.detalhe_do_lado(camada["estado"], kv, agora, _rotulos(esporte, estado["vermelho"]["draft"], sig, heroi)))
        if faltam:
            avisos.append("Jogador(es) fora da base (" + ", ".join(sorted(set(faltam))) + "): a prática no campeão "
                          "não entra; o pós-draft usa só a força recente dos campeões.")
    # ao vivo
    vivo = None
    if estado and modelos and float(estado.get("minuto") or 0) > 0:
        vivo = SL.p_ao_vivo(esporte, modelos, p_pos, estado, tier=_tier(esporte, jogo.get("liga") or ""),
                            n_mapa=int(jogo.get("mapa") or 1))

    nome_a, nome_b = az or az_raw, vm or vm_raw
    st.markdown(f"### {nome_a} (azul) x {nome_b} (vermelho)")
    for a in avisos:
        st.warning(a)
    cols = st.columns(3)
    for col, rot, p, sub in ((cols[0], "Pré-jogo", p_pre, origem or "—"),
                             (cols[1], "Pós-draft", p_pos if (camada and draft_ok) else None,
                              ("draft digitado" if (estado or {}).get("fonte") == "digitado" else "draft da fonte ao vivo")
                              if draft_ok else "esperando o draft"),
                             (cols[2], "Ao vivo", None if vivo is None else vivo["p"],
                              "—" if vivo is None else f"min {estado['minuto']:.0f} · {vivo['como']}")):
        col.metric(rot, _pct(p), help=sub)
        if p is not None:
            col.caption(f"{nome_a}: odd justa {_odd(p)} · {nome_b}: {_odd(1 - p)}")
        col.caption(sub)

    if estado:
        a, b = estado["azul"], estado["vermelho"]
        fonte = {"lolesports": "lolesports", "opendota": "OpenDota", "digitado": "digitado"}.get(estado.get("fonte"), "")
        extra = ""
        if esporte == "lol" and estado.get("fonte") == "lolesports":
            extra = (f" · torres {a.get('torres', 0)}–{b.get('torres', 0)} · dragões {len(a.get('dragoes') or [])}–"
                     f"{len(b.get('dragoes') or [])} · barões {a.get('baroes', 0)}–{b.get('baroes', 0)}")
        atraso = f" · atualizado há {jogo['atualizado_ha_s']} s" if jogo.get("atualizado_ha_s") is not None else ""
        st.caption(f"Estado ({fonte}{atraso}): minuto {estado['minuto']:.1f} · ouro {a.get('ouro', 0) - b.get('ouro', 0):+,.0f} · "
                   f"abates {a.get('abates', 0):.0f}–{b.get('abates', 0):.0f}{extra}".replace(",", "."))
    if linhas[0]:
        st.markdown("**Draft** — jogos de cada jogador naquele campeão (vitórias) e a força recente do campeão")
        c = st.columns(2)
        for col, nome, ls in ((c[0], nome_a, linhas[0]), (c[1], nome_b, linhas[1])):
            col.markdown(f"**{nome}**")
            col.dataframe([{"Jogador": x["jogador"] + ("" if x["na_base"] else " (fora da base)"),
                            "Campeão": x["campeao"],
                            "Jogos nele": x["jogos_no_campeao"] if x["na_base"] else "—",
                            "Vitórias": x["vitorias_no_campeao"] if x["na_base"] else "—",
                            "Campeão (recente)": f"{100 * x['aproveitamento_campeao']:.0f}%"} for x in ls],
                          hide_index=True, use_container_width=True)
    _render_polymarket(st, raiz=raiz, esporte=esporte, jogo=jogo, nome_a=nome_a, nome_b=nome_b,
                       p_mapa=(vivo["p"] if vivo else (p_pos if (camada and draft_ok) else p_pre)), resolver=resolver)


def _render_polymarket(st: Any, *, raiz: Path, esporte: str, jogo, nome_a, nome_b, p_mapa, resolver) -> None:
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
def _nomes_jogadores(raiz: Path) -> Dict[str, str]:
    import json
    try:
        return dict(json.loads((Path(raiz) / "nomes_jogadores_dota.json").read_text(encoding="utf-8")).get("nomes") or {})
    except (OSError, ValueError):
        return {}


def _rot_jogador(conta: str, nomes: Mapping[str, str]) -> str:
    if not conta:
        return "fora da base"
    return nomes.get(str(conta)) or f"conta {conta} (sem nome público)"


def render_draft_colado(st: Any, *, raiz: Path, esporte: str, times: List[str], camada: Optional[Dict[str, Any]],
                        heroi: Callable[[str], str]) -> None:
    import json
    from app_core import draft_colado_core as DC
    from app_core.live_between_maps_core import resolve_p_base
    st.markdown("#### Colar o draft — quanto a odd mexe e por quê")
    if camada is None:
        st.warning("Sem a camada de draft (camada_draft_dota.json): não dá para analisar o draft.")
        return
    ss = st.session_state
    lista = [""] + sorted(times)
    ini_a, ini_b = str(ss.get("teamA") or ""), str(ss.get("teamB") or "")
    c1, c2, c3 = st.columns([3, 3, 2])
    t1 = c1.selectbox("Time 1", lista, index=lista.index(ini_a) if ini_a in lista else 0, key="dc_t1")
    t2 = c2.selectbox("Time 2", lista, index=lista.index(ini_b) if ini_b in lista else 0, key="dc_t2")
    rad = c3.radio("Radiant", ["Time 1", "Time 2"], horizontal=True, key="dc_rad")
    texto = st.text_area("Heróis: uma linha por time (primeiro o Time 1), separados por vírgula — ou os 10 numa "
                         "linha só. Aceita apelidos (FV, AM, NP, WR...).", key="dc_texto", height=80,
                         placeholder="Faceless Void, Puck, Mars, Tusk, Rubick\nAnti-Mage, Invoker, NP, Axe, Centaur")
    ult = str(camada.get("ultimo_jogo") or "")[:10]
    if ult:
        st.caption(f"Base do draft até {ult[8:10]}/{ult[5:7]}/{ult[:4]} (jogos depois disso ainda não contam).")
    if not (t1 and t2) or t1 == t2:
        st.info("Escolha os dois times.")
        return
    if not str(texto or "").strip():
        return
    try:
        herois = json.loads((Path(raiz) / "herois_dota.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        herois = {}
    lados, _nao, aviso = DC.ler_draft(texto, DC.catalogo_dota(herois))
    if lados is None:
        st.warning(aviso)
        return
    estado, nomes = camada["estado"], _nomes_jogadores(raiz)
    elencos = camada.get("elencos") or {}
    escolhidos: List[List[str]] = []
    with st.expander("Quem joga cada herói (escolhido pelo histórico — corrija se estiver errado)", expanded=False):
        cols = st.columns(2)
        for col, time_, hs in ((cols[0], t1, lados[0]), (cols[1], t2, lados[1])):
            el = elencos.get(time_) or {}
            contas = [str(x) for x in (el.get("jogadores") or [])]
            visto = str(el.get("visto_em") or "")
            col.markdown(f"**{time_}**" + (f" — elenco do jogo de {visto[8:10]}/{visto[5:7]}" if visto else ""))
            if len(contas) < 5:
                col.warning(f"{time_} não tem elenco na base: a parte de cada jogador fica de fora "
                            "(só a força dos heróis conta).")
            quem = DC.atribuir(contas, hs, estado)
            opcoes = contas + [""]
            lado = []
            for h, q in zip(hs, quem):
                v = col.selectbox(heroi(h), opcoes, index=opcoes.index(q) if q in opcoes else len(opcoes) - 1,
                                  format_func=lambda c: _rot_jogador(c, nomes), key=f"dc_j_{time_}_{h}")
                lado.append(v)
            escolhidos.append(lado)
    p1, origem = resolve_p_base(session_state=ss, team_a=t1, team_b=t2,
                                artifact_path=str(Path(raiz) / "ml_artifact.json"))
    if p1 is None:
        st.warning(f"Sem preço pré-jogo para {t1} x {t2} ({origem}): a análise parte de 50%.")
        p1 = 0.5
    p1 = float(p1)
    t1_rad = rad == "Time 1"
    l1, l2 = list(zip(escolhidos[0], lados[0])), list(zip(escolhidos[1], lados[1]))
    azul, verm = (l1, l2) if t1_rad else (l2, l1)
    agora = datetime.now(timezone.utc).replace(tzinfo=None)
    r = DC.explicar(estado, camada["coef"], azul, verm, agora, p1 if t1_rad else 1 - p1)
    pos1 = r["p_pos"] if t1_rad else 1 - r["p_pos"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Pré-jogo (mapa)", _pct(p1), help=origem)
    m1.caption(f"{t1} {_odd(p1)} · {t2} {_odd(1 - p1)}")
    m2.metric("Pós-draft (mapa)", _pct(pos1), delta=f"{100 * (pos1 - p1):+.1f} pp para {t1}")
    m2.caption(f"{t1} {_odd(pos1)} · {t2} {_odd(1 - pos1)}")
    m3.metric("Lado Radiant", f"{r['pp_lado_azul']:+.1f} pp",
              help="Vantagem medida do lado Radiant (já somada no pós-draft).")
    m3.caption(f"Radiant: {t1 if t1_rad else t2}")
    if r["faltam"]:
        st.warning("Jogador fora da base (" + ", ".join(_rot_jogador(c, nomes) for c in sorted(set(r["faltam"])))
                   + "): a parte dos jogadores sai (regra da medição); contam só os heróis.")
    st.markdown("**Por quê** — cada jogador naquele herói comparado com a média dos 10 deste jogo "
                "(positivo = puxa a favor do time dele). Pós-draft = pré-jogo + lado + diferença entre os dois times.")
    cols = st.columns(2)
    por_lado: Dict[str, List[Dict[str, Any]]] = {"azul": [], "vermelho": []}
    for x in r["partes"]:
        por_lado[x["lado"]].append(x)
    for col, time_, ld in ((cols[0], t1, "azul" if t1_rad else "vermelho"),
                           (cols[1], t2, "vermelho" if t1_rad else "azul")):
        tot = sum(x["pp"] for x in por_lado[ld])
        col.markdown(f"**{time_}** — no duelo do draft: {tot:+.1f} pp")
        for x in sorted(por_lado[ld], key=lambda z: -z["pp"]):
            nome = _rot_jogador(x["conta"], nomes) if x["conta"] else "jogador não definido"
            if not r["usa_jogador"]:
                jog = "parte do jogador fora"
            elif x["jogos"] == 0:
                jog = f"nunca jogou de {heroi(x['heroi'])} na base ({x['pp_jogador']:+.1f})"
            else:
                jog = (f"ele no herói: {x['vitorias']}–{x['jogos'] - x['vitorias']} em {x['jogos']} jogos "
                       f"({x['pp_jogador']:+.1f})")
            col.markdown(f"**{x['pp']:+.1f} pp** · {nome} de {heroi(x['heroi'])}")
            col.caption(f"herói {100 * x['wr_heroi']:.0f}% nos jogos recentes ({x['pp_heroi']:+.1f}) · {jog}")
    st.caption("Medido às cegas em 2026: só a força recente do herói e o jogador naquele herói mexem no preço. "
               "Counter, dupla de heróis, time no herói, último pick, ban no herói do jogador e estilo de jogo "
               "foram testados e não melhoraram a previsão — por isso não aparecem como motivo.")
