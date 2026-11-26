# -*- coding: utf-8 -*-
"""MESA DO SUPER-LIVE (24/11/2026): varios jogos ao mesmo tempo, cada um na sua aba.

Pedidos dele:
  - "vao ter momentos que vao estar rolando 2-3-4-5 ate 10 jogos, ao mesmo tempo, seria legal que eu
    conseguisse analisar de forma unitaria cada uma deles" -> uma aba por jogo (ate 10) + "Todos os jogos";
  - "pegar os jogos que estao ocorrendo naquela hora, e deixar em aberto caso eu queira pegar um que o
    modelo ainda nao pegou" -> lista "Rolando agora" (agenda + fontes ao vivo) e "+ outro jogo";
  - o draft entra pelo que ELE cola: Dota = captura de tela da cyberscore (app_core/draft_imagem_core);
    LoL = texto com os 10 campeoes (draft_colado_core.ler_draft_lol). Nada de puxar draft sozinho.
  - a tela bonita aprovada em 24/11 (app_core/draft_visual_core), so com numero MEDIDO;
  - SEM Polymarket/"valor" na aba: a decisao de entrar e dele.
REGRA ABSOLUTA: nada some calado -- time da agenda que nao casa com a base aparece com o motivo.
Arquivo identico nos repositorios (LoL e Dota).
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from app_core import camada_draft_core as CD
from app_core import draft_colado_core as DC
from app_core import draft_visual_core as DV

MAX_JOGOS = 10
JANELA_ANTES_H, JANELA_DEPOIS_MIN = 4.0, 30.0
LADOS = {"dota": ("Radiant", "Dire"), "lol": ("Blue", "Red")}

# Colar a imagem com Ctrl+V (o Streamlit so aceita arrastar/escolher o arquivo): o script escuta o
# "colar" na pagina e entrega a imagem ao campo de captura VISIVEL (a aba aberta).
_JS_COLAR = """
<script>
(function(){
  const doc = window.parent.document;
  if (doc.__gabColar) return; doc.__gabColar = true;
  doc.addEventListener('paste', function(e){
    const fs = Array.from((e.clipboardData||{}).files||[]).filter(f => f.type.startsWith('image/'));
    if (!fs.length) return;
    const ins = Array.from(doc.querySelectorAll('[data-testid="stFileUploader"] input[type=file]'))
      .filter(i => i.closest('[data-testid="stFileUploader"]').offsetParent !== null);
    if (!ins.length) return;
    const dt = new DataTransfer(); dt.items.add(new File([fs[0]], 'captura.png', {type: fs[0].type}));
    ins[0].files = dt.files; ins[0].dispatchEvent(new Event('change', {bubbles: true}));
    e.preventDefault();
  });
})();
</script>
"""


# ------------------------------------------------------------------------------------------------
# Rolando agora
# ------------------------------------------------------------------------------------------------
def _dt(s: Any) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00").replace(" ", "T").replace("+00T", "+00"))
    except (TypeError, ValueError):
        try:
            d = datetime.strptime(str(s)[:19], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def rolando_agora(agenda: Mapping[str, Any], ao_vivo: List[Mapping[str, Any]], resolver: Callable[[str], str],
                  agora: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Pura. Jogos da agenda que comecaram nas ultimas 4 h (ou comecam em 30 min) + os das fontes ao
    vivo, sem repetir. Cada um: {a_bruto, b_bruto, a, b, liga, bo, inicio, fonte, motivo}."""
    agora = agora or datetime.now(timezone.utc)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _r(n):
        try:
            return resolver(n) or ""
        except Exception:  # noqa: BLE001
            return ""

    def _add(a_b, b_b, liga, bo, ini, fonte):
        a, b = _r(a_b), _r(b_b)
        chave = tuple(sorted([(a or a_b).lower(), (b or b_b).lower()]))
        if chave in out:
            out[chave]["fonte"] += " + " + fonte if fonte not in out[chave]["fonte"] else ""
            return
        faltam = [x for x, r in ((a_b, a), (b_b, b)) if not r]
        out[chave] = {"a_bruto": a_b, "b_bruto": b_b, "a": a, "b": b, "liga": liga or "", "bo": bo,
                      "inicio": ini, "fonte": fonte,
                      "motivo": ("" if not faltam else
                                 "não reconheci " + " e ".join(faltam) + " na base (time novo ou apelido não ensinado)")}

    for j in (agenda or {}).get("jogos") or []:
        ini = _dt(j.get("inicio_utc"))
        if ini is None:
            continue
        if agora - timedelta(hours=JANELA_ANTES_H) <= ini <= agora + timedelta(minutes=JANELA_DEPOIS_MIN):
            _add(str(j.get("teamA_raw") or ""), str(j.get("teamB_raw") or ""), j.get("league"), j.get("bo"), ini, "agenda")
    for j in ao_vivo or []:
        _add(str(j.get("azul") or ""), str(j.get("vermelho") or ""), j.get("liga"), j.get("bo"), None, "ao vivo")
    return sorted(out.values(), key=lambda x: (x["inicio"] is None, x["inicio"] or agora))


def _ao_vivo_cache(st: Any, esporte: str) -> Tuple[List[Dict[str, Any]], str]:
    k = f"_slm_vivo_{esporte}"
    c = st.session_state.get(k)
    if isinstance(c, tuple) and time.time() - c[0] < 60:
        return c[1], c[2]
    from app_core import super_live_core as SL
    try:
        if esporte == "dota":
            from app_core import valve_ao_vivo_core as VA
            ch = VA.chave(st)
            if ch:   # a lista ao vivo da propria Valve (mais confiavel que a da OpenDota), com lado certo
                from app_core import perfil_comp_core as PC
                raiz_ = Path(__file__).resolve().parents[1]
                from app_core import camada_draft_core as CD_
                el = (CD_.carregar(raiz_, "dota") or {}).get("elencos") or {}
                js = [{"azul": j["radiant"] or VA.time_pelos_jogadores(j["contas_lado"][0], el),
                       "vermelho": j["dire"] or VA.time_pelos_jogadores(j["contas_lado"][1], el), "liga": "", "bo": None}
                      for j in VA.buscar(ch, PC.carregar(raiz_, "dota"))]
                js = [j for j in js if j["azul"] and j["vermelho"]]
            else:
                js = SL.dota_ao_vivo()
        else:
            js = SL.lol_ao_vivo()
        mot = ""
    except Exception as e:  # noqa: BLE001
        js, mot = [], f"fonte ao vivo fora ({type(e).__name__})"
    st.session_state[k] = (time.time(), js, mot)
    return js, mot


# ------------------------------------------------------------------------------------------------
# Mesa (estado na sessao)
# ------------------------------------------------------------------------------------------------
def _mesa(ss: Any, esporte: str) -> List[Dict[str, Any]]:
    return ss.setdefault(f"slm_{esporte}", [])


def abrir(ss: Any, esporte: str, a: str = "", b: str = "", bo: Any = None, liga: str = "") -> Optional[int]:
    """Abre um jogo na mesa (ou devolve o que ja esta aberto com os mesmos times). None se a mesa esta cheia."""
    mesa = _mesa(ss, esporte)
    if a and b:
        for j in mesa:
            if {j["a"], j["b"]} == {a, b}:
                return j["id"]
    if len(mesa) >= MAX_JOGOS:
        return None
    n = int(ss.get(f"slm_{esporte}_n", 0)) + 1
    ss[f"slm_{esporte}_n"] = n
    # rotulo FIXO da aba: se mudasse a cada escolha de time, o Streamlit voltaria para a 1a aba no meio do uso
    mesa.append({"id": n, "a": a, "b": b, "bo": bo, "liga": liga,
                 "rotulo": f"Jogo {n}" + (f" · {a} x {b}" if a and b else "")})
    return n


def fechar(ss: Any, esporte: str, jid: int) -> None:
    ss[f"slm_{esporte}"] = [j for j in _mesa(ss, esporte) if j["id"] != jid]
    for k in [k for k in list(ss.keys()) if str(k).startswith(f"slm_{esporte}_{jid}_")]:
        del ss[k]
    ss.pop(f"slm_{esporte}_res_{jid}", None)


def _rotulo(j: Mapping[str, Any]) -> str:
    return str(j.get("rotulo") or f"Jogo {j['id']}")


# ------------------------------------------------------------------------------------------------
# Tela
# ------------------------------------------------------------------------------------------------
def render_mesa(st: Any, *, raiz: Path, esporte: str, times: List[str], resolver: Callable[[str], str],
                camada: Optional[Dict[str, Any]], modelos: Any) -> None:
    import streamlit.components.v1 as components
    ss = st.session_state
    raiz = Path(raiz)
    components.html(_JS_COLAR, height=0)
    if camada is None:
        st.warning(f"Sem a camada de draft (camada_draft_{esporte}.json): o pós-draft fica igual ao pré-jogo.")

    # rolando agora
    from app_core.gaveta_jogos_core import _agenda
    vivo, mot = _ao_vivo_cache(st, esporte)
    agora_lista = rolando_agora(_agenda(), vivo, resolver)
    with st.expander(f"Rolando agora ({len(agora_lista)})", expanded=not _mesa(ss, esporte)):
        if mot:
            st.caption(f"Fonte ao vivo: {mot}; a lista usa só a agenda.")
        if not agora_lista:
            st.caption("Nenhum jogo da agenda começou nas últimas 4 h nem começa nos próximos 30 min.")
        for i, g in enumerate(agora_lista):
            c1, c2 = st.columns([5, 1])
            hora = "" if g["inicio"] is None else (g["inicio"] - timedelta(hours=3)).strftime("%H:%M") + " · "
            c1.markdown(f"**{g['a'] or g['a_bruto']} x {g['b'] or g['b_bruto']}** — {hora}{g['liga']}"
                        + (f" · MD{g['bo']}" if g.get("bo") else "") + f" · _{g['fonte']}_")
            if g["motivo"]:
                c1.caption("⚠ " + g["motivo"] + ". Abra pelo “+ outro jogo” e escolha o time certo.")
            if c2.button("Abrir", key=f"slm_{esporte}_abrir_{i}", disabled=bool(g["motivo"])):
                if abrir(ss, esporte, g["a"], g["b"], g.get("bo"), g["liga"]) is None:
                    st.warning(f"A mesa já tem {MAX_JOGOS} jogos; feche um para abrir outro.")
                st.rerun()
    if st.button(r"\+ Outro jogo", key=f"slm_{esporte}_novo"):
        if abrir(ss, esporte) is None:
            st.warning(f"A mesa já tem {MAX_JOGOS} jogos; feche um para abrir outro.")
        st.rerun()

    mesa = _mesa(ss, esporte)
    if not mesa:
        st.info("Abra um jogo da lista “Rolando agora” ou pelo “+ outro jogo”.")
        return
    # o jogo aberto por ultimo vem PRIMEIRO (abre selecionado); "Todos os jogos" no fim
    ordem = list(reversed(mesa))
    abas = st.tabs([_rotulo(j) for j in ordem] + ["Todos os jogos"])
    for aba, j in zip(abas[:-1], ordem):
        with aba:
            try:
                _render_jogo(st, raiz=raiz, esporte=esporte, jogo=j, times=times, camada=camada, modelos=modelos,
                             resolver=resolver)
            except Exception as e:  # noqa: BLE001 -- um jogo quebrado nao derruba os outros
                st.error(f"Este jogo não abriu: {type(e).__name__}: {e}")
    with abas[-1]:
        _render_todos(st, esporte, mesa)


def _render_todos(st: Any, esporte: str, mesa: List[Dict[str, Any]]) -> None:
    linhas = []
    for j in mesa:
        r = st.session_state.get(f"slm_{esporte}_res_{j['id']}") or {}
        linhas.append({"Jogo": (f"{j['a']} x {j['b']}" if j.get("a") and j.get("b") else _rotulo(j)),
                       "Pré-jogo (T1)": f"{100 * r['pre']:.0f}% · {DV._odd(r['pre'])}" if r.get("pre") is not None else "—",
                       "Pós-draft (T1)": f"{100 * r['pos']:.0f}% · {DV._odd(r['pos'])}" if r.get("pos") is not None else "esperando o draft",
                       "Draft": r.get("draft", "—"),
                       "Ao vivo (T1)": f"{100 * r['vivo']:.0f}%" if r.get("vivo") is not None else "—"})
    st.dataframe(linhas, hide_index=True, use_container_width=True)
    st.caption("T1 = o Time 1 de cada aba. Clique na aba do jogo para ver o draft e os motivos.")


def _herois_dota(raiz: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    def _j(n):
        try:
            return json.loads((raiz / n).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return _j("herois_dota.json"), _j("herois_dota_img.json")


def _campeoes_lol(raiz: Path) -> Tuple[Dict[str, str], str]:
    try:
        d = json.loads((raiz / "campeoes_lol.json").read_text(encoding="utf-8"))
        return dict(d.get("campeoes") or {}), str(d.get("versao") or "")
    except (OSError, ValueError):
        return {}, ""


def _nomes_dota(raiz: Path) -> Dict[str, str]:
    try:
        return dict(json.loads((raiz / "nomes_jogadores_dota.json").read_text(encoding="utf-8")).get("nomes") or {})
    except (OSError, ValueError):
        return {}


def _ler_draft(st: Any, k: str, raiz: Path, esporte: str, valve: Optional[Dict[str, Any]] = None
               ) -> Tuple[Optional[Tuple[List[str], List[str]]],
                                                                     Tuple[List[str], List[str]], List[str]]:
    """-> (picks (T1, T2) na ordem pos 1..5, bans (esq, dir), avisos). Dota: captura ou texto; LoL: texto."""
    avisos: List[str] = []
    if esporte == "dota":
        nomes, _img = _herois_dota(raiz)
        up = st.file_uploader("Captura do draft (cyberscore): cole com Ctrl+V aqui na aba, arraste ou escolha o arquivo",
                              type=["png", "jpg", "jpeg", "webp"], key=f"{k}up")
        texto = st.text_input("…ou digite os 10 heróis, primeiro os 5 do Radiant e depois os 5 do Dire, na ordem "
                              "pos 1 a 5, separados por vírgula", key=f"{k}tx")
        if up is not None:
            from app_core import draft_imagem_core as RI
            refs = RI.carregar_refs(raiz)
            if refs is None:
                return None, ([], []), ["Faltam as referências dos heróis (herois_dota_ref.npz)."]
            cache_k = f"{k}rec_{up.file_id if hasattr(up, 'file_id') else up.name}"
            rec = st.session_state.get(cache_k)
            if rec is None:
                rec = RI.reconhecer(up.getvalue(), *refs)
                st.session_state[cache_k] = rec
            if rec.get("aviso"):
                avisos.append(rec["aviso"])
            if not rec.get("picks"):
                return None, ([], []), avisos
            pe, pd_ = rec["picks"]
            be, bd = rec["bans"]
            # conferencia: duvida aparece aberta; o resto fica a um clique
            op = sorted(nomes, key=lambda h: nomes[h])
            n_duv = len(rec.get("duvidas") or [])
            if n_duv:
                itens = []
                for lado, tipo, i in rec["duvidas"]:
                    x = ((pe if lado == "esquerda" else pd_) if tipo == "pick" else (be if lado == "esquerda" else bd))[i]
                    alt = " ou ".join(nomes.get(str(h), str(h)) for _s, h in x["alternativas"][:1])
                    itens.append(f"{tipo} {i + 1} da {lado}: **{nomes.get(str(x['id']), x['id'])}** (ou {alt}?)")
                st.warning("Confira antes de usar — reconheci com pouca folga: " + "; ".join(itens)
                           + ". Se estiver errado, corrija logo abaixo.")
            with st.expander(("⚠ Confira " + str(n_duv) + " herói(s) em dúvida") if n_duv else "Heróis reconhecidos (corrigir)",
                             expanded=bool(n_duv)):
                def corr(lista, rot):
                    out = []
                    cols = st.columns(max(1, len(lista)))
                    for i, (c, x) in enumerate(zip(cols, lista)):
                        hid = str(x["id"])
                        lab = f"{rot} {i + 1}" + (" ⚠" if x["duvida"] else "")
                        v = c.selectbox(lab, op, index=op.index(hid) if hid in op else 0,
                                        format_func=lambda h: nomes.get(h, h), key=f"{k}c_{rot}_{i}")
                        out.append(v)
                    return out
                ape = corr(pe, "Esq")
                apd = corr(pd_, "Dir")
                abe = corr(be, "Ban esq") if be else []
                abd = corr(bd, "Ban dir") if bd else []
            if len(ape) != 5 or len(apd) != 5:
                return None, (abe, abd), avisos
            return (ape, apd), (abe, abd), avisos
        if str(texto or "").strip():
            lados, _nao, aviso = DC.ler_draft(texto, DC.catalogo_dota(nomes))
            if lados is None:
                return None, ([], []), [aviso]
            return lados, ([], []), avisos
        if valve:   # sem captura nem texto: o draft do jogo ao vivo, pela Valve (lado e jogadores certos)
            st.session_state[f"{k}fonte"] = "valve"
            m = int(valve.get("duracao", 0) // 60)
            avisos.append(f"Draft preenchido pela Valve (jogo ao vivo{f', minuto {m}' if m else ', antes de começar'}). "
                          "Lado e jogadores vêm certos; as POSIÇÕES são estimadas"
                          + (" pelo patrimônio" if valve.get("duracao", 0) >= 480 else " (suportes por último)")
                          + " — cole a captura da cyberscore para as rotas exatas.")
            return valve["picks"], valve["bans"], avisos
        return None, ([], []), avisos
    camp, _v = _campeoes_lol(raiz)
    texto = st.text_area("Draft: os 10 campeões em sequência — top, jungle, mid, adc e suporte do lado Blue, depois do "
                         "Red (lixo no meio e nomes alternativos tudo bem)", key=f"{k}tx", height=70,
                         placeholder="renekton wukong ryze yunara lulu ambessa leesin orianna lucian milio")
    if not str(texto or "").strip():
        return None, ([], []), avisos
    lados, ign, aviso = DC.ler_draft_lol(texto, DC.catalogo_lol(camp))
    if lados is None:
        return None, ([], []), [aviso]
    if ign:
        avisos.append("Ignorei: " + ", ".join(ign) + ".")
    return lados, ([], []), avisos


MARGEM_LADO = 1.5   # em soma de log(1 + jogos): abaixo disto o lado fica para ele marcar


def lado_provavel(estado: Any, elencos: Mapping[str, Any], t1: str, t2: str, esq: List[str],
                  dir_: List[str]) -> Tuple[Optional[bool], float]:
    """Pura (24/11/2026, achado dele: "voce errou o lado, a Angry Birds esta Dire"). Quem esta a
    ESQUERDA (Radiant/Blue)? O elenco que tem historico com os herois da esquerda. Compara
    'T1 na esquerda' x 'T1 na direita' pela soma de log(1 + jogos de cada jogador no heroi dele).
    -> (True se T1 e a esquerda / False / None se nao da para saber com folga, margem)."""
    def enc(tm, hs):
        contas = [str(x) for x in ((elencos.get(tm) or {}).get("jogadores") or [])]
        if len(contas) < 5:
            return None
        ks = [CD.chave_campeao(h) for h in hs]
        quem = DC.atribuir(contas, ks, estado)
        return sum(math.log1p(estado.jogos_no_campeao(CD.chave_jogador(c), k)[1]) for c, k in zip(quem, ks) if c)
    v = [enc(t1, esq), enc(t2, dir_), enc(t1, dir_), enc(t2, esq)]
    if any(x is None for x in v):
        return None, 0.0
    margem = (v[0] + v[1]) - (v[2] + v[3])
    return (None if abs(margem) < MARGEM_LADO else margem > 0), margem


def _posicoes(raiz: Path, esporte: str) -> Optional[Dict[str, Any]]:
    """{conta: {posicao: jogos}} do Dota (posicoes_jogadores_dota.json), lido uma vez por versao do arquivo."""
    if esporte != "dota":
        return None
    arq = Path(raiz) / "posicoes_jogadores_dota.json"
    try:
        mt = arq.stat().st_mtime
    except OSError:
        return None
    if _POS_CACHE.get("mt") != mt:
        try:
            _POS_CACHE.update(mt=mt, d=json.loads(arq.read_text(encoding="utf-8")).get("contas") or {})
        except (OSError, ValueError):
            return None
    return _POS_CACHE.get("d")


_POS_CACHE: Dict[str, Any] = {}


def _valve_do_jogo(st: Any, raiz: Path, esporte: str, t1: str, t2: str,
                   resolver: Callable[[str], str]) -> Optional[Dict[str, Any]]:
    """O jogo ao vivo destes dois times pela Valve (cache de 20 s na sessao), ou None -- dizendo o porque."""
    from app_core import valve_ao_vivo_core as VA
    ch = VA.chave(st)
    if not ch:
        st.caption("Draft ao vivo pela Valve: desligado (falta a chave STEAM_API_KEY nos Secrets do app).")
        return None
    c = st.session_state.get("_slm_valve")
    if not (isinstance(c, tuple) and time.time() - c[0] < 20):
        try:
            from app_core import perfil_comp_core as PC
            c = (time.time(), VA.buscar(ch, PC.carregar(raiz, "dota")), "")
        except Exception as e:  # noqa: BLE001
            c = (time.time(), [], f"a Valve não respondeu ({type(e).__name__})")
        st.session_state["_slm_valve"] = c
    if c[2]:
        st.caption("Draft ao vivo pela Valve: " + c[2] + ".")
        return None
    from app_core import camada_draft_core as CD_
    cam = CD_.carregar(raiz, "dota") or {}
    vj = VA.achar(c[1], t1, t2, resolver, elencos=cam.get("elencos") or {})
    if vj is None:
        st.caption(f"Draft ao vivo pela Valve: {t1} x {t2} não está entre os {len(c[1])} jogos de liga ao vivo agora.")
    return vj


def _render_jogo(st: Any, *, raiz: Path, esporte: str, jogo: Dict[str, Any], times: List[str],
                 camada: Optional[Dict[str, Any]], modelos: Any, resolver: Callable[[str], str]) -> None:
    import streamlit.components.v1 as components
    from app_core.live_between_maps_core import resolve_p_base
    ss = st.session_state
    jid = jogo["id"]
    k = f"slm_{esporte}_{jid}_"
    lado1, lado2 = LADOS[esporte]
    lista = [""] + sorted(times)
    c1, c2, c4 = st.columns([3, 3, 1])
    t1 = c1.selectbox("Time 1", lista, index=lista.index(jogo["a"]) if jogo["a"] in lista else 0, key=f"{k}t1")
    t2 = c2.selectbox("Time 2", lista, index=lista.index(jogo["b"]) if jogo["b"] in lista else 0, key=f"{k}t2")
    if c4.button("Fechar", key=f"{k}fechar"):
        fechar(ss, esporte, jid)
        st.rerun()
    jogo["a"], jogo["b"] = t1, t2
    if not (t1 and t2) or t1 == t2:
        st.info("Escolha os dois times.")
        return
    try:
        from app_core.confianca_base_core import texto_ultimo_jogo
        st.caption(texto_ultimo_jogo(raiz, esporte, [t1, t2]))
    except Exception as e:  # noqa: BLE001
        st.caption(f"Último jogo de cada time: não abriu ({type(e).__name__})")
    vj = _valve_do_jogo(st, raiz, esporte, t1, t2, resolver) if esporte == "dota" else None
    ss.pop(f"{k}fonte", None)
    picks, bans, avisos = _ler_draft(st, k, raiz, esporte, valve=vj)
    da_valve = ss.get(f"{k}fonte") == "valve"
    for a in avisos:
        st.warning(a)
    p1, origem = resolve_p_base(session_state=ss, team_a=t1, team_b=t2, artifact_path=str(raiz / "ml_artifact.json"))
    if p1 is None:
        st.warning(f"Sem preço pré-jogo para {t1} x {t2} ({origem}): a análise parte de 50%.")
        p1 = 0.5
    p1 = float(p1)
    res = {"pre": p1, "pos": None, "draft": "—", "vivo": None}
    ss[f"slm_{esporte}_res_{jid}"] = res
    if picks is None:
        st.metric(f"Pré-jogo (mapa) — {t1}", f"{100 * p1:.1f}%", help=origem)
        st.caption(f"{t1} {DV._odd(p1)} · {t2} {DV._odd(1 - p1)} — cole o draft para ver o pós-draft.")
        return
    if camada is None:
        return
    # REGRA DE LADO: os 5 primeiros (a esquerda da captura/do texto) sao SEMPRE Radiant/Blue. Quem esta de
    # qual lado: detectado pelo historico do elenco nos herois; sem folga, ELE marca (nunca chute calado).
    kl, km = f"{k}esq", f"{k}esq_manual"
    if da_valve:      # a Valve diz quem e Radiant: sem detectar, sem chutar
        det, margem = (vj["radiant_base"] == t1), 99.0
    else:
        det, margem = lado_provavel(camada["estado"], camada.get("elencos") or {}, t1, t2, picks[0], picks[1])
    if not ss.get(km) and det is not None:
        ss[kl] = t1 if det else t2
    if kl in ss and ss[kl] not in (t1, t2):
        del ss[kl]
    extra = {} if kl in ss else {"index": None}     # sem valor ainda: nada marcado (nunca chute)
    esq = st.radio(f"Quem está à esquerda ({lado1})?", [t1, t2], horizontal=True, key=kl,
                   on_change=lambda: ss.__setitem__(km, True), **extra)
    if not ss.get(km) and det is not None:
        st.caption(f"Lado {'informado pela Valve' if da_valve else 'detectado pelo histórico dos jogadores nos heróis'} "
                   f"({esq} à esquerda). Se estiver errado, troque acima.")
    if esq is None:
        st.warning(f"Não dá para saber o lado pelos heróis com segurança: marque acima quem está à esquerda ({lado1}).")
        return
    t1_azul = esq == t1
    t_az, t_vm = (t1, t2) if t1_azul else (t2, t1)
    hs_az, hs_vm = picks[0], picks[1]
    # quem joga cada heroi (elenco do ultimo jogo de cada time na base)
    estado, elencos = camada["estado"], camada.get("elencos") or {}
    nomes_j = _nomes_dota(raiz) if esporte == "dota" else {}
    rot_j = (lambda c: nomes_j.get(str(c)) or (f"conta {c}" if c else "fora da base")) if esporte == "dota"         else (lambda c: str(c) or "fora da base")
    escolhidos = []
    with st.expander("Quem joga cada herói/campeão (pelo histórico — corrija se mudou o elenco)"):
        cols = st.columns(2)
        for col, tm, hs, ld in ((cols[0], t_az, hs_az, lado1), (cols[1], t_vm, hs_vm, lado2)):
            el = elencos.get(tm) or {}
            contas = [str(x) for x in (el.get("jogadores") or [])]
            col.markdown(f"**{tm}** ({ld})" + (f" — elenco de {str(el.get('visto_em'))[8:10]}/{str(el.get('visto_em'))[5:7]}"
                                               if el.get("visto_em") else " — sem elenco na base"))
            quem = DC.atribuir(contas, [CD.chave_campeao(h) for h in hs], estado, posicoes=_posicoes(raiz, esporte))
            if da_valve:   # a Valve diz quem joga cada heroi
                quem = [str(vj["contas"].get(str(h)) or q or "") for h, q in zip(hs, quem)]
                contas = contas + [c for c in quem if c and c not in contas]
            opcoes = contas + [""]
            escolhidos.append([col.selectbox(str(h), opcoes, index=opcoes.index(q) if q in opcoes else len(opcoes) - 1,
                                             format_func=rot_j, key=f"{k}j_{tm}_{h}") for h, q in zip(hs, quem)])
    azul = [(j, CD.chave_campeao(h)) for j, h in zip(escolhidos[0], hs_az)]
    verm = [(j, CD.chave_campeao(h)) for j, h in zip(escolhidos[1], hs_vm)]
    agora = datetime.now(timezone.utc).replace(tzinfo=None)
    p_pre_az = p1 if t1_azul else 1 - p1
    r = DC.explicar(estado, camada["coef"], azul, verm, agora, p_pre_az)
    pos1 = r["p_pos"] if t1_azul else 1 - r["p_pos"]
    res.update(pos=pos1)
    if r["faltam"]:
        st.caption("Jogador fora da base (" + ", ".join(rot_j(c) for c in sorted(set(r["faltam"])))
                   + "): a parte dos jogadores sai (regra da medição); contam só os heróis.")
    if esporte == "dota":
        nomes_h, img_h = _herois_dota(raiz)
        nome_h = lambda h: nomes_h.get(str(h), str(h))  # noqa: E731
        img = lambda h: DV.IMG_DOTA.format(img_h.get(str(h), ""))  # noqa: E731
    else:
        camp, ver = _campeoes_lol(raiz)
        nome_h = lambda h: camp.get(str(h), str(h))  # noqa: E731
        img = lambda h: DV.IMG_LOL.format(ver, h)  # noqa: E731
    partes = {"azul": [x for x in r["partes"] if x["lado"] == "azul"],
              "vermelho": [x for x in r["partes"] if x["lado"] == "vermelho"]}

    def jogs(ps, hs):
        out = []
        for x, h in zip(ps, hs):
            det = (f"ele: {x['vitorias']}–{x['jogos'] - x['vitorias']} de {nome_h(h)} · " if x["jogos"]
                   else f"nunca jogou de {nome_h(h)} na base · ") + f"campeão {100 * x['wr_heroi']:.0f}% recente"
            if esporte == "dota":
                det = det.replace("campeão", "herói")
            out.append({"nome": rot_j(x["conta"]) if x["conta"] else "jogador não definido", "heroi": nome_h(h),
                        "img": img(h), "pp": x["pp"], "detalhe": det})
        return out

    je, jd = jogs(partes["azul"], hs_az), jogs(partes["vermelho"], hs_vm)
    bans_d = [[{"heroi": nome_h(h), "img": img(h)} for h in lst] for lst in bans]
    d = {"esq": {"time": t_az, "lado": lado1, "jogadores": je, "bans": bans_d[0] if bans_d else []},
         "dir": {"time": t_vm, "lado": lado2, "jogadores": jd, "bans": bans_d[1] if len(bans_d) > 1 else []},
         "p_pre": p_pre_az, "p_pos": r["p_pos"], "lado_pp": r["pp_lado_azul"]}
    from app_core import perfil_comp_core as PC
    pfs = PC.carregar(raiz, esporte)
    if pfs:
        d["perfil"] = {"esq": PC.descrever(esporte, hs_az, pfs), "dir": PC.descrever(esporte, hs_vm, pfs)}
    d["veredito"] = DV.veredito(t_az, t_vm, lado1, lado2, p_pre_az, r["p_pos"], je, jd, r["pp_lado_azul"])
    dlt = 100 * (r["p_pos"] - p_pre_az) - r["pp_lado_azul"]
    res["draft"] = "equilibrado" if abs(dlt) < 0.3 else f"{t_az if dlt > 0 else t_vm} +{abs(dlt):.1f} pp"
    rotas = DV.ROTAS_DOTA if esporte == "dota" else DV.ROTAS_LOL
    components.html(DV.html_draft(d, rotas), height=(920 if esporte == "dota" else 860), scrolling=True)
    # ao vivo (opcional, digitado): o estado do mapa por cima do pos-draft
    with st.expander("Ao vivo (digite o minuto e o placar do mapa)"):
        cs = st.columns(4)
        minuto = cs[0].number_input("Minuto", 0.0, 90.0, 0.0, 1.0, key=f"{k}min")
        ouro = cs[1].number_input(f"Ouro {t1} − {t2}", -60000.0, 60000.0, 0.0, 500.0, key=f"{k}ouro")
        k1 = cs[2].number_input(f"Abates {t1}", 0, 150, 0, key=f"{k}k1")
        k2 = cs[3].number_input(f"Abates {t2}", 0, 150, 0, key=f"{k}k2")
        if minuto > 0 and modelos:
            from app_core import super_live_core as SL
            og = ouro if t1_azul else -ouro
            ka, kv = (k1, k2) if t1_azul else (k2, k1)
            est = {"fonte": "digitado", "minuto": float(minuto), "situacao": "",
                   "azul": {"draft": [], "ouro": max(og, 0.0), "abates": float(ka), "mortes": float(kv)},
                   "vermelho": {"draft": [], "ouro": max(-og, 0.0), "abates": float(kv), "mortes": float(ka)}}
            from app_core.super_live_tab_core import _tier
            v = SL.p_ao_vivo(esporte, modelos, r["p_pos"], est, tier=_tier(esporte, jogo.get("liga") or ""), n_mapa=1)
            if v:
                p_v1 = v["p"] if t1_azul else 1 - v["p"]
                res["vivo"] = p_v1
                st.metric(f"Ao vivo — {t1}", f"{100 * p_v1:.1f}%", help=v.get("como", ""))
                st.caption(f"{t1} {DV._odd(p_v1)} · {t2} {DV._odd(1 - p_v1)}")
        elif minuto > 0:
            st.caption("Os modelos ao vivo não estão carregados neste app.")
    # Polymarket/valor NAO aparece aqui (24/11/2026, ele: "nao quero ver isso, essa parte deixa comigo").

