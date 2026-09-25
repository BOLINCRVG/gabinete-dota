# -*- coding: utf-8 -*-
"""As tres tabelas no formato da bet365: Fair, Modelo e Pinnacle.

O PEDIDO (25/09/2026), com o print da bet365 ("Mapa 1 - Totais", Mais de /
Menos de):
    "eu quero o design igual eu te mandei... e eu clique dentro das linhas
     para que eu altere eles... a fair e padrao, nao altera, e se eu mudar a
     do modelo, ele vai mudar a da pinnacle tambem"

POR QUE UMA PAGINA PROPRIA DENTRO DA ABA. Com os componentes do Streamlit o
visual fica sempre "parecido", e clicar DENTRO da celula da linha nao existe.
Aqui a tela e HTML proprio: o desenho e o do print, e o clique responde no
navegador, na hora -- os numeros de TODAS as linhas clicaveis ja vem prontos
do app (`montar_linhas_justas`, campo `grade`). Nada vai a internet: nao ha
arquivo externo, fonte ou script de fora.

A LOGICA DO CLIQUE
  - FAIR: a linha justa do modelo e as odds dele. Fixa.
  - MODELO e PINNACLE andam JUNTAS: clicar na linha de qualquer uma abre as
    linhas disponiveis; a escolhida vale para as duas.
  - Comeca na linha PRINCIPAL da Pinnacle.
  - Na Pinnacle, o lado que paga mais que o modelo fica verde, com a vantagem.

DEPOIS DE VER NO SITE (25/09/2026): "so gostaria que fosse nas cores do
modelo mesmo, e que em cima disso tudo tenha a parte do vencedor, baseado no
BO que o calendario [da] por padrao". As cores sao as do app (ciano do
modelo, ambar da Pinnacle, paineis do global_css_core) e seguem o tema claro
ou escuro do navegador. O vencedor vem em cima: a SERIE no BO do confronto
aberto (o que o calendario escolheu) e o vencedor de cada mapa. O numero do
modelo e o MESMO do cartao da Visao Geral (vem pronto da aba).
"""
from __future__ import annotations

import json
from html import escape
from typing import Any, Dict, List, Mapping

_ROTULO_DO_TOTAL = {
    "kills": "Total de Kills", "towers": "Total de Torres", "dragons": "Total de Dragões",
    "barons": "Total de Barões", "inhibitors": "Total de Inibidores",
}


def _metrica_do_nome(mercado: str) -> str:
    m = str(mercado or "").lower()
    if "kills" in m:
        return "kills"
    for chave, pal in (("towers", "turret"), ("dragons", "dragon"), ("barons", "baron"),
                       ("inhibitors", "inhibitor")):
        if pal in m:
            return chave
    return ""


def linhas_da_tela(justas: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """As linhas de `montar_linhas_justas` no formato que a tela desenha."""
    fora: List[Dict[str, Any]] = []
    ordem_total = ["kills", "towers", "dragons", "barons", "inhibitors"]
    for x in justas:
        grade = [g for g in (x.get("grade") or []) if g.get("odd_modelo_a")]
        if not grade:
            continue
        forma = str(x.get("forma") or "")
        casa, fora_ = str(x.get("casa") or "Casa"), str(x.get("fora") or "Fora")
        if forma == "handicap":
            secao, rotulo, ordem = "Handicap de Kills", "", 0
        elif forma == "total_time":
            secao = "Kills por time"
            rotulo = casa if str(x.get("time") or "").lower() == "home" else fora_
            ordem = 0 if rotulo == casa else 1
        else:
            met = _metrica_do_nome(x.get("mercado"))
            secao = "Totais"
            rotulo = _ROTULO_DO_TOTAL.get(met, str(x.get("nome_pt") or x.get("mercado") or ""))
            ordem = ordem_total.index(met) if met in ordem_total else 9
        principal = next((g["linha"] for g in grade if g.get("principal")), None)
        fora.append({
            "mapa": int(x.get("periodo") or 0),
            "secao": secao, "rotulo": rotulo, "ordem": ordem,
            "tipo": "handicap" if forma == "handicap" else "total",
            "casa": casa, "fora": fora_,
            "com_vitoria": x.get("conta") == "com vitoria",
            "fair": {"linha": x.get("linha_modelo"), "oa": x.get("odd_modelo_a"),
                     "ob": x.get("odd_modelo_b")},
            "grade": [{k: g.get(k) for k in ("linha", "odd_modelo_a", "odd_modelo_b", "casa_a",
                                            "casa_b", "vantagem_a", "vantagem_b", "principal")}
                      for g in grade],
            "inicial": principal if principal is not None else x.get("linha_modelo"),
        })
    return fora


def vencedores_da_tela(linhas_do_jogo: List[Mapping[str, Any]],
                       achados_do_jogo: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Vencedor da serie (periodo 0) e de cada mapa: a odd da Pinnacle e, se o
    app calculou (so o confronto aberto), a do modelo e a vantagem.

    `linhas_do_jogo` sao as linhas cruas da casa; `achados_do_jogo` os de
    `montar_oportunidades` com metrica "vencedor" -- o numero do cartao.
    """
    por: Dict[int, Dict[str, Any]] = {}
    for l in linhas_do_jogo:
        if str(l.get("tipo") or "") != "moneyline" or str(l.get("unidade") or "").strip():
            continue
        lado = str(l.get("lado") or "").strip().lower()
        if lado not in ("home", "away"):
            continue
        try:
            per = int(l.get("periodo") or 0)
        except (TypeError, ValueError):
            continue
        r = por.setdefault(per, {"periodo": per})
        r["pin_a" if lado == "home" else "pin_b"] = l.get("odd")
    for a in achados_do_jogo:
        if str(a.get("metrica") or "") != "vencedor":
            continue
        m = str(a.get("mercado") or "").lower()
        if m.endswith("serie"):
            per = 0
        else:
            try:
                per = int(m.rsplit(" ", 1)[-1])
            except ValueError:
                continue
        r = por.setdefault(per, {"periodo": per})
        k = "a" if str(a.get("lado") or "").lower() == "home" else "b"
        r["modelo_" + k] = a.get("odd_justa")
        r["vant_" + k] = a.get("vantagem_pct")
    return [por[k] for k in sorted(por) if por[k].get("pin_a") and por[k].get("pin_b")]


def altura(linhas: List[Mapping[str, Any]]) -> int:
    """Altura da moldura, em pixels, para o mapa com mais linhas."""
    por_mapa: Dict[int, Dict[str, int]] = {}
    for l in linhas:
        por_mapa.setdefault(l["mapa"], {}).setdefault(l["secao"], 0)
        por_mapa[l["mapa"]][l["secao"]] += 1
    pior = 0
    for secoes in por_mapa.values():
        h = sum(44 + 34 + 46 * n for n in secoes.values()) + 12 * len(secoes)
        pior = max(pior, h)
    # + o vencedor da serie em cima e o do mapa (duas secoes de uma linha)
    return int(pior + 150 + 2 * (44 + 34 + 46 + 12))


def html_das_tabelas(linhas: List[Mapping[str, Any]], titulo_jogo: str = "",
                     vencedores: List[Mapping[str, Any]] = (), bo: Any = None,
                     casa: str = "", fora: str = "") -> str:
    dados = json.dumps(linhas, ensure_ascii=False).replace("</", "<\\/")
    venc = json.dumps({"linhas": list(vencedores), "bo": bo, "casa": casa, "fora": fora},
                      ensure_ascii=False).replace("</", "<\\/")
    from app_core.oportunidades_core import VANTAGEM_SUSPEITA_PCT
    return (_MODELO_HTML.replace("__DADOS__", dados).replace("__VENCEDOR__", venc)
            .replace("__SUSPEITA__", str(float(VANTAGEM_SUSPEITA_PCT)))
            .replace("__TITULO__", escape(titulo_jogo)))


_MODELO_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  :root { --texto:#e6e6e6; --fraco:#9aa0a6; --painel:rgba(255,255,255,0.035);
          --borda:rgba(255,255,255,0.12); --cab:rgba(255,255,255,0.06); --linha:rgba(255,255,255,0.07);
          --modelo:#2daac4; --pin:#d9a441; --boa:#3fb950; --ruim:#c9736a;
          --verde-fundo:rgba(63,185,80,0.14); --verde-borda:rgba(63,185,80,0.55); --menu:#1f2227; }
  @media (prefers-color-scheme: light) {
    :root { --texto:#1f2328; --fraco:#6a737d; --painel:rgba(0,0,0,0.025); --borda:rgba(0,0,0,0.12);
            --cab:rgba(0,0,0,0.045); --linha:rgba(0,0,0,0.07); --menu:#ffffff; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:transparent; color:var(--texto);
         font-family:"Source Sans Pro","Segoe UI",Arial,sans-serif; font-size:13.5px; }
  .topo { display:flex; gap:8px; align-items:center; padding:10px 2px 10px; flex-wrap:wrap; }
  .topo b { font-size:14.5px; margin-right:8px; }
  .mapa { background:var(--painel); color:var(--texto); border:1px solid var(--borda); border-radius:8px;
          padding:5px 14px; cursor:pointer; font-weight:600; font-family:inherit; }
  .mapa.ativo { background:rgba(45,170,196,0.16); border-color:var(--modelo); color:var(--modelo); }
  .colunas { display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap; }
  .coluna { flex:1 1 330px; min-width:300px; }
  .nome-col { font-weight:800; letter-spacing:0.07em; font-size:12px; padding:0 2px 6px;
              text-transform:uppercase; }
  .secao { background:var(--painel); border:1px solid var(--borda); border-top:3px solid var(--modelo);
           border-radius:12px; margin-bottom:12px; overflow:hidden; }
  .col-pinnacle .secao { border-top-color:var(--pin); }
  .secao h3 { margin:0; padding:10px 12px; font-size:13.5px; font-weight:700; }
  .cab { display:grid; background:var(--cab); font-weight:700; font-size:11.5px; opacity:0.85; }
  .cab div { padding:7px 6px; text-align:center; }
  .row { display:grid; border-top:1px solid var(--linha); align-items:stretch; }
  .row > div { padding:0 6px; display:flex; align-items:center; justify-content:center; min-height:45px; }
  .row .rot { justify-content:flex-start; padding-left:12px; }
  .g3 { grid-template-columns: 1.15fr 1fr 1fr; }
  .g2 { grid-template-columns: 1fr 1fr; }
  .cel { gap:7px; border-left:1px solid var(--linha); }
  .ln { color:var(--fraco); }
  .ln.clica { cursor:pointer; border-bottom:1px dashed var(--fraco); }
  .ln.clica:hover { color:var(--texto); border-bottom-color:var(--texto); }
  .odd { font-weight:800; }
  .col-fair .odd, .col-modelo .odd { color:var(--modelo); }
  .col-pinnacle .odd { color:var(--pin); }
  .cel.verde { background:var(--verde-fundo); box-shadow:inset 0 0 0 1px var(--verde-borda); }
  .v { font-size:11px; font-weight:700; }
  .v.b { color:var(--boa); } .v.r { color:var(--ruim); }
  .nada { color:var(--fraco); font-size:11.5px; }
  .alerta { font-size:9.5px; color:var(--pin); border:1px solid rgba(217,164,65,0.55); border-radius:8px;
            padding:0 5px; margin-left:4px; white-space:nowrap; }
  .menu { position:fixed; z-index:10; background:var(--menu); border:1px solid var(--borda); border-radius:8px;
          max-height:260px; overflow:auto; box-shadow:0 6px 18px rgba(0,0,0,0.35); min-width:210px; }
  .menu div { padding:6px 10px; cursor:pointer; display:flex; justify-content:space-between; gap:12px; }
  .menu div:hover { background:var(--cab); }
  .menu .sel { background:rgba(45,170,196,0.16); }
  .menu .pp { color:var(--pin); font-size:11px; }
  .nota { color:var(--fraco); font-size:11.5px; padding:6px 2px 0; line-height:1.45; }
  .tag { font-size:10px; color:var(--modelo); border:1px solid rgba(45,170,196,0.5); border-radius:8px;
         padding:0 5px; margin-left:6px; }
</style></head>
<body>
<div class="colunas" id="serie"></div>
<div class="topo" id="topo"></div>
<div class="colunas" id="colunas"></div>
<div class="nota">
  <b>Fair</b>: a linha justa do modelo (onde ele acha 50/50) — fixa.
  <b>Modelo</b> e <b>Pinnacle</b> andam juntas: clique no número da linha (sublinhado) em qualquer
  uma das duas e escolha outra linha. Começa na linha principal da Pinnacle. Em verde, o lado em que a
  Pinnacle paga mais que o modelo, com a vantagem.
</div>
<script>
const DADOS = __DADOS__;
const VENC = __VENCEDOR__;
const TITULO = "__TITULO__";
const sel = {};
let mapaAtual = null;
DADOS.forEach((m, i) => { m.id = i; sel[i] = m.inicial; });
const mapas = [...new Set(DADOS.map(m => m.mapa).concat(
  VENC.linhas.map(v => v.periodo).filter(p => p > 0)))].sort((a, b) => a - b);
mapaAtual = mapas[0];

function esc(x) { return String(x).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]); }
// Acima disto, desconfie do modelo antes da casa (mesma regra da aba:
// VANTAGEM_SUSPEITA_PCT em oportunidades_core). Medido 25/09: no vencedor o
// modelo e mais timido que o mercado (inclinacao 0,71).
const SUSPEITA = __SUSPEITA__;
function vantTxt(v) {
  if (v === null || v === undefined) return "";
  const t = `<span class="v ${v > 0 ? "b" : "r"}">${(v > 0 ? "+" : "") + v.toFixed(1)}%</span>`;
  return v > SUSPEITA ? t + `<span class="alerta">grande demais</span>` : t;
}
function fmtL(v) { return (Math.round(v * 10) / 10).toFixed(1); }
function fmtO(v) { return (v === null || v === undefined) ? null : Number(v).toFixed(2); }
function sinal(v) { return (v > 0 ? "+" : "") + fmtL(v); }
function rotulos(m, L) {
  if (m.tipo === "handicap") return [esc(m.casa) + " " + sinal(L), esc(m.fora) + " " + sinal(-L)];
  return [fmtL(L), fmtL(L)];
}
function acha(m, L) { return m.grade.find(g => Math.abs(g.linha - L) < 1e-9); }

function celula(m, L, odd, vant, clica, destaque) {
  const r = rotulos(m, L);
  return [0, 1].map(k => {
    const o = fmtO(odd[k]);
    if (o === null) return `<div class="cel"><span class="ln ${clica ? "clica" : ""}" data-m="${m.id}">${r[k]}</span><span class="nada">não oferece</span></div>`;
    const v = vant ? vant[k] : null;
    const verde = destaque && v !== null && v !== undefined && v > 0;
    const vt = destaque ? vantTxt(v) : "";
    const lnTxt = m.tipo === "handicap" ? r[k].split(" ").slice(-1)[0] : r[k];
    return `<div class="cel ${verde ? "verde" : ""}"><span class="ln ${clica ? "clica" : ""}" data-m="${m.id}">${lnTxt}</span><span class="odd">${o}</span>${vt}</div>`;
  }).join("");
}

function vencHTML(tabela, titulo, v) {
  let h = `<div class="secao"><h3>${titulo}</h3>`;
  h += `<div class="cab g2"><div>${esc(VENC.casa)}</div><div>${esc(VENC.fora)}</div></div><div class="row g2">`;
  [["a", "pin_a", "vant_a"], ["b", "pin_b", "vant_b"]].forEach(([k, pk, vk]) => {
    if (tabela === "pinnacle") {
      const vv = v[vk];
      const verde = vv !== null && vv !== undefined && vv > 0;
      const vt = vantTxt(vv);
      h += `<div class="cel ${verde ? "verde" : ""}"><span class="odd">${fmtO(v[pk])}</span>${vt}</div>`;
    } else {
      const o = fmtO(v["modelo_" + k]);
      h += o === null ? `<div class="cel"><span class="nada">abra este confronto</span></div>`
                      : `<div class="cel"><span class="odd">${o}</span></div>`;
    }
  });
  return h + `</div></div>`;
}

function secaoHTML(tabela, nome, itens) {
  const hand = itens[0].tipo === "handicap";
  let h = `<div class="secao"><h3>Mapa ${mapaAtual} - ${nome}</h3>`;
  if (hand) {
    const m = itens[0];
    h += `<div class="cab g2"><div>${esc(m.casa)}</div><div>${esc(m.fora)}</div></div>`;
  } else {
    h += `<div class="cab g3"><div></div><div>Mais de</div><div>Menos de</div></div>`;
  }
  itens.forEach(m => {
    let L, odd, vant = null, clica = false, dest = false;
    if (tabela === "fair") {
      L = m.fair.linha; odd = [m.fair.oa, m.fair.ob];
    } else {
      L = sel[m.id]; const g = acha(m, L) || {}; clica = true;
      if (tabela === "modelo") odd = [g.odd_modelo_a, g.odd_modelo_b];
      else { odd = [g.casa_a, g.casa_b]; vant = [g.vantagem_a, g.vantagem_b]; dest = true; }
    }
    const tag = (m.com_vitoria && tabela !== "pinnacle") ? `<span class="tag">c/ vitória</span>` : "";
    if (hand) h += `<div class="row g2">${celula(m, L, odd, vant, clica, dest)}</div>`;
    else h += `<div class="row g3"><div class="rot">${esc(m.rotulo)}${tag}</div>${celula(m, L, odd, vant, clica, dest)}</div>`;
  });
  return h + `</div>`;
}

const COLS = [["fair", "Fair (modelo)", "var(--modelo)"], ["modelo", "Modelo", "var(--modelo)"],
              ["pinnacle", "Pinnacle", "var(--pin)"]];

function desenhaSerie() {
  const serie = VENC.linhas.find(v => v.periodo === 0);
  const bo = VENC.bo ? ` (BO${VENC.bo})` : "";
  document.getElementById("serie").innerHTML = COLS.map(([t, nome, cor]) =>
    `<div class="coluna col-${t}"><div class="nome-col" style="color:${cor}">${nome}</div>` +
    (serie ? vencHTML(t, "Série - Vencedor" + bo, serie) : "") + `</div>`).join("");
}

function desenha() {
  document.getElementById("topo").innerHTML = `<b>${TITULO}</b>` + mapas.map(p =>
    `<button class="mapa ${p === mapaAtual ? "ativo" : ""}" data-mapa="${p}">Mapa ${p}</button>`).join("");
  const doMapa = DADOS.filter(m => m.mapa === mapaAtual);
  const secoes = ["Totais", "Handicap de Kills", "Kills por time"];
  const vm = VENC.linhas.find(v => v.periodo === mapaAtual);
  document.getElementById("colunas").innerHTML = COLS.map(([t, nome, cor]) => {
    let h = `<div class="coluna col-${t}">`;
    if (vm) h += vencHTML(t, `Mapa ${mapaAtual} - Vencedor`, vm);
    secoes.forEach(s => {
      const itens = doMapa.filter(m => m.secao === s).sort((a, b) => a.ordem - b.ordem);
      if (itens.length) h += secaoHTML(t, s, itens);
    });
    return h + `</div>`;
  }).join("");
}

function fechaMenu() { const x = document.querySelector(".menu"); if (x) x.remove(); }

desenhaSerie();
document.addEventListener("click", ev => {
  const bm = ev.target.closest(".mapa");
  if (bm) { mapaAtual = Number(bm.dataset.mapa); fechaMenu(); desenha(); return; }
  const ln = ev.target.closest(".ln.clica");
  if (ln) {
    fechaMenu();
    const m = DADOS[Number(ln.dataset.m)];
    const menu = document.createElement("div"); menu.className = "menu";
    menu.innerHTML = m.grade.map(g => {
      const r = rotulos(m, g.linha);
      const pin = (g.casa_a && g.casa_b) ? "Pinnacle" : "";
      const pp = g.principal ? "principal" : pin;
      return `<div class="${Math.abs(g.linha - sel[m.id]) < 1e-9 ? "sel" : ""}" data-l="${g.linha}">
                <span>${m.tipo === "handicap" ? r[0] : r[0]}</span><span class="pp">${pp}</span></div>`;
    }).join("");
    const box = ln.getBoundingClientRect();
    menu.style.left = Math.min(box.left, window.innerWidth - 230) + "px";
    menu.style.top = Math.min(box.bottom + 4, window.innerHeight - 270) + "px";
    menu.addEventListener("click", e2 => {
      const it = e2.target.closest("[data-l]");
      if (!it) return;
      sel[m.id] = Number(it.dataset.l); fechaMenu(); desenha();
      e2.stopPropagation();
    });
    document.body.appendChild(menu);
    const s = menu.querySelector(".sel"); if (s) s.scrollIntoView({block: "center"});
    ev.stopPropagation();
    return;
  }
  if (!ev.target.closest(".menu")) fechaMenu();
});
desenha();
</script>
</body></html>
"""
