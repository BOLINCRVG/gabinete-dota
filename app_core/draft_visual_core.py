# -*- coding: utf-8 -*-
"""TELA DO DRAFT (24/11/2026): o draft desenhado como no jogo -- bans em cima, os 5 de cada lado com a
foto do heroi, as ROTAS no centro e o veredito embaixo. Ele: "fazer igual e o draft do jogo mesmo,
aparecer os 10 picks e os 10 bans, e mostrar no centro, quem tem vantagem na rota [...] e iria mostrar
a foto dos bonecos [...] 'o draft da esquerda e melhor por x motivo. Mas, se o jogo chegar a x
minutos, ele tende a ser melhor para o time da direita por x motivo'".

So entra numero MEDIDO: cada jogador vale o que a camada de draft mede (heroi + jogador no heroi,
app_core/draft_colado_core.explicar). A rota soma os jogadores dela. O bloco "se o jogo passar de X
minutos" fica marcado "em medicao" ate passar no teste as cegas -- nao se inventa frase.
Rotas do Dota: safe (pos 1+5) de um x off (pos 3+4) do outro; mid x mid. Arquivo identico nos repos.
"""
from __future__ import annotations

import html as _h
from typing import Any, Dict, List, Mapping, Optional, Sequence

IMG_DOTA = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{}.png"
COR = {"Radiant": "#5bbd5e", "Dire": "#e0533d", "Blue": "#4f8fe8", "Red": "#e0533d"}

IMG_LOL = "https://ddragon.leagueoflegends.com/cdn/{}/img/champion/{}.png"

# (rotulo, posicoes da esquerda, posicoes da direita) -- indices 0..4 = pos 1..5 (Dota) /
# top, jungle, mid, adc, suporte (LoL)
ROTAS_DOTA = (("Safe da esquerda × Off da direita", (0, 4), (2, 3)),
              ("Mid", (1,), (1,)),
              ("Off da esquerda × Safe da direita", (2, 3), (0, 4)))
ROTAS_LOL = (("Top", (0,), (0,)), ("Jungle", (1,), (1,)), ("Mid", (2,), (2,)), ("Bot", (3, 4), (3, 4)))


def _e(s: Any) -> str:
    return _h.escape(str(s if s is not None else ""))


def _pp(x: float) -> str:
    return f"{x:+.1f}".replace(".", ",") + " pp"


def _odd(p: float) -> str:
    return "—" if p <= 0 else f"{1 / p:.2f}".replace(".", ",")


CSS = """
<style>
.dv{font-family:'Inter',system-ui,sans-serif;background:#0d1117;color:#e6e6e6;border-radius:14px;padding:18px 18px 14px;}
.dv *{box-sizing:border-box}
.dv-top{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px}
.dv-time{display:flex;flex-direction:column;gap:2px}
.dv-time.d{align-items:flex-end;text-align:right}
.dv-nome{font-size:22px;font-weight:800;letter-spacing:.2px}
.dv-lado{font-size:11px;font-weight:700;letter-spacing:1.6px;text-transform:uppercase}
.dv-placar{display:flex;align-items:baseline;gap:14px;font-variant-numeric:tabular-nums}
.dv-p{font-size:34px;font-weight:800}
.dv-x{color:#6b7280;font-size:14px}
.dv-sub{font-size:12px;color:#9aa4b2;text-align:center;margin-top:2px}
.dv-bans{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:14px 0 6px}
.dv-bt{font-size:10px;letter-spacing:1.4px;color:#6b7280;margin-bottom:3px}
.dv-bans .d .dv-bt{text-align:right}
.dv-banrow{display:flex;gap:4px;flex-wrap:wrap}
.dv-banrow.d{justify-content:flex-end}
.dv-ban{position:relative;width:46px;height:26px;border-radius:3px;overflow:hidden;filter:grayscale(.85) brightness(.8)}
.dv-ban img{width:100%;height:100%;object-fit:cover}
.dv-ban:after{content:"";position:absolute;left:-4px;right:-4px;top:50%;border-top:2px solid #e0533d;transform:rotate(-28deg)}
.dv-rotulo{font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:#6b7280;margin:10px 0 4px}
.dv-rota{display:grid;grid-template-columns:1fr 210px 1fr;gap:10px;align-items:center;background:#151b23;border:1px solid #232b36;border-radius:10px;padding:8px;margin-bottom:8px}
.dv-lado-r{display:flex;flex-direction:column;gap:6px}
.dv-j{display:flex;align-items:center;gap:8px}
.dv-j.d{flex-direction:row-reverse;text-align:right}
.dv-j img{width:96px;height:54px;border-radius:5px;object-fit:cover;border:1px solid #2b3440}
.dv-jn{font-weight:700;font-size:14px}
.dv-jh{font-size:12px;color:#9aa4b2}
.dv-jx{font-size:11px;color:#7d8794}
.dv-jpp{font-size:12px;font-weight:700;font-variant-numeric:tabular-nums}
.dv-mid{text-align:center}
.dv-mid .t{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#9aa4b2;margin-bottom:6px}
.dv-bar{position:relative;height:10px;background:#232b36;border-radius:6px;overflow:hidden}
.dv-bar .c{position:absolute;left:50%;top:-2px;bottom:-2px;width:2px;background:#4b5563}
.dv-bar .f{position:absolute;top:0;bottom:0}
.dv-mid .v{font-size:13px;font-weight:800;margin-top:6px;font-variant-numeric:tabular-nums}
.dv-ver{margin-top:10px;background:#111827;border-left:4px solid var(--c);border-radius:8px;padding:12px 14px}
.dv-ver h4{margin:0 0 6px;font-size:16px}
.dv-ver li{font-size:13px;color:#cbd5e1;margin:2px 0}
.dv-tempo{margin-top:8px;font-size:12px;color:#7d8794;border:1px dashed #2b3440;border-radius:8px;padding:8px 12px}
.pos{color:#5bbd5e}.neg{color:#e0533d}
</style>
"""


def _jog(j: Mapping[str, Any], dir_: bool) -> str:
    cls = "pos" if j["pp"] >= 0 else "neg"
    return (f'<div class="dv-j{" d" if dir_ else ""}"><img src="{_e(j["img"])}" alt="">'
            f'<div><div class="dv-jn">{_e(j["nome"])}</div><div class="dv-jh">{_e(j["heroi"])}</div>'
            f'<div class="dv-jx">{_e(j["detalhe"])}</div>'
            f'<div class="dv-jpp {cls}">{_pp(j["pp"])}</div></div></div>')


def html_draft(d: Mapping[str, Any], rotas=ROTAS_DOTA) -> str:
    """d = {'esq': lado, 'dir': lado, 'p_pre': P(esq) antes, 'p_pos': P(esq) depois, 'lado_pp': pp a favor
    da esquerda pelo lado do mapa, 'veredito': [titulo, motivos...], 'tempo': texto ou None}
    lado = {'time', 'lado' ('Radiant'|'Dire'), 'jogadores': 5 x {nome, heroi, img, pp, detalhe} na ordem
    pos 1..5, 'bans': [{heroi, img}]}"""
    E, D = d["esq"], d["dir"]
    ce, cd = COR.get(E["lado"], "#5bbd5e"), COR.get(D["lado"], "#e0533d")
    p0, p1 = float(d["p_pre"]), float(d["p_pos"])
    delta = 100 * (p1 - p0)
    out = [CSS, '<div class="dv">', '<div class="dv-top">',
           f'<div class="dv-time"><span class="dv-lado" style="color:{ce}">{_e(E["lado"])}</span>'
           f'<span class="dv-nome">{_e(E["time"])}</span></div>',
           '<div><div class="dv-placar">'
           f'<span class="dv-p" style="color:{ce}">{100 * p1:.0f}%</span><span class="dv-x">mapa</span>'
           f'<span class="dv-p" style="color:{cd}">{100 * (1 - p1):.0f}%</span></div>'
           f'<div class="dv-sub">odd justa {_odd(p1)} × {_odd(1 - p1)} · antes do draft {100 * p0:.0f}% '
           f'({"+" if delta >= 0 else "−"}{abs(delta):.1f} pp {"esquerda" if delta >= 0 else "direita"})</div></div>',
           f'<div class="dv-time d"><span class="dv-lado" style="color:{cd}">{_e(D["lado"])}</span>'
           f'<span class="dv-nome">{_e(D["time"])}</span></div>', '</div>']
    tem_ban = bool(E.get("bans") or D.get("bans"))
    if tem_ban:
        out.append('<div class="dv-bans">')
    for L, dir_ in (((E, False), (D, True)) if tem_ban else ()):
        out.append(f'<div class="{"d" if dir_ else ""}"><div class="dv-bt">BANS</div><div class="dv-banrow{" d" if dir_ else ""}">' + "".join(
            f'<div class="dv-ban" title="{_e(b["heroi"])}"><img src="{_e(b["img"])}" alt=""></div>' for b in L["bans"])
            + "</div></div>")
    if tem_ban:
        out.append("</div>")
    out.append('<div class="dv-rotulo">Rotas — o que o jogador naquele herói soma (medido)</div>')
    je, jd = E["jogadores"], D["jogadores"]
    for rot, pe, pd_ in rotas:
        se = sum(je[i]["pp"] for i in pe if i < len(je))
        sd = sum(jd[i]["pp"] for i in pd_ if i < len(jd))
        dif = se - sd
        larg = min(50.0, abs(dif) * 8)
        # a barra cresce do centro para o lado de quem leva a rota
        barra = (f'<div class="f" style="right:50%;width:{larg}%;background:{ce}"></div>' if dif > 0 else
                 f'<div class="f" style="left:50%;width:{larg}%;background:{cd}"></div>')
        quem = (E["time"] if dif > 0 else D["time"]) if abs(dif) >= 0.05 else "equilibrado"
        out.append('<div class="dv-rota"><div class="dv-lado-r">' + "".join(_jog(je[i], False) for i in pe if i < len(je))
                   + f'</div><div class="dv-mid"><div class="t">{_e(rot)}</div><div class="dv-bar"><div class="c"></div>{barra}</div>'
                   f'<div class="v" style="color:{ce if dif > 0 else cd}">{_e(quem)} {_pp(abs(dif)) if quem != "equilibrado" else ""}</div></div>'
                   '<div class="dv-lado-r">' + "".join(_jog(jd[i], True) for i in pd_ if i < len(jd)) + "</div></div>")
    v = list(d.get("veredito") or [])
    cor_v = ce if p1 >= p0 else cd
    if v:
        out.append(f'<div class="dv-ver" style="--c:{cor_v}"><h4>{_e(v[0])}</h4><ul>'
                   + "".join(f"<li>{_e(x)}</li>" for x in v[1:]) + "</ul></div>")
    t = d.get("tempo")
    out.append('<div class="dv-tempo">' + (_e(t) if t else
               "Se o jogo passar de X minutos: <b>em medição</b> — a curva de cada draft por duração só aparece "
               "aqui depois de passar no teste às cegas.") + "</div>")
    out.append("</div>")
    return "".join(out)


def veredito(esq: str, dir_: str, lado_esq: str, lado_dir: str, p_pre: float, p_pos: float,
             jog_esq: Sequence[Mapping[str, Any]], jog_dir: Sequence[Mapping[str, Any]], lado_pp: float) -> List[str]:
    """Pura: titulo + ate 4 motivos, dos numeros medidos (maior |pp| primeiro)."""
    dlt = 100 * (p_pos - p_pre) - lado_pp
    if abs(dlt) < 0.3:
        tit = "Draft equilibrado: os dois lados se anulam (menos de 0,3 pp)."
    else:
        ganha, lado = (esq, lado_esq) if dlt > 0 else (dir_, lado_dir)
        lado_txt = "esquerda" if dlt > 0 else "direita"
        tit = f"O draft da {lado_txt} ({lado} · {ganha}) é melhor: {_pp(abs(dlt))} no mapa."
    motivos = []
    todos = [(j, esq) for j in jog_esq] + [(j, dir_) for j in jog_dir]
    for j, tm in sorted(todos, key=lambda x: -abs(x[0]["pp"]))[:4]:
        motivos.append(f'{j["nome"]} de {j["heroi"]} ({tm}): {_pp(j["pp"])} — {j["detalhe"]}')
    if abs(lado_pp) >= 0.1:
        motivos.append(f"Lado do mapa: {_pp(lado_pp)} para a esquerda ({lado_esq}).")
    return [tit] + motivos
