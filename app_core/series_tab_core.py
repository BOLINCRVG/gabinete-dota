"""Series tab renderer."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional


import pandas as pd

from app_core.tab_ui_core import render_tab_hero, render_tab_section


# FORMA DO DIA (29/10/2026, ele: "a analise e muito perdida no +1.5 e no -1.5"). Os mapas
# de uma serie NAO sao independentes: o time que esta bem no dia tende a ganhar os dois.
# Medido no LoL (2.326 BO3 de 2026): 2-0 real acima do previsto por p*p (favorito 70-80%:
# 61,8% previsto x 70,1% real). Correcao: a chance de cada mapa vem de uma Beta com media
# p e concentracao KAPPA (mapas trocaveis entre si). KAPPA=3 escolhido em jan-mai/2026;
# cego jun-out (972 series): placar da serie +0,0194 IC95 [+0,0024; +0,0355], "teve
# 2-0?" +0,0157.
# DOTA (15/11/2026, medido): BO3 de 2025-2026 (7.972 series, chance da serie por um Elo simples,
# todo kappa reproduzindo a MESMA chance de serie). Escolhendo kappa so com o passado, em 3 blocos
# seguidos de ~2.000 series o escolhido foi sempre 3 e o placar exato melhorou +0,0184 IC95
# [+0,0115; +0,0252] (favorito 60%+: 2-0 previsto por p*p 49,7%, real 54,4%). O bloco mais novo
# sozinho (dez/2025-nov/2026) deu +0,0090 com IC cruzando o zero: efeito menor em 2026, a
# acompanhar. BO5 do Dota: 187 series, nao da para medir -> mapas independentes.
def _e_dota() -> bool:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "dota_paths.py").exists()


E_DOTA = _e_dota()


def kappa_do_bo(bo: Any) -> Optional[float]:
    """A concentracao da forma do dia para o formato: LoL 3 (BO3 e BO5); Dota 3 so no BO3."""
    try:
        b = int(bo)
    except (TypeError, ValueError):
        return None
    if E_DOTA:
        return 3.0 if b == 3 else None
    return 3.0


KAPPA_SERIE = kappa_do_bo(3)


def _momento(p: float, a: int, b: int, kappa: Optional[float]) -> float:
    """E[t^a (1-t)^b] com t ~ Beta(p*kappa, (1-p)*kappa); kappa None = mapas independentes."""
    q = 1.0 - p
    if kappa is None:
        return (p ** a) * (q ** b)
    al, be = p * kappa, q * kappa
    num = 1.0
    for i in range(a):
        num *= al + i
    for j in range(b):
        num *= be + j
    den = 1.0
    for k in range(a + b):
        den *= kappa + k
    return num / den


def score_probs(p: float, bo: int, kappa: Optional[float] = "padrao") -> dict[str, float]:
    """Placar da serie a partir da chance do mapa. kappa: concentracao da "forma do dia"
    (padrao: KAPPA_SERIE; None = mapas independentes, a conta antiga)."""
    k = kappa_do_bo(bo) if kappa == "padrao" else kappa
    p = float(p)
    if k is not None:
        p = min(max(p, 1e-6), 1.0 - 1e-6)          # a Beta pede 0 < p < 1
    m = lambda a, b: _momento(p, a, b, k)
    if int(bo) == 3:
        return {
            "A_2-0": m(2, 0),
            "A_2-1": 2 * m(2, 1),
            "B_2-1": 2 * m(1, 2),
            "B_2-0": m(0, 2),
        }
    if int(bo) == 5:
        return {
            "A_3-0": m(3, 0),
            "A_3-1": 3 * m(3, 1),
            "A_3-2": 6 * m(3, 2),
            "B_3-2": 6 * m(2, 3),
            "B_3-1": 3 * m(1, 3),
            "B_3-0": m(0, 3),
        }
    return {}


def chance_do_mapa_coerente(p_serie: Any, bo: int, kappa: Optional[float] = "padrao") -> Optional[float]:
    """A chance do mapa que, com a forma do dia, reproduz a chance da SERIE do proprio modelo.

    29/10/2026, ele: "o -1.5 da odd 1,83 quando a MD3 e 1,10 -- essa odd nunca vai ter".
    Era isso: a serie passa pelo ajuste de serie (gamma) e sai 1,10 (91%), mas o -1,5 saia
    da chance CRUA do mapa (74% -> 1,83). Os dois numeros nao conversavam. Resolvendo a
    chance do mapa que bate com os 91%, o -1,5 sai 1,27 -- coerente com o ML da serie."""
    try:
        ps = float(p_serie)
    except (TypeError, ValueError):
        return None
    if not (0.0 < ps < 1.0) or int(bo) not in (3, 5):
        return None
    k = kappa_do_bo(bo) if kappa == "padrao" else kappa

    def _serie(m):
        return sum(v for c, v in score_probs(m, int(bo), kappa=k).items() if c.startswith("A_"))
    lo, hi = 1e-4, 1 - 1e-4
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _serie(mid) < ps else (lo, mid)
    return (lo + hi) / 2



# ---- A ABA SERIES REMODELADA (29/10/2026, ele: "remodelar de vez essa aba series") ----------
# Uma tela so, tudo saindo do MESMO numero (o ML da serie do card) e da forma do dia:
#   vencedor da serie -> placar exato -> handicaps de mapas -> total de mapas -> totais da serie.
# Cada aposta: chance, justa, a melhor odd das casas dele (quando a casa oferece) e o valor.
# Os totais da serie somam os mapas jogados com o NUMERO DE MAPAS do placar (2-0 joga 2,
# 2-1 joga 3) -- antes a simulacao jogava os mapas como independentes e nao conversava com o
# +-1,5. As linhas dos totais entram sozinhas (a mediana do modelo e uma de cada lado).
DESTAQUE = 1.72                     # a regra dele: justa abaixo de 1,72


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mapas_jogados(probs: Mapping[str, float]) -> dict[int, float]:
    """{n mapas: chance} a partir do placar (2-0 = 2 mapas, 3-1 = 4...)."""
    fora: dict[int, float] = {}
    for k, v in probs.items():
        a, b = k.split("_")[1].split("-")
        n = int(a) + int(b)
        fora[n] = fora.get(n, 0.0) + float(v)
    return fora


def p_mais_serie(n_mapas: Mapping[int, float], media: float, dp: float, linha: float) -> Optional[float]:
    """P(total da serie > linha): a soma de n mapas ~ Normal(n*media, raiz(n)*dp), misturada
    pelo numero de mapas."""
    if not (math.isfinite(media) and math.isfinite(dp)) or dp <= 0:
        return None
    return float(sum(pn * (1.0 - _phi((float(linha) - n * media) / (math.sqrt(n) * dp)))
                     for n, pn in n_mapas.items()))


def linhas_auto_serie(n_mapas: Mapping[int, float], media: float, dp: float, passo: float) -> list[float]:
    """A linha .5 mais perto da mediana do modelo e uma de cada lado (passo da metrica)."""
    lo, hi = 0.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = p_mais_serie(n_mapas, media, dp, mid)
        if p is None:
            return []
        lo, hi = (mid, hi) if p > 0.5 else (lo, mid)
    centro = math.floor((lo + hi) / 2) + 0.5
    return [centro - passo, centro, centro + passo]


def mercados_da_serie(p_serie_a: float, bo: int, nome_a: str, nome_b: str) -> list[dict[str, Any]]:
    """Vencedor, placar, handicaps e total de mapas -- coerentes com o ML da serie. Pura."""
    m = chance_do_mapa_coerente(p_serie_a, int(bo)) if kappa_do_bo(bo) is not None else None
    if m is None:
        # sem forma do dia (Dota): a chance do mapa que, independente, da a mesma serie
        lo, hi = 1e-4, 1 - 1e-4
        for _ in range(60):
            mid = (lo + hi) / 2
            ps = sum(v for c, v in score_probs(mid, int(bo), kappa=None).items() if c.startswith("A_"))
            lo, hi = (mid, hi) if ps < p_serie_a else (lo, mid)
        m = (lo + hi) / 2
    pr = score_probs(m, int(bo))
    pa = sum(v for c, v in pr.items() if c.startswith("A_"))
    fora: list[dict[str, Any]] = [
        {"secao": "Vencedor da série", "aposta": nome_a, "p": pa, "chave": "venc_A"},
        {"secao": "Vencedor da série", "aposta": nome_b, "p": 1.0 - pa, "chave": "venc_B"},
    ]
    for k, v in pr.items():
        quem, placar = k.split("_")
        a, b = placar.split("-")
        txt = f"{nome_a} {a}-{b}" if quem == "A" else f"{nome_b} {a}-{b}"
        fora.append({"secao": "Placar exato", "aposta": txt, "p": float(v), "chave": f"placar_{k}"})
    # handicaps: A -h cobre se A ganha por mais de h mapas
    def _saldo_a(k):
        quem, placar = k.split("_")
        a, b = (int(x) for x in placar.split("-"))
        return (a - b) if quem == "A" else (b - a)
    hs = (1.5,) if int(bo) == 3 else (1.5, 2.5)
    for h in hs:
        p_a_menos = sum(v for k, v in pr.items() if _saldo_a(k) > h)
        p_b_menos = sum(v for k, v in pr.items() if -_saldo_a(k) > h)
        for nome, p, ch in ((f"{nome_a} -{h:g}", p_a_menos, f"hc_A_-{h:g}"),
                            (f"{nome_b} +{h:g}", 1.0 - p_a_menos, f"hc_B_+{h:g}"),
                            (f"{nome_b} -{h:g}", p_b_menos, f"hc_B_-{h:g}"),
                            (f"{nome_a} +{h:g}", 1.0 - p_b_menos, f"hc_A_+{h:g}")):
            fora.append({"secao": "Handicap de mapas", "aposta": nome.replace(".", ","), "p": float(p), "chave": ch})
    nm = mapas_jogados(pr)
    for L in ((2.5,) if int(bo) == 3 else (3.5, 4.5)):
        p_mais = sum(v for n, v in nm.items() if n > L)
        fora.append({"secao": "Total de mapas", "aposta": f"Mais de {L:g} mapas".replace(".", ","), "p": p_mais,
                     "chave": f"mapas_over_{L:g}"})
        fora.append({"secao": "Total de mapas", "aposta": f"Menos de {L:g} mapas".replace(".", ","), "p": 1 - p_mais,
                     "chave": f"mapas_under_{L:g}"})
    for r in fora:
        r["n_mapas"] = nm
    return fora


def odds_das_casas_serie(jogo: Any, inv: bool, esp: str, casas: list) -> dict[str, tuple]:
    """{chave: (melhor odd, casa)} -- vencedor da serie e +-1,5/2,5 de mapas das casas dele."""
    from app_core.suas_casas_core import linhas, linhas_handicap_mapas
    best: dict[str, tuple] = {}

    def _poe(k, odd, casa):
        if odd and float(odd) > 1.0 and (k not in best or float(odd) > best[k][0]):
            best[k] = (float(odd), str(casa))
    if not jogo:
        return best
    for l in linhas(jogo, inv, esp, casas_dele=casas):
        if int(l.get("periodo") or 0) == 0:
            _poe("venc_A", l["odd_a"], l["casa"])
            _poe("venc_B", l["odd_b"], l["casa"])
    com_pin = any("pinnacle" in str(c).lower() for c in casas)
    for l in linhas_handicap_mapas(jogo, inv, esp, casas_dele=casas):
        if l.get("slug") == "pinnacle" and not com_pin:      # a Pinnacle so entra como regua
            continue
        h = float(l["handicap_a"])
        _poe(f"hc_A_{h:+g}", l["odd_a"], l["casa"])
        _poe(f"hc_B_{-h:+g}", l["odd_b"], l["casa"])
    # totais da serie (mapa 0): mapas, kills da serie, kills de cada time
    from app_core.suas_casas_core import _marcas, linhas_de_mapa, versao_da_marca
    rot = {"over": "Mais", "under": "Menos"}
    linhas_casa: dict[str, set] = {}
    for l in linhas_de_mapa(jogo, inv, esp, casas_dele=casas):
        if int(l["periodo"]) != 0 or l["lado"] not in rot or (l["slug"] == "pinnacle" and not com_pin):
            continue
        L = float(l["pontos"])
        if l["unidade"] == "maps" and l["tipo"] == "total":
            _poe(f"mapas_{l['lado']}_{L:g}", l["odd"], l["casa_aposta"])
        elif l["unidade"] == "kills" and l["tipo"] == "total":
            _poe(f"tot_kills_{rot[l['lado']]}_{L:g}", l["odd"], l["casa_aposta"])
            linhas_casa.setdefault("kills", set()).add(L)
        elif l["unidade"] == "kills" and l["tipo"] == "team_total":
            q = "A" if l.get("time") == "home" else "B"
            _poe(f"tk_{q}_{rot[l['lado']]}_{L:g}", l["odd"], l["casa_aposta"])
            linhas_casa.setdefault(f"tk_{q}", set()).add(L)
    # placar exato: (diferenca de mapas, time 1 da OddsPapi, time 2) -> placar_A_2-0 ...
    todas = jogo.get("linhas") or {}
    for marca, nome in _marcas(casas):
        slug = versao_da_marca(todas, marca)
        if not slug:
            continue
        for unidade, tipo, mapa, pontos, oa, ob, _q in todas[slug]:
            if tipo != "placar":
                continue
            d = int(round(float(pontos)))
            for quem, odd in ((("B", "A") if inv else ("A", "B"))[0], oa), ((("B", "A") if inv else ("A", "B"))[1], ob):
                best_k = [f"placar_{quem}_{w}-{w - d}" for w in (2, 3) if w - d >= 0]
                for k in best_k:
                    _poe(k, odd, nome)
    best["_linhas"] = {k: sorted(v) for k, v in linhas_casa.items()}
    return best


def _linhas_da_casa(odds: Mapping[str, Any], chave_linhas: str, prefixo: str) -> list[float]:
    """As linhas da serie que as casas dele abriram com os DOIS lados pagando entre 1,50 e 2,60
    (a principal e as vizinhas; as alternativas de 1,10 x 6,00 ficam de fora)."""
    fora = []
    for L in (odds.get("_linhas") or {}).get(chave_linhas, []):
        mais, menos = odds.get(f"{prefixo}_Mais_{L:g}"), odds.get(f"{prefixo}_Menos_{L:g}")
        if mais and menos and 1.50 <= mais[0] <= 2.60 and 1.50 <= menos[0] <= 2.60:
            fora.append(float(L))
    return fora


def _tabela(st: Any, linhas_: list[dict[str, Any]], odds: Mapping[str, tuple]) -> None:
    rows = []
    for r in linhas_:
        p = r.get("p")
        if p is None or not (0.0 < p < 1.0):
            continue
        justa = 1.0 / p
        casa = odds.get(r.get("chave", ""))
        valor = None if not casa else (p * casa[0] - 1.0) * 100.0
        rows.append({"": "★" if justa < DESTAQUE else "", "Aposta": r["aposta"], "Chance": f"{100 * p:.1f}%",
                     "Justa": round(justa, 2), "Melhor casa": f"{casa[1]} {casa[0]:.2f}" if casa else "—",
                     "Valor": f"{valor:+.1f}%" if valor is not None else "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_series_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(st, title="Series", subtitle="Tudo da série num lugar só, saindo do mesmo número: "
                    "vencedor, placar, handicaps, total de mapas e totais da série.")
    ml_info = ctx["ml_info"]
    team_a, team_b, bo = ctx["teamA"], ctx["teamB"], int(ctx["bo"])
    p_map_cal = float(ctx["p_map_cal"])
    state_get_float_fn = ctx["state_get_float_fn"]
    parse_lines_fn = ctx["parse_lines_fn"]
    totals_avg = ctx["totals_avg"]
    profile_a, profile_b = ctx["profile_a"], ctx["profile_b"]

    league_a = league_b = ""
    if ml_info is not None:
        league_a = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
        league_b = str(((ml_info.get("red") or {}).get("league")) or "").strip()
    nome_a = ctx["team_with_league_label_fn"](team_a, league_a)
    nome_b = ctx["team_with_league_label_fn"](team_b, league_b)

    wp = ctx["winner_pricing_bundle_fn"](
        p_team_a_map=state_get_float_fn(st.session_state, "_p_map_used", float(p_map_cal)),
        p_team_a_series=state_get_float_fn(st.session_state, "_p_series_used", float("nan")),
        bo_for_series=int(bo), prob_win_series_fn=ctx["prob_win_series_fn"],
        odd_from_prob_fn=ctx["odd_from_prob_fn"])
    o_a, o_b = wp.get("oddA_series"), wp.get("oddB_series")
    if int(bo) not in (3, 5) or o_a is None or o_b is None:
        st.info("Sem preço de série para este formato (BO1/BO2). Use o ML do mapa nas outras abas.")
        return
    ia, ib = 1.0 / float(o_a), 1.0 / float(o_b)
    p_serie_a = ia / (ia + ib)

    # as casas dele
    odds: dict[str, tuple] = {}
    try:
        from pathlib import Path
        from app_core.suas_casas_core import CASAS_PADRAO, jogo_das_casas
        tg = ctx.get("team_games")
        if tg is not None:
            raiz = Path(__file__).resolve().parents[1]
            esp = "dota" if (raiz / "dota_paths.py").exists() else "lol"
            casas = list(st.session_state.get("suas_casas_escolhidas") or []) or list(CASAS_PADRAO)
            times = sorted({str(x) for x in tg["team"].dropna().unique()})
            jogo, inv = jogo_das_casas(raiz, str(team_a), str(team_b), times)
            odds = odds_das_casas_serie(jogo, inv, esp, casas)
            if not jogo:
                st.caption("Este confronto não está na coleta das suas casas: só a justa do modelo.")
    except Exception:
        odds = {}

    merc = mercados_da_serie(p_serie_a, int(bo), nome_a, nome_b)
    st.caption(f"{nome_a} x {nome_b} · BO{bo} · ★ = justa abaixo de {DESTAQUE:.2f}".replace(".", ",")
               + (" · forma do dia ligada (os mapas da série andam juntos)" if kappa_do_bo(bo) is not None else ""))
    for secao in ("Vencedor da série", "Handicap de mapas", "Placar exato", "Total de mapas"):
        render_tab_section(st, title=secao, subtitle="")
        _tabela(st, [r for r in merc if r["secao"] == secao], odds)

    # totais da serie
    nm = merc[0]["n_mapas"]
    render_tab_section(st, title="Totais da série", subtitle=(
        "A soma dos mapas jogados, com o número de mapas do placar acima (2-0 joga 2, 2-1 joga 3). "
        "As linhas entram sozinhas; adicione outras abaixo."))
    passos = {"kills": 3.0, "towers": 2.0, "dragons": 1.0, "barons": 1.0}
    rotulos = {"kills": "Kills", "towers": "Torres", "dragons": "Dragões", "barons": "Barões"}
    if E_DOTA:                                  # Dota: o "baron" do motor e o Roshan
        rotulos["barons"] = "Roshans"
    with st.expander("Adicionar linhas da série", expanded=False):
        extra = {m: parse_lines_fn(st.text_input(f"{rotulos[m]} (série) — ex: 75.5", key=f"series_{m}_txt"))
                 for m in passos}
    linhas_tot: list[dict[str, Any]] = []
    for m, passo in passos.items():
        t = totals_avg.get(m) if hasattr(totals_avg, "get") else None
        if t is None:
            continue
        media, dp = float(getattr(t, "mean", float("nan"))), float(getattr(t, "sd", float("nan")))
        lins = sorted(set(linhas_auto_serie(nm, media, dp, passo)) | {float(x) for x in (extra.get(m) or [])}
                      | set(_linhas_da_casa(odds, m, f"tot_{m}")))
        for L in lins:
            p = p_mais_serie(nm, media, dp, L)
            if p is None:
                continue
            for lado, pp in (("Mais", p), ("Menos", 1.0 - p)):
                linhas_tot.append({"aposta": f"{rotulos[m]} {lado} de {L:g}".replace(".", ","), "p": pp,
                                   "chave": f"tot_{m}_{lado}_{L:g}"})
    _tabela(st, linhas_tot, odds)

    # kills de cada time na serie
    try:
        cm, cs = ctx["combine_mean_fn"], ctx["combine_sd_fn"]
        tk = []
        for q, nome, pf, pa_ in (("A", nome_a, profile_a, profile_b), ("B", nome_b, profile_b, profile_a)):
            media = float(cm(pf.kills_for, pa_.kills_against))
            dp = float(max(cs(pf.kills_for_sd, pa_.kills_against_sd), 1.0))
            for L in sorted(set(linhas_auto_serie(nm, media, dp, 2.0)) | set(_linhas_da_casa(odds, f"tk_{q}", f"tk_{q}"))):
                p = p_mais_serie(nm, media, dp, L)
                if p is None:
                    continue
                for lado, pp in (("Mais", p), ("Menos", 1.0 - p)):
                    tk.append({"aposta": f"Kills {nome} {lado} de {L:g}".replace(".", ","), "p": pp,
                               "chave": f"tk_{q}_{lado}_{L:g}"})
        if tk:
            render_tab_section(st, title="Kills de cada time na série", subtitle="")
            _tabela(st, tk, odds)
    except Exception:
        pass
