# -*- coding: utf-8 -*-
"""Mede e gera a forma do handicap de kills e do total de kills por time.

O QUE ELE PEDIU (24/09/2026): que o modelo calcule o handicap de kills e o
total de kills de cada time -- os mercados em que ele gira, e que a aba de
Oportunidades mostrava so com o preco da Pinnacle.

O QUE ESTE PROGRAMA FAZ
  1. Percorre os jogos EM ORDEM (data, gameid), um jogo por vez, e para cada
     um calcula os kills esperados de cada time SO com o que veio antes --
     pela mesma conta Dixon-Coles da producao (`mlcore.abates_por_time.
     medias_dc`), feita de forma incremental para caber no tempo. O teste
     `test_o_incremental_bate_com_a_producao` prova a igualdade.
  2. Escolhe como medir o espalhamento (bruto, pela raiz, por uma reta) com
     ajuste em 2024 e conferencia em 2025. 2026 nao entra na escolha.
  3. Ajusta a forma final em 2024-2025 e mede UMA vez, cego, em 2026.
  4. So grava o mercado que passar: ganho sobre chutar a taxa-base com o
     limite inferior do intervalo (agrupado por jogo) acima de zero, e erro
     de calibracao condicional abaixo de 2 pontos percentuais. Mercado por
     mercado -- no Dota (24/09/2026) o handicap passou e o total por time
     nao, e so o handicap foi gravado.

AS LINHAS MEDIDAS sao as que a Pinnacle oferece: em volta da diferenca (ou
do total) esperada, 5 de cada lado. Linha longe da media e facil para
qualquer modelo e inflaria o ganho -- media em linha que ninguem oferece nao
diz nada sobre o dinheiro dele.

Uso: python tools/gerar_abates_por_time.py [--sem-gravar]
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

def _achar_raiz() -> Path:
    """Raiz pelo git, nunca pela pasta do arquivo (arquivo IDENTICO no LoL
    e no Dota; ver a armadilha no CLAUDE.md do Dota)."""
    import subprocess
    aqui = Path(__file__).resolve().parent
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(aqui),
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return aqui.parent


ROOT = _achar_raiz()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OS PERIODOS saem dos anos que o historico tem, pela mesma regra nos dois
# jogos -- o ultimo ano e SEMPRE o cego:
#   3 anos ou mais (LoL: 2024-2026): escolhe ajustando no 1o ano e
#     conferindo no 2o; ajuste final em todos menos o ultimo.
#   2 anos (Dota: 2025-2026): escolhe ajustando no 1o semestre do 1o ano e
#     conferindo no 2o semestre; ajuste final no 1o ano.
QUANTIS = 401
DESLOCAMENTOS = range(-5, 6)
ECE_MAXIMO = 0.02
REAMOSTRAS = 1000


def registros_ponto_no_tempo(tg):
    """[{gameid, ano, kx, ky, ex, ey}] -- expectativa so com o passado.

    Mesma conta de `medias_dc`, incremental: somas decaidas por time e do
    nivel geral, decaindo pela meia-vida ate a data do jogo.
    """
    import pandas as pd

    from mlcore.lines_engine import DC_BLEND_HL_DAYS, DC_BLEND_MIN_EFF
    hl, minimo = float(DC_BLEND_HL_DAYS), float(DC_BLEND_MIN_EFF)

    df = tg.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kf"] = pd.to_numeric(df["kills_for"], errors="coerce")
    df["ka"] = pd.to_numeric(df["kills_against"], errors="coerce")
    df = df.dropna(subset=["date", "kf", "ka"])
    df = df.sort_values(["date", "gameid", "team"], kind="mergesort")

    estado = {}            # time -> [soma_feitos, soma_sofridos, peso, data]
    geral = [0.0, 0.0, None]   # soma kills (por linha de time), peso, data
    fora = []

    def _decair(reg, quando, i_data):
        ant = reg[i_data]
        if ant is not None:
            f = 0.5 ** (max((quando - ant).total_seconds() / 86400.0, 0.0) / hl)
            for k in range(i_data):
                reg[k] *= f
        reg[i_data] = quando

    for gid, g in df.groupby("gameid", sort=False):
        if len(g) != 2:
            continue
        x, y = g.iloc[0], g.iloc[1]
        quando = x["date"]
        _decair(geral, quando, 2)
        tx, ty = str(x["team"]), str(y["team"])
        for t in (tx, ty):
            if t in estado:
                _decair(estado[t], quando, 3)
        if geral[1] >= minimo and geral[1] > 1e-9 and tx in estado and ty in estado:
            lm = geral[0] / geral[1]
            sx, sy = estado[tx], estado[ty]
            if min(sx[2], sy[2]) >= minimo and lm > 1e-9:
                att_x, def_x = sx[0] / sx[2], sx[1] / sx[2]
                att_y, def_y = sy[0] / sy[2], sy[1] / sy[2]
                fora.append({"gameid": str(gid), "ano": int(quando.year),
                             "sem": "%da" % quando.year if quando.month <= 6 else "%db" % quando.year,
                             "kx": float(x["kf"]), "ky": float(y["kf"]),
                             "ex": att_x * def_y / lm, "ey": att_y * def_x / lm})
        # so DEPOIS de prever, o jogo entra no historico
        for t, kf, ka in ((tx, float(x["kf"]), float(x["ka"])),
                          (ty, float(y["kf"]), float(y["ka"]))):
            s = estado.setdefault(t, [0.0, 0.0, 0.0, quando])
            s[0] += kf
            s[1] += ka
            s[2] += 1.0
        geral[0] += float(x["kf"]) + float(y["kf"])
        geral[1] += 2.0
    return fora


def _quantis(valores, n=QUANTIS):
    import numpy as np
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    return [float(q) for q in np.quantile(v, np.linspace(0.0, 1.0, n))]


def _reta_da_largura(totais, residuos):
    """a + b x total, pelo desvio medido direto em cinco faixas de total."""
    import numpy as np
    tot = np.asarray(totais, dtype=float)
    res = np.asarray(residuos, dtype=float)
    cortes = np.quantile(tot, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    pts = []
    for i in range(5):
        m = (tot >= cortes[i]) & ((tot <= cortes[i + 1]) if i == 4 else (tot < cortes[i + 1]))
        if int(m.sum()) >= 100:
            pts.append((float(tot[m].mean()), float(res[m].std())))
    if len(pts) < 2:
        return float(res.std()), 0.0
    b, a = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    return float(a), float(b)


def ajustar(regs, modo_h, modo_t, recalibrar=True):
    """O artefato (so os dois blocos) a partir de `regs`.

    `recalibrar`: mede no proprio treino a INCLINACAO da diferenca (o
    Dixon-Coles exagera: 0,75-0,84) e a reta dos kills por time (ele previa
    de menos em 2024-2025). Sem isso, a forma carregaria esses vicios
    escondidos -- e eles mudam de ano para ano.
    """
    import numpy as np

    from mlcore.abates_por_time import _escala
    D = np.array([r["kx"] - r["ky"] for r in regs], dtype=float)
    M = np.array([r["ex"] - r["ey"] for r in regs], dtype=float)
    h = {"modo": modo_h, "inclinacao": 1.0}
    if recalibrar and float((M * M).sum()) > 1e-9:
        # pela origem: simetrico, sem lado
        h["inclinacao"] = float((D * M).sum() / (M * M).sum())
    Mh = h["inclinacao"] * M
    if modo_h == "reta":
        h["a"], h["b"] = _reta_da_largura([r["ex"] + r["ey"] for r in regs], list(D - Mh))
    res_h = []
    for r, d, m in zip(regs, D, Mh):
        e = _escala(h, r["ex"] + r["ey"])
        z = (d - m) / e
        res_h += [z, -z]          # as DUAS direcoes: simetrica, sem lado
    h["quantis"] = _quantis(res_h)

    K = np.array([r["kx"] for r in regs] + [r["ky"] for r in regs], dtype=float)
    E = np.array([r["ex"] for r in regs] + [r["ey"] for r in regs], dtype=float)
    t = {"modo": modo_t, "centro": 0.0, "inclinacao": 1.0}
    if recalibrar:
        b, a = np.polyfit(E, K, 1)
        t["centro"], t["inclinacao"] = float(a), float(b)
    Et = t["centro"] + t["inclinacao"] * E
    res_t = []
    for k, e in zip(K, Et):
        sc = _escala(t, e)
        if sc:
            res_t.append((k - e) / sc)
    t["quantis"] = _quantis(res_t)
    return {"handicap": h, "total_time": t}


def _lado(gid):
    """Direcao fixa e sem vies para medir: metade dos jogos de cada jeito."""
    return int(hashlib.md5(gid.encode("utf-8")).hexdigest(), 16) % 2 == 0


def apostas(regs, art):
    """[(gameid, mercado, p_modelo, deu, linha)] nas linhas que a casa oferece.

    Handicap nas DUAS direcoes de cada jogo: sortear uma so poe ruido de
    lado na medicao (medido: +0,34 kill de vies so pelo sorteio em 2026). A
    incerteza continua agrupada por jogo.
    """
    from mlcore.abates_por_time import prob_cobre_handicap, prob_total_do_time_acima
    hreg, treg = art["handicap"], art["total_time"]
    fora = []
    for r in regs:
        for a, b in (("x", "y"), ("y", "x")):
            ka, kb, ea, eb = r["k" + a], r["k" + b], r["e" + a], r["e" + b]
            # A LINHA vem da media CRUA, igual para toda versao comparada:
            # se andasse com a media de cada versao, cada uma seria medida em
            # apostas diferentes e a escolha nao seria justa.
            base = math.floor(-(ea - eb)) + 0.5
            for j in DESLOCAMENTOS:
                L = base + j
                p = prob_cobre_handicap(ea, eb, L, art)
                if p is not None:
                    fora.append((r["gameid"], "handicap", p, 1.0 if ka + L > kb else 0.0, L))
        for k, e in ((r["kx"], r["ex"]), (r["ky"], r["ey"])):
            base_t = math.floor(e) + 0.5
            for j in DESLOCAMENTOS:
                L = base_t + j
                p = prob_total_do_time_acima(e, L, art)
                if p is not None:
                    fora.append((r["gameid"], "total_time", p, 1.0 if k > L else 0.0, L))
    return fora


def taxas_base(treino):
    """A TAXA-BASE POR LINHA DE VERDADE, so do treino: com que frequencia
    "cobrir -7,5" ou "mais de 15,5 kills" aconteceu, sem saber quem joga.

    A primeira versao usava a taxa por DISTANCIA ate a media do modelo -- o
    que ja carrega a informacao do modelo e zera o ganho por construcao.
    """
    import bisect
    difs = sorted([r["kx"] - r["ky"] for r in treino] + [r["ky"] - r["kx"] for r in treino])
    ks = sorted([r["kx"] for r in treino] + [r["ky"] for r in treino])

    def _h(L):     # P(D + L > 0) = P(D > -L)
        return 1.0 - bisect.bisect_right(difs, -L) / len(difs)

    def _t(L):
        return 1.0 - bisect.bisect_right(ks, L) / len(ks)
    return {"handicap": _h, "total_time": _t}


def medir(treino, teste, art, com_ic=True):
    """Por mercado: Brier do modelo, da taxa-base, ganho, IC 95% e ECE."""
    import numpy as np
    base = taxas_base(treino)
    fora = {}
    todas = apostas(teste, art)
    for merc in ("handicap", "total_time"):
        linhas = [(gid, p, y, min(max(base[merc](L), 1e-3), 1 - 1e-3))
                  for gid, m, p, y, L in todas if m == merc]
        if not linhas:
            continue
        gids = sorted({g for g, *_ in linhas})
        idx = {g: i for i, g in enumerate(gids)}
        dm = np.zeros(len(gids))
        db = np.zeros(len(gids))
        cont = np.zeros(len(gids))
        for g, p, y, tb in linhas:
            i = idx[g]
            dm[i] += (p - y) ** 2
            db[i] += (tb - y) ** 2
            cont[i] += 1
        bm, bb = dm.sum() / cont.sum(), db.sum() / cont.sum()
        reg = {"apostas": len(linhas), "jogos": len(gids), "brier_modelo": bm,
               "brier_taxa_base": bb, "ganho": bb - bm}
        if com_ic:
            rng = np.random.default_rng(20260924)
            gan = []
            for _ in range(REAMOSTRAS):
                s = rng.integers(0, len(gids), len(gids))
                gan.append(db[s].sum() / cont[s].sum() - dm[s].sum() / cont[s].sum())
            reg["ic95"] = (float(np.quantile(gan, 0.025)), float(np.quantile(gan, 0.975)))
        # calibracao CONDICIONAL: dentro de cada faixa de probabilidade
        faixas = [(0, .15), (.15, .25), (.25, .35), (.35, .45), (.45, .55),
                  (.55, .65), (.65, .75), (.75, .85), (.85, 1.01)]
        ece = 0.0
        tab = []
        for lo, hi in faixas:
            sub = [(p, y) for _, p, y, _ in linhas if lo <= p < hi]
            if not sub:
                continue
            pm = sum(p for p, _ in sub) / len(sub)
            ob = sum(y for _, y in sub) / len(sub)
            ece += len(sub) * abs(pm - ob)
            tab.append((lo, hi, len(sub), pm, ob))
        reg["ece"] = ece / len(linhas)
        reg["faixas"] = tab
        fora[merc] = reg
    return fora


def _csvs_padrao():
    """Os CSVs do repositorio: os do LoL pela funcao de sempre; no Dota, os
    `*_oe_like.csv` das duas pastas."""
    try:
        from tools.nota_honesta import default_csvs
        return [str(c) for c in default_csvs()]
    except Exception:
        return sorted(str(p) for pasta in ("CSV_HISTORICO", "CSV_DIARIO")
                      for p in (ROOT / pasta).glob("*_oe_like.csv"))


def periodos(anos):
    """(escolha_ajuste, escolha_confere, ajuste_final, cego) -- ver o topo."""
    anos = sorted(anos)
    if len(anos) >= 3:
        return ([str(anos[0])], [str(anos[1])], [str(a) for a in anos[:-1]], str(anos[-1]))
    if len(anos) == 2:
        return (["%da" % anos[0]], ["%db" % anos[0]], [str(anos[0])], str(anos[1]))
    raise SystemExit("historico curto demais para uma prova cega")


def _do(regs, rotulos):
    return [r for r in regs if str(r["ano"]) in rotulos or r["sem"] in rotulos]


def main() -> int:
    import pandas as pd

    from mlcore.stats_teams import build_team_games, load_team_rows

    gravar = "--sem-gravar" not in sys.argv
    csvs = _csvs_padrao()
    print("[abates] raiz:", ROOT)
    tg = pd.concat([build_team_games(load_team_rows(str(c))) for c in csvs],
                   ignore_index=True)
    regs = registros_ponto_no_tempo(tg)
    anos = sorted({r["ano"] for r in regs})
    esc_aj, esc_conf, aj_final, cego = periodos(anos)
    print("[abates] jogos previsiveis por ano:",
          {a: sum(1 for r in regs if r["ano"] == a) for a in anos})
    print("[abates] escolha: ajusta %s, confere %s | final: ajusta %s, CEGO %s"
          % (esc_aj, esc_conf, aj_final, cego))

    # 1) ESCOLHA do modo e da recalibracao. O ano cego fica fora de toda escolha.
    tr_esc, te_esc = _do(regs, esc_aj), _do(regs, esc_conf)
    melhor = {}
    for merc, modos in (("handicap", ("bruto", "reta")), ("total_time", ("bruto", "raiz"))):
        notas = {}
        for modo in modos:
            for recal in (False, True):
                art = ajustar(tr_esc, modo if merc == "handicap" else "bruto",
                              modo if merc == "total_time" else "bruto", recalibrar=recal)
                m = medir(tr_esc, te_esc, art, com_ic=False)
                notas[(modo, recal)] = m[merc]["brier_modelo"]
        melhor[merc] = min(notas, key=notas.get)
        print("[abates] %s: Brier na conferencia %s -> %s" % (
            merc, {"%s%s" % (k[0], "+recal" if k[1] else ""): round(float(v), 5)
                   for k, v in notas.items()}, melhor[merc]))

    # 2) AJUSTE final e PROVA cega.
    treino, teste = _do(regs, aj_final), _do(regs, [cego])
    art_h = ajustar(treino, melhor["handicap"][0], "bruto", recalibrar=melhor["handicap"][1])
    art_t = ajustar(treino, "bruto", melhor["total_time"][0], recalibrar=melhor["total_time"][1])
    art = {"handicap": art_h["handicap"], "total_time": art_t["total_time"]}
    res = medir(treino, teste, art)
    for merc, reg in res.items():
        lo, hi = reg["ic95"]
        ok = lo > 0 and reg["ece"] < ECE_MAXIMO
        print()
        print("== %s (cego %s): %d apostas em %d jogos" % (merc, cego, reg["apostas"], reg["jogos"]))
        print("   Brier modelo %.5f | taxa-base %.5f | ganho %+.5f  IC95 [%+.5f, %+.5f]"
              % (reg["brier_modelo"], reg["brier_taxa_base"], reg["ganho"], lo, hi))
        print("   calibracao condicional (ECE): %.2f pp  -> %s" % (100 * reg["ece"],
                                                               "PASSA" if ok else "NAO PASSA"))
        for lo_, hi_, n, pm, ob in reg["faixas"]:
            print("     %3.0f%%-%3.0f%%  %6d  modelo %5.1f%%  aconteceu %5.1f%%  %+5.1fpp"
                  % (100 * lo_, 100 * hi_, n, 100 * pm, 100 * ob, 100 * (pm - ob)))
        res[merc]["passou"] = ok
    # MERCADO POR MERCADO: grava so o que passou. O que nao passou fica
    # sem numero na tela, com o motivo -- nunca um preco reprovado.
    aprovados = [m for m, r in res.items() if r["passou"]]
    for m in list(art):
        if m not in aprovados:
            del art[m]
    if not aprovados:
        print("\n[abates] NADA passou. Nada gravado; a tela continua sem este preco.")
        return 1
    print("\n[abates] passaram: %s | nao passaram: %s" % (
        aprovados, [m for m in res if m not in aprovados]))
    if not gravar:
        print("[abates] (sem gravar, a pedido)")
        return 0
    art["_gerado_em"] = date.today().isoformat()
    art["_anos_de_ajuste"] = sorted({r["ano"] for r in treino})
    art["_periodos"] = {"escolha_ajuste": esc_aj, "escolha_confere": esc_conf,
                        "ajuste_final": aj_final, "cego": cego}
    art["_prova_cega"] = {m: {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in r.items() if k != "faixas"}
                          for m, r in res.items()}
    art["_o_que_e"] = ("quantis do residuo de kills (diferenca, espelhada; e por "
                       "time) em volta da media Dixon-Coles; mlcore/abates_por_time.py")
    alvo = ROOT / "abates_por_time_forma.json"
    # Os blocos COM A CHANCE DE VITORIA vem de outra regua
    # (tools/abates_com_vitoria.py) e de outro periodo: regravar os de sempre
    # nao pode apaga-los.
    if alvo.exists():
        try:
            velho = json.loads(alvo.read_text(encoding="utf-8"))
            for k, v in velho.items():
                if "com_vitoria" in k:
                    art[k] = v
        except ValueError:
            pass
    alvo.write_text(json.dumps(art, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n[abates] passou e gravou %s" % alvo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
