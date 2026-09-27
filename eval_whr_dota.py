# -*- coding: utf-8 -*-
"""Fase 1c — primeiro WHR do Dota (reusa mlcore/whr_core do LoL, que e agnostico).

Protocolo (adaptado a 1 semestre de dados — nao ha ano-cego):
- Replay prequencial: TODA predicao e feita ANTES de adicionar o jogo.
- TUNE  = partidas tier professional 2026-02-15..2026-04-30 (escolhe w2 e config).
- BLIND = partidas tier professional 2026-05-01..fim (so o veredito final).
- Eval exige ambos os times com >=5 jogos no fitter.
- Identidade de time = team_id do OpenDota (estavel a renames).

Perguntas que este eval responde com numero:
1. w2* do Dota (hipotese do usuario: momento importa mais -> w2* > 60 do LoL);
2. ligas "excluded" (Destiny/UDPL/...) ajudam como DADO DE TREINO? (56% das
   partidas professional tem time que tambem joga nelas — mas ha risco 322);
3. termo de side Radiant rolante (meia-vida 30d, prequencial) melhora?

Uso: python eval_whr_dota.py
"""
from __future__ import annotations

import json
import math
import sys

# Console cp1252 no Windows derruba qualquer print com acento/emoji — e o estouro
# vem no fim, depois do trabalho todo feito. Ver tools/find_silent_failures.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass
import time
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402

TIERS_CACHE = Path(__file__).with_name("league_tiers.csv")
TUNE_FROM, TUNE_TO = "2026-02-15", "2026-04-30"
BLIND_FROM = "2026-05-01"
MIN_GAMES_EVAL = 5
SIDE_HALFLIFE_DAYS = 30.0
SIDE_MIN_EFF = 50.0
SIDE_MAX_OFFSET = 0.4  # natural (logit); 0.4 ~ 60/40


# A OpenDota classifica liga em premium > professional > amateur. O TI e marcado
# como PREMIUM: ate 16/08/2026 o universo do produto era so "professional", e o
# torneio inteiro ficava FORA do modelo (TI 2025 144 partidas + TI 2026 109, este
# em andamento). Era o maior evento do ano invisivel pro modelo. Universo = os
# dois tiers de cima; amateur/excluded seguem fora.
PRO_TIERS = ("professional", "premium")

# O cache de tiers e uma foto: sem prazo de validade, liga NOVA fica "sem_tier" e
# some do universo calada (visto em 16/08: BLAST Slam VI China Qualifier e ESL
# challenger China caidos assim). Revalida por semana, com queda-pro-cache se a
# API falhar — dado velho e melhor que erro.
TIERS_CACHE_MAX_AGE_DAYS = 7.0


def is_pro_tier(s: pd.Series) -> pd.Series:
    """Mascara: a linha esta no universo profissional do produto?"""
    return s.astype(str).str.strip().str.lower().isin(PRO_TIERS)


def _fetch_tiers_df() -> pd.DataFrame:
    from opendota_client import get_leagues
    return pd.DataFrame(get_leagues()).rename(columns={"leagueid": "league_id"})[["league_id", "tier"]]


def load_tiers(ligas_usadas=None) -> dict[int, str]:
    """{league_id: tier}. Renova o cache se ele estiver velho OU se faltar
    nele alguma liga de `ligas_usadas`.

    POR QUE O SEGUNDO GATILHO (27/09/2026): o prazo de 7 dias olha a data do
    ARQUIVO, e no robo da nuvem o checkout do git cria o arquivo de novo a
    cada rodada -- ele sempre parecia novo e nunca renovou. Ficou parado em
    16/08 e toda liga criada depois virou "sem_tier" e sumiu do CSV calada:
    246 partidas profissionais (PGL Wallachia S9, qualificatorias da BLAST
    Slam...) e o CSV diario congelado em 10/09. Liga que aparece nas partidas
    e nao esta no cache e prova de que ele esta velho, sem depender de data.
    """
    df = None
    stale = True
    if TIERS_CACHE.exists():
        try:
            age_days = (time.time() - TIERS_CACHE.stat().st_mtime) / 86400.0
            stale = age_days > TIERS_CACHE_MAX_AGE_DAYS
        except OSError:
            stale = True
        try:
            df = pd.read_csv(TIERS_CACHE)
        except (OSError, ValueError):
            df = None
    if df is not None and not stale and ligas_usadas is not None:
        try:
            conhecidas = set(pd.to_numeric(df["league_id"], errors="coerce")
                             .dropna().astype(int))
            faltando = {int(x) for x in ligas_usadas if int(x) > 0} - conhecidas
            stale = bool(faltando)
        except (KeyError, TypeError, ValueError):
            stale = True
    if df is None or stale:
        try:
            fresh = _fetch_tiers_df()
            if not fresh.empty:
                fresh.to_csv(TIERS_CACHE, index=False)
                df = fresh
        except Exception:
            if df is None:
                raise
    out = {}
    for r in df.itertuples(index=False):
        try:
            out[int(r.league_id)] = str(r.tier)
        except Exception:
            continue
    return out


def canonical_team_names(m: pd.DataFrame,
                         rep_por_id: dict[int, int] | None = None,
                         api_names: dict[int, str] | None = None) -> dict[int, str]:
    """team_id -> nome canonico. A IDENTIDADE do time e o team_id, nao o nome.

    Achado 16/08/2026, no universo que alimenta o modelo (professional+premium):
      - 85 team_id tem MAIS DE UM nome (a org trocou de nome). Sem unificar, a
        historia se parte e um time de ponta vira "desconhecido":
        BB=BB Team=BetBoom Team=BoomBoys sao 520 jogos que o modelo via como 78;
        PARIVISION=PVISION=TEAM VISION sao 409 que ele via como 50. Tem ate caso
        de so a caixa mudar (Business Club) e de homoglifo cirilico (EYE Gaming).
      - 155 NOMES sao reusados por varios team_id — nome generico/placeholder
        (Elite Eclipse 32 ids, Dominion 31, Shinigami Gaming 19). Sem separar, o
        modelo soma 19 times diferentes num rating so.
    Chavear por team_id conserta os dois lados de uma vez.

    Canonico = o nome MAIS RECENTE daquele id (a tela mostra como o time se chama
    hoje). Quando dois ids terminam com o mesmo nome, o de mais jogos fica com o
    nome limpo e os outros recebem sufixo — senao a colisao desfaria a separacao.

    NOTA para quem mexer: o canonico muda quando a org se renomeia, entao a chave
    do modelo muda junto. Isso e seguro porque a esteira do Dota regenera o CSV
    inteiro e retreina todo dia; num fluxo incremental seria preciso um mapa de
    apelidos persistente.
    """
    ultimo: dict[tuple[int, str], float] = {}
    jogos: dict[int, int] = {}
    for id_col, nome_col in (("tid", "teamname"), ("oid", "opp_teamname")):
        sub = m[[id_col, nome_col, "ord_key"]].dropna()
        for r in sub.itertuples(index=False):
            i = int(r[0])
            n = str(r[1]).strip()
            if i <= 0 or not n:
                continue
            jogos[i] = jogos.get(i, 0) + 1
            k = (i, n)
            t = float(r[2])
            if t > ultimo.get(k, float("-inf")):
                ultimo[k] = t

    melhor: dict[int, tuple[float, str]] = {}
    for (i, n), t in ultimo.items():
        if t > melhor.get(i, (float("-inf"), ""))[0]:
            melhor[i] = (t, n)
    canon = {i: v[1] for i, v in melhor.items()}

    # NOME DO GRUPO = o mais RECENTE do grupo inteiro, nao o do id representante.
    # Bug visto por ele em 16/08/2026 olhando a casa de apostas: o time do TI
    # aparecia como "Tundra Esports" (nome do id antigo, que tinha mais jogos e
    # por isso virou representante) enquanto a OpenDota e o proprio app diziam
    # "Iron Wing". Quem manda e o nome ATUAL — e a API confirma: o id 10150413
    # se chama "Iron Wing" (tag IW) e tem exatamente o elenco da Tundra.
    if rep_por_id:
        recente_do_grupo: dict[int, tuple[float, str]] = {}
        for i, (t, n) in melhor.items():
            r = rep_por_id.get(i, i)
            if t > recente_do_grupo.get(r, (float("-inf"), ""))[0]:
                recente_do_grupo[r] = (t, n)
        for i in list(canon):
            r = rep_por_id.get(i, i)
            if r in recente_do_grupo:
                canon[i] = recente_do_grupo[r][1]

    # Colisao e por GRUPO, nunca por id: depois da unificacao acima todos os ids
    # do mesmo time dividem o nome de proposito, e tratar isso como colisao
    # colocaria sufixo e DESFARIA a uniao (Iron Wing e Tundra voltariam a ser dois).
    def _grupo(i: int) -> int:
        return (rep_por_id or {}).get(i, i)

    # NOME PELA API vence o derivado das partidas: e o nome de HOJE, enquanto o
    # das partidas so muda quando o time volta a jogar. Aplicado ao grupo inteiro
    # a partir do id ATIVO (o aposentado ainda devolve o nome antigo).
    if api_names:
        melhor_api: dict[int, tuple[float, str]] = {}
        for i, (t, _n) in melhor.items():
            nm = str(api_names.get(i) or "").strip()
            if not nm:
                continue
            r = (rep_por_id or {}).get(i, i)
            if t > melhor_api.get(r, (float("-inf"), ""))[0]:
                melhor_api[r] = (t, nm)
        for i in list(canon):
            r = (rep_por_id or {}).get(i, i)
            if r in melhor_api:
                canon[i] = melhor_api[r][1]

    donos: dict[str, set[int]] = {}
    for i, n in canon.items():
        donos.setdefault(n, set()).add(_grupo(i))
    for n, grupos in donos.items():
        if len(grupos) < 2:
            continue
        jogos_do_grupo = {g: sum(jogos.get(i, 0) for i in canon if _grupo(i) == g)
                          for g in grupos}
        for g in sorted(grupos, key=lambda x: -jogos_do_grupo.get(x, 0))[1:]:
            for i in [k for k in canon if _grupo(k) == g]:
                canon[i] = f"{n} #{g}"
    return canon


# ---------------------------------------------------------------------
# IDENTIDADE PELA API (ideia dele, 16/08/2026)
# ---------------------------------------------------------------------
# A OpenDota responde direto duas coisas que eu vinha DEDUZINDO dos jogos:
#   /teams/{id}          -> o nome ATUAL do time
#   /teams/{id}/players  -> is_current_team_member = o elenco de HOJE
# Isso e melhor que inferir do ultimo jogo, que fica velho ate o time jogar.
#
# Duas restricoes medidas antes de desenhar:
#  - id APOSENTADO devolve elenco parcial (a Tundra antiga volta com 2 de 5),
#    entao a API vale para o lado ATIVO do vinculo; o historico continua vindo
#    das partidas.
#  - sao 2415 ids no universo, mas so 245 jogaram nos ultimos 90 dias. Consultar
#    so os ativos cabe folgado na cota (a lista inteira nao caberia).
TEAM_API_CACHE = Path(__file__).with_name("team_api_cache.json")
TEAM_API_MAX_AGE_DAYS = 7.0
API_JANELA_ATIVOS_DIAS = 90


def _api_team_fetch(tid: int) -> dict | None:
    from opendota_client import _request
    t = _request(f"teams/{tid}")
    if not isinstance(t, dict):
        return None
    pl = _request(f"teams/{tid}/players")
    atuais = [int(p["account_id"]) for p in (pl or [])
              if isinstance(p, dict) and p.get("is_current_team_member") and p.get("account_id")]         if isinstance(pl, list) else []
    return {"name": str(t.get("name") or "").strip(), "roster": sorted(atuais)}


def team_api_info(ids: set[int], *, permitir_rede: bool = True) -> dict[int, dict]:
    """{team_id: {"name":..., "roster":[account_id]}} para os ids pedidos.

    Cache em disco com validade de 7 dias. Se a rede falhar, devolve o que tem —
    identidade com dado velho e melhor que identidade nenhuma, e o chamador tem
    a inferencia por partidas como rede de seguranca.
    """
    cache: dict[str, dict] = {}
    if TEAM_API_CACHE.exists():
        try:
            velho = float(time.time() - TEAM_API_CACHE.stat().st_mtime) / 86400.0
            cache = json.loads(TEAM_API_CACHE.read_text(encoding="utf-8"))
            if velho > TEAM_API_MAX_AGE_DAYS:
                cache = {}          # foto vencida: refaz
        except (OSError, ValueError):
            cache = {}
    faltando = [i for i in sorted(ids) if str(i) not in cache]
    if faltando and permitir_rede:
        for i in faltando:
            try:
                d = _api_team_fetch(int(i))
            except Exception:
                d = None
            if d is not None:
                cache[str(i)] = d
        try:
            TEAM_API_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return {int(k): v for k, v in cache.items() if int(k) in ids}


def ids_ativos(m: pd.DataFrame, dias: int = API_JANELA_ATIVOS_DIAS) -> set[int]:
    """Quem jogou na janela recente — o unico recorte que vale consultar."""
    if m.empty or "ord_key" not in m.columns:
        return set()
    corte = float(m["ord_key"].max()) - dias * 86400.0
    rec = m[m["ord_key"] >= corte]
    return {int(i) for i in set(rec["tid"]) | set(rec["oid"]) if int(i) > 0}


ROSTER_MIN_COMUM = 5      # elenco IDENTICO; com 4 entravam erros (HEROIC+LGD Gaming)
ROSTER_ULTIMOS_N = 20     # "elenco recente" = titulares dos ultimos N jogos do id
ROSTER_FOLGA_DIAS = 21    # sobreposicao tolerada entre as janelas de atividade


def _rosters_por_team_id(caminho: Path) -> dict[int, list[tuple[float, frozenset[int]]]]:
    """team_id -> [(instante, contas do time naquele jogo)], cronologico."""
    por_id: dict[int, list[tuple[float, frozenset[int]]]] = {}
    try:
        fh = open(caminho, encoding="utf-8")
    except OSError:
        return por_id
    with fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            st = float(d.get("start_time") or 0)
            jog = d.get("players") or []
            for tid_raw, is_rad in ((d.get("radiant_team_id"), True),
                                    (d.get("dire_team_id"), False)):
                try:
                    tid = int(tid_raw or 0)
                except (TypeError, ValueError):
                    continue
                if tid <= 0:
                    continue
                accs = {int(p["account_id"]) for p in jog
                        if p.get("account_id") and (p.get("isRadiant") is is_rad)}
                if accs:
                    por_id.setdefault(tid, []).append((st, frozenset(accs)))
    for v in por_id.values():
        v.sort()
    return por_id


def roster_identity_map(m: pd.DataFrame, *, raw_path: Path | None = None,
                        api_rosters: dict[int, list[int]] | None = None) -> dict[int, int]:
    """team_id -> team_id REPRESENTANTE, unindo ids que sao o mesmo elenco.

    CAMADA 2 da identidade (16/08/2026). A camada 1 (team_id) conserta rebrand,
    mas quando a organizacao se RE-REGISTRA ela ganha id novo e a historia parte
    de novo. Medido: Iron Wing (40 jogos, e o time do TI que aparecia com ZERO no
    modelo) e a Tundra Esports (478) — elenco identico. HULIGANI (31) e L1GA TEAM
    (358). Nigma Galaxy tem duas encarnacoes com janelas contiguas.

    Tres guardas, todos aprendidos medindo:
      1) elenco IDENTICO (5 de 5). Com 4 entravam erros reais — HEROIC + LGD
         Gaming dividiam 4 contas e sao organizacoes diferentes.
      2) sem atividade simultanea: dois ids que jogam na mesma janela nao podem
         ser o mesmo time (poderiam ate se enfrentar).
      3) transitividade nao atropela o guarda 2: se A~B e A~C mas B e C jogavam
         ao mesmo tempo, a uniao para — senao o encadeamento junta rivais.
    """
    if raw_path is None:
        raw_path = Path(__file__).with_name("raw_matches.jsonl")

    jogos: dict[int, int] = {}
    janela: dict[int, list[float]] = {}
    for r in m.itertuples(index=False):
        for i in (int(getattr(r, "tid", 0)), int(getattr(r, "oid", 0))):
            if i <= 0:
                continue
            jogos[i] = jogos.get(i, 0) + 1
            k = float(r.ord_key)
            w = janela.get(i)
            janela[i] = [min(w[0], k), max(w[1], k)] if w else [k, k]

    # ELENCO ATUAL PELA API tem prioridade sobre o inferido: ele diz quem joga
    # HOJE, enquanto a inferencia olha os ultimos 20 jogos e envelhece junto com
    # o time. Vale so para o lado ATIVO — id aposentado devolve elenco parcial.
    api = api_rosters or {}
    recente: dict[int, set[int]] = {}
    for i, lst in _rosters_por_team_id(raw_path).items():
        if i not in jogos:
            continue
        cont: dict[int, int] = {}
        for _, s in lst[-ROSTER_ULTIMOS_N:]:
            for a in s:
                cont[a] = cont.get(a, 0) + 1
        do_api = api.get(i)
        if do_api and len(do_api) == 5:
            recente[i] = set(do_api)          # a API sabe melhor
        else:
            recente[i] = {a for a, _ in sorted(cont.items(), key=lambda kv: -kv[1])[:5]}

    folga = ROSTER_FOLGA_DIAS * 86400.0

    def simultaneos(a: int, b: int) -> bool:
        wa, wb = janela.get(a), janela.get(b)
        if not wa or not wb:
            return True
        return (min(wa[1], wb[1]) - max(wa[0], wb[0])) > folga

    por_jog: dict[int, set[int]] = {}
    for i, s in recente.items():
        for a in s:
            por_jog.setdefault(a, set()).add(i)

    pares: set[tuple[int, int]] = set()
    for ids in por_jog.values():
        if len(ids) < 2 or len(ids) > 12:   # conta em 13+ ids = stand-in/ruido
            continue
        ids_l = sorted(ids)
        for x in range(len(ids_l)):
            for y in range(x + 1, len(ids_l)):
                pares.add((ids_l[x], ids_l[y]))

    cand = []
    for a, b in pares:
        if len(recente.get(a, ()) & recente.get(b, ())) < ROSTER_MIN_COMUM:
            continue
        if simultaneos(a, b):
            continue
        cand.append((jogos.get(a, 0) + jogos.get(b, 0), a, b))
    cand.sort(reverse=True)   # une primeiro o que tem mais historia em jogo

    grupo: dict[int, set[int]] = {}
    pai: dict[int, int] = {}

    def raiz(i: int) -> int:
        return pai.get(i, i)

    for _, a, b in cand:
        ra, rb = raiz(a), raiz(b)
        if ra == rb:
            continue
        ga = grupo.get(ra, {ra})
        gb = grupo.get(rb, {rb})
        # guarda 3: nenhuma dupla entre os grupos pode ser simultanea
        if any(simultaneos(x, y) for x in ga for y in gb):
            continue
        novo = ga | gb
        # representante = o id com mais jogos (carrega a historia maior)
        rep = max(novo, key=lambda i: jogos.get(i, 0))
        for i in novo:
            pai[i] = rep
        grupo[rep] = novo
        for i in (ra, rb):
            if i != rep:
                grupo.pop(i, None)
    return {i: raiz(i) for i in jogos}


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).with_name("data.csv"), low_memory=False)
    m = df[df["side"] == "radiant"].drop_duplicates("match_id").copy()
    m["dt"] = pd.to_datetime(m["date"], errors="coerce", utc=True)
    m["st"] = pd.to_numeric(m["start_time"], errors="coerce")
    # ordena por start_time quando existe (ordem real do dia); senao pela data
    m["ord_key"] = m["st"].fillna(m["dt"].astype("int64") // 10**9)
    m["tid"] = pd.to_numeric(m["team_id"], errors="coerce").fillna(0).astype(int)
    m["oid"] = pd.to_numeric(m["opp_team_id"], errors="coerce").fillna(0).astype(int)
    m["win"] = pd.to_numeric(m["result"], errors="coerce").fillna(0).astype(int)
    m["league_id"] = pd.to_numeric(m["league_id"], errors="coerce").fillna(0).astype(int)
    m = m[(m["tid"] > 0) & (m["oid"] > 0) & m["dt"].notna()]
    tiers = load_tiers(ligas_usadas=m["league_id"].unique())
    m["tier"] = m["league_id"].map(tiers).fillna("sem_tier")
    m["day"] = m["dt"].dt.date.map(lambda d: d.toordinal())
    m = m.sort_values(["ord_key", "match_id"]).reset_index(drop=True)
    return _aplicar_identidade(m)


_IDENT_CACHE: dict[tuple, dict[int, str]] = {}


def _aplicar_identidade(m: pd.DataFrame) -> pd.DataFrame:
    """Reescreve teamname/opp_teamname com a identidade canonica do time.

    AQUI e nao no dota_adapter (conserto 16/08/2026): aplicar so no adaptador
    deixava o CSV do modelo dizendo "Tundra Esports" enquanto o motor do app
    (dota_engine, que le direto daqui) dizia "Iron Wing" — dois nomes para o mesmo
    time em duas telas, e o modelo chaveado por um deles. Fazendo no load_matches,
    TODO consumidor ve a mesma identidade.

    DOTA_TEAM_IDENTITY=0 volta ao nome cru (escape hatch e uso do gate).
    """
    import os
    if os.environ.get("DOTA_TEAM_IDENTITY", "1") == "0" or m.empty:
        return m
    chave = (len(m), int(m["match_id"].iloc[-1]) if "match_id" in m.columns else 0)
    canon = _IDENT_CACHE.get(chave)
    if canon is None:
        # A API responde nome e elenco ATUAIS; consulta so os ~245 ids que
        # jogaram nos ultimos 90 dias (o universo tem 2415 e nao caberia na cota).
        # DOTA_IDENTITY_API=0 desliga a rede e cai na inferencia por partidas.
        api_nomes: dict[int, str] = {}
        api_elencos: dict[int, list[int]] = {}
        if os.environ.get("DOTA_IDENTITY_API", "1") != "0":
            try:
                info = team_api_info(ids_ativos(m))
                api_nomes = {i: d.get("name", "") for i, d in info.items() if d.get("name")}
                api_elencos = {i: d.get("roster", []) for i, d in info.items() if d.get("roster")}
            except Exception:
                api_nomes, api_elencos = {}, {}
        rep = {}
        if os.environ.get("DOTA_ROSTER_IDENTITY", "1") != "0":
            try:
                rep = roster_identity_map(m, api_rosters=api_elencos)
            except (OSError, ValueError, KeyError):
                rep = {}
        canon = canonical_team_names(m, rep, api_names=api_nomes)
        canon = {i: canon.get(rep.get(i, i), canon.get(i, "")) for i in canon}
        _IDENT_CACHE.clear()          # so a leitura corrente interessa
        _IDENT_CACHE[chave] = canon
    out = m.copy()
    out["teamname"] = [canon.get(int(i), str(n)) or str(n)
                       for i, n in zip(out["tid"], out["teamname"])]
    out["opp_teamname"] = [canon.get(int(i), str(n)) or str(n)
                           for i, n in zip(out["oid"], out["opp_teamname"])]
    return out


class SideTracker:
    """Taxa Radiant decaida (meia-vida em dias), atualizada prequencialmente."""

    def __init__(self, halflife_days: float = SIDE_HALFLIFE_DAYS):
        self.hl = float(halflife_days)
        self.sw = 0.0
        self.swy = 0.0
        self.last_day: int | None = None

    def _decay_to(self, day: int) -> None:
        if self.last_day is not None and day > self.last_day:
            f = 0.5 ** ((day - self.last_day) / self.hl)
            self.sw *= f
            self.swy *= f
        self.last_day = day if self.last_day is None else max(self.last_day, day)

    def offset_nat(self, day: int) -> float:
        self._decay_to(day)
        if self.sw < SIDE_MIN_EFF:
            return 0.0
        p = min(max(self.swy / self.sw, 1e-6), 1 - 1e-6)
        off = math.log(p / (1 - p))
        return min(max(off, -SIDE_MAX_OFFSET), SIDE_MAX_OFFSET)

    def add(self, day: int, radiant_won: int) -> None:
        self._decay_to(day)
        self.sw += 1.0
        self.swy += float(radiant_won)


def run(m: pd.DataFrame, *, w2: float, use_excluded: bool, use_side: bool,
        score_from: str, score_to: str | None, full_every: int = 500) -> dict:
    fitter = WHRFitter(w2_elo_per_day=w2)
    side = SideTracker()
    t_from = pd.Timestamp(score_from, tz="UTC")
    t_to = pd.Timestamp(score_to, tz="UTC") if score_to else None
    ll = br = 0.0
    n = hits = 0
    count = 0
    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        is_prof = r.tier == "professional"
        in_window = (r.dt >= t_from) and (t_to is None or r.dt <= t_to)
        if is_prof and in_window:
            ga = len(fitter.games.get(a, ()))
            gb = len(fitter.games.get(b, ()))
            if ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL:
                logit = (fitter.rating_nat(a) - fitter.rating_nat(b))
                if use_side:
                    logit += side.offset_nat(int(r.day))
                p = 1.0 / (1.0 + math.exp(-logit))
                pc = min(max(p, 1e-6), 1 - 1e-6)
                y = int(r.win)
                ll += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
                br += (p - y) ** 2
                hits += 1 if ((p >= 0.5) == (y == 1)) else 0
                n += 1
        if r.tier == "excluded" and not use_excluded:
            continue
        fitter.add_game(a, b, int(r.win), int(r.day))
        fitter.refresh_pair(a, b, iters=2)
        if is_prof:
            side.add(int(r.day), int(r.win))
        count += 1
        if count % full_every == 0:
            fitter.full_sweeps(sweeps=1, iters=1)
    return {"logloss": ll / n if n else float("nan"),
            "brier": br / n if n else float("nan"),
            "acc": hits / n if n else float("nan"), "n": n}


def main() -> int:
    m = load_matches()
    n_prof = int((m["tier"] == "professional").sum())
    print(f"[dota-whr] partidas={len(m)} (professional={n_prof}) "
          f"janela {m['dt'].min():%Y-%m-%d}..{m['dt'].max():%Y-%m-%d}", flush=True)

    print("\n[1] varredura de w2 no TUNE (excluded ON, side OFF):", flush=True)
    best_w2, best_ll = None, float("inf")
    for w2 in (20.0, 60.0, 120.0, 250.0, 500.0):
        t0 = time.monotonic()
        res = run(m, w2=w2, use_excluded=True, use_side=False,
                  score_from=TUNE_FROM, score_to=TUNE_TO)
        print(f"  w2={w2:>5.0f} -> ll={res['logloss']:.4f} acc={res['acc']*100:.1f}% "
              f"n={res['n']} ({time.monotonic()-t0:.0f}s)", flush=True)
        if res["logloss"] < best_ll:
            best_ll, best_w2 = res["logloss"], w2
    print(f"  w2* = {best_w2:.0f} (LoL usa 60 — hipotese do usuario: Dota mais alto)", flush=True)

    print(f"\n[2] grade 2x2 no TUNE com w2*={best_w2:.0f}:", flush=True)
    grid = {}
    for use_excluded in (True, False):
        for use_side in (True, False):
            key = f"excl={'on ' if use_excluded else 'off'} side={'on ' if use_side else 'off'}"
            res = run(m, w2=float(best_w2), use_excluded=use_excluded, use_side=use_side,
                      score_from=TUNE_FROM, score_to=TUNE_TO)
            grid[(use_excluded, use_side)] = res
            print(f"  {key} -> ll={res['logloss']:.4f} acc={res['acc']*100:.1f}% n={res['n']}", flush=True)
    best_cfg = min(grid, key=lambda k: grid[k]["logloss"])
    print(f"  config* = excluded={'ON' if best_cfg[0] else 'OFF'} side={'ON' if best_cfg[1] else 'OFF'}", flush=True)

    print(f"\n[3] BLIND (>= {BLIND_FROM}), config* vs baseline (excl ON, side OFF, w2=60):", flush=True)
    blind_best = run(m, w2=float(best_w2), use_excluded=best_cfg[0], use_side=best_cfg[1],
                     score_from=BLIND_FROM, score_to=None)
    blind_base = run(m, w2=60.0, use_excluded=True, use_side=False,
                     score_from=BLIND_FROM, score_to=None)
    for label, res in (("config*", blind_best), ("baseline", blind_base)):
        print(f"  {label:9} ll={res['logloss']:.4f} brier={res['brier']:.4f} "
              f"acc={res['acc']*100:.1f}% n={res['n']}", flush=True)
    dif = blind_base["logloss"] - blind_best["logloss"]
    print(f"  dif ll (base - config*) = {dif:+.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
