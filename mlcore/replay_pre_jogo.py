# -*- coding: utf-8 -*-
"""REPLAY DO PRÉ-JOGO — uma fonte só para medir o número que a tela serve.

Motivo (26/08/2026): existem TRÊS implementações da probabilidade pré-jogo,
com contagem de etapas diferente, e a nota oficial mede a mais simples. A
causa raiz é que o caminho servido não é chamável fora da interface
(`run_ml_pre_engine_layers` recebe `st`, formatadores e callbacks), então quem
precisa medir REIMPLEMENTA — e cada um para num ponto diferente.

Este módulo existe para a medição parar de reescrever o caminho. Não altera
nada em produção: é leitura.

DUAS DECISÕES DE DESENHO QUE IMPORTAM:

1) O CACHE GUARDA O DADO CRU, não as probabilidades.
   Montar as amostras leva ~20 min (lineup cache sobre 28 mil jogos); montar
   as probabilidades a partir delas leva segundos. Guardando o cru, dá para
   testar QUALQUER variante do caminho sem repagar a carga.

2) CADA VARIANTE REAJUSTA OS PRÓPRIOS CALIBRADORES.
   Sem isso a ablação seria fraudada a favor do status quo: remover uma etapa
   e manter um Platt que foi ajustado COM ela presente compara a etapa contra
   uma versão sabotada de si mesma.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .artifact import MLArtifact
from .calibration import fit_platt_from_raw_probs
from .calibration_diagnostics import _apply_microseg_calibration, _read_json_file_safe
from .fit_ml_params import _build_lineup_cache, _fuse_probs_by_precision, _make_samples
from .players_layer import PlayersArtifact, load_match_player_rows
from .utils import clip_prob, logit, sigmoid

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "evidence" / "_cache_pre_jogo.json"

# VERSAO do formato do cache. Subir isto invalida todo cache antigo de uma vez
# — util quando a FORMA do que se guarda muda, nao so as entradas.
CACHE_VERSAO = 1


def _assinatura_das_entradas(*, artifact, players_artifact, csv_train,
                             csv_current, params, val_cutoff) -> str:
    """Digest das ENTRADAS que geraram o cache (divida D9, 06/09/2026).

    Antes, `carregar()` devolvia o cache se o ARQUIVO EXISTISSE, sem conferir
    nada. O cache era de 26/08 23:34 e o robo atualizou o ml_artifact.json
    todo dia ate 06/09 — entao o gate deu numeros IDENTICOS depois de trocar
    o modelo, ate a quinta casa. Modelo diferente, nota igual, por 11 dias.

    A ideia vem de cache endereçado por ENTRADA: a chave e um hash das
    entradas da acao, e "qualquer modificacao nessas entradas produz uma chave
    diferente, tornando o resultado antigo inacessivel sob a chave nova e
    portanto impossivel de ser reusado por engano". Assim nao existe
    "esquecer de invalidar": mudou a entrada, o cache erra sozinho.
    https://blog.engflow.com/2024/05/13/the-many-caches-of-bazel/
    (conferido abrindo a pagina em 06/09/2026)

    Usa tamanho+mtime em vez de hash do conteudo: e o mesmo criterio que o
    resto do app ja aplica em `_file_sig`, e ler ~300 MB de CSV a cada chamada
    so para decidir se pode usar o cache anularia o proposito do cache.
    CONSEQUENCIA ACEITA: arquivo reescrito com mesmo tamanho E mesmo mtime
    passa despercebido. Para forcar, apague o arquivo de cache.
    """
    partes = [f"v{CACHE_VERSAO}", f"corte={val_cutoff}"]
    for rotulo, caminho in (("artifact", artifact),
                            ("players", players_artifact),
                            ("params", params)):
        f = ROOT / str(caminho)
        try:
            st = f.stat()
            partes.append(f"{rotulo}={f.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            partes.append(f"{rotulo}=AUSENTE")
    for rotulo, pasta in (("csv_train", csv_train), ("csv_current", csv_current)):
        base = ROOT / str(pasta)
        itens = []
        if base.is_dir():
            for x in sorted(base.glob("*.csv")):
                try:
                    st = x.stat()
                    itens.append(f"{x.name}:{st.st_size}:{int(st.st_mtime)}")
                except OSError:
                    itens.append(f"{x.name}:ILEGIVEL")
        partes.append(f"{rotulo}=[{','.join(itens)}]")
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()[:32]


@dataclass
class Amostra:
    """O dado CRU de um jogo — tudo que o caminho servido consome."""
    gameid: str
    league: str
    split: str
    side: str
    bo: str
    phase: str
    month: str
    y: int
    p_team: float
    p_players: float
    n_team: float
    n_players: float
    delta_eff: float


def _ms(p: float, a: Amostra, tabela: dict) -> float:
    """Microssegmento. Os argumentos sao NOMEADOS na funcao original — passar a
    amostra posicionalmente levanta TypeError, que um `except` largo engoliria,
    fazendo a etapa nao fazer nada CALADA. Aconteceu comigo em 26/08: eu teria
    medido "o microssegmento nao ajuda" sem ele nunca ter sido aplicado."""
    if not tabela:
        return float(p)
    return float(_apply_microseg_calibration(
        float(p), league=a.league, split=a.split, side=a.side,
        bo=a.bo, phase=a.phase, table=tabela))


def cadeia(amostras: list[Amostra], *, tabela: dict, platt_time=None,
           platt_fusao=None, usar_ms1=True, usar_platt_time=True,
           usar_fusao=True, usar_ms2=False, usar_platt_fusao=False) -> list[float]:
    """Roda o caminho servido com as etapas que forem pedidas.

    OS DEFAULTS SAO OS DE PRODUCAO (corrigido em 08/09/2026, divida D8).

    Ate 07/09 `usar_ms2` e `usar_platt_fusao` vinham True, e o docstring ja
    dizia "roda o caminho servido" — mas o APP NAO RODA essas duas etapas: elas
    dependem de callbacks que nunca sao injetados no caminho servido (o
    contrato MlPreEngineInput nem tem os campos). Quem escrevia codigo novo
    usando os defaults media uma cadeia que nao existe, e isso derrubou TRES
    chamadores: o gate (D8), a sombra, e a auditoria do pre-jogo (D5).

    A cura e a mesma que a fonte da divergencia treino/servico prescreve —
    "garanta que treino e producao se imitem o mais perto possivel". Aqui isso
    vira: o default e producao, e quem quer as etapas mortas PEDE.
    https://developers.google.com/machine-learning/crash-course/production-ml-systems/monitoring
    (conferida abrindo a pagina em 06/09/2026)

    Para ABLACAO, passe usar_ms2=True / usar_platt_fusao=True de proposito."""
    saida = []
    for a in amostras:
        p = float(a.p_team)
        if usar_ms1:
            p = _ms(p, a, tabela)
        if usar_platt_time and platt_time is not None:
            p = float(clip_prob(platt_time.predict_proba(p)))
        if usar_fusao:
            p = float(clip_prob(_fuse_probs_by_precision(
                p_team=p, p_players=float(a.p_players), n_team=float(a.n_team),
                n_players=float(a.n_players), team_scale=1.0, players_scale=1.0)))
        if usar_ms2:
            p = _ms(p, a, tabela)
        if usar_platt_fusao and platt_fusao is not None:
            p = float(clip_prob(platt_fusao.predict_proba(p)))
        saida.append(p)
    return saida


def variante(treino: list[Amostra], alvo: list[Amostra], *, tabela: dict,
             treino_final: list[Amostra] | None = None,
             familia_final: str = "platt", **chaves) -> list[float]:
    """Uma variante do caminho, com os calibradores REAJUSTADOS para ela.

    E o que torna a ablacao justa: cada versao do pipeline e avaliada com os
    calibradores que ELA teria, nao com os da versao completa."""
    y_tr = [a.y for a in treino]
    pt = None
    if chaves.get("usar_platt_time", True):
        base = cadeia(treino, tabela=tabela, usar_ms1=chaves.get("usar_ms1", True),
                      usar_platt_time=False, usar_fusao=False, usar_ms2=False,
                      usar_platt_fusao=False)
        pt = fit_platt_from_raw_probs(y_tr, base)
    pf = None
    if chaves.get("usar_platt_fusao", False):
        # O calibrador FINAL pode ser ajustado noutro conjunto: a distorcao que
        # ele corrige e da temporada, e o treino sao temporadas ANTIGAS. Ajustar
        # no val (recente) testa se o problema e defasagem em vez de metodo.
        fonte = treino_final if treino_final else treino
        base = cadeia(fonte, tabela=tabela, platt_time=pt,
                      usar_ms1=chaves.get("usar_ms1", True),
                      usar_platt_time=chaves.get("usar_platt_time", True),
                      usar_fusao=chaves.get("usar_fusao", True),
                      usar_ms2=chaves.get("usar_ms2", False),
                      usar_platt_fusao=False)
        y_fin = [a.y for a in (treino_final if treino_final else treino)]
        if familia_final == "beta":
            from .calibration import fit_beta_from_raw_probs
            pf = fit_beta_from_raw_probs(y_fin, base)
        else:
            pf = fit_platt_from_raw_probs(y_fin, base)
    return cadeia(alvo, tabela=tabela, platt_time=pt, platt_fusao=pf, **chaves)


def p_regua(amostras: list[Amostra], w_lineup: float) -> list[float]:
    """O que a nota oficial mede: sigmoid(logit(p_elo) + w_lineup * delta).

    CUIDADO — DEFEITO CONHECIDO, achado em 08/09/2026 (divida D5):
    `delta_eff` chega ZERO em 100% dos jogos. A causa e que `Amostra` monta o
    campo com `getattr(s, "delta_players_eff", 0.0)` e a classe `Sample`
    (mlcore/fit_samples_core.py) NAO TEM campo de delta nenhum — o delta e
    calculado durante a montagem e descartado. O `getattr` com default devolve
    0.0 calado, para sempre.

    Consequencia: com delta = 0, esta funcao devolve o ELO SOZINHO, nao
    "Elo + jogadores". Conferido: a saida bate com `p_team` em 1,11e-16.
    Entao a "divergencia regua x servido" de 3,36 pp que a divida D5 citava
    NAO era medicao da divergencia — era, em boa parte, este zero.

    Consertar de verdade exige levar o delta ate a `Sample`, que o TREINO usa.
    E frente propria. Ate la, esta funcao AVISA em vez de mentir calada."""
    deltas = [abs(float(a.delta_eff)) for a in amostras]
    if amostras and max(deltas, default=0.0) < 1e-12:
        print("[replay] AVISO: delta_eff e ZERO em todos os jogos — p_regua "
              "esta devolvendo o Elo SOZINHO, nao Elo+jogadores. Ver o "
              "docstring desta funcao (divida D5). Qualquer comparacao "
              "'regua x servido' feita agora mede outra coisa.")
    return [float(sigmoid(logit(float(a.p_team)) + w_lineup * float(a.delta_eff)))
            for a in amostras]


def carregar(*, artifact: str = "ml_artifact.json",
             players_artifact: str = "players_artifact.json",
             csv_train: str = "CSV_HISTORICO", csv_current: str = "CSV_DIARIO",
             params: str = "params_fitted.json", val_cutoff: str = "2026-03-31",
             usar_cache: bool = True) -> dict:
    """{'treino','val','teste': [Amostra], 'tabela': {...}, 'w_lineup': float}"""
    assinatura = _assinatura_das_entradas(
        artifact=artifact, players_artifact=players_artifact,
        csv_train=csv_train, csv_current=csv_current, params=params,
        val_cutoff=val_cutoff)

    if usar_cache and CACHE.exists():
        try:
            b = json.loads(CACHE.read_text(encoding="utf-8"))
            guardada = b.get("assinatura")
            if guardada != assinatura:
                # Falar ALTO: cache silencioso e o que congelou a regua por 11
                # dias. Quem roda uma regua tem que saber que ela recalculou.
                print(f"[replay] cache VENCIDO ({'sem assinatura, gerado antes de 06/09/2026' if not guardada else 'as entradas mudaram'}) — recalculando.")
                raise ValueError("assinatura diferente")
            return {"treino": [Amostra(**d) for d in b["treino"]],
                    "val": [Amostra(**d) for d in b["val"]],
                    "teste": [Amostra(**d) for d in b["teste"]],
                    "tabela": b["tabela"], "w_lineup": b["w_lineup"]}
        except (OSError, ValueError, TypeError, KeyError):
            pass

    import pandas as pd

    from .io_oracle import load_oracle_elixir_games
    from .ml_eval import build_split

    caminhos = []
    for pasta in (csv_train, csv_current):
        base = ROOT / pasta
        if base.is_dir():
            caminhos.extend(str(x) for x in sorted(base.glob("*.csv")))
    quadros = []
    for c in caminhos:
        try:
            quadros.append(load_oracle_elixir_games(c, keep_playoffs=None))
        except (OSError, ValueError, KeyError):
            continue
    jogos = pd.concat(quadros, ignore_index=True).drop_duplicates("gameid")

    art = MLArtifact.load_json(str(ROOT / artifact))
    p_art = PlayersArtifact.load_json(str(ROOT / players_artifact))
    # `or 0.85` era BUG (achado em 08/09/2026): em Python 0.0 e falso, entao o
    # valor REAL do artefato -- que hoje e 0.0 -- era descartado e trocado por
    # 0,85. O replay passava a medir uma camada de jogadores com peso que a
    # producao nao usa. Agora so cai no default quando a chave FALTA mesmo.
    _wl = (((art.meta or {}).get("players_layer") or {}).get("w_lineup"))
    w_lineup = float(0.85 if _wl is None else _wl)
    pr = load_match_player_rows(caminhos, keep_playoffs=None,
                                gameids=set(jogos["gameid"].astype(str)))
    brutas = _make_samples(art=art, p_art=p_art, games=jogos,
                           lineup_cache=_build_lineup_cache(pr), w_lineup=w_lineup)
    split = build_split(brutas, val_cutoff=val_cutoff)
    tabela = (_read_json_file_safe(ROOT / params).get("microseg_calibration") or {}).get("table") or {}

    def _conv(itens):
        fora = []
        for s in itens:
            fora.append(Amostra(
                gameid=str(getattr(s, "gameid", "") or ""),
                league=str(getattr(s, "league", "") or ""),
                split=str(getattr(s, "split", "") or ""),
                side=str(getattr(s, "side", "") or ""),
                bo=str(getattr(s, "bo_bucket", "") or ""),
                phase=str(getattr(s, "phase", "") or ""),
                month=str(getattr(s, "month", "") or ""),
                y=int(s.y), p_team=float(s.p_team), p_players=float(s.p_players),
                n_team=float(s.n_team), n_players=float(s.n_players),
                # `Sample` nao tem campo de delta (conferido em 08/09/2026):
                # este getattr devolve 0.0 SEMPRE. Fica explicito em vez de
                # parecer que le alguma coisa. Ver p_regua() e a divida D5.
                delta_eff=float(getattr(s, "delta_players_eff", 0.0) or 0.0)))
        return fora

    saida = {"treino": _conv(split.train.samples), "val": _conv(split.val.samples),
             "teste": _conv(split.test.samples), "tabela": tabela, "w_lineup": w_lineup}
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(
            {"treino": [asdict(a) for a in saida["treino"]],
             "val": [asdict(a) for a in saida["val"]],
             "teste": [asdict(a) for a in saida["teste"]],
             "tabela": tabela, "w_lineup": w_lineup,
             "assinatura": assinatura,
             "gerado_em": datetime.now().isoformat(timespec="seconds")},
            ensure_ascii=False),
            encoding="utf-8")
    except OSError:
        pass
    return saida
