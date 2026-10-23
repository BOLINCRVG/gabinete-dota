# -*- coding: utf-8 -*-
"""VENCEDOR: modelo x Pinnacle x Polymarket, e o valor CRUZADO (23/10/2026).

Pedido dele: a Polymarket como segunda referencia, e "uma maneira de cruzar os
dados que estao dando valor so na comparacao entre casas e o valor na analise
de dados tambem". Valor que so aparece contra UMA casa pode ser so aquela casa
fora do lugar; valor contra as DUAS e o modelo discordando do mercado inteiro.

AS TRES PROBABILIDADES do time da tela, no mesmo formato (BO do confronto):
  - modelo: p_map no BO1, p_series no BO3/BO5 -- os numeros do card, sem
    recalcular nada (o projeto ja pagou por ter tres pre-jogos diferentes);
  - Pinnacle: a odd SEM a margem (1/odd, normalizada pelos dois lados);
  - Polymarket: o meio do livro, que ja e probabilidade (bolsa, sem margem).

A VANTAGEM e sempre contra o PRECO QUE SE PAGA, nao contra a probabilidade:
  - Pinnacle: p_modelo * odd - 1 (a odd com margem, a que ela paga);
  - Polymarket: p_modelo / preco_de_compra - 1, com preco_de_compra = meio +
    meio spread (o "ask": e o que custa comprar o lado).

O SELO tem a Pinnacle como regua (ver `selo`): forte (Polymarket barata
contra a Pinnacle E o modelo concorda), entre casas, so do modelo, suspeito.
E um aviso separado quando as DUAS CASAS discordam entre si (>5 pontos).

O que NAO faz: nao inventa o lado que a casa nao tem. Sem Polymarket para o
jogo, a coluna fica vazia e a tela diz por que.
"""
from __future__ import annotations

import json
import unicodedata
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from app_core.oportunidades_core import VANTAGEM_SUSPEITA_PCT

ARQUIVO = "polymarket_lol.json"
LIQUIDEZ_MINIMA = 10_000.0       # US$: abaixo disso o preco mexe com pouco dinheiro
SPREAD_MAXIMO = 0.05              # 5 pontos entre compra e venda
CASAS_DISCORDAM = 0.05            # 5 pontos de probabilidade
IDADE_VELHA_H = 3.0


def _chave(nome: str) -> str:
    t = str(nome or "").strip().lower().replace("ø", "o").replace("æ", "ae")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", t)


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def carregar_polymarket(raiz: Path, agora: Optional[datetime] = None) -> Dict[str, Any]:
    """{'partidas': [...], 'idade_horas', 'velha', 'motivo'}."""
    if (Path(raiz) / "dota_paths.py").exists():
        # Arquivo identico nos dois repos; o coletor so existe no LoL por ora.
        return {"partidas": [], "motivo": "a Polymarket ainda nao e coletada para o Dota"}
    p = Path(raiz) / ARQUIVO
    if not p.exists():
        return {"partidas": [], "motivo": "o robo ainda nao coletou a Polymarket"}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"partidas": [], "motivo": "arquivo da Polymarket ilegivel"}
    idade = None
    try:
        dt = datetime.fromisoformat(str(doc.get("coletado_em")))
        idade = ((agora or datetime.now(timezone.utc)) - dt).total_seconds() / 3600.0
    except (TypeError, ValueError):
        pass
    return {"partidas": list(doc.get("partidas") or []), "idade_horas": idade,
            "velha": idade is not None and idade > IDADE_VELHA_H, "coletado_em": doc.get("coletado_em")}


def achar_na_polymarket(partidas: List[Mapping[str, Any]], time_a: str, time_b: str,
                        resolver: Optional[Callable[[str], str]] = None) -> Optional[Dict[str, Any]]:
    """A partida da Polymarket deste confronto, ja na orientacao do Time 1
    (p_a = chance do time_a). None se nao houver."""
    alvo_a, alvo_b = _chave(time_a), _chave(time_b)

    def _nome(n: str) -> str:
        if resolver is not None:
            try:
                r = resolver(n)
                if r:
                    return _chave(r)
            except Exception:
                pass
        return _chave(n)

    for p in partidas:
        a, b = _nome(str(p.get("time_a") or "")), _nome(str(p.get("time_b") or ""))
        if (a, b) == (alvo_a, alvo_b):
            return dict(p)
        if (a, b) == (alvo_b, alvo_a):
            q = dict(p)
            q["time_a"], q["time_b"] = p.get("time_b"), p.get("time_a")
            q["p_a"], q["p_b"] = p.get("p_b"), p.get("p_a")
            return q
    return None


def pinnacle_sem_margem(odd_a: Any, odd_b: Any) -> Optional[float]:
    a, b = _num(odd_a), _num(odd_b)
    if not a or not b or a <= 1 or b <= 1:
        return None
    ia, ib = 1 / a, 1 / b
    return ia / (ia + ib)


def polymarket_vale(poly: Optional[Mapping[str, Any]]) -> bool:
    if not poly:
        return False
    liq, spr = _num(poly.get("liquidez")), _num(poly.get("spread"))
    return liq is not None and liq >= LIQUIDEZ_MINIMA and spr is not None and spr <= SPREAD_MAXIMO


def comparar_vencedor(p_modelo_a: Any, *, pin_odd_a: Any = None, pin_odd_b: Any = None,
                      poly: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Uma linha por time, com as tres probabilidades, as vantagens e o selo. Pura.

    A PINNACLE E A REGUA (pedido dele, 23/10/2026: "a pinnacle vai ser quase
    sempre melhor regulada"). Alem do modelo contra cada casa, mede a casa
    contra a casa: comprar na Polymarket vale a pena SE a Pinnacle estiver
    certa? `vant_entre_casas` = p_pinnacle_sem_margem / preco_de_compra - 1.
    Esse valor existe sem modelo nenhum; o modelo concordando o reforca."""
    pm = _num(p_modelo_a)
    if pm is None or not (0 < pm < 1):
        return {"linhas": [], "motivo": "o modelo nao tem probabilidade para este confronto"}
    pin_a = pinnacle_sem_margem(pin_odd_a, pin_odd_b)
    poly_a = _num((poly or {}).get("p_a"))
    spread = _num((poly or {}).get("spread")) or 0.0
    vale = polymarket_vale(poly)
    linhas = []
    for lado, p, odd, p_pin, p_poly in (
            ("a", pm, _num(pin_odd_a), pin_a, poly_a),
            ("b", 1 - pm, _num(pin_odd_b), None if pin_a is None else 1 - pin_a,
             None if poly_a is None else 1 - poly_a)):
        v_pin = None if not odd else round(100 * (p * odd - 1), 1)
        compra = None if p_poly is None else min(0.999, p_poly + spread / 2)
        v_poly = None if not compra else round(100 * (p / compra - 1), 1)
        v_casas = None if (not compra or p_pin is None) else round(100 * (p_pin / compra - 1), 1)
        modelo_concorda = p_pin is not None and p > p_pin
        linhas.append({"lado": lado, "p_modelo": round(p, 4), "p_pinnacle": p_pin and round(p_pin, 4),
                       "odd_pinnacle": odd, "p_polymarket": p_poly and round(p_poly, 4),
                       "preco_compra_poly": compra and round(compra, 4),
                       "vant_pinnacle": v_pin, "vant_polymarket": v_poly,
                       "vant_entre_casas": v_casas,
                       "selo": selo(v_pin, v_casas, vale, modelo_concorda)})
    discordam = (pin_a is not None and poly_a is not None and abs(pin_a - poly_a) >= CASAS_DISCORDAM)
    return {"linhas": linhas, "casas_discordam": discordam,
            "diferenca_casas": None if not discordam else round(abs(pin_a - poly_a), 4)}


def selo(v_pin: Optional[float], v_casas: Optional[float], poly_vale: bool,
         modelo_concorda: bool) -> str:
    """Do mais forte ao mais fraco, com a Pinnacle como regua.

    - "forte": a Polymarket vende mais barato do que a Pinnacle acha justo E o
      modelo tambem ve o time acima da Pinnacle -- duas fontes independentes
      apontando o mesmo lado;
    - "entre casas": a Polymarket esta barata contra a Pinnacle, mas o modelo
      nao concorda (ou nao tem opiniao) -- vale pela diferenca de mercado;
    - "modelo": so o modelo ve valor contra a odd da Pinnacle;
    - "suspeito": qualquer vantagem acima de VANTAGEM_SUSPEITA_PCT -- no
      modelo, mais provavel ele errar; entre casas, mais provavel preco velho.
    Polymarket com pouco dinheiro ou spread largo nao conta como referencia."""
    vs = [v for v in (v_pin, v_casas if poly_vale else None) if v is not None]
    if any(v > VANTAGEM_SUSPEITA_PCT for v in vs):
        return "suspeito"
    casas = poly_vale and v_casas is not None and v_casas > 0
    if casas and modelo_concorda:
        return "forte"
    if casas:
        return "entre casas"
    if v_pin is not None and v_pin > 0:
        return "modelo"
    return ""


ROTULO_SELO = {
    "forte": "VALOR FORTE: Polymarket barata contra a Pinnacle, e o modelo concorda",
    "entre casas": "valor entre casas: Polymarket barata contra a Pinnacle (o modelo nao confirma)",
    "modelo": "valor so do modelo contra a odd da Pinnacle",
    "suspeito": "vantagem grande demais: confira o preco e desconfie do numero",
    "": "",
}


def odds_pinnacle_do_confronto(odds: Mapping[str, Any], time_a: str, time_b: str,
                               resolver: Optional[Callable[[str], str]], bo: Any):
    """(odd do Time 1, odd do Time 2) do vencedor da SERIE na Pinnacle, ou (None, None).

    So conta se o BO da casa e o da tela: BO5 precificado como BO3 e outro numero."""
    if resolver is None:
        return None, None
    from app_core.oportunidades_core import bo_bate, mesmo_confronto
    for jogo in odds.get("jogos") or []:
        lado = mesmo_confronto(jogo, time_a, time_b, resolver)
        if lado is None or not bo_bate(jogo, bo):
            continue
        casa = fora = None
        for l in odds.get("linhas") or []:
            if not isinstance(l, dict) or l.get("jogo") != jogo.get("id"):
                continue
            if str(l.get("tipo") or "") != "moneyline" or str(l.get("unidade") or "").strip():
                continue
            try:
                if int(l.get("periodo") or 0) != 0:
                    continue
            except (TypeError, ValueError):
                continue
            ld = str(l.get("lado") or "").lower()
            if ld == "home":
                casa = l.get("odd")
            elif ld == "away":
                fora = l.get("odd")
        if casa and fora:
            return (casa, fora) if lado else (fora, casa)
    return None, None


def _pct(x: Any) -> str:
    v = _num(x)
    return "-" if v is None else f"{100 * v:.1f}%"


def _vant(x: Any) -> str:
    v = _num(x)
    return "-" if v is None else f"{v:+.1f}%"


def render_vencedor_tres_fontes(st: Any, *, raiz: Path, odds: Mapping[str, Any],
                                confronto: Mapping[str, Any],
                                resolver: Optional[Callable[[str], str]]) -> None:
    """O bloco do topo da aba: o confronto aberto, nas tres fontes."""
    st.markdown("### Vencedor: modelo x Pinnacle x Polymarket")
    time_a = str(confronto.get("teamA") or "")
    time_b = str(confronto.get("teamB") or "")
    poly_doc = carregar_polymarket(raiz)
    if not (time_a and time_b):
        st.caption("Abra um confronto (Time 1 x Time 2) para comparar o vencedor nas tres fontes.")
    else:
        try:
            bo = int(confronto.get("bo") or 1)
        except (TypeError, ValueError):
            bo = 1
        p_modelo = confronto.get("p_map") if bo == 1 else confronto.get("p_series")
        poly = achar_na_polymarket(poly_doc.get("partidas") or [], time_a, time_b, resolver)
        if poly is not None and int(poly.get("bo") or bo) != bo:
            st.caption(f"A Polymarket tem este confronto em BO{poly.get('bo')} e a tela esta em BO{bo}: ficou de fora.")
            poly = None
        pin_a, pin_b = odds_pinnacle_do_confronto(odds, time_a, time_b, resolver, bo)
        r = comparar_vencedor(p_modelo, pin_odd_a=pin_a, pin_odd_b=pin_b, poly=poly)
        if not r["linhas"]:
            st.info(r.get("motivo") or "Sem numeros para comparar.")
        else:
            nomes = {"a": time_a, "b": time_b}
            tabela = [{
                "Time": nomes[l["lado"]],
                "Modelo": _pct(l["p_modelo"]),
                "Pinnacle (sem margem)": _pct(l["p_pinnacle"]),
                "Polymarket": _pct(l["p_polymarket"]),
                "Vantagem x Pinnacle": _vant(l["vant_pinnacle"]),
                "Vantagem x Polymarket": _vant(l["vant_polymarket"]),
                "Polymarket x Pinnacle": _vant(l["vant_entre_casas"]),
                "Selo": ROTULO_SELO.get(l["selo"], ""),
            } for l in r["linhas"]]
            st.dataframe(tabela, hide_index=True, width="stretch")
            for l in r["linhas"]:
                if l["selo"] in ("forte", "entre casas"):
                    st.success(f"{nomes[l['lado']]}: {ROTULO_SELO[l['selo']]}.")
                elif l["selo"] == "suspeito":
                    st.warning(f"{nomes[l['lado']]}: {ROTULO_SELO['suspeito']}.")
            if r.get("casas_discordam"):
                st.info("As duas casas discordam entre si em {:.0f} pontos: mercado dividido."
                        .format(100 * r["diferenca_casas"]))
            avisos = []
            if pin_a is None:
                avisos.append("Pinnacle sem este confronto (ou em outro BO)")
            if poly is None:
                avisos.append("Polymarket sem este confronto" + (
                    f" ({poly_doc['motivo']})" if poly_doc.get("motivo") else ""))
            elif not polymarket_vale(poly):
                avisos.append("Polymarket com pouco dinheiro ou spread largo neste jogo: referencia fraca")
            if avisos:
                st.caption(" · ".join(avisos) + ".")
    partidas = poly_doc.get("partidas") or []
    if partidas:
        with st.expander(f"Polymarket: {len(partidas)} jogo(s) de LoL abertos"
                         + (" (dado de mais de 3 horas)" if poly_doc.get("velha") else "")):
            st.dataframe([{
                "Jogo": f"{p.get('time_a')} x {p.get('time_b')}",
                "Torneio": p.get("torneio"), "BO": p.get("bo"),
                "Inicio (UTC)": str(p.get("inicio_utc") or "")[:16],
                "Chance do 1o nome": _pct(p.get("p_a")),
                "Spread": _pct(p.get("spread")),
                "Liquidez (US$)": f"{_num(p.get('liquidez')) or 0:,.0f}",
            } for p in partidas], hide_index=True, width="stretch")
            st.caption("Coletado pelo robo de hora em hora (a Polymarket e bloqueada no Brasil; "
                       "o robo roda nos EUA). Preco = meio do livro, antes do inicio do jogo.")
