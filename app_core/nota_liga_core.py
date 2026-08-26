"""[COPIA DO REPO DO LOL — mantenha os dois iguais.]

No Dota "liga" e um TORNEIO (DreamLeague, PGL Wallachia, ESL One). O resto
vale igual, e os numeros daqui sao outros: os piores torneios batem ECE 0.19
e 0.25, muito acima de qualquer liga de LoL — e exatamente por isso o aviso
importa mais aqui.

NOTA POR LIGA na tela — onde o modelo merece confiança (pedido dele 24/07).

A nota geral (8.63) é média de ligas muito diferentes. Medido na season 2026:
a LPL tem a MELHOR calibração de todas (ECE 0.019) e mesmo assim a nota mais
baixa (6.60) — porque os times são parelhos e o jogo é genuinamente imprevisível.
Ligas regionais (TCL/HM/ROL/EM) batem 10.0 porque o gap entre times é enorme.

Por isso a tela mostra as DUAS coisas, que respondem perguntas diferentes:
  nota        -> quanta previsibilidade existe ali (quanto edge dá pra extrair)
  calibração  -> se o % que o modelo diz é honesto (se o EV calculado é real)

Fonte: evidence/nota_por_liga.json — no LoL escrito por tools/nota_por_liga.py,
no Dota por dota_scorecard.py (mesmo schema, de proposito).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ECE: erro médio entre "o modelo disse X%" e "aconteceu Y%".
_ECE_BOA = 0.06
_ECE_RUIM = 0.10


def load_notas(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(root) / "evidence" / "nota_por_liga.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def nota_for_league(root: Path, league: str) -> dict[str, Any] | None:
    lg = str(league or "").strip().upper()
    if not lg:
        return None
    for row in load_notas(root).get("ligas") or []:
        if str(row.get("liga", "")).strip().upper() == lg:
            return row
    return None


def _calib_label(ece: float) -> str:
    if ece <= _ECE_BOA:
        return "honesta"
    if ece <= _ECE_RUIM:
        return "ok"
    return "DESCONFIE"


def render_nota_liga_badge(*, st: Any, root: Path, league: str, box: Any = None) -> None:
    """Uma linha no confronto: como o modelo se sai NESTA liga."""
    row = nota_for_league(root, league)
    if not row:
        return
    tgt = box if box is not None else st
    ece = float(row.get("ece", 0.0) or 0.0)
    lbl = _calib_label(ece)
    icon = "🟢" if lbl == "honesta" else ("🟡" if lbl == "ok" else "🔴")
    tgt.caption(
        f"{icon} **{row.get('liga')}**: nota **{row.get('nota')}** · "
        f"acerto {100.0 * float(row.get('acuracia', 0) or 0):.0f}% · "
        f"calibração {ece:.3f} ({lbl}) · {row.get('n')} jogos na season. "
        "Nota = previsibilidade da liga; calibração = se o % do modelo é honesto."
    )


def render_nota_liga_table(*, st: Any, root: Path, pd: Any) -> None:
    """Tabela completa (aba Rankings): onde confiar no modelo."""
    data = load_notas(root)
    ligas = data.get("ligas") or []
    if not ligas:
        st.caption("Nota por liga ainda não medida — rode `python tools/nota_por_liga.py`.")
        return
    df = pd.DataFrame(ligas)
    df["calibração"] = df["ece"].apply(lambda e: f"{float(e):.3f} ({_calib_label(float(e))})")
    df["acerto"] = df["acuracia"].apply(lambda a: f"{100.0 * float(a):.0f}%")
    # tier so entra se existir: no Dota "liga" e um torneio e nao ha tier de
    # liga pra mostrar — coluna vazia na tela dele e sujeira, nao informacao.
    cols = ["liga", "tier", "n", "nota", "acerto", "calibração"]
    if "tier" not in df.columns or not df["tier"].astype(str).str.strip().any():
        cols.remove("tier")
    df = df[cols].sort_values("nota", ascending=False)
    st.markdown(
        f"#### Nota do modelo por liga — season {data.get('season', '')} "
        f"(geral: **{data.get('nota_geral', '?')}**)"
    )
    st.dataframe(df, width="stretch", height=min(560, 40 + 35 * len(df)), hide_index=True)
    st.caption(
        "**Nota** (mesma régua do scorecard: aleatório 5,0 · elite 9,5) mede quanta "
        "previsibilidade existe na liga — ERLs pontuam alto porque o gap entre times é "
        "grande; LPL/LCK pontuam baixo porque os times são parelhos, não porque o modelo "
        "erra mais. **Calibração** é o que diz se o % é honesto: aí sim, ≤0,06 confia, "
        ">0,10 desconfia do EV."
    )
