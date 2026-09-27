# -*- coding: utf-8 -*-
# app_core/tab_dispatcher.py
# Centraliza definições e despacho de tabs do Streamlit.
# Fonte única de verdade para rótulos e ordem das tabs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constantes de rótulo — use estas em vez de strings literais espalhadas
# ---------------------------------------------------------------------------

TAB_PARAMS = "Parâmetros"
TAB_VISAO = "Visão Geral"
TAB_CONS = "Consistência das linhas"
TAB_RANKINGS = "Rankings"
TAB_JOGOS = "Jogos"
TAB_SERIES = "Series"
TAB_STATSCAMPS = "StatsCamps"
TAB_CAMPEOES = "Campeões"
TAB_PLAYERS = "Players"
TAB_IDENTIDADE = "Identidade"
TAB_LINE_ANALYSIS = "Análise de Linha"
TAB_OPORTUNIDADES = "Oportunidades"

# Ordem canônica das tabs secundárias
SECONDARY_TAB_ORDER: tuple[str, ...] = (
    TAB_RANKINGS,
    TAB_JOGOS,
    TAB_SERIES,
    TAB_STATSCAMPS,
    TAB_LINE_ANALYSIS,
    TAB_OPORTUNIDADES,
    TAB_CAMPEOES,
    TAB_PLAYERS,
    TAB_IDENTIDADE,
)

# Todas as tabs conhecidas (primárias + secundárias)
ALL_TABS: tuple[str, ...] = (
    TAB_PARAMS,
    TAB_VISAO,
    TAB_CONS,
    *SECONDARY_TAB_ORDER,
)


# ---------------------------------------------------------------------------
# TabConfig — metadados por tab
# ---------------------------------------------------------------------------

@dataclass
class TabConfig:
    label: str
    render_fn: Callable[[], None]


# ---------------------------------------------------------------------------
# Helpers de despacho
# ---------------------------------------------------------------------------



# Onde as falhas de aba ficam registradas, para os testes de tela poderem
# reprovar. Sem isto, isolar a falha viraria ESCONDER a falha: o `at.exception`
# do AppTest fica vazio e a suite passa com a aba quebrada.
CHAVE_FALHAS_DE_ABA = "_tab_falhas"


def _texto_da_falha(tab_name: str, erro: BaseException) -> str:
    """Mensagem para o dono, que nao le traceback.

    O caso do arquivo desatualizado ganha texto proprio porque tem conserto
    conhecido e nao e defeito do codigo: a Streamlit Community Cloud recarrega
    o arquivo principal quando chega um commit, mas NAO recarrega os modulos
    importados — eles ficam no `sys.modules` da maquina antiga. O arquivo
    principal novo entao chama uma funcao velha e o encaixe nao bate.
      - https://discuss.streamlit.io/t/apps-deployed-on-streamlit-community-cloud-reacts-only-on-main-app-file-changes-but-not-on-changes-in-the-py-modules/36626
      - https://github.com/streamlit/streamlit/issues/366
      (ambas abertas em 13/09/2026, depois de o site cair por isso)
    """
    msg = str(erro)
    desencontro = isinstance(erro, TypeError) and (
        "unexpected keyword argument" in msg
        or "missing 1 required" in msg
        or "missing required" in msg
        or "positional argument" in msg
    )
    if desencontro:
        return (
            f"A aba **{tab_name}** esta rodando com um arquivo "
            "desatualizado do servidor. O site continua de pe; "
            "so esta aba parou.\n\n"
            "**Conserto:** no painel do Streamlit, canto inferior "
            "direito, **Manage app -> Reboot app**. Publicar um "
            "commit novo nao resolve sozinho - so o reinicio baixa "
            "os arquivos de dentro.\n\n"
            f"Detalhe tecnico: `{msg}`"
        )
    return (
        f"A aba **{tab_name}** deu erro e nao foi desenhada. "
        "O resto do site continua funcionando.\n\n"
        f"Detalhe tecnico: `{type(erro).__name__}: {msg}`"
    )


def iniciar_registro_de_falhas(st: Any) -> list:
    """Zera o registro de abas quebradas no comeco do desenho da pagina.

    Quem zera e o inicio da pagina, UMA VEZ SO — nao cada bloco de abas. As
    tabs principais (Parametros, Visao Geral, Consistencia) sao desenhadas
    antes das secundarias; se o despacho das secundarias zerasse, a falha de
    uma tab principal sumiria antes de alguem ver.
    """
    falhas: list[tuple[str, str]] = []
    try:
        st.session_state[CHAVE_FALHAS_DE_ABA] = falhas
    except (AttributeError, TypeError, KeyError):
        pass
    return falhas


def _onde_quebrou(erro: BaseException) -> str:
    """arquivo:linha do ponto mais fundo DO NOSSO codigo.

    So o tipo e a mensagem nao bastam: em 15/09/2026 um "'dict' object is not
    callable" na Visao Geral me custou uma caçada, porque a funcao tem
    centenas de linhas e varios blocos protegidos. O quadro mais fundo que
    seja nosso aponta o lugar em um segundo.
    """
    import traceback
    try:
        quadros = traceback.extract_tb(erro.__traceback__)
    except Exception:
        return ""
    nosso = [q for q in quadros
             if ("app_core" in str(q.filename) or "mlcore" in str(q.filename)
                 or "plays_app" in str(q.filename))]
    q = (nosso or quadros or [None])[-1]
    if q is None:
        return ""
    import os
    return "{}:{}".format(os.path.basename(str(q.filename)), q.lineno)


def _registrar_falha(st: Any, tab_name: str, erro: BaseException) -> None:
    onde = _onde_quebrou(erro)
    marca = "{}: {}{}".format(type(erro).__name__, erro,
                              " [{}]".format(onde) if onde else "")
    try:
        st.session_state[CHAVE_FALHAS_DE_ABA].append((tab_name, marca))
    except (AttributeError, TypeError, KeyError, IndexError):
        pass


def desenhar_aba(st: Any, tab_ref: Any, tab_name: str,
                 render_fn: Callable[[], None] | None) -> bool:
    """Desenha UMA aba sem deixar a falha dela chegar no resto da pagina.

    Devolve True se desenhou inteira. Vale para as tabs principais tambem —
    elas sao desenhadas soltas no arquivo principal e, sem isto, continuariam
    podendo derrubar o app como o StatsCamps derrubou em 13/09/2026.

    So `Exception` e capturada, de proposito: `st.rerun()` e `st.stop()`
    levantam `ScriptControlException`, que herda de `BaseException` e passa
    reto por aqui (conferido no streamlit 1.51.0). Capturar tudo quebraria
    qualquer botao que recarrega a pagina.
    """
    if tab_ref is None or render_fn is None:
        return False
    with tab_ref:
        try:
            render_fn()
            return True
        except Exception as erro:  # noqa: BLE001 — ver docstring
            _registrar_falha(st, tab_name, erro)
            try:
                st.error(_texto_da_falha(tab_name, erro))
            except (AttributeError, TypeError):
                pass
            return False


def dispatch_secondary_tabs(
    st: Any,
    tab_refs: dict[str, Any],
    renderers: dict[str, Callable[[], None]],
    order: tuple[str, ...] = SECONDARY_TAB_ORDER,
) -> None:
    """Percorre as tabs secundárias na ordem canônica e chama cada renderer.

    Cada aba e desenhada isolada: uma aba com defeito custa uma aba, nao o
    app inteiro. Ver `desenhar_aba`.
    """
    try:
        st.session_state[CHAVE_FALHAS_DE_ABA]
    except (AttributeError, TypeError, KeyError):
        # Chamado sem o inicio da pagina ter rodado (testes, ou um app que
        # so tem abas secundarias). Ter registro vazio e melhor que nenhum.
        iniciar_registro_de_falhas(st)

    for tab_name in order:
        desenhar_aba(st, tab_refs.get(tab_name), tab_name,
                     renderers.get(tab_name))
