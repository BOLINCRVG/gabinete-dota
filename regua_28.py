# -*- coding: utf-8 -*-
"""RÉGUA DAS 28 — "estamos no melhor possível?" com número, não com sensação.

Adaptação do ML Test Score (Breck, Cai, Nielsen, Salib & Sculley — Google,
IEEE Big Data 2017): 28 testes fixos em quatro áreas, aplicados aos dois
Gabinetes.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║                                                                             ║
║ 1. A LISTA DOS 28 e as quatro áreas vêm do artigo, lido direto (as tabelas  ║
║    I a IV). Não adaptei de memória — faltavam-me as áreas de Dados e        ║
║    Modelo, e inventá-las seria o oposto do ponto da régua.                  ║
║                                                                             ║
║ 2. PONTUAÇÃO, do próprio artigo:                                            ║
║      meio ponto  — o teste é executado À MÃO e o resultado documentado      ║
║      um ponto    — existe sistema que o executa AUTOMÁTICO e repetido       ║
║    "Régua que só roda quando eu rodo vale metade de uma régua."             ║
║                                                                             ║
║ 3. A NOTA FINAL É O MÍNIMO das quatro áreas, não a média — também do        ║
║    artigo: "escolhemos o mínimo porque acreditamos que as quatro áreas são  ║
║    importantes, e um sistema precisa considerar todas para subir a nota".   ║
║    É o que impede a nossa força em desenvolvimento de modelo de esconder    ║
║    que monitoramento está perto de zero.                                    ║
║                                                                             ║
║ 4. A ESCALA DE LEITURA (Tabela V do artigo):                                ║
║      0        mais projeto de pesquisa que sistema em produção              ║
║      (0,1]    vale considerar buracos sérios de confiabilidade              ║
║      (1,2]    primeira passada de produtização; falta investimento          ║
║      (2,3]    razoavelmente testado, mas muito ainda podia ser automatizado ║
║      (3,5]    forte, adequado a sistema crítico                             ║
║      >5       excepcional                                                   ║
║                                                                             ║
║ 5. POR QUE PARTE É VERIFICADA E PARTE É DECLARADA:                          ║
║    o artigo é uma rubrica preenchida pela equipe. Onde dá para checar        ║
║    mecanicamente (existe passo na esteira? existe arquivo de evidência?),   ║
║    a régua checa — porque declaração envelhece e verificação não. Onde não  ║
║    dá, a declaração é obrigada a NOMEAR a evidência.                        ║
║ 6. POR QUE A DECLARAÇÃO VIVE NO REPO E NOMEIA O SEU DONO:                  ║
║    este .py é o mesmo nos dois Gabinetes; com as declarações dentro dele,  ║
║    medir o Dota herdava afirmações escritas olhando o LoL. Mover para      ║
║    dentro do repo não basta — o arquivo também se copia —, então a régua   ║
║    confere o SUJEITO, como nas atestações de cadeia de suprimentos:        ║
║    "nomes podem ser reusados, mas o resumo criptográfico identifica o      ║
║    conteúdo exato", e uma atestação de uma versão não vale para outro      ║
║    binário com o mesmo rótulo. Sujeito errado vale ZERO.                   ║
║    kusari.dev/learning-center/attestation/                                 ║
║                                                                            ║
║ 7. POR QUE A DECLARAÇÃO É VERSIONADA (exceção no .gitignore):              ║
║    ela é ENTRADA da régua, não saída. A prática é guardar a política em    ║
║    repositório versionado, com revisão obrigatória, "garantindo trilha     ║
║    auditável de toda mudança de política". Ignorada, a esteira roda com    ║
║    zero declarações e a nota despenca sozinha na nuvem, sem nada ter mudado.║
║    cloudaware.com/blog/devsecops-compliance/                               ║
║                                                                            ║
║ 8. POR QUE TODA DECLARAÇÃO VENCE EM 90 DIAS:                               ║
║    carimbo de "verificado em tal dia" que ninguém confere não é controle,  ║
║    é decoração — foi o que eu fiz na primeira versão. A prática contra     ║
║    "compliance drift" é EXPIRAÇÃO com ciclo de revisão trimestral: um      ║
║    controle perfeitamente conforme vira não-conformidade no dia em que a   ║
║    certificação vence. Sem data, ou vencida, a declaração vale ZERO e a    ║
║    nota cai sozinha até alguém reverificar.                                ║
║                                                                            ║
║ 9. POR QUE OS CAMINHOS SÃO ESCRITOS, e não procurados no disco:            ║
║    as ferramentas moram em tools/ no LoL e na raiz no Dota. A primeira     ║
║    correção foi varrer o disco pelo NOME quando o caminho falhasse — e ela ║
║    achou cópias dentro de backups/. Descoberta implícita é anti-padrão por ║
║    resolver ambíguo sem avisar: "misturar referência explícita e implícita"║
║    faz o sistema pegar a origem errada (é o mecanismo do dependency        ║
║    confusion). Os dois caminhos possíveis ficam ESCRITOS na lista de itens.║
║    cloudsmith.com/blog/eliminating-ambiguity-the-case-for-explicit-docker-image-paths║
╚═════════════════════════════════════════════════════════════════════════════╝

Uso: python tools/regua_28.py [--root OUTRO_REPO]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

def _raiz_do_repo(inicio: Path) -> Path:
    """A raiz do repositorio, sem depender de ONDE este arquivo mora.

    Segunda vez que este bug aparece: no LoL o script vive em tools/ e no Dota
    na raiz, entao parents[1] fixo resolvia para a Area de Trabalho. A copia do
    Dota varria o Desktop INTEIRO — enxergava as ferramentas do LoL como se
    fossem dela e gravava o resultado em Desktop/evidence/. A nota do Dota que
    eu vinha reportando media a Area de Trabalho, nao o repositorio.

    Mesma cura ja escrita em tools/livro_caixa.py, pelo mesmo motivo:
    `git rev-parse --show-toplevel` e o jeito canonico; travessia atras de .git
    fica de reserva se o git nao estiver no PATH."""
    try:
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           cwd=str(inicio.parent), capture_output=True,
                           text=True, encoding="utf-8", timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    atual = inicio.resolve().parent
    for cand in [atual, *atual.parents]:
        if (cand / ".git").exists():
            return cand
    return atual


RAIZ_PADRAO = _raiz_do_repo(Path(__file__))

# --------------------------------------------------------------------------
# Verificadores mecanicos. Devolvem (nota, evidencia). Nota: 0, 0.5 ou 1.0.
# --------------------------------------------------------------------------


def _existe(raiz: Path, *padroes: str):
    """Acha a ferramenta pelos caminhos DECLARADOS — nunca por busca no disco.

    Os dois Gabinetes guardam as ferramentas em lugares diferentes: no LoL
    vivem em tools/, no Dota na raiz. A primeira correcao foi cair para uma
    varredura por NOME quando o caminho falhasse, e ela achou copias dentro de
    backups/ — remendei ordenando por profundidade, o que so escondeu o
    problema. Descoberta implicita e anti-padrao justamente por isso: mistura
    referencia explicita com implicita e resolve para o alvo errado sem avisar
    (o mesmo mecanismo do "dependency confusion"). Entao os dois caminhos
    possiveis ficam ESCRITOS na lista de itens, e o que nao esta escrito nao
    conta.
    """
    return [p for pad in padroes for p in raiz.glob(pad) if p.exists()]


def _na_esteira(raiz: Path, *trechos: str):
    """O passo roda numa esteira (automatico = 1 ponto) ou so a mao (0,5)?"""
    for yml in (raiz / ".github" / "workflows").glob("*.y*ml"):
        try:
            texto = yml.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for t in trechos:
            if t in texto:
                return yml.name
    return None


def v_na_esteira(*trechos: str, descricao: str):
    """O passo roda numa esteira? Nomeia o ARQUIVO ONDE ACHOU.

    Existe porque a primeira versao reportava o primeiro resultado do glob em
    vez do arquivo onde o trecho estava, produzindo evidencia sem sentido
    ("nota_semanal.yml roda em retreino.yml"). Nota certa, texto mentiroso —
    e num arquivo cujo proposito e ser verificavel, texto mentiroso e pior."""
    def verificar(raiz: Path):
        yml = _na_esteira(raiz, *trechos)
        if yml:
            return 1.0, f"{descricao} ({yml})"
        return 0.0, f"{descricao}: nao encontrado em nenhuma esteira"
    return verificar


def v_suite_na_esteira(raiz: Path):
    """A suite INTEIRA roda na esteira E deu verde ha pouco? (22/10/2026)

    Casar texto no YAML nao prova que o passo roda (armadilha do CLAUDE.md).
    A prova e o que o robo GRAVOU: evidence/suite_esteira.json, verde e com
    no maximo 3 dias. Workflow sem resultado recente nao ganha ponto."""
    yml = _na_esteira(raiz, "tools/suite_na_esteira.py")
    if not yml:
        return 0.0, "a suite inteira nao roda em esteira nenhuma"
    arq = raiz / "evidence" / "suite_esteira.json"
    try:
        d = json.loads(arq.read_text(encoding="utf-8"))
        idade = (datetime.now() - datetime.fromisoformat(str(d["gerado_em"]))).days
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0, f"{yml} existe, mas evidence/suite_esteira.json nao foi gravado"
    if not d.get("ok"):
        return 0.0, (f"{yml}: a suite FALHOU na ultima rodada ({d.get('falhas')} falhas, "
                     f"{d.get('erros')} erros)")
    if idade > 3:
        return 0.0, f"{yml}: ultimo resultado verde tem {idade} dias (limite 3)"
    return 1.0, f"suite inteira verde na esteira ({yml}): {d.get('testes')} testes, {idade}d"


def v_ensaio_na_suite(ferramenta: str, teste: str, descricao: str):
    """Ponto inteiro: a ferramenta existe, o TESTE que a exercita existe, e a
    suite (que o robo roda todo dia) esta verde e recente. So a ferramenta:
    meio ponto. (22/10/2026)"""
    def verificar(raiz: Path):
        tem_f = (raiz / ferramenta).exists()
        tem_t = (raiz / "mlcore" / "tests" / teste).exists()
        if tem_f and tem_t:
            nota, txt = v_suite_na_esteira(raiz)
            if nota >= 1.0:
                return 1.0, f"{descricao} ({ferramenta} + {teste}, na suite verde do robo)"
            return 0.5, f"{descricao}: existe, mas a suite nao esta verde no robo ({txt})"
        if tem_f:
            return 0.5, f"{ferramenta} existe, mas nenhum teste o roda sozinho"
        return 0.0, f"{descricao}: nao existe"
    return verificar


def v_desempenho(raiz: Path):
    """Desempenho computacional medido sozinho e recente? (22/10/2026) -- o robo
    roda tools/medir_desempenho.py e grava evidence/desempenho.json; vale ponto
    inteiro com medicao de ate 3 dias."""
    yml = _na_esteira(raiz, "tools/medir_desempenho.py")
    if not yml:
        return 0.0, "tempo de treino e de resposta nao sao medidos em lugar nenhum"
    try:
        d = json.loads((raiz / "evidence" / "desempenho.json").read_text(encoding="utf-8"))
        idade = (datetime.now() - datetime.fromisoformat(str(d["gerado_em"]))).days
    except (OSError, ValueError, KeyError, TypeError):
        return 0.5, f"{yml} mede, mas evidence/desempenho.json ainda nao foi gravado"
    if idade > 3:
        return 0.5, f"{yml} mede, mas a ultima medicao tem {idade} dias"
    u = d.get("ultimo") or {}
    return 1.0, (f"tempo medido no robo ({yml}): carregar {u.get('carregar_s')}s, "
                 f"previsoes {u.get('prever_s')}s, suite {u.get('suite_s')}s")


def v_dependabot(raiz: Path):
    """Mudanca de dependencia notifica? (22/10/2026) -- .github/dependabot.yml vigiando
    o pip. O GitHub roda sozinho e avisa por pull request: e automatico."""
    arq = raiz / ".github" / "dependabot.yml"
    try:
        txt = arq.read_text(encoding="utf-8")
    except OSError:
        return 0.0, "nao ha dependabot nem renovate em .github/; nada notifica"
    if 'package-ecosystem: "pip"' in txt or "package-ecosystem: pip" in txt:
        return 1.0, "dependabot.yml vigia o pip (e as actions) toda semana"
    return 0.5, "dependabot.yml existe, mas nao vigia o pip"


def v_reversao(raiz: Path):
    """Da para voltar atras, e isso e ENSAIADO sozinho? (22/10/2026)

    Ponto inteiro: tools/reverter_modelo.py + o ensaio (test_reverter_modelo.py)
    rodando na suite verde do robo. So a ferramenta antiga (rollback_baseline,
    que le copias fora do git) ou a nova sem ensaio: meio ponto."""
    tem_nova = (raiz / "tools" / "reverter_modelo.py").exists()
    tem_ensaio = (raiz / "mlcore" / "tests" / "test_reverter_modelo.py").exists()
    if tem_nova and tem_ensaio:
        nota, txt = v_suite_na_esteira(raiz)
        if nota >= 1.0:
            return 1.0, "reverter_modelo.py + ensaio de reversao na suite verde do robo"
        return 0.5, f"ferramenta e ensaio existem, mas a suite nao esta verde no robo: {txt}"
    if tem_nova or _existe(raiz, "tools/rollback_baseline.py", "rollback_baseline.py"):
        return 0.5, "ferramenta de reversao existe, mas sem ensaio automatico"
    return 0.0, "ferramenta nao existe"


def v_ferramenta(*, arquivos: tuple[str, ...], esteira: tuple[str, ...] = ()):
    """Existe a ferramenta? Roda sozinha? -> 1,0. So a mao? -> 0,5. Nao existe? 0."""
    def verificar(raiz: Path):
        achados = _existe(raiz, *arquivos)
        if not achados:
            return 0.0, "ferramenta nao existe"
        yml = _na_esteira(raiz, *esteira) if esteira else None
        nome = achados[0].name
        if yml:
            return 1.0, f"{nome} roda em {yml}"
        return 0.5, f"{nome} existe, mas so roda a mao"
    return verificar


def v_ferramenta_ou(arquivos: tuple[str, ...], esteira: tuple[str, ...], senao):
    """Ferramenta na esteira quando ela existe neste repo; senao, o verificador reserva."""
    auto = v_ferramenta(arquivos=arquivos, esteira=esteira)
    def verificar(raiz: Path):
        return auto(raiz) if _existe(raiz, *arquivos) else senao(raiz)
    return verificar


VALIDADE_DECLARACAO_DIAS = 90
ARQ_DECLARACOES = "evidence/regua_28_declaracoes.json"


def _sujeito(raiz: Path) -> str:
    """Quem e este repositorio, na voz do proprio git."""
    try:
        r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=str(raiz),
                           capture_output=True, text=True, encoding="utf-8", timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return raiz.name


def _declaracoes(raiz: Path) -> tuple[dict, str]:
    """As declaracoes sao DO REPOSITORIO medido, e so valem NELE.

    Antes elas moravam neste arquivo. Como o arquivo e o mesmo nos dois
    Gabinetes, medir o Dota herdava as afirmacoes escritas olhando o LoL — a
    regua creditava ao Dota coisas que so existem aqui.

    Mover para dentro do repo nao basta: o arquivo tambem se copia. Por isso a
    declaracao NOMEIA o seu sujeito e a regua confere. E a regra das atestacoes
    de cadeia de suprimentos, onde a afirmacao se prende ao artefato exato:
    "nomes podem ser reusados, mas o resumo criptografico identifica o conteudo
    exato" — uma atestacao de uma versao nao vale para outro binario com o
    mesmo rotulo. Aqui o sujeito e o remoto do git, e declaracao com sujeito
    errado vale ZERO em vez de virar credito emprestado."""
    cache = _declaracoes.__dict__.setdefault("_c", {})
    chave = str(raiz)
    if chave not in cache:
        p = raiz / ARQ_DECLARACOES
        try:
            bruto = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache[chave] = ({}, f"{ARQ_DECLARACOES} ausente ou ilegivel")
            return cache[chave]
        dono = str(bruto.get("repo") or "").strip()
        sujeito = _sujeito(raiz)
        if not dono:
            cache[chave] = ({}, "declaracoes sem campo 'repo': nao da para saber de quem sao")
        elif dono not in sujeito:
            cache[chave] = ({}, f"declaracoes sao de '{dono}', mas este repo e '{sujeito}'")
        else:
            cache[chave] = (bruto.get("declaracoes") or {}, "")
    return cache[chave]


def declarado(area: str, num: int):
    """Uma declaracao COM VALIDADE, lida do repositorio medido.

    O carimbo "verificado em tal dia" nao serve para nada se ninguem confere o
    carimbo — foi o que eu fiz na primeira versao, reintroduzindo dentro da
    regua a mesma doenca que o livro-caixa existe para tratar. Ver fonte 6."""
    def verificar(raiz: Path):
        d = _declaracoes(raiz)[0].get(f"{area} {num}")
        if not d:
            return 0.0, f"NAO DECLARADO neste repositorio ({ARQ_DECLARACOES})"
        just = str(d.get("justificativa") or "")
        carimbo = d.get("verificado_em")
        if not carimbo:
            return 0.0, f"SEM DATA DE VERIFICACAO (vale zero): {just}"
        try:
            quando = datetime.fromisoformat(str(carimbo))
        except ValueError:
            return 0.0, f"data de verificacao ilegivel: {just}"
        idade = (datetime.now() - quando).days
        if idade > VALIDADE_DECLARACAO_DIAS:
            return 0.0, (f"VENCIDA ha {idade - VALIDADE_DECLARACAO_DIAS}d "
                         f"(verificada em {carimbo}): {just}")
        resta = VALIDADE_DECLARACAO_DIAS - idade
        return float(d.get("nota") or 0.0), (f"{just} [verificada em {carimbo}, "
                                             f"vence em {resta}d]")
    return verificar


def v_evidencia(nome: str, cadencia_dias: float):
    """Existe evidencia recente? Usa o mesmo criterio do tools/frescor.py."""
    def verificar(raiz: Path):
        p = raiz / "evidence" / nome
        if not p.exists():
            return 0.0, f"evidence/{nome} nao existe"
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            carimbo = d.get("gerado_em") or d.get("generated_at")
            quando = datetime.fromisoformat(str(carimbo).replace("Z", "+00:00").split("+")[0])
            idade = (datetime.now() - quando).days
        except (OSError, ValueError, TypeError):
            return 0.5, f"evidence/{nome} existe mas SEM carimbo de data"
        if idade > 3 * cadencia_dias:
            return 0.5, f"evidence/{nome} tem {idade}d (velha)"
        return 1.0, f"evidence/{nome} tem {idade}d"
    return verificar


# --------------------------------------------------------------------------
# Os 28. (area, numero, titulo, verificador OU (nota, justificativa))
# Declaracao e obrigada a NOMEAR a evidencia (fonte 5).
# --------------------------------------------------------------------------
ITENS = [
    # ---- DADOS ----
    # 22/10/2026: no LoL o contrato de dados roda no robo antes do incremental
    # e bloqueia o modelo se falhar. Onde a ferramenta nao existe (Dota), vale a
    # declaracao.
    ("dados", 1, "expectativas das features num esquema",
     v_ferramenta_ou(("tools/run_phase1_data_contract.py", "tools/contrato_dados_dota.py"),
                     ("run_phase1_data_contract.py", "contrato_dados_dota.py"),
                     declarado("dados", 1))),
    ("dados", 2, "todas as features sao beneficas",
     declarado("dados", 2)),
    # 23/10/2026: tempo por previsao de cada camada e peso de cada bloco nos
    # arquivos do modelo, medidos na suite todo dia, com teto (5 ms, 20 MB).
    ("dados", 3, "nenhuma feature custa demais",
     v_ensaio_na_suite("tools/custo_por_feature.py", "test_custo_por_feature.py",
                       "custo de cada camada medido e abaixo do teto")),
    ("dados", 4, "features atendem requisitos de nivel superior",
     declarado("dados", 4)),
    # 23/10/2026: so o identificador PROFISSIONAL publico entra (playerid no LoL,
    # account_id no Dota); o vigia barra apelido Steam, nome real, pais, avatar,
    # e-mail etc. em todo arquivo de dados, todo dia, na suite.
    ("dados", 5, "controles de privacidade no pipeline",
     v_ensaio_na_suite("tools/vigia_privacidade.py", "test_vigia_privacidade.py",
                       "nenhum dado pessoal alem do identificador profissional")),
    ("dados", 6, "feature nova entra rapido",
     declarado("dados", 6)),
    # 22/10/2026: a suite INTEIRA passou a rodar na esteira (suite.yml ->
    # tools/suite_na_esteira.py, que grava evidence/suite_esteira.json). Os
    # testes destes itens fazem parte dela, entao "roda sozinho" passa a valer.
    ("dados", 7, "codigo de feature testado",
     v_ferramenta(arquivos=("mlcore/tests/test_*features*.py",
                            "mlcore/tests/test_*data*.py"),
                  esteira=("tools/suite_na_esteira.py",))),
    # ---- MODELO ----
    ("modelo", 1, "especificacao revisada e versionada",
     declarado("modelo", 1)),
    ("modelo", 2, "metricas offline e online se correlacionam",
     declarado("modelo", 2)),
    ("modelo", 3, "hiperparametros ajustados",
     declarado("modelo", 3)),
    ("modelo", 4, "impacto do envelhecimento conhecido",
     declarado("modelo", 4)),
    ("modelo", 5, "modelo simples nao e melhor",
     declarado("modelo", 5)),
    ("modelo", 6, "qualidade suficiente em cada FATIA",
     v_ferramenta(arquivos=("tools/nota_por_liga.py", "dota_scorecard.py"),
                  esteira=("nota_por_liga.py", "dota_scorecard.py"))),
    ("modelo", 7, "testado quanto a inclusao",
     declarado("modelo", 7)),
    # ---- INFRA ----
    # 23/10/2026: medido -- duas rodadas do treino dao o mesmo modelo (fora os
    # carimbos de hora); o ensaio roda na suite todo dia.
    ("infra", 1, "treino reproduzivel",
     v_ensaio_na_suite("mlcore/train_offline.py", "test_treino_reproduzivel.py",
                       "treinar duas vezes no mesmo CSV da o mesmo modelo")),
    ("infra", 2, "especificacao com teste unitario",
     v_ferramenta(arquivos=("mlcore/tests/test_artifact.py",),
                  esteira=("tools/suite_na_esteira.py",))),
    ("infra", 3, "pipeline com teste de integracao",
     v_suite_na_esteira),
    ("infra", 4, "qualidade validada ANTES de servir",
     v_na_esteira("Guardas pos-retreino", "Guardas pós-retreino",
                  descricao="guarda que barra modelo quebrado antes de publicar")),
    ("infra", 5, "modelo depuravel",
     declarado("infra", 5)),
    # 23/10/2026: o canario compara o modelo novo com o do ar nos mesmos jogos
    # e barra antes do commit, nos robos diarios dos dois repos.
    ("infra", 6, "canario antes de servir",
     v_ferramenta_ou(("tools/canario_modelo.py",), ("canario_modelo.py",), declarado("infra", 6))),
    ("infra", 7, "da para reverter", v_reversao),
    # ---- MONITORAMENTO ----
    ("monitor", 1, "mudanca de dependencia notifica",
     v_dependabot),
    # 23/10/2026: o contrato de dados confere as invariantes de TODO CSV novo
    # no robo, antes de ele chegar ao modelo (que serve do mesmo CSV).
    ("monitor", 2, "invariantes valem na entrada",
     v_ferramenta_ou(("tools/run_phase1_data_contract.py", "tools/contrato_dados_dota.py"),
                     ("run_phase1_data_contract.py", "contrato_dados_dota.py"),
                     declarado("monitor", 2))),
    ("monitor", 3, "treino e servico NAO divergem",
     declarado("monitor", 3)),
    ("monitor", 4, "modelo nao esta velho demais",
     v_ferramenta(arquivos=("tools/frescor.py", "frescor.py"), esteira=("frescor.py",))),
    ("monitor", 5, "numericamente estavel",
     v_ensaio_na_suite("tools/vigia_numerico.py", "test_vigia_numerico.py",
                       "nenhum NaN/infinito nos arquivos do modelo")),
    ("monitor", 6, "desempenho computacional nao regrediu",
     v_desempenho),
    ("monitor", 7, "qualidade de previsao nao regrediu",
     v_ferramenta(arquivos=("tools/ml_fusion_scorecard.py", "ml_fusion_scorecard.py", "tools/portao_qualidade_dota.py"),
                  esteira=("ml_fusion_scorecard.py", "portao_qualidade_dota.py"))),
]


AREAS = {"dados": "DADOS", "modelo": "MODELO", "infra": "INFRAESTRUTURA", "monitor": "MONITORAMENTO"}


def leitura(nota: float) -> str:
    if nota <= 0:
        return "mais projeto de pesquisa que sistema em producao"
    if nota <= 1:
        return "vale considerar buracos serios de confiabilidade"
    if nota <= 2:
        return "primeira passada de produtizacao; falta investimento"
    if nota <= 3:
        return "razoavelmente testado, mas muito ainda podia ser automatizado"
    if nota <= 5:
        return "forte, adequado a sistema critico"
    return "excepcional"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="evidence/regua_28.json")
    args = ap.parse_args()
    raiz = Path(args.root).resolve() if args.root else RAIZ_PADRAO

    print(f"[raiz] {raiz}")
    d, recusa = _declaracoes(raiz)
    if recusa:
        print(f"[declaracoes] NENHUMA vale aqui: {recusa}")
    else:
        print(f"[declaracoes] {len(d)} lidas de {ARQ_DECLARACOES}")
    print()
    por_area: dict[str, float] = {k: 0.0 for k in AREAS}
    linhas = []
    for area, num, titulo, regra in ITENS:
        if callable(regra):
            nota, evid = regra(raiz)
        else:
            nota, evid = regra
        por_area[area] += nota
        linhas.append({"area": area, "n": num, "item": titulo,
                       "nota": nota, "evidencia": evid})

    for area, rotulo in AREAS.items():
        itens = [x for x in linhas if x["area"] == area]
        print(f"=== {rotulo}: {por_area[area]:.1f} / 7 ===")
        for x in itens:
            marca = {0.0: "  ", 0.5: "½ ", 1.0: "✓ "}[x["nota"]]
            print(f"  {marca}{x['n']}. {x['item'][:52]:<54}{x['evidencia'][:56]}")
        print()

    nota = min(por_area.values())
    pior = [a for a, v in por_area.items() if v == nota]
    print("=" * 78)
    print(f"NOTA (o MINIMO das quatro areas, fonte 3): {nota:.1f}")
    print(f"  area que segura: {', '.join(AREAS[a] for a in pior)}")
    print(f"  leitura: {leitura(nota)}")
    print(f"  media (que NAO e a nota, e serve so de contraste): "
          f"{sum(por_area.values()) / 4:.1f}")
    print("=" * 78)

    # Uma linha resumindo o VENCIMENTO, para o vigia semanal poder escalar.
    # Sem isto a expiracao seria decadencia silenciosa: a nota cairia sozinha
    # daqui a 90 dias e ninguem saberia — o defeito que esta regua ataca.
    vencidas = [x for x in linhas if x["evidencia"].startswith(("VENCIDA", "SEM DATA"))]
    if recusa:
        print(f"ATENCAO: declaracoes RECUSADAS ({recusa}) — a nota abaixo ignora todas.")
    elif vencidas:
        print(f"ATENCAO: {len(vencidas)} declaracao(oes) VENCIDA(s) ou sem data — "
              f"a nota caiu sozinha ate alguem reverificar:")
        for x in vencidas:
            print(f"         {x['area']} {x['n']}. {x['item']}")
    else:
        print("todas as declaracoes estao dentro da validade "
              f"({VALIDADE_DECLARACAO_DIAS} dias).")
    print()

    destino = raiz / str(args.out)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "nota": nota, "por_area": por_area, "itens": linhas,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[salvo] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
