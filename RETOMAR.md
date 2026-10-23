# RETOMAR — Gabinete Dota, depois de formatar o PC

Escrito em 27/08/2026. **O plano completo, com as frentes e a próxima ação, está
no repositório do LoL, em `RETOMAR.md` na raiz de `BOLINAKT/gabinete-lol`.** Os
dois projetos andam juntos e a maioria das decisões vale para os dois.

Aqui fica só o que é específico do Dota.

## Onde isto mora agora — 04/09/2026

`C:\Gabinetes\GabineteDOTA`. **Saiu da Área de Trabalho de propósito**: acento no
caminho faz o LightGBM não abrir modelo nenhum, o retreino morrer com um erro que
parece de permissão, e a suíte dar 30 falhas em vez de 15. Ver a armadilha no
`CLAUDE.md`. Não mover de volta.

## Reconstruir

Python **3.13** (em 04/09/2026 rodou com a 3.13.15; não passar para 3.14).
Clonar **em pasta sem acento**:

```
git clone https://github.com/BOLINCRVG/gabinete-dota.git C:\Gabinetes\GabineteDOTA
cd C:\Gabinetes\GabineteDOTA
pip install -r requirements.txt
```

Aqui **não há `.venv`**: os pacotes vão no Python do sistema. Consequência a
saber — qualquer `pip install` global futuro pode mexer nas versões de que este
projeto depende.

O modelo (`ml_artifact.json`), os dados (`data.csv`, `CSV_DOTA/`), o
`live_models/` e as notas (`evidence/`) **já vêm no clone** — conferido arquivo
por arquivo em 27/08.

O que **não** vem no clone: `.streamlit/secrets.toml` (está no `.gitignore`, e é
onde moram as chaves da OpenDota e da PandaScore). Ele sobreviveu à mudança de
27/08 só porque a pasta inteira foi copiada. Guarde-o à parte.

Provar que voltou inteiro:

```
python regua_28.py      # esperado: nota 2,0, segurada por INFRA e MONITORAMENTO
python frescor.py
python dota_scorecard.py
```

## A suíte — leia antes de rodar

**Rodar a suíte RETREINA os modelos de produção.** Um teste chama
`launch_retrain_background` contra o `live_models/` real, e ele sobrescreve os 12
arquivos dos 4 minutos. **Copie `live_models/` antes** (1,8 MB). A cópia de
04/09/2026 está em `C:\Gabinetes\_backup_dota_pre_move_20260904\`.

Para restaurar, **use a cópia física, não `git checkout`**: este repo tem
`core.autocrlf = true` e nenhum `.gitattributes`, então o checkout reescreve
`lgb_model.txt` com CRLF e engorda o arquivo em ~2.200 bytes. Confira por md5.

**15 falhas e 30 erros conhecidos**, de antes de agosto/2026 — não são regressão
nova e nunca foram atacados. Mas esse número **só aparece com
`GABINETE_VIEWER=0`**, porque o `secrets.toml` daqui liga o modo visualização, e
ele bloqueia subprocesso — o que derruba 7 testes do `run_cmd` sem que nada
esteja quebrado:

```
set GABINETE_VIEWER=0 && python -m unittest discover -s mlcore\tests -p "test_*.py"
```

Medido em 04/09/2026 (4.674 testes, ~2 min):

| caminho | viewer | falhas | erros |
|---|---|---|---|
| com acento | 1 | 30 | 30 |
| com acento | 0 | 23 | 30 |
| sem acento | 1 | 22 | 30 |
| **sem acento** | **0** | **15** | **30** |

Os 4 `lgb_model.txt` aparecem como modificados no `git status` desde 04/09.
**Os arquivos em disco são os originais** (md5 conferido); o que mudou foi o
relato do git. Eles sempre diferiram do commitado por 2 bytes — as duas últimas
linhas terminam em CRLF e o resto em LF — e o git não mostrava porque o cache de
`stat` dele nunca comparou o conteúdo. Não commitar por reflexo. Detalhe completo
na dívida D7 do `RETOMAR.md` do LoL.

## Modelo congelado, engenharia fechada — 22 e 23/10/2026

O que entrou e o que foi medido está na seção 2-F do `RETOMAR.md` do LoL.
Específico daqui: contrato de dados (`tools/contrato_dados_dota.py`), canário
(`tools/canario_modelo.py`) e trava de qualidade (`tools/portao_qualidade_dota.py`)
no `dados_diarios_dota.yml` — se o canário ou a trava barram, os DADOS saem e o
MODELO anterior fica. Os modelos ao vivo retreinam toda segunda
(`retreino_ao_vivo_dota.yml`). A suíte não retreina mais os modelos ao vivo de
produção (`GABINETE_SEM_TREINO=1`).

**Achado não mexido (modelo congelado):** depois do corte, a probabilidade crua
dá logloss 0,70 mesmo aprendendo jogo a jogo, pior que moeda; congelar 30 dias
custa +0,041 (`evidence/envelhecimento_dota.json`). A nota 0,57 do
`dota_scorecard.py` é dentro da amostra. Primeira coisa a olhar quando a frente
de acertividade voltar.

## Estado

- **Régua das 28: 2,0.** Áreas que seguram: INFRAESTRUTURA e MONITORAMENTO.
  As 20 declarações em `evidence/regua_28_declaracoes.json` foram verificadas
  contra ESTE repositório em 27/08 e **vencem em 90 dias** — quando vencerem, a
  nota cai sozinha e a esteira diária abre uma issue avisando. Isso é de
  propósito: carimbo que ninguém confere é decoração.
- Onde o Dota é **melhor** que o LoL: tem semente fixa no treino
  (`dota_ml.py:382`, `random_state=7`), e protege as features `delta_*` com
  `math.isfinite` antes de subtrair.
- Onde é **pior**: o pipeline persiste `account_id` do Steam
  (`eval_whr_dota.py` monta `{team_id: roster:[account_id]}`). Isso é
  identificador de conta de uma pessoa, não pseudônimo profissional, e não há
  controle de privacidade nenhum.

## O que já foi medido e fechado aqui

- **Identidade de time = `team_id` + elenco, nunca o nome.** Foi o maior ganho
  do Dota: o TI vai de 0,68 para 6,29.
- **Calibrar o modelo ao vivo PIORA** — perdeu em 15 de 15 células. O que ganha
  é parar o treino pelo refinamento (+0,00193, IC acima de zero), mas isso não
  passa na barra de aceite da casa.
- **Totais: só kills tem sinal** (+0,0139 sobre a taxa-base). Barracas são
  piores que a taxa-base. Teto declarado após 3 ataques; o caminho seguinte é
  prever a duração da partida (frente 10 no plano do LoL).
- **Draft e jogadores estão bloqueados por dado**: 60 de ~2.000 partidas têm
  draft. O desenho recomendado é estimar pares de heróis em partidas PÚBLICAS e
  usar as profissionais só para força de time.
