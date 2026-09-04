# Gabinete Dota — leia isto antes de qualquer coisa

Este arquivo é carregado sozinho em toda sessão nova nesta pasta. Ele existe
para que o assistente não precise ser reapresentado ao projeto.

## O que é

Modelo de apostas em Dota 2 profissional, irmão do **Gabinete LoL**
(`BOLINAKT/gabinete-lol`). Dados vindos da API da OpenDota, com esteira diária.
O dono é apostador profissional, não engenheiro: **explique em português claro,
sem jargão, e nunca peça que ele confira código para validar um número.**

**O plano de trabalho vive no repositório do LoL** — o foco atual é o LoL, e o
Dota entra depois. Ver lá: `MODELO_LOL.md` (a fila de trabalho) e `RETOMAR.md`.
Aqui, `RETOMAR.md` traz só o que é específico do Dota.

Regra que faz isso sair barato: **arquivo compartilhado fica idêntico nos dois
repositórios** (`regua_28.py`, `frescor.py`, `livro_caixa.py`,
`mlcore/calibration.py`).

## AS REGRAS DE TRABALHO — não negociáveis

1. **Uma frente por vez**, terminada por completo. Só avançar quando ele mandar.
2. **Pesquisar ANTES de propor.** Não sugerir a ideia primeiro: ir à internet,
   ver se já existe, como se faz, e se já se sabe que é bom (copiar) ou ruim
   (nem começar). Pesquisar de novo depois de implementado.
3. **Dúvida → pesquisar. "Eu acho que assim é melhor" → pesquisar.** Vale
   inclusive para consertar bug.
4. **A fonte fica ESCRITA no arquivo**, em bloco FONTES. Pesquisa que não ficou
   escrita conta como pesquisa que não aconteceu.
5. **A regra vale para o código que JÁ EXISTE, não só para o novo.** Toda peça
   que a frente tocar é auditada contra pesquisa: aquilo é mesmo o estado da
   arte? já se sabe que é ruim? existe jeito melhor conhecido? A resposta, com
   a fonte, entra no arquivo — inclusive quando a conclusão for "está certo,
   mantém". Peça sem fonte é peça não auditada, mesmo que funcione.
6. **Tudo testado. Só passa o que for medidamente melhor.** Teste cego separado
   do de calibração, intervalo de confiança, e a barra da casa: **ganho > 0,002
   de logloss, p < 0,05, limite inferior do intervalo acima de zero — as três
   juntas.** Nunca medir no conjunto usado para calibrar.
7. **Nota nunca sai da cabeça**: programa reproduzível, com carimbo de quando
   foi medida. Em mercado binário, mostrar o ganho **sobre a taxa-base**.

## Fora do plano: CLV

Decisão dele, por falta de tempo. "Melhorou" só pode significar **"prevê
melhor"**, nunca "é lucrativo". Não reabrir sozinho.

## O que já está medido e FECHADO aqui (não refazer)

- **Identidade de time = `team_id` + elenco, nunca o nome.** Maior ganho do
  Dota: no TI vai de 0,68 para 6,29.
- **Calibrar o modelo ao vivo PIORA** — perdeu 15 de 15. Não insistir.
- **Totais: só kills tem sinal** (+0,0139 sobre a taxa-base). Barracas são
  piores que a taxa-base. Teto declarado; o caminho seguinte é prever a duração.
- **Draft e jogadores bloqueados por dado**: 60 de ~2.000 partidas têm draft.
- **`premium` ≠ `professional`**: o TI inteiro ficava fora do treino por isso.

## Armadilhas que já morderam (não repetir)

- **Raiz do repositório**: nunca `Path(__file__).parents[N]` — aqui as
  ferramentas vivem na raiz e no LoL vivem em `tools/`, então isso resolvia para
  a Área de Trabalho e media a pasta errada por semanas. Usar
  `git rev-parse --show-toplevel`.
- **Afirmação sobre um repo dentro de código compartilhado** vira crédito
  emprestado. Declaração é dado local e nomeia o seu dono.
- **Divergência treino/serviço** já apareceu aqui (`scope_meta`) e não há vigia.
- **Acento no caminho da pasta cega o modelo ao vivo, e a suíte MENTE.** O
  LightGBM não abre arquivo por caminho absoluto que tenha acento. Com o repo
  em `...\Área de Trabalho\...` a suíte dava 30 falhas em vez de 15, e o
  retreino diário morria com `Model file ... is not available for writes` — que
  parece permissão e não é. Medido em 04/09/2026. **O projeto tem que morar em
  caminho sem acento** (hoje: `C:\Gabinetes\GabineteDOTA`).
- **RODAR A SUÍTE RETREINA OS MODELOS DE PRODUÇÃO.** Um teste chama
  `launch_retrain_background` contra o `live_models/` real em vez de uma pasta
  temporária, e ele sobrescreve os 12 arquivos dos 4 minutos. Não há chave de
  ambiente para desligar. **Copie `live_models/` antes de rodar a suíte** (são
  1,8 MB). Enquanto o repo esteve em caminho com acento isso passava
  despercebido, porque o acento é que impedia a escrita.
- **`git checkout` NÃO restaura modelo do jeito que você espera.** Este repo tem
  `core.autocrlf = true` e nenhum `.gitattributes`, então o checkout reescreve
  `lgb_model.txt` com CRLF e engorda o arquivo em ~2.200 bytes. Para voltar ao
  estado exato, restaure de uma cópia física, não do git.

## Comandos

```
python regua_28.py        # confiabilidade do sistema (hoje: 2,0)
python frescor.py         # as notas são de quando?
python dota_scorecard.py  # a nota por torneio
```

A suíte roda 4.674 testes em ~2 min e tem **15 falhas e 30 erros conhecidos**,
anteriores a agosto/2026. Não são regressão nova e nunca foram atacados.

**Esse 15 só aparece com `GABINETE_VIEWER=0`.** O `.streamlit/secrets.toml`
deste repo tem `GABINETE_VIEWER = "1"`, e o modo visualização bloqueia
subprocesso — o que faz 7 testes do `run_cmd` falharem sem que nada esteja
quebrado. Rode assim:

```
set GABINETE_VIEWER=0 && python -m unittest discover -s mlcore\tests -p "test_*.py"
```

Medido em 04/09/2026, e as quatro combinações ficam assim:

| caminho | viewer | falhas | erros |
|---|---|---|---|
| com acento | 1 | 30 | 30 |
| com acento | 0 | 23 | 30 |
| sem acento | 1 | 22 | 30 |
| **sem acento** | **0** | **15** | **30** |

Antes de rodar, leia a armadilha do retreino acima e copie `live_models/`.
