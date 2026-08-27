# RETOMAR — Gabinete Dota, depois de formatar o PC

Escrito em 27/08/2026. **O plano completo, com as frentes e a próxima ação, está
no repositório do LoL, em `RETOMAR.md` na raiz de `BOLINAKT/gabinete-lol`.** Os
dois projetos andam juntos e a maioria das decisões vale para os dois.

Aqui fica só o que é específico do Dota.

## Reconstruir

Python 3.13. `git clone https://github.com/BOLINCRVG/gabinete-dota.git`, depois
`pip install -r requirements.txt`. O modelo (`ml_artifact.json`), os dados
(`data.csv`, `CSV_DOTA/`), o `live_models/` e as notas (`evidence/`) **já vêm no
clone** — conferido arquivo por arquivo em 27/08.

Nada aqui precisa de cópia manual. (No LoL precisa: `ops/baselines/`, ~68 MB.)

Provar que voltou inteiro:

```
python regua_28.py      # esperado: nota 2,0, segurada por INFRA e MONITORAMENTO
python frescor.py
python dota_scorecard.py
```

**A suíte do Dota tem 15 falhas e 30 erros conhecidos**, de antes desta rodada.
Não são regressão nova; nunca foram atacados.

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
