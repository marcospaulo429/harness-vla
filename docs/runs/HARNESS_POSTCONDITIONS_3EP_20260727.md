# Harness beta: pos-condicoes fisicas em 3 episodios

## Classificacao

- **paper-compatible:** feedback fechado separado de sucesso da tarefa e estado
  fisico estruturado. A fonte primaria indicada nos documentos locais como
  `arXiv:2607.08448v2` nao foi resolvida externamente nesta etapa.
- **beta-only:** tolerancias de `2.0` voxels para `move_to` e `12.0` voxels para
  `place`, attachment PyRep como evidencia autoritativa e formato das metricas.

Esta avaliacao mede pos-condicoes do mock scripted. Nao valida um VLA aprendido.

## Identidade

- run: `harness_demo_3ep_20260727_214125`;
- commit avaliado: `92f5555`;
- modelo: `gemma4:12b`, Ollama CPU, temperatura 0, `think=false`;
- episodios: `[0, 15, 38]`;
- runtime: CoppeliaSim 4.1 + PyRep, Xvfb/Mesa llvmpipe;
- duracao: 1.206 s (20 min 06 s);
- estado: completa, com traces, sidecars, overlays, frames e tres GIFs.

## Resultado agregado

| Metrica | Resultado |
|---|---:|
| `task_success` | 1/3 |
| Turns / env steps | 32 / 60 |
| Parse / compile / semantic rejects | 0 / 0 / 0 |
| Env steps com `action_success=1` | 60/60 |
| Pos-condicoes verificadas | 25/32 |
| Pos-condicoes falhas | 7/32 |
| `postcondition_met` | 25 |
| `unverified` | 6 |
| `target_pose_not_reached` | 1 |
| Observacoes RGB-D | 264 |
| Erro RGB-D medio / maximo | 1,74 cm / 5,76 cm |

O resultado confirma o objetivo da etapa: uma acao aceita pelo simulador nao
prova o resultado fisico da primitiva. `action_success` continua sendo a metrica
oficial por env step; `primitive_postcondition_met` e a metrica por primitiva.
As duas contagens possuem denominadores diferentes e nao devem ser somadas ou
comparadas como se fossem a mesma unidade.

## Resultado por episodio

| Episodio | Task | Turns | Steps | Pos-condicao | Estado final |
|---:|---:|---:|---:|---:|---|
| 0 | 1 | 8 | 16 | 4 ok / 4 falhas | `held=null`, `placed=[]` |
| 15 | 0 | 12 | 26 | 11 ok / 1 falha | `held=object 5`, `placed=[object 7]` |
| 38 | 0 | 12 | 18 | 10 ok / 2 falhas | `held=null`, `placed=[]`, `remaining=[object 9]` |

### Episodio 0: pick and place

- quatro `move_to` chegaram exatamente ao voxel comandado;
- um `move_to` foi aceito, mas terminou a `15,81` voxels do alvo;
- tres grasps foram `grasp_unverified` por `wrong_object_attached`;
- o predicado oficial marcou `task_success=1` durante o ultimo fechamento, sem
  uma primitiva `place` verificada;
- por isso `placed=[]` nao e sobrescrito pelo sucesso oficial.

A divergencia e preservada como diagnostico: benchmark task success, attachment
e estado local medem contratos diferentes. Nao ha evidencia suficiente para
forcar o tracker a declarar placement.

### Episodio 15: stacking

- tres grasps foram verificados por attachment;
- o primeiro place desapegou o objeto, mas ficou sem evidencia espacial valida e
  permaneceu `unverified`;
- um `release` separado confirmou desapego;
- o segundo place foi verificado com distancia de `3,32` voxels ao destino;
- estado final: `object 7` colocado e `object 5` ainda segurado, coerente com
  `task_success=0`.

`remaining` lista todos os objetos com papel `manipulable`, inclusive
manipulaveis irrelevantes para a instrucao. Isso e estado de cena, nao uma lista
inferida de metas da linguagem.

### Episodio 38: wiping

- dez movimentos cumpriram a pose comandada;
- dois grasps ficaram `unverified` por geometria ambigua;
- a esponja permaneceu em `remaining` e a tarefa terminou sem sucesso.

## Comparacao com a etapa RGB-D

Run anterior: `harness_demo_3ep_20260727_205436`.

| Metrica | RGB-D | Pos-condicoes |
|---|---:|---:|
| `task_success` | 1/3 | 1/3 |
| Turns | 32 | 32 |
| Env steps | 67 | 60 |
| Parse / compile errors | 0 / 0 | 0 / 0 |
| Erro RGB-D medio | 1,82 cm | 1,74 cm |

O feedback novo alterou a sequencia do planner nos episodios 15 e 38, reduzindo
steps sem melhorar `task_success`. Isso e comportamento diferente, nao evidencia
de melhoria de politica. A capacidade nova aprovada e explicar divergencias
fisicas por primitiva.

## Gate

Etapa aprovada:

- `move_to` distingue pose aceita de pose alcancada;
- grasp, release e place exigem evidencia apropriada;
- `held`, `placed` e `remaining` aparecem em cada turn e no episodio;
- `action_success`, pos-condicao e `task_success` permanecem separados;
- todos os sete falsos sucessos de primitiva observados possuem razao explicita;
- suite local: `100 passed`; diagnosticos e `git diff --check` limpos;
- smoke de um episodio e run comparavel de tres episodios completaram.

Riscos restantes:

- `remaining` ainda e scene-level, nao instruction-level;
- o episodio 0 mostra desacordo entre predicado oficial e attachment;
- `sim_mask` e attachment PyRep continuam sinais privilegiados beta-only;
- thresholds ainda nao foram calibrados em uma amostra maior.
# Harness beta: pos-condicoes fisicas em 3 episodios

## Classificacao

- **paper-compatible:** feedback fechado separando execucao, pos-condicao da
  primitiva e sucesso oficial da tarefa. A fonte primaria indicada nos
  documentos locais como `arXiv:2607.08448v2` nao estava resolvivel durante esta
  etapa, portanto nenhuma alegacao nova foi promovida a paper-confirmed.
- **beta-only:** tolerancias de 2 voxels para `move_to`, 12 voxels para `place`,
  attachment PyRep como evidencia autoritativa e estado `held/placed/remaining`.

## Identidade

- run: `harness_demo_3ep_20260727_214125`;
- commit: `92f5555`;
- modelo: Gemma 4 12B, Ollama CPU, temperatura 0, `think=false`;
- episodios: `[0, 15, 38]`;
- budgets: 12 turns e 30 env steps;
- runtime: CoppeliaSim 4.1, PyRep, Xvfb e Mesa llvmpipe;
- duracao de parede: 20 min 06 s;
- estado: completa, com traces, sidecars, overlays, frames e GIFs.

## Hipotese e gate

Hipotese: `action_success` do simulador pode aceitar uma acao 7-D sem que o
resultado fisico pretendido da primitiva ocorra. Pose final, attachment e
re-grounding RGB-D devem tornar essa divergencia explicita sem mudar trajetorias.

Resultado: aprovado. Todos os 60 env steps tiveram `action_success=1`, mas 7 das
32 primitivas falharam sua pos-condicao. A nova metrica detectou essas falhas sem
confundi-las com `task_success`.

## Resultado por episodio

| Episodio | Tarefa | Task success | Turns | Steps | Pos-condicao | Duracao |
|---:|---|---:|---:|---:|---:|---:|
| 0 | pick star | 1 | 8 | 16 | 4/8 | 283,12 s |
| 15 | stack cylinders | 0 | 12 | 26 | 11/12 | 483,45 s |
| 38 | wipe area | 0 | 12 | 18 | 10/12 | 434,38 s |
| **Total** | - | **1/3** | **32** | **60** | **25/32** | **1.200,95 s** |

A run RGB-D anterior (`harness_demo_3ep_20260727_205436`) tambem obteve 1/3,
32 turns e zero erros de parse/compile. Os env steps cairam de 67 para 60 porque
o novo feedback alterou as decisoes do LLM; isso nao e evidencia de melhora de
tarefa.

## Pos-condicoes por tipo

| Tipo | Cumpridas | Falhas | Evidencia |
|---|---:|---:|---|
| `move_to` | 20 | 1 | distancia EE-comando <= 2 voxels |
| `vla_act:grasp` | 3 | 5 | attachment alvo ou fallback geometrico |
| `vla_act:place` | 1 | 1 | detach e objeto a <= 12 voxels do destino |
| `release` | 1 | 0 | abertura executada e nenhum attachment |

Razoes agregadas:

- `postcondition_met`: 25;
- `unverified`: 6;
- `target_pose_not_reached`: 1.

O caso discriminante ocorreu no episodio 0, turn 6: o simulador retornou
`action_success=1`, mas o end-effector terminou a `15,81` voxels do alvo. O
trace registrou `target_pose_not_reached` e realimentou isso ao planner.

## Estado fisico

O trace registra, apos cada primitiva:

- `held`: objeto com grasp verificado;
- `placed`: objeto com place local verificado;
- `remaining`: todos os objetos da cena com papel `manipulable` que nao estao
  segurados nem colocados.

Estado final observado:

- episodio 0: nenhum objeto segurado ou colocado;
- episodio 15: `object 5` segurado e `object 7` colocado;
- episodio 38: nenhum objeto segurado ou colocado; `object 9` restante.

`remaining` e scene-wide, nao task-specific. Distratores manipulaveis aparecem
nessa lista. Alem disso, `placed` representa somente uma primitiva `place`
verificada; o episodio 0 terminou com `task_success=1` durante um grasp e por
isso nao gerou um registro `placed`. Isso e uma diferenca semantica documentada,
nao prova erro no predicado oficial do benchmark.

## Diagnostico

1. A separacao `env_step action_success` / `primitive postcondition` /
   `task_success` agora e observavel e nao colapsada.
2. No wiping, os dois grasps ficaram `ambiguous_geometry`; 10 movimentos aceitos
   nao compensaram a ausencia de grasp verificado.
3. No stacking, um place foi localmente valido a `3,32` voxels do destino, mas a
   tarefa permaneceu incompleta. Proximidade local nao substitui o predicado de
   sequencia completo.
4. Os thresholds sao instrumentacao beta. Esta run valida o contrato e nao
   calibra valores otimos.

## Artefatos

Diretorio local:

`EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_20260727_214125/base/`

Arquivos principais:

- `results/summary.json`;
- `results/episode_{1,2,3}_res.json`;
- `results/trace_episode_{1,2,3}.jsonl`;
- `images/episode_N/episode_N.gif`;
- `grounding/episode_N/` com sidecars e overlays RGB-D.

## Decisao

Etapa 2 aprovada por capacidade mensuravel antes ausente e sem regressao de
`task_success`, parse ou compile. O proximo incremento e persistencia incremental
do trace com flush e reconstrucao apos interrupcao.
