# Harness beta: avaliacao RGB-D de 3 episodios

## Classificacao

- **paper-confirmed:** observacao RGB-D calibrada e world coordinates com
  reobservacao entre decisoes.
- **paper-compatible:** `frame_id`, provenance, metricas contra oracle, overlays
  e sidecars JSON.
- **beta-only:** associacao pixel-objeto por `sim_mask` perfeita e backend
  `vla_act` scripted.

Esta run valida geometria e observabilidade. Ela nao e uma avaliacao de percepcao
visual independente nem de um VLA aprendido.

## Identidade

- run: `harness_demo_3ep_20260727_205436`;
- commit-base: `c9aecd0` com mudancas RGB-D ainda nao commitadas durante a run;
- modelo: `gemma4:12b`, Ollama, CPU, `think=false`, temperatura 0;
- simulador: CoppeliaSim 4.1 + PyRep em Xvfb/Mesa llvmpipe;
- episodios: `[0, 15, 38]`;
- budgets: 12 turns e 30 env steps;
- duracao: 1.221 s (20 min 26 s);
- estado: completa.

## Gate geometrico

Um smoke real comparou a projecao `pixel + depth + K + T_camera_world` com
`front_point_cloud` do PyRep no mesmo frame:

- objetos comparados: 7;
- erro medio: `7,85e-8 m`;
- erro maximo: `1,39e-7 m`;
- tolerancia do gate: `1e-5 m`;
- resultado: aprovado.

## Comparacao com baseline

Baseline: `harness_demo_3ep_20260727_183547`.

| Episodio | Sucesso baseline/RGB-D | Turns | Steps | Tempo baseline | Tempo RGB-D |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 / 1 | 8 / 8 | 16 / 16 | 288,73 s | 310,59 s |
| 15 | 0 / 0 | 12 / 12 | 27 / 27 | 478,85 s | 479,28 s |
| 38 | 0 / 0 | 12 / 12 | 24 / 24 | 430,07 s | 430,72 s |
| **Total** | **1 / 1** | **32 / 32** | **67 / 67** | **1.197,65 s** | **1.220,60 s** |

Foram identicos entre as runs:

- 32/32 primitivas;
- coordenadas voxel em todos os turns;
- outcomes de grasp/place;
- `task_success`;
- zero parse errors e zero compile errors.

Classificacao:

- comportamento: **neutro**, sem regressao;
- observabilidade: **melhora**, capacidade antes ausente;
- overhead total: aproximadamente 1,9%.

## Metricas RGB-D

| Episodio | Frames | Observacoes de objeto | Erro medio | Mediana | P95 | Maximo |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 119 | 1,61 cm | 1,37 cm | 2,69 cm | 3,53 cm |
| 15 | 28 | 91 | 1,18 cm | 1,13 cm | 1,60 cm | 1,72 cm |
| 38 | 25 | 75 | 2,94 cm | 2,32 cm | 5,45 cm | 5,76 cm |
| **Total** | **57** | **285** | **1,82 cm** | - | - | **5,76 cm** |

A metrica e `visible surface centroid -> simulator object origin`. Ela inclui
um vies geometrico esperado e nao deve ser interpretada como erro puro de sensor.
O oracle nunca e enviado ao planner.

## Diagnostico

1. O stacking falhou apesar de erro RGB-D estavel em torno de 1,2 cm. O gargalo
   continua sendo sequenciamento/estado do planner e contato do mock.
2. O wiping apresentou o maior erro geometrico, mas a falha principal continuou
   sendo a ausencia de grasp verificado da esponja.
3. No stacking, `object 6` desapareceu depois da manipulacao e `object 7` teve
   deslocamento grande enquanto segurado. O world map torna esses eventos
   mensuraveis, mas IDs ainda sao fornecidos pelo simulador.

## Artefatos

Diretorio local:

`EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_20260727_205436/base/`

Inclui:

- `results/summary.json`;
- tres resultados e tres traces JSONL;
- sidecar JSON e overlay PNG por frame em `grounding/episode_N/`;
- frames normais e GIFs em `images/episode_N/episode_N.gif`;
- configuracao completa da run.

## Decisao do gate

Etapa aprovada para avancar:

- geometria confirmada contra PyRep;
- provenance completo;
- oracle isolado do planner;
- retorno legado preservado;
- comportamento identico a baseline;
- suite leve passou antes da run.

O proximo incremento e pos-condicao uniforme e estado fisico estruturado. A
substituicao de `sim_mask` por percepcao visual real permanece pendente e nao e
necessaria para validar o proximo contrato.
