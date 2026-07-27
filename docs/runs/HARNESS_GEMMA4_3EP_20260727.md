# Harness VLA Gemma 4 12B — avaliação headless de 3 episódios

## Classificação

- **paper-confirmed:** planner emite uma primitiva JSON por turno e recebe nova observação após a execução.
- **paper-compatible:** transporte local via Ollama e desativação do reasoning interno para preservar a saída JSON executável.
- **beta-only:** CoppeliaSim 4.1 em Xvfb/Mesa `llvmpipe`, captura de GIF e `vla_act` scripted.

O paper não especifica protocolo HTTP nem exige exposição do reasoning interno. A opção `disable_thinking` usa a API nativa `/api/chat` do Ollama com `think=false`; o contrato de saída do planner não muda.

## Identidade da run

- run: `harness_demo_3ep_20260727_183547`
- estado: completa
- data: 2026-07-27
- commit-base: `ff8d7c7`
- modelo: `gemma4:12b` (Ollama 0.32.4, 7.6 GB no disco, aproximadamente 9.9 GiB carregado)
- inferência: CPU, contexto 4096, temperatura 0, `max_tokens=1024`, `think=false`
- simulador: CoppeliaSim Pro 4.1.0 + PyRep 4.1.0.3
- renderização: Xvfb 1024x768x24 + Mesa llvmpipe OpenGL 4.5, sem display físico
- resolução capturada: 256x256 RGB
- eval set: `base`
- índices selecionados: `[0, 15, 38]`
- budgets: 12 turns e 30 env steps por episódio
- duração: 1,197.64 s (20 min 02 s)

A GPU não foi usada porque NVML reportou incompatibilidade entre driver e biblioteca. A máquina tinha 125 GiB de RAM e 1.4 TiB de disco livre; o modelo coube confortavelmente em CPU.

## Resultado agregado

| Métrica | Resultado |
|---|---:|
| Tarefas resolvidas (`task_success`) | 1/3 |
| Success rate | 33.33% |
| Turns | 32 |
| Env steps | 67 |
| `action_success` | 67/67 |
| Erros de formato | 0 |
| Rejeições semânticas | 0 |
| Rejeições por falta de progresso | 0 |
| `move_to` | 19 (59.38%) |
| `vla_act` | 13 (40.62%) |

Outcomes de `vla_act`: três `grasp_verified`, seis `grasp_unverified`, um `empty_grasp` e três places executados com status `success`. `action_success` significa que o simulador aceitou o movimento; não prova grasp, placement ou conclusão da tarefa.

## Resultado por episódio

| Índice | Instrução | Sucesso | Turns | Steps | Tempo |
|---:|---|---:|---:|---:|---:|
| 0 | Pick up the star and place it into the silver container. | 1 | 8 | 16 | 288.73 s |
| 15 | Stack the maroon cylinder and the navy cylinder in sequence. | 0 | 12 | 27 | 478.85 s |
| 38 | Wipe the horizontal area. | 0 | 12 | 24 | 430.07 s |

### Episódio 0: sucesso oficial com ressalva física

O planner produziu cinco `move_to` e três tentativas de grasp. As três foram classificadas como `grasp_unverified`; na última, o fechamento da garra deslocou o objeto para a região de sucesso e o benchmark retornou `task_success=1`. Portanto, o resultado conta como sucesso oficial, mas não demonstra grasp verificado nem a sequência grasp-transport-place pretendida.

### Episódio 15: sequência de stacking incorreta

Foram três grasps verificados e três places executados, sem erro de formato ou contrato. Mesmo assim, o predicado final permaneceu falso. O planner repetiu manipulação de objetos já usados e esgotou os 12 turns. Classificação principal: planejamento/sequência inadequada; contato e placement também permanecem fatores possíveis.

### Episódio 38: grasp da ferramenta não obtido

O planner alternou oito `move_to` e quatro tentativas de grasp da esponja. Três ficaram `grasp_unverified` e uma foi `empty_grasp`; a fase de wiping nunca começou. Classificação principal: execução física/contato, seguida de planejamento repetitivo até o budget.

## Artefatos

Diretório local, ignorado pelo Git:

`EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_20260727_183547/base/`

Conteúdo:

- `results/summary.json`;
- três `episode_N_res.json`;
- três `trace_episode_N.jsonl`;
- 70 PNGs (6,628,719 bytes);
- três GIFs (1,931,236 bytes):
  - `images/episode_1/episode_1.gif`;
  - `images/episode_2/episode_2.gif`;
  - `images/episode_3/episode_3.gif`.

Também foi executado um smoke físico determinístico sem LLM. Ele concluiu 7/7 ações, obteve attachment verificado, `task_success=1` e salvou 16 frames em `EmbodiedBench/running/grasp_feedback_physics_1ep/20260727_181838/`.

## Interpretação

O Gemma 4 12B eliminou erros de saída nesta amostra e mostrou recuperação após feedback, mas 3 episódios não sustentam alegação de melhoria estatística sobre a baseline histórica de 10 episódios com Qwen. Os conjuntos e tamanhos são diferentes. O resultado confirma que o pipeline headless, o modelo local e o contrato JSON funcionam; os principais gargalos continuam sendo contato real do mock `vla_act`, sequência multiobjeto e detecção de progresso físico.
