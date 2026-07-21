# Grounding e grasp — diagnóstico de 21/07/2026

## Objetivo

Separar falhas do planner, do contrato semântico e da execução física usando o
mesmo episódio `base`, índice 0, em modo headless.

## 1. LLM antes do guard semântico

Run: `harness_grounding_grasp_1ep_20260721_110703`.

- 12 turns e 30 passos do ambiente;
- 1 `compile_error`;
- zero reward e `task_success=0`;
- o modelo insistiu em `place` legado com `target`, sem grasp verificado;
- duração do episódio: 18,60 s.

Interpretação: ações aceitas pelo simulador não provaram contato nem progresso.

## 2. LLM após o guard semântico

Run: `harness_grounding_grasp_1ep_20260721_111642`.

- 12 turns, 12 rejeições semânticas e zero passos do ambiente;
- todas as ações inválidas foram bloqueadas antes de `env.step`;
- `task_success=0`.

Interpretação: o guard eliminou movimento inútil, mas o Qwen 0.5B não conseguiu
corrigir o contrato canônico `object + destination` a partir do feedback. Segurança
melhorou; capacidade de planejamento não.

## 3. Diagnóstico físico determinístico

Run: `grasp_feedback_physics_1ep/20260721_112933`.

O runner sem LLM executou exatamente um grasp e um place canônicos:

- objeto: `object 3`, estrela física `star_normal0`;
- destino: `object 1`, primeiro recipiente;
- quatro subações de grasp e três de place;
- attachment apareceu após fechar a garra e permaneceu após o lift;
- lift do objeto: 5 voxels;
- lift da garra: 6 voxels;
- distância final objeto–garra: aproximadamente 1,73 voxels;
- residual de co-motion: aproximadamente 1,41 voxels;
- attachment desapareceu após release;
- `task_success=1`;
- duração total: aproximadamente 6,74 s.

## Conclusão

**Evidência confirmada:** o pipeline físico scripted consegue aproximar, anexar,
elevar, transportar e soltar o objeto correto quando recebe bindings corretos.

**Evidência confirmada:** o guard semântico impede `place` inválido de consumir
passos físicos.

**Hipótese principal:** no protocolo LLM atual, o gargalo dominante é o planner de
0,5B e sua aderência ao contrato, não a trajetória física determinística.

Não se declara melhora de taxa de sucesso da política: o sucesso físico foi um
teste determinístico isolado, não uma avaliação do planner.
