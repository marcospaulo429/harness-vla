# Plano de implementação: Harness VLA multi-turn no LIBERO

**Estado:** em implementação  
**Início:** 2026-08-06  
**Commit-base:** `5f88f28`  
**Fonte científica:** arXiv:2607.08448v2, §2.1–2.3, §3.3 e Apêndices A–C, E.1–E.3

Este é o documento vivo do primeiro loop funcional do Harness VLA no LIBERO.
Cada etapa deve registrar aqui o contrato implementado, arquivos alterados,
testes, runs e limitações. O objetivo não é codificar uma sequência fixa, mas
permitir que o planner escolha uma primitiva, espere sua pós-condição, receba
feedback físico e escolha a próxima até sucesso oficial ou budget.

## 1. Classificação científica

### Paper-confirmed

- o planner emite uma primitiva por turno;
- o worker executa somente essa primitiva e devolve observação e diagnóstico;
- o planner espera a pós-condição antes de emitir o próximo comando;
- `vla_act` executa contato em chunks com early return por `tau`;
- primitivas analíticas executam a estrutura fora do contato;
- pós-condição local não equivale a `task_success`;
- o episódio termina apenas no predicado oficial ou em budget/erro explícito.

### Paper-compatible

- loop síncrono no mesmo processo antes do REPL mediado por arquivos;
- planner escolhe alvo nominal e modo; worker resolve `xyz` pelo RGB-D;
- alturas de aproximação e release, tolerâncias e budgets concretos;
- resumo de feedback, guards semânticos e classificação de falha recuperável;
- biblioteca inicial reduzida a `vla_act`, `move_to` e `release`.

### Beta-only

- máscara de instância e contato privilegiados do simulador;
- métricas oracle separadas do payload do planner;
- instrumentação adicional para diagnosticar sequência e recuperação.

## 2. Invariantes

1. O planner emite exatamente um objeto JSON e uma primitiva por turno.
2. O planner nunca emite torques, joints, chunks VLA ou coordenadas oracle.
3. O planner só pode referenciar alvos grounded apresentados no turno.
4. `move_to` executável recebe `xyz` finito no frame mundo, resolvido pelo worker.
5. Transporte de objeto exige gripper explicitamente fechado (`+1`).
6. `release` encapsula a convenção nativa e envia gripper aberto (`-1`).
7. `primitive_success`, `tau_satisfied`, `env_done` e `task_success` são campos distintos.
8. Cada turno reobserva o ambiente antes de o planner decidir novamente.
9. `max_turns`, horizon global e budgets locais são verificados antes da execução.
10. O sucesso final vem somente de `env.check_success()` ou predicado oficial equivalente.

## 3. Vocabulário inicial

### `vla_act`

```json
{
  "action": "vla_act",
  "prompt": "pick up and lift the black bowl",
  "target": "akita_black_bowl_1",
  "max_chunks": 20,
  "tau": "lift_and_grasp"
}
```

### `move_to`

```json
{
  "action": "move_to",
  "target": "plate_1",
  "mode": "above",
  "gripper": "close"
}
```

Modos iniciais permitidos:

- `above`: destino grounded mais clearance configurado;
- `release_pose`: destino grounded mais altura de release configurada.

### `release`

```json
{"action": "release"}
```

Não adicionar `grasp`, `place`, `verify`, `push` ou outra primitiva fora do
vocabulário publicado. A sequência esperada de pick-and-place é uma propriedade
avaliada no trace, nunca uma máquina de estados imposta ao planner.

## 4. Estado apresentado ao planner

O planner recebe somente estado semântico e feedback resumido:

```json
{
  "instruction": "official task instruction",
  "grounded_targets": ["akita_black_bowl_1", "plate_1"],
  "holding": "akita_black_bowl_1",
  "last_action": "vla_act",
  "last_feedback": {
    "primitive_success": true,
    "reason": "lift_and_grasp_satisfied",
    "task_success": false
  },
  "budget": {
    "turns_remaining": 7,
    "actions_remaining": 164
  }
}
```

Poses, geometrias MuJoCo, máscaras, contatos brutos e métricas oracle ficam no
worker/trace e não entram no prompt do planner.

## 5. Feedback unificado

Cada turno persiste:

```json
{
  "turn": 1,
  "invocation": {},
  "feedback": {
    "action": "vla_act",
    "primitive_success": true,
    "task_success": false,
    "termination_reason": "lift_and_grasp_satisfied",
    "steps_executed": 53,
    "holding": "akita_black_bowl_1",
    "recoverable": false
  }
}
```

O trace detalhado da primitiva permanece aninhado no registro do turno. Imagens
não entram no JSONL.

## 6. Guards

- rejeitar ação fora de `vla_act`, `move_to`, `release`;
- rejeitar target ausente ou fora do grounding atual;
- rejeitar campos e enums inválidos;
- rejeitar `move_to` de transporte com gripper aberto;
- rejeitar `release` quando `holding` é nulo;
- rejeitar `vla_act` acima do cap restante;
- parar antes de executar se horizon ou turn budget acabou;
- não converter parse/compile/budget/env done em sucesso;
- registrar toda rejeição com raw output e razão.

## 7. Resolução geométrica inicial

Para `move_to(target, mode)`:

```text
above:
  xyz = grounded_target_xyz + [0, 0, approach_clearance_m]

release_pose:
  xyz = grounded_target_xyz + [0, 0, release_height_m]
```

Valores iniciais são configuração paper-compatible, registrados no manifest.
O destino é re-grounded imediatamente antes de cada `move_to`. Grounding
indisponível falha fechado; a primeira versão não reutiliza coordenada antiga.

## 8. Budgets e término

- `max_turns`: limite de decisões do planner;
- `horizon`: total de `env.step` após settling;
- `max_chunks`: cap local de `vla_act`, limitado pelo horizon restante;
- `max_move_steps`: cap local de `move_to`;
- `release_steps`: cap local de abertura do gripper.

Razões de término mínimas:

- `task_success`;
- `max_turns_exhausted`;
- `horizon_exhausted`;
- `planner_parse_error`;
- `primitive_compile_error`;
- `grounding_failure`;
- `grasp_lost`;
- `env_done`;
- `release_completed_task_incomplete`.

## 9. Etapas de implementação

### Etapa A — Contratos e planner

- [x] Parser estrito das três primitivas.
- [x] Prompt multi-turn com estado, feedback e budget.
- [x] Validação de target e enums.
- [x] Testes de parse, target spoofing e uma ação por turno.

### Etapa B — Release físico

- [x] Executor fechado por budget para abrir o gripper parado.
- [x] Pós-condição local separada de sucesso oficial.
- [x] Testes de convenção `-1`, env done, task success e budget.

### Etapa C — Worker multi-turn fake

- [x] Feedback unificado.
- [x] Loop com `max_turns` e horizon global.
- [x] Dispatch de `vla_act`, `move_to` e `release` por injeção de executores.
- [x] Trace JSONL por turno.
- [x] Testes de fluxo feliz e falhas recuperáveis.
- [ ] Manifest, episode e summary finais (Etapa E).

### Etapa D — Grounding e execução nativos

- [x] Re-grounding antes de cada `move_to`.
- [x] Resolução `above` e `release_pose` sem pose oracle.
- [x] Integração do executor OSC existente.
- [ ] Monitor de grasp durante transporte.
- [x] Vídeo contínuo de todas as fases.

### Etapa E — Runner e smoke real

- [x] CLI com budgets e offsets no manifest.
- [ ] GPU/Ollama/policy/disk gate.
- [ ] Health check eager do checkpoint na V100.
- [ ] Task 0/state 0, seed 7, uma run diagnóstica.
- [ ] Três repetições idênticas após validar a primeira.

### Etapa F — Expansão controlada

- [ ] Múltiplos initial states da task 0.
- [ ] Dez tarefas LIBERO-Spatial.
- [ ] Comparação pareada com VLA direta.
- [ ] Só depois: `move_pose`, rotações e lifecycle de memórias.

## 10. Critérios de aceitação do primeiro marco

1. Fake executa `vla_act -> move_to -> move_to -> release` em quatro turnos.
2. O planner recebe feedback do turno anterior, não pose física bruta.
3. A sequência não é hard-coded no evaluator.
4. Todos os budgets param antes de uma ação excedente.
5. Gripper fica fechado durante os dois `move_to` e abre no `release`.
6. `release` sem sucesso oficial gera novo turno se houver budget.
7. Trace reconstrói decisão, execução, pós-condição e estado final por turno.
8. Testes existentes continuam passando.

## 11. Diário de implementação

### 2026-08-06 — Planejamento

- Confirmado no paper: uma primitiva por turno, espera do worker, feedback e
  replanejamento são contratos explícitos.
- Confirmado no código: `vla_act`, grounding, `lift_and_grasp`, compilador OSC e
  executor de pose existem isoladamente.
- Lacunas locais: planner aceita somente `vla_act`; não há executor de release,
  feedback unificado ou loop multi-turn.
- Decisão: implementar primeiro contratos e fakes; nenhuma run pesada antes de
  a suíte completa passar.

### 2026-08-06 — Etapa A: planner multi-turn

- Adicionado `libero_multi_turn_planner.py`, separado do smoke legado.
- O parser aceita somente `vla_act`, `move_to` e `release`, uma por turno.
- `move_to` aceita alvo grounded, modo `above|release_pose` e gripper fechado;
  `xyz`, tolerâncias e campos extras vindos do LLM são rejeitados.
- O prompt inclui instrução, targets, feedback semântico e budget, sem pose do
  robô, máscara, contato bruto ou oracle.
- Teste focado: `17 passed`.

### 2026-08-06 — Etapa B: release físico

- Adicionado executor bounded que mantém os seis deltas de pose em zero e usa
  `-1` para abrir o gripper nativo.
- Release local completo retorna `primitive_success=true`, mas somente o
  predicado oficial pode produzir `task_success=true`.
- Teste focado do executor analítico: `15 passed`.

### 2026-08-06 — Etapa C: worker multi-turn injetável

- Adicionado loop síncrono com executores injetados, feedback semântico e um
  registro JSONL por decisão.
- O evaluator despacha a ação escolhida pelo planner; não contém sequência de
  pick-and-place.
- Horizon é convertido em cap local antes de cada chamada e a execução
  retornada é validada contra esse cap.
- `release` sem holding e transporte sem grasp são bloqueados antes de
  `env.step`; release incompleto pode reabrir o planejamento.
- Limitação intencional do primeiro marco: `move_to` é apenas transporte com
  objeto segurado. Movimento analítico pré-grasp fica fora desta fatia.
- Teste focado: `11 passed`; conjunto planner/executor/evaluator: `43 passed`.

### 2026-08-06 — Etapa D: adapters nativos

- Adicionados adapters para resolver target/modo por RGB-D, re-groundar antes
  de cada movimento e reutilizar o executor OSC com gripper fechado.
- O adapter VLA re-infere com observação atualizada, respeita o horizon no meio
  do chunk e atribui `holding` somente quando `lift_and_grasp` é satisfeito.
- Falhas transitórias de tau falham fechado e permanecem no trace.
- Teste focado: `11 passed`; conjunto relacionado: `77 passed`.
- Risco pendente: monitorar preservação bilateral do grasp durante transporte.

### 2026-08-06 — Etapa E leve: runner e artifacts

- Adicionados runner CLI e função transacional de episódio com manifest antes
  do reset, settling fora do horizon, trace, MP4, episode e summary.
- Exceções preservam artifacts parciais e marcam o manifest `incomplete`.
- O manifest marca corretamente `harness_complete=false`, pois esta fatia não
  possui biblioteca completa, memórias nem REPL por arquivos.
- Classificações paper-confirmed, paper-compatible e beta-only são registradas
  separadamente, sem classificar a run inteira como mecanismo do paper.
- Modelo padrão alinhado ao pedido: `gemma4:12b`, com `--think` explícito.
- Teste focado: `5 passed`; runner e planner após revisão: `22 passed`.

### 2026-08-06 — Gate pré-run

- Suíte completa após implementação leve: `336 passed`.
- `git diff --check`: passou.
- Hardware: 8x Tesla V100-SXM2-32GB, compute capability 7.0.
- GPU escolhida: índice físico 2, com 32.085 MB livres no gate.
- Ollama: `gemma4:12b` instalado, com capability `thinking`.
- Disco: 703 GB livres em `/home`.
- Nenhum processo CoppeliaSim/LIBERO e nenhuma policy na porta 8000.
- Policy server iniciado com `TORCH_COMPILE_DISABLE=1` e
  `TORCHDYNAMO_DISABLE=1`; health check ainda pendente.

## 12. Registro de validação

| Etapa | Validação | Resultado |
|---|---|---|
| Planejamento | `git diff --check` | passou |
| A | `pytest tests/test_libero_multi_turn_planner.py -q` | `17 passed` |
| B | `pytest tests/test_libero_analytic_executor.py -q` | `15 passed` |
| C | `pytest tests/test_libero_multi_turn_evaluator.py -q` | `11 passed` |
| A–C | testes relacionados | `43 passed` |
| D | `pytest tests/test_libero_native_multi_turn.py -q` | `11 passed` |
| D relacionado | adapters e módulos relacionados | `77 passed` |
| E leve | `pytest tests/test_libero_multi_turn_run.py -q` | `5 passed` |
| Revisão E | runner + planner | `22 passed` |
| Suíte completa | `pytest tests -q` | `336 passed` |
| Diff | `git diff --check` | passou |
| Preflight GPU/Ollama/disco/processos | comandos locais | passou |
