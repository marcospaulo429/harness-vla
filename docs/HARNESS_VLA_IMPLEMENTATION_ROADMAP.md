# Roadmap de implementação fiel ao Harness VLA v2

> Fonte obrigatória: arXiv:2607.08448v2. Antes de iniciar cada milestone, o
> `pesquisador-paper` deve reconfirmar a seção, o contrato e os detalhes que o
> paper deixa específicos ao benchmark.

## Classificação

- **paper-confirmed**: mecanismo e papel descritos pelo paper;
- **paper-compatible**: detalhe necessário, não especificado pelo paper;
- **beta-only**: instrumentação local, sem alegação de fazer parte do método.

## Protocolo para cada milestone

1. Verificação no paper v2.
2. Teste unitário/fake sem simulador.
3. Smoke test físico isolado, quando aplicável.
4. Um episódio fixo headless.
5. Comparação com a baseline nos mesmos índices e budgets.
6. Análise de traces por subagente.
7. Documentação da run e commit pequeno.
8. Só então avançar ao próximo milestone.

## Ordem de implementação

### M1 — Pós-condições e feedback uniforme

- **Tipo:** paper-confirmed; thresholds são paper-compatible.
- **Fonte:** §2.1, §2.3, Apêndices B e C.
- **Objetivo:** separar execução válida, pós-condição da primitiva e sucesso da tarefa.
- **Estado:** núcleo M1 implementado; grasp possui attachment/geometria, termination reason e guard de place.
- **Próximo incremento:** pós-condição de release/place e `move_to` por tolerância.
- **Aceitação:** nenhum `action_success` isolado vira sucesso semântico.

### M2 — Contrato `vla_act(prompt, max_chunks, τ)`

- **Tipo:** paper-confirmed; catálogo/threshold de `τ` é benchmark-specific.
- **Fonte:** §2.3, Apêndices B, C e E.5.
- **Objetivo:** budget por chunks, early return e razão de término.
- **Estado:** contrato backend-neutral implementado e testado com fake; adapter VLA real permanece no M10.
- **Teste inicial:** backend fake satisfaz `τ` ou atinge o cap sem inferência extra.
- **Aceitação:** resultado registra chunks pedidos/usados, `tau_satisfied` e cap;
  budget esgotado permanece uma razão de término compatível com continuação.
- **Persistência:** registros por chunk entram no trace incremental do M3.

### M3 — Trace incremental resistente a crash

- **Tipo:** registro por turn é paper-confirmed; atomicidade é paper-compatible.
- **Fonte:** §2.2, Apêndices A e E.
- **Objetivo:** persistir uma linha JSONL por decisão, inclusive rejeições.
- **Aceitação:** crash não perde turns concluídos; resumo reconstruível do trace.

### M4 — Task Specific Memory offline

- **Tipo:** paper-confirmed para memoria semantica/procedural parametrizada; schema e gates locais sao beta-only.
- **Fonte:** §2.2, Apêndices A e E.3.
- **Objetivo:** gerar audit JSON semântico e command JSONL procedural somente de rollout verificado.
- **Aceitação:** uma primitiva por linha, ordem preservada, sem coordenadas literais quando há binding simbólico.
- **Estado:** concluido no commit `ddefadb`; 117 testes passaram e 0/3 rollouts reais foram promovidos, como esperado pelas pos-condicoes nao verificadas.

### M5 — Retrieval e re-grounding de Task Specific Memory

- **Tipo:** paper-confirmed para prior estrutural e re-grounding atual; algoritmo de selecao e matching local sao paper-compatible/beta-only.
- **Fonte:** §2.2, §3.2, Apêndices A/C/E.3.
- **Objetivo:** usar a memória como prior estrutural e resolver posições na cena atual.
- **Aceitação:** posições trocadas mudam bindings, não copiam xyz da seed.
- **Estado:** resolver offline concluido no commit `4b7c986`; gate de tres cenas aprovado. Retrieval automatico, prompt e deployment continuam pendentes.

### M6 — Fases bootstrap e deployment

- **Tipo:** paper-confirmed.
- **Fonte:** §2.2 e Apêndice C.
- **Objetivo:** reset/budget amplo e escrita de memória somente no bootstrap; deployment estrito e read-only.
- **Aceitação:** seed de bootstrap nunca entra na métrica final.

### M7 — Global Memory auditável e incremental

- **Tipo:** paper-confirmed; política de promoção é paper-compatible.
- **Fonte:** §2.2, Apêndices A e E.4.
- **Objetivo:** candidatos com provenance, deduplicação e memória congelada em deployment.
- **Aceitação:** nenhuma regra sem trace de evidência; atualização idempotente.

### M8 — REPL mediado por arquivos

- **Tipo:** paper-confirmed.
- **Fonte:** Apêndices A e E.2.
- **Objetivo:** `command.json`, estados/logs indexados e sinal de conclusão entre planner e worker.
- **Aceitação:** exatamente uma execução por comando e retomada sem duplicação.

### M9 — Percepção isolada RGB-D/world map

- **Tipo:** paper-confirmed.
- **Fonte:** §2.1–2.2, §3.3, Apêndice E.2.
- **Objetivo:** remover coordenadas privilegiadas do planner e exigir pixels + world map.
- **Primeiro teste:** API pura pixels → mediana robusta de world coordinates.
- **Aceitação:** toda coordenada tem provenance de câmera/pixels.

### M10 — Backend VLA frozen real

- **Tipo:** paper-confirmed.
- **Fonte:** §2.3, §3.1, Apêndices B e D.
- **Objetivo:** substituir mock por checkpoint aprendido, visual e congelado.
- **Aceitação:** ações vêm do backend VLA; o mesmo checkpoint é usado no baseline direto e no harness.

### M11 — Vocabulário por embodiment

- **Tipo:** paper-confirmed.
- **Fonte:** §2.3 e Apêndice B.
- **Ordem:** `move_pose`; arm binding; somente com RoboCasa, `navigate_to`/`move_base`.
- **Aceitação:** disponibilidade corresponde ao benchmark e não muda em deployment.

### M12 — Protocolos experimentais completos

- **Tipo:** paper-confirmed.
- **Fonte:** §3 e Apêndices C/F.
- **Objetivo:** LIBERO, LIBERO-Pro, RoboCasa365 e RoboTwin C2R com seeds, baselines e métricas oficiais.
- **Aceitação:** manifests rejeitam vazamento de seed/memória e checkpoint divergente.

## Pontos de parada

- Após M3: arquitetura auditável e resistente a crash.
- Após M7: lifecycle e memória do paper, ainda sem VLA real.
- Após M10: candidato a reprodução funcional em um benchmark.
- Após M12: candidato a reprodução experimental comparável à v2.

Até M10, resultados do EB-Manipulation continuam sendo validação da beta, não
reprodução dos números científicos do paper.
