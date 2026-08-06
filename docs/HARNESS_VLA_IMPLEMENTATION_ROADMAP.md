# Roadmap de implementação e avaliação do Harness VLA v2

Atualizado em 2026-08-06. Este é o único documento de planejamento futuro do
repositório. O estado consolidado está em `HARNESS_VLA_BETA_REPORT.md` e os
resultados históricos em `docs/runs/`.

## 1. Regras de escopo

Antes de implementar um mecanismo, o `pesquisador-paper` deve confirmar a seção,
o contrato e o que permanece específico ao benchmark em arXiv:2607.08448v2.

- **paper-confirmed:** mecanismo e papel publicados;
- **paper-compatible:** adaptação necessária, não especificada pelo paper;
- **beta-only:** instrumentação ou extensão local, sem alegação de fidelidade.

Componentes P0 publicados têm precedência sobre extensões beta-only. Cada
mudança deve ter teste leve, smoke fixo, comparação pareada quando aplicável,
trace analisável e commit próprio.

## 2. Estado dos componentes

| Componente | Classe | Estado | Próximo gate |
|---|---|---|---|
| Planner, JSON e loop fechado | paper-confirmed | implementado | regressão Etapa E |
| Biblioteca fixa e guards | paper-confirmed | implementado | vocabulário LIBERO |
| Pós-condições físicas | paper-confirmed | parcial avançado | cobertura uniforme |
| Trace incremental | paper-confirmed/compatible | implementado | manifesto de eval |
| Runtime `vla_act` por chunks | paper-confirmed | implementado | ampliar predicados `tau` |
| Backend pi0.5/RLinf | paper-confirmed | smoke nativo validado | Harness LIBERO completo |
| RGB-D/world map | paper-confirmed | parcial | remover `sim_mask`/oracle |
| Task Specific Memory | paper-confirmed | runtime fail-closed | lifecycle LIBERO |
| Bootstrap/deployment | paper-confirmed | guards no evaluator | run LIBERO oficial |
| Global Memory incremental | paper-confirmed | candidatos auditados | promoção causal |
| REPL mediado por arquivos | paper-confirmed | ausente | worker fake idempotente |
| Protocolos LIBERO/Pro | paper-confirmed | baseline 20 rollouts + smokes | Harness pareado |
| EB-Navigation | beta-only | 3 episódios, 2/3 | manter separado do P0 |

## 3. Fase I: fechar o lifecycle do método

### P0.1 — Task Specific Memory no runtime

**Fonte:** §2.2, Apêndices A e E.3.
**Contrato:** uma seed de referência produz memória semântica e procedural;
novas seeds reutilizam a estrutura, mas re-groundeiam toda geometria.

Entregas:

1. retrieval por identidade/estrutura de tarefa;
2. seleção explícita e hash no manifesto;
3. injeção da memória resolvida no prompt;
4. progresso auditável sobre os passos recuperados;
5. fallback zero-shot quando não houver binding seguro.

Aceitação:

- seed verificada gera pacote determinístico;
- position swap altera bindings, não a estrutura;
- coordenadas literais da seed nunca entram no deployment;
- ausência/ambiguidade rejeita a memória sem ação física.

### P0.2 — Bootstrap e deployment reais

**Fonte:** §2.2, Apêndice C.
**Contrato:** exploração/reset/escrita somente no bootstrap; avaliação held-out
com memória congelada e sem incluir a seed de referência na métrica.

Entregas:

1. manifest de fase obrigatório;
2. partição de seeds e budgets por fase;
3. guards de reset e escrita no evaluator;
4. hashes de memória antes/depois da run;
5. denominador de deployment independente do bootstrap.

Aceitação:

- escrita ou reset proibido falha antes de executar ação;
- seed bootstrap não aparece no resumo de deployment;
- hashes permanecem constantes durante deployment.

### P0.3 — Global Memory incremental

**Fonte:** §2.2, Apêndices A e E.4.
**Contrato:** regras de sucesso, modelos de falha e evidência negativa são
extraídos das interações, refinados no bootstrap e congelados no deployment.

Entregas:

1. interpretação estruturada dos candidatos do ledger;
2. promoção/rejeição com provenance;
3. deduplicação e substituição auditável;
4. preservação de negative evidence;
5. renderização somente de regras promovidas.

Aceitação:

- nenhuma regra sem trace/turn/hash;
- reprocessar o mesmo trace é idempotente;
- deployment não muda o arquivo nem o hash da memória.

### P0.4 — Percepção isolada

**Fonte:** §2.1–2.2, §3.3, Apêndice E.2.
**Contrato:** o planner localiza visualmente e consulta RGB-D/world map; estado
privilegiado pode ser usado somente para métricas.

Entregas:

1. payload multimodal do planner;
2. seleção visual de objeto/pixel;
3. consulta robusta ao world map;
4. remoção de `sim_mask`, poses e bboxes oracle do caminho de decisão;
5. re-localização após movimentos e oclusões.

Aceitação:

- toda coordenada registra câmera/frame/pixel/depth/calibração;
- auditor detecta e rejeita campo privilegiado no payload;
- position swap e oclusão mantêm identidade ou falham de forma explícita.

### P0.5 — Contrato planner-facing de `vla_act`

**Fonte:** §2.3, Apêndices B e E.5.
**Contrato:** `vla_act(prompt, max_chunks, tau)` executa observações vivas até o
predicado ou cap, sem inferência adicional após early return.

Entregas:

1. schema de prompt, cap e predicado permitido por interação;
2. validação semântica no planner;
3. trace com cap solicitado/usado e definição de `tau`;
4. métricas de chunks e razão de término.

Aceitação:

- fake e backend real param exatamente no primeiro chunk elegível;
- budget esgotado é continuação possível, não falso sucesso;
- trace reconstrói integralmente a chamada.

Estado em 2026-08-06: schema planner-facing, cap, trace e early return foram
validados com pi0.5/RLinf real e Gemma thinking em task 0/state 0. O smoke usou
somente `tau=task_success`; predicados de grasp/contato ainda faltam. O
compilador OSC e o executor fechado de pose passaram em testes, e a projeção
RGB-D foi validada em um smoke isolado com segmentação privilegiada. Esses
componentes ainda não estão ligados ao planner-facing evaluator.

## 4. Fase II: reprodução funcional no LIBERO

### P0.6 — Harness LIBERO nativo

**Fonte:** §2.3, §3.1–3.2, Apêndices B, C e D.
**Contrato:** baseline direta e Harness usam o mesmo pi0.5/RLinf frozen, com
controle analítico fora do contato e `vla_act` no contato.

Entregas:

1. evaluator Harness no embodiment nativo;
2. vocabulário LIBERO, incluindo `move_pose`;
3. observações e ações nativas, sem conversão EB;
4. checkpoint SHA e configuração no manifesto;
5. baseline e Harness pareados.

Aceitação incremental:

1. uma tarefa: bootstrap `s0`, deployment `s1`;
2. 10 tarefas × 10 seeds em LIBERO-Spatial;
3. quatro suites LIBERO;
4. nenhuma divergência de checkpoint ou seed entre braços da comparação.

### P1.1 — REPL mediado por arquivos

**Fonte:** Apêndices A e E.2.
**Contrato:** planner e worker trocam comando, estado, log e flag de conclusão;
o worker é o único proprietário do simulador.

Aceitação:

- cada comando é consumido exatamente uma vez;
- estados são monotônicos;
- crash/restart não duplica ação nem perde turno concluído.

## 5. Fase III: protocolo experimental

### 5.1 Escada de avaliação

1. **Testes/fakes:** regressão de contratos sem simulador.
2. **Etapa E:** `[0,15,38]`, somente como regressão beta-only.
3. **VLA direto LIBERO-Spatial:** ampliar o smoke 9/10 para 10 seeds por tarefa.
4. **Harness zero-shot:** mesmo checkpoint, sem Task/Global Memory.
5. **Harness + Task Memory:** bootstrap `s0`, deployment held-out.
6. **Harness + Global Memory:** mesma partição e budget.
7. **Harness completo:** ambas as memórias congeladas no deployment.
8. **Quatro suites LIBERO:** protocolo pareado completo.
9. **LIBERO-Pro:** células oficiais de perturbação.
10. **EB-Navigation:** generalização beta-only, separada das alegações do paper.
11. **RoboCasa365 e RoboTwin C2R:** cobertura final.

### 5.2 Matriz mínima de ablações

| Braço | Task Memory | Global Memory | Finalidade |
|---|---:|---:|---|
| VLA direto | não | não | baseline frozen |
| Harness zero-shot | não | não | efeito do planner/primitivas |
| Harness task-only | sim | não | efeito da memória específica |
| Harness global-only | não | sim | efeito das regras globais |
| Harness completo | sim | sim | método completo |

Depois dessa matriz: cap de `vla_act` e planners publicados, sempre alterando um
único fator por vez. Ablations locais de threshold ou feedback devem ser marcadas
**paper-compatible**, não atribuídas ao paper.

### 5.3 Métricas obrigatórias

- `task_success` por tarefa, suite, seed e perturbação;
- intervalos de confiança e número de rollouts completos;
- chunks e chamadas de `vla_act`;
- primitivas analíticas por tipo;
- parse, compile e rejeição semântica;
- pós-condições físicas;
- falhas de grounding, planejamento, contato e infraestrutura;
- tempo total, por turno e por inferência;
- hashes de checkpoint, prompt e memórias.

## 6. Gates operacionais

Antes de qualquer avaliação pesada:

1. `git status` e commit identificados;
2. testes leves e `git diff --check`;
3. GPU escolhida por memória livre e utilização;
4. Ollama/modelo e servidor VLA verificados;
5. espaço em disco confirmado;
6. uma única instância do simulador;
7. run nomeada em `evaluation_runs/<test_id>/`.

Após a avaliação:

1. validar contagem de episódios e JSON/JSONL;
2. marcar `complete`, `interrupted` ou `crashed`;
3. preservar artefatos parciais;
4. analisar traces antes de outra mudança;
5. versionar resumo pequeno em `docs/runs/` quando comparável.

## 7. Fora da prioridade atual

- otimizar o mock de contato para aumentar score no EB-Manipulation;
- adaptar indefinidamente o pi0.5/RLinf ao embodiment do CoppeliaSim;
- adicionar skills não publicadas antes dos P0;
- iniciar RoboCasa/mobile ou RoboTwin/bimanual antes do LIBERO completo.
