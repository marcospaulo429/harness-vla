# Harness beta: Task Specific Memory offline em 3 episodios

## Classificacao

**paper-compatible.** Os documentos locais atribuem `audit.json` semantico e
`commands.jsonl` procedural ao paper, mas a fonte primaria indicada como
`arXiv:2607.08448v2` nao estava verificavel. Schema, criterios de promocao,
hashes, escrita atomica e relatorio de rejeicao sao escolhas beta-only.

## Escopo

Esta etapa somente gera memoria offline. Ela nao recupera memoria, nao injeta
exemplos no planner e nao altera trajetorias. Portanto nao ha alegacao de melhora
de `task_success`.

Commit: `ddefadb`.

## Contrato implementado

Uma memoria aceita contem:

- `audit.json` com instrucao, contagens, criterios e hashes SHA-256;
- `commands.jsonl` com uma primitiva por linha e ordem preservada;
- bindings simbolicos `{label, roles}` em vez de IDs de episodio;
- apenas parametros nao espaciais, como modo, gripper e orientacao.

A saida rejeita qualquer campo `xyz`, voxel, pose, frame, mask, point cloud,
coordenada de objeto, acao compilada ou nome interno do simulador.

## Promocao conservadora

Um rollout somente e aceito quando:

1. `task_success=1`;
2. o trace esta completo e sua contagem concorda com `num_turns`;
3. nao ha parse, compile, semantic ou no-progress rejection;
4. existe ao menos uma primitiva de contato;
5. toda primitiva executada possui pos-condicao verificada;
6. todo alvo possui label e roles simbolicos;
7. o payload passa pela whitelist do schema.

Qualquer retry fisicamente nao verificado rejeita o rollout inteiro. A etapa nao
remove tentativas ruins para fabricar uma demonstracao ideal.

## Avaliacao fixa

Fonte: os mesmos episodios `[0, 15, 38]` da run
`harness_demo_3ep_20260727_222947`.

| Episodio | Task success | Promovido | Razoes |
|---:|---:|---:|---|
| 0 | 1 | nao | `unverified_primitive_postcondition` |
| 15 | 0 | nao | `task_not_successful`, `unverified_primitive_postcondition` |
| 38 | 0 | nao | `task_not_successful`, `unverified_primitive_postcondition` |

Resultado: **0/3 aceitos, 3/3 rejeitados corretamente**.

O episodio 0 e o teste mais importante: o benchmark marcou sucesso, mas houve
rasps nao verificados e um movimento fora da tolerancia. Ele nao virou seed.

O relatorio local foi salvo em:

`EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_20260727_222947/base/task_memory/promotion_report.json`

## Validacao

- 117 testes passaram;
- testes focados cobrem determinismo, uma primitiva por linha, trace truncado,
  mismatch de turns, statuses semanticos, schema e campos proibidos;
- revisao independente encontrou cinco gaps, todos corrigidos antes do commit;
- diagnosticos do editor e `git diff --check` passaram.

## Decisao

Infraestrutura M4 aprovada, mas nenhuma seed real esta disponivel. M5 pode
implementar retrieval e re-grounding com fixtures sinteticas, mas nao deve ser
ativado em avaliacao ate existir memoria de bootstrap fisicamente verificada.
