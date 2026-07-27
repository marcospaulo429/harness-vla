# Harness beta: retrieval explicito e re-grounding em 3 cenas

## Classificacao e fonte

**paper-confirmed:** Task Specific Memory e recuperada como prior estrutural e
seus argumentos espaciais sao re-grounded da observacao atual.

A fonte primaria `arXiv:2607.08448v2` foi verificada diretamente:

- Secao 2.2: o rollout bem-sucedido e parametrizado, substituindo coordenadas
  concretas por consultas simbolicas; no deployment o planner recupera a trace e
  a fundamenta com RGB-D atual.
- Apendice A: a trace e um esqueleto da solucao, nao uma trajetoria open-loop.
- Apendice E.3: nunca reutilizar `xyz`, pixels, poses ou coordenadas da seed;
  relocalizar entidades a partir de imagens e world maps atuais.

**paper-compatible:** o paper nao especifica algoritmo de busca entre memorias.
A beta exige que o chamador selecione explicitamente o pacote da tarefa.
Normalizacao textual, match exato por label, conjunto de roles e rejeicao de
ambiguidade sao escolhas beta-only.

## Mudanca

Commit: `4b7c986`.

O modulo de Task Specific Memory agora:

- carrega e valida `audit.json` e `commands.jsonl`;
- verifica schema, ordem, newline final e hash SHA-256 dos comandos;
- resolve `{label, roles}` contra `object_labels`, `object_roles` e
  `object_coords` da observacao atual;
- exige correspondencia unica;
- produz `xyz` atual somente para `move_to`;
- mantem IDs atuais em `vla_act`, sem inventar coordenadas;
- rejeita qualquer coordenada armazenada na seed.

O resolver e puro: nao altera a memoria original e ainda nao e injetado no
planner Gemma.

## Gate de tres cenas

Nao existe seed real promovida na Etapa 4. Por isso, executar tres episodios
fisicos com memoria seria impossivel sem fabricar uma demonstracao. O gate desta
etapa usou uma memoria sintetica identica em tres cenas controladas:

| Cena | Grounding atual | Resultado |
|---|---|---|
| `position_a` | `red-a -> [1,2,3]` | resolvido para `[1,2,3]` |
| `position_swap` | `red-b -> [21,22,23]` | resolvido para `[21,22,23]` |
| `object_absent` | nenhum `red cube` manipulavel | rejeitado antes da execucao |

A troca de posicao alterou o `xyz` resolvido sem alterar o comando simbolico. A
cena sem objeto falhou de forma explicita, sem coordenada fallback.

## Validacao

- 124 testes passaram;
- testes focados cobrem swap, ausencia, ambiguidade, role mismatch, adulteracao
  de hash, ordem e rejeicao de coordenadas da seed;
- o payload resolvido foi conferido contra o compilador de `move_to` e
  `vla_act`;
- `git diff --check` passou.

## Limites

Retrieval automatico, selecao entre tarefas, contexto no prompt e execucao
memory-backed continuam ausentes. Nao houve mudanca de `task_success`; ativacao
experimental depende de uma seed fisicamente verificada e de isolamento formal
entre bootstrap e deployment.
