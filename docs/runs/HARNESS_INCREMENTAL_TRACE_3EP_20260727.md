# Harness beta: trace incremental em 3 episodios

## Classificacao

- **paper-compatible:** um registro por decisao e resumo reconstruivel. A fonte
  primaria indicada como `arXiv:2607.08448v2` nao estava verificavel nesta etapa.
- **beta-only:** `fsync` por turn, tolerancia a uma ultima linha truncada,
  `trace_summary_episode_N.json` e `run_manifest.json` atomico.

## Identidade

- run: `harness_demo_3ep_20260727_222947`;
- commit: `f9ff6fcf47789428a5c8a224517f4ab2d8ead174`;
- modelo: Gemma 4 12B, Ollama CPU, temperatura 0, `think=false`;
- episodios: `[0, 15, 38]`;
- duracao: 20 min 00 s;
- runtime: CoppeliaSim 4.1, PyRep, Xvfb e Mesa llvmpipe;
- manifesto: `completed`, 3 episodios concluidos.

## Gate de persistencia

Todos os 32 turns foram persistidos incrementalmente:

| Episodio | Linhas | Newline final | Summary reconstruido | Episode result |
|---:|---:|---:|---:|---:|
| 0 | 8 | sim | valido | concorda |
| 15 | 12 | sim | valido | concorda |
| 38 | 12 | sim | valido | concorda |

Para cada episodio, a reconstrucao usando somente o JSONL concordou com o
resultado oficial em:

- numero de turns;
- numero de env steps;
- `task_success`;
- pos-condicoes cumpridas e falhas;
- razoes de termino.

Testes sem simulador tambem confirmaram:

- registros completos sobrevivem a interrupcao sem `close`;
- somente uma ultima linha sem newline e JSON truncado e ignorada;
- corrupcao em linha completa levanta erro;
- inicializacao remove trace obsoleto com a mesma identidade;
- manifesto e substituido atomicamente;
- commit e resolvido mesmo quando o container nao possui o executavel Git.

## Resultado comportamental

| Metrica | Pos-condicoes (`214125`) | Trace incremental (`222947`) |
|---|---:|---:|
| `task_success` | 1/3 | 1/3 |
| Turns | 32 | 32 |
| Env steps | 60 | 57 |
| Pos-condicoes cumpridas | 25 | 21 |
| Pos-condicoes falhas | 7 | 11 |
| Parse / compile / semantic rejects | 0 / 0 / 0 | 0 / 0 / 0 |

A persistencia nao altera trajetorias ou prompts. A diferenca de sequencia vem
da variacao do grounding e do feedback entre rollouts, nao do `fsync`.

## Finding separado: destino fora do workspace

No episodio 15, turn 9, o planner chamou place de `object 7` em `object 8`.
O grounding observado para o destino foi `[-14, 88, 20]`, fora do intervalo
voxel valido. O place desapegou o objeto, mas terminou a 56,94 voxels da
estimativa do destino e foi classificado como
`released_outside_destination_tolerance`.

Nos turns 10-12, o planner tentou `move_to` com `xyz=[-14, 88, 20]`. A biblioteca
limitou a acao compilada para `[0, 88, 20]`, e o ambiente rejeitou cada tentativa:

- reward `-1`;
- `action_success=0`;
- feedback `Target is outside of workspace`;
- `execution_status=failed`;
- `primitive_postcondition_met=false`;
- `termination_reason=target_pose_not_reached`.

Portanto a falha foi capturada corretamente. Sua categoria e
percepcao/coordenadas instaveis seguida de planejamento repetitivo, nao falha de
persistencia.

## Artefatos

Diretorio:

`EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_20260727_222947/base/`

Inclui manifesto, config, tres traces JSONL, tres resumos reconstruidos, resultados,
sidecars RGB-D, overlays, frames e GIFs.

## Decisao

Etapa 3 aprovada. O trace e auditavel e resistente a perda dos turns concluidos.
A proxima etapa e geracao offline de Task Specific Memory. Ela nao sera injetada
no planner nem promovida a partir das runs atuais: o unico episodio com
`task_success=1` possui tres grasps nao verificados.
