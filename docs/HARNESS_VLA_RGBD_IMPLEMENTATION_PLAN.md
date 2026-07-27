# Harness VLA: plano experimental sem backend VLA

Data de inicio: 2026-07-27

## Escopo e alegacoes

Este plano adia temporariamente o backend VLA aprendido devido a indisponibilidade
de CUDA. O `vla_act` continua sendo um mock scripted. Resultados obtidos antes
da integracao de um VLA frozen real sao validacao da beta, nao reproducao dos
resultados cientificos de arXiv:2607.08448v2.

Classificacao:

- **paper-confirmed**: mecanismo e papel descritos no paper;
- **paper-compatible**: detalhe necessario, mas nao especificado pelo paper;
- **beta-only**: instrumentacao ou limitacao local.

## Protocolo fixo

Cada etapa segue obrigatoriamente esta sequencia:

1. registrar hipotese e criterio de falsificacao;
2. implementar a menor mudanca testavel;
3. executar diagnosticos e testes leves focados;
4. executar a suite local relevante;
5. executar os mesmos episodios `[0, 15, 38]` com Gemma 4 12B, temperatura 0;
6. preservar config, traces, frames, GIFs, resumo e estado da run;
7. comparar com a baseline e registrar melhora, neutralidade ou regressao;
8. avancar apenas quando o contrato da etapa estiver validado e nao houver
   regressao inexplicada de interface ou observabilidade.

Melhora de `task_success` nao e gate obrigatorio para etapas de infraestrutura.
O gate exige que a metrica diretamente afetada melhore ou que uma capacidade
antes ausente seja medida corretamente.

Baseline fixa:

- commit: `c9aecd0`;
- run: `harness_demo_3ep_20260727_183547`;
- episodios: `[0, 15, 38]`;
- `task_success`: `1/3`;
- turns: `32`;
- env steps: `67`;
- parse errors: `0`;
- compile errors: `0`;
- grasp verificado: `3/10`;
- grasp nao verificado: `6/10`;
- empty grasp: `1/10`.

## Metricas comuns

Por episodio e agregadas:

- `task_success` e `action_success`, sempre separados;
- turns, env steps e duracao;
- parse errors, compile errors e rejeicoes semanticas;
- primitivas por tipo e modo;
- pos-condicao por primitiva e razao de termino;
- grasps verificados, nao verificados e vazios;
- objetos segurados, colocados e restantes;
- repeticoes sem progresso;
- identidade selecionada e mudancas de identidade;
- idade da estimativa usada por uma acao;
- erro RGB-D 3D medio, mediano, p95 e maximo;
- cobertura visual por camera e objetos sem estimativa;
- fonte, camera, frame, pixel, depth e calibracao de cada estimativa.

## Etapa 1 - RGB-D, world map e isolamento

**Classificacao:** paper-confirmed; formato do artefato e metricas sao
paper-compatible; mascaras perfeitas do simulador sao beta-only.

**Hipotese:** um snapshot RGB-D calibrado produz coordenadas mundo auditaveis e
IDs estaveis sem consultar poses oracle. O oracle e usado somente para metricas.

**Estado:** concluida em 2026-07-27. Gate aprovado com 91 testes, smoke de
projecao com erro maximo `1,39e-7 m` e run comparavel de tres episodios. O
comportamento permaneceu identico a baseline e foram adicionadas 285 observacoes
auditaveis. Ver `docs/runs/HARNESS_RGBD_3EP_20260727.md`.

Incrementos:

1. funcoes puras de depth metrico e projecao pixel-depth para mundo;
2. snapshot com `frame_id`, camera e calibracao;
3. artefato de grounding com estimativa, voxel e provenance;
4. metricas contra pose oracle isoladas do payload do planner;
5. overlays e sidecars auditaveis;
6. world map persistente entre frames e metricas de identity switch/staleness;
7. remocao gradual de coordenadas privilegiadas do input do planner.

Gates:

- projecao sintetica e round-trip passam;
- projecao concorda com a nuvem PyRep dentro de tolerancia numerica;
- retorno legado permanece compativel;
- nenhuma pose/bbox oracle entra em `planner_coords` ou prompt;
- todo ponto possui provenance;
- tres episodios completos geram metricas e overlays;
- qualquer regressao em parse/compile e investigada antes de avancar.

Nota: a primeira versao usa `sim_mask` para associar pixels a objetos. Isso
valida geometria e instrumentacao, mas ainda nao e percepcao visual independente.

## Etapa 2 - Pos-condicoes e estado fisico uniforme

**Classificacao:** paper-compatible enquanto a fonte primaria indicada como
`arXiv:2607.08448v2` nao puder ser resolvida externamente; os documentos locais
atribuem o mecanismo ao paper, mas isso nao substitui verificacao primaria.
Tolerancias numericas e transformacao de sinais PyRep em estado estruturado sao
beta-only.

**Hipotese:** `action_success` esconde divergencias fisicas que podem ser
detectadas por pose final, attachment e grounding RGB-D, sem mudar a trajetoria
nem consultar oracle.

**Criterio de falsificacao:** a etapa falha se os traces nao distinguirem
movimento aceito de pose nao alcancada, ou se grasp/release/place forem marcados
como verificados sem a evidencia exigida.

**Thresholds beta:** `move_to_tolerance=2.0` voxels e
`place_tolerance=12.0` voxels. Estes valores instrumentam a beta e nao sao
atribuidos ao paper.

**Estado:** concluida em 2026-07-27 no commit `92f5555`. A run fixa de tres
episodios manteve `task_success=1/3` e revelou 7 falhas de pos-condicao entre 32
primitivas, apesar de `action_success=1` nos 60 env steps. Ver
`docs/runs/HARNESS_POSTCONDITIONS_3EP_20260727.md`.

Entregas:

- `move_to` validado por tolerancia de pose;
- `release/place` validado por attachment e estado final;
- estado estruturado `held`, `placed`, `remaining`;
- categorias explicitas para falha de grasp;
- `action_success`, pos-condicao e `task_success` nunca colapsados.

Gate: os traces explicam todas as divergencias entre movimento aceito e resultado
fisico nos tres episodios.

## Etapa 3 - Trace incremental resistente a crash

**Classificacao:** paper-compatible enquanto a fonte primaria indicada como
`arXiv:2607.08448v2` nao estiver verificavel; `fsync`, tolerancia a linha
truncada, resumo reconstruido e manifesto atomico sao beta-only.

**Estado:** concluida em 2026-07-27 nos commits `e893aa2` e `f9ff6fc`. A run fixa
de tres episodios preservou 32/32 turns em JSONL, e os resumos reconstruidos
concordaram com turns, env steps, `task_success` e pos-condicoes dos resultados
oficiais. Ver `docs/runs/HARNESS_INCREMENTAL_TRACE_3EP_20260727.md`.

Entregas:

- append e flush por decisao, inclusive rejeicoes;
- resumo reconstruivel somente do JSONL;
- teste de interrupcao apos N turns;
- manifesto de run com commit, config, episodios e estado.

Gate: uma interrupcao simulada preserva todos os turns concluidos, sem linha
parcial valida.

## Etapa 4 - Task Specific Memory offline

**Classificacao:** paper-confirmed para memoria semantica/procedural e
parametrizacao simbolica, conforme Secao 2.2 e Apendices A/E.3 de
`arXiv:2607.08448v2`, verificado apos a implementacao. O schema, criterios
conservadores, hashes e escrita atomica sao beta-only.

**Estado:** concluida em 2026-07-27 no commit `ddefadb`. O gerador offline
produz memoria simbolica deterministica apenas de rollout integralmente
verificado. Nos mesmos tres episodios da Etapa 3, rejeitou corretamente 3/3
candidatos; inclusive o episodio com `task_success=1` continha pos-condicoes
nao verificadas. Ver `docs/runs/HARNESS_TASK_MEMORY_OFFLINE_3EP_20260727.md`.

Entregas:

- `audit.json` semantico;
- `commands.jsonl` procedural, uma primitiva por linha;
- geracao apenas de rollout com sucesso fisico verificado;
- bindings simbolicos sem coordenadas literais.

Gate: memoria de seed e reproduzivel, rejeita rollout suspeito e passa testes de
schema e ordenacao. Gate de infraestrutura aprovado; nenhuma seed real foi
promovida e a memoria ainda nao e consumida pelo planner.

## Etapa 5 - Retrieval e re-grounding

**Classificacao:** paper-confirmed para recuperar a trace como prior estrutural e
re-groundear toda geometria da observacao atual (Secao 2.2 e Apendice E.3).
Selecao explicita do pacote, normalizacao de labels e rejeicao de ambiguidade
sao paper-compatible/beta-only porque o algoritmo de busca nao e especificado.

**Estado:** resolver offline concluido em 2026-07-27 no commit `4b7c986`.
Carregamento, hash, binding unico e re-grounding passaram 124 testes e um gate
de tres cenas: duas posicoes foram resolvidas com coordenadas atuais diferentes
e uma cena sem objeto foi rejeitada. Ver
`docs/runs/HARNESS_TASK_MEMORY_REGROUNDING_3SCENE_20260727.md`. Retrieval
automatico e injecao no planner permanecem pendentes por nao haver seed real.

Entregas:

- retrieval por tarefa;
- binding de roles e labels na cena atual;
- re-grounding via world map;
- testes de position swap e objeto ausente.

Gate offline aprovado: trocar posicoes altera coordenadas resolvidas sem alterar
a estrutura da memoria nem copiar xyz da seed. O gate de deployment permanece
pendente.

## Etapa 6 - Bootstrap e deployment

**Classificacao:** paper-confirmed; budgets locais sao paper-compatible.

Entregas:

- bootstrap com exploracao e escrita;
- deployment estrito e read-only;
- separacao de seeds e metricas;
- manifests que rejeitam vazamento.

Gate: episodios de bootstrap nunca entram na metrica de deployment e nenhuma
memoria muda durante deployment.

## Etapa 7 - Global Memory incremental

**Classificacao:** paper-confirmed; promocao e deduplicacao sao
paper-compatible.

Entregas:

- candidatos derivados de evidencias;
- provenance ate trace/turn;
- deduplicacao e atualizacao idempotente;
- congelamento em deployment;
- preservacao de evidencia negativa.

Gate: nenhuma regra existe sem evidencia rastreavel e reprocessar os mesmos
traces nao altera a memoria.

## Etapa 8 - REPL mediado por arquivos

**Classificacao:** paper-confirmed; timeout e retry sao paper-compatible.

Entregas:

- `command.json`, `state_NN.json`, `log_NN.json`, `done_NN.flag`;
- exatamente uma execucao por comando;
- retomada idempotente apos crash.

Gate: worker fake e worker do simulador passam testes de duplicacao,
interrupcao e retomada.

## Regra de parada

Interromper uma etapa e registrar bloqueio quando:

- o teste discriminante falsificar a hipotese;
- houver regressao nao explicada nos contratos existentes;
- a avaliacao depender de GPU ou VLA real;
- o simulador apresentar crash repetivel sem artefato diagnostico suficiente.

Nao ajustar offsets, forcas ou trajetorias do mock para aumentar artificialmente
`task_success`. Mudancas comportamentais devem permanecer reutilizaveis quando o
backend VLA real for conectado.
