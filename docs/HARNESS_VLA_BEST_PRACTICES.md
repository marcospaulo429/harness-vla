# Boas práticas do projeto Harness VLA

Este documento define como desenvolver, avaliar e diagnosticar a beta sem
perder resultados, misturar causas ou sobrecarregar o contexto do orquestrador.

## 1. Organização do trabalho

1. Transforme a solicitação em entregáveis verificáveis.
2. Use subagentes de leitura para exploração, traces, paper e simulador.
3. Paralelize apenas tarefas independentes.
4. Mantenha uma única edição por arquivo de cada vez.
5. Integre resultados no contexto principal somente após conferir evidências.
6. Faça um commit pequeno por mudança lógica.

### Gate obrigatório de fidelidade científica

**Restrição vigente: só é permitido implementar mecanismos que o paper
(arXiv:2607.08448v2) descreve.** Extensões fora do paper estão proibidas até
que os componentes publicados estejam completos e as tarefas do benchmark
sejam resolvidas com eles. Instrumentação de auditoria e adaptação mínima ao
benchmark (`paper-compatible`) são aceitas; novas primitivas, heurísticas de
política ou skills fora do vocabulário publicado não são.

Antes de cada feature, consulte arXiv:2607.08448v2 e registre:

- seção/apêndice que descreve a ideia;
- contrato que precisa ser preservado;
- detalhes que o paper não especifica;
- classificação `paper-confirmed`, `paper-compatible` ou `beta-only`.

Heurísticas locais, thresholds e instrumentação são permitidos para tornar a
beta testável, mas devem ser descritos como escolhas locais. Não atribua ao
paper algoritmos ou valores que ele não publicou.

### Modelos dos subagentes

- orquestrador: GPT-5.6 Sol por padrão;
- tarefas mecânicas e coleta factual: modelo leve, como Claude Haiku 4.5;
- revisão e análise moderada: Haiku ou Sonnet;
- arquitetura, causa-raiz e implementação cruzada: GPT-5.6 Sol;
- decisão crítica/ambígua: GPT-5.6 Sol com revisão forte independente.

## 2. Hierarquia de evidência

Da mais forte para a mais fraca:

1. predicado `task_success` do benchmark;
2. estado físico mensurável, attachment e deslocamento do objeto;
3. reward e pós-condições específicas da primitiva;
4. `action_success` do ambiente;
5. interpretação visual de frames;
6. texto de reasoning do LLM.

`action_success=1` significa apenas que o simulador aceitou/executou o movimento.
Não significa que o objeto foi agarrado ou que a tarefa foi resolvida.

## 3. Ciclo de alteração

1. Fixar episódios, modelo, temperatura e budget da baseline.
2. Registrar a hipótese da mudança.
3. Rodar testes leves.
4. Rodar um episódio diagnóstico.
5. Inspecionar trace e frames.
6. Rodar o conjunto comparável somente se o diagnóstico for válido.
7. Comparar métricas e comportamento.
8. Registrar a run, inclusive se falhar ou crashar.

Nunca mude prompt, modelo, primitiva e percepção na mesma comparação: isso
impede atribuir a melhora a uma causa.

## 4. Identidade de uma run

Cada experimento deve registrar:

- ID com data/hora e nome curto;
- commit Git;
- configuração completa;
- modelo e endpoint;
- índices dos episódios;
- versão do prompt/memória;
- duração e estado: completa, interrompida ou crash;
- métricas agregadas;
- análise curta das falhas.

Artefatos pesados permanecem em `EmbodiedBench/running/`. Resumos pequenos e
comparáveis devem ser copiados para `docs/runs/` e versionados no Git.

## 5. Métricas mínimas

- episódios iniciados e concluídos;
- `task_success`;
- turns e env steps;
- parse errors;
- compile errors por motivo;
- primitivas por tipo e modo;
- `action_success` separado de sucesso da tarefa;
- grasp vazio/attachment quando instrumentado;
- tempo total e por episódio;
- crash ou encerramento incompleto.

## 6. Análise de traces

Para cada episódio:

1. ler a instrução;
2. reconstruir a sequência de primitivas/status;
3. verificar targets e coordenadas;
4. comparar pose antes/depois;
5. conferir ações compiladas;
6. verificar reward, feedback e predicado final;
7. identificar repetição sem progresso;
8. separar erro de planner, percepção, contato e infraestrutura.

Preserve sempre `raw_output`: ele explica por que o parser rejeitou a resposta.

## 7. Contato e grasp

Um grasp só deve ser considerado provável quando houver evidência pós-ação:

- objeto deslocou junto com o end-effector;
- objeto está anexado/segurado, quando a API fornecer esse estado;
- diferença relativa objeto–gripper permaneceu estável durante o lift;
- gripper não fechou completamente sem resistência, se disponível.

Após grasp vazio:

1. registrar `empty_grasp`;
2. re-localizar o alvo;
3. alterar staging, orientação ou offset;
4. só então repetir `vla_act`.

## 8. Grounding e identidade de objetos

- IDs (`object 1`) precisam permanecer estáveis entre frames.
- Nome semântico e ID técnico devem ser campos separados.
- Destino e objeto manipulável não podem compartilhar um target ambíguo.
- O prompt deve enumerar targets válidos.
- O compilador pode tolerar formatação, mas não deve adivinhar semântica perigosa.
- Após oclusão, reidentifique por mask ID, proximidade e histórico.

## 9. Simulador e renderização

- CWD: diretório `EmbodiedBench`.
- Carregar `.harness_env.sh`.
- Adicionar EB-Manipulation ao `PYTHONPATH`.
- Confirmar Ollama antes de iniciar.
- Não executar duas instâncias do CoppeliaSim em paralelo.
- Preferir `headless=True` e render `rgb_array`/offscreen.
- Salvar frames da câmera sem depender de janela interativa.
- Gerar GIF/vídeo depois do episódio.
- Após `SIGSEGV`, preservar imagens e traces parciais e encerrar processos órfãos.

A GUI `OPENGL3_WINDOWED` apresentou crash no renderer Qt/OpenGL durante a demo
de 21/07/2026; não deve ser usada como modo padrão de avaliação.

## 10. Testes e validação

Antes de uma run pesada:

1. diagnósticos dos arquivos alterados;
2. `git diff --check`;
3. testes unitários e integração sem simulador;
4. smoke test do modelo/API;
5. um episódio headless com captura de frames.

Depois:

1. verificar número de resultados esperado;
2. confirmar `summary.json` apenas em run completa;
3. validar JSON/JSONL;
4. procurar processos remanescentes;
5. gerar resumo histórico.

## 11. Commits

O commit deve informar o que mudou e por quê. Mudanças experimentais devem
incluir a métrica observada, sem afirmar causalidade não demonstrada.

Exemplos:

- `Add post-grasp displacement diagnostics`
- `Use offscreen rendering for visual demos`
- `Document incomplete 3-episode demo failure analysis`

## 12. Critério de melhoria

Uma mudança é melhoria somente quando:

- usa os mesmos episódios/budget da baseline;
- reduz a falha-alvo sem introduzir regressão maior;
- melhora evidência física ou `task_success`;
- é repetível em mais de uma execução.

Melhor JSON sem melhora física é progresso de interface, não de manipulação.
Movimento visualmente plausível sem predicado final é diagnóstico, não sucesso.