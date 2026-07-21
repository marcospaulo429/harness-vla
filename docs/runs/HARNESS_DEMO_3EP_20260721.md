# Análise da demo visual de 3 episódios — 2026-07-21

## Estado da run

- ID: `harness_demo_3ep_20260721_102341`
- modelo: `qwen2.5:0.5b-instruct`
- episódios selecionados: `[0, 15, 38]`
- modo: GUI OpenGL + captura de `front_rgb`
- resultado: **run incompleta** por `SIGSEGV` no renderer OpenGL do CoppeliaSim
- episódio 1: concluído
- episódio 2: concluído sem executar ações físicas
- episódio 3: frames parciais, sem trace/result final devido ao crash

Os artefatos brutos permanecem em `EmbodiedBench/running/` e não são
versionados no Git.

## Episódio 1

Instrução: pegar a estrela e colocá-la no recipiente prateado.

### Sequência

- turns 1–4: três `parse_error` e um `compile_error`;
- turn 5: `vla_act(grasp, target="object 1")`;
- turns 6–12: sete chamadas repetidas de
  `vla_act(place, target="object 1")`;
- 25 subações aceitas pelo simulador;
- reward zero em todas;
- `task_success=0`.

### Por que parece pegar/soltar no vazio

O planner usa `object 1` tanto no grasp quanto nas tentativas de place. Ele
nunca seleciona explicitamente o **recipiente prateado** como destino. Portanto,
a primitiva de place desce e abre a garra perto do objeto ou de sua posição
reestimada, não no recipiente correto.

Além disso, `action_success=1` só confirma que a trajetória foi executada. O mock
`vla_act` não verifica se o objeto realmente ficou preso à garra. Depois da
primeira tentativa, as coordenadas de `object 1` variam:

`[73,15,18] → [73,17,24] → [73,19,24] → [73,20,18] → ...`

Isso é compatível com objeto deslocado/solto, oclusão ou identidade instável. O
trace atual não contém attachment ou teste de deslocamento suficiente para
determinar qual dessas causas ocorreu.

## Episódio 2

Instrução: empilhar cilindros maroon e navy.

- 12 turns;
- 9 `parse_error`;
- 3 `compile_error`;
- zero ações físicas;
- `task_success=0`.

O modelo produziu JSON com alternativas usando `|`, estruturas sem campo
`action`, modo inexistente `stack` e finalmente targets descritivos como:

`object 1: [63, 32, 17] (maroon cylinder)`

A biblioteca aceita somente IDs presentes na tabela, como `object 1`.

## Episódio 3

O terceiro episódio produziu frames parciais, mas o processo caiu antes de
gravar `episode_3_res.json` e `trace_episode_3.jsonl`. Não é possível fazer uma
análise causal completa do planner desse episódio.

## Crash de renderização

O stack trace termina em:

- `QOpenGLContext::defaultFramebufferObject`;
- `libsimExtOpenGL3Renderer.so`;
- renderização de depth/light do vision sensor.

Isso classifica o encerramento como falha de infraestrutura/renderização, não
como falha física do robô. Próximas demos devem usar execução headless e captura
offscreen, gerando GIFs após o encerramento.

## Métricas dos episódios com trace

| Métrica | Valor |
|---|---:|
| turns | 24 |
| parse errors | 12 (50%) |
| compile errors | 4 (16,7%) |
| turns executados | 8 (33,3%) |
| subações executadas | 25 |
| task success | 0/2 |

## Causas priorizadas

1. **Sem verificação de grasp:** movimento válido é tratado como sucesso local.
2. **Destino não grounded:** recipiente não recebe ID/semântica utilizável.
3. **Planner tiny:** JSON inválido e repetição sem progresso.
4. **Identidade/coordenadas possivelmente instáveis após oclusão/contato.**
5. **GUI OpenGL instável:** crash interrompeu a terceira análise.

## Próximo experimento mínimo

Antes de repetir três episódios:

1. registrar posição do objeto antes/depois do grasp e distância ao gripper;
2. emitir feedback explícito `empty_grasp`/`grasp_unverified`;
3. separar semanticamente objeto e destino na percepção/prompt;
4. executar apenas um pick-and-place fixo;
5. usar `headless=True`, render offscreen e frames front + overhead;
6. só ampliar para três episódios após obter trace completo sem crash.