# Harness VLA com OpenVLA em CPU - avaliacao de 3 episodios

## Classificacao cientifica

- **paper-confirmed:** o planner emite uma primitiva por turno, recebe nova observacao apos a execucao e o sucesso da tarefa e medido pelo predicado do ambiente.
- **paper-compatible:** adaptacao HTTP, conversao da acao continua de 7 dimensoes para o contrato do EB-Manipulation e limites de seguranca dos deltas.
- **beta-only:** uso do checkpoint `openvla/openvla-7b-finetuned-libero-object` fora do dominio LIBERO, thresholds locais, instrumentacao RGB-D e esta avaliacao em CPU.

Esta run valida a integracao de um VLA real. Ela nao reproduz o Harness VLA do paper e nao mede a capacidade do OpenVLA no dominio em que foi treinado.

## Identidade da run

- run: `openvla_cpu_3ep_20260728_134022`
- estado: completa
- commit: `b8c7091ff840cec8a8ba7a2034204d0e8ca11a8b`
- planner: `gemma4:12b` via Ollama, CPU, temperatura 0, thinking desativado
- VLA: `openvla/openvla-7b-finetuned-libero-object`, CPU, `unnorm_key=libero_object`
- simulador: CoppeliaSim 4.1 + PyRep, headless, captura RGB 256x256
- eval set: `base`
- indices: `[0, 15, 38]`
- budgets: 12 turns, 30 env steps e ate 8 chunks OpenVLA por chamada
- limites do adapter: translacao `0.05`, rotacao `0.5`
- duracao observada: 1.908 s (31 min 48 s)

Artefatos locais:

`openvla_cpu_eval/gemma4_12b/openvla_cpu_3ep_20260728_134022/base/`

## Resultado

| Indice | Tarefa | `task_success` | Turns | Env steps | `action_success` |
|---:|---|---:|---:|---:|---:|
| 0 | Pick up the star and place it into the silver container. | 0 | 12 | 30 | 30/30 |
| 15 | Stack the maroon cylinder and the navy cylinder in sequence. | 0 | 12 | 30 | 22/30 |
| 38 | Wipe the horizontal area. | 0 | 9 | 30 | 27/30 |
| **Total** | | **0/3** | **33** | **90** | **79/90 (87,8%)** |

`action_success` indica apenas que uma acao foi aceita pelo simulador. O predicado oficial `task_success` permaneceu falso nos tres episodios.

Nao houve erro de parse, compile error, erro do backend, semantic reject ou crash. Foram 14 pos-condicoes cumpridas e 19 falhas: 12 `target_pose_not_reached` e 7 `unverified`.

## Metricas do VLA

| Metrica | Resultado |
|---|---:|
| Chamadas `vla_act` | 10 |
| Chunks/inferencias | 67 |
| Latencia media | 8,43 s |
| Latencia p95 | 8,65 s |
| `grasp_verified` | 3/7 |
| Place verificado | 0/2 |
| `budget_exhausted` | 7/10 chamadas |
| `target_attached` | 1/10 chamadas |
| `environment_done` | 2/10 chamadas |
| Delta translacional saturado | 61/67 chunks (91,0%) |
| Delta rotacional saturado | 0/67 chunks |

Os valores brutos de garra foram discretos: 33 chunks emitiram `0.0` e 34 emitiram aproximadamente `0.996`. Portanto, os traces nao sustentam a hipotese de comando de garra sempre neutro. A saturacao translacional frequente e compativel com incompatibilidade de escala entre LIBERO e EB-Manipulation, mas nao prova sozinha a causa das falhas fisicas.

O grounding teve 393 observacoes, erro medio de 0,0199 m e maximo de 0,0874 m. Por episodio, a media foi 0,0162 m, 0,0116 m e 0,0387 m. O terceiro episodio teve os maiores outliers, mas nao ha evidencia de troca semantica do alvo.

## Diagnostico por episodio

### Episodio 0: grasp sem transporte preservado

O turn 2 anexou fisicamente a estrela. No turn 3, o planner chamou `move_to` com `gripper="open"`; esse valor realmente abre a garra no EB-Manipulation e o attachment desapareceu. Os turns 3-6 e 8-11 terminaram entre 15,75 e 22,52 voxels do alvo. O place do turn 7 ocorreu sem attachment e nao foi verificado.

Causa primaria: **planejamento/sequencia inadequada**. Contribuintes: poses baixas ou orientacao herdada levando a estagnacao de IK/colisao, e `held_object_id` local obsoleto apos o detach. Abrir a garra explica a perda do objeto, mas nao explica sozinho `target_pose_not_reached`, pois outros `move_to` com garra aberta chegaram ao alvo no episodio 15.

### Episodio 15: release antes do stacking

O primeiro grasp foi chamado longe do objeto e seus oito chunks falharam com `Could not create path`. Depois de staging, o turn 3 anexou o primeiro cilindro em um unico chunk. O turn 4 abriu a garra durante o transporte; o planner ainda executou um `release` generico no turn 6, sem placement verificado. Os grasps do segundo cilindro nos turns 10 e 12 falharam.

Causa primaria: **planejamento/sequencia inadequada**. Contribuintes: execucao fisica/IK no primeiro grasp e contato insuficiente nos grasps finais. O simulador aceitou 22 acoes, mas nenhuma sequencia de stacking satisfez o predicado oficial.

### Episodio 38: wiping tratado como place

O turn 4 anexou a esponja. O turn 5 abriu a garra e perdeu o attachment. Em seguida, o planner tentou `vla_act mode="place"` com destino `object 5`, em vez de manter contato e executar uma trajetoria de limpeza. A chamada terminou longe do destino, com tres `Could not create path` finais.

Causa primaria: **planejamento/sequencia inadequada**. Contribuintes: execucao fisica/contato e erro de grounding maior neste episodio. Attachment da esponja nao equivale a limpar a regiao.

## Comparacao com a baseline

A baseline `harness_demo_3ep_20260727_183547` usou os mesmos indices e budgets e obteve `task_success=1/3`. Esta run obteve `0/3`; portanto, **nao houve melhora** e a diferenca observada foi de -33,33 pontos percentuais.

A comparacao e util para o pipeline, mas nao isola uma politica: a baseline usava `vla_act` scripted, enquanto esta run usa OpenVLA real, outro commit e outra dinamica. O unico sucesso da baseline tambem tinha ressalva fisica: a estrela entrou na regiao de sucesso sem grasp verificado. Com apenas tres tarefas heterogeneas, nao ha base para significancia estatistica.

## Videos

Cada MP4 tem 31 frames, 256x256, 2 fps e foi validado por decodificacao:

- `images/episode_1/episode_1.mp4` - estrela e recipiente;
- `images/episode_2/episode_2.mp4` - stacking dos cilindros;
- `images/episode_3/episode_3.mp4` - wiping com esponja.

Os GIFs e os PNGs originais permanecem nas mesmas pastas.

## Proximos experimentos minimos

1. Repetir uma seed apos attachment comparando transporte com garra fechada e aberta; medir attachment, erro de pose e deslocamento objeto-efetor.
2. Repetir um unico alvo preservando a orientacao OpenVLA e usando a orientacao analitica da baseline; isso discrimina orientacao/IK de erro de coordenada.
3. Executar um episodio por tipo com sequencia fixa: grasp verificado, transporte fechado e release apenas no destino; no wiping, manter attachment durante toda a trajetoria.

Esses testes devem manter seeds, indices, budgets e predicado oficial. Aumentar apenas o budget do VLA nao e a primeira opcao: 7/10 chamadas ja consumiram oito chunks sem resolver a incompatibilidade de sequencia.