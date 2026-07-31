# Run: Etapa E — thinking + sinal de false-success no place (2026-07-31)

## Configuração

- Data: 2026-07-31 02:15 — 1/3
- Commit do código: `5bac398` (sinal de benchmark no feedback do place quando a
  postcondition beta é atingida mas o benchmark não sinaliza sucesso — §3.3
  false-success model, paper-confirmed) sobre `c94799c` (num_ctx=8192,
  request_timeout=1800, resiliência a timeout).
- Script: `run_harness_demo_3ep.py gemma4:12b --think`, episódios [0, 15, 38]
- Saída: `.../harness_demo_3ep_think_20260731_021532/base`
- GIFs: `videos_think/etapaE_episode_*.gif`

## Resultados

| Episódio | Tarefa | task_success | Turnos | Steps | Observação |
|---|---|---|---|---|---|
| 1 | pick-and-place | **1.0** | 3 | 6 | sucesso perfeito, 0 parse_error, run mais rápida (~6 min) |
| 2 | stack | 0.0 | 12 | 11 | stack executado (grasp_verified + place); geometria errada |
| 3 | wipe | 0.0 | 12 | 27 | grasp_unverified 5x + re-staging; esponja nunca anexa |

Total: **1/3** (mesmo placar de A/C/D, com causas residuais agora isoladas).

## Fatos confirmados

1. **Sinal de false-success funcionou parcialmente.** No ep2, o place (t4) recebeu
   "Benchmark task signal: NOT successful yet …". O modelo ainda ruminou 6 turnos
   (t5–t10, `raw_output` vazio com ~15.4k chars de thinking = teto de 4096 tokens),
   mas **saiu da ruminação** em t11 (`move_to`) e t12 (`release`) — na Etapa D
   ruminava até o fim do episódio.

2. **Causa raiz do ep2 (stack) — geometria do place (beta-only).**
   `vla_act place` compila a descida para `on.z = z_destino`
   ([primitives.py](../../EmbodiedBench/embodiedbench/planner/harness/primitives.py#L665))
   — a altura do **próprio** objeto de destino. Para empilhar, o objeto segurado
   precisaria ir para `z_destino + altura_do_objeto` (em cima), não no mesmo nível.
   `target_voxel [53,40,17]` = centróide de object 6; os dois cilindros ficaram no
   mesmo z (interpenetração), então o benchmark nunca marca stack.
   Verificação do paper (`pesquisador-paper`): a geometria de contato/soltura é
   responsabilidade do **VLA frozen**, ausente na beta (OpenVLA não roda em CPU).
   Corrigir isso analiticamente é **beta-only** (compensa a ausência do VLA), não
   um mecanismo do paper.

3. **Causa raiz do ep3 (wipe) — pose de grasp da esponja (categoria 6, física).**
   Diagnóstico do `diagnostico-simulador`: o corpo graspável é `sponge0` (collider
   ~58×125×33 mm), menor e deslocado em relação ao mesh visual `sponge_visual0`
   (~88×176×51 mm) que o grounding publica. O attach depende do
   `Panda_gripper_attachProxSensor` (volume estreito ~20×100×50 mm) detectar
   `sponge0`. A pose comprovadamente detectável no ep 38 é offset local
   (+10.8,+37.3,+5.4) mm com orientação ~(-180°,0°,-93.2°). O `vla_act grasp`
   analítico desce ao centróide visual e mantém a orientação corrente — nunca
   intercepta o collider. `rotate_wrist/rotate_pitch` em torno do centróide não
   reproduzem a transformação 6-DoF necessária. Resolver isso exige pose de grasp
   6-DoF (papel do VLA) — **beta-only/paper-compatible**, fora do escopo paper-only.

## Conclusão de fronteira (paper-only)

Os mecanismos do **planner do Harness** que implementamos e que o paper especifica
(thinking, memória global, failure models, reconciliação de estado, grounding,
vocabulário de primitivas) estão **funcionando**: na tarefa cujo contato o
substituto analítico consegue satisfazer (pick-place), o sucesso é **reprodutível**
(ep1 em A/C/D/E, com 0 parse_error na E). As duas falhas residuais (stack, wipe)
estão no **executor de contato**, que o paper delega a um **VLA frozen** que não
temos rodando. Completá-las requer o VLA real ou compensação **beta-only** de
física — ambos explicitamente "além do paper".

## Comparativo (episódios [0, 15, 38])

| Run | Sucessos | ep1 turnos | parse_errors | Falha residual dominante |
|---|---|---|---|---|
| baseline scripted | 1/3 | — | — | — |
| OpenVLA CPU | 0/3 | — | — | VLA inviável em CPU |
| Etapa A | 1/3 | 9 | 15 | formato/truncamento |
| Etapa C | 1/3 | 3 | 10 | estado + truncamento |
| Etapa D | 1/3 | 3 | ~6 | ruminação de false-success |
| Etapa E | 1/3 | 3 | ~6 (só ep2) | contato: geometria/attach (papel do VLA) |
