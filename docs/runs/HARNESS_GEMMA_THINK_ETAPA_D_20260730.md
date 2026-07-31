# Run: Etapa D — thinking + num_ctx efetivo + re-staging de grasp (2026-07-30)

## Configuração

- Data: 2026-07-30 23:07 — duração 3h07
- Commit do código: `592d7fb` (inclui `c94799c`: num_ctx=8192, num_predict=4096,
  request_timeout=1800, resiliência a timeout; e `592d7fb`: failure model de
  grasp repetido na mesma pose)
- Nota: uma tentativa anterior desta etapa com `num_ctx=16384` crashou por
  socket timeout no turno 3 (turno único > 600 s em CPU) — motivou o `c94799c`.
- Script: `run_harness_demo_3ep.py gemma4:12b --think`, episódios [0, 15, 38]
- Saída: `.../harness_demo_3ep_think_20260730_230747/base`
- GIFs: `videos_think/etapaD_episode_*.gif`

## Resultados

| Episódio | Tarefa | task_success | Turnos | parse_error | Observação |
|---|---|---|---|---|---|
| 1 | pick-and-place | **1.0** | 3 | **0** | zero erros de formato; sequência mínima |
| 2 | stack | 0.0 | 12 | 6 | stack completo executado em 4 turnos; depois ruminação |
| 3 | wipe | 0.0 | ~14 | baixo | re-staging ativo (rotate_wrist/rotate_pitch entre grasps) |

Total: 1/3. Evolução qualitativa forte apesar do placar igual.

## Fatos novos

1. **num_ctx=8192 eliminou o truncamento em turnos normais** — ep1 com 0
   parse_error (Etapas A/C: 3 e 0-8 por episódio).
2. **Ep2 executou o stack completo pela primeira vez** (t1-t4: move_to →
   grasp_verified → move_to postcondition_met → vla_act place
   postcondition_met). A falha mudou de formato/estado para **false success**:
   o place ficou dentro da tolerância beta de 12 voxels, o benchmark não
   sinalizou sucesso, e o feedback não comunicava isso. O modelo ruminou ~15k
   chars/turno por 6 turnos perguntando se a tarefa já estava concluída
   (novo teto de 4096 tokens de thinking atingido nesses turnos).
3. **Ep3 usou o fluxo de recuperação do paper**: rotate_wrist 90°, rotate_pitch
   e novos offsets entre tentativas de grasp — a regra seed `592d7fb` foi
   seguida. A esponja continuou sem anexar (causa física pendente); houve 1
   "Could not create path" (IK) sem crash.

## Fix derivado desta run

Commit `5bac398` (paper-confirmed §3.3, failure model de false success):
o feedback de place com postcondition atingida agora informa
"Benchmark task signal: NOT successful yet - keep acting" quando o ambiente
não sinaliza sucesso. Aplicado na Etapa E.

## Comparativo entre etapas (mesmos episódios [0, 15, 38])

| Run | Sucessos | ep1 turnos | parse_errors totais |
|---|---|---|---|
| baseline scripted | 1/3 | — | — |
| OpenVLA CPU | 0/3 | — | — |
| Etapa A (think 2048) | 1/3 | 9 | 15 |
| Etapa C (+Etapa B estado) | 1/3 | 3 | 10 |
| Etapa D (+num_ctx/re-stage) | 1/3 | 3 | ~6 (concentrados na ruminação do ep2) |
