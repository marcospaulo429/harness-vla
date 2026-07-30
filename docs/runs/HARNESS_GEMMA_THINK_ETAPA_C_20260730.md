# Run: gemma4:12b + thinking + reconciliação de estado (Etapa C) — 2026-07-30

## Configuração

- Data: 2026-07-30 20:16 — duração 2h32
- Commit do código: `e311e67` (inclui Etapa B `b5c6092`: reconciliação de
  `held_object_id`, estado físico no prompt, failure model de detach; e o
  aumento de `num_predict` para 8192 — **ineficaz**, ver diagnóstico)
- Script: `run_harness_demo_3ep.py gemma4:12b --think`, episódios [0, 15, 38]
- Saída: `.../harness_demo_3ep_think_20260730_201633/base`
- GIFs: `videos_think/etapaC_episode_*.gif`

## Resultados

| Episódio | Tarefa | task_success | Turnos | parse_error | vs Etapa A |
|---|---|---|---|---|---|
| 1 | pick-and-place | **1.0** | **3** (6 steps) | **0** | 9 turnos, 3 parse_error |
| 2 | stack | 0.0 | 12 | 8 | 12 turnos, 7 parse_error |
| 3 | wipe | 0.0 | 12 | 2 | 12 turnos, 5 parse_error |

Total: 1/3 (igual Etapa A), mas o episódio 1 ficou 3× mais eficiente
(3 turnos vs 9; 629 s vs 2185 s).

## Efeitos da Etapa B observados (paper-confirmed §2.1/§2.2)

- Ep2 t3: `released_outside_destination_tolerance` — o place prematuro foi
  detectado com evidência de attachment e `held_object_id` foi limpo na hora
  (na Etapa A esse estado ficava obsoleto).
- Ep2 t7: re-grasp verificado após a falha — o planner leu o estado
  autoritativo do prompt e corrigiu o plano.

## Diagnóstico principal: truncamento persistiu

Os 10 parse_error da run ainda têm thinking de 7.3–8.3k chars truncado no meio
da frase. O `num_predict=8192` foi ineficaz porque o **`num_ctx` default do
Ollama (4096)** limita prompt+thinking+saída. Fix real no commit `59aa1cd`
(planner envia `num_ctx=16384` no modo think) — aplicado APÓS esta run.

## Diagnóstico secundário: wipe (ep3)

Loop de `grasp_unverified (ambiguous_geometry)` e `empty_grasp` no objeto 9:
o grasp da esponja nunca gera evidência de attachment. O paper não publica
primitiva de wiping; investigação pendente sobre por que o attachment não
ocorre fisicamente (categoria 6/3).

## Próxima run

Etapa D (em andamento): idêntica + `num_ctx=16384` efetivo. Expectativa:
eliminar parse_errors e destravar o ep2, cujo plano já está correto.
