# Run: gemma4:12b + thinking nativo — 3 episódios fixos (2026-07-30)

## Configuração

- Data: 2026-07-30 17:24 — duração 2h50 (3413 s/episódio em média)
- Commit do código: `7578ab7` (thinking support; **não** inclui Etapa B `b5c6092`)
- Script: `EmbodiedBench/run_harness_demo_3ep.py gemma4:12b --think`
- Episódios: índices fixos [0, 15, 38] (pick-and-place, stack, wipe)
- Planner: gemma4:12b via Ollama nativo `/api/chat`, `think=true`,
  temperature 0, `max_tokens 2048`, 12 turnos, 30 env steps
- Saída: `EmbodiedBench/running/eb_manipulation_harness/gemma4_12b/harness_demo_3ep_think_20260730_172440/base`
- GIFs copiados para `videos_think/`

## Classificação de fidelidade

- Thinking nativo do modelo: **paper-compatible** (o paper usa planners de
  raciocínio forte, mas não exige chain-of-thought; a separação
  thinking/content preserva o contrato JSON de uma primitiva por turno).
- Budget de tokens: **beta-only** (infraestrutura local).

## Resultados

| Episódio | Tarefa | task_success | Turnos | parse_error | Observação |
|---|---|---|---|---|---|
| 1 (idx 0) | pick-and-place estrela | **1.0** | 9 | 3 | grasp_verified → transporte → place concluiu a tarefa |
| 2 (idx 15) | stack cilindros | 0.0 | 12 | 7 | plano correto, mas 7/12 turnos queimados em parse_error |
| 3 (idx 38) | wipe | 0.0 | 12 | 5 | grasp_unverified/empty_grasp repetidos no objeto 9 |

Total: **1/3** — igual ao baseline scripted (1/3) e melhor que OpenVLA CPU (0/3),
mas o episódio 1 falhava nas runs anteriores, então a composição de acertos mudou.

## Diagnóstico principal (categoria 1 — formato do LLM)

Todos os 15 parse_error da run têm `raw_text` vazio e `planner_thinking` com
6.4–7.4k caracteres truncados no meio da frase: o thinking consumiu o budget
inteiro de 2048 tokens e o JSON nunca foi emitido. Turnos bem-sucedidos tinham
thinking de 600–1600 caracteres. A ruminação longa ocorre tipicamente após
`target_pose_not_reached`, quando o modelo re-deriva o contrato de `move_to`.

**Fix aplicado** (commit `e311e67`): `max_tokens 8192` no modo think.

## Diagnóstico secundário

- Ep2: após o place `unverified` no turno 9, o planner re-grasped corretamente
  no turno 12 — comportamento de re-tentativa correto, faltou orçamento de turnos.
- Ep3 (wipe): `grasp_unverified (ambiguous_geometry)` 3× e depois `empty_grasp`.
  O paper não publica primitiva de wiping (contrato ausente); tratamento via
  grasp/place é limitação conhecida.

## Próxima run

Etapa C: thinking + reconciliação de estado (commit `b5c6092`) + budget 8192,
mesmos episódios, para medir o efeito conjunto.
