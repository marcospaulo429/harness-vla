# Navigation and LIBERO Evaluations - 2026-08-06

**Commit:** `0827e21760fb0b39a4385caec61fad40d93d10b3`

## Results summary

| Evaluation | Result | Evidence class |
|---|---:|---|
| EB-Navigation Harness, episodes 0-2 | 2/3 (66.7%) | beta-only generalization |
| Native pi0.5/RLinf, LIBERO-Spatial, attempt 1 | 19/20 (95%) | native frozen-VLA baseline |
| Native pi0.5/RLinf, LIBERO-Spatial, attempt 2 | 20/20 (100%) | native frozen-VLA baseline |
| LIBERO VLA-only runtime smoke | 1/1 | reduced protocol smoke |
| LIBERO planner-facing `vla_act` smoke | 1/1 | partial Harness smoke |

The two 20-rollout attempts used the same checkpoint, task suite, task/state
selection, seed and action budget. Their 19/20 and 20/20 scores are both
retained; this observed variation prevents reporting only the best attempt.

## Canonical artifacts

- Navigation metrics and traces:
  `evaluation_runs/harness_nav_3ep_think_20260805_231701/base/results/`
- Navigation videos:
  `evaluation_runs/harness_nav_3ep_think_20260805_231701/videos/`
- Native LIBERO 19/20 attempt:
  `evaluation_runs/20260805_2345_m10_rlinf_native_libero_spatial_20ep/`
- Native LIBERO 20/20 attempt with 20 uniquely named videos:
  `evaluation_runs/20260805_2359_m10_rlinf_native_libero_spatial_20ep_videos_fixed/`
- VLA-only smoke:
  `evaluation_runs/libero_vla_only_task0_state0_20260806/`
- Planner-facing smoke:
  `evaluation_runs/libero_harness_vla_only_gemma_think_task0_state0_20260806/`

## EB-Navigation Harness

Gemma4:12b with thinking ran the fixed episodes `[0, 1, 2]` using the eight
discrete navigation actions. It solved Bread and Pot and failed Toaster at the
12-turn cap.

- success: 2/3;
- environment steps: 41;
- planner turns: 26;
- parse errors: 0;
- mean final target distance: 1.015 m;
- failed episode final/minimum distance: 1.256/1.239 m;
- decisions: 21 `move_forward`, 2 `turn_left`, 3 `turn_right`;
- visual artifacts: three GIFs, one per episode.

This validates the closed-loop planner/action/feedback infrastructure in a
second embodiment. It does not reproduce a benchmark from the Harness VLA
paper.

## Native LIBERO baseline

The frozen checkpoint was evaluated on all ten LIBERO-Spatial tasks with two
initial states per task, seed 7 and horizon 220.

- attempt 1: 19/20; task 5/state 0 failed; metrics complete, but an upstream
  filename collision preserved only 11 videos;
- attempt 2: 20/20; all 20 MP4 files have unique task/state names and decode at
  224x224, with 78-133 frames each;
- checkpoint revision: `6222623f635769bfc73c9472e29fab9b7fd8e027`;
- runtime: V100 eager mode, no `torch.compile`.

The second attempt is the canonical visual baseline. The first remains as
repeatability evidence, not as the preferred result.

## Paired LIBERO smoke summary

Two paired smoke tests on LIBERO spatial task 0, state 0 (seed 7) with identical configuration but different planner modes. Both achieved 100% success (1/1 success).

| Attribute | Value |
|-----------|-------|
| **Task Suite** | LIBERO spatial |
| **Task ID** | 0 |
| **Initial State** | 0 |
| **Seed** | 7 |
| **Replan Steps** | 5 |
| **Max Chunks** | 44 |
| **Horizon** | 220 |

---

## Run 1: `libero_vla_only_task0_state0_20260806`

**Backend:** Frozen pi0.5/RLinf checkpoint (paper-confirmed) through the local websocket adapter (paper-compatible)  
**Planner:** Disabled (VLA-only)  
**Classification:** Smoke/reduced protocol (beta-only)

### Configuration
- `run_type`: `vla_only_smoke`
- `harness_complete`: `false`
- `analytic_primitives_available`: `false`
- `perception`: Text-only task instruction
- `task_memory`: `false`
- `global_memory`: `false`

### Results
- **Episodes:** 1
- **Success Rate:** 100% (1/1)
- **Termination:** `tau_satisfied`
- **Chunks Executed:** 16 / 44
- **Actions Executed:** 78
- **Budget Exhausted:** `false`

### Video Integrity
- **File:** `videos/task_000_state_000_success.mp4`
- **Size:** 40 KB (~40614 bytes)
- **Decoded Frames:** 79
- **Resolution:** 224x224x3

---

## Run 2: `libero_harness_vla_only_gemma_think_task0_state0_20260806`

**Backend:** Frozen pi0.5/RLinf checkpoint (paper-confirmed) through the local websocket adapter in V100 eager mode (paper-compatible)  
**Planner:** Gemma4:12b with thinking enabled (beta-only in this reduced smoke)  
**Classification:** Smoke/reduced protocol (beta-only)

### Configuration
- `run_type`: `harness_vla_only_smoke`
- `harness_complete`: `false`
- `analytic_primitives_available`: `false`
- `perception`: Text-only task instruction
- `task_memory`: `false`
- `global_memory`: `false`
- `planner_enabled`: `true`
- `planner_model`: `gemma4:12b`

### Planner Output
The Gemma model generated coherent reasoning and issued a single `vla_act` invocation with identical max_chunks (44) and tau constraint:

```json
{
  "action": "vla_act",
  "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
  "max_chunks": 44,
  "tau": "task_success"
}
```

### Results
- **Episodes:** 1
- **Success Rate:** 100% (1/1)
- **Termination:** `tau_satisfied`
- **Chunks Executed:** 16 / 44
- **Actions Executed:** 80
- **Budget Exhausted:** `false`

### Video Integrity
- **File:** `videos/task_000_state_000_success.mp4`
- **Size:** 42 KB (~42004 bytes)
- **Decoded Frames:** 81
- **Resolution:** 224x224x3

---

## Limitations & Non-Claims

- **Harness Completeness:** `false` - No full harness initialization; VLA-only or VLA+planner smoke.
- **No Analytic Primitives:** Grasp, placement, and navigation primitives are _not_ available.
- **No Memory Systems:** No task memory, global memory, or RGB-D perception.
- **Single Episode (N=1):** Results are indicative only; no statistical significance.
- **No Performance Claims:** These runs do not demonstrate generalization, learned gains, or competitive benchmarking.
- **Task Instruction Only:** Perception limited to text-based task descriptions.

---

## Metadata

- **Created:** 2026-08-06 00:35–00:37 UTC
- **Hardware:** V100 (eager execution)
- **Status:** Beta (protocol validation only)
