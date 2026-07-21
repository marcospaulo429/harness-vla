# Harness VLA — beta scope and what is NOT implemented yet

> For the complete architecture, installation history, paper comparison, and
> 10-episode evaluation results, see [HARNESS_VLA_BETA_REPORT.md](HARNESS_VLA_BETA_REPORT.md).

This repository contains a **simplified beta** of the Harness VLA framework
(arXiv:2607.08448) built on top of EmbodiedBench's **EB-Manipulation** environment.
The goal of this beta is a **sanity check**: validate that the harness
architecture (an LLM planner orchestrating a fixed library of analytic primitives
plus one retryable contact-rich primitive, guided by a Global Memory) runs
end-to-end in a simple simulator before applying it to a harder problem.

This document lists what the paper describes that is **intentionally left out**
or **stubbed** in this beta, so the gap to a full implementation is explicit.

## What the beta DOES implement

- **Fixed primitive library** (`embodiedbench/planner/harness/primitives.py`):
  analytic primitives `move_to`, `rotate_wrist`, `rotate_pitch`, `set_gripper`,
  `release`, plus a single contact-rich primitive `vla_act`. Primitives compile
  to the environment's 7-D discrete voxel actions and are unit tested without the
  simulator.
- **Closed-loop planner** (`harness_planner.py`): the LLM emits exactly ONE JSON
  primitive invocation per turn; the loop executes it, re-perceives, and iterates.
- **Global Memory** (`global_memory.py`): a **fixed manual seed** of
  task-independent success rules and failure models, injected into the prompt.
- **Perception as text**: stable object IDs, semantic roles, labels and voxel
  coordinates from `form_harness_grounding_for_input` are passed to the planner
  (language-only). Bindings are refreshed from each observation.
- **Audit traces**: a per-episode JSONL trace of every primitive, its compiled
  actions, and environment feedback.
- **Contact-outcome inspection**: grasp attachment is authoritative when the
  simulator exposes it; lift, distance and co-motion geometry provide supporting
  evidence. Primitive post-condition remains separate from benchmark success.
- **Semantic safety guards**: grasp/place role validation, verified-held-object
  requirement and bounded rejection of repeated no-progress calls.
- **Deterministic runs**: fixed seed, temperature 0.

## What is NOT implemented (deferred)

### 1. Real frozen VLA for `vla_act`
`vla_act` is a **mock scripted** contact primitive: it expands a
`grasp`/`place`/`push` intent into a short burst of analytic sub-actions. There is
**no frozen Vision-Language-Action policy**. The full framework replaces this with
a trained VLA invoked at the moment of contact.

### 2. Task Specific Memory (few-shot / bootstrapping)
The paper's per-task procedural (JSONL) + semantic (JSON) memory that is built
during an exploratory phase is **not implemented**. The beta is **zero-shot
only** — no exploration and no accumulated task traces. Per-turn scene bindings
are refreshed, but this is not Task Specific Memory retrieval or seed transfer.

### 3. Automatic Global Memory updates
Global Memory is a **static hand-written seed**. The paper distills success rules
and failure models from execution traces over time; here nothing is learned or
written back.

### 4. Multimodal / visual perception
The beta is **language-only**: it consumes the object coordinate table as text and
does **not** feed RGB-D images, masks, or rendered coordinate overlays to the
planner. Perception isolation via RGB-D + world maps (paper) is not implemented.

### 5. File-mediated REPL protocol (Appendix A)
The paper's asynchronous REPL contract (`command.json` / `state_NN.json` /
`done_NN.flag` handshake between planner and executor) is replaced by a simple
**in-process loop**. There is no separate process or file-mediated protocol.

### 6. Extended primitive set (mobile / bimanual)
Only the tabletop arm primitives are implemented. Mobile-base primitives
(`navigate_to` / `move_base`) and bimanual coordination are **out of scope**.

### 7. Generalized primitive post-conditions
Grasp now has a dedicated outcome classifier using attachment and quantitative
geometry, and place/release has conservative attachment-aware termination
feedback. Uniform post-condition predicates for every analytic primitive,
including pose-tolerance checks for motion, are still deferred. Local geometric
thresholds are beta configuration details, not universal thresholds from the paper.

### 8. Other benchmarks
Only EB-Manipulation is wired. LIBERO / RoboCasa / RoboTwin and the other
EmbodiedBench environments (alfred, habitat, navigation) are not targeted by the
harness path.

### 9. Model
The default `model_name` is a **tiny** model (`qwen2.5:0.5b-instruct`) chosen only
to validate the pipeline. Real evaluation requires swapping in a capable model via
`base_url` / `model_name` (any OpenAI-compatible endpoint).

## How to run the beta

```bash
# 1. Serve a tiny model with Ollama (OpenAI-compatible endpoint)
ollama pull qwen2.5:0.5b-instruct
ollama serve            # exposes http://localhost:11434/v1

# 2. Run the harness evaluator on a small subset
cd EmbodiedBench
python -m embodiedbench.main env=eb-man-harness eval_sets=[base] down_sample_ratio=0.1
```

Swap the model by editing `embodiedbench/configs/eb-man-harness.yaml`
(`model_name`, `base_url`). Any OpenAI-compatible server (Ollama, llama.cpp
`llama-server`, vLLM) works by changing `base_url` only.
