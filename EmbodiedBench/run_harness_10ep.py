"""Ad-hoc runner for the Harness VLA beta on EB-Manipulation (10-episode sanity run).

Bypasses Hydra to give explicit control over which episodes run. Selects 10
episodes spread across the four base tasks so the analysis covers task diversity.
Run from the EmbodiedBench root with the eb_manipulation dir on PYTHONPATH and the
CoppeliaSim env vars exported (see .harness_env.sh).
"""

import sys

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)

# base set = 48 episodes: pick_cube[0-11], place_sorter[12-23],
# stack[24-35], wipe[36-47]. Spread 10 indices across all four tasks.
SELECTED = [0, 5, 10, 15, 19, 24, 29, 34, 38, 43]

config = {
    'model_name': sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5:0.5b-instruct',
    'base_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'temperature': 0.0,
    'max_tokens': 1024,
    'down_sample_ratio': 1.0,
    'selected_indexes': SELECTED,
    'eval_sets': ['base'],
    'resolution': 256,
    'language_only': 1,
    'exp_name': 'harness_beta_10ep',
    'max_turns': 12,
    'max_env_steps': 30,
    'approach_dz': 8,
    'lift_dz': 6,
    'global_memory_path': '',
}

if __name__ == '__main__':
    evaluator = EB_ManipulationHarnessEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    print('DONE')
