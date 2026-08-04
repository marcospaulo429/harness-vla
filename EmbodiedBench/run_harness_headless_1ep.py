"""Run one fixed headless episode to diagnose grounding and grasp feedback."""

from datetime import datetime
from pathlib import Path
import sys

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)
from embodiedbench.evaluator.run_artifacts import (
    create_episode_gifs,
    create_run_root,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5:0.5b-instruct'
EXPERIMENT = f"harness_grounding_grasp_1ep_{datetime.now():%Y%m%d_%H%M%S}"

config = {
    'model_name': MODEL,
    'base_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'temperature': 0.0,
    'max_tokens': 1024,
    'disable_thinking': MODEL.startswith('gemma4:'),
    'down_sample_ratio': 1.0,
    'selected_indexes': [0],
    'eval_sets': ['base'],
    'resolution': 256,
    'language_only': 1,
    'exp_name': EXPERIMENT,
    'max_turns': 12,
    'max_env_steps': 30,
    'approach_dz': 8,
    'lift_dz': 6,
    'global_memory_path': '',
    'headless': True,
    'render_mode': None,
    'save_images': True,
    'grasp_object_lift_threshold': 3.0,
    'grasp_max_gripper_object_distance': 8.0,
    'grasp_max_comotion_residual': 2.0,
    'grasp_empty_object_motion_threshold': 1.0,
    'grasp_min_gripper_lift': 3.0,
}


if __name__ == '__main__':
    runs_root = Path(__file__).resolve().parents[1] / 'evaluation_runs'
    run_root = create_run_root(runs_root, EXPERIMENT)
    config['run_root'] = str(run_root)
    evaluator = EB_ManipulationHarnessEvaluator(config)
    evaluator.check_config_valid()
    try:
        evaluator.evaluate_main()
    finally:
        for gif_path in create_episode_gifs(evaluator.log_path, run_root):
            print(f'GIF: {gif_path}')
        print(f'ARTIFACTS: {run_root}')
