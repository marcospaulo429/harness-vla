"""Run one fixed headless episode to diagnose grounding and grasp feedback."""

from datetime import datetime
import sys

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)


config = {
    'model_name': sys.argv[1] if len(sys.argv) > 1 else 'qwen2.5:0.5b-instruct',
    'base_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'temperature': 0.0,
    'max_tokens': 1024,
    'down_sample_ratio': 1.0,
    'selected_indexes': [0],
    'eval_sets': ['base'],
    'resolution': 256,
    'language_only': 1,
    'exp_name': f"harness_grounding_grasp_1ep_{datetime.now():%Y%m%d_%H%M%S}",
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
    evaluator = EB_ManipulationHarnessEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    print(f'DONE: {evaluator.log_path}')
