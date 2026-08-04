"""Run the beta-only OpenVLA HTTP backend on three EB-Man episodes.

OpenVLA LIBERO is a frozen alternative backend, not a reproduction of the
Harness VLA paper.  This runner does not load or serve the model; start a
compatible CPU inference server separately and pass its HTTP endpoint.
"""

import argparse
from datetime import datetime
from pathlib import Path

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)
from embodiedbench.evaluator.run_artifacts import (
    create_episode_gifs,
    create_run_root,
)


SELECTED = [0, 15, 38]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openvla-url', default='http://127.0.0.1:8000/predict')
    parser.add_argument('--planner-model', default='gemma4:12b')
    parser.add_argument('--planner-base-url', default='http://localhost:11434/v1')
    parser.add_argument(
        '--output-root',
        default=str(Path(__file__).resolve().parents[1] / 'evaluation_runs'),
    )
    parser.add_argument('--max-chunks', type=int, default=8)
    parser.add_argument('--max-delta-xyz', type=float, default=0.05)
    parser.add_argument('--max-delta-rotation', type=float, default=0.5)
    parser.add_argument('--openvla-timeout', type=float, default=600.0)
    parser.add_argument(
        '--selected-indexes',
        type=int,
        nargs='+',
        default=SELECTED,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = f'openvla_cpu_{len(args.selected_indexes)}ep_{datetime.now():%Y%m%d_%H%M%S}'
    run_root = create_run_root(args.output_root, experiment)
    config = {
        'model_name': args.planner_model,
        'base_url': args.planner_base_url,
        'api_key': 'ollama',
        'temperature': 0.0,
        'max_tokens': 1024,
        'disable_thinking': True,
        'down_sample_ratio': 1.0,
        'selected_indexes': args.selected_indexes,
        'eval_sets': ['base'],
        'resolution': 256,
        'language_only': 1,
        'exp_name': experiment,
        'run_root': str(run_root),
        'max_turns': 12,
        'max_env_steps': 30,
        'move_to_tolerance': 2.0,
        'place_tolerance': 12.0,
        'global_memory_path': '',
        'headless': True,
        'render_mode': 'rgb_array',
        'save_images': True,
        'vla_backend': 'openvla_http',
        'openvla_url': args.openvla_url,
        'openvla_timeout': args.openvla_timeout,
        'openvla_max_chunks': args.max_chunks,
        'openvla_max_delta_xyz': args.max_delta_xyz,
        'openvla_max_delta_rotation': args.max_delta_rotation,
        'openvla_gripper_convention': 'libero_minus_open_plus_close',
        'openvla_rotation_frame': 'local',
        'openvla_unnorm_key': 'libero_object',
    }
    evaluator = EB_ManipulationHarnessEvaluator(config)
    evaluator.check_config_valid()
    try:
        evaluator.evaluate_main()
    finally:
        for gif_path in create_episode_gifs(evaluator.log_path, run_root):
            print(f'GIF: {gif_path}')
        print(f'ARTIFACTS: {run_root}')


if __name__ == '__main__':
    main()