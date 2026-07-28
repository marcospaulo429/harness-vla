"""Run the beta-only OpenVLA HTTP backend on three EB-Man episodes.

OpenVLA LIBERO is a frozen alternative backend, not a reproduction of the
Harness VLA paper.  This runner does not load or serve the model; start a
compatible CPU inference server separately and pass its HTTP endpoint.
"""

import argparse
from datetime import datetime
from pathlib import Path

from PIL import Image

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)


SELECTED = [0, 15, 38]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--openvla-url', default='http://127.0.0.1:8000/predict')
    parser.add_argument('--planner-model', default='gemma4:12b')
    parser.add_argument('--planner-base-url', default='http://localhost:11434/v1')
    parser.add_argument(
        '--output-root',
        default='/home/marcos/harness-vla/openvla_cpu_eval',
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


def create_episode_gifs(log_path):
    for episode_dir in sorted((Path(log_path) / 'images').glob('episode_*')):
        image_paths = sorted(
            episode_dir.glob('*_front_rgb.png'),
            key=lambda path: int(path.name.split('_step_')[1].split('_')[0]),
        )
        frames = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                frames.append(image.convert('RGB'))
        if frames:
            gif_path = episode_dir / f'{episode_dir.name}.gif'
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=700,
                loop=0,
            )
            print(f'GIF: {gif_path}')


def main():
    args = parse_args()
    experiment = f'openvla_cpu_{len(args.selected_indexes)}ep_{datetime.now():%Y%m%d_%H%M%S}'
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
        'output_root': args.output_root,
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
    evaluator.evaluate_main()
    create_episode_gifs(evaluator.log_path)
    print(f'DONE: {evaluator.log_path}')


if __name__ == '__main__':
    main()