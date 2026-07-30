"""Run three Harness VLA episodes and save reviewable GIFs.

The selected episodes cover pick-and-place, shape sorting, and wiping. Stable
offscreen rendering is used because CoppeliaSim 4.1's windowed OpenGL3 renderer
can crash during long frame-capture runs. Front-camera frames and one GIF per
episode are persisted under the experiment folder.
"""

from datetime import datetime
from pathlib import Path
import sys

from PIL import Image

from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
    EB_ManipulationHarnessEvaluator,
)


SELECTED = [0, 15, 38]
THINK = '--think' in sys.argv
EXPERIMENT = (
    f"harness_demo_3ep_{'think_' if THINK else ''}{datetime.now():%Y%m%d_%H%M%S}"
)
MODEL = next(
    (arg for arg in sys.argv[1:] if not arg.startswith('--')),
    'qwen2.5:0.5b-instruct',
)

config = {
    'model_name': MODEL,
    'base_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'temperature': 0.0,
    'max_tokens': 2048 if THINK else 1024,
    'disable_thinking': MODEL.startswith('gemma4:') and not THINK,
    'enable_thinking': THINK,
    'down_sample_ratio': 1.0,
    'selected_indexes': SELECTED,
    'eval_sets': ['base'],
    'resolution': 256,
    'language_only': 1,
    'exp_name': EXPERIMENT,
    'max_turns': 12,
    'max_env_steps': 30,
    'approach_dz': 8,
    'lift_dz': 6,
    'move_to_tolerance': 2.0,
    'place_tolerance': 12.0,
    'global_memory_path': '',
    'headless': True,
    'render_mode': 'rgb_array',
    'save_images': True,
}


def create_episode_gifs(log_path):
    image_root = Path(log_path) / 'images'
    for episode_dir in sorted(image_root.glob('episode_*')):
        frames = []
        for image_path in sorted(
            episode_dir.glob('*.png'),
            key=lambda path: int(path.name.split('_step_')[1].split('_')[0]),
        ):
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


if __name__ == '__main__':
    evaluator = EB_ManipulationHarnessEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
    create_episode_gifs(evaluator.log_path)
    print(f'DONE: {evaluator.log_path}')