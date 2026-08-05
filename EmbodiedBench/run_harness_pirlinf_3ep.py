"""Run Harness VLA episodes with the frozen pi0.5/RLinf websocket backend.

Mirrors run_harness_demo_3ep.py but routes vla_act through the OpenPI
websocket policy server (M10). Requires the server to be READY first.
"""

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


def _episodes_from_argv(default):
    for arg in sys.argv[1:]:
        if arg.startswith('--episodes='):
            return [int(x) for x in arg.split('=', 1)[1].split(',') if x]
    return default


SELECTED = _episodes_from_argv([0, 15, 38])
THINK = '--think' in sys.argv
EXPERIMENT = (
    f"harness_pirlinf_{len(SELECTED)}ep_{'think_' if THINK else ''}"
    f"{datetime.now():%Y%m%d_%H%M%S}"
)
MODEL = next(
    (arg for arg in sys.argv[1:] if not arg.startswith('--')),
    'gemma4:12b',
)

config = {
    'model_name': MODEL,
    'base_url': 'http://localhost:11434/v1',
    'api_key': 'ollama',
    'temperature': 0.0,
    # 8192 generation tokens: with think enabled, Gemma's reasoning alone can
    # exceed 4096, returning an empty final answer (observed as parse_error).
    'max_tokens': 8192 if THINK else 1024,
    'num_ctx': 16384 if THINK else None,
    'request_timeout': 1800.0 if THINK else 600.0,
    'disable_thinking': MODEL.startswith('gemma4:') and not THINK,
    'enable_thinking': THINK,
    'down_sample_ratio': 1.0,
    'selected_indexes': SELECTED,
    'eval_sets': ['base'],
    'resolution': 256,
    'language_only': 1,
    'exp_name': EXPERIMENT,
    'max_turns': 12,
    'max_env_steps': 150,
    'approach_dz': 8,
    'lift_dz': 6,
    'move_to_tolerance': 2.0,
    'place_tolerance': 12.0,
    'global_memory_path': '',
    'headless': True,
    'render_mode': 'rgb_array',
    'save_images': True,
    # M10 frozen VLA backend (paper-confirmed mechanism; transport is
    # a paper-compatible integration choice).
    'vla_backend': 'pirlinf_websocket',
    'pirlinf_host': '127.0.0.1',
    'pirlinf_port': 8010,
    'pirlinf_replan_steps': 5,
    'pirlinf_timeout': 120.0,
    'pirlinf_max_chunks': 8,
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
