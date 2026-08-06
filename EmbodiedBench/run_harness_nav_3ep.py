"""Run the beta-only Harness extension on EB-Navigation."""

from datetime import datetime
from pathlib import Path
import sys

from embodiedbench.evaluator.eb_navigation_harness_evaluator import (
    EB_NavigationHarnessEvaluator,
)
from embodiedbench.evaluator.run_artifacts import create_run_root
from embodiedbench.evaluator.run_artifacts import create_episode_gifs


def _episodes_from_argv(default):
    for argument in sys.argv[1:]:
        if argument.startswith("--episodes="):
            return [
                int(value)
                for value in argument.split("=", 1)[1].split(",")
                if value
            ]
    return default


SELECTED = _episodes_from_argv([0, 1, 2])
THINK = "--think" in sys.argv
EXPERIMENT = (
    f"harness_nav_{len(SELECTED)}ep_{'think_' if THINK else ''}"
    f"{datetime.now():%Y%m%d_%H%M%S}"
)
MODEL = next(
    (argument for argument in sys.argv[1:] if not argument.startswith("--")),
    "gemma4:12b",
)
MEMORY_PATH = (
    Path(__file__).resolve().parent
    / "embodiedbench"
    / "planner"
    / "harness"
    / "nav_global_memory.json"
)

config = {
    "model_name": MODEL,
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "temperature": 0.0,
    "max_tokens": 8192 if THINK else 1024,
    "num_ctx": 16384 if THINK else None,
    "request_timeout": 1800.0 if THINK else 600.0,
    "disable_thinking": MODEL.startswith("gemma4:") and not THINK,
    "enable_thinking": THINK,
    "down_sample_ratio": 1.0,
    "selected_indexes": SELECTED,
    "eval_sets": ["base"],
    "resolution": 300,
    "exp_name": EXPERIMENT,
    "max_turns": 12,
    "max_env_steps": 30,
    "global_memory_path": str(MEMORY_PATH),
}


if __name__ == "__main__":
    runs_root = Path(__file__).resolve().parents[1] / "evaluation_runs"
    run_root = create_run_root(runs_root, EXPERIMENT)
    config["run_root"] = str(run_root)
    evaluator = EB_NavigationHarnessEvaluator(config)
    evaluator.check_config_valid()
    try:
        evaluator.evaluate_main()
    finally:
        if evaluator.log_path:
            for gif_path in create_episode_gifs(evaluator.log_path, run_root):
                print(f"GIF: {gif_path}")
        print(f"ARTIFACTS: {run_root}")