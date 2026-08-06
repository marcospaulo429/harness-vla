"""Run one native LIBERO task/state through the multi-turn Harness VLA."""

import argparse
from datetime import datetime
from pathlib import Path

from embodiedbench.evaluator.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
)
from embodiedbench.evaluator.libero_multi_turn_run import (
    run_libero_multi_turn_episode,
)
from embodiedbench.evaluator.libero_native_multi_turn import LiberoNativeOffsets
from embodiedbench.evaluator.run_artifacts import create_run_root
from embodiedbench.planner.harness.libero_multi_turn_planner import (
    LiberoMultiTurnPlanner,
)
from embodiedbench.planner.harness.pirlinf_backend import PiRLinfWebsocketBackend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--max-chunks", type=int, default=20)
    parser.add_argument("--max-move-steps", type=int, default=40)
    parser.add_argument("--release-steps", type=int, default=10)
    parser.add_argument("--above-offset-m", type=float, default=0.10)
    parser.add_argument("--release-pose-offset-m", type=float, default=0.05)
    parser.add_argument("--position-tolerance-m", type=float, default=0.01)
    parser.add_argument("--planner-model", default="gemma4:12b")
    parser.add_argument(
        "--planner-base-url", default="http://localhost:11434/v1"
    )
    parser.add_argument("--think", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_runs",
    )
    parser.add_argument(
        "--test-id",
        default="libero_multi_turn_%s"
        % datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return parser.parse_args()


def create_libero_episode(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from openpi_client import image_tools

    suite_type = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_type()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_index < len(initial_states):
        raise IndexError("initial-state-index is outside the task's initial states")
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = SegmentationRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(args.seed)
    return env, str(task.language), initial_states[args.initial_state_index], image_tools


def main():
    args = parse_args()
    run_root = create_run_root(args.output_root, args.test_id)
    env = None
    try:
        env, instruction, initial_state, image_tools = create_libero_episode(args)
        planner = LiberoMultiTurnPlanner(
            args.planner_model,
            base_url=args.planner_base_url,
            think=args.think,
            required_tau="lift_and_grasp",
        )
        backend = PiRLinfWebsocketBackend(
            args.host,
            args.port,
            replan_steps=args.replan_steps,
        )
        budgets = LiberoMultiTurnBudgets(
            max_turns=args.max_turns,
            horizon=args.horizon,
            max_chunks_cap=args.max_chunks,
            max_move_steps=args.max_move_steps,
            release_steps=args.release_steps,
        )
        offsets = LiberoNativeOffsets(
            above_m=args.above_offset_m,
            release_pose_m=args.release_pose_offset_m,
        )
        result = run_libero_multi_turn_episode(
            env=env,
            backend=backend,
            planner=planner,
            initial_state=initial_state,
            instruction=instruction,
            run_root=run_root,
            task_suite=args.task_suite,
            task_id=args.task_id,
            initial_state_index=args.initial_state_index,
            seed=args.seed,
            budgets=budgets,
            offsets=offsets,
            position_tolerance=args.position_tolerance_m,
            resize_with_pad=image_tools.resize_with_pad,
            convert_to_uint8=image_tools.convert_to_uint8,
        )
        print("SUCCESS: %s" % result["episode"]["task_success"])
        print("ARTIFACTS: %s" % run_root)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()