"""Artifact-backed orchestration for one native LIBERO multi-turn episode."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from embodiedbench.evaluator.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoMultiTurnEvaluator,
)
from embodiedbench.evaluator.libero_native_multi_turn import (
    LiberoNativeOffsets,
    make_native_move_executor,
    make_native_release_executor,
    make_native_vla_executor,
)
from embodiedbench.evaluator.libero_vla_smoke import LIBERO_DUMMY_ACTION
from embodiedbench.planner.harness.libero_grounding import (
    calibration_from_sim,
    depth_to_meters,
    ground_instance,
)
from embodiedbench.planner.harness.trace_io import (
    resolve_git_commit,
    write_json_atomic,
)


SETTLING_STEPS = 10


def _default_video_writer(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=10)


def _positive_finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be finite and greater than zero" % name) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("%s must be finite and greater than zero" % name)
    return number


def _unpack_step(step_result):
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, reward, bool(done), info
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, reward, bool(terminated or truncated), info
    raise ValueError("env.step must return four or five values")


def _video_frame(observation: Mapping[str, Any], camera: str) -> np.ndarray:
    image = np.asarray(observation["%s_image" % camera])
    return np.ascontiguousarray(image[::-1, ::-1])


def _native_grounder(env, observation, camera: str, frame_id: int):
    depth = np.asarray(observation["%s_depth" % camera])
    height, width = depth.shape[:2]
    sim = env.env.sim
    calibration = calibration_from_sim(sim, camera, height, width, frame_id)
    grounding_frame = frame_id

    def grounder(current_observation, target):
        nonlocal grounding_frame
        grounding_frame += 1
        current_calibration = replace(calibration, frame_id=grounding_frame)
        metric_depth = depth_to_meters(
            sim, current_observation["%s_depth" % camera]
        )
        return ground_instance(
            current_observation,
            env.instance_to_id,
            target,
            current_calibration,
            metric_depth,
        )

    return grounder


def run_libero_multi_turn_episode(
    *,
    env,
    backend,
    planner,
    initial_state,
    instruction: str,
    run_root,
    task_suite: str,
    task_id: int,
    initial_state_index: int,
    seed: int,
    budgets: LiberoMultiTurnBudgets,
    offsets: LiberoNativeOffsets,
    position_tolerance: float,
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
    grounder=None,
    vla_executor=None,
    move_executor=None,
    release_executor=None,
    available_targets: Optional[Sequence[str]] = None,
    camera: str = "agentview",
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] = _default_video_writer,
    backend_name: str = "pirlinf_websocket",
) -> Dict[str, Any]:
    """Run one episode and durably persist its manifest, trace, video and result."""
    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    if not isinstance(budgets, LiberoMultiTurnBudgets):
        raise ValueError("budgets must be a LiberoMultiTurnBudgets")
    if not isinstance(offsets, LiberoNativeOffsets):
        raise ValueError("offsets must be a LiberoNativeOffsets")
    tolerance = _positive_finite(position_tolerance, "position_tolerance")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if not isinstance(camera, str) or not camera:
        raise ValueError("camera must be a non-empty string")

    targets = list(
        getattr(env, "obj_of_interest", ())
        if available_targets is None
        else available_targets
    )
    if not targets or any(not isinstance(target, str) or not target for target in targets):
        raise ValueError("available_targets must contain non-empty strings")

    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    manifest_path = root / "run_manifest.json"
    manifest = {
        "run_type": "libero_multi_turn",
        "status": "in_progress",
        "harness_complete": False,
        "implementation_scope": "reduced_multi_turn_primitive_library",
        "scientific_classification": {
            "paper_confirmed": [
                "one_primitive_per_turn",
                "execute_observe_feedback_replan",
                "vla_act_move_to_release_roles",
                "official_task_success_termination",
            ],
            "paper_compatible": [
                "in_process_loop",
                "rgbd_target_mode_resolution",
                "configured_offsets_tolerances_budgets",
            ],
            "beta_only": [
                "privileged_instance_segmentation",
                "privileged_contact_state",
            ],
        },
        "task_memory": False,
        "global_memory": False,
        "perception": "rgbd_with_privileged_instance_segmentation",
        "privileged_segmentation": True,
        "privileged_contact_state": True,
        "planner_receives_oracle_coordinates": False,
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "planner_model": getattr(planner, "model_name", None),
        "think": bool(getattr(planner, "think", False)),
        "backend": backend_name,
        "backend_host": getattr(backend, "host", None),
        "backend_port": getattr(backend, "port", None),
        "available_targets": targets,
        "config": {
            "settling_steps": SETTLING_STEPS,
            "budgets": asdict(budgets),
            "offsets_m": asdict(offsets),
            "position_tolerance_m": tolerance,
            "camera": camera,
        },
        "git_commit": resolve_git_commit(Path(__file__)),
    }
    write_json_atomic(manifest_path, manifest)

    frames = []
    video_path = None
    try:
        env.reset()
        observation = env.set_init_state(initial_state)
        frames.append(_video_frame(observation, camera))
        for _ in range(SETTLING_STEPS):
            observation, _, _, _ = _unpack_step(
                env.step(list(LIBERO_DUMMY_ACTION))
            )
            frames.append(_video_frame(observation, camera))

        frame_callback = lambda current: frames.append(_video_frame(current, camera))
        active_vla_executor = vla_executor or make_native_vla_executor(
            env,
            backend,
            resize_with_pad=resize_with_pad,
            convert_to_uint8=convert_to_uint8,
            frame_callback=frame_callback,
        )
        if move_executor is None:
            active_grounder = grounder or _native_grounder(
                env, observation, camera, SETTLING_STEPS
            )
            active_move_executor = make_native_move_executor(
                env,
                active_grounder,
                offsets=offsets,
                position_tolerance=tolerance,
                frame_callback=frame_callback,
            )
        else:
            active_move_executor = move_executor
        active_release_executor = release_executor or make_native_release_executor(
            env, frame_callback=frame_callback
        )
        evaluator = LiberoMultiTurnEvaluator(
            planner,
            vla_executor=active_vla_executor,
            move_executor=active_move_executor,
            release_executor=active_release_executor,
            trace_path=root / "trace.jsonl",
        )
        result = evaluator.run(
            instruction.strip(),
            observation,
            available_targets=targets,
            budgets=budgets,
        )

        video_status = "success" if result.task_success else "failure"
        video_path = video_root / (
            "task_%03d_state_%03d_%s.mp4"
            % (task_id, initial_state_index, video_status)
        )
        video_writer(video_path, frames)
        episode = {
            "run_type": manifest["run_type"],
            "task_suite": task_suite,
            "task_id": task_id,
            "initial_state_index": initial_state_index,
            "seed": seed,
            "instruction": instruction.strip(),
            "task_success": result.task_success,
            "primitive_success": result.primitive_success,
            "env_done": result.env_done,
            "termination_reason": result.termination_reason,
            "turns_executed": result.turns_executed,
            "actions_executed": result.steps_executed,
            "horizon": budgets.horizon,
            "holding": result.holding,
            "video": str(video_path.relative_to(root)),
        }
        summary = {
            "episodes": 1,
            "successes": int(result.task_success),
            "task_success_rate": float(result.task_success),
            "turns_executed": result.turns_executed,
            "actions_executed": result.steps_executed,
            "budget_exhausted": result.termination_reason
            in ("max_turns_exhausted", "horizon_exhausted"),
            "termination_reason": result.termination_reason,
        }
        write_json_atomic(root / "episode.json", episode)
        write_json_atomic(root / "summary.json", summary)
        manifest.update(
            {
                "status": "completed",
                "task_success": result.task_success,
                "termination_reason": result.termination_reason,
                "video": str(video_path.relative_to(root)),
            }
        )
        write_json_atomic(manifest_path, manifest)
        return {"episode": episode, "summary": summary, "manifest": manifest}
    except Exception as exc:
        if frames:
            video_path = video_root / (
                "task_%03d_state_%03d_incomplete.mp4"
                % (task_id, initial_state_index)
            )
            try:
                video_writer(video_path, frames)
            except Exception:
                video_path = None
        manifest.update(
            {
                "status": "incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if video_path is not None:
            manifest["video"] = str(video_path.relative_to(root))
        write_json_atomic(manifest_path, manifest)
        raise


__all__ = ["SETTLING_STEPS", "run_libero_multi_turn_episode"]