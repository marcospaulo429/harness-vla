"""Minimal native LIBERO smoke for a frozen, chunk-producing VLA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from embodiedbench.planner.harness.pirlinf_backend import PiRLinfObservation
from embodiedbench.planner.harness.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    resolve_git_commit,
    write_json_atomic,
)
from embodiedbench.planner.harness.vla_runtime import VLARuntime


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def prepare_pirlinf_observation(
    observation: Dict[str, Any],
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
) -> PiRLinfObservation:
    """Apply the OpenPI LIBERO image and state preprocessing contract."""
    front = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    front = convert_to_uint8(resize_with_pad(front, 224, 224))
    wrist = convert_to_uint8(resize_with_pad(wrist, 224, 224))
    gripper_pose = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            np.asarray(observation["robot0_eef_quat"]),
        )
    )
    return PiRLinfObservation(
        front_rgb=np.asarray(front, dtype=np.uint8),
        wrist_rgb=np.asarray(wrist, dtype=np.uint8),
        gripper_pose=gripper_pose,
        gripper_qpos=np.asarray(observation["robot0_gripper_qpos"]),
        mode="task",
    )


def _unpack_step(step_result):
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, reward, bool(done), info
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, reward, bool(terminated or truncated), info
    raise ValueError("env.step must return four or five values")


def _default_video_writer(path: Path, frames: Sequence[np.ndarray]) -> None:
    import imageio.v2 as imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=10)


def run_libero_vla_smoke(
    *,
    env,
    backend,
    initial_state,
    prompt: str,
    run_root,
    task_suite: str,
    task_id: int,
    initial_state_index: int,
    seed: int,
    replan_steps: int,
    max_chunks: int,
    horizon: int,
    resize_with_pad: Callable[..., np.ndarray],
    convert_to_uint8: Callable[[np.ndarray], np.ndarray],
    planner=None,
    host: str = "127.0.0.1",
    port: int = 8000,
    video_writer: Callable[[Path, Sequence[np.ndarray]], None] = _default_video_writer,
) -> Dict[str, Any]:
    """Run one native LIBERO task/state smoke and persist its audit artifacts."""
    for name, value in (
        ("replan_steps", replan_steps),
        ("max_chunks", max_chunks),
        ("horizon", horizon),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("%s must be an integer greater than zero" % name)

    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be an existing directory")
    video_root = root / "videos"
    video_root.mkdir(exist_ok=True)
    trace_path = root / "trace.jsonl"
    initialize_jsonl(trace_path)

    run_type = "harness_vla_only_smoke" if planner is not None else "vla_only_smoke"
    manifest = {
        "run_type": run_type,
        "harness_complete": False,
        "analytic_primitives_available": False,
        "task_memory": False,
        "global_memory": False,
        "perception": "text_only_task_instruction",
        "planner_enabled": planner is not None,
        "planner_model": getattr(planner, "model_name", None),
        "planner_raw_output": None,
        "planner_thinking": None,
        "planner_invocation": None,
        "backend": "pirlinf_websocket",
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "host": host,
        "port": port,
        "replan_steps": replan_steps,
        "max_chunks": max_chunks,
        "horizon": horizon,
        "git_commit": resolve_git_commit(Path(__file__)),
    }

    env.reset()
    raw_observation = env.set_init_state(initial_state)
    for _ in range(10):
        raw_observation, _, _, _ = _unpack_step(env.step(list(LIBERO_DUMMY_ACTION)))

    live_observation = prepare_pirlinf_observation(
        raw_observation, resize_with_pad, convert_to_uint8
    )
    video_frames = [live_observation.front_rgb]
    executed_action_count = 0
    task_success = False
    chunk_traces = []
    runtime_chunk_limit = min(max_chunks, int(math.ceil(float(horizon) / replan_steps)))
    effective_prompt = prompt
    planner_parse_error = False
    if planner is not None:
        invocation, raw_output = planner.act(prompt, runtime_chunk_limit)
        thinking = getattr(planner, "last_thinking", None)
        manifest.update(
            {
                "planner_raw_output": raw_output,
                "planner_thinking": thinking,
                "planner_invocation": invocation,
            }
        )
        if invocation is None:
            planner_parse_error = True
        else:
            effective_prompt = invocation["prompt"]
            runtime_chunk_limit = invocation["max_chunks"]
    write_json_atomic(root / "run_manifest.json", manifest)

    def execute_chunk(chunk):
        nonlocal executed_action_count, live_observation, raw_observation, task_success
        executed_actions = []
        rewards = []
        dones = []
        successes = []
        for raw_action in chunk.raw_deltas:
            if executed_action_count >= horizon or task_success:
                break
            action = np.asarray(raw_action, dtype=float)
            raw_observation, reward, done, _ = _unpack_step(
                env.step(action.tolist())
            )
            executed_action_count += 1
            checker = getattr(env, "check_success", None)
            task_success = bool(done) or bool(checker() if callable(checker) else False)
            executed_actions.append(action.tolist())
            rewards.append(float(reward))
            dones.append(bool(done))
            successes.append(task_success)
            live_observation = prepare_pirlinf_observation(
                raw_observation, resize_with_pad, convert_to_uint8
            )
            video_frames.append(live_observation.front_rgb)
            if task_success:
                break

        trace = {
            "prompt": effective_prompt,
            "chunk_index": len(chunk_traces) + 1,
            "inference_duration_s": float(chunk.inference_duration_s),
            "full_chunk_length": int(chunk.full_chunk_length),
            "executed_actions": executed_actions,
            "rewards": rewards,
            "dones": dones,
            "task_successes": successes,
            "tau_satisfied": task_success,
            "planner_raw_output": manifest["planner_raw_output"],
            "planner_thinking": manifest["planner_thinking"],
            "planner_invocation": manifest["planner_invocation"],
        }
        chunk_traces.append(trace)
        append_jsonl_record(trace_path, trace)
        return live_observation

    runtime_result = None
    if planner_parse_error:
        append_jsonl_record(
            trace_path,
            {
                "event": "planner_parse_error",
                "prompt": prompt,
                "planner_raw_output": manifest["planner_raw_output"],
                "planner_thinking": manifest["planner_thinking"],
                "planner_invocation": None,
                "tau_satisfied": False,
            },
        )
    else:
        runtime_result = VLARuntime(backend).run(
            live_observation,
            effective_prompt,
            runtime_chunk_limit,
            lambda _: task_success,
            execute_chunk,
        )

    status = "success" if task_success else "failure"
    video_name = "task_%03d_state_%03d_%s.mp4" % (
        task_id,
        initial_state_index,
        status,
    )
    video_path = video_root / video_name
    video_writer(video_path, video_frames)

    episode = {
        "run_type": run_type,
        "task_suite": task_suite,
        "task_id": task_id,
        "initial_state_index": initial_state_index,
        "prompt": effective_prompt,
        "planner_raw_output": manifest["planner_raw_output"],
        "planner_thinking": manifest["planner_thinking"],
        "planner_invocation": manifest["planner_invocation"],
        "task_success": task_success,
        "tau_satisfied": False if runtime_result is None else runtime_result.tau_satisfied,
        "termination_reason": (
            "planner_parse_error"
            if runtime_result is None
            else runtime_result.termination_reason
        ),
        "chunks_executed": 0 if runtime_result is None else runtime_result.chunks_executed,
        "actions_executed": executed_action_count,
        "horizon": horizon,
        "video": str(video_path.relative_to(root)),
    }
    summary = {
        "episodes": 1,
        "successes": int(task_success),
        "task_success_rate": float(task_success),
        "budget_exhausted": (
            runtime_result is not None
            and runtime_result.termination_reason == "budget_exhausted"
        ),
        "termination_reason": episode["termination_reason"],
    }
    write_json_atomic(root / "episode.json", episode)
    write_json_atomic(root / "summary.json", summary)
    return {"episode": episode, "summary": summary, "manifest": manifest}


__all__ = [
    "LIBERO_DUMMY_ACTION",
    "prepare_pirlinf_observation",
    "run_libero_vla_smoke",
]