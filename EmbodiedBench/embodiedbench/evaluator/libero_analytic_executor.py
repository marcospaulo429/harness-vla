"""Closed-loop execution for native LIBERO analytic pose primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from embodiedbench.planner.harness.libero_primitives import (
    LiberoPrimitiveError,
    compile_move_action,
    pose_postcondition,
)


def _unpack_step(step_result):
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return observation, float(reward), bool(done), info
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, float(reward), bool(terminated or truncated), info
    raise LiberoPrimitiveError("env.step must return four or five values")


@dataclass(frozen=True)
class LiberoPrimitiveExecution:
    observation: Dict[str, Any]
    primitive_success: bool
    task_success: bool
    termination_reason: str
    steps_executed: int
    trace: List[Dict[str, Any]]


def execute_pose_primitive(
    env,
    observation: Dict[str, Any],
    target_xyz: Sequence[float],
    *,
    gripper: str,
    max_steps: int,
    position_tolerance: float,
    target_quaternion: Optional[Sequence[float]] = None,
    rotation_tolerance: Optional[float] = None,
) -> LiberoPrimitiveExecution:
    """Recompile and execute OSC deltas until a physical stop condition fires."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise LiberoPrimitiveError("max_steps must be an integer greater than zero")

    current = observation
    trace = []
    checker = getattr(env, "check_success", None)
    for step_index in range(max_steps + 1):
        postcondition = pose_postcondition(
            current,
            target_xyz,
            position_tolerance=position_tolerance,
            target_quaternion=target_quaternion,
            rotation_tolerance=rotation_tolerance,
        )
        task_success = bool(checker() if callable(checker) else False)
        if postcondition["postcondition_met"]:
            return LiberoPrimitiveExecution(
                current, True, task_success, "postcondition_met", step_index, trace
            )
        if task_success:
            return LiberoPrimitiveExecution(
                current, False, True, "task_success", step_index, trace
            )
        if step_index == max_steps:
            return LiberoPrimitiveExecution(
                current, False, False, "step_budget_exhausted", step_index, trace
            )

        action = compile_move_action(
            current,
            target_xyz,
            gripper=gripper,
            target_quaternion=target_quaternion,
        )
        current, reward, done, info = _unpack_step(env.step(action.tolist()))
        trace.append(
            {
                "step": step_index + 1,
                "action": action.tolist(),
                "reward": reward,
                "env_done": done,
                "info": info,
            }
        )
        if done:
            task_success = bool(checker() if callable(checker) else False)
            return LiberoPrimitiveExecution(
                current, False, task_success, "env_done", step_index + 1, trace
            )

    raise AssertionError("unreachable")


__all__ = ["LiberoPrimitiveExecution", "execute_pose_primitive"]