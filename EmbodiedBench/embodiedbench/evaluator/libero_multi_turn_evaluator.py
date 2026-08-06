"""Synchronous, injectable worker loop for LIBERO multi-turn planning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from embodiedbench.evaluator.libero_analytic_executor import (
    LiberoPrimitiveExecution,
)
from embodiedbench.planner.harness.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
)


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be an integer greater than zero" % name)


@dataclass(frozen=True)
class LiberoMultiTurnBudgets:
    max_turns: int
    horizon: int
    max_chunks_cap: int
    max_move_steps: int
    release_steps: int

    def __post_init__(self) -> None:
        for name in (
            "max_turns",
            "horizon",
            "max_chunks_cap",
            "max_move_steps",
            "release_steps",
        ):
            _positive_integer(getattr(self, name), name)


@dataclass(frozen=True)
class LiberoVLAExecution:
    """VLA result with explicit semantic effects outside the physical trace."""

    execution: LiberoPrimitiveExecution
    tau_satisfied: bool
    holding: Optional[str]


@dataclass(frozen=True)
class LiberoMultiTurnResult:
    observation: Dict[str, Any]
    holding: Optional[str]
    primitive_success: bool
    task_success: bool
    env_done: bool
    termination_reason: str
    turns_executed: int
    steps_executed: int


PrimitiveExecutor = Callable[..., LiberoPrimitiveExecution]
VLAExecutor = Callable[..., LiberoVLAExecution]


class LiberoMultiTurnEvaluator:
    """Run one planner-selected primitive per turn until success or budget."""

    def __init__(
        self,
        planner,
        *,
        vla_executor: VLAExecutor,
        move_executor: PrimitiveExecutor,
        release_executor: PrimitiveExecutor,
        trace_path,
        file_repl_bridge=None,
    ) -> None:
        self.planner = planner
        self.vla_executor = vla_executor
        self.move_executor = move_executor
        self.release_executor = release_executor
        self.trace_path = Path(trace_path)
        self.file_repl_bridge = file_repl_bridge

    def run(
        self,
        instruction: str,
        observation: Dict[str, Any],
        *,
        available_targets: Sequence[str],
        budgets: LiberoMultiTurnBudgets,
        holding: Optional[str] = None,
        object_labels: Optional[Dict[str, str]] = None,
        object_roles: Optional[Dict[str, Sequence[str]]] = None,
        memory_context: Optional[str] = None,
        protocol_metadata: Optional[Dict[str, Any]] = None,
    ) -> LiberoMultiTurnResult:
        initialize_jsonl(self.trace_path)
        current_observation = observation
        current_holding = holding
        last_action = None
        last_feedback = None
        steps_executed = 0
        primitive_success = False
        labels = dict(object_labels or {})
        roles = {
            target: list(target_roles)
            for target, target_roles in (object_roles or {}).items()
        }

        for turn_index in range(1, budgets.max_turns + 1):
            actions_remaining = budgets.horizon - steps_executed
            if actions_remaining <= 0:
                return self._result(
                    current_observation,
                    current_holding,
                    primitive_success,
                    False,
                    False,
                    "horizon_exhausted",
                    turn_index - 1,
                    steps_executed,
                )

            state = {
                "instruction": instruction,
                "grounded_targets": list(available_targets),
                "holding": current_holding,
                "last_action": last_action,
                "last_feedback": last_feedback,
                "budget": {
                    "turns_remaining": budgets.max_turns - turn_index + 1,
                    "actions_remaining": actions_remaining,
                },
            }
            invocation, raw_output = self.planner.act_turn(
                instruction,
                state,
                available_targets=available_targets,
                max_chunks_cap=min(budgets.max_chunks_cap, actions_remaining),
                **(
                    {"memory_context": memory_context}
                    if memory_context is not None
                    else {}
                ),
            )
            planner_thinking = getattr(self.planner, "last_thinking", None)

            if invocation is None:
                feedback = self._feedback(
                    None,
                    False,
                    False,
                    False,
                    "planner_parse_error",
                    0,
                    current_holding,
                    False,
                )
                self._record(
                    turn_index,
                    raw_output,
                    planner_thinking,
                    None,
                    feedback,
                    [],
                    labels,
                    roles,
                )
                return self._result(
                    current_observation,
                    current_holding,
                    False,
                    False,
                    False,
                    "planner_parse_error",
                    turn_index,
                    steps_executed,
                )

            guard_reason = self._guard_invocation(invocation, current_holding)
            if guard_reason is not None:
                feedback = self._feedback(
                    invocation.get("action"),
                    False,
                    False,
                    False,
                    guard_reason,
                    0,
                    current_holding,
                    False,
                )
                self._record(
                    turn_index,
                    raw_output,
                    planner_thinking,
                    invocation,
                    feedback,
                    [],
                    labels,
                    roles,
                )
                return self._result(
                    current_observation,
                    current_holding,
                    False,
                    False,
                    False,
                    guard_reason,
                    turn_index,
                    steps_executed,
                )

            action = invocation["action"]
            step_budget = self._step_budget(action, actions_remaining, budgets)
            if action == "vla_act" and invocation.get("max_chunks", 0) > min(
                budgets.max_chunks_cap, actions_remaining
            ):
                feedback = self._feedback(
                    action,
                    False,
                    False,
                    False,
                    "primitive_compile_error",
                    0,
                    current_holding,
                    False,
                )
                self._record(
                    turn_index,
                    raw_output,
                    planner_thinking,
                    invocation,
                    feedback,
                    [],
                    labels,
                    roles,
                )
                return self._result(
                    current_observation,
                    current_holding,
                    False,
                    False,
                    False,
                    "primitive_compile_error",
                    turn_index,
                    steps_executed,
                )

            tau_satisfied = None
            protocol_turn = None
            if self.file_repl_bridge is not None:
                dispatched, protocol_turn = self.file_repl_bridge.dispatch(
                    invocation,
                    current_observation,
                    max_steps=step_budget,
                    turn=turn_index,
                )
                if action == "vla_act":
                    vla_result = dispatched
                    execution = vla_result.execution
                    tau_satisfied = bool(vla_result.tau_satisfied)
                    current_holding = vla_result.holding
                else:
                    execution = dispatched
            elif action == "vla_act":
                vla_result = self.vla_executor(
                    invocation, current_observation, max_steps=step_budget
                )
                execution = vla_result.execution
                tau_satisfied = bool(vla_result.tau_satisfied)
                current_holding = vla_result.holding
            elif action == "move_to":
                execution = self.move_executor(
                    invocation, current_observation, max_steps=step_budget
                )
            else:
                execution = self.release_executor(
                    invocation, current_observation, max_steps=step_budget
                )

            self._validate_execution(execution, step_budget)
            current_observation = execution.observation
            steps_executed += execution.steps_executed
            env_done = execution.termination_reason == "env_done"
            if execution.termination_reason == "grasp_lost":
                current_holding = None
            if action == "release" and execution.primitive_success:
                current_holding = None

            recoverable = bool(
                not execution.task_success
                and not env_done
                and execution.termination_reason
                == "release_completed_task_incomplete"
            )
            feedback = self._feedback(
                action,
                execution.primitive_success,
                execution.task_success,
                env_done,
                execution.termination_reason,
                execution.steps_executed,
                current_holding,
                recoverable,
                tau_satisfied=tau_satisfied,
            )
            for semantic_field in ("target", "mode"):
                if semantic_field in invocation:
                    feedback[semantic_field] = invocation[semantic_field]
            self._record(
                turn_index,
                raw_output,
                planner_thinking,
                invocation,
                feedback,
                execution.trace,
                labels,
                roles,
                protocol_metadata,
                protocol_turn,
            )
            primitive_success = execution.primitive_success
            last_action = action
            last_feedback = feedback

            if execution.task_success:
                return self._result(
                    current_observation,
                    current_holding,
                    primitive_success,
                    True,
                    env_done,
                    "task_success",
                    turn_index,
                    steps_executed,
                )
            if env_done:
                return self._result(
                    current_observation,
                    current_holding,
                    primitive_success,
                    False,
                    True,
                    "env_done",
                    turn_index,
                    steps_executed,
                )
            if steps_executed >= budgets.horizon:
                return self._result(
                    current_observation,
                    current_holding,
                    primitive_success,
                    False,
                    False,
                    "horizon_exhausted",
                    turn_index,
                    steps_executed,
                )

        return self._result(
            current_observation,
            current_holding,
            primitive_success,
            False,
            False,
            "max_turns_exhausted",
            budgets.max_turns,
            steps_executed,
        )

    @staticmethod
    def _guard_invocation(
        invocation: Dict[str, object], holding: Optional[str]
    ) -> Optional[str]:
        action = invocation.get("action")
        if action not in ("vla_act", "move_to", "release"):
            return "primitive_compile_error"
        if action == "release" and holding is None:
            return "release_without_holding"
        if action == "move_to" and holding is None:
            return "grasp_lost"
        if action == "move_to" and invocation.get("gripper") != "close":
            return "grasp_lost"
        if action == "vla_act" and holding not in (None, invocation.get("target")):
            return "holding_incompatible"
        return None

    @staticmethod
    def _step_budget(
        action: str,
        actions_remaining: int,
        budgets: LiberoMultiTurnBudgets,
    ) -> int:
        if action == "move_to":
            return min(actions_remaining, budgets.max_move_steps)
        if action == "release":
            return min(actions_remaining, budgets.release_steps)
        return actions_remaining

    @staticmethod
    def _validate_execution(
        execution: LiberoPrimitiveExecution, step_budget: int
    ) -> None:
        steps = execution.steps_executed
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 0
            or steps > step_budget
        ):
            raise ValueError("executor exceeded its step budget")

    @staticmethod
    def _feedback(
        action,
        primitive_success: bool,
        task_success: bool,
        env_done: bool,
        termination_reason: str,
        steps_executed: int,
        holding: Optional[str],
        recoverable: bool,
        *,
        tau_satisfied: Optional[bool] = None,
    ) -> Dict[str, Any]:
        feedback = {
            "action": action,
            "primitive_success": bool(primitive_success),
            "task_success": bool(task_success),
            "env_done": bool(env_done),
            "termination_reason": termination_reason,
            "steps_executed": steps_executed,
            "holding": holding,
            "recoverable": bool(recoverable),
        }
        if tau_satisfied is not None:
            feedback["tau_satisfied"] = bool(tau_satisfied)
        return feedback

    def _record(
        self,
        turn: int,
        raw_output: str,
        planner_thinking,
        invocation,
        feedback: Dict[str, Any],
        primitive_trace,
        object_labels: Dict[str, str],
        object_roles: Dict[str, Sequence[str]],
        protocol_metadata: Optional[Dict[str, Any]] = None,
        protocol_turn: Optional[Dict[str, Any]] = None,
    ) -> None:
        primitive = invocation.get("action") if isinstance(invocation, dict) else None
        append_jsonl_record(
            self.trace_path,
            {
                "turn": turn,
                "planner_raw_output": raw_output,
                "planner_thinking": planner_thinking,
                "invocation": invocation,
                "feedback": feedback,
                "primitive_trace": primitive_trace,
                "primitive": primitive,
                "primitive_postcondition_met": feedback["primitive_success"],
                "is_contact": primitive == "vla_act",
                "status": feedback["termination_reason"],
                "termination_reason": feedback["termination_reason"],
                "task_success": feedback["task_success"],
                "episode_status": "completed",
                "object_labels": object_labels,
                "object_roles": object_roles,
                "step_results": primitive_trace,
                "protocol": {
                    "metadata": dict(protocol_metadata or {}),
                    "turn": protocol_turn,
                },
            },
        )

    @staticmethod
    def _result(
        observation,
        holding,
        primitive_success,
        task_success,
        env_done,
        termination_reason,
        turns_executed,
        steps_executed,
    ) -> LiberoMultiTurnResult:
        return LiberoMultiTurnResult(
            observation=observation,
            holding=holding,
            primitive_success=bool(primitive_success),
            task_success=bool(task_success),
            env_done=bool(env_done),
            termination_reason=termination_reason,
            turns_executed=turns_executed,
            steps_executed=steps_executed,
        )


__all__ = [
    "LiberoMultiTurnBudgets",
    "LiberoMultiTurnEvaluator",
    "LiberoMultiTurnResult",
    "LiberoVLAExecution",
]