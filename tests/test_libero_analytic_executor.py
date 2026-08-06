import numpy as np
import pytest

from embodiedbench.evaluator.libero_analytic_executor import execute_pose_primitive
from embodiedbench.planner.harness.libero_primitives import LiberoPrimitiveError


class _FakeOscEnv:
    def __init__(self, xyz=(0.0, 0.0, 0.5), done_after=None, success_after=None):
        self.xyz = np.asarray(xyz, dtype=float)
        self.done_after = done_after
        self.success_after = success_after
        self.actions = []

    def observation(self):
        return {
            "robot0_eef_pos": self.xyz.copy(),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        }

    def step(self, action):
        self.actions.append(action)
        self.xyz += np.asarray(action[:3]) * 0.05
        done = self.done_after is not None and len(self.actions) >= self.done_after
        return self.observation(), 0.0, done, {"source": "fake"}

    def check_success(self):
        return self.success_after is not None and len(self.actions) >= self.success_after


def test_closed_loop_move_reaches_pose_and_preserves_closed_gripper():
    env = _FakeOscEnv()

    result = execute_pose_primitive(
        env,
        env.observation(),
        [0.12, -0.02, 0.5],
        gripper="close",
        max_steps=4,
        position_tolerance=1e-6,
    )

    assert result.primitive_success is True
    assert result.task_success is False
    assert result.termination_reason == "postcondition_met"
    assert result.steps_executed == 3
    np.testing.assert_allclose(env.xyz, [0.12, -0.02, 0.5])
    assert all(action[6] == 1.0 for action in env.actions)
    assert len(result.trace) == 3


def test_task_success_stops_without_becoming_primitive_success():
    env = _FakeOscEnv(success_after=1)

    result = execute_pose_primitive(
        env,
        env.observation(),
        [0.2, 0.0, 0.5],
        gripper="close",
        max_steps=5,
        position_tolerance=1e-6,
    )

    assert result.primitive_success is False
    assert result.task_success is True
    assert result.termination_reason == "task_success"
    assert result.steps_executed == 1


def test_env_done_is_not_reported_as_task_or_primitive_success():
    env = _FakeOscEnv(done_after=1)

    result = execute_pose_primitive(
        env,
        env.observation(),
        [0.2, 0.0, 0.5],
        gripper="open",
        max_steps=5,
        position_tolerance=1e-6,
    )

    assert result.primitive_success is False
    assert result.task_success is False
    assert result.termination_reason == "env_done"
    assert env.actions[0][6] == -1.0


def test_step_budget_exhaustion_is_explicit():
    env = _FakeOscEnv()

    result = execute_pose_primitive(
        env,
        env.observation(),
        [0.2, 0.0, 0.5],
        gripper="close",
        max_steps=2,
        position_tolerance=1e-6,
    )

    assert result.primitive_success is False
    assert result.task_success is False
    assert result.termination_reason == "step_budget_exhausted"
    assert result.steps_executed == 2


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_invalid_step_budgets_fail_closed(max_steps):
    env = _FakeOscEnv()
    with pytest.raises(LiberoPrimitiveError):
        execute_pose_primitive(
            env,
            env.observation(),
            [0.0, 0.0, 0.5],
            gripper="close",
            max_steps=max_steps,
            position_tolerance=0.01,
        )