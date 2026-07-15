"""Unit tests for the fixed primitive library (no simulator required)."""

import pytest

from embodiedbench.planner.harness.primitives import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PoseState,
    PrimitiveError,
    PrimitiveLibrary,
    VOXEL_SIZE,
)


@pytest.fixture
def lib():
    return PrimitiveLibrary(approach_dz=8, lift_dz=6)


@pytest.fixture
def coords():
    return {"object 1": [50, 60, 20], "object 2": [30, 40, 25]}


def test_primitive_set_is_fixed():
    assert PrimitiveLibrary.CONTACT_PRIMITIVE == "vla_act"
    assert set(PrimitiveLibrary.PRIMITIVES) == {
        "move_to",
        "rotate_wrist",
        "rotate_pitch",
        "set_gripper",
        "release",
        "vla_act",
    }


def test_move_to_explicit_xyz(lib):
    pose = PoseState(x=10, y=10, z=10, gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "move_to", "xyz": [50, 60, 20]}, pose)
    assert res.name == "move_to"
    assert res.actions == [[50, 60, 20, pose.roll, pose.pitch, pose.yaw, GRIPPER_OPEN]]
    assert not res.is_contact


def test_move_to_object_name(lib, coords):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "target": "object 2"}, pose, coords)
    assert res.actions[0][:3] == [30, 40, 25]


def test_move_to_tolerant_name(lib, coords):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "target": "Object2"}, pose, coords)
    assert res.actions[0][:3] == [30, 40, 25]


def test_move_to_gripper_override(lib):
    pose = PoseState(gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "move_to", "xyz": [1, 2, 3], "gripper": "close"}, pose)
    assert res.actions[0][6] == GRIPPER_CLOSED


def test_rotate_wrist_holds_position(lib):
    pose = PoseState(x=5, y=6, z=7, yaw=10)
    res = lib.compile({"action": "rotate_wrist", "target_yaw": 40}, pose)
    assert res.actions[0][:3] == [5, 6, 7]
    assert res.actions[0][5] == 40


def test_rotate_pitch_holds_position(lib):
    pose = PoseState(x=5, y=6, z=7, pitch=10)
    res = lib.compile({"action": "rotate_pitch", "target_pitch": 33}, pose)
    assert res.actions[0][4] == 33


def test_set_gripper(lib):
    pose = PoseState(gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "set_gripper", "gripper": "close"}, pose)
    assert res.actions[0][6] == GRIPPER_CLOSED


def test_release_opens_gripper(lib):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    res = lib.compile({"action": "release"}, pose)
    assert res.actions[-1][6] == GRIPPER_OPEN


def test_release_with_lift(lib):
    pose = PoseState(x=1, y=2, z=50, gripper=GRIPPER_CLOSED)
    res = lib.compile({"action": "release", "lift": True}, pose)
    assert len(res.actions) == 2
    assert res.actions[1][2] == 56  # lifted by lift_dz


def test_vla_act_grasp_sequence(lib, coords):
    pose = PoseState(x=0, y=0, z=0, gripper=GRIPPER_OPEN)
    res = lib.compile({"action": "vla_act", "target": "object 1", "mode": "grasp"}, pose, coords)
    assert res.is_contact
    assert res.meta["mode"] == "grasp"
    # 4 sub-actions: approach(open), descend(open), close, lift
    assert len(res.actions) == 4
    approach, descend, close, lift = res.actions
    assert approach[2] == 20 + 8 and approach[6] == GRIPPER_OPEN
    assert descend[2] == 20 and descend[6] == GRIPPER_OPEN
    assert close[6] == GRIPPER_CLOSED
    assert lift[2] == 20 + 6 and lift[6] == GRIPPER_CLOSED


def test_vla_act_place_sequence(lib, coords):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    res = lib.compile({"action": "vla_act", "target": "object 2", "mode": "place"}, pose, coords)
    assert res.meta["mode"] == "place"
    assert len(res.actions) == 3
    assert res.actions[-1][6] == GRIPPER_OPEN  # release at the end


def test_vla_act_prompt_implies_place(lib, coords):
    pose = PoseState(gripper=GRIPPER_CLOSED)
    res = lib.compile(
        {"action": "vla_act", "target": "object 1", "prompt": "place the cube in the box"},
        pose,
        coords,
    )
    assert res.meta["mode"] == "place"


def test_vla_act_push(lib, coords):
    pose = PoseState()
    res = lib.compile(
        {"action": "vla_act", "target": "object 1", "mode": "push", "direction": [5, 0, 0]},
        pose,
        coords,
    )
    assert res.meta["mode"] == "push"
    assert res.actions[-1][0] == 55


def test_unknown_primitive_raises(lib):
    with pytest.raises(PrimitiveError):
        lib.compile({"action": "teleport", "xyz": [1, 2, 3]}, PoseState())


def test_missing_action_raises(lib):
    with pytest.raises(PrimitiveError):
        lib.compile({"xyz": [1, 2, 3]}, PoseState())


def test_unknown_target_raises(lib, coords):
    with pytest.raises(PrimitiveError):
        lib.compile({"action": "move_to", "target": "object 9"}, PoseState(), coords)


def test_clamping_within_bounds(lib):
    pose = PoseState()
    res = lib.compile({"action": "move_to", "xyz": [999, -5, 50]}, pose)
    assert res.actions[0][0] == VOXEL_SIZE
    assert res.actions[0][1] == 0


def test_actions_are_plain_ints(lib, coords):
    res = lib.compile({"action": "vla_act", "target": "object 1"}, PoseState(), coords)
    for action in res.actions:
        assert all(isinstance(v, int) for v in action)
        assert len(action) == 7
