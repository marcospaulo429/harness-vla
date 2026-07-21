"""Compatibility tests for grasp attachment introspection without PyRep startup."""

from pathlib import Path


def _load_env_method_source():
    path = (
        Path(__file__).parents[1]
        / "EmbodiedBench/embodiedbench/envs/eb_manipulation/EBManEnv.py"
    )
    source = path.read_text(encoding="utf-8")
    start = source.index("    def get_grasped_object_names(self):")
    end = source.index("\n    def close(", start)
    namespace = {}
    exec("class _Env:\n" + source[start:end], namespace)
    return namespace["_Env"]


def test_grasped_object_names_supports_task_robot_gripper():
    env_type = _load_env_method_source()
    obj = type("Obj", (), {"get_name": lambda self: "cube"})()
    gripper = type("Gripper", (), {"get_grasped_objects": lambda self: [obj]})()
    robot = type("Robot", (), {"gripper": gripper})()
    env = env_type()
    env.task = type("Task", (), {"_robot": robot})()
    env.env = None
    assert env.get_grasped_object_names() == ["cube"]
    assert env.get_grasp_attachment_evidence() == (["cube"], True)


def test_grasped_object_names_keeps_minimal_mocks_compatible():
    env_type = _load_env_method_source()
    env = env_type()
    env.task = object()
    env.env = object()
    assert env.get_grasped_object_names() == []
    assert env.get_grasp_attachment_evidence() == ([], False)