"""Stable harness grounding tests without launching the simulator."""

import sys
import types

import numpy as np

pyrep = types.ModuleType("pyrep")
pyrep_objects = types.ModuleType("pyrep.objects")
pyrep_objects.VisionSensor = type("VisionSensor", (), {})
sys.modules.setdefault("pyrep", pyrep)
sys.modules.setdefault("pyrep.objects", pyrep_objects)
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
scipy = types.ModuleType("scipy")
scipy_spatial = types.ModuleType("scipy.spatial")
scipy_transform = types.ModuleType("scipy.spatial.transform")
scipy_transform.Rotation = type("Rotation", (), {})
sys.modules.setdefault("scipy", scipy)
sys.modules.setdefault("scipy.spatial", scipy_spatial)
sys.modules.setdefault("scipy.spatial.transform", scipy_transform)

from embodiedbench.envs.eb_manipulation import eb_man_utils


def _ground(monkeypatch, task_class, visible):
    masks = {}
    clouds = {}
    mask_id_to_name = {}
    object_informations = {}
    for index, (sim_name, point) in enumerate(visible.items(), 1):
        mask_id_to_name[index] = sim_name
        object_informations[sim_name] = {"id": index}
    for camera in eb_man_utils.CAMERAS:
        mask = np.zeros((1, max(1, len(visible))), dtype=int)
        cloud = np.zeros((1, max(1, len(visible)), 3), dtype=float)
        for column, point in enumerate(visible.values()):
            mask[0, column] = column + 1
            cloud[0, column] = point
        masks[camera] = mask
        clouds[camera] = cloud
    monkeypatch.setattr(eb_man_utils, "_get_point_cloud_dict_for_input", lambda obs, cameras: (clouds, [], []))
    monkeypatch.setattr(eb_man_utils, "_get_mask_dict_for_input", lambda obs: masks)
    return eb_man_utils.form_harness_grounding_for_input(
        {"object_informations": object_informations}, task_class, ["front_rgb"]
    )


def test_grounding_ids_stay_stable_when_positions_swap(monkeypatch):
    first = _ground(monkeypatch, "pick", {
        "star_normal_visual0": [0.0, -0.2, 0.8],
        "cube_basic0": [0.0, 0.2, 0.8],
    })
    second = _ground(monkeypatch, "pick", {
        "star_normal_visual0": [0.0, 0.2, 0.8],
        "cube_basic0": [0.0, -0.2, 0.8],
    })
    assert first[3] == second[3]
    assert first[3]["object 3"] == "star_normal_visual0"
    assert first[3]["object 9"] == "cube_basic0"
    assert first[0]["object 3"] != second[0]["object 3"]


def test_grounding_roles_and_known_index(monkeypatch):
    coords, roles, labels, mapping = _ground(
        monkeypatch, "place", {"shape_sorter_visual": [0.0, 0.0, 0.8]}
    )
    assert list(coords) == ["object 21"]
    assert roles["object 21"] == ["destination"]
    assert roles["object 1"] == ["manipulable"]
    assert labels["object 21"] == "shape sorter"
    assert len(mapping) == 21


def test_structural_roles_for_pick_wipe_and_stack(monkeypatch):
    _, pick_roles, _, _ = _ground(monkeypatch, "pick", {})
    _, wipe_roles, _, wipe_mapping = _ground(monkeypatch, "wipe", {})
    _, stack_roles, _, _ = _ground(monkeypatch, "stack", {})
    assert pick_roles["object 1"] == ["destination"]
    sponge_id = next(key for key, value in wipe_mapping.items() if value == "sponge_visual0")
    assert wipe_roles[sponge_id] == ["manipulable"]
    assert wipe_roles["object 1"] == ["destination"]
    assert all(value == ["manipulable", "destination"] for value in stack_roles.values())