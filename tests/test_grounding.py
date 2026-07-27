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
from embodiedbench.envs.eb_manipulation.rgbd_grounding import (
    RGBDCalibration,
    depth_to_meters,
    make_provenance,
    pixel_depth_to_world,
    representative_mask_pixel,
    validate_rgbd_shapes,
    world_to_pixel,
    compute_oracle_metrics,
    summarize_oracle_frames,
)


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
    monkeypatch.setattr(
        eb_man_utils,
        "form_harness_grounding_artifact_for_input",
        lambda obs, task, cameras: _legacy_artifact(task, visible, object_informations, masks, clouds),
    )
    return eb_man_utils.form_harness_grounding_for_input(
        {"object_informations": object_informations}, task_class, ["front_rgb"]
    )


def _legacy_artifact(task_class, visible, object_informations, masks, clouds):
    task_handler = eb_man_utils.TASK_HANDLERS[task_class]()
    known = list(task_handler.sim_name_to_real_name)
    id_to_sim_name = {f"object {index + 1}": name for index, name in enumerate(known)}
    coords = {}
    for sim_name, point in visible.items():
        object_id = next(key for key, name in id_to_sim_name.items() if name == sim_name)
        coords[object_id] = list(eb_man_utils.point_to_voxel_index(np.asarray(point)))
    return {
        "planner_coords": coords,
        "roles": {
            object_id: eb_man_utils._harness_roles(task_class, sim_name)
            for object_id, sim_name in id_to_sim_name.items()
        },
        "labels": {
            object_id: task_handler.sim_name_to_real_name[sim_name]
            for object_id, sim_name in id_to_sim_name.items()
        },
        "id_to_sim_name": id_to_sim_name,
    }


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


def test_depth_to_meters_supports_normalized_and_metric_depth():
    normalized = np.array([[0.0, 0.5, 1.0]])
    assert np.allclose(depth_to_meters(normalized, 0.2, 2.2), [[0.2, 1.2, 2.2]])
    assert np.allclose(depth_to_meters(normalized, 0.2, 2.2, True), normalized)


def test_pixel_depth_world_round_trip_with_translated_camera():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1.0, -2.0, 0.5]
    pixel = [60.0, 20.0]
    world = pixel_depth_to_world(pixel, 2.0, intrinsics, camera_to_world)
    assert np.allclose(world, [1.2, -2.4, 2.5])
    projected, depth = world_to_pixel(world, intrinsics, camera_to_world)
    assert np.allclose(projected, pixel)
    assert depth == 2.0


def test_rgbd_shape_validation_rejects_unsynchronized_resolution():
    validate_rgbd_shapes(np.zeros((4, 5, 3)), np.zeros((4, 5)))
    try:
        validate_rgbd_shapes(np.zeros((4, 5, 3)), np.zeros((5, 4)))
    except ValueError as exc:
        assert "resolutions" in str(exc)
    else:
        raise AssertionError("mismatched RGB-D shapes must be rejected")


def test_representative_pixel_always_belongs_to_non_convex_mask():
    pixels = np.array([[0, 0], [0, 2], [2, 0]])
    row, column = representative_mask_pixel(pixels)
    assert [row, column] in pixels.tolist()


def test_grounding_provenance_marks_transitional_sim_mask():
    calibration = RGBDCalibration(
        camera="front",
        frame_id=7,
        intrinsics=np.eye(3),
        camera_to_world=np.eye(4),
        near=0.1,
        far=3.0,
    )
    provenance = make_provenance(calibration, [12, 34], 0.8)
    assert provenance["frame_id"] == 7
    assert provenance["camera"] == "front"
    assert provenance["pixel_selection"] == "sim_mask"
    assert provenance["privileged_segmentation"] is True
    assert "pose" not in provenance


def test_oracle_metrics_are_separate_from_planner_coordinates():
    artifact = {
        "frame_id": 3,
        "planner_coords": {"object 1": [10, 20, 30]},
        "objects": {
            "object 1": {
                "sim_name": "cube0",
                "world_xyz": [0.1, 0.2, 0.3],
            }
        },
    }
    before = dict(artifact["planner_coords"])
    metrics = compute_oracle_metrics(
        artifact, {"cube0": {"pose": [0.1, 0.2, 0.4, 0, 0, 0, 1]}}
    )
    assert np.isclose(metrics["mean_error_m"], 0.1)
    assert artifact["planner_coords"] == before
    assert "oracle_world_xyz" not in artifact["objects"]["object 1"]


def test_oracle_metric_summary_aggregates_object_observations():
    summary = summarize_oracle_frames([
        {"frame_id": 1, "objects": [{"surface_to_origin_error_m": 0.01}]},
        {"frame_id": 2, "objects": [
            {"surface_to_origin_error_m": 0.02},
            {"surface_to_origin_error_m": 0.03},
        ]},
        {"frame_id": 2, "objects": [{"surface_to_origin_error_m": 1.0}]},
    ])
    assert summary["frame_count"] == 2
    assert summary["object_observation_count"] == 3
    assert np.isclose(summary["mean_error_m"], 0.02)