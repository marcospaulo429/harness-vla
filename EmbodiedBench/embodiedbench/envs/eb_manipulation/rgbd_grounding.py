"""Auditable RGB-D geometry helpers for Harness grounding."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class RGBDCalibration:
    camera: str
    frame_id: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    near: float
    far: float
    depth_in_meters: bool = False


def depth_to_meters(depth, near, far, depth_in_meters=False):
    """Return metric depth without mutating the input array."""
    depth_array = np.asarray(depth, dtype=float)
    if depth_in_meters:
        return depth_array.copy()
    if far <= near:
        raise ValueError("far clipping plane must be greater than near clipping plane")
    return near + depth_array * (far - near)


def pixel_depth_to_world(pixel_uv, depth_m, intrinsics, camera_to_world):
    """Project an image pixel with metric z-depth into world coordinates."""
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth_m must be a positive finite value")
    calibration = np.asarray(intrinsics, dtype=float)
    transform = np.asarray(camera_to_world, dtype=float)
    if calibration.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    if transform.shape != (4, 4):
        raise ValueError("camera_to_world must have shape (4, 4)")
    u, v = (float(value) for value in pixel_uv)
    ray = np.linalg.solve(calibration, np.array([u, v, 1.0]))
    camera_point = ray * float(depth_m)
    world_point = transform @ np.append(camera_point, 1.0)
    return world_point[:3]


def world_to_pixel(world_xyz, intrinsics, camera_to_world):
    """Project a world coordinate into pixel coordinates and metric z-depth."""
    calibration = np.asarray(intrinsics, dtype=float)
    transform = np.asarray(camera_to_world, dtype=float)
    if calibration.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("invalid calibration shapes")
    camera_point = np.linalg.inv(transform) @ np.append(
        np.asarray(world_xyz, dtype=float), 1.0
    )
    if camera_point[2] <= 0:
        raise ValueError("world point is not in front of the camera")
    homogeneous_pixel = calibration @ camera_point[:3]
    return homogeneous_pixel[:2] / homogeneous_pixel[2], float(camera_point[2])


def validate_rgbd_shapes(rgb, depth):
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if depth_array.shape != rgb_array.shape[:2]:
        raise ValueError("rgb and depth resolutions must match")


def representative_mask_pixel(matching_pixels):
    """Return the mask pixel nearest its median, including non-convex masks."""
    pixels = np.asarray(matching_pixels, dtype=int)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels):
        raise ValueError("matching_pixels must have shape (n, 2) with n > 0")
    median = np.median(pixels, axis=0)
    distances = np.sum((pixels - median) ** 2, axis=1)
    row, column = pixels[int(np.argmin(distances))]
    return int(row), int(column)


def make_provenance(calibration: RGBDCalibration, pixel_uv, depth_m):
    """Build JSON-safe provenance for one RGB-D estimate."""
    return {
        "version": 1,
        "method": "pinhole_pixel_depth_to_world",
        "camera": calibration.camera,
        "frame_id": int(calibration.frame_id),
        "pixel_uv": [float(pixel_uv[0]), float(pixel_uv[1])],
        "depth_m": float(depth_m),
        "depth_encoding": "meters" if calibration.depth_in_meters else "normalized_linear",
        "calibration_source": "observation.misc",
        "pixel_selection": "sim_mask",
        "privileged_segmentation": True,
    }


def calibration_from_observation(obs: Dict[str, Any], camera: str, frame_id: int):
    misc = obs["misc"]
    return RGBDCalibration(
        camera=camera,
        frame_id=frame_id,
        intrinsics=np.asarray(misc[f"{camera}_camera_intrinsics"], dtype=float),
        camera_to_world=np.asarray(misc[f"{camera}_camera_extrinsics"], dtype=float),
        near=float(misc[f"{camera}_camera_near"]),
        far=float(misc[f"{camera}_camera_far"]),
        depth_in_meters=bool(misc.get(f"{camera}_camera_depth_in_meters", False)),
    )


def compute_oracle_metrics(artifact, object_informations):
    """Compare RGB-D surface estimates with simulator poses for evaluation only."""
    records = []
    skipped = []
    for object_id, estimate in artifact.get("objects", {}).items():
        sim_name = estimate.get("sim_name")
        oracle = object_informations.get(sim_name, {})
        pose = oracle.get("pose")
        if pose is None or len(pose) < 3:
            skipped.append(object_id)
            continue
        estimate_xyz = np.asarray(estimate["world_xyz"], dtype=float)
        oracle_xyz = np.asarray(pose[:3], dtype=float)
        error_m = float(np.linalg.norm(estimate_xyz - oracle_xyz))
        records.append({
            "object_id": object_id,
            "sim_name": sim_name,
            "estimate_world_xyz": estimate_xyz.tolist(),
            "oracle_world_xyz": oracle_xyz.tolist(),
            "surface_to_origin_error_m": error_m,
        })
    errors = np.asarray(
        [record["surface_to_origin_error_m"] for record in records], dtype=float
    )
    return {
        "frame_id": int(artifact.get("frame_id", 0)),
        "metric_definition": "rgbd_visible_surface_centroid_to_sim_object_origin",
        "valid_object_count": int(len(records)),
        "skipped_object_ids": skipped,
        "mean_error_m": float(np.mean(errors)) if len(errors) else None,
        "median_error_m": float(np.median(errors)) if len(errors) else None,
        "p95_error_m": float(np.percentile(errors, 95)) if len(errors) else None,
        "max_error_m": float(np.max(errors)) if len(errors) else None,
        "objects": records,
    }


def summarize_oracle_frames(frame_metrics):
    """Aggregate object-level RGB-D errors across observation frames."""
    unique_frames = []
    seen_frame_ids = set()
    for index, frame in enumerate(frame_metrics):
        frame_id = frame.get("frame_id")
        key = ("frame", frame_id) if frame_id is not None else ("index", index)
        if key in seen_frame_ids:
            continue
        seen_frame_ids.add(key)
        unique_frames.append(frame)
    errors = [
        record["surface_to_origin_error_m"]
        for frame in unique_frames
        for record in frame.get("objects", [])
    ]
    values = np.asarray(errors, dtype=float)
    return {
        "frame_count": int(len(unique_frames)),
        "object_observation_count": int(len(values)),
        "mean_error_m": float(np.mean(values)) if len(values) else None,
        "median_error_m": float(np.median(values)) if len(values) else None,
        "p95_error_m": float(np.percentile(values, 95)) if len(values) else None,
        "max_error_m": float(np.max(values)) if len(values) else None,
    }