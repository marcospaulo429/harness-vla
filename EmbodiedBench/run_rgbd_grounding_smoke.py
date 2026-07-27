"""Validate RGB-D projection against PyRep point clouds on one observation."""

import copy
import json

import numpy as np

from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
from embodiedbench.envs.eb_manipulation.eb_man_utils import (
    form_harness_grounding_artifact_for_input,
)
from embodiedbench.envs.eb_manipulation.rgbd_grounding import (
    calibration_from_observation,
    depth_to_meters,
    pixel_depth_to_world,
)


def main():
    env = EBManEnv(
        eval_set="base",
        render_mode="rgb_array",
        img_size=(256, 256),
        down_sample_ratio=1.0,
        selected_indexes=[0],
        headless=True,
        log_path="running/rgbd_grounding_smoke",
    )
    try:
        _, observation = env.reset()
        obs = vars(copy.deepcopy(observation))
        artifact = form_harness_grounding_artifact_for_input(
            obs, env.task_class, ["front_rgb"]
        )
        calibration = calibration_from_observation(
            obs, "front", artifact["frame_id"]
        )
        depth_m = depth_to_meters(
            obs["front_depth"], calibration.near, calibration.far,
            calibration.depth_in_meters,
        )
        errors = []
        comparisons = []
        for object_id, estimate in artifact["objects"].items():
            sample = next(
                (item for item in estimate["samples"] if item["camera"] == "front"),
                None,
            )
            if sample is None:
                continue
            column, row = (int(round(value)) for value in sample["pixel_uv"])
            projected = pixel_depth_to_world(
                [column, row], depth_m[row, column], calibration.intrinsics,
                calibration.camera_to_world,
            )
            pyrep_point = np.asarray(obs["front_point_cloud"][row, column], dtype=float)
            error_m = float(np.linalg.norm(projected - pyrep_point))
            errors.append(error_m)
            comparisons.append({
                "object_id": object_id,
                "pixel_uv": [column, row],
                "projection_world_xyz": projected.tolist(),
                "pyrep_world_xyz": pyrep_point.tolist(),
                "error_m": error_m,
            })
        result = {
            "frame_id": artifact["frame_id"],
            "comparison_count": len(comparisons),
            "max_projection_error_m": max(errors) if errors else None,
            "mean_projection_error_m": float(np.mean(errors)) if errors else None,
            "comparisons": comparisons,
        }
        print(json.dumps(result, indent=2))
        if not errors:
            raise RuntimeError("no front-camera object samples were available")
        if max(errors) > 1e-5:
            raise RuntimeError(
                f"RGB-D projection differs from PyRep by {max(errors):.6g} m"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
