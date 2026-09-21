"""Offline DA3 camera estimation for Pixal3D ordered multi-image input."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from .scene_geometry import scale_intrinsics, w2c_opencv_to_c2w_blender


def _consistent_cameras(intrinsics_source: list[np.ndarray], w2c: np.ndarray, tolerance: float) -> tuple[list[int], np.ndarray]:
    values = np.asarray([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]] for k in intrinsics_source])
    median = np.median(values, axis=0)
    relative = np.abs(values - median) / np.maximum(np.abs(median), 1.0)
    accepted = [index for index in range(len(values)) if np.all(relative[index] <= tolerance)]
    if len(accepted) < 2:
        raise ValueError("DA3 camera intrinsics are not cross-frame consistent")
    for index in accepted:
        w2c_opencv_to_c2w_blender(w2c[index])
    return accepted, np.median(values[accepted], axis=0)


def estimate_transforms(
    intrinsics_processed, w2c, depth, *, source_hw: tuple[int, int], processed_hw: tuple[int, int],
) -> dict:
    """Convert DA3's camera solution into Pixal3D's canonical Blender c2w gauge.

    DA3 determines relative cameras only. A single rigid transform and uniform
    relative-scale normalization place the first estimated camera at Pixal3D's
    canonical front pose; no per-view pose is invented or replaced.
    """

    depth_values = np.asarray(depth, dtype=np.float64)
    camera_values = np.asarray(w2c, dtype=np.float64)
    if depth_values.ndim != 3 or len(depth_values) < 2 or len(depth_values) > 4:
        raise ValueError("DA3 camera estimation requires two to four depth maps")
    if len(camera_values) != len(depth_values):
        raise ValueError("DA3 camera estimation returned inconsistent view counts")
    source_h, source_w = source_hw
    intrinsics_source = [
        scale_intrinsics(k, source_hw=processed_hw, processed_hw=source_hw)
        for k in np.asarray(intrinsics_processed, dtype=np.float64)
    ]
    accepted, _ = _consistent_cameras(intrinsics_source, camera_values, 0.15)
    if accepted != list(range(len(depth_values))):
        raise ValueError("DA3 camera calibration rejected one or more connected images as inconsistent")

    valid_depth = depth_values[0][np.isfinite(depth_values[0]) & (depth_values[0] > 0)]
    if not len(valid_depth):
        raise ValueError("DA3 camera calibration returned no finite positive reference depth")
    reference_depth = float(np.median(valid_depth))
    if not math.isfinite(reference_depth) or reference_depth <= 1e-8:
        raise ValueError("DA3 camera calibration returned an invalid relative scale")

    blender = np.asarray([w2c_opencv_to_c2w_blender(matrix) for matrix in camera_values])
    first = blender[0]
    target = first[:3, 3] - first[:3, 2] * reference_depth
    canonical_rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    world_rotation = canonical_rotation @ first[:3, :3].T
    relative_scale = 3.0 / reference_depth

    frames = []
    for index, (camera, intrinsic) in enumerate(zip(blender, intrinsics_source)):
        transformed = np.eye(4, dtype=np.float64)
        transformed[:3, :3] = world_rotation @ camera[:3, :3]
        transformed[:3, 3] = world_rotation @ (camera[:3, 3] - target) * relative_scale
        if not np.isfinite(transformed).all() or not np.allclose(
            transformed[:3, :3].T @ transformed[:3, :3], np.eye(3), atol=1e-5, rtol=0
        ) or not np.isclose(np.linalg.det(transformed[:3, :3]), 1.0, atol=1e-5, rtol=0):
            raise ValueError(f"DA3 camera {index} is not a finite proper rigid transform")
        focal_x = float(intrinsic[0, 0])
        fov = 2.0 * math.atan(source_w / (2.0 * focal_x))
        if not math.isfinite(fov) or not 0 < fov < math.pi:
            raise ValueError(f"DA3 camera {index} produced an invalid horizontal FOV")
        frames.append({
            "file_path": f"{index:04d}.png",
            "transform_matrix": transformed.tolist(),
            "camera_angle_x": fov,
        })
    return {
        "mesh_scale": 1.0,
        "frames": frames,
        "provenance": {
            "cameraEstimator": "Depth Anything 3 Base",
            "cameraConvention": "blender-c2w",
            "gauge": "first-view canonical rigid transform plus uniform DA3-relative scale",
        },
    }


def run(job: dict) -> Path:
    import cv2
    from .scene_prepare_worker import _run_da3

    if not isinstance(job, dict) or job.get("schema") != "modly.pixal3d-mv-camera-job.v1":
        raise ValueError("Invalid Pixal3D MV camera job schema")
    frame_paths = [Path(value).resolve(strict=True) for value in job.get("frame_paths", [])]
    if not 2 <= len(frame_paths) <= 4 or len(set(frame_paths)) != len(frame_paths):
        raise ValueError("Pixal3D MV camera job requires two to four unique ordered frames")
    root = frame_paths[0].parent
    if any(path.parent != root or path.is_symlink() or not path.is_file() for path in frame_paths):
        raise ValueError("Pixal3D MV camera frames must be regular files in one private directory")
    dimensions = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim not in (2, 3):
            raise ValueError(f"Pixal3D MV camera frame cannot be decoded: {path.name}")
        dimensions.append(image.shape[:2])
    if len(set(dimensions)) != 1:
        raise ValueError("Pixal3D MV requires all connected images to have matching dimensions")
    da3_root = Path(job["da3_root"]).resolve(strict=True)
    params = {
        "process_resolution": int(job.get("process_resolution", 504)),
        "ref_view_strategy": "saddle_balanced",
    }
    depth, _confidence, intrinsics, w2c = _run_da3(frame_paths, da3_root, params)
    transforms = estimate_transforms(
        intrinsics, w2c, depth, source_hw=dimensions[0], processed_hw=depth.shape[1:],
    )
    output = Path(job["output_path"])
    if output.parent.resolve(strict=True) != root or output.name != "transforms.json" or output.is_symlink():
        raise ValueError("Pixal3D MV transforms output must remain in the private frame directory")
    output.write_text(json.dumps(transforms, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one Pixal3D MV camera job JSON path")
        job_path = Path(sys.argv[1]).resolve(strict=True)
        result = run(json.loads(job_path.read_text(encoding="utf-8")))
        print(json.dumps({"type": "done", "transforms_path": str(result)}), flush=True)
    except Exception as exc:
        print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
