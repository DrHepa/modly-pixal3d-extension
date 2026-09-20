"""Clean-room geometry primitives for scene preparation.

The module contains no third-party WorldSculpt community code. Camera inputs use
OpenCV world-to-camera convention; WorldSculpt outputs use Blender camera-to-world.
All scene scale produced by DA3 Base remains explicitly relative.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


_BLENDER_CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def _matrix(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite {shape} matrix")
    return matrix


def scale_intrinsics(intrinsics, *, source_hw: tuple[int, int], processed_hw: tuple[int, int]) -> np.ndarray:
    """Scale K from its declared source resolution to the actual processed resolution."""
    source_h, source_w = source_hw
    processed_h, processed_w = processed_hw
    if min(source_h, source_w, processed_h, processed_w) <= 0:
        raise ValueError("source and processed resolutions must be positive")
    intrinsic = _matrix(intrinsics, (3, 3), "intrinsics").copy()
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError("intrinsics require positive focal lengths and homogeneous final row")
    sx = processed_w / source_w
    sy = processed_h / source_h
    intrinsic[0, 0] *= sx
    intrinsic[0, 1] *= sx
    intrinsic[0, 2] *= sx
    intrinsic[1, 0] *= sy
    intrinsic[1, 1] *= sy
    intrinsic[1, 2] *= sy
    return intrinsic


def w2c_opencv_to_c2w_blender(w2c) -> np.ndarray:
    """Invert an OpenCV w2c matrix and convert camera axes to Blender convention."""
    world_to_camera = _matrix(w2c, (4, 4), "w2c")
    if not np.allclose(world_to_camera[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("w2c must be homogeneous")
    rotation = world_to_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("w2c rotation must be rigid and proper")
    return np.linalg.inv(world_to_camera) @ _BLENDER_CAMERA_AXIS_FLIP


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if radius < 0:
        raise ValueError("erosion radius cannot be negative")
    if radius == 0:
        return result.copy()
    padded = np.pad(result, radius, mode="constant", constant_values=False)
    eroded = np.ones_like(result, dtype=bool)
    size = 2 * radius + 1
    for y in range(size):
        for x in range(size):
            eroded &= padded[y:y + result.shape[0], x:x + result.shape[1]]
    return eroded


def unproject_masked_points(
    *,
    depth,
    confidence,
    mask,
    intrinsics,
    w2c,
    source_hw: tuple[int, int],
    processed_hw: tuple[int, int],
    erosion_radius: int = 1,
    confidence_percentile: float = 40.0,
    depth_percentiles: tuple[float, float] = (5.0, 95.0),
) -> np.ndarray:
    """Robustly unproject accepted masked depth samples into relative world space."""
    depth_map = np.asarray(depth, dtype=np.float64)
    conf_map = np.asarray(confidence, dtype=np.float64)
    binary_mask = _erode(np.asarray(mask, dtype=bool), erosion_radius)
    if depth_map.shape != processed_hw or conf_map.shape != processed_hw or binary_mask.shape != processed_hw:
        raise ValueError("depth, confidence and mask must match processed_hw")
    low, high = depth_percentiles
    if not (0 <= confidence_percentile <= 100 and 0 <= low <= high <= 100):
        raise ValueError("percentiles must be within 0..100")
    valid = binary_mask & np.isfinite(depth_map) & (depth_map > 0) & np.isfinite(conf_map)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    confidence_cut = np.percentile(conf_map[valid], confidence_percentile)
    valid &= conf_map >= confidence_cut
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    depth_low, depth_high = np.percentile(depth_map[valid], [low, high])
    valid &= (depth_map >= depth_low) & (depth_map <= depth_high)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    intrinsic = scale_intrinsics(intrinsics, source_hw=source_hw, processed_hw=processed_hw)
    inv_k = np.linalg.inv(intrinsic)
    pixels = np.stack((xs, ys, np.ones_like(xs)), axis=0).astype(np.float64)
    camera = (inv_k @ pixels) * depth_map[ys, xs]
    world_to_camera = _matrix(w2c, (4, 4), "w2c")
    camera_to_world = np.linalg.inv(world_to_camera)
    homogeneous = np.vstack((camera, np.ones((1, camera.shape[1]))))
    return (camera_to_world @ homogeneous)[:3].T


def percentile_aabb(
    frame_points: Sequence[np.ndarray],
    *,
    accepted_indices: Iterable[int],
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    indices = tuple(dict.fromkeys(int(index) for index in accepted_indices))
    if not indices:
        raise ValueError("at least one accepted frame is required")
    selected = []
    for index in indices:
        if index < 0 or index >= len(frame_points):
            raise ValueError("accepted frame index is out of range")
        points = np.asarray(frame_points[index], dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("accepted frame points must be finite Nx3 arrays")
        if len(points):
            selected.append(points)
    if not selected:
        raise ValueError("accepted frames contain no points")
    lower, upper = percentiles
    if not (0 <= lower <= upper <= 100):
        raise ValueError("AABB percentiles must be within 0..100")
    points = np.concatenate(selected, axis=0)
    return np.stack((np.percentile(points, lower, axis=0), np.percentile(points, upper, axis=0)))


def reprojection_coverage(points_world, mask, intrinsics, w2c) -> float:
    points = np.asarray(points_world, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    intrinsic = _matrix(intrinsics, (3, 3), "intrinsics")
    world_to_camera = _matrix(w2c, (4, 4), "w2c")
    if points.ndim != 2 or points.shape[1:] != (3,) or binary.ndim != 2:
        raise ValueError("points must be Nx3 and mask must be two-dimensional")
    mask_area = int(binary.sum())
    if not len(points) or mask_area == 0:
        return 0.0
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > 1e-8
    camera = camera[valid]
    if not len(camera):
        return 0.0
    projected = (intrinsic @ (camera / camera[:, 2:3]).T).T
    xy = np.rint(projected[:, :2]).astype(np.int64)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < binary.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < binary.shape[0])
    xy = xy[inside]
    if not len(xy):
        return 0.0
    supported = {(int(x), int(y)) for x, y in xy if binary[y, x]}
    return min(1.0, len(supported) / mask_area)


def aabb_iou(first, second) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != (2, 3) or b.shape != (2, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("AABBs must be finite 2x3 arrays")
    a_size = np.maximum(0.0, a[1] - a[0])
    b_size = np.maximum(0.0, b[1] - b[0])
    intersection = np.maximum(0.0, np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]))
    intersection_volume = float(np.prod(intersection))
    union = float(np.prod(a_size) + np.prod(b_size) - intersection_volume)
    return intersection_volume / union if union > 0 else 0.0


def deduplicate_labeled_aabbs(instances: Sequence[dict], *, iou_threshold: float = 0.5) -> list[dict]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be within 0..1")
    ordered = sorted(instances, key=lambda item: (-float(item.get("score", 0)), str(item.get("label", ""))))
    kept: list[dict] = []
    for instance in ordered:
        label = instance.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Instance label is required for label-aware deduplication")
        if any(existing["label"] == label and aabb_iou(existing["aabb_world"], instance["aabb_world"]) >= iou_threshold for existing in kept):
            continue
        kept.append(instance)
    return kept
