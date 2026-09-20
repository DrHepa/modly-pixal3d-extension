"""Isolated clean-room SAM3 + DA3 Base capture-to-scene worker.

This file uses only documented official APIs. It contains no community
WorldSculpt node code and never downloads model weights at runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np

from pixal3d_extension.scene_geometry import (
    deduplicate_labeled_aabbs,
    percentile_aabb,
    reprojection_coverage,
    scale_intrinsics,
    unproject_masked_points,
    w2c_opencv_to_c2w_blender,
)
from pixal3d_extension.scene_prepare_contract import load_capture_manifest
from pixal3d_extension.worldsculpt_contract import validate_scene

SAM3_SOURCE_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM3_WEIGHT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_WEIGHT_REVISION = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"


def emit(message: dict) -> None:
    print(json.dumps(message, sort_keys=True), flush=True)


def _integer(params: dict, name: str, default: int, low: int, high: int) -> int:
    value = params.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not low <= result <= high:
        raise ValueError(f"{name} must be within {low}..{high}")
    return result


def _number(params: dict, name: str, default: float, low: float, high: float) -> float:
    value = params.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be within {low}..{high}")
    return result


def _parameters(raw: object) -> dict:
    params = raw if isinstance(raw, dict) else {}
    labels_value = params.get("object_labels", "chair,table,sofa,bed,cabinet,lamp,plant,appliance")
    if not isinstance(labels_value, str):
        raise ValueError("object_labels must be comma-separated text")
    labels = tuple(dict.fromkeys(label.strip() for label in labels_value.split(",") if label.strip()))
    if not labels or len(labels) > 16 or any(len(label) > 80 for label in labels):
        raise ValueError("object_labels requires 1..16 concise labels")
    process_resolution = _integer(params, "process_resolution", 504, 280, 1008)
    if process_resolution not in {392, 504, 630}:
        raise ValueError("process_resolution must be one of 392, 504, or 630")
    result = {
        "labels": labels,
        "max_frames": _integer(params, "max_frames", 16, 2, 64),
        "frame_stride": _integer(params, "frame_stride", 1, 1, 120),
        "process_resolution": process_resolution,
        "mask_erosion": _integer(params, "mask_erosion", 1, 0, 8),
        "minimum_geometry_points": _integer(params, "minimum_geometry_points", 128, 16, 100000),
        "confidence_percentile": _number(params, "confidence_percentile", 40, 0, 100),
        "depth_trim_low": _number(params, "depth_trim_low", 5, 0, 49),
        "depth_trim_high": _number(params, "depth_trim_high", 95, 51, 100),
        "aabb_percentile_low": _number(params, "aabb_percentile_low", 2, 0, 25),
        "aabb_percentile_high": _number(params, "aabb_percentile_high", 98, 75, 100),
        "min_reprojection_coverage": _number(params, "min_reprojection_coverage", 0.1, 0, 1),
        "dedup_iou_threshold": _number(params, "dedup_iou_threshold", 0.5, 0, 1),
        "camera_consistency_tolerance": _number(params, "camera_consistency_tolerance", 0.15, 0.01, 0.5),
        "sam_score_threshold": _number(params, "sam_score_threshold", 0.5, 0, 1),
    }
    if result["depth_trim_low"] >= result["depth_trim_high"] or result["aabb_percentile_low"] >= result["aabb_percentile_high"]:
        raise ValueError("lower percentiles must be below upper percentiles")
    return result


def _safe_file(root: Path, value: str, label: str) -> Path:
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be relative")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"{label} is missing or unsafe")
    return path


def _capture_frames(manifest: dict, capture_root: Path, destination: Path, params: dict) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    images: list[np.ndarray] = []
    if manifest["kind"] == "frames":
        candidates = manifest["frames"][::params["frame_stride"]][:params["max_frames"]]
        for entry in candidates:
            image = cv2.imread(str(_safe_file(capture_root, entry["path"], f"frame {entry['index']}")), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (entry["height"], entry["width"]):
                raise ValueError(f"Capture frame {entry['index']} dimensions or encoding changed")
            images.append(image)
    else:
        video = manifest["video"]
        source = _safe_file(capture_root, video["path"], "capture video")
        reader = cv2.VideoCapture(str(source))
        if not reader.isOpened():
            raise ValueError("Capture video cannot be decoded")
        frame_index = 0
        try:
            while True:
                ok, image = reader.read()
                if not ok:
                    break
                if frame_index % params["frame_stride"] == 0 and len(images) < params["max_frames"]:
                    if image.shape[:2] != (video["height"], video["width"]):
                        raise ValueError("Capture video decoded dimensions differ from its manifest")
                    images.append(image)
                frame_index += 1
        finally:
            reader.release()
        if frame_index != video["frameCount"]:
            raise ValueError("Capture video decoded frame count differs from its manifest")
    if len(images) < 2:
        raise ValueError("Scene estimation requires at least two deterministically selected frames")
    size = images[0].shape[:2]
    if any(image.shape[:2] != size for image in images):
        raise ValueError("Scene estimation requires consistent frame dimensions")
    paths = []
    for index, image in enumerate(images):
        path = destination / f"{index:04d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write normalized frame {index}")
        paths.append(path)
    return paths


def _as_w2c(extrinsics) -> np.ndarray:
    values = np.asarray(extrinsics, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError("DA3 must return one 3x4 or 4x4 w2c matrix per frame")
    if values.shape[1:] == (3, 4):
        padded = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
        padded[:, :3, :4] = values
        values = padded
    for matrix in values:
        w2c_opencv_to_c2w_blender(matrix)
    return values


def _run_da3(frame_paths: list[Path], da3_root: Path, params: dict):
    from pixal3d_extension.da3_official_adapter import load_depth_anything3

    DepthAnything3 = load_depth_anything3()
    model = DepthAnything3.from_pretrained(str(da3_root), local_files_only=True).to("cuda")
    prediction = model.inference(
        image=[str(path) for path in frame_paths],
        process_res=params["process_resolution"],
        process_res_method="upper_bound_resize",
        ref_view_strategy="middle",
        export_format="mini_npz",
    )
    depth = np.asarray(prediction.depth, dtype=np.float64)
    confidence = np.asarray(prediction.conf if prediction.conf is not None else np.ones_like(depth), dtype=np.float64)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    w2c = _as_w2c(prediction.extrinsics)
    if depth.ndim != 3 or confidence.shape != depth.shape or intrinsics.shape != (len(frame_paths), 3, 3) or len(w2c) != len(frame_paths):
        raise ValueError("DA3 returned inconsistent depth/camera shapes")
    return depth, confidence, intrinsics, w2c


def _extract_sam_outputs(outputs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(outputs, dict):
        raise ValueError("SAM3 returned invalid outputs")
    ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []))
    scores = np.asarray(outputs.get("out_scores", outputs.get("out_probs", np.ones(len(ids)))), dtype=np.float64).reshape(-1)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2 and len(ids) == 1:
        masks = masks[None]
    if masks.shape[:1] != ids.shape or masks.ndim != 3:
        raise ValueError("SAM3 object ids and masks are inconsistent")
    if len(scores) != len(ids):
        scores = np.ones(len(ids), dtype=np.float64)
    return ids, masks > 0, scores


def _run_sam3(frame_dir: Path, sam_root: Path, labels: tuple[str, ...], threshold: float) -> list[dict]:
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(checkpoint_path=str(sam_root / "sam3.pt"), gpus_to_use=[0])
    tracked: dict[tuple[str, int], dict] = {}
    try:
        for label in labels:
            started = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
            session_id = started["session_id"]
            try:
                initial = predictor.handle_request({
                    "type": "add_prompt", "session_id": session_id, "frame_index": 0,
                    "text": label, "output_prob_thresh": threshold,
                })
                responses = [initial]
                responses.extend(predictor.handle_stream_request({
                    "type": "propagate_in_video", "session_id": session_id,
                    "propagation_direction": "both", "output_prob_thresh": threshold,
                    "evict_cached_frame_outputs": True,
                }))
                for response in responses:
                    frame_index = int(response["frame_index"])
                    ids, masks, scores = _extract_sam_outputs(response["outputs"])
                    for object_id, mask, score in zip(ids, masks, scores):
                        item = tracked.setdefault((label, int(object_id)), {"label": label, "score": 0.0, "masks": {}})
                        item["score"] = max(item["score"], float(score))
                        item["masks"][frame_index] = mask
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
    finally:
        predictor.shutdown()
    return list(tracked.values())


def _consistent_cameras(intrinsics_source: list[np.ndarray], w2c: np.ndarray, tolerance: float) -> tuple[list[int], np.ndarray]:
    values = np.asarray([[k[0, 0], k[1, 1], k[0, 2], k[1, 2]] for k in intrinsics_source])
    median = np.median(values, axis=0)
    relative = np.abs(values - median) / np.maximum(np.abs(median), 1.0)
    accepted = [index for index in range(len(values)) if np.all(relative[index] <= tolerance)]
    if len(accepted) < 2:
        raise ValueError("DA3 camera intrinsics are not cross-frame consistent")
    # Conversion validates rigid proper axes for every accepted camera.
    for index in accepted:
        w2c_opencv_to_c2w_blender(w2c[index])
    return accepted, np.median(values[accepted], axis=0)


def run(job: dict) -> Path:
    if job.get("schema") != "modly.scene-prep-job.v1":
        raise ValueError("Invalid scene-prep job schema")
    workspace = Path(job["workspace_dir"]).resolve(strict=True)
    output = Path(job["output_dir"]).resolve(strict=True)
    if not output.is_relative_to(workspace):
        raise ValueError("Scene-prep output escapes workspace")
    sam_root = Path(job["sam_root"]).resolve(strict=True)
    da3_root = Path(job["da3_root"]).resolve(strict=True)
    params = _parameters(job.get("params"))
    manifest, capture_root = load_capture_manifest(Path(job["capture_manifest_path"]), workspace)
    staging = Path(tempfile.mkdtemp(prefix=".scene-estimate-", dir=output))
    final = output / f"scene-estimated-{uuid.uuid4().hex}"
    try:
        emit({"type": "progress", "pct": 5, "step": "Selecting deterministic capture frames"})
        input_frames = staging / "_input_frames"
        frame_paths = _capture_frames(manifest, capture_root, input_frames, params)
        source_h, source_w = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR).shape[:2]

        emit({"type": "progress", "pct": 20, "step": "Running official DA3 Base multiview"})
        depth, confidence, intrinsics_processed, w2c = _run_da3(frame_paths, da3_root, params)
        depth_hw = depth.shape[1:]
        intrinsics_source = [scale_intrinsics(k, source_hw=depth_hw, processed_hw=(source_h, source_w)) for k in intrinsics_processed]
        accepted_frames, median_k = _consistent_cameras(intrinsics_source, w2c, params["camera_consistency_tolerance"])

        emit({"type": "progress", "pct": 45, "step": "Running official SAM3 text-guided tracking"})
        tracks = _run_sam3(input_frames, sam_root, params["labels"], params["sam_score_threshold"])
        if not tracks:
            raise ValueError("SAM3 found no tracked instances for the requested labels")

        candidates = []
        for track in tracks:
            frame_points: list[np.ndarray] = []
            accepted_for_instance: list[int] = []
            masks_by_frame: dict[int, np.ndarray] = {}
            for global_index in accepted_frames:
                full_mask = track["masks"].get(global_index)
                points = np.empty((0, 3), dtype=np.float64)
                if full_mask is not None:
                    full_mask = cv2.resize(full_mask.astype(np.uint8), (source_w, source_h), interpolation=cv2.INTER_NEAREST) > 0
                    small_mask = cv2.resize(full_mask.astype(np.uint8), (depth_hw[1], depth_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
                    points = unproject_masked_points(
                        depth=depth[global_index], confidence=confidence[global_index], mask=small_mask,
                        intrinsics=intrinsics_source[global_index], w2c=w2c[global_index],
                        source_hw=(source_h, source_w), processed_hw=depth_hw,
                        erosion_radius=params["mask_erosion"], confidence_percentile=params["confidence_percentile"],
                        depth_percentiles=(params["depth_trim_low"], params["depth_trim_high"]),
                    )
                    coverage = reprojection_coverage(points, small_mask, intrinsics_processed[global_index], w2c[global_index])
                    if len(points) >= params["minimum_geometry_points"] and coverage >= params["min_reprojection_coverage"]:
                        accepted_for_instance.append(len(frame_points))
                        masks_by_frame[global_index] = full_mask
                frame_points.append(points)
            if not accepted_for_instance:
                continue
            box = percentile_aabb(
                frame_points,
                accepted_indices=accepted_for_instance,
                percentiles=(params["aabb_percentile_low"], params["aabb_percentile_high"]),
            )
            if np.max(box[1] - box[0]) <= 1e-4:
                continue
            candidates.append({**track, "aabb_world": box.tolist(), "accepted_masks": masks_by_frame})
        candidates = deduplicate_labeled_aabbs(candidates, iou_threshold=params["dedup_iou_threshold"])
        if not candidates:
            raise ValueError("No SAM3 instance passed DA3 geometry and reprojection checks")

        emit({"type": "progress", "pct": 80, "step": "Publishing canonical relative-scale scene"})
        for output_index, global_index in enumerate(accepted_frames):
            shutil.copyfile(frame_paths[global_index], staging / f"{output_index:04d}.png")
        instances = []
        for pass_index, candidate in enumerate(candidates, start=1):
            obj = f"obj{pass_index:02d}"
            mask_dir = staging / "masks" / obj
            mask_dir.mkdir(parents=True)
            for output_index, global_index in enumerate(accepted_frames):
                mask = candidate["accepted_masks"].get(global_index, np.zeros((source_h, source_w), dtype=bool))
                cv2.imwrite(str(mask_dir / f"{output_index:04d}.png"), mask.astype(np.uint8) * 255)
            instances.append({
                "pass_index": pass_index,
                "label": candidate["label"],
                "score": candidate["score"],
                "aabb_world": candidate["aabb_world"],
            })
        transforms = {
            "fl_x": float(median_k[0]), "fl_y": float(median_k[1]),
            "cx": float(median_k[2]), "cy": float(median_k[3]),
            "w": source_w, "h": source_h,
            "frames": [
                {"file_path": f"{output_index:04d}.png", "transform_matrix": w2c_opencv_to_c2w_blender(w2c[global_index]).tolist()}
                for output_index, global_index in enumerate(accepted_frames)
            ],
            "instances": instances,
            "provenance": {
                "producer": "pixal3d-clean-room-scene-prep-v1",
                "sam3": {"source": "facebookresearch/sam3", "sourceRevision": SAM3_SOURCE_REVISION, "checkpoint": "facebook/sam3", "weightRevision": SAM3_WEIGHT_REVISION, "scale": "n/a"},
                "da3": {"source": "ByteDance-Seed/Depth-Anything-3", "sourceRevision": DA3_SOURCE_REVISION, "checkpoint": "depth-anything/DA3-BASE", "weightRevision": DA3_WEIGHT_REVISION},
                "scale": {"mode": "relative", "metric": False, "reason": "DA3 Base multiview has no metric calibration"},
                "camera": {"inputConvention": "opencv-w2c", "outputConvention": "blender-c2w", "intrinsicsSource": "DA3 processed-resolution K scaled to declared source resolution"},
                "capture": {
                    "manifest": manifest.get("provenance"),
                    "kind": manifest["kind"],
                    "frameStride": params["frame_stride"],
                    "maximumFrames": params["max_frames"],
                    "selectedFrames": len(frame_paths),
                },
            },
        }
        (staging / "transforms.json").write_text(json.dumps(transforms, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "scene-manifest.json").write_text(json.dumps({
            "schema": "modly.scene-manifest.v1", "sceneRoot": ".",
            "assets": [{"path": "transforms.json", "kind": "worldsculpt-transforms"}],
            "provenance": transforms["provenance"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(input_frames)
        validate_scene(staging)
        os.replace(staging, final)
        emit({"type": "progress", "pct": 100, "step": "Scene preparation complete"})
        return final / "scene-manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one scene-prep job JSON path")
        job_path = Path(sys.argv[1]).resolve(strict=True)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        result = run(job)
        emit({"type": "done", "scene_manifest_path": str(result)})
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
