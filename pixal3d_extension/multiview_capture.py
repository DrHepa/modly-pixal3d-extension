"""Adapt a calibrated capture to TencentARC's ordered, posed MV view bundle.

Capture supplies media, not estimated cameras. Calibration is explicit metadata;
this adapter never fabricates poses or starts a separate estimation model.
"""

from __future__ import annotations

import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .scene_prepare_contract import load_capture_manifest, validate_workspace_output_parent


def _cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Pixal3D MV generation cancelled")


def _number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"MV {label} must be finite numeric calibration")
    return float(value)


def _calibration(capture: dict, count: int) -> tuple[list[dict], float]:
    metadata = capture.get("multiview")
    if not isinstance(metadata, dict) or metadata.get("cameraConvention") != "blender-c2w":
        raise ValueError(
            "Pixal3D MV requires calibrated capture metadata: multiview.cameraConvention "
            "must be blender-c2w and multiview.cameras must provide real poses and FOVs. "
            "Raw captures need camera calibration; this node does not estimate poses."
        )
    cameras = metadata.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != count:
        raise ValueError("MV calibration requires one ordered camera per capture frame")
    mesh_scale = _number(metadata.get("meshScale", 1.0), "meshScale")
    if mesh_scale <= 0:
        raise ValueError("MV meshScale must be positive")
    validated = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict) or type(camera.get("index")) is not int or camera["index"] != index:
            raise ValueError("MV camera indices must match capture frame order")
        matrix = camera.get("transformMatrix")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
            raise ValueError(f"MV camera {index} requires a 4x4 transformMatrix")
        matrix = [[_number(value, f"camera {index} transformMatrix") for value in row] for row in matrix]
        if any(abs(value - expected) > 1e-6 for value, expected in zip(matrix[3], (0, 0, 0, 1))):
            raise ValueError(f"MV camera {index} requires an affine camera-to-world matrix")
        rotation = [row[:3] for row in matrix[:3]]
        for row in range(3):
            for column in range(3):
                dot = sum(rotation[row][k] * rotation[column][k] for k in range(3))
                if abs(dot - (1 if row == column else 0)) > 1e-4:
                    raise ValueError(f"MV camera {index} rotation must be orthonormal")
        determinant = sum(rotation[0][i] * (
            rotation[1][(i + 1) % 3] * rotation[2][(i + 2) % 3]
            - rotation[1][(i + 2) % 3] * rotation[2][(i + 1) % 3]
        ) for i in range(3))
        if abs(determinant - 1) > 1e-4 or sum(row[3] ** 2 for row in matrix[:3]) <= 1e-12:
            raise ValueError(f"MV camera {index} requires a proper rotation and nonzero camera distance")
        fov = _number(camera.get("cameraAngleX"), f"camera {index} cameraAngleX")
        if not 0 < fov < math.pi:
            raise ValueError(f"MV camera {index} cameraAngleX must be in (0, pi) radians")
        validated.append({"file_path": f"{index:04d}.png", "transform_matrix": matrix, "camera_angle_x": fov})
    return validated, mesh_scale


def validate_mv_capture(manifest_path: Path, workspace_dir: Path, num_views: int) -> tuple[dict, Path, list[dict], float]:
    """Validate the complete capture and calibration before writing any output."""
    if type(num_views) is not int or not 1 <= num_views <= 16:
        raise ValueError("num_views must be between 1 and 16")
    capture, root = load_capture_manifest(Path(manifest_path), Path(workspace_dir))
    if capture["kind"] == "frames":
        if capture.get("video") is not None:
            raise ValueError("Frame capture must not also declare video")
        count = len(capture["frames"])
    else:
        if capture.get("frames") not in (None, []):
            raise ValueError("Video capture must not also declare frames")
        count = capture["video"]["frameCount"]
    cameras, mesh_scale = _calibration(capture, count)
    if num_views > count:
        raise ValueError(f"num_views {num_views} exceeds the capture frame count {count}")
    return capture, root, cameras[:num_views], mesh_scale


@contextmanager
def prepare_capture_views(manifest_path: Path, workspace_dir: Path, output_dir: Path, num_views: int, *, cancel_event=None):
    """Yield a disposable upstream view directory; leave source media untouched."""
    _cancelled(cancel_event)
    capture, root, cameras, mesh_scale = validate_mv_capture(manifest_path, workspace_dir, num_views)
    output = validate_workspace_output_parent(output_dir, workspace_dir, "MV output directory")
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pixal3d-mv-views-", dir=output) as temporary:
        staged = Path(temporary)
        if capture["kind"] == "frames":
            for index, frame in enumerate(capture["frames"][:num_views]):
                _cancelled(cancel_event)
                with Image.open(root / frame["path"].replace("\\", "/")) as image:
                    if image.format not in {"PNG", "JPEG"} or image.size != (frame["width"], frame["height"]):
                        raise ValueError(f"Capture frame {index} dimensions or encoding changed")
                    mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                    image.convert(mode).save(staged / cameras[index]["file_path"])
        else:
            import cv2

            video = capture["video"]
            reader = cv2.VideoCapture(str(root / video["path"].replace("\\", "/")))
            if not reader.isOpened():
                reader.release()
                raise ValueError("Capture video cannot be decoded")
            count = 0
            try:
                while True:
                    _cancelled(cancel_event)
                    ok, image = reader.read()
                    if not ok:
                        break
                    if image.shape[:2] != (video["height"], video["width"]):
                        raise ValueError("Capture video decoded dimensions differ from its manifest")
                    if count < num_views:
                        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(staged / cameras[count]["file_path"])
                    count += 1
            finally:
                reader.release()
            if count != video["frameCount"]:
                raise ValueError("Capture video decoded frame count differs from its manifest")
        _cancelled(cancel_event)
        (staged / "transforms.json").write_text(json.dumps({"mesh_scale": mesh_scale, "frames": cameras}), encoding="utf-8")
        yield staged
