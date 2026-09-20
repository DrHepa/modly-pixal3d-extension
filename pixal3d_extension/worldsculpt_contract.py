"""Fail-closed, local contracts for a future WorldSculpt scene runner.

No Modly node invokes this module yet. The upstream scripts can exit successfully
while silently skipping crops, reconstructions, or scene GLB export.
"""

from __future__ import annotations

import json
import math
import numbers
import struct
import zipfile
from pathlib import Path


ADAPTER_DIRS = (
    "ss_ft64_mv_lora_ibr_texverse",
    "shape_ft1024_mv_lora_ibr_texverse_fixedmem05",
)
STEP = 15000
_MARKER = ".worldsculpt-private-output"
_MAX_MESH_PT_BYTES = 4 * 1024**3


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _numeric(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _local_file(root: Path, path: Path, label: str) -> Path:
    if any(part.is_symlink() for part in (path, *path.parents) if _inside(root, part)) or not _inside(root, path.resolve()) or not path.is_file():
        raise ValueError(f"{label} must be a local regular file, not a symlink")
    return path


def _matrix(value: object, label: str, *, rigid: bool = False):
    import numpy as np

    matrix = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else np.asarray([])
    if matrix.shape != (4, 4) or not all(_numeric(v) for v in matrix.flat):
        raise ValueError(f"{label} requires a finite 4x4 matrix")
    matrix = matrix.astype(np.float64)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f"{label} requires a homogeneous matrix")
    rotation = matrix[:3, :3]
    if abs(np.linalg.det(rotation)) < 1e-8:
        raise ValueError(f"{label} is singular")
    if rigid and (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
                  or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4, rtol=0)):
        raise ValueError(f"{label} must be a rigid proper camera transform")
    return matrix


def _mask_in_crop(mask, box, size) -> int:
    import numpy as np

    width, height = size
    h, w = mask.shape
    x0, y0, x1, y1 = box
    if (w, h) != (width, height):
        x0, x1 = math.floor(x0 * w / width), math.ceil(x1 * w / width)
        y0, y1 = math.floor(y0 * h / height), math.ceil(y1 * h / height)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1] > 127))


def _usable_crop(box, c2w, intrinsic, mask, size) -> bool:
    """Mirror the official default projection, 0.005 pad and 3x crop limit."""
    import numpy as np

    center = (box[0] + box[1]) / 2
    cube_size = float((box[1] - box[0]).max())
    w2c = np.linalg.inv(c2w @ np.diag([1, -1, -1, 1]))
    center_camera = w2c @ np.append(center, 1)
    if center_camera[2] <= 1e-3:
        return False
    projected_center = intrinsic @ (center_camera[:3] / center_camera[2])
    corners = np.array([(x, y, z) for x in (-.5, .5)
                        for y in (-.5, .5) for z in (-.5, .5)]) * cube_size + center
    camera_corners = (w2c @ np.column_stack((corners, np.ones(8))).T).T[:, :3]
    valid = camera_corners[:, 2] > 1e-3
    if valid.sum() < 4:
        return False
    projected = (intrinsic @ (camera_corners[valid] / camera_corners[valid, 2:3]).T).T
    extent = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    side = round(extent * 1.005)
    if not math.isfinite(side) or side <= 0 or side > 3 * max(size):
        return False
    x0 = round(projected_center[0] - side / 2)
    y0 = round(projected_center[1] - side / 2)
    return _mask_in_crop(mask, (x0, y0, x0 + side, y0 + side), size) >= max(500, .001 * size[0] * size[1])


def validate_scene(scene_dir: Path) -> tuple[str, ...]:
    """Return instances with a real >=500-pixel, projectable crop candidate.

    This mirrors the published inference.sh crop options for the world-aligned
    AABB path. OBB/anchor orientation and mask-fit scaling need runtime proof
    before a WorldSculpt node can be enabled.
    """
    import numpy as np
    from PIL import Image

    scene_dir = Path(scene_dir).resolve(strict=True)
    metadata = _local_file(scene_dir, scene_dir / "transforms.json", "transforms.json")
    data = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("transforms.json must contain a scene object")
    for key in ("fl_x", "fl_y", "cx", "cy", "w", "h"):
        if not _numeric(data.get(key)) or (key in {"fl_x", "fl_y", "w", "h"} and data[key] <= 0):
            raise ValueError(f"invalid camera intrinsic: {key}")
    size = (data["w"], data["h"])
    if any(type(v) is not int for v in size):
        raise ValueError("camera w and h must be integer pixels")
    intrinsic = np.array([[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]])
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("WorldSculpt requires nonempty frames")
    cameras = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not isinstance(frame.get("file_path"), str):
            raise ValueError(f"frame {index} has no file_path")
        path = scene_dir / frame["file_path"]
        if path.suffix.lower() != ".png":
            raise ValueError(f"frame {index} must reference a local PNG")
        _local_file(scene_dir, path, f"frame {index} PNG")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != size:
                raise ValueError(f"frame {index} must be a decodable full-size PNG")
        cameras.append(_matrix(frame.get("transform_matrix"), f"frame {index} c2w", rigid=True))
    instances = data.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("WorldSculpt requires instances")
    eligible = []
    seen = set()
    for instance in instances:
        if not isinstance(instance, dict) or type(instance.get("pass_index")) is not int or instance["pass_index"] < 0:
            raise ValueError("instance requires a nonnegative pass_index")
        index = instance["pass_index"]
        if index in seen:
            raise ValueError(f"duplicate pass_index {index}")
        seen.add(index)
        box = instance.get("aabb_world")
        if box is None:
            continue
        if not isinstance(box, list) or np.asarray(box).shape != (2, 3) or not all(_numeric(v) for row in box for v in row):
            raise ValueError(f"invalid aabb_world for pass_index {index}")
        box = np.asarray(box, dtype=np.float64)
        extents = box[1] - box[0]
        if np.any(extents < 0):
            raise ValueError(f"inverted aabb_world for pass_index {index}")
        if extents.max() < 1e-3:
            continue
        if "gt_scales" in instance or "obj_pose_world" in instance:
            if not ("gt_scales" in instance and "obj_pose_world" in instance):
                raise ValueError(f"incomplete OBB metadata for pass_index {index}")
            scales = instance["gt_scales"]
            if not isinstance(scales, list) or len(scales) != 3 or not all(_numeric(v) and v > 0 for v in scales):
                raise ValueError(f"invalid gt_scales for pass_index {index}")
            _matrix(instance["obj_pose_world"], f"obj{index:02d} pose", rigid=True)
            raise ValueError("OBB crop eligibility requires a separately verified projection path")
        name = f"obj{index:02d}"
        present_masks = []
        for frame_index, camera in enumerate(cameras):
            mask = scene_dir / "masks" / name / f"{frame_index:04d}.png"
            if not mask.exists() and not mask.is_symlink():
                continue
            _local_file(scene_dir, mask, f"{name} mask")
            with Image.open(mask) as image:
                image.load()
                if image.format != "PNG":
                    raise ValueError(f"{name} mask must be a decodable PNG")
                gray = np.asarray(image.convert("L"))
            present_masks.append((camera, gray))
        for camera, gray in present_masks:
            if np.count_nonzero(gray > 127) < max(500, .001 * gray.size):
                continue
            if _usable_crop(box, camera, intrinsic, gray, size):
                eligible.append(name)
                break
    if not eligible:
        raise ValueError("scene has no crop-eligible masked instances")
    return tuple(eligible)


def validate_adapters(root: Path) -> None:
    """Require distinct, local LoRA stage trees at the exact official step."""
    root = Path(root).resolve(strict=True)
    stages = []
    for directory in ADAPTER_DIRS:
        stage = root / directory
        if stage.is_symlink() or not stage.is_dir() or not _inside(root, stage.resolve()):
            raise ValueError(f"invalid WorldSculpt adapter stage: {directory}")
        stages.append(stage.resolve())
        for relative in ("config.json", f"ckpts/denoiser_step{STEP:07d}.pt", f"ckpts/mv_aggregator_step{STEP:07d}.pt"):
            file = stage / relative
            if not file.exists():
                raise FileNotFoundError(f"WorldSculpt local adapter missing: {directory}/{relative}")
            _local_file(root, file, f"WorldSculpt adapter {directory}/{relative}")
        config = json.loads((stage / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or "models" not in config or "trainer" not in config:
            raise ValueError(f"invalid WorldSculpt config: {directory}/config.json")
    if len(set(stages)) != len(stages):
        raise ValueError("WorldSculpt adapter stages alias one another")


def prepare_case_root(case_root: Path) -> Path:
    """Create a new, private, never-reused output root before invoking upstream."""
    case_root = Path(case_root)
    case_root.mkdir(parents=False, exist_ok=False)
    if case_root.is_symlink():
        raise ValueError("WorldSculpt output root cannot be a symlink")
    (case_root / _MARKER).write_text("private WorldSculpt output\n", encoding="ascii")
    return case_root


def _load_upstream_mesh(mesh: Path):
    """Load the upstream NumPy float64 transform without general pickle access."""
    import numpy as np
    import torch

    if mesh.stat().st_size > _MAX_MESH_PT_BYTES:
        raise ValueError("mesh.pt exceeds the validation size limit")
    with zipfile.ZipFile(mesh) as archive:
        if sum(info.file_size for info in archive.infolist()) > _MAX_MESH_PT_BYTES:
            raise ValueError("mesh.pt exceeds the decoded size limit")
    reconstruct = np._core.multiarray._reconstruct if hasattr(np, "_core") else np.core.multiarray._reconstruct
    allowed = [reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]
    expected = {"numpy.ndarray", "numpy.dtype", "numpy.core.multiarray._reconstruct",
                "numpy._core.multiarray._reconstruct"}
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(mesh))
    if not unsafe <= expected:
        raise ValueError(f"mesh.pt contains unexpected pickle globals: {sorted(unsafe - expected)}")
    with torch.serialization.safe_globals(allowed):
        state = torch.load(mesh, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or type(state.get("T_canon_to_metric")) is not np.ndarray or state["T_canon_to_metric"].dtype != np.dtype("float64"):
        raise ValueError("mesh.pt requires an upstream float64 NumPy transform")
    return state


def validate_output(case_root: Path, eligible: tuple[str, ...]) -> Path:
    """Validate fresh per-object tensors and a GLB containing every instance."""
    import numpy as np
    import torch
    import trimesh

    case_root = Path(case_root)
    if case_root.is_symlink():
        raise ValueError("WorldSculpt output root cannot be a symlink")
    case_root = case_root.resolve(strict=True)
    marker = _local_file(case_root, case_root / _MARKER, "private output marker")
    if not eligible or len(set(eligible)) != len(eligible):
        raise ValueError("eligible instances must be nonempty and unique")
    recon_root = case_root / "_recon"
    actual = {entry.name for entry in recon_root.iterdir() if entry.is_dir()}
    if actual != set(eligible):
        raise ValueError(f"WorldSculpt reconstruction set differs from eligible instances: {sorted(actual ^ set(eligible))}")
    for name in eligible:
        if not name.startswith("obj") or not name[3:].isdigit():
            raise ValueError(f"invalid eligible instance: {name}")
        mesh = _local_file(case_root, case_root / "_recon" / name / "mesh.pt", f"{name} mesh.pt")
        if mesh.stat().st_mtime_ns < marker.stat().st_mtime_ns:
            raise ValueError(f"stale reconstruction: {mesh}")
        try:
            state = _load_upstream_mesh(mesh)
        except Exception as exc:
            raise ValueError(f"invalid {name} mesh.pt") from exc
        if not isinstance(state, dict) or state.get("mode") not in {"offcenter_mv", "offcenter_mv_geom"}:
            raise ValueError(f"invalid {name} mesh.pt state")
        vertices, faces = state.get("vertices"), state.get("faces")
        if not isinstance(vertices, torch.Tensor) or not isinstance(faces, torch.Tensor) or vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3 or faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
            raise ValueError(f"invalid {name} mesh.pt geometry")
        if not torch.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError(f"invalid {name} mesh.pt geometry values")
        transform = state.get("T_canon_to_metric")
        if isinstance(transform, torch.Tensor):
            transform = transform.numpy()
        _matrix(transform, f"{name} metric transform")
    glb = _local_file(case_root, case_root / "_scene" / "scene.glb", "scene.glb")
    if glb.stat().st_mtime_ns < marker.stat().st_mtime_ns:
        raise ValueError("stale scene.glb")
    with glb.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise ValueError("WorldSculpt scene.glb has an invalid GLB header")
    magic, version, length = struct.unpack("<4sII", header)
    if magic != b"glTF" or version != 2 or length != glb.stat().st_size:
        raise ValueError("WorldSculpt scene.glb has an invalid GLB header")
    try:
        loaded = trimesh.load(glb, force="scene")
    except Exception as exc:
        raise ValueError("WorldSculpt scene.glb is not loadable") from exc
    if not loaded.geometry:
        raise ValueError("WorldSculpt scene.glb has no geometry")
    names = set(loaded.geometry)
    unexpected = {geometry for geometry in names if not any(
        geometry == name or geometry.startswith(name + "__") for name in eligible)}
    if unexpected:
        raise ValueError(f"WorldSculpt scene.glb contains unexpected geometry: {sorted(unexpected)}")
    for name in eligible:
        matched = [geometry for geometry in names if geometry == name or geometry.startswith(name + "__")]
        if not matched:
            raise ValueError(f"WorldSculpt scene.glb omits {name}")
        for geometry in matched:
            surface = loaded.geometry[geometry]
            if (not isinstance(surface, trimesh.Trimesh) or len(surface.vertices) < 3
                    or len(surface.faces) < 1 or not np.isfinite(surface.vertices).all()
                    or not np.isfinite(surface.faces).all() or surface.faces.min() < 0
                    or surface.faces.max() >= len(surface.vertices)):
                raise ValueError(f"WorldSculpt scene.glb has invalid geometry for {geometry}")
    return glb


def validate_crops(case_root: Path, eligible: tuple[str, ...]) -> None:
    """Gate reconstruction on actual official crop outputs, not box estimates."""
    from PIL import Image

    case_root = Path(case_root).resolve(strict=True)
    _local_file(case_root, case_root / _MARKER, "private output marker")
    crops_root = case_root / "_crops"
    if crops_root.is_symlink() or not crops_root.is_dir():
        raise ValueError("WorldSculpt crop root must be a local directory")
    actual = {entry.name for entry in crops_root.iterdir() if entry.is_dir()}
    if actual != set(eligible):
        raise ValueError(f"WorldSculpt crop set differs from eligible instances: {sorted(actual ^ set(eligible))}")
    for name in eligible:
        folder = case_root / "_crops" / name
        metadata = _local_file(case_root, folder / "transforms.json", f"{name} crop metadata")
        data = json.loads(metadata.read_text(encoding="utf-8"))
        frames = data.get("frames") if isinstance(data, dict) else None
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{name} has no usable crop frames")
        if data.get("pass_index") != int(name[3:]):
            raise ValueError(f"{name} crop pass_index mismatch")
        rotation = data.get("R_box")
        if not isinstance(rotation, list) or len(rotation) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in rotation):
            raise ValueError(f"{name} has invalid R_box")
        _matrix([row + [0] for row in rotation] + [[0, 0, 0, 1]], f"{name} R_box", rigid=True)
        anchor = data.get("anchor_full_idx")
        if type(anchor) is not int or anchor < 0 or sum(frame.get("is_anchor") is True for frame in frames if isinstance(frame, dict)) != 1:
            raise ValueError(f"{name} has invalid anchor metadata")
        seen_frames = set()
        for frame in frames:
            if not isinstance(frame, dict) or type(frame.get("full_frame_idx")) is not int or type(frame.get("crop_size")) is not int or frame["crop_size"] <= 0:
                raise ValueError(f"{name} has invalid crop frame metadata")
            if frame["full_frame_idx"] in seen_frames:
                raise ValueError(f"{name} has duplicate crop frame index")
            seen_frames.add(frame["full_frame_idx"])
            image_size = frame.get("image_size_px")
            if type(image_size) is not int or image_size <= 0:
                raise ValueError(f"{name} has invalid image_size_px")
            if frame.get("is_anchor") is True and frame["full_frame_idx"] != anchor:
                raise ValueError(f"{name} anchor does not match saved frame")
            filename = frame.get("file_path")
            if filename != f'{frame["full_frame_idx"]:04d}.png':
                raise ValueError(f"{name} crop file metadata mismatch")
            crop = _local_file(case_root, folder / filename, f"{name} crop PNG")
            with Image.open(crop) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGBA" or image.size != (image_size, image_size) or not image.getchannel("A").getbbox():
                    raise ValueError(f"{name} has invalid or empty RGBA crop")
            _matrix(frame.get("transform_matrix"), f"{name} crop c2w", rigid=True)
            intrinsics = frame.get("K_image_pix")
            if not isinstance(intrinsics, list) or len(intrinsics) != 3 or any(not isinstance(row, list) or len(row) != 3 or not all(_numeric(v) for v in row) for row in intrinsics):
                raise ValueError(f"{name} has invalid crop intrinsics")
            if intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0:
                raise ValueError(f"{name} has invalid crop focal length")
