"""Clean-room capture/scene custody and canonical annotated-scene publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path, PureWindowsPath

from .worldsculpt import resolve_scene_manifest
from .worldsculpt_contract import validate_scene


CAPTURE_SCHEMA = "modly.capture-manifest.v1"
CAPTURE_MANIFEST = "capture-manifest.json"
SCENE_SCHEMA = "modly.scene-manifest.v1"


def validate_workspace_output_parent(output_dir: Path, workspace_dir: Path, label: str) -> Path:
    """Validate workspace authority without creating or following output paths."""
    workspace = Path(workspace_dir).resolve(strict=True)
    raw = Path(output_dir)
    raw_text = os.fspath(output_dir)
    if not raw_text or "\x00" in raw_text or PureWindowsPath(raw_text).is_absolute() and os.name != "nt":
        raise ValueError(f"{label} must be a native workspace path")
    if any(part in {".", ".."} for part in raw.parts):
        raise ValueError(f"{label} must not contain traversal segments")
    candidate = raw if raw.is_absolute() else workspace / raw
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the workspace") from exc
    if not relative.parts:
        raise ValueError(f"{label} cannot be the workspace root")
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlinks")
    resolved = candidate.resolve(strict=False)
    if resolved == workspace or not resolved.is_relative_to(workspace):
        raise ValueError(f"{label} must be inside a workspace-owned subdirectory")
    return resolved


def _inside(path: Path, root: Path, label: str, *, require_file: bool | None = None) -> Path:
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its allowed root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlinks")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes its allowed root")
    if require_file is True and not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    if require_file is False and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty relative path")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in normalized.split("/")):
        raise ValueError(f"{label} must be a safe relative path")
    return Path(*normalized.split("/"))


def load_capture_manifest(manifest_path: Path, workspace_dir: Path) -> tuple[dict, Path]:
    """Revalidate the host capture envelope inside the isolated worker."""
    workspace = Path(workspace_dir).resolve(strict=True)
    manifest = _inside(Path(manifest_path), workspace, "capture manifest", require_file=True)
    if manifest.name != CAPTURE_MANIFEST:
        raise ValueError(f"Capture input must be {CAPTURE_MANIFEST}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"Capture schema must be {CAPTURE_SCHEMA}")
    root_value = data.get("captureRoot")
    if root_value == ".":
        root = manifest.parent
    else:
        root = workspace / _safe_relative(root_value, "captureRoot")
    root = _inside(root, workspace, "capture root", require_file=False)
    provenance = data.get("provenance")
    if (not isinstance(provenance, dict) or not isinstance(provenance.get("source"), str)
            or not provenance["source"].strip()
            or provenance.get("ordering") not in {"manifest-index", "decode-index"}):
        raise ValueError("Capture provenance requires a source and deterministic ordering")
    kind = data.get("kind")
    if kind == "frames":
        if provenance["ordering"] != "manifest-index":
            raise ValueError("Frame capture ordering must be manifest-index")
        frames = data.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Frame capture requires frames")
        seen = set()
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or frame.get("index") != index:
                raise ValueError("Capture frame order must be contiguous")
            relative = _safe_relative(frame.get("path"), f"frame {index}")
            if relative.as_posix() in seen:
                raise ValueError("Capture frame paths must be unique")
            seen.add(relative.as_posix())
            path = _inside(root / relative, root, f"frame {index}", require_file=True)
            if type(frame.get("byteSize")) is not int or frame["byteSize"] != path.stat().st_size:
                raise ValueError(f"Capture frame {index} size changed")
            if any(type(frame.get(field)) is not int or frame[field] <= 0 for field in ("width", "height")):
                raise ValueError(f"Capture frame {index} dimensions are invalid")
    elif kind == "video":
        if provenance["ordering"] != "decode-index":
            raise ValueError("Video capture ordering must be decode-index")
        video = data.get("video")
        if not isinstance(video, dict):
            raise ValueError("Video capture requires video metadata")
        path = _inside(root / _safe_relative(video.get("path"), "video"), root, "video", require_file=True)
        if type(video.get("byteSize")) is not int or video["byteSize"] != path.stat().st_size:
            raise ValueError("Capture video size changed")
        if any(type(video.get(field)) is not int or video[field] <= 0 for field in ("width", "height", "frameCount")):
            raise ValueError("Capture video dimensions are invalid")
    else:
        raise ValueError("Capture kind must be frames or video")
    return data, root


def _copy_file(source_root: Path, destination_root: Path, relative: Path, label: str) -> None:
    source = _inside(source_root / relative, source_root, label, require_file=True)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f"{label} destination cannot be a symlink")
    shutil.copyfile(source, destination)


def normalize_annotated_scene(scene_manifest_path: Path, workspace_dir: Path, output_dir: Path) -> Path:
    """Validate and atomically copy only the canonical WorldSculpt scene contract."""
    workspace = Path(workspace_dir).resolve(strict=True)
    manifest_input = _inside(Path(scene_manifest_path), workspace, "scene manifest", require_file=True)
    scene_root = resolve_scene_manifest(manifest_input, workspace)
    eligible = validate_scene(scene_root)
    transforms_path = _inside(scene_root / "transforms.json", scene_root, "transforms.json", require_file=True)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    output_parent = validate_workspace_output_parent(output_dir, workspace, "scene output directory")
    if output_parent == scene_root or output_parent.is_relative_to(scene_root):
        raise ValueError("scene output directory must not overlap the input scene")
    output_parent.mkdir(parents=True, exist_ok=True)
    output_parent = _inside(output_parent, workspace, "scene output directory", require_file=False)
    staging = Path(tempfile.mkdtemp(prefix=".scene-normalize-", dir=output_parent))
    final = output_parent / f"scene-normalized-{uuid.uuid4().hex}"
    try:
        _copy_file(scene_root, staging, Path("transforms.json"), "transforms.json")
        for frame_index, frame in enumerate(transforms["frames"]):
            _copy_file(scene_root, staging, _safe_relative(frame["file_path"], f"frame {frame_index}"), f"frame {frame_index}")
        for instance in transforms["instances"]:
            obj = f"obj{instance['pass_index']:02d}"
            for frame_index in range(len(transforms["frames"])):
                relative = Path("masks") / obj / f"{frame_index:04d}.png"
                candidate = scene_root / relative
                if candidate.exists() or candidate.is_symlink():
                    _copy_file(scene_root, staging, relative, f"{obj} mask {frame_index}")
        normalized = json.loads((staging / "transforms.json").read_text(encoding="utf-8"))
        provenance = normalized.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        scale = provenance.get("scale")
        if not isinstance(scale, dict) or scale.get("mode") not in {"metric", "relative", "unknown"}:
            scale = {"mode": "unknown"}
        provenance.update({"scale": scale, "normalizer": "pixal3d-clean-room-v1"})
        normalized["provenance"] = provenance
        (staging / "transforms.json").write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scene_manifest = {
            "schema": SCENE_SCHEMA,
            "sceneRoot": ".",
            "assets": [{"path": "transforms.json", "kind": "worldsculpt-transforms"}],
            "provenance": {"sourceManifest": manifest_input.relative_to(workspace).as_posix(), "normalizer": "pixal3d-clean-room-v1"},
        }
        (staging / "scene-manifest.json").write_text(json.dumps(scene_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_scene(staging)
        os.replace(staging, final)
        return final / "scene-manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
