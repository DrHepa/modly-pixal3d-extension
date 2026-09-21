#!/usr/bin/env python3
"""Real RTX 50-series Blackwell validation helper for the Pixal3D candidate lane.

The module is import-safe on Linux CI: heavy PyTorch/Pixal3D imports happen only
inside the real validation command. Static/unit tests exercise the fail-closed
file, path, GLB, workflow, and schema contracts without GPU hardware.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any
import urllib.request


def _bootstrap_is_reparse_or_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def _bootstrap_contained_regular_file(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"bootstrap_path_escape: {relative_path} resolves outside extension root {root}") from exc
    if _bootstrap_is_reparse_or_symlink(candidate) or not candidate.is_file():
        raise RuntimeError(f"bootstrap_missing_regular_file: {relative_path} under extension root {root}")
    return candidate


def _bootstrap_extension_root() -> Path:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    expected_validation_dir = root / "tools" / "validation"
    if script_path.parent != expected_validation_dir:
        raise RuntimeError(f"bootstrap_unexpected_layout: expected helper under {expected_validation_dir}, found {script_path}")
    for relative_path in (
        "generator.py",
        "pixal3d_extension/__init__.py",
        "pixal3d_extension/assets.py",
        "pixal3d_extension/naf_checkpoint.py",
    ):
        _bootstrap_contained_regular_file(root, relative_path)
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != root_text]
    sys.path.insert(0, root_text)
    return root


EXTENSION_ROOT = _bootstrap_extension_root()

from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET
from pixal3d_extension.naf_checkpoint import verify_naf_checkpoint

HEX_SHA256_LENGTH = 64
# Scale-aware degeneracy threshold:
# compare twice-area against max-edge-length-squared * 2^-40.  The terms share
# length^2 units, so tiny but well-shaped triangles remain valid while repeated,
# identical, and collinear triangles fail independent of scene scale.
TRIANGLE_AREA_RELATIVE_EPSILON = 2.0**-40
BLACKWELL_REQUIRED_IMPORTS = [
    "cumesh_vb",
    "flex_gemm_ap",
    "o_voxel_vb_ap",
    "drtk",
    "flash_attn",
    "nvdiffrast",
    "nvdiffrec_render",
    "natten",
]
BLACKWELL_REQUIRED_UPSTREAM_IMPORTS = ["cumesh", "flex_gemm", "o_voxel"]
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_ETAG_TIMEOUT": "1",
    "HF_HUB_DOWNLOAD_TIMEOUT": "1",
}


class ValidationError(RuntimeError):
    """Fail-closed validation error with a stable code."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def as_gate(self) -> dict[str, Any]:
        return {"status": "failed", "code": self.code, "message": str(self), **self.details}


class NetworkDenied:
    """Deny common Python network paths during Windows validation generation."""

    def __init__(self) -> None:
        self._restore: list[tuple[Any, str, Any]] = []
        self._targets: list[str] = []
        self._entered = False

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        self._restore.append((owner, name, getattr(owner, name)))
        owner_name = getattr(owner, "__name__", owner.__class__.__name__)
        self._targets.append(f"{owner_name}.{name}")
        setattr(owner, name, replacement)

    def __enter__(self) -> "NetworkDenied":
        def denied(*_args: Any, **_kwargs: Any) -> Any:
            raise ValidationError("network_denied", "Network access is denied during Blackwell hardware validation")

        # Network denial covers socket.create_connection for Windows process execution.
        self._patch(socket, "create_connection", denied)
        self._patch(socket, "socket", denied)
        # Network denial covers urllib.request.urlopen for Windows process execution.
        self._patch(urllib.request, "urlopen", denied)
        try:
            import requests.sessions  # type: ignore[import-not-found]
        except Exception:
            pass
        else:
            self._patch(requests.sessions.Session, "request", denied)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        while self._restore:
            owner, name, original = self._restore.pop()
            setattr(owner, name, original)
        return False

    def evidence(self) -> dict[str, Any]:
        if not self._entered or not self._targets:
            raise ValidationError("network_denial_not_executed", "Network denial guard was not installed during generation")
        return {
            "status": "passed",
            "scope": "generator_load_and_generation",
            "denied_symbols": list(self._targets),
            "offline_environment": OFFLINE_ENVIRONMENT,
        }


def set_offline_environment() -> dict[str, str]:
    previous = {key: os.environ.get(key, "") for key in OFFLINE_ENVIRONMENT}
    for key, value in OFFLINE_ENVIRONMENT.items():
        os.environ[key] = value
    return previous


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def validate_regular_file(path: Path | str, *, code: str = "file_missing") -> Path:
    path = Path(path)
    if _is_reparse_or_symlink(path):
        raise ValidationError("reparse_or_symlink_rejected", f"Path must not be a symlink or reparse point: {path}")
    if not path.is_file():
        raise ValidationError(code, f"Required regular file does not exist: {path}")
    return path


def validate_file_contract(path: Path | str, expected_sha256: str, expected_size_bytes: int) -> dict[str, Any]:
    path = validate_regular_file(path)
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != HEX_SHA256_LENGTH or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValidationError("invalid_expected_sha256", "Expected SHA256 must be 64 lowercase hex characters")
    size = path.stat().st_size
    if size != int(expected_size_bytes):
        raise ValidationError("size_mismatch", f"Expected {expected_size_bytes} bytes, found {size}", details={"actual_size_bytes": size})
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValidationError("sha256_mismatch", "SHA256 verification failed", details={"actual_sha256": digest})
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def assert_path_contained(path: Path | str, root: Path | str) -> dict[str, str]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValidationError("path_escape", f"Path escapes validation root: {path}", details={"root": str(root)}) from exc
    if not path.exists():
        raise ValidationError("path_missing", f"Path does not exist inside validation root: {path}", details={"root": str(root)})
    return {"path": str(path), "root": str(root)}


def validate_generated_glb_output(path: Path | str, *, output_dir: Path | str, workspace_dir: Path | str) -> dict[str, Any]:
    path = validate_regular_file(path, code="generated_glb_missing")
    if path.suffix.lower() != ".glb":
        raise ValidationError("generated_output_not_glb", f"Generated output must be a .glb file: {path}")
    resolved = path.resolve()
    output_root = Path(output_dir).resolve()
    workspace_root = Path(workspace_dir).resolve()
    try:
        resolved.relative_to(output_root)
    except ValueError as output_exc:
        try:
            resolved.relative_to(workspace_root)
        except ValueError as workspace_exc:
            raise ValidationError(
                "generated_glb_path_escape",
                "Generated GLB must be a regular file inside output_dir or workspace_dir before parsing/hash",
                details={"path": str(resolved), "output_dir": str(output_root), "workspace_dir": str(workspace_root)},
            ) from workspace_exc
        if output_root != workspace_root:
            raise ValidationError(
                "generated_glb_outside_output_dir",
                "Generated GLB must stay inside the requested output_dir",
                details={"path": str(resolved), "output_dir": str(output_root)},
            ) from output_exc
    return validate_glb(resolved)


def _read_glb_chunks(data: bytes) -> tuple[dict[str, Any], bytes | None]:
    if len(data) < 20:
        raise ValidationError("glb_too_small", "GLB file is too small to contain a valid header and JSON chunk")
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValidationError("invalid_glb_magic", "GLB magic must be glTF")
    if version != 2:
        raise ValidationError("invalid_glb_version", "GLB version must be 2")
    if declared_length != len(data):
        raise ValidationError("glb_length_mismatch", "GLB declared length must match file size", details={"declared_length": declared_length, "actual_length": len(data)})

    offset = 12
    json_chunk: dict[str, Any] | None = None
    bin_chunk: bytes | None = None
    chunk_count = 0
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValidationError("truncated_chunk_header", "GLB chunk header is truncated")
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(data):
            raise ValidationError("truncated_chunk_payload", "GLB chunk payload is truncated")
        payload = data[offset:chunk_end]
        offset = chunk_end
        chunk_count += 1
        if chunk_type == b"JSON":
            if json_chunk is not None:
                raise ValidationError("duplicate_json_chunk", "GLB contains more than one JSON chunk")
            json_chunk = json.loads(payload.rstrip(b" \t\r\n\x00").decode("utf-8"))
        elif chunk_type == b"BIN\x00":
            if bin_chunk is not None:
                raise ValidationError("duplicate_bin_chunk", "GLB contains more than one BIN chunk")
            bin_chunk = payload
        else:
            raise ValidationError("unknown_glb_chunk", f"Unsupported GLB chunk type: {chunk_type!r}")
    if json_chunk is None:
        raise ValidationError("missing_json_chunk", "GLB is missing its JSON chunk")
    return {"json": json_chunk, "chunk_count": chunk_count}, bin_chunk


def _non_negative_int(value: Any, *, code: str, message: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(code, message, details={"value": value}) from exc
    if parsed < 0:
        raise ValidationError(code, message, details={"value": value})
    return parsed


def _validate_accessor_positions(
    *,
    mesh_index: int,
    primitive_index: int,
    position_index: int,
    accessors: list[Any],
    buffer_views: list[Any],
    bin_chunk: bytes,
) -> list[tuple[float, float, float]]:
    if position_index < 0 or position_index >= len(accessors):
        raise ValidationError("missing_position_accessor", "POSITION references an invalid accessor", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index})
    accessor = accessors[position_index]
    if not isinstance(accessor, dict):
        raise ValidationError("invalid_position_accessor", "POSITION accessor must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index})
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        raise ValidationError("unsupported_position_accessor", "POSITION accessor must be FLOAT VEC3", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index})
    count = _non_negative_int(accessor.get("count"), code="empty_position_accessor", message="POSITION accessor must contain vertices")
    if count <= 0:
        raise ValidationError("empty_position_accessor", "POSITION accessor must contain vertices", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index})
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(buffer_views):
        raise ValidationError("missing_position_bufferview", "POSITION accessor has no valid bufferView", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index})
    view = buffer_views[view_index]
    if not isinstance(view, dict):
        raise ValidationError("invalid_position_bufferview", "POSITION bufferView must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "bufferView": view_index})
    if int(view.get("buffer", 0)) != 0:
        raise ValidationError("unsupported_buffer_index", "Only buffer 0 is supported for validation", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "bufferView": view_index})
    view_offset = _non_negative_int(view.get("byteOffset") or 0, code="invalid_position_range", message="POSITION bufferView byteOffset must be non-negative")
    accessor_offset = _non_negative_int(accessor.get("byteOffset") or 0, code="invalid_position_range", message="POSITION accessor byteOffset must be non-negative")
    view_length = _non_negative_int(view.get("byteLength"), code="invalid_position_range", message="POSITION bufferView byteLength must be non-negative")
    stride = _non_negative_int(view.get("byteStride") or 12, code="invalid_position_stride", message="POSITION byteStride must be a non-negative integer")
    component_size = 4
    element_size = 12
    effective_offset = view_offset + accessor_offset
    if effective_offset % component_size != 0:
        raise ValidationError(
            "misaligned_position_accessor",
            "POSITION effective byte offset must align to FLOAT component size",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "byteOffset": accessor_offset, "bufferViewByteOffset": view_offset, "component_size": component_size},
        )
    if stride < element_size or stride % component_size != 0 or stride > 252:
        raise ValidationError(
            "invalid_position_stride",
            "POSITION byteStride must fit FLOAT VEC3 data and align to component size",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "stride": stride, "component_size": component_size},
        )
    byte_length_required = 12 if count == 1 else ((count - 1) * stride) + 12
    if accessor_offset + byte_length_required > view_length:
        raise ValidationError(
            "position_data_truncated",
            "POSITION accessor range extends beyond its bufferView",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index, "bufferView": view_index},
        )
    base = view_offset + accessor_offset
    end = base + byte_length_required
    if end > len(bin_chunk):
        raise ValidationError(
            "position_data_truncated",
            "POSITION data extends beyond the BIN chunk",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index, "bufferView": view_index},
        )
    positions: list[tuple[float, float, float]] = []
    for index in range(count):
        start = base + index * stride
        xyz = struct.unpack_from("<fff", bin_chunk, start)
        if not all(math.isfinite(value) for value in xyz):
            raise ValidationError("non_finite_position", "POSITION contains non-finite coordinates", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "vertex_index": index})
        positions.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return positions


INDEX_COMPONENTS = {
    5121: ("<B", 1, "UNSIGNED_BYTE"),
    5123: ("<H", 2, "UNSIGNED_SHORT"),
    5125: ("<I", 4, "UNSIGNED_INT"),
}


def _validate_index_values_for_primitive(
    primitive: dict[str, Any],
    accessors: list[Any],
    buffer_views: list[Any],
    bin_chunk: bytes,
    *,
    position_count: int,
    mesh_index: int,
    primitive_index: int,
) -> tuple[int, bool, list[int]]:
    index_accessor = primitive.get("indices")
    if index_accessor is None:
        return 0, False, []
    if not isinstance(index_accessor, int) or index_accessor < 0 or index_accessor >= len(accessors):
        raise ValidationError("invalid_indices_accessor", "Primitive indices reference an invalid accessor", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": index_accessor})
    accessor = accessors[index_accessor]
    if not isinstance(accessor, dict):
        raise ValidationError("invalid_indices_accessor", "Primitive indices accessor must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index})
    count = _non_negative_int(accessor.get("count"), code="invalid_indices_accessor", message="Primitive indices accessor count must be non-negative")
    if count <= 0:
        raise ValidationError("empty_indices_accessor", "Indexed TRIANGLES primitive must contain at least one index", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": index_accessor})
    component_type = accessor.get("componentType")
    if component_type not in INDEX_COMPONENTS:
        raise ValidationError(
            "unsupported_index_component_type",
            "Primitive indices must use an unsigned integer component type",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "componentType": component_type, "allowed": sorted(INDEX_COMPONENTS)},
        )
    if accessor.get("type") != "SCALAR":
        raise ValidationError("unsupported_index_accessor_type", "Primitive indices accessor must be SCALAR", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "type": accessor.get("type")})
    view_index = accessor.get("bufferView")
    if not isinstance(view_index, int) or view_index < 0 or view_index >= len(buffer_views):
        raise ValidationError("missing_indices_bufferview", "Primitive indices accessor has no valid bufferView", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": index_accessor})
    view = buffer_views[view_index]
    if not isinstance(view, dict):
        raise ValidationError("invalid_indices_bufferview", "Primitive indices bufferView must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "bufferView": view_index})
    if int(view.get("buffer", 0)) != 0:
        raise ValidationError("unsupported_index_buffer", "Only buffer 0 is supported for primitive indices validation", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "bufferView": view_index})

    fmt, component_size, component_name = INDEX_COMPONENTS[component_type]
    view_offset = _non_negative_int(view.get("byteOffset") or 0, code="invalid_index_range", message="Indices bufferView byteOffset must be non-negative")
    view_length = _non_negative_int(view.get("byteLength"), code="invalid_index_range", message="Indices bufferView byteLength must be non-negative")
    accessor_offset = _non_negative_int(accessor.get("byteOffset") or 0, code="invalid_index_range", message="Indices accessor byteOffset must be non-negative")
    stride = _non_negative_int(view.get("byteStride") or component_size, code="invalid_index_stride", message="Indices byteStride must be a non-negative integer")
    if accessor_offset % component_size != 0 or view_offset % component_size != 0:
        raise ValidationError(
            "misaligned_index_accessor",
            f"Indices accessor and bufferView offsets must align to {component_name}",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "component_size": component_size, "byteOffset": accessor_offset, "bufferViewByteOffset": view_offset},
        )
    if stride < component_size or stride % component_size != 0:
        raise ValidationError(
            "invalid_index_stride",
            f"Indices byteStride must be at least and aligned to {component_name}",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "component_size": component_size, "byteStride": stride},
        )
    byte_length_required = 0 if count == 0 else ((count - 1) * stride) + component_size
    if accessor_offset + byte_length_required > view_length:
        raise ValidationError(
            "index_data_truncated",
            "Indices accessor range extends beyond its bufferView",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": index_accessor, "bufferView": view_index, "count": count},
        )
    base = view_offset + accessor_offset
    end = base + byte_length_required
    if end > len(bin_chunk):
        raise ValidationError(
            "index_data_truncated",
            "Indices data extends beyond the BIN chunk",
            details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": index_accessor, "bufferView": view_index, "count": count},
        )
    values: list[int] = []
    for index in range(count):
        start = base + index * stride
        value = int(struct.unpack_from(fmt, bin_chunk, start)[0])
        values.append(value)
        if value >= position_count:
            raise ValidationError(
                "index_out_of_range",
                "Primitive index references a vertex outside its POSITION accessor",
                details={"mesh_index": mesh_index, "primitive_index": primitive_index, "index_position": index, "index_value": value, "position_count": position_count},
            )
    return count, True, values


def _squared_distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum((left[axis] - right[axis]) ** 2 for axis in range(3))


def _is_nondegenerate_triangle(vertices: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]) -> bool:
    a, b, c = vertices
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    twice_area = math.sqrt(sum(component * component for component in cross))
    max_edge_squared = max(_squared_distance(a, b), _squared_distance(b, c), _squared_distance(c, a))
    if max_edge_squared <= 0 or not math.isfinite(twice_area) or not math.isfinite(max_edge_squared):
        return False
    return twice_area > (max_edge_squared * TRIANGLE_AREA_RELATIVE_EPSILON)


def _count_nondegenerate_faces(positions: list[tuple[float, float, float]], indices: list[int] | None) -> int:
    nondegenerate = 0
    if indices is not None:
        for offset in range(0, len(indices), 3):
            triangle = (positions[indices[offset]], positions[indices[offset + 1]], positions[indices[offset + 2]])
            if _is_nondegenerate_triangle(triangle):
                nondegenerate += 1
        return nondegenerate
    for offset in range(0, len(positions), 3):
        triangle = (positions[offset], positions[offset + 1], positions[offset + 2])
        if _is_nondegenerate_triangle(triangle):
            nondegenerate += 1
    return nondegenerate


def _position_geometry(document: dict[str, Any], bin_chunk: bytes | None) -> dict[str, Any]:
    meshes = document.get("meshes") or []
    accessors = document.get("accessors") or []
    buffer_views = document.get("bufferViews") or []
    if not meshes:
        raise ValidationError("missing_mesh", "GLB must contain at least one mesh")
    if bin_chunk is None:
        raise ValidationError("missing_bin_chunk", "GLB mesh validation requires a BIN chunk")

    positions: list[tuple[float, float, float]] = []
    position_primitive_count = 0
    face_count = 0
    nondegenerate_face_count = 0
    has_indices = False
    for mesh_index, mesh in enumerate(meshes):
        for primitive_index, primitive in enumerate(mesh.get("primitives") or []):
            if not isinstance(primitive, dict):
                raise ValidationError("invalid_primitive", "GLB primitive must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index})
            attributes = primitive.get("attributes") or {}
            if not isinstance(attributes, dict):
                raise ValidationError("invalid_primitive_attributes", "GLB primitive attributes must be an object", details={"mesh_index": mesh_index, "primitive_index": primitive_index})
            position_index = attributes.get("POSITION")
            if position_index is None:
                continue
            if not isinstance(position_index, int):
                raise ValidationError("missing_position_accessor", "POSITION references an invalid accessor", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "accessor": position_index})
            mode = primitive.get("mode", 4)
            if mode != 4:
                raise ValidationError("unsupported_primitive_mode", "Only TRIANGLES mode is supported for face-count validation", details={"mesh_index": mesh_index, "primitive_index": primitive_index, "mode": mode})
            primitive_positions = _validate_accessor_positions(
                mesh_index=mesh_index,
                primitive_index=primitive_index,
                position_index=position_index,
                accessors=accessors,
                buffer_views=buffer_views,
                bin_chunk=bin_chunk,
            )
            index_count, primitive_has_indices, index_values = _validate_index_values_for_primitive(
                primitive,
                accessors,
                buffer_views,
                bin_chunk,
                position_count=len(primitive_positions),
                mesh_index=mesh_index,
                primitive_index=primitive_index,
            )
            triangle_count_basis = index_count if primitive_has_indices else len(primitive_positions)
            if not primitive_has_indices and triangle_count_basis < 3:
                raise ValidationError(
                    "empty_triangle_primitive",
                    "Non-indexed TRIANGLES primitive must contain at least three POSITION vertices",
                    details={"mesh_index": mesh_index, "primitive_index": primitive_index, "count": triangle_count_basis},
                )
            if triangle_count_basis % 3 != 0:
                raise ValidationError(
                    "triangle_count_not_multiple_of_three",
                    "TRIANGLES primitive must have a vertex/index count that is a multiple of 3",
                    details={"mesh_index": mesh_index, "primitive_index": primitive_index, "count": triangle_count_basis, "indexed": primitive_has_indices},
                )
            has_indices = has_indices or primitive_has_indices
            face_count += triangle_count_basis // 3
            nondegenerate_face_count += _count_nondegenerate_faces(primitive_positions, index_values if primitive_has_indices else None)
            positions.extend(primitive_positions)
            position_primitive_count += 1
    if position_primitive_count <= 0:
        raise ValidationError("missing_position_accessor", "GLB must contain a mesh primitive with POSITION data")
    if face_count <= 0:
        raise ValidationError("missing_faces", "GLB must contain at least one validated TRIANGLES face")
    if nondegenerate_face_count <= 0:
        raise ValidationError(
            "missing_nondegenerate_faces",
            "GLB must contain at least one nondegenerate TRIANGLES face after index expansion",
            details={"face_count": face_count, "area_relative_epsilon": TRIANGLE_AREA_RELATIVE_EPSILON},
        )
    return {
        "positions": positions,
        "position_primitive_count": position_primitive_count,
        "face_count": face_count,
        "nondegenerate_face_count": nondegenerate_face_count,
        "area_relative_epsilon": TRIANGLE_AREA_RELATIVE_EPSILON,
        "has_indices": has_indices,
    }


def validate_glb(path: Path | str) -> dict[str, Any]:
    path = validate_regular_file(path, code="glb_missing")
    data = path.read_bytes()
    chunk_meta, bin_chunk = _read_glb_chunks(data)
    document = chunk_meta["json"]
    geometry = _position_geometry(document, bin_chunk)
    positions = geometry["positions"]
    mins = [min(vertex[axis] for vertex in positions) for axis in range(3)]
    maxs = [max(vertex[axis] for vertex in positions) for axis in range(3)]
    return {
        "path": str(path),
        "magic": "glTF",
        "version": 2,
        "size_bytes": len(data),
        "sha256": sha256_file(path),
        "chunk_count": chunk_meta["chunk_count"],
        "mesh_count": len(document.get("meshes") or []),
        "primitive_count": sum(len(mesh.get("primitives") or []) for mesh in document.get("meshes") or []),
        "position_primitive_count": geometry["position_primitive_count"],
        "vertex_count": len(positions),
        "face_count": geometry["face_count"],
        "nondegenerate_face_count": geometry["nondegenerate_face_count"],
        "area_relative_epsilon": geometry["area_relative_epsilon"],
        "bbox_min": [0 if abs(value) == 0 else value for value in mins],
        "bbox_max": [0 if abs(value) == 0 else value for value in maxs],
        "finite_geometry": True,
        "has_indices": geometry["has_indices"],
    }


def write_synthetic_glb_for_test(path: Path | str) -> Path:
    path = Path(path)
    positions = struct.pack("<fffffffff", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    json_doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(positions)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962}],
        "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_payload = json.dumps(json_doc, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((4 - len(json_payload) % 4) % 4)
    bin_payload = positions + (b"\x00" * ((4 - len(positions) % 4) % 4))
    total_length = 12 + 8 + len(json_payload) + 8 + len(bin_payload)
    glb = b"".join([
        struct.pack("<4sII", b"glTF", 2, total_length),
        struct.pack("<I4s", len(json_payload), b"JSON"),
        json_payload,
        struct.pack("<I4s", len(bin_payload), b"BIN\x00"),
        bin_payload,
    ])
    path.write_bytes(glb)
    return path


def _import_with_torch_compile_disabled(module_name: str) -> Any:
    import torch

    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return importlib.import_module(module_name)

    def identity_compile(function: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if function is None:
            return lambda inner: inner
        return function

    torch.compile = identity_compile
    try:
        return importlib.import_module(module_name)
    finally:
        torch.compile = original_compile


def validate_runtime(expected_torch: str, expected_torchvision: str, expected_cuda: str, expected_sm: str, expected_natten: str) -> dict[str, Any]:
    import torch
    from importlib.metadata import version

    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torchvision_version": version("torchvision"),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "imports": [],
        "upstream_imports": [],
    }
    if payload["torch_cuda_available"]:
        major, minor = torch.cuda.get_device_capability()
        payload["gpu_sm"] = f"{major}{minor}"
        payload["gpu_name"] = torch.cuda.get_device_name()

    errors: list[str] = []
    expected = {
        "torch_version": expected_torch,
        "torchvision_version": expected_torchvision,
        "torch_cuda_version": expected_cuda,
        "gpu_sm": expected_sm,
    }
    for key, expected_value in expected.items():
        if str(payload.get(key)) != str(expected_value):
            errors.append(f"{key} must be {expected_value}; found {payload.get(key)!r}")
    if payload["torch_cuda_available"] is not True:
        errors.append("torch CUDA must be available")

    for module_name in BLACKWELL_REQUIRED_IMPORTS:
        module = _import_with_torch_compile_disabled(module_name) if module_name == "natten" else importlib.import_module(module_name)
        payload["imports"].append(module_name)
        if module_name == "natten":
            payload["natten_version"] = version("natten")
            payload["natten_has_libnatten"] = bool(getattr(module, "HAS_LIBNATTEN", False))
    sys.modules.setdefault("cumesh", importlib.import_module("cumesh_vb"))
    sys.modules.setdefault("flex_gemm", importlib.import_module("flex_gemm_ap"))
    sys.modules.setdefault("o_voxel", importlib.import_module("o_voxel_vb_ap"))
    for module_name in BLACKWELL_REQUIRED_UPSTREAM_IMPORTS:
        importlib.import_module(module_name)
        payload["upstream_imports"].append(module_name)
    if payload.get("natten_version") != expected_natten:
        errors.append(f"natten_version must be {expected_natten}; found {payload.get('natten_version')!r}")
    if payload.get("natten_has_libnatten") is not True:
        errors.append("natten.HAS_LIBNATTEN must be True")
    payload["ok"] = not errors
    payload["validation_errors"] = errors
    if errors:
        raise ValidationError("runtime_contract_failed", "Blackwell runtime contract failed", details=payload)
    return payload


def _relative_after_shared_base(sentinel: str) -> Path:
    prefix = "models/pixal3d/_shared/pixal3d-base/"
    if not sentinel.startswith(prefix):
        raise ValidationError("unexpected_primary_sentinel", f"Unexpected primary sentinel path: {sentinel}")
    return Path(*sentinel[len(prefix):].split("/"))


def resolve_blackwell_weights_layout(authoritative_weights_path: Path | str) -> dict[str, Any]:
    root = Path(authoritative_weights_path).resolve()
    if (root / "pipeline.json").is_file():
        base_model_dir = root
        parts = root.parts
        suffix = ("models", "pixal3d", "_shared", "pixal3d-base")
        if tuple(parts[-4:]) == suffix:
            modly_home = Path(*parts[:-4]) if parts[:-4] else root.anchor
        else:
            modly_home = root.parent.parent.parent.parent if len(root.parents) >= 4 else root
    elif (root / "models" / "pixal3d" / "_shared" / "pixal3d-base" / "pipeline.json").is_file():
        modly_home = root
        base_model_dir = root / "models" / "pixal3d" / "_shared" / "pixal3d-base"
    else:
        raise ValidationError(
            "blackwell_weights_root_invalid",
            "ModlyWeightsPath must be either the Modly home or the pixal3d-base directory containing pipeline.json",
            details={"path": str(root)},
        )
    return {
        "authoritative_weights_path": str(root),
        "modly_home": str(Path(modly_home).resolve()),
        "base_model_dir": str(base_model_dir.resolve()),
        "generator_model_dir": str((Path(modly_home).resolve() / "models" / "pixal3d" / "generate")),
    }


def validate_blackwell_assets(authoritative_weights_path: Path | str) -> dict[str, Any]:
    layout = resolve_blackwell_weights_layout(authoritative_weights_path)
    base_model_dir = Path(layout["base_model_dir"])
    modly_home = Path(layout["modly_home"])
    checked: dict[str, Any] = {"primary": [], "auxiliary": {}}

    for sentinel in PRIMARY_ASSET.sentinel_paths:
        path = base_model_dir / _relative_after_shared_base(sentinel)
        validate_regular_file(path)
        checked["primary"].append({"logical": sentinel, "path": str(path), "size_bytes": path.stat().st_size})

    for key in ("dino", "rmbg", "moge"):
        manifest = AUXILIARY_ASSETS[key]
        asset_checked = []
        for sentinel in manifest.sentinel_paths:
            path = base_model_dir / _relative_after_shared_base(sentinel)
            validate_regular_file(path)
            asset_checked.append({"logical": sentinel, "path": str(path), "size_bytes": path.stat().st_size})
        checked["auxiliary"][key] = asset_checked

    naf = AUXILIARY_ASSETS["naf"]
    naf_entries = []
    for sentinel in naf.sentinel_paths:
        path = modly_home / Path(*sentinel.split("/"))
        validate_regular_file(path)
        verify_naf_checkpoint(path)
        naf_entries.append({"logical": sentinel, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    checked["auxiliary"]["naf"] = naf_entries
    return {"status": "passed", "layout": layout, "checked": checked, "auxiliary_mode": "strict", "network_available": False}


@contextmanager
def strict_offline_pipeline_patch(extension_dir: Path, modly_home: Path):
    sys.path.insert(0, str(extension_dir))
    import pixal3d_extension.pipeline_patch as pipeline_patch

    original_patch = pipeline_patch.patch_pipeline

    def strict_no_write_patch(workspace_root: str | Path, *, auxiliary_mode: str | None = "strict", network_available: bool | None = False) -> dict[str, Any]:
        result = pipeline_patch.validate_pipeline_patch(workspace_root, auxiliary_mode="strict", network_available=False)
        if result.get("status") != "ready":
            raise RuntimeError(json.dumps(result, sort_keys=True))
        return {
            "status": "patched",
            "code": "strict_offline_pipeline_already_ready",
            "auxiliary_mode": "strict",
            "network_available": False,
            "write_mode": "no_write_validation",
            "validation": result,
        }

    pipeline_patch.patch_pipeline = strict_no_write_patch
    try:
        yield
    finally:
        pipeline_patch.patch_pipeline = original_patch


def run_generation(extension_dir: Path, model_dir: Path, workspace_dir: Path, fixture_image: Path, output_dir: Path) -> dict[str, Any]:
    set_offline_environment()
    asset_gate = validate_blackwell_assets(model_dir)
    base_model_dir = Path(asset_gate["layout"]["base_model_dir"])
    modly_home = Path(asset_gate["layout"]["modly_home"])
    generator_model_dir = Path(asset_gate["layout"]["generator_model_dir"])

    sys.path.insert(0, str(extension_dir))
    from generator import Pixal3DGenerator

    generator = Pixal3DGenerator(model_dir=generator_model_dir, workspace_dir=workspace_dir)
    generator.MODEL_NODE_ID = "generate"
    generator.shared_model_dirs = {"pixal3d-base": str(base_model_dir)}
    generator.outputs_dir = output_dir
    params = {"low_vram": "low_vram", "resolution": 1024, "texture_size": 1024, "manual_fov": "-1", "seed": 1}
    progress: list[dict[str, Any]] = []
    start = time.time()
    with NetworkDenied() as network_denial, strict_offline_pipeline_patch(extension_dir, modly_home):
        network_denial_gate = network_denial.evidence()
        generator.load()
        glb_path = Path(generator.generate(fixture_image.read_bytes(), params, lambda pct, msg: progress.append({"pct": pct, "message": msg}), None))
    elapsed = time.time() - start
    generator.unload()
    logical_unload = {"status": "passed", "is_loaded": generator.is_loaded(), "scope": "logical_generator_unload_only"}
    if generator.is_loaded():
        raise ValidationError("unload_failed", "Generator remained logically loaded after unload()")
    return {
        "asset_gate": asset_gate,
        "network_denial": network_denial_gate,
        "output": validate_generated_glb_output(glb_path, output_dir=output_dir, workspace_dir=workspace_dir),
        "params": params,
        "progress_events": progress,
        "elapsed_seconds": round(elapsed, 3),
        "unload": logical_unload,
    }


def second_runtime_probe(venv_python: Path, extension_dir: Path, expected_torch: str, expected_torchvision: str, expected_cuda: str, expected_sm: str, expected_natten: str) -> dict[str, Any]:
    code = (
        "import importlib.util, json, pathlib\n"
        f"module_path = pathlib.Path({str(extension_dir / 'tools' / 'validation' / 'blackwell_real_generation.py')!r})\n"
        "spec = importlib.util.spec_from_file_location('blackwell_real_generation_child', module_path)\n"
        "helper = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(helper)\n"
        "helper.set_offline_environment()\n"
        "runtime = helper.validate_runtime("
        f"{expected_torch!r}, {expected_torchvision!r}, {expected_cuda!r}, {expected_sm!r}, {expected_natten!r})\n"
        "print(json.dumps({'status':'passed','scope':'separate child process-level cleanup proof','runtime_contract': runtime}, sort_keys=True))\n"
    )
    completed = subprocess.run([str(venv_python), "-c", code], cwd=extension_dir, text=True, capture_output=True)
    parsed: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            parsed = None
    result = {"returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:], "parsed": parsed}
    if completed.returncode != 0 or not parsed or parsed.get("status") != "passed":
        raise ValidationError("second_runtime_probe_failed", "Process-level restart/runtime validation failed", details=result)
    return {"status": "passed", **result, **parsed}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_validate(args: argparse.Namespace) -> int:
    evidence_dir = Path(args.evidence_dir).resolve()
    gates: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "schema_version": "pixal3d.blackwell.validation/v2",
        "status": "failed",
        "candidate": {
            "artifact_path": str(Path(args.candidate_artifact).resolve()),
            "expected_artifact_sha256": args.expected_artifact_sha256.lower(),
            "expected_artifact_size_bytes": int(args.expected_artifact_size_bytes),
            "wheelhouse_archive": str(Path(args.candidate_wheelhouse_archive).resolve()),
            "wheelhouse_sha256": args.candidate_wheelhouse_sha256.lower(),
            "wheelhouse_size_bytes": int(args.candidate_wheelhouse_size_bytes),
        },
        "run": {"source_commit": args.source_commit, "github_run_id": args.github_run_id, "github_run_url": args.github_run_url},
        "host": {"platform": sys.platform, "python": sys.version, "executable": sys.executable},
        "gates": gates,
        "cancellation": {"status": "not_supported_by_harness", "reason": "Manual hardware validation reports cancellation as untested; it does not fake a Modly cancel event."},
    }
    try:
        set_offline_environment()
        gates["network_denial_boundary"] = {"status": "configured", "class": "NetworkDenied", "offline_environment": OFFLINE_ENVIRONMENT}
        gates["artifact_contract"] = {"status": "passed", **validate_file_contract(args.candidate_artifact, args.expected_artifact_sha256, int(args.expected_artifact_size_bytes))}
        gates["wheelhouse_contract"] = {"status": "passed", **validate_file_contract(args.candidate_wheelhouse_archive, args.candidate_wheelhouse_sha256, int(args.candidate_wheelhouse_size_bytes))}
        for key, path in {"extension_dir": args.extension_dir, "workspace_dir": args.workspace_dir, "output_dir": args.output_dir}.items():
            assert_path_contained(path, args.work_root)
        gates["path_containment"] = {"status": "passed", "work_root": str(Path(args.work_root).resolve())}
        gates["asset_sentinels"] = validate_blackwell_assets(args.model_dir)
        gates["runtime"] = {"status": "passed", **validate_runtime(args.expected_torch, args.expected_torchvision, args.expected_cuda, args.expected_sm, args.expected_natten)}
        generation = run_generation(Path(args.extension_dir).resolve(), Path(args.model_dir).resolve(), Path(args.workspace_dir).resolve(), Path(args.fixture_image).resolve(), Path(args.output_dir).resolve())
        if generation["asset_gate"]["status"] != "passed":
            raise ValidationError("asset_gate_not_executed", "Generation did not execute the strict local asset gate", details={"asset_gate": generation["asset_gate"]})
        if generation["network_denial"]["status"] != "passed":
            raise ValidationError("network_denial_not_executed", "Generation did not execute the network denial gate", details={"network_denial": generation["network_denial"]})
        gates["generation"] = {"status": "passed", **generation}
        gates["restart_second_runtime"] = second_runtime_probe(Path(args.venv_python).resolve(), Path(args.extension_dir).resolve(), args.expected_torch, args.expected_torchvision, args.expected_cuda, args.expected_sm, args.expected_natten)
        evidence["status"] = "passed"
        return_code = 0
    except Exception as exc:  # noqa: BLE001 - evidence helper must persist all failures.
        if isinstance(exc, ValidationError):
            gates.setdefault(exc.code, exc.as_gate())
            evidence["failure_code"] = exc.code
            evidence["message"] = str(exc)
        else:
            evidence["failure_code"] = "unexpected_validation_error"
            evidence["message"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        evidence_path = evidence_dir / "blackwell-validation.json"
        write_json(evidence_path, evidence)
        print(json.dumps({"status": evidence["status"], "evidence": str(evidence_path), "failure_code": evidence.get("failure_code")}, sort_keys=True))
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Pixal3D Blackwell candidate on real Windows RTX 50 hardware.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-real")
    for name in [
        "candidate-artifact", "expected-artifact-sha256", "expected-artifact-size-bytes", "candidate-wheelhouse-archive",
        "candidate-wheelhouse-sha256", "candidate-wheelhouse-size-bytes", "extension-dir", "model-dir", "workspace-dir",
        "fixture-image", "output-dir", "evidence-dir", "work-root", "venv-python",
    ]:
        validate.add_argument(f"--{name}", required=True)
    validate.add_argument("--expected-torch", default="2.7.1+cu128")
    validate.add_argument("--expected-torchvision", default="0.22.1+cu128")
    validate.add_argument("--expected-cuda", default="12.8")
    validate.add_argument("--expected-sm", default="120")
    validate.add_argument("--expected-natten", default="0.21.6")
    validate.add_argument("--source-commit", default="")
    validate.add_argument("--github-run-id", default="")
    validate.add_argument("--github-run-url", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-real":
        return run_validate(args)
    parser.error(f"unsupported command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
