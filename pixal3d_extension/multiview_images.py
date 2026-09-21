"""Ordered Modly multiple-image adapter with offline DA3 camera estimation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from .scene_prepare import _run_worker, offline_environment, validate_da3_weights
from .scene_prepare_contract import validate_workspace_output_parent
from .scene_prepare_lane import python_path, validate_da3_runtime


MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
FILE_SHARE_READ = 0x00000001
FILE_ATTRIBUTE_DIRECTORY = 0x00000010
FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400


def _windows_native_path(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.abspath(value))


def _windows_stable_read(
    candidate: Path, workspace: Path, label: str, *, _opened_hook: Callable[[Path], None] | None = None,
) -> tuple[Path, bytes]:
    """Read through one locked Win32 handle and prove its final path custody."""

    import ctypes
    from ctypes import wintypes

    GENERIC_READ = 0x80000000
    OPEN_EXISTING = 3
    FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", FILETIME),
            ("ftLastAccessTime", FILETIME),
            ("ftLastWriteTime", FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandle
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(BY_HANDLE_FILE_INFORMATION)]
    get_info.restype = wintypes.BOOL
    get_final_path = kernel32.GetFinalPathNameByHandleW
    get_final_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_final_path.restype = wintypes.DWORD
    read_file = kernel32.ReadFile
    read_file.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    read_file.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    # Share only reads. Existing or new writers/deleters/renamers conflict with
    # this handle, so metadata, path custody, and bytes refer to one stable file.
    handle = create_file(
        str(candidate), GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise OSError(ctypes.get_last_error(), f"{label} cannot be opened as a locked regular file")
    try:
        if _opened_hook is not None:
            _opened_hook(candidate)
        info = BY_HANDLE_FILE_INFORMATION()
        if not get_info(handle, ctypes.byref(info)):
            raise OSError(ctypes.get_last_error(), f"{label} handle metadata cannot be read")
        if info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"{label} must not be a Windows reparse point")
        if info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY:
            raise ValueError(f"{label} must be a regular file")
        size = (int(info.nFileSizeHigh) << 32) | int(info.nFileSizeLow)
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"{label} must be no larger than 64 MiB")

        capacity = 32768
        buffer = ctypes.create_unicode_buffer(capacity)
        length = get_final_path(handle, buffer, capacity, 0)
        if not length or length >= capacity:
            raise OSError(ctypes.get_last_error(), f"{label} final handle path cannot be resolved")
        final_path = _windows_native_path(buffer.value)
        lexical_path = _windows_native_path(str(candidate))
        workspace_path = _windows_native_path(str(workspace))
        try:
            inside = os.path.commonpath([workspace_path, final_path]) == workspace_path
        except ValueError:
            inside = False
        if not inside:
            raise ValueError(f"{label} final handle path escapes the Modly workspace")
        if final_path != lexical_path:
            raise ValueError(f"{label} must not traverse a Windows reparse point")

        chunks: list[bytes] = []
        remaining = size
        while remaining:
            request = min(1024 * 1024, remaining)
            chunk = ctypes.create_string_buffer(request)
            count = wintypes.DWORD()
            if not read_file(handle, chunk, request, ctypes.byref(count), None):
                raise OSError(ctypes.get_last_error(), f"{label} locked handle read failed")
            if count.value == 0:
                raise OSError(f"{label} changed size during locked handle read")
            chunks.append(chunk.raw[:count.value])
            remaining -= count.value
        return Path(final_path), b"".join(chunks)
    finally:
        close_handle(handle)


@dataclass(frozen=True)
class ValidatedImage:
    data: bytes
    format: str
    width: int
    height: int
    digest: str


def _check_cancel(cancel_event: Any | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Pixal3D MV generation cancelled")


def _decode(data: bytes, label: str) -> ValidatedImage:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{label} must be a nonempty supported image no larger than 64 MiB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} must contain a supported image") from exc
    if image_format not in SUPPORTED_FORMATS or width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"{label} must contain a supported image (PNG, JPEG, or WebP) within safe dimensions")
    return ValidatedImage(data, image_format, width, height, hashlib.sha256(data).hexdigest())


def _workspace_file(raw: object, workspace: Path, label: str) -> tuple[Path, bytes]:
    if not isinstance(raw, str) or not raw or raw != raw.strip() or "\x00" in raw:
        raise TypeError(f"{label} must be a nonempty workspace image path")
    parsed = Path(raw)
    if PureWindowsPath(raw).is_absolute() and os.name != "nt":
        raise ValueError(f"{label} must be a native workspace path")
    if any(part in {".", ".."} for part in parsed.parts):
        raise ValueError(f"{label} must not contain traversal segments")
    candidate = parsed if parsed.is_absolute() else workspace / parsed
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside the Modly workspace") from exc
    if os.name == "nt":
        try:
            return _windows_stable_read(candidate, workspace, label)
        except OSError as exc:
            raise ValueError(f"{label} must remain a locked regular file inside the Modly workspace") from exc

    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not use symlinks")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError(f"{label} must be a regular file inside the Modly workspace")

    # Open every component relative to an already-open workspace descriptor.
    # O_NOFOLLOW closes the symlink-swap window between custody validation and
    # the immutable byte snapshot used by staging.
    directory_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
    opened = [directory_fd]
    try:
        for part in relative.parts[:-1]:
            directory_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
            opened.append(directory_fd)
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"{label} must be a regular file inside the Modly workspace")
            chunks = []
            total = 0
            while True:
                chunk = os.read(file_fd, min(1024 * 1024, MAX_IMAGE_BYTES + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ValueError(f"{label} must be no larger than 64 MiB")
            return resolved, b"".join(chunks)
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise ValueError(f"{label} must remain a regular non-symlink file inside the Modly workspace") from exc
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def validate_ordered_images(
    primary_image_bytes: bytes, extra_image_paths: object, workspace_dir: str | Path,
    *, max_connected: int = 3, purpose: str = "Pixal3D MV",
) -> list[ValidatedImage]:
    workspace = Path(workspace_dir).resolve(strict=True)
    if not isinstance(extra_image_paths, list):
        raise TypeError("extra_image_paths must be an ordered list supplied by Modly")
    if len(extra_image_paths) > max_connected:
        words = {4: "four", 8: "eight"}
        total = max_connected + 1
        raise ValueError(f"{purpose} accepts at most {words.get(total, total)} connected images")
    connected_paths: list[tuple[int, str]] = []
    for index, value in enumerate(extra_image_paths, start=2):
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"view {index} must be None or a workspace image path")
        connected_paths.append((index, value))
    if len(connected_paths) < 1:
        raise ValueError(f"{purpose} requires at least two connected images")
    snapshots = [
        (index, *_workspace_file(value, workspace, f"view {index}"))
        for index, value in connected_paths
    ]
    paths = [item[1] for item in snapshots]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{purpose} extra image paths must not contain duplicate or repeated paths")
    images = [_decode(bytes(primary_image_bytes), "primary view")]
    for index, _path, data in snapshots:
        images.append(_decode(data, f"view {index}"))
    digests = [image.digest for image in images]
    if len(set(digests)) != len(digests):
        raise ValueError(f"{purpose} connected images must not contain duplicate content")
    dimensions = {(image.width, image.height) for image in images}
    if len(dimensions) != 1:
        raise ValueError(f"{purpose} connected images must have matching dimensions")
    return images


def _stage_pngs(images: list[ValidatedImage], target: Path) -> list[Path]:
    paths = []
    for index, item in enumerate(images):
        destination = target / f"{index:04d}.png"
        with Image.open(io.BytesIO(item.data)) as image:
            mode = "RGBA" if "A" in image.getbands() else "RGB"
            image.convert(mode).save(destination, format="PNG")
        paths.append(destination)
    return paths


def _estimate_cameras(
    frame_paths: list[Path], da3_root: Path, staging: Path, extension_root: Path,
    *, progress_cb=None, cancel_event=None,
) -> Path:
    job = {
        "schema": "modly.pixal3d-mv-camera-job.v1",
        "frame_paths": [str(path) for path in frame_paths],
        "da3_root": str(Path(da3_root).resolve(strict=True)),
        "output_path": str(staging / "transforms.json"),
        "process_resolution": 504,
    }
    handle = tempfile.NamedTemporaryFile(
        "w", prefix=".mv-camera-job-", suffix=".json", dir=staging, delete=False, encoding="utf-8"
    )
    job_path = Path(handle.name)
    try:
        json.dump(job, handle, sort_keys=True)
        handle.close()
        result = _run_worker(
            [str(python_path(extension_root)), "-m", "pixal3d_extension.multiview_camera_worker", str(job_path)],
            cwd=extension_root,
            env=offline_environment(staging / ".da3-cache", extension_root),
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            output_field="transforms_path",
            label="Pixal3D MV camera calibration",
        )
        if result != staging / "transforms.json" or not result.is_file():
            raise RuntimeError("Pixal3D MV camera calibration did not produce private transforms")
        return result
    finally:
        try:
            handle.close()
        except Exception:
            pass
        job_path.unlink(missing_ok=True)


def run_multiview_from_images(
    *, primary_image_bytes: bytes, extra_image_paths: list[str], workspace_dir: str | Path,
    output_dir: str | Path, da3_root: str | Path, mv_root: str | Path, base_root: str | Path,
    naf_path: str | Path, params: dict[str, Any], progress_cb: Callable[[int, str], None] | None = None,
    cancel_event: Any | None = None, inference_runner: Callable[..., Any] | None = None,
) -> Path:
    """Estimate real ordered cameras, then reuse the calibrated MV cascade."""

    _check_cancel(cancel_event)
    images = validate_ordered_images(primary_image_bytes, extra_image_paths, workspace_dir)
    _check_cancel(cancel_event)
    validate_da3_weights(Path(da3_root))
    extension_root = Path(__file__).resolve().parent.parent
    validate_da3_runtime(extension_root)
    _check_cancel(cancel_event)
    workspace = Path(workspace_dir).resolve(strict=True)
    output = validate_workspace_output_parent(Path(output_dir), workspace, "MV output directory")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pixal3d-mv-images-", dir=output) as directory:
        staging = Path(directory)
        frames = _stage_pngs(images, staging)
        _check_cancel(cancel_event)
        if progress_cb:
            progress_cb(3, "Estimating consistent cameras with DA3 Base")
        _estimate_cameras(
            frames, Path(da3_root), staging, extension_root,
            progress_cb=progress_cb, cancel_event=cancel_event,
        )
        (staging / "scene-manifest.json").write_text(json.dumps({
            "schema": "modly.scene-manifest.v1", "sceneRoot": ".",
            "provenance": {"source": "ordered Modly image ports", "cameraEstimator": "DA3 Base"},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _check_cancel(cancel_event)
        from .multiview import run_multiview

        internal_params = dict(params)
        internal_params["num_views"] = len(images)
        return run_multiview(
            scene_manifest_path=staging / "scene-manifest.json", workspace_dir=workspace,
            mv_root=mv_root, base_root=base_root, naf_path=naf_path, output_dir=output,
            params=internal_params, inference_runner=inference_runner,
            progress_cb=progress_cb, cancel_event=cancel_event,
        )
