"""Primary-venv orchestration for isolated SAM3 + DA3 scene preparation."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

from .scene_prepare_contract import load_capture_manifest, validate_workspace_output_parent
from .scene_prepare_lane import python_path, validate_runtime
from .process_tree import popen_process_group_kwargs, terminate_process_tree
from .worldsculpt import resolve_scene_manifest
from .worldsculpt_contract import validate_scene
from .scene_video_input import parse_video_input, stage_video_input


SAM3_GROUP = "sam3"
DA3_GROUP = "da3-base"
SAM3_FILES = ("config.json", "sam3.pt", "LICENSE")
DA3_FILES = ("config.json", "model.safetensors")


def _regular(root: Path, relative: str, label: str) -> Path:
    root = Path(root).resolve(strict=True)
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise FileNotFoundError(f"{label} missing; download or repair the shared group in Modly Models UI: {relative}")
    return path


def validate_scene_prepare_weights(sam_root: Path, da3_root: Path) -> None:
    for relative in SAM3_FILES:
        _regular(sam_root, relative, "SAM3 gated weights")
    for relative in DA3_FILES:
        _regular(da3_root, relative, "DA3 Base weights")


def validate_da3_weights(da3_root: Path) -> None:
    """Validate the UI-managed DA3 subset used by MV camera calibration."""
    for relative in DA3_FILES:
        _regular(da3_root, relative, "DA3 Base weights")


def offline_environment(cache_root: Path, package_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env.pop(key, None)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(cache_root),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hub"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
    })
    if package_root is not None:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(Path(package_root).resolve()) + (os.pathsep + existing if existing else "")
    return env


def worker_command(extension_root: Path, job_path: Path) -> list[str]:
    root = Path(extension_root)
    return [str(python_path(root)), "-m", "pixal3d_extension.scene_prepare_worker", str(job_path)]


def _typed_path(value, expected_kind: str) -> Path:
    kind = getattr(value, "kind", None)
    path = getattr(value, "path", value)
    if kind is not None and kind != expected_kind:
        raise ValueError(f"Expected {expected_kind} typed input, got {kind}")
    if not isinstance(path, (str, os.PathLike, Path)):
        raise TypeError(f"{expected_kind} input must contain a filesystem path")
    return Path(path)


def _run_worker(
    command: list[str], *, cwd: Path, env: dict[str, str], progress_cb=None,
    cancel_event=None, output_field: str = "scene_manifest_path", label: str = "Scene preparation",
) -> Path:
    process = subprocess.Popen(
        command,
        cwd=Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        **popen_process_group_kwargs(),
    )
    assert process.stdout is not None and process.stderr is not None
    chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue()

    def read_chunks(name: str, stream) -> None:
        try:
            while True:
                chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                if not chunk:
                    return
                chunks.put((name, chunk))
        finally:
            chunks.put((name, None))

    readers = [
        threading.Thread(target=read_chunks, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=read_chunks, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    stderr_tail: list[str] = []
    output: Path | None = None
    buffers = {"stdout": b"", "stderr": b""}
    active = {"stdout", "stderr"}

    def consume(name: str, raw: bytes) -> None:
        nonlocal output
        line = raw.decode("utf-8", errors="replace").rstrip("\r")
        if name == "stderr":
            stderr_tail.append(line)
            stderr_tail[:] = stderr_tail[-80:]
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            stderr_tail.append(line)
            stderr_tail[:] = stderr_tail[-80:]
            return
        if message.get("type") == "progress" and progress_cb:
            progress_cb(int(message.get("pct", 0)), str(message.get("step", "")))
        elif message.get("type") == "done":
            output = Path(message[output_field])
        elif message.get("type") == "error":
            raise RuntimeError(str(message.get("message", f"{label} worker failed")))

    try:
        while active or process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                terminate_process_tree(process)
                raise RuntimeError(f"{label} cancelled")
            try:
                name, chunk = chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            if chunk is None:
                active.discard(name)
                if buffers[name]:
                    consume(name, buffers[name])
                    buffers[name] = b""
                continue
            buffers[name] += chunk
            while b"\n" in buffers[name]:
                raw, buffers[name] = buffers[name].split(b"\n", 1)
                consume(name, raw)
        if process.wait() != 0 or output is None:
            raise RuntimeError(f"{label} worker failed: " + "\n".join(stderr_tail[-20:]))
        return output
    finally:
        terminate_process_tree(process, force=process.poll() is None)
        for reader in readers:
            reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()


def _run_scene_from_capture(
    *,
    capture_manifest: Path,
    workspace_dir: Path,
    output_dir: Path,
    sam_root: Path,
    da3_root: Path,
    params: dict,
    progress_cb=None,
    cancel_event=None,
) -> Path:
    workspace = Path(workspace_dir).resolve(strict=True)
    capture_manifest = Path(capture_manifest).resolve(strict=True)
    load_capture_manifest(capture_manifest, workspace)
    destination = validate_workspace_output_parent(output_dir, workspace, "scene-prep output directory")
    validate_scene_prepare_weights(sam_root, da3_root)
    extension_root = Path(__file__).resolve().parent.parent
    validate_runtime(extension_root)
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve(strict=True)
    if not destination.is_relative_to(workspace):
        raise ValueError("Scene-prep output directory must be inside the workspace")
    cache = destination / ".scene-prep-cache"
    cache.mkdir(exist_ok=True)
    job = {
        "schema": "modly.scene-prep-job.v1",
        "capture_manifest_path": str(capture_manifest),
        "workspace_dir": str(workspace),
        "output_dir": str(destination),
        "sam_root": str(Path(sam_root).resolve(strict=True)),
        "da3_root": str(Path(da3_root).resolve(strict=True)),
        "params": dict(params),
    }
    handle = tempfile.NamedTemporaryFile("w", prefix=".scene-prep-job-", suffix=".json", dir=destination, delete=False, encoding="utf-8")
    job_path = Path(handle.name)
    try:
        json.dump(job, handle, sort_keys=True)
        handle.close()
        result = _run_worker(
            worker_command(extension_root, job_path),
            cwd=extension_root,
            env=offline_environment(cache, extension_root),
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
        scene = resolve_scene_manifest(result, workspace)
        validate_scene(scene)
        return result
    finally:
        try:
            handle.close()
        except Exception:
            pass
        job_path.unlink(missing_ok=True)


def run_scene_from_estimates(
    *, capture_input, workspace_dir: Path, output_dir: Path, sam_root: Path,
    da3_root: Path, params: dict, progress_cb=None, cancel_event=None,
) -> Path:
    """Private backward-compatible capture adapter; not a public manifest node."""
    capture_manifest = _typed_path(capture_input, "capture")
    return _run_scene_from_capture(
        capture_manifest=capture_manifest, workspace_dir=workspace_dir, output_dir=output_dir,
        sam_root=sam_root, da3_root=da3_root, params=params,
        progress_cb=progress_cb, cancel_event=cancel_event,
    )


def run_scene_from_images(
    *, primary_image_bytes: bytes, extra_image_paths: list, workspace_dir: Path,
    output_dir: Path, sam_root: Path, da3_root: Path, params: dict,
    progress_cb=None, cancel_event=None,
) -> Path:
    """Adapt eight fixed ordered Modly image ports to the private capture ABI."""
    from .multiview_images import _stage_pngs, validate_ordered_images

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Scene preparation cancelled")
    workspace = Path(workspace_dir).resolve(strict=True)
    images = validate_ordered_images(
        primary_image_bytes, extra_image_paths, workspace,
        max_connected=7, purpose="Scene preparation",
    )
    output = validate_workspace_output_parent(output_dir, workspace, "scene-prep output directory")
    output.mkdir(parents=True, exist_ok=True)
    connected_ports = [1, *[index for index, value in enumerate(extra_image_paths, start=2) if value is not None]]
    with tempfile.TemporaryDirectory(prefix=".scene-images-", dir=output) as directory:
        staging = Path(directory)
        frame_paths = _stage_pngs(images, staging)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Scene preparation cancelled")
        frames = [
            {
                "index": index,
                "path": path.name,
                "width": image.width,
                "height": image.height,
                "byteSize": path.stat().st_size,
            }
            for index, (path, image) in enumerate(zip(frame_paths, images, strict=True))
        ]
        manifest = staging / "capture-manifest.json"
        manifest.write_text(json.dumps({
            "schema": "modly.capture-manifest.v1",
            "captureRoot": ".",
            "kind": "frames",
            "frames": frames,
            "provenance": {
                "source": "ordered Modly image ports",
                "ordering": "manifest-index",
                "sourcePorts": connected_ports,
            },
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        internal_params = dict(params)
        internal_params.update({"max_frames": len(images), "frame_stride": 1})
        return _run_scene_from_capture(
            capture_manifest=manifest, workspace_dir=workspace, output_dir=output,
            sam_root=sam_root, da3_root=da3_root, params=internal_params,
            progress_cb=progress_cb, cancel_event=cancel_event,
        )


def _probe_video(staged_video: Path, extension_root: Path, progress_cb=None, cancel_event=None) -> dict:
    if progress_cb:
        progress_cb(2, "Validating and decoding scene video")
    metadata_path = _run_worker(
        [str(python_path(extension_root)), "-m", "pixal3d_extension.scene_video_probe_worker", str(staged_video)],
        cwd=extension_root,
        env=offline_environment(staged_video.parent / ".scene-prep-cache", extension_root),
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        output_field="metadata_path",
        label="Scene video validation",
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    finally:
        metadata_path.unlink(missing_ok=True)
    if not isinstance(metadata, dict) or any(type(metadata.get(key)) is not int or metadata[key] <= 0 for key in ("width", "height", "frameCount")):
        raise RuntimeError("Scene video validation returned invalid metadata")
    return metadata


def run_scene_from_video(
    *, video_input, workspace_dir: Path, output_dir: Path, sam_root: Path,
    da3_root: Path, params: dict, progress_cb=None, cancel_event=None,
) -> Path:
    """Validate one typed local video, sample it deterministically, and prepare a scene."""
    workspace = Path(workspace_dir).resolve(strict=True)
    output = validate_workspace_output_parent(output_dir, workspace, "scene-prep output directory")
    output.mkdir(parents=True, exist_ok=True)
    extension_root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix=".scene-video-", dir=output) as directory:
        staging = Path(directory)
        staged_video = stage_video_input(video_input, workspace, staging, cancel_event)
        validate_scene_prepare_weights(sam_root, da3_root)
        validate_runtime(extension_root)
        metadata = _probe_video(staged_video, extension_root, progress_cb, cancel_event)
        manifest = staging / "capture-manifest.json"
        manifest.write_text(json.dumps({
            "schema": "modly.capture-manifest.v1",
            "captureRoot": ".",
            "kind": "video",
            "video": {
                "path": staged_video.name,
                "width": metadata["width"],
                "height": metadata["height"],
                "frameCount": metadata["frameCount"],
                "byteSize": staged_video.stat().st_size,
            },
            "provenance": {"source": "typed Modly video input", "ordering": "decode-index"},
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return _run_scene_from_capture(
            capture_manifest=manifest, workspace_dir=workspace, output_dir=output,
            sam_root=sam_root, da3_root=da3_root, params=dict(params),
            progress_cb=progress_cb, cancel_event=cancel_event,
        )
