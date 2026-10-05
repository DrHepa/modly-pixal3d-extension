"""Isolated future Modly video-envelope parsing and workspace custody."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PureWindowsPath


MAX_VIDEO_BYTES = 512 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"}


def parse_video_input(value) -> Path:
    """Parse the future host envelope without coupling the runtime to host classes."""
    if isinstance(value, dict):
        kind, path = value.get("kind"), value.get("path")
    else:
        kind, path = getattr(value, "kind", None), getattr(value, "path", None)
    if kind != "video":
        if kind is None:
            raise TypeError("Scene video input must be a typed video envelope with kind and path")
        raise ValueError("Scene video input envelope must declare kind=video")
    if not isinstance(path, (str, os.PathLike)) or not os.fspath(path):
        raise TypeError("Scene video input envelope path must be a nonempty local path")
    return Path(path)


def _recognized_video(header: bytes, extension: str) -> bool:
    if extension in {".mp4", ".m4v", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if extension in {".mkv", ".webm"}:
        return header.startswith(b"\x1a\x45\xdf\xa3")
    if extension == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    return False


def _copy_stream(source, target, cancel_event=None) -> None:
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Scene video preparation cancelled")
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        target.write(chunk)


def stage_video_input(value, workspace_dir: Path, destination_dir: Path, cancel_event=None) -> Path:
    """Copy one stable, workspace-owned video snapshot into private staging."""
    workspace = Path(workspace_dir).resolve(strict=True)
    raw = parse_video_input(value)
    raw_text = os.fspath(raw)
    if "\x00" in raw_text or PureWindowsPath(raw_text).is_absolute() and os.name != "nt":
        raise ValueError("Scene video path must be a native workspace path")
    if any(part in {".", ".."} for part in raw.parts):
        raise ValueError("Scene video path must not contain traversal segments")
    candidate = raw if raw.is_absolute() else workspace / raw
    try:
        relative = candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("Scene video must remain inside the Modly workspace") from exc
    extension = candidate.suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise ValueError("Scene video extension must be MP4, M4V, MOV, MKV, WebM, or AVI")
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Scene video must not use symlinks or reparse points")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(workspace) or not resolved.is_file():
        raise ValueError("Scene video must be a regular file inside the Modly workspace")
    size = resolved.stat().st_size
    if size <= 0 or size > MAX_VIDEO_BYTES:
        raise ValueError("Scene video must be nonempty and no larger than 512 MiB")
    destination = Path(destination_dir) / ("capture" + extension)
    if os.name == "nt":
        with resolved.open("rb") as source:
            header = source.read(16)
            if not _recognized_video(header, extension):
                raise ValueError("Scene video content does not match its supported extension")
            source.seek(0)
            with destination.open("xb") as target:
                _copy_stream(source, target, cancel_event)
    else:
        directory_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
        opened = [directory_fd]
        try:
            for part in relative.parts[:-1]:
                directory_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                opened.append(directory_fd)
            file_fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size != size:
                    raise ValueError("Scene video changed during custody validation")
                header = os.read(file_fd, 16)
                if not _recognized_video(header, extension):
                    raise ValueError("Scene video content does not match its supported extension")
                os.lseek(file_fd, 0, os.SEEK_SET)
                with os.fdopen(os.dup(file_fd), "rb") as source, destination.open("xb") as target:
                    _copy_stream(source, target, cancel_event)
                if os.fstat(file_fd).st_size != size:
                    raise ValueError("Scene video changed during custody snapshot")
            finally:
                os.close(file_fd)
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)
    if destination.stat().st_size != size:
        destination.unlink(missing_ok=True)
        raise ValueError("Scene video snapshot size mismatch")
    return destination
