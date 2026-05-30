from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable


HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_SELECTOR_KEYS = {"os", "arch", "python_tag", "accelerator_lane"}
DownloadFn = Callable[[str, Path], int | None]


class WheelhouseError(RuntimeError):
    def __init__(self, code: str, message: str, *, observation: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.observation = {
            "status": "failed",
            "failure_code": code,
            "downloads_started": False,
            "installs_started": False,
            **(observation or {}),
        }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise WheelhouseError(code, message)


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _safe_join(root: Path, relative_path: str) -> Path:
    if not _is_safe_relative_path(relative_path):
        raise WheelhouseError("unsafe_archive_path", "Archive member escapes the wheelhouse cache")
    destination = root / relative_path
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise WheelhouseError("unsafe_archive_path", "Archive member escapes the wheelhouse cache") from exc
    return destination


def _validate_hash(value: Any, code: str = "missing_hash") -> None:
    _require(isinstance(value, str) and bool(HEX_SHA256.fullmatch(value)), code, "Expected a lowercase sha256 hex digest")


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("schema_version") == "release-wheelhouse/v1", "invalid_schema", "Unsupported wheelhouse manifest schema")
    _require(manifest.get("extension_id") == "pixal3d", "invalid_extension", "Manifest does not describe Pixal3D")
    _require(isinstance(manifest.get("wheelhouse_version"), str) and manifest["wheelhouse_version"], "missing_version", "Missing wheelhouse version")

    release = manifest.get("release") or {}
    tag = release.get("tag")
    _require(isinstance(tag, str) and tag and tag != "latest", "mutable_release_tag", "Release tag must be pinned and not latest")
    _require(bool(HEX_COMMIT.fullmatch(str(release.get("immutable_commit", "")))), "missing_immutable_commit", "Missing immutable release commit")

    cache = manifest.get("cache") or {}
    _require(_is_safe_relative_path(str(cache.get("root", ""))), "unsafe_cache_path", "Cache root must stay relative to the extension")
    _require("{asset_id}" in str(cache.get("key_template", "")), "invalid_cache_key_template", "Cache key template must include asset_id")

    assets = manifest.get("assets")
    _require(isinstance(assets, list) and bool(assets), "missing_assets", "At least one asset lane is required")
    seen_ids: set[str] = set()
    for asset in assets:
        asset_id = asset.get("id")
        _require(isinstance(asset_id, str) and asset_id and asset_id not in seen_ids, "invalid_asset_id", "Asset ids must be unique")
        seen_ids.add(asset_id)
        _require(_is_safe_relative_path(str(asset.get("filename", ""))), "unsafe_asset_path", "Asset filename must be a safe relative path")
        _require(isinstance(asset.get("size_bytes"), int) and asset["size_bytes"] > 0, "invalid_asset_size", "Asset size must be positive")
        _validate_hash(asset.get("sha256"))
        selectors = asset.get("selectors") or {}
        _require(REQUIRED_SELECTOR_KEYS <= set(selectors), "missing_selectors", "Asset selectors are incomplete")

    fallback = manifest.get("fallback") or {}
    _require(_is_safe_relative_path(str(fallback.get("vendored_wheels", ""))), "unsafe_fallback_path", "Vendored wheels path must stay relative")
    _require(fallback.get("require_hashes") is True, "fallback_hashes_required", "Fallback wheels must require hashes")
    wheels = fallback.get("wheels") or []
    _require(isinstance(wheels, list) and bool(wheels), "missing_fallback_hashes", "Fallback wheels must list hashes")
    for wheel in wheels:
        _require(_is_safe_relative_path(str(wheel.get("filename", ""))), "unsafe_fallback_wheel", "Fallback wheel filename must be safe")
        _require(isinstance(wheel.get("size_bytes"), int) and wheel["size_bytes"] > 0, "invalid_fallback_size", "Fallback wheel size must be positive")
        _validate_hash(wheel.get("sha256"), "missing_fallback_hash")


def _cache_key(manifest: dict[str, Any], asset: dict[str, Any]) -> str:
    template = manifest["cache"]["key_template"]
    return template.format(
        extension_id=manifest["extension_id"],
        wheelhouse_version=manifest["wheelhouse_version"],
        asset_id=asset["id"],
    )


def _asset_url(manifest: dict[str, Any], asset: dict[str, Any]) -> str:
    explicit_url = asset.get("url")
    if isinstance(explicit_url, str) and explicit_url:
        return explicit_url
    release = manifest["release"]
    return "https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}".format(
        owner=release["owner"],
        repo=release["repo"],
        tag=release["tag"],
        filename=asset["filename"],
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected_sha256: str, mismatch_code: str = "checksum_mismatch") -> None:
    if _sha256_file(path) != expected_sha256:
        raise WheelhouseError(mismatch_code, "sha256 verification failed")


def _download_url(url: str, destination: Path) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
    except urllib.error.HTTPError as exc:
        code = "auth_required" if exc.code in {401, 403} else "network"
        raise WheelhouseError(code, f"GitHub release download failed with HTTP {exc.code}") from exc
    except OSError as exc:
        raise WheelhouseError("network", "GitHub release download failed") from exc
    return destination.stat().st_size


def _cache_paths(manifest: dict[str, Any], asset: dict[str, Any], workspace_root: Path) -> dict[str, Path]:
    cache_root = workspace_root / manifest["cache"]["root"]
    key = asset.get("cache_key") or _cache_key(manifest, asset)
    if not _is_safe_relative_path(key):
        raise WheelhouseError("unsafe_cache_path", "Cache key must stay relative to the extension")
    asset_root = cache_root / key
    suffix = Path(asset["filename"]).suffix or ".archive"
    return {
        "root": asset_root,
        "archive": asset_root / f"archive{suffix}",
        "extracted": asset_root / "extracted",
        "marker": asset_root / ".modly-wheelhouse.json",
    }


def _read_cache_marker(marker_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache_marker(marker_path: Path, manifest: dict[str, Any], asset: dict[str, Any]) -> None:
    marker_path.write_text(
        json.dumps(
            {
                "asset_id": asset["id"],
                "sha256": asset["sha256"],
                "release_tag": manifest["release"]["tag"],
                "wheelhouse_version": manifest["wheelhouse_version"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _verified_extracted_cache(paths: dict[str, Path], asset: dict[str, Any]) -> bool:
    marker = _read_cache_marker(paths["marker"])
    return bool(
        paths["extracted"].is_dir()
        and marker
        and marker.get("asset_id") == asset["id"]
        and marker.get("sha256") == asset["sha256"]
    )


def _extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_join(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_tar(archive_path: Path, destination: Path, mode: str) -> None:
    with tarfile.open(archive_path, mode) as archive:
        for member in archive.getmembers():
            target = _safe_join(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _extract_archive(archive_path: Path, destination: Path, compression: str) -> None:
    if compression == "zip":
        _extract_zip(archive_path, destination)
        return
    if compression in {"tar", "tar.gz", "tgz"}:
        mode = "r:gz" if compression in {"tar.gz", "tgz"} else "r:"
        _extract_tar(archive_path, destination, mode)
        return
    if compression == "tar.zst":
        raise WheelhouseError("unsupported_compression", "tar.zst extraction requires the release runtime extractor")
    raise WheelhouseError("unsupported_compression", f"Unsupported wheelhouse archive compression: {compression}")


def _ready_observation(
    manifest: dict[str, Any],
    asset: dict[str, Any],
    paths: dict[str, Path],
    *,
    cache_hit: bool,
    downloaded: bool,
    downloads_started: bool,
    bytes_downloaded: int,
) -> dict[str, Any]:
    return {
        "status": "ready",
        "selected_asset": asset["id"],
        "cache_hit": cache_hit,
        "downloaded": downloaded,
        "sha256_verified": True,
        "bytes_downloaded": bytes_downloaded,
        "release_tag": manifest["release"]["tag"],
        "downloads_started": downloads_started,
        "installs_started": False,
        "wheelhouse_path": str(paths["extracted"]),
    }


def select_asset(manifest: dict[str, Any], runtime_evidence: dict[str, str]) -> dict[str, Any]:
    matches = [
        asset
        for asset in manifest.get("assets", [])
        if all(asset.get("selectors", {}).get(key) == runtime_evidence.get(key) for key in REQUIRED_SELECTOR_KEYS)
    ]
    if not matches:
        raise WheelhouseError("unsupported_lane", "No wheelhouse asset matches this runtime lane")
    if len(matches) > 1:
        raise WheelhouseError("ambiguous_lane", "More than one wheelhouse asset matches this runtime lane")
    selected = dict(matches[0])
    selected["cache_key"] = _cache_key(manifest, selected)
    return selected


def prepare_wheelhouse(
    manifest: dict[str, Any],
    runtime_evidence: dict[str, str],
    workspace_root: Path,
    *,
    downloader: DownloadFn | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest)
    asset = select_asset(manifest, runtime_evidence)
    paths = _cache_paths(manifest, asset, workspace_root)
    paths["root"].mkdir(parents=True, exist_ok=True)

    if _verified_extracted_cache(paths, asset):
        return _ready_observation(
            manifest,
            asset,
            paths,
            cache_hit=True,
            downloaded=False,
            downloads_started=False,
            bytes_downloaded=0,
        )

    if paths["archive"].exists():
        _verify_file(paths["archive"], asset["sha256"])
        _extract_to_cache(paths, manifest, asset)
        return _ready_observation(
            manifest,
            asset,
            paths,
            cache_hit=True,
            downloaded=False,
            downloads_started=False,
            bytes_downloaded=0,
        )

    download = downloader or _download_url
    url = _asset_url(manifest, asset)
    tmp_archive = paths["root"] / f"download-{paths['archive'].name}"
    try:
        bytes_downloaded = download(url, tmp_archive)
        bytes_downloaded = tmp_archive.stat().st_size if bytes_downloaded is None else int(bytes_downloaded)
        _verify_file(tmp_archive, asset["sha256"])
        tmp_archive.replace(paths["archive"])
        _extract_to_cache(paths, manifest, asset)
    except WheelhouseError as exc:
        exc.observation.update(
            {
                "downloads_started": True,
                "installs_started": False,
                "selected_asset": asset["id"],
                "release_tag": manifest["release"]["tag"],
            }
        )
        raise
    finally:
        if tmp_archive.exists():
            tmp_archive.unlink()

    return _ready_observation(
        manifest,
        asset,
        paths,
        cache_hit=False,
        downloaded=True,
        downloads_started=True,
        bytes_downloaded=bytes_downloaded,
    )


def _extract_to_cache(paths: dict[str, Path], manifest: dict[str, Any], asset: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(dir=paths["root"]) as tmp_dir:
        tmp_extract = Path(tmp_dir) / "extracted"
        tmp_extract.mkdir()
        _extract_archive(paths["archive"], tmp_extract, asset.get("compression", ""))
        if paths["extracted"].exists():
            shutil.rmtree(paths["extracted"])
        tmp_extract.replace(paths["extracted"])
    _write_cache_marker(paths["marker"], manifest, asset)


def _wheel_is_compatible(filename: str, runtime_evidence: dict[str, str]) -> bool:
    if filename.endswith("-py3-none-any.whl"):
        return True
    python_tag = runtime_evidence["python_tag"]
    arch = {"aarch64": "aarch64", "x64": "x86_64"}.get(runtime_evidence["arch"], runtime_evidence["arch"])
    os_name = runtime_evidence["os"]
    platform_tag = f"{os_name}_{arch}"
    return f"-{python_tag}-{python_tag}-" in filename and platform_tag in filename


def resolve_verified_fallback(manifest: dict[str, Any], workspace_root: Path, runtime_evidence: dict[str, str]) -> dict[str, Any]:
    validate_manifest(manifest)
    fallback = manifest["fallback"]
    fallback_root = workspace_root / fallback["vendored_wheels"]
    wheel_records = fallback.get("wheels") or []

    for wheel in wheel_records:
        if not _wheel_is_compatible(wheel["filename"], runtime_evidence):
            raise WheelhouseError("fallback_incompatible_lane", "Vendored fallback wheel does not match this runtime lane")

    verified: list[str] = []
    for wheel in wheel_records:
        path = fallback_root / wheel["filename"]
        if not path.exists():
            raise WheelhouseError("fallback_missing_wheel", "Vendored fallback wheel is missing")
        if path.stat().st_size != wheel["size_bytes"]:
            raise WheelhouseError("fallback_size_mismatch", "Vendored fallback wheel size changed")
        _verify_file(path, wheel["sha256"], "fallback_checksum_mismatch")
        verified.append(str(path))

    return {
        "status": "ready",
        "cache_hit": False,
        "downloaded": False,
        "downloads_started": False,
        "installs_started": False,
        "sha256_verified": True,
        "fallback_path": str(fallback_root),
        "wheel_count": len(verified),
        "wheels": verified,
    }


def detect_runtime_lane() -> dict[str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux":
        os_name = "linux"
    elif system.startswith("win"):
        os_name = "windows"
    elif system == "darwin":
        os_name = "macos"
    else:
        os_name = system

    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine, machine)
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return {"os": os_name, "arch": arch, "python_tag": python_tag, "accelerator_lane": "cuda124"}
