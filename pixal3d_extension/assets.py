from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path


@dataclass(frozen=True)
class AssetManifest:
    key: str
    repo_id: str
    local_root: str
    sentinels: tuple[str, ...]
    local_reference: str = "root"

    @property
    def sentinel_paths(self) -> tuple[str, ...]:
        return tuple(str(PurePosixPath(self.local_root) / sentinel) for sentinel in self.sentinels)

    @property
    def local_reference_path(self) -> str:
        if self.local_reference == "root":
            return self.local_root
        if self.local_reference == "sentinel":
            if len(self.sentinels) != 1:
                raise ValueError(f"asset {self.key!r} must have exactly one sentinel for sentinel local references")
            return self.sentinel_paths[0]
        raise ValueError(f"unsupported local reference kind for asset {self.key!r}: {self.local_reference!r}")


PRIMARY_ASSET = AssetManifest(
    key="primary",
    repo_id="TencentARC/Pixal3D",
    local_root="models/pixal3d/generate",
    sentinels=(
        "pipeline.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
)

AUXILIARY_ASSETS = {
    "dino": AssetManifest(
        key="dino",
        repo_id="camenduru/dinov3-vitl16-pretrain-lvd1689m",
        local_root="models/pixal3d/auxiliary/dinov3",
        sentinels=("config.json", "preprocessor_config.json", "model.safetensors"),
    ),
    "rmbg": AssetManifest(
        key="rmbg",
        repo_id="camenduru/RMBG-2.0",
        local_root="models/pixal3d/auxiliary/rmbg",
        sentinels=("config.json", "preprocessor_config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"),
    ),
    "moge": AssetManifest(
        key="moge",
        repo_id="Ruicheng/moge-2-vitl",
        local_root="models/pixal3d/auxiliary/moge",
        sentinels=("model.pt",),
        local_reference="sentinel",
    ),
}

AUXILIARY_BOOTSTRAP_ALLOWLIST = {
    key: {
        "repo_id": manifest.repo_id,
        "local_root": manifest.local_root,
        "files": list(manifest.sentinels),
    }
    for key, manifest in AUXILIARY_ASSETS.items()
}

AUXILIARY_SOURCE_MODES = {"default", "auto", "remote", "local", "offline", "strict"}
LOCAL_REQUIRED_AUXILIARY_SOURCE_MODES = {"local", "offline", "strict"}
AuxiliaryDownloader = Callable[..., Any]

LOCALIZABLE_RUNTIME_DEPENDENCIES = (
    {
        "key": "moge",
        "source": "Ruicheng/moge-2-vitl",
        "local_root": AUXILIARY_ASSETS["moge"].local_root,
        "local_checkpoint": AUXILIARY_ASSETS["moge"].sentinel_paths[0],
        "runtime_hook": "inference.load_moge_model",
        "offline_status": "local_first_with_remote_or_hf_cache_fallback",
        "localization_status": "wired_to_local_checkpoint_when_available",
    },
)

UNLOCALIZED_RUNTIME_DEPENDENCIES = (
    {
        "key": "naf",
        "source": "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth",
        "current_behavior": "Packaged NAF code calls torch.hub.load_state_dict_from_url(), so the checkpoint must already be in Torch cache or it may use the network.",
        "offline_status": "torch_cache_or_network_fallback",
        "localization_status": "checkpoint_not_declared_as_local_asset",
    },
)

REQUIRED_SENTINEL_PATHS = [
    *PRIMARY_ASSET.sentinel_paths,
    *(sentinel for manifest in AUXILIARY_ASSETS.values() for sentinel in manifest.sentinel_paths),
]


def normalize_auxiliary_source_mode(mode: str | None) -> str:
    normalized = (mode or "default").strip().lower()
    if normalized == "auto":
        normalized = "default"
    if normalized not in AUXILIARY_SOURCE_MODES:
        raise ValueError(f"unsupported auxiliary source mode: {mode!r}")
    return normalized


def requires_local_auxiliary(mode: str | None) -> bool:
    return normalize_auxiliary_source_mode(mode) in LOCAL_REQUIRED_AUXILIARY_SOURCE_MODES


def _missing_for_manifest(workspace_root: Path, manifest: AssetManifest) -> list[str]:
    layout = resolve_modly_layout(workspace_root)
    return [relative for relative in manifest.sentinel_paths if not resolve_storage_path(layout, relative).is_file()]


def _asset_status(workspace_root: Path, manifest: AssetManifest) -> dict[str, Any]:
    layout = resolve_modly_layout(workspace_root)
    resolved_root = resolve_storage_path(layout, manifest.local_root)
    logical_value = manifest.local_reference_path
    resolved_value = resolve_storage_path(layout, logical_value)
    missing = _missing_for_manifest(workspace_root, manifest)
    return {
        "key": manifest.key,
        "repo_id": manifest.repo_id,
        "logical_root": manifest.local_root,
        "resolved_path": str(resolved_root),
        "logical_value": logical_value,
        "resolved_value": str(resolved_value),
        "local_reference": manifest.local_reference,
        "sentinels": list(manifest.sentinel_paths),
        "missing": missing,
        "complete": not missing,
    }


def check_auxiliary_sentinels(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root)
    assets = {key: _asset_status(root, manifest) for key, manifest in AUXILIARY_ASSETS.items()}
    missing = [relative for asset in assets.values() for relative in asset["missing"]]
    if missing:
        return {
            "status": "blocked",
            "code": "missing_auxiliary_assets",
            "missing": missing,
            "assets": assets,
            "generation_allowed": False,
        }
    return {
        "status": "ready",
        "code": "auxiliary_assets_ready",
        "missing": [],
        "assets": assets,
        "generation_allowed": True,
    }


def _default_auxiliary_downloader(
    *,
    repo_id: str,
    filename: str,
    destination: str | Path,
    token: str | None = None,
    revision: str | None = None,
) -> str:
    """Download one allowlisted auxiliary file through Hugging Face Hub.

    The import stays inside this function so importing pixal3d_extension.assets
    never performs network-facing setup or requires huggingface_hub until an
    explicit bootstrap path calls it.
    """

    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {"repo_id": repo_id, "filename": filename}
    if token is not None:
        kwargs["token"] = token
    if revision is not None:
        kwargs["revision"] = revision
    cached_path = Path(hf_hub_download(**kwargs))
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, destination_path)
    return str(destination_path)


def _manifest_relative_file(manifest: AssetManifest, sentinel_path: str) -> PurePosixPath:
    return PurePosixPath(sentinel_path).relative_to(PurePosixPath(manifest.local_root))


def _bootstrap_failure(
    workspace_root: Path,
    code: str,
    message: str,
    *,
    downloads_started: bool,
    attempted: list[dict[str, str]],
    error: str | None = None,
) -> dict[str, Any]:
    sentinel_status = check_auxiliary_sentinels(workspace_root)
    result: dict[str, Any] = {
        "status": "failed",
        "code": code,
        "message": message,
        "downloads_started": downloads_started,
        "installs_started": False,
        "attempted": attempted,
        "allowlist": AUXILIARY_BOOTSTRAP_ALLOWLIST,
        "sentinel_status": sentinel_status,
        "missing": sentinel_status.get("missing", []),
        "generation_allowed": False,
    }
    if error:
        result["error"] = error
    return result


def bootstrap_auxiliary_assets(
    workspace_root: str | Path,
    downloader: AuxiliaryDownloader | None = None,
    *,
    force: bool = False,
    token: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Populate exact allowlisted DINO/RMBG/MoGe auxiliary assets.

    Files are staged under the Modly model storage tree first. Final sentinel
    files are replaced only after every allowlisted file has been staged
    successfully, so a failed bootstrap cannot leave freshly downloaded partials
    looking like a complete auxiliary install.
    """

    root = Path(workspace_root)
    before = check_auxiliary_sentinels(root)
    if before["status"] == "ready" and not force:
        return {
            "status": "ready",
            "code": "auxiliary_assets_ready",
            "downloads_started": False,
            "installs_started": False,
            "force": False,
            "allowlist": AUXILIARY_BOOTSTRAP_ALLOWLIST,
            "sentinel_status": before,
            "missing": [],
            "generation_allowed": True,
        }

    layout = resolve_modly_layout(root)
    auxiliary_parent = resolve_storage_path(layout, "models/pixal3d/auxiliary")
    stage_root = auxiliary_parent / f".bootstrap-{uuid.uuid4().hex}.tmp"
    active_downloader = downloader or _default_auxiliary_downloader
    downloads_started = False
    attempted: list[dict[str, str]] = []
    staged: dict[str, dict[str, Any]] = {}

    try:
        auxiliary_parent.mkdir(parents=True, exist_ok=True)
        stage_root.mkdir(parents=True, exist_ok=False)

        for key, manifest in AUXILIARY_ASSETS.items():
            staged[key] = {
                "repo_id": manifest.repo_id,
                "logical_root": manifest.local_root,
                "files": [],
                "downloaded": [],
                "reused": [],
            }
            for sentinel in manifest.sentinel_paths:
                relative_file = _manifest_relative_file(manifest, sentinel)
                filename = str(relative_file)
                stage_path = stage_root / key / Path(*relative_file.parts)
                final_path = resolve_storage_path(layout, sentinel)
                stage_path.parent.mkdir(parents=True, exist_ok=True)

                if final_path.is_file() and not force:
                    shutil.copy2(final_path, stage_path)
                    staged[key]["reused"].append(filename)
                else:
                    downloads_started = True
                    attempted.append({"asset": key, "repo_id": manifest.repo_id, "filename": filename})
                    download_result = active_downloader(
                        repo_id=manifest.repo_id,
                        filename=filename,
                        destination=stage_path,
                        token=token,
                        revision=revision,
                    )
                    if not stage_path.is_file() and isinstance(download_result, (str, os.PathLike)):
                        candidate = Path(download_result)
                        if candidate.is_file():
                            shutil.copy2(candidate, stage_path)
                    staged[key]["downloaded"].append(filename)

                if not stage_path.is_file():
                    raise RuntimeError(f"downloader did not create allowlisted file: {manifest.repo_id}/{filename}")
                staged[key]["files"].append(filename)

        for key, manifest in AUXILIARY_ASSETS.items():
            for sentinel in manifest.sentinel_paths:
                relative_file = _manifest_relative_file(manifest, sentinel)
                stage_path = stage_root / key / Path(*relative_file.parts)
                if not stage_path.is_file():
                    raise RuntimeError(f"staged auxiliary file missing before final copy: {manifest.key}/{relative_file}")

        for key, manifest in AUXILIARY_ASSETS.items():
            for sentinel in manifest.sentinel_paths:
                relative_file = _manifest_relative_file(manifest, sentinel)
                stage_path = stage_root / key / Path(*relative_file.parts)
                final_path = resolve_storage_path(layout, sentinel)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage_path, final_path)

    except Exception as exc:
        shutil.rmtree(stage_root, ignore_errors=True)
        return _bootstrap_failure(
            root,
            "auxiliary_bootstrap_failed",
            "failed to bootstrap Pixal3D DINO/RMBG/MoGe auxiliary assets",
            downloads_started=downloads_started,
            attempted=attempted,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)

    after = check_auxiliary_sentinels(root)
    if after["status"] != "ready":
        return {
            "status": "blocked",
            "code": "missing_auxiliary_assets",
            "message": "auxiliary bootstrap completed but required sentinels are still missing",
            "downloads_started": downloads_started,
            "installs_started": False,
            "force": force,
            "attempted": attempted,
            "allowlist": AUXILIARY_BOOTSTRAP_ALLOWLIST,
            "assets": staged,
            "sentinel_status": after,
            "missing": after.get("missing", []),
            "generation_allowed": False,
        }
    return {
        "status": "ready",
        "code": "auxiliary_assets_bootstrapped" if downloads_started or force else "auxiliary_assets_ready",
        "downloads_started": downloads_started,
        "installs_started": False,
        "force": force,
        "attempted": attempted,
        "allowlist": AUXILIARY_BOOTSTRAP_ALLOWLIST,
        "assets": staged,
        "sentinel_status": after,
        "missing": [],
        "generation_allowed": True,
    }


def resolve_auxiliary_sources(
    workspace_root: str | Path,
    *,
    mode: str | None = "default",
    network_available: bool | None = True,
) -> dict[str, Any]:
    """Resolve auxiliary sources without probing the network.

    Local paths are used only when all sentinels for that individual auxiliary
    asset exist. In default mode, a missing local auxiliary keeps the current
    remote Hugging Face/cache fallback only when the caller says network fallback
    is available. MoGe resolves to its local checkpoint file path, while DINO/RMBG
    resolve to local model directories. Strict/local/offline modes fail before any
    HF-facing import can happen.
    """

    normalized_mode = normalize_auxiliary_source_mode(mode)
    root = Path(workspace_root)
    aux_status = check_auxiliary_sentinels(root)
    network_allowed = bool(network_available) and normalized_mode != "offline"
    prefer_local = normalized_mode != "remote"
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    fallback_used = False

    def local_source(manifest: AssetManifest, asset: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": "local",
            "value": asset["resolved_value"],
            "repo_id": manifest.repo_id,
            "logical_root": manifest.local_root,
            "logical_value": asset["logical_value"],
            "local_reference": manifest.local_reference,
        }

    for key, manifest in AUXILIARY_ASSETS.items():
        asset = aux_status["assets"][key]
        if prefer_local and asset["complete"]:
            sources[key] = local_source(manifest, asset)
            continue

        if asset["complete"] and not network_allowed:
            sources[key] = local_source(manifest, asset)
            continue

        if normalized_mode in LOCAL_REQUIRED_AUXILIARY_SOURCE_MODES or not network_allowed:
            missing.extend(asset["missing"])
            continue

        fallback_used = True
        sources[key] = {
            "kind": "remote",
            "value": manifest.repo_id,
            "repo_id": manifest.repo_id,
            "logical_root": manifest.local_root,
        }

    unresolved = sorted(set(missing))
    if unresolved:
        return {
            "status": "blocked",
            "code": "missing_auxiliary_assets",
            "mode": normalized_mode,
            "network_available": bool(network_available),
            "missing": unresolved,
            "assets": aux_status["assets"],
            "sources": sources,
            "unlocalized_runtime_dependencies": list(UNLOCALIZED_RUNTIME_DEPENDENCIES),
            "localizable_runtime_dependencies": list(LOCALIZABLE_RUNTIME_DEPENDENCIES),
            "generation_allowed": False,
        }

    return {
        "status": "fallback" if fallback_used else "ready",
        "code": "remote_auxiliary_fallback" if fallback_used else "local_auxiliary_assets_ready",
        "mode": normalized_mode,
        "network_available": bool(network_available),
        "missing": [],
        "assets": aux_status["assets"],
        "sources": sources,
        "unlocalized_runtime_dependencies": list(UNLOCALIZED_RUNTIME_DEPENDENCIES),
        "localizable_runtime_dependencies": list(LOCALIZABLE_RUNTIME_DEPENDENCIES),
        "generation_allowed": True,
    }


def check_asset_sentinels(workspace_root: str | Path, *, require_aux: bool = True) -> dict:
    root = Path(workspace_root)
    missing_primary = _missing_for_manifest(root, PRIMARY_ASSET)

    if missing_primary:
        return {
            "status": "blocked",
            "code": "missing_primary_assets",
            "missing": missing_primary,
            "generation_allowed": False,
        }
    if require_aux:
        aux_result = check_auxiliary_sentinels(root)
        if aux_result["status"] != "ready":
            return aux_result
    return {"status": "ready", "code": "assets_ready", "missing": [], "generation_allowed": True}
