from __future__ import annotations

from pathlib import Path

from pixal3d_extension.assets import (
    AUXILIARY_ASSETS,
    LOCALIZABLE_RUNTIME_DEPENDENCIES,
    PRIMARY_ASSET,
    UNLOCALIZED_RUNTIME_DEPENDENCIES,
    check_asset_sentinels,
    normalize_auxiliary_source_mode,
    resolve_auxiliary_sources,
)
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import validate_pipeline_patch

SUPPORTED_RUNTIME_LANES = {
    "linux-aarch64-cp312-cuda124",
    "linux-x64-cp312-cuda124",
    "windows-x64-cp311-cuda124",
    "windows-x64-cp312-cuda124",
}
SUPPORTED_RUNTIME_LANE = "linux-aarch64-cp312-cuda124"

SETUP_REQUIRED_PATHS = [
    "venv",
    "models/pixal3d/generate",
    "models/pixal3d/auxiliary/dinov3",
    "models/pixal3d/auxiliary/rmbg",
    "models/pixal3d/auxiliary/moge",
    "models/pixal3d/auxiliary/naf",
    "models/pixal3d/readiness.json",
]

AUXILIARY_MODELS = [
    {"id": manifest.repo_id, "logical_root": manifest.local_root, "status": "required"}
    for manifest in AUXILIARY_ASSETS.values()
]
LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS = list(LOCALIZABLE_RUNTIME_DEPENDENCIES)
RUNTIME_DEPENDENCY_STATUS = list(UNLOCALIZED_RUNTIME_DEPENDENCIES)


def _legacy_code(result: dict) -> dict:
    if result.get("code") == "missing_primary_assets":
        return {**result, "code": "missing_assets"}
    return result


def check_setup_readiness(workspace_root: str | Path, *, require_aux: bool = True) -> dict:
    layout = resolve_modly_layout(workspace_root)
    missing_setup_paths = [relative for relative in SETUP_REQUIRED_PATHS if not resolve_storage_path(layout, relative).exists()]
    if missing_setup_paths:
        return {
            "status": "blocked",
            "code": "setup_not_prepared",
            "setup_ready": False,
            "generation_allowed": False,
            "missing_setup_paths": missing_setup_paths,
            "model_storage_root": str((layout.models_root / "pixal3d").resolve()),
        }

    asset_result = _legacy_code(check_asset_sentinels(layout.ext_dir, require_aux=require_aux))
    if asset_result["status"] != "ready":
        return {
            "status": "prepared_waiting_for_weights",
            "code": "weights_missing_expected",
            "setup_ready": True,
            "generation_allowed": False,
            "missing": asset_result.get("missing", []),
            "generation_readiness_code": asset_result.get("code"),
            "model_storage_root": str((layout.models_root / "pixal3d").resolve()),
        }

    return {
        "status": "prepared",
        "code": "setup_prepared",
        "setup_ready": True,
        "generation_allowed": False,
        "missing": [],
        "generation_readiness_code": asset_result.get("code"),
        "model_storage_root": str((layout.models_root / "pixal3d").resolve()),
    }


def check_readiness(
    workspace_root: str | Path,
    *,
    require_aux: bool = True,
    auxiliary_mode: str | None = "default",
    network_available: bool | None = True,
    runtime_lane: str | None = SUPPORTED_RUNTIME_LANE,
    runtime_validated: bool = False,
    import_validation: dict | None = None,
    allow_remote_runtime_dependencies: bool = False,
) -> dict:
    asset_result = check_asset_sentinels(workspace_root, require_aux=False)
    if asset_result["status"] != "ready":
        result = _legacy_code(asset_result)
        result["unknown_auxiliaries"] = AUXILIARY_MODELS
        result["localizable_runtime_dependencies"] = LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS
        result["runtime_dependencies"] = RUNTIME_DEPENDENCY_STATUS
        return result

    if not require_aux:
        return {
            "status": "ready",
            "code": "ready",
            "missing": [],
            "generation_allowed": True,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
        }

    try:
        normalized_auxiliary_mode = normalize_auxiliary_source_mode(auxiliary_mode)
    except ValueError as exc:
        return {"status": "blocked", "code": "invalid_auxiliary_mode", "message": str(exc), "generation_allowed": False}

    auxiliary_source = resolve_auxiliary_sources(
        workspace_root,
        mode=normalized_auxiliary_mode,
        network_available=network_available,
    )
    if auxiliary_source["status"] == "blocked":
        return {
            "status": "blocked",
            "code": "missing_auxiliary_assets",
            "missing": auxiliary_source.get("missing", []),
            "auxiliary_source": auxiliary_source,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
            "generation_allowed": False,
        }

    patch_result = validate_pipeline_patch(
        workspace_root,
        auxiliary_mode=normalized_auxiliary_mode,
        network_available=network_available,
    )
    if patch_result["status"] != "ready":
        return {
            "status": "blocked",
            "code": patch_result["code"],
            "pipeline_patch": patch_result,
            "auxiliary_source": auxiliary_source,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
            "generation_allowed": False,
        }

    if runtime_lane not in SUPPORTED_RUNTIME_LANES:
        return {
            "status": "blocked",
            "code": "unsupported_lane",
            "runtime_lane": runtime_lane,
            "supported_runtime_lanes": sorted(SUPPORTED_RUNTIME_LANES),
            "auxiliary_source": auxiliary_source,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
            "generation_allowed": False,
        }

    unresolved_runtime_dependencies = auxiliary_source.get("unlocalized_runtime_dependencies", [])
    if normalized_auxiliary_mode in {"offline", "strict"} and unresolved_runtime_dependencies and not allow_remote_runtime_dependencies:
        return {
            "status": "blocked",
            "code": "offline_runtime_dependencies_unresolved",
            "message": "Some Pixal3D runtime dependencies still depend on cache or network fallback.",
            "auxiliary_source": auxiliary_source,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
            "generation_allowed": False,
        }

    if not runtime_validated:
        return {
            "status": "blocked",
            "code": "runtime_not_validated",
            "runtime_lane": runtime_lane,
            "auxiliary_source": auxiliary_source,
            "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
            "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
            "generation_allowed": False,
        }

    return {
        "status": "ready",
        "code": "ready",
        "missing": [],
        "generation_allowed": True,
        "pipeline_patch": patch_result,
        "auxiliary_source": auxiliary_source,
        "runtime_lane": runtime_lane,
        "localizable_runtime_dependencies": LOCALIZABLE_RUNTIME_DEPENDENCY_STATUS,
        "runtime_dependencies": RUNTIME_DEPENDENCY_STATUS,
        "import_validation": import_validation or {},
    }
