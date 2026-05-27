from __future__ import annotations

from pathlib import Path

from pixal3d_extension.assets import AUXILIARY_ASSETS, PRIMARY_ASSET, check_asset_sentinels
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path
from pixal3d_extension.pipeline_patch import validate_pipeline_patch

SUPPORTED_RUNTIME_LANE = "linux-aarch64-cp312-nvidia-cuda"

SETUP_REQUIRED_PATHS = [
    "venv",
    "models/pixal3d/generate",
    "models/pixal3d/aux/dinov3",
    "models/pixal3d/aux/rmbg",
    "models/pixal3d/readiness.json",
]

AUXILIARY_MODELS = [
    {"id": manifest.repo_id, "logical_root": manifest.local_root, "status": "required"}
    for manifest in AUXILIARY_ASSETS.values()
]


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
    runtime_lane: str | None = SUPPORTED_RUNTIME_LANE,
    runtime_validated: bool = False,
    import_validation: dict | None = None,
) -> dict:
    asset_result = check_asset_sentinels(workspace_root, require_aux=require_aux)
    if asset_result["status"] != "ready":
        result = _legacy_code(asset_result)
        result["unknown_auxiliaries"] = AUXILIARY_MODELS
        return result

    if not require_aux:
        return {"status": "ready", "code": "ready", "missing": [], "generation_allowed": True}

    patch_result = validate_pipeline_patch(workspace_root)
    if patch_result["status"] != "ready":
        return {"status": "blocked", "code": patch_result["code"], "pipeline_patch": patch_result, "generation_allowed": False}

    if runtime_lane != SUPPORTED_RUNTIME_LANE:
        return {
            "status": "blocked",
            "code": "unsupported_lane",
            "runtime_lane": runtime_lane,
            "supported_runtime_lane": SUPPORTED_RUNTIME_LANE,
            "generation_allowed": False,
        }

    if not runtime_validated:
        return {"status": "blocked", "code": "runtime_not_validated", "runtime_lane": runtime_lane, "generation_allowed": False}

    return {
        "status": "ready",
        "code": "ready",
        "missing": [],
        "generation_allowed": True,
        "pipeline_patch": patch_result,
        "runtime_lane": runtime_lane,
        "import_validation": import_validation or {},
    }
