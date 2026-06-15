from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pixal3d_extension.assets import AUXILIARY_ASSETS, resolve_auxiliary_sources
from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path


DINO_SOURCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REPLACEMENT = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
RMBG_SOURCE = "briaai/RMBG-2.0"
RMBG_REPLACEMENT = "camenduru/RMBG-2.0"

PIPELINE_PATH = Path("models/pixal3d/generate/pipeline.json")
BACKUP_PATH = Path("models/pixal3d/generate/pipeline.json.modly-original")
READINESS_METADATA_PATH = Path("models/pixal3d/readiness.json")
PATCHER_VERSION = "2026-06-13"

_TARGETS = {
    "dino": {
        "source": DINO_SOURCE,
        "remote": DINO_REPLACEMENT,
        "path": ("args", "image_cond_model", "args", "model_name"),
    },
    "rmbg": {
        "source": RMBG_SOURCE,
        "remote": RMBG_REPLACEMENT,
        "path": ("args", "rembg_model", "args", "model_name"),
    },
}


def _failure(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "blocked", "code": code, "message": message, "generation_allowed": False, **extra}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if not path.is_file():
        return None, _failure("missing_pipeline_json", "pipeline.json is required before patching"), ""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, _failure("invalid_pipeline_json", f"pipeline.json is invalid JSON: {exc.msg}"), text
    if not isinstance(data, dict):
        return None, _failure("invalid_pipeline_json", "pipeline.json must contain a JSON object"), text
    return data, None, text


def _get_nested(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    current: Any = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _read_metadata(root: Path) -> dict[str, Any]:
    layout = resolve_modly_layout(root)
    path = resolve_storage_path(layout, str(READINESS_METADATA_PATH))
    if not path.is_file():
        return {"extension_id": "pixal3d"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"extension_id": "pixal3d"}
    return data if isinstance(data, dict) else {"extension_id": "pixal3d"}


def _write_metadata(root: Path, metadata: dict[str, Any]) -> None:
    layout = resolve_modly_layout(root)
    path = resolve_storage_path(layout, str(READINESS_METADATA_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _actual_replacement_refs(auxiliary_resolution: dict[str, Any]) -> dict[str, str]:
    return {key: str(source["value"]) for key, source in auxiliary_resolution.get("sources", {}).items()}


def _metadata_replacement_refs(auxiliary_resolution: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for key, source in auxiliary_resolution.get("sources", {}).items():
        if source.get("kind") == "local":
            refs[key] = f"local:{source['logical_root']}"
        else:
            refs[key] = str(source["repo_id"])
    return refs


def _replacement_kinds(auxiliary_resolution: dict[str, Any]) -> dict[str, str]:
    return {key: str(source["kind"]) for key, source in auxiliary_resolution.get("sources", {}).items()}


def _local_auxiliary_roots() -> dict[str, str]:
    return {key: manifest.local_root for key, manifest in AUXILIARY_ASSETS.items()}


def _is_previous_local_patch(metadata: dict[str, Any], key: str) -> bool:
    patch = metadata.get("pipeline_patch", {})
    if not isinstance(patch, dict):
        return False
    return patch.get("replacement_kinds", {}).get(key) == "local"


def _inspect_refs(
    data: dict[str, Any],
    expected_refs: dict[str, str],
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    diagnostics: list[dict[str, Any]] = []
    update_needed = False
    metadata = metadata or {}

    for key, target in _TARGETS.items():
        expected = expected_refs[key]
        try:
            value = _get_nested(data, target["path"])
        except KeyError:
            diagnostics.append({"key": key, "code": "missing_key", "expected": expected, "actual": None})
            continue

        if value == expected:
            continue

        upgradeable_values = {target["source"], target["remote"]}
        if value in upgradeable_values or _is_previous_local_patch(metadata, key):
            update_needed = True
            continue

        diagnostics.append(
            {
                "key": key,
                "code": "unexpected_ref",
                "expected": expected,
                "source": target["source"],
                "remote_fallback": target["remote"],
                "actual": value,
            }
        )

    return diagnostics, update_needed


def _metadata_matches(root: Path, auxiliary_resolution: dict[str, Any], data: dict[str, Any]) -> bool:
    metadata = _read_metadata(root).get("pipeline_patch", {})
    if not isinstance(metadata, dict):
        return False
    if metadata.get("status") != "patched":
        return False
    if metadata.get("replacement_refs") != _metadata_replacement_refs(auxiliary_resolution):
        return False
    if metadata.get("replacement_kinds") != _replacement_kinds(auxiliary_resolution):
        return False
    if metadata.get("local_auxiliary_roots") != _local_auxiliary_roots():
        return False
    diagnostics, update_needed = _inspect_refs(data, _actual_replacement_refs(auxiliary_resolution), metadata={"pipeline_patch": metadata})
    return not diagnostics and not update_needed


def _write_patch_metadata(
    root: Path,
    *,
    original_text: str,
    patched_text: str,
    auxiliary_resolution: dict[str, Any],
) -> None:
    metadata = _read_metadata(root)
    metadata["pipeline_patch"] = {
        "status": "patched",
        "patcher_version": PATCHER_VERSION,
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_path": str(PIPELINE_PATH),
        "backup_path": str(BACKUP_PATH),
        "auxiliary_mode": auxiliary_resolution.get("mode", "default"),
        "auxiliary_resolution_code": auxiliary_resolution.get("code"),
        "replacement_kinds": _replacement_kinds(auxiliary_resolution),
        "original_refs": {"dino": DINO_SOURCE, "rmbg": RMBG_SOURCE},
        "remote_fallback_refs": {"dino": DINO_REPLACEMENT, "rmbg": RMBG_REPLACEMENT},
        "replacement_refs": _metadata_replacement_refs(auxiliary_resolution),
        "local_auxiliary_roots": _local_auxiliary_roots(),
        "unlocalized_runtime_dependencies": auxiliary_resolution.get("unlocalized_runtime_dependencies", []),
        "hashes": {"before": _hash_text(original_text), "after": _hash_text(patched_text)},
        "validation": "expected_substitutions_applied",
    }
    metadata["generation_allowed"] = False
    _write_metadata(root, metadata)


def _resolve_or_fail(
    root: Path,
    *,
    auxiliary_mode: str | None,
    network_available: bool | None,
) -> dict[str, Any]:
    try:
        auxiliary_resolution = resolve_auxiliary_sources(root, mode=auxiliary_mode, network_available=network_available)
    except ValueError as exc:
        return _failure("invalid_auxiliary_mode", str(exc))
    if auxiliary_resolution["status"] == "blocked":
        return _failure(
            "missing_auxiliary_assets",
            "local DINO/RMBG auxiliary assets are required for this mode",
            auxiliary_source=auxiliary_resolution,
            missing=auxiliary_resolution.get("missing", []),
        )
    return auxiliary_resolution


def patch_pipeline(
    workspace_root: str | Path,
    *,
    auxiliary_mode: str | None = "default",
    network_available: bool | None = True,
) -> dict[str, Any]:
    root = Path(workspace_root)
    auxiliary_resolution = _resolve_or_fail(root, auxiliary_mode=auxiliary_mode, network_available=network_available)
    if auxiliary_resolution.get("status") == "blocked":
        return auxiliary_resolution

    layout = resolve_modly_layout(root)
    pipeline_path = resolve_storage_path(layout, str(PIPELINE_PATH))
    data, error, original_text = _load_json(pipeline_path)
    if error is not None:
        return error
    assert data is not None

    expected_refs = _actual_replacement_refs(auxiliary_resolution)
    metadata = _read_metadata(root)
    diagnostics, update_needed = _inspect_refs(data, expected_refs, metadata=metadata)
    if diagnostics:
        return _failure("pipeline_patch_mismatch", "pipeline.json does not match expected upstream, fallback, or local refs", diagnostics=diagnostics)

    if not update_needed and _metadata_matches(root, auxiliary_resolution, data):
        return {
            "status": "patched",
            "code": "pipeline_patch_current",
            "idempotent": True,
            "metadata_path": str(READINESS_METADATA_PATH),
            "backup_path": str(BACKUP_PATH),
            "auxiliary_source": auxiliary_resolution,
            "replacement_refs": expected_refs,
            "generation_allowed": False,
        }

    backup_path = resolve_storage_path(layout, str(BACKUP_PATH))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        backup_path.write_text(original_text, encoding="utf-8")

    patched = copy.deepcopy(data)
    for key, replacement in expected_refs.items():
        _set_nested(patched, _TARGETS[key]["path"], replacement)
    patched_text = json.dumps(patched, indent=2, sort_keys=True) + "\n"
    if patched_text != original_text:
        pipeline_path.write_text(patched_text, encoding="utf-8")

    _write_patch_metadata(root, original_text=original_text, patched_text=patched_text, auxiliary_resolution=auxiliary_resolution)

    return {
        "status": "patched",
        "code": "pipeline_patch_applied" if update_needed else "pipeline_patch_metadata_repaired",
        "idempotent": False,
        "metadata_path": str(READINESS_METADATA_PATH),
        "backup_path": str(BACKUP_PATH),
        "auxiliary_source": auxiliary_resolution,
        "replacement_refs": expected_refs,
        "generation_allowed": False,
    }


def validate_pipeline_patch(
    workspace_root: str | Path,
    *,
    auxiliary_mode: str | None = "default",
    network_available: bool | None = True,
) -> dict[str, Any]:
    root = Path(workspace_root)
    auxiliary_resolution = _resolve_or_fail(root, auxiliary_mode=auxiliary_mode, network_available=network_available)
    if auxiliary_resolution.get("status") == "blocked":
        return auxiliary_resolution

    layout = resolve_modly_layout(root)
    data, error, _text = _load_json(resolve_storage_path(layout, str(PIPELINE_PATH)))
    if error is not None:
        return error
    assert data is not None
    diagnostics, update_needed = _inspect_refs(data, _actual_replacement_refs(auxiliary_resolution), metadata=_read_metadata(root))
    if diagnostics or update_needed or not _metadata_matches(root, auxiliary_resolution, data):
        return _failure(
            "pipeline_substitution_required",
            "pipeline substitutions and patch metadata are required for the resolved auxiliary source mode",
            auxiliary_source=auxiliary_resolution,
            diagnostics=diagnostics,
        )
    return {
        "status": "ready",
        "code": "pipeline_patch_ready",
        "generation_allowed": True,
        "metadata_path": str(READINESS_METADATA_PATH),
        "auxiliary_source": auxiliary_resolution,
    }


def restore_pipeline(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root)
    layout = resolve_modly_layout(root)
    backup_path = resolve_storage_path(layout, str(BACKUP_PATH))
    pipeline_path = resolve_storage_path(layout, str(PIPELINE_PATH))
    if not backup_path.is_file():
        return _failure("missing_pipeline_backup", "original pipeline backup is required for restore")
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
    metadata = _read_metadata(root)
    metadata["pipeline_patch"] = {
        "status": "restored",
        "pipeline_path": str(PIPELINE_PATH),
        "backup_path": str(BACKUP_PATH),
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata["generation_allowed"] = False
    _write_metadata(root, metadata)
    return {"status": "restored", "code": "pipeline_restored", "metadata_path": str(READINESS_METADATA_PATH), "generation_allowed": False}
