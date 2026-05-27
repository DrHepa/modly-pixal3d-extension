from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path


DINO_SOURCE = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_REPLACEMENT = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
RMBG_SOURCE = "briaai/RMBG-2.0"
RMBG_REPLACEMENT = "camenduru/RMBG-2.0"

PIPELINE_PATH = Path("models/pixal3d/generate/pipeline.json")
BACKUP_PATH = Path("models/pixal3d/generate/pipeline.json.modly-original")
READINESS_METADATA_PATH = Path("models/pixal3d/readiness.json")
PATCHER_VERSION = "2026-05-26"

_TARGETS = {
    "dino": (DINO_SOURCE, DINO_REPLACEMENT, ("args", "image_cond_model", "args", "model_name")),
    "rmbg": (RMBG_SOURCE, RMBG_REPLACEMENT, ("args", "rembg_model", "args", "model_name")),
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


def _diagnose_refs(data: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    diagnostics = []
    already_patched = True
    for key, (source, replacement, path) in _TARGETS.items():
        try:
            value = _get_nested(data, path)
        except KeyError:
            diagnostics.append({"key": key, "code": "missing_key", "expected": source, "actual": None})
            already_patched = False
            continue
        if value == source:
            already_patched = False
            continue
        if value == replacement:
            continue
        diagnostics.append({"key": key, "code": "unexpected_ref", "expected": source, "replacement": replacement, "actual": value})
        already_patched = False
    return diagnostics, already_patched and not diagnostics


def _metadata_matches(root: Path) -> bool:
    metadata = _read_metadata(root).get("pipeline_patch", {})
    return metadata.get("status") == "patched" and metadata.get("replacement_refs") == {"dino": DINO_REPLACEMENT, "rmbg": RMBG_REPLACEMENT}


def patch_pipeline(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root)
    layout = resolve_modly_layout(root)
    pipeline_path = resolve_storage_path(layout, str(PIPELINE_PATH))
    data, error, original_text = _load_json(pipeline_path)
    if error is not None:
        return error
    assert data is not None

    diagnostics, already_patched = _diagnose_refs(data)
    if diagnostics:
        return _failure("pipeline_patch_mismatch", "pipeline.json does not match expected upstream refs", diagnostics=diagnostics)

    if already_patched:
        if not _metadata_matches(root):
            return _failure("pipeline_patch_mismatch", "pipeline.json is patched but metadata is missing or mismatched")
        return {
            "status": "patched",
            "code": "pipeline_patch_current",
            "idempotent": True,
            "metadata_path": str(READINESS_METADATA_PATH),
            "backup_path": str(BACKUP_PATH),
            "generation_allowed": False,
        }

    backup_path = resolve_storage_path(layout, str(BACKUP_PATH))
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        backup_path.write_text(original_text, encoding="utf-8")

    patched = copy.deepcopy(data)
    for _key, (_source, replacement, path) in _TARGETS.items():
        _set_nested(patched, path, replacement)
    patched_text = json.dumps(patched, indent=2, sort_keys=True) + "\n"
    pipeline_path.write_text(patched_text, encoding="utf-8")

    metadata = _read_metadata(root)
    metadata["pipeline_patch"] = {
        "status": "patched",
        "patcher_version": PATCHER_VERSION,
        "patched_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_path": str(PIPELINE_PATH),
        "backup_path": str(BACKUP_PATH),
        "original_refs": {"dino": DINO_SOURCE, "rmbg": RMBG_SOURCE},
        "replacement_refs": {"dino": DINO_REPLACEMENT, "rmbg": RMBG_REPLACEMENT},
        "local_auxiliary_roots": {"dino": "models/pixal3d/aux/dinov3", "rmbg": "models/pixal3d/aux/rmbg"},
        "hashes": {"before": _hash_text(original_text), "after": _hash_text(patched_text)},
        "validation": "expected_substitutions_applied",
    }
    metadata["generation_allowed"] = False
    _write_metadata(root, metadata)

    return {
        "status": "patched",
        "code": "pipeline_patch_applied",
        "idempotent": False,
        "metadata_path": str(READINESS_METADATA_PATH),
        "backup_path": str(BACKUP_PATH),
        "generation_allowed": False,
    }


def validate_pipeline_patch(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root)
    layout = resolve_modly_layout(root)
    data, error, _text = _load_json(resolve_storage_path(layout, str(PIPELINE_PATH)))
    if error is not None:
        return error
    assert data is not None
    diagnostics, already_patched = _diagnose_refs(data)
    if diagnostics or not already_patched or not _metadata_matches(root):
        return _failure("pipeline_substitution_required", "pipeline substitutions and patch metadata are required")
    return {"status": "ready", "code": "pipeline_patch_ready", "generation_allowed": True, "metadata_path": str(READINESS_METADATA_PATH)}


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
