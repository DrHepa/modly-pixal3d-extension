from dataclasses import dataclass
from pathlib import Path, PurePosixPath


EXTENSION_ID = "pixal3d"
MODELS_PREFIX = "models"


@dataclass(frozen=True)
class ModlyLayout:
    modly_home: Path
    ext_dir: Path
    models_root: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "modly_home": str(self.modly_home),
            "ext_dir": str(self.ext_dir),
            "models_root": str(self.models_root),
        }


def is_safe_relative_path(value: str) -> bool:
    value = normalize_logical_path(value)
    if not value or value.startswith(("/", "file://", "http://", "https://")):
        return False
    if ":" in value:
        return False
    parts = PurePosixPath(value).parts
    return all(part not in {"", ".", ".."} and not part.startswith(".") for part in parts)


def normalize_logical_path(value: str) -> str:
    return value.replace("\\", "/")


def require_safe_relative_path(value: str) -> str:
    if not is_safe_relative_path(value):
        raise ValueError(f"unsafe relative path: {value!r}")
    return normalize_logical_path(value)


def is_contained_path(root: str | Path, candidate: str | Path) -> bool:
    root_path = Path(root).resolve()
    candidate_path = Path(candidate).resolve()
    return candidate_path == root_path or root_path in candidate_path.parents


def require_contained_path(root: str | Path, value: str, *, allowed_root: str | Path | None = None, allow_absolute: bool = False) -> Path:
    base = Path(root).resolve()
    raw = Path(value)
    if raw.is_absolute():
        if not allow_absolute:
            raise ValueError(f"unsafe absolute path: {value!r}")
        candidate = raw.resolve()
    else:
        safe_value = require_safe_relative_path(value)
        candidate = (base / Path(*PurePosixPath(safe_value).parts)).resolve()

    if not is_contained_path(base, candidate):
        raise ValueError(f"path escapes workspace root: {value!r}")
    if allowed_root is not None and not is_contained_path(allowed_root, candidate):
        raise ValueError(f"path escapes allowed root: {value!r}")
    return candidate


def resolve_modly_layout(workspace_root: str | Path, *, ext_dir: str | Path | None = None) -> ModlyLayout:
    resolved_ext_dir = Path(ext_dir or workspace_root).expanduser().resolve()
    if resolved_ext_dir.name == EXTENSION_ID and resolved_ext_dir.parent.name == "extensions":
        modly_home = resolved_ext_dir.parent.parent.resolve()
    else:
        modly_home = resolved_ext_dir
    return ModlyLayout(modly_home=modly_home, ext_dir=resolved_ext_dir, models_root=(modly_home / MODELS_PREFIX).resolve())


def resolve_storage_path(layout: ModlyLayout, logical_path: str) -> Path:
    safe_path = require_safe_relative_path(logical_path)
    relative = PurePosixPath(safe_path)
    if relative.parts and relative.parts[0] == MODELS_PREFIX:
        return (layout.modly_home / Path(*relative.parts)).resolve()
    return (layout.ext_dir / Path(*relative.parts)).resolve()
