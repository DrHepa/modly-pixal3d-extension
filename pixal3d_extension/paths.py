from dataclasses import dataclass
from pathlib import Path, PurePosixPath


EXTENSION_ID = "pixal3d"
MODELS_PREFIX = "models"
PIXAL3D_GENERATE_SUFFIX = (MODELS_PREFIX, EXTENSION_ID, "generate")
WORKSPACE_SEGMENT = "workspace"
WINDOWS_RESERVED_SEGMENTS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


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
    return all(
        part not in {"", ".", ".."}
        and not part.startswith(".")
        and part.split(".", 1)[0].upper() not in WINDOWS_RESERVED_SEGMENTS
        for part in parts
    )


def normalize_logical_path(value: str) -> str:
    return value.replace("\\", "/")


def _split_logical_path(value: str | Path) -> tuple[str, tuple[str, ...]]:
    text = normalize_logical_path(str(value))
    while len(text) > 1 and text.endswith("/") and not (len(text) == 3 and text[1] == ":"):
        text = text[:-1]

    if len(text) >= 3 and text[0].isalpha() and text[1] == ":" and text[2] == "/":
        return text[:3], tuple(segment for segment in text[3:].split("/") if segment)
    if text.startswith("/"):
        return "/", tuple(segment for segment in text[1:].split("/") if segment)
    return "", tuple(segment for segment in text.split("/") if segment)


def _join_logical_path(prefix: str, segments: tuple[str, ...]) -> str:
    if prefix:
        if not segments:
            return prefix
        return f"{prefix}{'/'.join(segments)}" if prefix.endswith("/") else f"{prefix}/{'/'.join(segments)}"
    return "/".join(segments)


def _logical_path_from_parts(prefix: str, segments: tuple[str, ...]) -> Path | None:
    if not prefix and not segments:
        return None
    return Path(_join_logical_path(prefix, segments))


def derive_modly_home_from_model_dir(model_dir: str | Path | None) -> Path | None:
    """Derive the Modly home from a local Pixal3D generate model directory.

    The derivation is string/segment based instead of filesystem based so it
    works for Windows-style paths observed by Linux-hosted contract tests and
    does not depend on user-specific absolute roots existing on this machine.
    """

    if model_dir is None:
        return None
    prefix, segments = _split_logical_path(model_dir)
    suffix_length = len(PIXAL3D_GENERATE_SUFFIX)
    if len(segments) < suffix_length:
        return None
    if tuple(segment.lower() for segment in segments[-suffix_length:]) != PIXAL3D_GENERATE_SUFFIX:
        return None
    return _logical_path_from_parts(prefix, segments[:-suffix_length])


def derive_modly_home_from_workspace_dir(workspace_dir: str | Path | None) -> Path | None:
    """Derive the Modly home from a Modly workspace directory or child path."""

    if workspace_dir is None:
        return None
    prefix, segments = _split_logical_path(workspace_dir)
    for index in range(len(segments) - 1, -1, -1):
        if segments[index].lower() != WORKSPACE_SEGMENT:
            continue
        return _logical_path_from_parts(prefix, segments[:index])
    return None


def derive_modly_home(*, model_dir: str | Path | None = None, workspace_dir: str | Path | None = None) -> Path | None:
    """Derive the Modly home, preferring the model storage path when present."""

    return derive_modly_home_from_model_dir(model_dir) or derive_modly_home_from_workspace_dir(workspace_dir)


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
