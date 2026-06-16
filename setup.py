from __future__ import annotations

import argparse
import json
import subprocess
import sys
import venv as stdlib_venv
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pixal3d_extension.assets import (
    AUXILIARY_ASSETS,
    LOCALIZABLE_RUNTIME_DEPENDENCIES,
    PRIMARY_ASSET,
    UNLOCALIZED_RUNTIME_DEPENDENCIES,
    bootstrap_auxiliary_assets,
)
from pixal3d_extension.pipeline_patch import patch_pipeline, restore_pipeline
from pixal3d_extension.paths import ModlyLayout, resolve_modly_layout, resolve_storage_path
from pixal3d_extension.readiness import check_readiness, check_setup_readiness
from modly_wheelhouse import (
    WheelhouseError,
    detect_runtime_lane,
    load_manifest,
    prepare_wheelhouse,
    resolve_verified_fallback,
    validate_manifest,
)


EXTENSION_ID = "pixal3d"
VENV_DIR = "venv"
VENV_MARKER = ".modly-prepared"
WHEELHOUSE_DIR = "wheels"
WHEELHOUSE_MANIFEST = "wheelhouse.manifest.json"

LOCAL_WHEEL_PACKAGES = [
    "utils3d==1.3+modly.headless",
    "pipeline==1.0.0+modly",
    "moge==2.0.0+modly",
    "naf==0.1.0+modly",
    "o-voxel==0.0.1",
    "cumesh==0.0.1",
    "flex-gemm==1.0.0",
    "nvdiffrast==0.4.0",
    "nvdiffrec-render==0.0.0",
    "pixal3d-core==0.1.0+modly",
]

WINDOWS_LOCAL_WHEEL_PACKAGES = [
    "utils3d==1.3+modly.headless",
    "pipeline==1.0.0+modly",
    "moge==2.0.0+modly",
    "naf==0.1.0+modly",
    "o-voxel-vb-ap==0.0.1",
    "cumesh-vb==1.0",
    "flex-gemm-ap==1.0.0",
    "drtk==0.1.0",
    "flash-attn==2.8.3",
    "nvdiffrast==0.4.0",
    "nvdiffrec-render==0.0.1",
    "pixal3d-core==0.1.0+modly",
]

OPTIONAL_NATTEN_PACKAGES = ["natten==0.21.0"]
LINUX_METADATA_PACKAGES = ["triton"]
WINDOWS_METADATA_PACKAGES = ["triton-windows"]
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
PYTORCH_PIP_FLAGS = ["--no-cache-dir", "--retries", "5", "--timeout", "60"]
PYTORCH_DIRECT_PIP_FLAGS = [*PYTORCH_PIP_FLAGS, "--no-deps"]
PYTORCH_CUDA_PACKAGES = ["torch==2.6.0+cu124", "torchvision==0.21.0+cu124"]
PYTORCH_AARCH64_PACKAGES = ["torch==2.12.0", "torchvision==0.27.0"]
PIP_BOOTSTRAP_PACKAGE = "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl#sha256=9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"

RUNTIME_DIRS = [
    "models/pixal3d/generate",
    "models/pixal3d/auxiliary/dinov3",
    "models/pixal3d/auxiliary/rmbg",
    "models/pixal3d/auxiliary/moge",
    "models/pixal3d/auxiliary/naf",
]
READINESS_METADATA = "models/pixal3d/readiness.json"


class SetupPathConflict(RuntimeError):
    code = "setup_path_conflict"

    def __init__(self, logical_path: str, path: Path, *, expected: str = "directory") -> None:
        self.logical_path = logical_path
        self.path = path
        self.expected = expected
        super().__init__(f"Setup path conflict: expected {expected} at {path}")

    @property
    def observation(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "failure_code": self.code,
            "logical_path": self.logical_path,
            "path": str(self.path),
            "expected": self.expected,
            "actual": "file-or-non-directory",
            "message": f"Expected {self.expected}, but a file or non-directory already exists at {self.path}.",
        }


def _workspace_item(layout: ModlyLayout, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe setup path: {relative_path}")
    return resolve_storage_path(layout, relative_path)


def _first_non_directory(path: Path) -> Path | None:
    current = path
    while current != current.parent:
        if current.exists() and not current.is_dir():
            return current
        current = current.parent
    return None


def _ensure_setup_directory(layout: ModlyLayout, relative_path: str, created: list[dict[str, str]], skipped: list[dict[str, str]]) -> Path:
    path = _workspace_item(layout, relative_path)
    conflict = _first_non_directory(path)
    if conflict is not None:
        raise SetupPathConflict(relative_path, conflict)
    if path.exists():
        skipped.append({"path": relative_path, "reason": "already-exists"})
    else:
        path.mkdir(parents=True, exist_ok=True)
        created.append({"path": relative_path, "kind": "directory"})
    return path


def _readiness_payload() -> dict[str, Any]:
    return {
        "extension_id": EXTENSION_ID,
        "weights_downloaded": False,
        "generation_ready": False,
        "next_step": "Download model assets from Modly UI.",
        "auxiliary_assets": _auxiliary_asset_bootstrap_plan(),
        "localizable_runtime_dependencies": list(LOCALIZABLE_RUNTIME_DEPENDENCIES),
        "unlocalized_runtime_dependencies": list(UNLOCALIZED_RUNTIME_DEPENDENCIES),
    }


def _auxiliary_asset_bootstrap_plan() -> dict[str, Any]:
    return {
        "logical_roots": {key: manifest.local_root for key, manifest in AUXILIARY_ASSETS.items()},
        "sentinels": {key: list(manifest.sentinel_paths) for key, manifest in AUXILIARY_ASSETS.items()},
        "download_sources": {key: manifest.source for key, manifest in AUXILIARY_ASSETS.items()},
        "source_kinds": {key: manifest.source_kind for key, manifest in AUXILIARY_ASSETS.items()},
        "bootstrap_command": "python3 setup.py --bootstrap-auxiliary-assets --workspace-root <extension-dir> --json",
        "bootstrap_intent": "explicitly downloads only the allowlisted DINO/RMBG/MoGe/NAF files into models/pixal3d/auxiliary; normal setup does not do this hidden download",
        "fallback_policy": "default mode may use remote/HF/Torch-cache fallback only when local sentinels are missing and network fallback is available; local/offline/strict modes require these files first. NATTEN/libnatten strict kernels remain a separate runtime validation from NAF checkpoint localization.",
    }


def _model_download_plan() -> dict[str, Any]:
    return {
        "status": "planned",
        "downloads_started": False,
        "installs_started": False,
        "primary": {
            "repo_id": PRIMARY_ASSET.repo_id,
            "logical_root": PRIMARY_ASSET.local_root,
            "sentinels": list(PRIMARY_ASSET.sentinel_paths),
        },
        "auxiliary": _auxiliary_asset_bootstrap_plan(),
        "localizable_runtime_dependencies": list(LOCALIZABLE_RUNTIME_DEPENDENCIES),
        "runtime_dependencies": list(UNLOCALIZED_RUNTIME_DEPENDENCIES),
        "note": "Normal setup emits the asset plan only; use --bootstrap-auxiliary-assets for the explicit DINO/RMBG/MoGe/NAF auxiliary bootstrap, and use Modly-owned model download/repair for Pixal3D primary weights. Local NAF checkpoint availability does not prove NATTEN/libnatten strict kernel availability.",
    }


def _create_prepare_paths(layout: ModlyLayout) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    created: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    venv_path = _workspace_item(layout, VENV_DIR)
    if venv_path.exists() and not venv_path.is_dir():
        raise SetupPathConflict(VENV_DIR, venv_path)
    if (venv_path / "pyvenv.cfg").exists():
        skipped.append({"path": VENV_DIR, "reason": "already-exists"})
    else:
        stdlib_venv.EnvBuilder(with_pip=True).create(venv_path)
        created.append({"path": VENV_DIR, "kind": "directory"})
    (venv_path / VENV_MARKER).write_text(f"{EXTENSION_ID}\n", encoding="utf-8")

    for relative_path in RUNTIME_DIRS:
        _ensure_setup_directory(layout, relative_path, created, skipped)

    readiness_path = _workspace_item(layout, READINESS_METADATA)
    conflict = _first_non_directory(readiness_path.parent)
    if conflict is not None:
        raise SetupPathConflict(READINESS_METADATA, conflict)
    if readiness_path.exists():
        if not readiness_path.is_file():
            raise SetupPathConflict(READINESS_METADATA, readiness_path, expected="file")
        skipped.append({"path": READINESS_METADATA, "reason": "already-exists"})
    else:
        readiness_path.parent.mkdir(parents=True, exist_ok=True)
        created.append({"path": READINESS_METADATA, "kind": "file"})
    readiness_path.write_text(json.dumps(_readiness_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return created, skipped


def _run_setup_command(args: list[str], *, cwd: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_attempts = 3 if _is_retryable_pip_json_command(args) else 1
    completed = None
    for attempt in range(1, max_attempts + 1):
        completed = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
        attempts.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
            }
        )
        if completed.returncode == 0 or not _looks_like_pip_json_decode_failure(completed.stderr):
            break
    assert completed is not None
    return {
        "args": args,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "ok": completed.returncode == 0,
        "attempts": attempts,
    }


def _is_retryable_pip_json_command(args: list[str]) -> bool:
    return len(args) >= 4 and args[1:4] == ["-m", "pip", "install"]


def _looks_like_pip_json_decode_failure(stderr: str) -> bool:
    return "json.decoder.JSONDecodeError" in stderr and "pip/_internal/index" in stderr


def _install_prepare_dependencies(workspace_root: Path, *, wheelhouse_path: Path | None = None) -> dict[str, Any]:
    venv_python = _venv_python_path(workspace_root)
    wheelhouse = wheelhouse_path or (workspace_root / WHEELHOUSE_DIR)
    if not venv_python.exists():
        return {"status": "failed", "code": "venv_python_missing", "venv_python": str(venv_python), "commands": []}
    if not wheelhouse.exists():
        return {"status": "failed", "code": "wheelhouse_missing", "wheelhouse": str(wheelhouse), "commands": []}

    local_wheel_packages = _local_wheel_packages_for_wheelhouse(wheelhouse)
    torch_install_command = _torch_install_command_for_wheelhouse(venv_python, wheelhouse)
    commands = [
        [str(venv_python), "-m", "pip", "install", "--no-deps", PIP_BOOTSTRAP_PACKAGE],
        torch_install_command,
        [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, "-r", "requirements.txt"],
        [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, *_metadata_dependency_packages_for_wheelhouse(wheelhouse)],
        [str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheelhouse), *local_wheel_packages],
    ]
    if _wheelhouse_contains_natten(wheelhouse):
        commands.append([str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheelhouse), *OPTIONAL_NATTEN_PACKAGES])
    results: list[dict[str, Any]] = []
    for command in commands:
        result = _run_setup_command(command, cwd=workspace_root)
        results.append(result)
        if not result["ok"]:
            return {"status": "failed", "code": "dependency_install_failed", "failed_command": command, "commands": results}
    pip_check = _run_setup_command([str(venv_python), "-m", "pip", "check"], cwd=workspace_root)
    runtime_check = _runtime_cuda_check(venv_python, workspace_root, wheelhouse)
    pip_check_acceptable = pip_check["ok"] or _is_known_aarch64_nvidia_platform_check_false_positive(pip_check, wheelhouse)
    if not pip_check_acceptable or not runtime_check["ok"]:
        return {
            "status": "failed",
            "code": "dependency_runtime_check_failed" if pip_check_acceptable else "dependency_metadata_check_failed",
            "failed_command": runtime_check["args"] if pip_check_acceptable else pip_check["args"],
            "commands": results,
            "pip_check": pip_check,
            "runtime_check": runtime_check,
        }
    return {
        "status": "installed",
        "code": "dependencies_installed",
        "venv_python": str(venv_python),
        "wheelhouse": str(wheelhouse),
        "local_wheel_packages": local_wheel_packages,
        "optional_natten_packages": OPTIONAL_NATTEN_PACKAGES,
        "natten_runtime": _natten_runtime_status(venv_python, workspace_root),
        "pip_check": pip_check,
        "runtime_check": runtime_check,
        "commands": results,
    }


def _is_known_aarch64_nvidia_platform_check_false_positive(pip_check: dict[str, Any], wheelhouse: Path) -> bool:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return False
    output = f"{pip_check.get('stdout_tail', '')}\n{pip_check.get('stderr_tail', '')}"
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines == ["nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"]


def _runtime_cuda_check(venv_python: Path, workspace_root: Path, wheelhouse: Path) -> dict[str, Any]:
    native_imports = _native_import_modules_for_wheelhouse(wheelhouse)
    upstream_imports = _upstream_import_modules_for_wheelhouse(wheelhouse)
    if _wheelhouse_contains_natten(wheelhouse):
        native_imports.append("natten")
    code = (
        "import importlib, json, sys\n"
        "payload = {'ok': False}\n"
        "def import_with_torch_compile_disabled(name):\n"
        "    if name != 'natten':\n"
        "        return importlib.import_module(name)\n"
        "    import torch\n"
        "    original_compile = getattr(torch, 'compile', None)\n"
        "    if original_compile is None:\n"
        "        return importlib.import_module(name)\n"
        "    def identity_compile(function=None, *args, **kwargs):\n"
        "        if function is None:\n"
        "            return lambda inner: inner\n"
        "        return function\n"
        "    torch.compile = identity_compile\n"
        "    try:\n"
        "        return importlib.import_module(name)\n"
        "    finally:\n"
        "        torch.compile = original_compile\n"
        "try:\n"
        "    import torch\n"
        "    payload['torch_version'] = torch.__version__\n"
        "    payload['torch_cuda_version'] = torch.version.cuda\n"
        "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())\n"
        f"    imports = {native_imports!r}\n"
        "    for name in imports:\n"
        "        import_with_torch_compile_disabled(name)\n"
        f"    aliases = {_windows_native_aliases_for_wheelhouse(wheelhouse)!r}\n"
        "    for upstream_name, windows_name in aliases.items():\n"
        "        sys.modules.setdefault(upstream_name, importlib.import_module(windows_name))\n"
        f"    upstream_imports = {upstream_imports!r}\n"
        "    for name in upstream_imports:\n"
        "        importlib.import_module(name)\n"
        "    payload['imports'] = imports\n"
        "    payload['upstream_imports'] = upstream_imports\n"
        "    payload['ok'] = bool(torch.version.cuda and torch.cuda.is_available())\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )
    result = _run_setup_command([str(venv_python), "-c", code], cwd=workspace_root)
    try:
        payload = json.loads(result.get("stdout_tail", "").strip().splitlines()[-1])
    except Exception:
        payload = {"ok": False, "error": "runtime CUDA probe did not return JSON"}
    return {**result, **payload, "ok": bool(result.get("ok") and payload.get("ok"))}


def _native_import_modules_for_wheelhouse(wheelhouse: Path) -> list[str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return ["cumesh_vb", "flex_gemm_ap", "o_voxel_vb_ap", "nvdiffrast", "nvdiffrec_render"]
    return ["cumesh", "flex_gemm", "o_voxel", "nvdiffrast", "nvdiffrec_render"]


def _upstream_import_modules_for_wheelhouse(wheelhouse: Path) -> list[str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return ["cumesh", "flex_gemm", "o_voxel"]
    return []


def _windows_native_aliases_for_wheelhouse(wheelhouse: Path) -> dict[str, str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return {"cumesh": "cumesh_vb", "flex_gemm": "flex_gemm_ap", "o_voxel": "o_voxel_vb_ap"}
    return {}


def _wheelhouse_contains_natten(wheelhouse: Path) -> bool:
    return any(wheelhouse.glob("natten-*.whl"))


def _local_wheel_packages_for_wheelhouse(wheelhouse: Path) -> list[str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return WINDOWS_LOCAL_WHEEL_PACKAGES
    return LOCAL_WHEEL_PACKAGES


def _metadata_dependency_packages_for_wheelhouse(wheelhouse: Path) -> list[str]:
    if any(wheelhouse.glob("*win_amd64.whl")):
        return WINDOWS_METADATA_PACKAGES
    return LINUX_METADATA_PACKAGES


def _torch_install_command_for_wheelhouse(venv_python: Path, wheelhouse: Path) -> list[str]:
    wheelhouse_text = str(wheelhouse).replace("\\", "/")
    if "windows-x64-cp312-cuda124" in wheelhouse_text or any(wheelhouse.glob("*win_amd64.whl")):
        return [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, "--index-url", PYTORCH_CUDA_INDEX_URL, *PYTORCH_CUDA_PACKAGES]
    if "linux-x64-cp312-cuda124" in wheelhouse_text:
        return [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, "--index-url", PYTORCH_CUDA_INDEX_URL, *PYTORCH_CUDA_PACKAGES]
    return [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, *PYTORCH_AARCH64_PACKAGES]


def _natten_runtime_status(venv_python: Path, workspace_root: Path) -> dict[str, Any]:
    result = _run_setup_command(
        [
            str(venv_python),
            "-c",
            (
                "import json\n"
                "def import_natten_with_torch_compile_disabled():\n"
                "    import importlib, torch\n"
                "    original_compile = getattr(torch, 'compile', None)\n"
                "    if original_compile is None:\n"
                "        return importlib.import_module('natten')\n"
                "    def identity_compile(function=None, *args, **kwargs):\n"
                "        if function is None:\n"
                "            return lambda inner: inner\n"
                "        return function\n"
                "    torch.compile = identity_compile\n"
                "    try:\n"
                "        return importlib.import_module('natten')\n"
                "    finally:\n"
                "        torch.compile = original_compile\n"
                "try:\n"
                "    natten = import_natten_with_torch_compile_disabled()\n"
                "    print(json.dumps({'importable': True, 'version': getattr(natten, '__version__', None), "
                "'HAS_LIBNATTEN': bool(getattr(natten, 'HAS_LIBNATTEN', False))}, sort_keys=True))\n"
                "except Exception as exc:\n"
                "    print(json.dumps({'importable': False, 'HAS_LIBNATTEN': False, 'error': f'{type(exc).__name__}: {exc}'}, sort_keys=True))\n"
            ),
        ],
        cwd=workspace_root,
    )
    try:
        payload = json.loads(result.get("stdout_tail", "").strip().splitlines()[-1])
    except Exception:
        payload = {"importable": False, "HAS_LIBNATTEN": False, "error": "natten runtime probe did not return JSON"}
    payload["strict_naf_available"] = bool(payload.get("importable") and payload.get("HAS_LIBNATTEN"))
    payload["fallback_required"] = not payload["strict_naf_available"]
    return payload


def _venv_python_path(workspace_root: Path) -> Path:
    venv_root = workspace_root / VENV_DIR
    candidates = []
    if sys.platform.startswith("win"):
        candidates.extend([venv_root / "Scripts" / "python.exe", venv_root / "bin" / "python"])
    else:
        candidates.extend([venv_root / "bin" / "python", venv_root / "Scripts" / "python.exe"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _prepare_wheelhouse_for_setup(workspace_root: Path) -> dict[str, Any]:
    manifest = load_manifest(workspace_root / WHEELHOUSE_MANIFEST)
    runtime_evidence = detect_runtime_lane()
    try:
        return prepare_wheelhouse(manifest, runtime_evidence, workspace_root)
    except WheelhouseError as exc:
        if exc.code not in {"network", "auth_required"}:
            raise
        fallback = resolve_verified_fallback(manifest, workspace_root, runtime_evidence)
        return {
            **fallback,
            "fallback_mode": manifest.get("fallback", {}).get("mode", "migration_only"),
            "fallback_reason": exc.code,
            "release_failure": exc.observation,
        }


def _prepared_wheelhouse_path(observation: dict[str, Any]) -> Path:
    path = observation.get("wheelhouse_path") or observation.get("fallback_path")
    if not isinstance(path, str) or not path:
        raise WheelhouseError("wheelhouse_path_missing", "Wheelhouse preparation did not return a local install path")
    return Path(path)


def _wheelhouse_manifest_observation(workspace_root: Path) -> dict[str, Any]:
    manifest_path = workspace_root / WHEELHOUSE_MANIFEST
    if not manifest_path.exists():
        return {"status": "missing", "path": WHEELHOUSE_MANIFEST}
    try:
        manifest = load_manifest(manifest_path)
        validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, WheelhouseError) as exc:
        code = getattr(exc, "code", "invalid_manifest")
        return {"status": "failed", "failure_code": code, "path": WHEELHOUSE_MANIFEST}
    return {
        "status": "available",
        "release_tag": manifest["release"]["tag"],
        "wheelhouse_version": manifest["wheelhouse_version"],
        "asset_count": len(manifest["assets"]),
    }


def _load_payload(raw_payload: str | None) -> dict[str, Any]:
    if not raw_payload:
        return {}
    payload = json.loads(raw_payload)
    return {key: payload[key] for key in {"ext_dir"} if key in payload}


def _coerce_payload_arg(explicit_payload: str | None, positional_payload: str | None, unknown_args: list[str]) -> str | None:
    if explicit_payload:
        return explicit_payload
    if positional_payload:
        return positional_payload
    if unknown_args and unknown_args[0].lstrip().startswith("{"):
        return " ".join(unknown_args)
    return None


def run_setup(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Prepare the Pixal3D Modly extension.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--patch-pipeline", action="store_true")
    parser.add_argument("--restore-pipeline", action="store_true")
    parser.add_argument("--download-plan", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--bootstrap-auxiliary-assets", action="store_true", help="Explicitly download the allowlisted local DINO/RMBG/MoGe/NAF auxiliary assets.")
    parser.add_argument("--force-auxiliary-assets", action="store_true", help="Redownload allowlisted DINO/RMBG/MoGe/NAF auxiliary files during explicit bootstrap.")
    parser.add_argument("--auxiliary-mode", choices=["default", "auto", "remote", "local", "offline", "strict"], default="default")
    parser.add_argument("--offline", action="store_true", help="Disable remote auxiliary fallback for readiness/patch planning.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--payload-json")
    parser.add_argument("positional_payload_json", nargs="?", help="Modly install payload JSON. Modly may pass this as a positional argument.")
    args, unknown_args = parser.parse_known_args(argv)

    raw_payload = _coerce_payload_arg(args.payload_json, args.positional_payload_json, unknown_args)
    setup_inputs = _load_payload(raw_payload)
    layout = resolve_modly_layout(args.workspace_root, ext_dir=setup_inputs.get("ext_dir"))
    workspace_root = layout.ext_dir
    prepare_requested = args.prepare or bool(raw_payload)

    result: dict[str, Any] = {
        "extension_id": EXTENSION_ID,
        "entrypoint": "setup.py",
        "workspace_root": str(workspace_root),
        "resolved_paths": layout.as_dict(),
        "downloads_started": False,
        "installs_started": False,
        "wheelhouse_manifest": _wheelhouse_manifest_observation(workspace_root),
        "auxiliary_mode": "offline" if args.offline else args.auxiliary_mode,
        "network_available": not args.offline,
        "auxiliary_assets": _auxiliary_asset_bootstrap_plan(),
        "localizable_runtime_dependencies": list(LOCALIZABLE_RUNTIME_DEPENDENCIES),
        "runtime_dependencies": list(UNLOCALIZED_RUNTIME_DEPENDENCIES),
    }

    if prepare_requested:
        try:
            created, skipped = _create_prepare_paths(layout)
        except SetupPathConflict as exc:
            return {
                **result,
                "status": "failed",
                "failure_code": exc.code,
                "path_conflict": exc.observation,
                "created": [],
                "skipped": [],
                "downloads_started": False,
                "installs_started": False,
                "setup_readiness": check_setup_readiness(workspace_root),
                "next_steps": [
                    "remove or rename the conflicting file/non-directory path",
                    "rerun setup",
                ],
            }
        wheelhouse_prepare = None
        dependency_install = None
        if not args.skip_install:
            try:
                wheelhouse_prepare = _prepare_wheelhouse_for_setup(workspace_root)
                dependency_install = _install_prepare_dependencies(
                    workspace_root,
                    wheelhouse_path=_prepared_wheelhouse_path(wheelhouse_prepare),
                )
            except (OSError, json.JSONDecodeError, WheelhouseError) as exc:
                code = getattr(exc, "code", "wheelhouse_prepare_failed")
                wheelhouse_prepare = getattr(
                    exc,
                    "observation",
                    {"status": "failed", "failure_code": code, "downloads_started": False, "installs_started": False},
                )
                return {
                    **result,
                    "status": "failed",
                    "created": created,
                    "skipped": skipped,
                    "wheelhouse_prepare": wheelhouse_prepare,
                    "dependency_install": None,
                    "installs_started": False,
                    "setup_readiness": check_setup_readiness(workspace_root),
                    "next_steps": ["preseed a verified wheelhouse release asset or fix the wheelhouse manifest"],
                }
        install_failed = dependency_install is not None and dependency_install.get("status") != "installed"
        result.update(
            {
                "status": "failed" if install_failed else "prepared",
                "created": created,
                "skipped": skipped,
                "wheelhouse_prepare": wheelhouse_prepare,
                "dependency_install": dependency_install,
                "installs_started": dependency_install is not None,
                "setup_readiness": check_setup_readiness(workspace_root),
                "next_steps": ["fix dependency installation failure", "rerun setup"]
                if install_failed
                else ["download model assets from Modly UI", "rerun readiness", "run generation"],
            }
        )
        if args.bootstrap_auxiliary_assets and not install_failed:
            auxiliary_bootstrap = bootstrap_auxiliary_assets(workspace_root, force=args.force_auxiliary_assets)
            bootstrap_success = auxiliary_bootstrap.get("status") == "ready"
            result.update(
                {
                    "status": "prepared" if bootstrap_success else "failed",
                    "code": auxiliary_bootstrap.get("code"),
                    "downloads_started": bool(result.get("downloads_started") or auxiliary_bootstrap.get("downloads_started")),
                    "auxiliary_bootstrap": auxiliary_bootstrap,
                    "setup_readiness": check_setup_readiness(workspace_root),
                    "next_steps": ["rerun readiness", "patch pipeline", "run generation"]
                    if bootstrap_success
                    else ["check network/auth for Hugging Face/GitHub auxiliary assets", "rerun explicit auxiliary bootstrap"],
                }
            )
        return result

    auxiliary_mode = "offline" if args.offline else args.auxiliary_mode
    network_available = not args.offline

    if args.download_plan:
        return {**result, **_model_download_plan(), "status": "download_plan"}
    if args.download_models:
        return {
            **result,
            "status": "blocked",
            "code": "model_download_managed_by_modly",
            "downloads_started": False,
            "installs_started": False,
            "download_plan": _model_download_plan(),
            "message": "Use the Modly model download/repair flow to transfer Pixal3D primary and auxiliary weights; setup.py does not perform hidden downloads.",
        }
    if args.bootstrap_auxiliary_assets:
        auxiliary_bootstrap = bootstrap_auxiliary_assets(workspace_root, force=args.force_auxiliary_assets)
        success = auxiliary_bootstrap.get("status") == "ready"
        return {
            **result,
            "status": "bootstrap_auxiliary_assets" if success else "failed",
            "code": auxiliary_bootstrap.get("code"),
            "downloads_started": bool(auxiliary_bootstrap.get("downloads_started")),
            "installs_started": False,
            "auxiliary_bootstrap": auxiliary_bootstrap,
            "setup_readiness": check_setup_readiness(workspace_root),
            "next_steps": ["rerun readiness", "patch pipeline", "run generation"]
            if success
            else ["check network/auth for Hugging Face/GitHub auxiliary assets", "rerun explicit auxiliary bootstrap"],
        }
    if args.patch_pipeline:
        return {**result, **patch_pipeline(workspace_root, auxiliary_mode=auxiliary_mode, network_available=network_available)}
    if args.restore_pipeline:
        return {**result, **restore_pipeline(workspace_root)}
    if args.readiness:
        return {
            **result,
            "status": "readiness",
            "readiness": check_readiness(workspace_root, auxiliary_mode=auxiliary_mode, network_available=network_available),
        }

    return {
        **result,
        "status": "dry_run",
        "prepare_command": "python3 setup.py --prepare --json",
        "wheelhouse": WHEELHOUSE_DIR,
        "wheelhouse_manifest_path": WHEELHOUSE_MANIFEST,
        "local_wheel_packages": LOCAL_WHEEL_PACKAGES,
        "download_plan": _model_download_plan(),
    }


def main() -> None:
    result = run_setup()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
