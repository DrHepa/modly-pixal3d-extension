from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from pixal3d_extension.readiness import TRANSFORMERS_VERSION, check_readiness, check_setup_readiness
from modly_wheelhouse import (
    WheelhouseError,
    detect_runtime_lane,
    load_manifest,
    prepare_wheelhouse,
    resolve_verified_fallback,
    select_asset,
    validate_manifest,
)


EXTENSION_ID = "pixal3d"
VENV_DIR = "venv"
VENV_MARKER = ".modly-prepared"
WHEELHOUSE_DIR = "wheels"
WHEELHOUSE_MANIFEST = "wheelhouse.manifest.json"
MV_CORE_WHEEL = "wheels/mv/pixal3d_core-0.1.0+modly-py3-none-any.whl"
MV_CORE_WHEEL_SHA256 = "3ad32043cc429091d2bb2435e3e4256406f94da2dadfe92c4d36a44aa7c2309c"

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
BLACKWELL_RUNTIME_LANE = "windows-x64-cp311-cuda128-blackwell"
BLACKWELL_REQUIRED_IMPORTS = ["cumesh_vb", "flex_gemm_ap", "o_voxel_vb_ap", "nvdiffrast", "nvdiffrec_render", "natten"]
BLACKWELL_REQUIRED_UPSTREAM_IMPORTS = ["cumesh", "flex_gemm", "o_voxel"]
PIP_BOOTSTRAP_PACKAGE = "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl#sha256=9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"

RUNTIME_DIRS = [
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


class SetupPayloadError(ValueError):
    code = "invalid_runtime_evidence"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            "Setup payload must be exactly one non-empty JSON object; pass it once via "
            "--payload-json or as Modly's single positional JSON argument. "
            f"{detail}"
        )

    @property
    def observation(self) -> dict[str, Any]:
        return {
            "extension_id": EXTENSION_ID,
            "entrypoint": "setup.py",
            "status": "failed",
            "failure_code": self.code,
            "downloads_started": False,
            "installs_started": False,
            "message": str(self),
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
        "download_sources": {"naf": AUXILIARY_ASSETS["naf"].source},
        "source_kinds": {"naf": AUXILIARY_ASSETS["naf"].source_kind},
        "modly_managed_sources": {key: AUXILIARY_ASSETS[key].repo_id for key in ("dino", "rmbg", "moge")},
        "bootstrap_command": "python3 setup.py --bootstrap-auxiliary-assets --workspace-root <extension-dir> --json",
        "bootstrap_intent": "explicitly downloads only the allowlisted NAF checkpoint into models/pixal3d/auxiliary/naf; DINO/RMBG/MoGe are managed by Modly shared weight groups",
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
        "note": "Normal setup emits the asset plan only; use Modly to download the shared Pixal3D/DINO/RMBG/MoGe group. Use --bootstrap-auxiliary-assets only for the NAF checkpoint. Local NAF availability does not prove NATTEN/libnatten strict kernel availability.",
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


def _runtime_evidence_for_wheelhouse(wheelhouse: Path) -> dict[str, str]:
    text = str(wheelhouse).replace("\\", "/").lower()
    if "windows-x64-cp311-cuda128-blackwell" in text:
        return {
            "os": "windows",
            "arch": "x64",
            "python_tag": "cp311",
            "accelerator_lane": "cuda128-blackwell",
            "cuda_version": "12.8",
            "gpu_sm": "120",
        }
    if any(wheelhouse.glob("*win_amd64.whl")):
        python_tag = "cp311" if any(wheelhouse.glob("*-cp311-cp311-win_amd64.whl")) else "cp312"
        return {"os": "windows", "arch": "x64", "python_tag": python_tag, "accelerator_lane": "cuda124"}
    if "linux-x64" in text:
        return {"os": "linux", "arch": "x64", "python_tag": "cp312", "accelerator_lane": "cuda124"}
    return {"os": "linux", "arch": "aarch64", "python_tag": "cp312", "accelerator_lane": "cuda124"}


def _dependency_policy(runtime_evidence: dict[str, str]) -> dict[str, Any]:
    lane = "{os}-{arch}-{python_tag}-{accelerator_lane}".format(**runtime_evidence)
    if lane == BLACKWELL_RUNTIME_LANE:
        return {
            "lane": lane,
            "torch": "2.7.1+cu128",
            "torchvision": "0.22.1+cu128",
            "torch_index_url": "https://download.pytorch.org/whl/cu128",
            "natten": "0.21.6",
            "expected_torch_cuda": "12.8",
            "required_gpu_sm": "120",
            "require_libnatten": True,
            "strict_validation": True,
            "required_imports": list(BLACKWELL_REQUIRED_IMPORTS),
            "required_upstream_imports": list(BLACKWELL_REQUIRED_UPSTREAM_IMPORTS),
        }
    if runtime_evidence.get("os") == "linux" and runtime_evidence.get("arch") == "aarch64":
        return {
            "lane": lane,
            "torch": "2.12.0",
            "torchvision": "0.27.0",
            "torch_index_url": None,
            "natten": "0.21.0",
            "strict_validation": False,
        }
    return {
        "lane": lane,
        "torch": "2.6.0+cu124",
        "torchvision": "0.21.0+cu124",
        "torch_index_url": PYTORCH_CUDA_INDEX_URL,
        "natten": "0.21.0",
        "strict_validation": False,
    }


def _dependency_install_plan(workspace_root: Path, wheelhouse: Path, runtime_evidence: dict[str, str]) -> dict[str, Any]:
    venv_python = _venv_python_path(workspace_root)
    policy = _dependency_policy(runtime_evidence)
    torch_command = [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS]
    if policy["torch_index_url"]:
        torch_command.extend(["--index-url", policy["torch_index_url"]])
    torch_command.extend([f"torch=={policy['torch']}", f"torchvision=={policy['torchvision']}"])
    natten_command = None
    if _wheelhouse_contains_natten(wheelhouse):
        natten_command = [
            str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheelhouse),
            f"natten=={policy['natten']}",
        ]
    return {
        "policy": policy,
        "torch_command": torch_command,
        "natten_command": natten_command,
        "local_wheel_packages": _local_wheel_packages_for_wheelhouse(wheelhouse),
        "metadata_packages": _metadata_dependency_packages_for_wheelhouse(wheelhouse),
    }


def _install_prepare_dependencies(
    workspace_root: Path,
    *,
    wheelhouse_path: Path | None = None,
    runtime_evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    venv_python = _venv_python_path(workspace_root)
    wheelhouse = wheelhouse_path or (workspace_root / WHEELHOUSE_DIR)
    if not venv_python.exists():
        return {"status": "failed", "code": "venv_python_missing", "venv_python": str(venv_python), "commands": []}
    if not wheelhouse.exists():
        return {"status": "failed", "code": "wheelhouse_missing", "wheelhouse": str(wheelhouse), "commands": []}
    mv_wheel = SCRIPT_DIR / MV_CORE_WHEEL
    if not mv_wheel.is_file() or hashlib.sha256(mv_wheel.read_bytes()).hexdigest() != MV_CORE_WHEEL_SHA256:
        return {"status": "failed", "code": "mv_core_wheel_missing_or_invalid", "wheel": str(mv_wheel), "commands": []}

    runtime_evidence = runtime_evidence or _runtime_evidence_for_wheelhouse(wheelhouse)
    plan = _dependency_install_plan(workspace_root, wheelhouse, runtime_evidence)
    local_wheel_packages = plan["local_wheel_packages"]
    torch_install_command = plan["torch_command"]
    commands = [
        [str(venv_python), "-m", "pip", "install", "--no-deps", PIP_BOOTSTRAP_PACKAGE],
        torch_install_command,
        [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, "-r", "requirements.txt"],
        [str(venv_python), "-m", "pip", "install", *PYTORCH_PIP_FLAGS, *plan["metadata_packages"]],
        [str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheelhouse), *local_wheel_packages],
        [str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", str(mv_wheel)],
    ]
    if plan["natten_command"] is not None:
        commands.append(plan["natten_command"])
    results: list[dict[str, Any]] = []
    for command in commands:
        result = _run_setup_command(command, cwd=workspace_root)
        results.append(result)
        if not result["ok"]:
            return {"status": "failed", "code": "dependency_install_failed", "failed_command": command, "commands": results}
    pip_check = _run_setup_command([str(venv_python), "-m", "pip", "check"], cwd=workspace_root)
    runtime_check = _runtime_cuda_check(venv_python, workspace_root, wheelhouse, runtime_evidence)
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
        "optional_natten_packages": [f"natten=={plan['policy']['natten']}"],
        "runtime_evidence": runtime_evidence,
        "dependency_policy": plan["policy"],
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


def _validate_runtime_probe(probe: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not probe.get("ok"):
        errors.append(str(probe.get("error") or "runtime probe failed"))
    if policy.get("strict_validation"):
        expected = {
            "torch_version": policy["torch"],
            "torchvision_version": policy["torchvision"],
            "torch_cuda_version": policy["expected_torch_cuda"],
            "gpu_sm": policy["required_gpu_sm"],
            "natten_version": policy["natten"],
        }
        for key, expected_value in expected.items():
            if str(probe.get(key)) != str(expected_value):
                errors.append(f"{key} must be {expected_value}; found {probe.get(key)!r}")
        if probe.get("torch_cuda_available") is not True:
            errors.append("torch CUDA must be available")
        if policy.get("require_libnatten") and probe.get("natten_has_libnatten") is not True:
            errors.append("natten.HAS_LIBNATTEN must be True")
        for key, policy_key in (("imports", "required_imports"), ("upstream_imports", "required_upstream_imports")):
            missing = sorted(set(policy.get(policy_key, [])) - set(probe.get(key, [])))
            if missing:
                errors.append(f"missing {key}: {', '.join(missing)}")
    return {**probe, "ok": not errors, "validation_errors": errors}


def _runtime_cuda_check(
    venv_python: Path,
    workspace_root: Path,
    wheelhouse: Path,
    runtime_evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    runtime_evidence = runtime_evidence or _runtime_evidence_for_wheelhouse(wheelhouse)
    policy = _dependency_policy(runtime_evidence)
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
        "    from importlib.metadata import version\n"
        "    payload['transformers_version'] = version('transformers')\n"
        f"    if payload['transformers_version'] != {TRANSFORMERS_VERSION!r}:\n"
        f"        raise RuntimeError('Pixal3D requires transformers=={TRANSFORMERS_VERSION}')\n"
        "    payload['torch_version'] = torch.__version__\n"
        "    payload['torchvision_version'] = version('torchvision')\n"
        "    payload['torch_cuda_version'] = torch.version.cuda\n"
        "    payload['torch_cuda_available'] = bool(torch.cuda.is_available())\n"
        "    if payload['torch_cuda_available']:\n"
        "        major, minor = torch.cuda.get_device_capability()\n"
        "        payload['gpu_sm'] = f'{major}{minor}'\n"
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
        "    if 'natten' in imports:\n"
        "        natten = sys.modules.get('natten') or importlib.import_module('natten')\n"
        "        payload['natten_version'] = version('natten')\n"
        "        payload['natten_has_libnatten'] = bool(getattr(natten, 'HAS_LIBNATTEN', False))\n"
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
    combined = {
        **result,
        **payload,
        "ok": bool(result.get("ok") and payload.get("ok") and payload.get("transformers_version") == TRANSFORMERS_VERSION),
    }
    return _validate_runtime_probe(combined, policy)


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


def _prepare_wheelhouse_for_setup(
    workspace_root: Path,
    *,
    runtime_evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(workspace_root / WHEELHOUSE_MANIFEST)
    runtime_evidence = runtime_evidence or detect_runtime_lane()
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
    if raw_payload is None:
        return {}
    try:
        payload = json.loads(raw_payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SetupPayloadError("The supplied value is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise SetupPayloadError("JSON arrays, scalars, booleans, and null are not valid setup payloads.")
    if not payload:
        raise SetupPayloadError("The object is empty and does not contain Modly runtime evidence.")
    return {key: payload[key] for key in {"ext_dir", "cuda_version", "gpu_sm"} if key in payload}


def _coerce_payload_arg(explicit_payload: str | None, positional_payload: str | None, unknown_args: list[str]) -> str | None:
    if explicit_payload is not None:
        return explicit_payload
    if positional_payload is not None:
        return positional_payload
    if unknown_args and unknown_args[0].lstrip().startswith("{"):
        return " ".join(unknown_args)
    return None


def run_setup(argv: list[str] | None = None) -> dict[str, Any]:
    argv = list(sys.argv[1:] if argv is None else argv)
    payload_flag_count = sum(
        argument == "--payload-json" or argument.startswith("--payload-json=")
        for argument in argv
    )
    parser = argparse.ArgumentParser(description="Prepare the Pixal3D Modly extension.")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repair-worldsculpt", action="store_true", help="Offline repair of the separate validated WorldSculpt environment; does not alter the primary venv.")
    parser.add_argument("--repair-scene-prep", action="store_true", help="Repair the isolated Python 3.12 SAM3 + DA3 environment; model weights remain managed by Modly.")
    parser.add_argument("--skip-scene-prep", action="store_true", help="Skip provisioning the optional scene-preparation environment during general setup.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--patch-pipeline", action="store_true")
    parser.add_argument("--restore-pipeline", action="store_true")
    parser.add_argument("--download-plan", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--bootstrap-auxiliary-assets", action="store_true", help="Explicitly download the allowlisted NAF checkpoint.")
    parser.add_argument("--force-auxiliary-assets", action="store_true", help="Redownload the allowlisted NAF checkpoint during explicit bootstrap.")
    parser.add_argument("--auxiliary-mode", choices=["default", "auto", "remote", "local", "offline", "strict"], default="default")
    parser.add_argument("--offline", action="store_true", help="Disable remote auxiliary fallback for readiness/patch planning.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--payload-json")
    parser.add_argument("positional_payload_json", nargs="?", help="Modly install payload JSON. Modly may pass this as a positional argument.")
    args, unknown_args = parser.parse_known_args(argv)

    # Modly's Repair invokes setup.py with legacy positional interpreter/ext-dir/SM/CUDA,
    # while newer hosts send a JSON payload. Both are ordinary preparation paths.
    legacy_repair = bool(
        payload_flag_count == 0
        and args.positional_payload_json
        and not args.positional_payload_json.lstrip().startswith("{")
        and len(unknown_args) >= 2
    )
    try:
        if payload_flag_count > 1:
            raise SetupPayloadError("--payload-json was supplied more than once.")
        if args.payload_json is not None and args.positional_payload_json is not None:
            raise SetupPayloadError("Both explicit and positional payload forms were supplied.")
        if not legacy_repair and unknown_args:
            raise SetupPayloadError("Unexpected extra arguments make the payload ambiguous.")
        raw_payload = None if legacy_repair else _coerce_payload_arg(args.payload_json, args.positional_payload_json, unknown_args)
        setup_inputs = _load_payload(raw_payload)
    except SetupPayloadError as exc:
        return exc.observation
    if legacy_repair:
        # Older three-argument invocations still retain their ext-dir behavior;
        # the current four-argument contract additionally supplies complete lane evidence.
        setup_inputs = {"ext_dir": unknown_args[0]}
        if len(unknown_args) >= 3:
            setup_inputs.update({"gpu_sm": unknown_args[1], "cuda_version": unknown_args[2]})
    layout = resolve_modly_layout(args.workspace_root, ext_dir=setup_inputs.get("ext_dir"))
    workspace_root = layout.ext_dir
    prepare_requested = args.prepare or bool(raw_payload) or legacy_repair

    try:
        runtime_evidence = detect_runtime_lane(setup_inputs)
    except WheelhouseError as exc:
        return {
            "extension_id": EXTENSION_ID,
            "entrypoint": "setup.py",
            "workspace_root": str(workspace_root),
            "resolved_paths": layout.as_dict(),
            **exc.observation,
            "message": str(exc),
        }

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
        "runtime_evidence": runtime_evidence,
    }

    if prepare_requested and ("cuda_version" in setup_inputs or "gpu_sm" in setup_inputs):
        try:
            manifest = load_manifest(workspace_root / WHEELHOUSE_MANIFEST)
            validate_manifest(manifest)
            selected = select_asset(manifest, runtime_evidence)
        except (OSError, json.JSONDecodeError, WheelhouseError) as exc:
            code = getattr(exc, "code", "invalid_manifest")
            observation = getattr(
                exc,
                "observation",
                {"status": "failed", "failure_code": code, "downloads_started": False, "installs_started": False},
            )
            return {
                **result,
                **observation,
                "message": str(exc),
                "runtime_lane_preflight": {"status": "failed", "failure_code": code, "runtime_evidence": runtime_evidence},
            }
        result["runtime_lane_preflight"] = {
            "status": "matched",
            "selected_asset": selected["id"],
            "runtime_evidence": runtime_evidence,
        }

    if args.repair_worldsculpt:
        if args.skip_install:
            return {
                **result,
                "status": "skipped",
                "worldsculpt_lane": {"status": "skipped", "reason": "installation disabled by --skip-install"},
            }
        from pixal3d_extension.worldsculpt_lane import repair
        try:
            lane = repair(workspace_root)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            return {**result, "status": "failed", "failure_code": "worldsculpt_lane_failed", "reason": str(exc), "installs_started": True}
        return {**result, "status": "prepared", "worldsculpt_lane": lane, "installs_started": True}

    if args.repair_scene_prep:
        if args.skip_install:
            return {
                **result,
                "status": "skipped",
                "scene_prep_lane": {"status": "skipped", "reason": "installation disabled by --skip-install"},
                "installs_started": False,
            }
        from pixal3d_extension.scene_prepare_lane import capability, repair
        support = capability()
        if not support["supported"]:
            return {
                **result,
                "status": "failed",
                "failure_code": "scene_prep_unsupported_platform",
                "reason": support["reason"],
                "scene_prep_lane": support,
                "installs_started": False,
            }
        try:
            lane = repair(workspace_root)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            return {**result, "status": "failed", "failure_code": "scene_prep_lane_failed", "reason": str(exc), "installs_started": True}
        return {**result, "status": "prepared", "scene_prep_lane": lane, "installs_started": True}

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
                wheelhouse_prepare = _prepare_wheelhouse_for_setup(workspace_root, runtime_evidence=runtime_evidence)
                dependency_install = _install_prepare_dependencies(
                    workspace_root,
                    wheelhouse_path=_prepared_wheelhouse_path(wheelhouse_prepare),
                    runtime_evidence=runtime_evidence,
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
        if args.skip_install:
            result["worldsculpt_lane"] = {
                "status": "skipped",
                "reason": "installation disabled by --skip-install",
            }
        elif not install_failed and (workspace_root / "worldsculpt-wheelhouse.manifest.json").is_file() and (workspace_root / "venv/bin/python").is_file() and sys.platform == "linux" and platform.machine() == "aarch64":
            from pixal3d_extension import worldsculpt_lane
            try:
                info = worldsculpt_lane._probe(workspace_root / "venv/bin/python")
                try:
                    worldsculpt_lane._platform_gate(info)
                except RuntimeError:
                    result["worldsculpt_lane"] = {"status": "unsupported-platform"}
                else:
                    result["worldsculpt_lane"] = worldsculpt_lane.repair(workspace_root)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                result.update(status="failed", failure_code="worldsculpt_lane_failed", reason=str(exc))
        if not install_failed and not args.skip_install and not args.skip_scene_prep:
            from pixal3d_extension import scene_prepare_lane
            support = scene_prepare_lane.capability()
            if not support["supported"]:
                result["scene_prep_lane"] = support
            else:
                try:
                    result["scene_prep_lane"] = scene_prepare_lane.repair(workspace_root)
                    result["installs_started"] = True
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                    result["scene_prep_lane"] = {
                        "status": "unprepared",
                        "supported": True,
                        "reason": str(exc),
                    }
                    result["installs_started"] = True
                    result["next_steps"] = [
                        *result.get("next_steps", []),
                        "install Python 3.12 and a CUDA >=12.6 runtime for scene-from-images/scene-from-video",
                        "rerun setup.py --repair-scene-prep",
                    ]
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
