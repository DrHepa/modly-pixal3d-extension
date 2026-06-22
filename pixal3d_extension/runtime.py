from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.machinery
import faulthandler
import os
import random
import re
import sys
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Any, Callable

from pixal3d_extension.assets import bootstrap_auxiliary_assets, normalize_auxiliary_source_mode, resolve_auxiliary_sources, requires_local_auxiliary
from pixal3d_extension.paths import derive_modly_home, is_contained_path, require_contained_path
from pixal3d_extension.pipeline_patch import patch_pipeline, validate_pipeline_patch

PIXAL3D_MODEL_SOURCE = "TencentARC/Pixal3D"
PIXAL3D_TEXTURE_SIZE_ENV = "PIXAL3D_TEXTURE_SIZE"
SUPPORTED_TEXTURE_SIZES = (1024, 2048)
DEFAULT_TEXTURE_SIZE = 1024
SUPPORTED_RUNTIME_LANES = {
    "linux-aarch64-cp312-cuda124",
    "linux-x64-cp312-cuda124",
    "windows-x64-cp311-cuda124",
    "windows-x64-cp312-cuda124",
}
SUPPORTED_RUNTIME_LANE = "linux-aarch64-cp312-cuda124"


def _enable_crash_diagnostics() -> None:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        return


def _diagnostic_checkpoint(label: str) -> None:
    print(f"[Pixal3D Diagnostic] {label}", file=sys.stderr, flush=True)


def _default_pipeline_factory(source: str) -> Any:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    module = importlib.import_module("pixal3d.pipelines")
    return module.Pixal3DImageTo3DPipeline.from_pretrained(source)


def _prepare_runtime_compat() -> None:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")
    import torch

    if not hasattr(torch.nn.Module, "all_tied_weights_keys"):
        torch.nn.Module.all_tied_weights_keys = {}


def _install_windows_native_module_aliases() -> None:
    """Expose Windows exact-stack native wheels under upstream module names.

    The Windows wheels are published as alias packages (`cumesh_vb`,
    `flex_gemm_ap`, `o_voxel_vb_ap`) while Pixal3D upstream imports the Linux
    module names (`cumesh`, `flex_gemm`, `o_voxel`). Keep the installed package
    names exact-stack, but provide import aliases before importing upstream
    inference code.
    """

    if os.name != "nt":
        return

    aliases = {
        "cumesh": "cumesh_vb",
        "flex_gemm": "flex_gemm_ap",
        "o_voxel": "o_voxel_vb_ap",
    }
    for upstream_name, windows_name in aliases.items():
        if upstream_name in sys.modules:
            if upstream_name == "o_voxel":
                _install_windows_o_voxel_python_compat(sys.modules[upstream_name])
            continue
        try:
            module = importlib.import_module(windows_name)
            sys.modules[upstream_name] = module
            if upstream_name == "o_voxel":
                _install_windows_o_voxel_python_compat(module)
        except ModuleNotFoundError:
            continue


def _install_windows_o_voxel_python_compat(o_voxel_module: Any) -> None:
    """Expose Python modules omitted by the Windows o_voxel alias wheel.

    `o_voxel_vb_ap` ships the native `_C` extension and conversion helpers but
    omits upstream Python modules used by Pixal3D (`o_voxel.postprocess` and
    `o_voxel.rasterize`). Install package attributes and submodule aliases so
    upstream imports keep using `o_voxel.*` while native calls resolve to the
    Windows wheel.
    """

    if os.name != "nt":
        return

    if "o_voxel._C" not in sys.modules:
        try:
            sys.modules["o_voxel._C"] = importlib.import_module("o_voxel_vb_ap._C")
            setattr(o_voxel_module, "_C", sys.modules["o_voxel._C"])
        except ModuleNotFoundError:
            pass

    compat_modules = {
        "postprocess": "pixal3d_extension.o_voxel_compat.postprocess",
        "rasterize": "pixal3d_extension.o_voxel_compat.rasterize",
    }
    for attr_name, compat_name in compat_modules.items():
        full_name = f"o_voxel.{attr_name}"
        if full_name in sys.modules:
            setattr(o_voxel_module, attr_name, sys.modules[full_name])
            continue
        compat_module = importlib.import_module(compat_name)
        sys.modules[full_name] = compat_module
        setattr(o_voxel_module, attr_name, compat_module)


def _import_module_with_torch_compile_disabled(module_name: str) -> Any:
    """Import a module while suppressing torch.compile import-time side effects.

    NATTEN v0.21.x decorates some helpers with torch.compile at import time.
    On the Modly Windows cp311/cu124 lane this can enter torch._inductor and a
    Triton package whose public API does not expose AttrsDescriptor. The native
    libnatten kernels are still useful for NAF; the compile-time decorator is not
    required for Pixal3D generation. Keep the patch scoped to the import so the
    rest of PyTorch keeps its normal behavior.
    """

    if os.name != "nt":
        return importlib.import_module(module_name)

    try:
        import torch
    except Exception:
        return importlib.import_module(module_name)

    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return importlib.import_module(module_name)

    def identity_compile(function: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if function is None:
            return lambda inner: inner
        return function

    torch.compile = identity_compile
    try:
        return importlib.import_module(module_name)
    finally:
        torch.compile = original_compile


def _install_natten_fallback() -> None:
    """Provide a slow but correct top-level natten fallback when absent.

    Upstream NAF first tries `natten.functional` and falls back to `from natten
    import na2d` on any import error. We intentionally only provide the top-level
    module so the legacy import path still fails and upstream selects the recent
    `na2d(...)` call shape.
    """

    if "natten" in sys.modules:
        return

    try:
        _import_module_with_torch_compile_disabled("natten")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "natten":
            return

    import torch
    import torch.nn.functional as torch_functional

    def na2d(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: Any,
        dilation: Any,
        stride: int = 1,
        backend: str | None = None,
    ) -> torch.Tensor:
        del backend
        if stride != 1:
            raise NotImplementedError("natten fallback only supports stride=1")
        if q.ndim != 5 or k.ndim != 5 or v.ndim != 5:
            raise ValueError("natten fallback expects q/k/v shaped as [b, h, w, n, d]")
        if q.shape != k.shape:
            raise ValueError("natten fallback expects q and k to share the same shape")
        if v.shape[:4] != q.shape[:4]:
            raise ValueError("natten fallback expects v to share q/k [b, h, w, n] dimensions")

        def normalize_pair(value: Any, name: str) -> tuple[int, int]:
            if isinstance(value, int):
                pair = (value, value)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                pair = (int(value[0]), int(value[1]))
            else:
                raise ValueError(f"natten fallback expected {name} to be an int or length-2 tuple")

            if pair[0] <= 0 or pair[1] <= 0:
                raise ValueError(f"natten fallback expected positive {name}")
            return pair

        kernel_h, kernel_w = normalize_pair(kernel_size, "kernel_size")
        dilation_h, dilation_w = normalize_pair(dilation, "dilation")
        if kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError("natten fallback requires odd kernel_size values")

        b, h, w, n, d_qk = q.shape
        d_v = v.shape[-1]
        bn = b * n
        neighborhood_size = kernel_h * kernel_w
        padding = ((kernel_h // 2) * dilation_h, (kernel_w // 2) * dilation_w)

        q_heads = q.permute(0, 3, 4, 1, 2).reshape(bn, d_qk, h, w)
        k_heads = k.permute(0, 3, 4, 1, 2).reshape(bn, d_qk, h, w)
        v_heads = v.permute(0, 3, 4, 1, 2).reshape(bn, d_v, h, w)

        q_dense = q_heads.reshape(bn, d_qk, h * w).transpose(1, 2)
        k_neighbors = torch_functional.unfold(
            k_heads,
            kernel_size=(kernel_h, kernel_w),
            dilation=(dilation_h, dilation_w),
            padding=padding,
            stride=1,
        ).reshape(bn, d_qk, neighborhood_size, h * w).permute(0, 3, 2, 1)
        v_neighbors = torch_functional.unfold(
            v_heads,
            kernel_size=(kernel_h, kernel_w),
            dilation=(dilation_h, dilation_w),
            padding=padding,
            stride=1,
        ).reshape(bn, d_v, neighborhood_size, h * w).permute(0, 3, 2, 1)

        valid_neighbors = torch_functional.unfold(
            torch.ones((1, 1, h, w), device=q.device, dtype=q.dtype),
            kernel_size=(kernel_h, kernel_w),
            dilation=(dilation_h, dilation_w),
            padding=padding,
            stride=1,
        ).reshape(1, neighborhood_size, h * w).transpose(1, 2) > 0

        scores = (q_dense.unsqueeze(2) * k_neighbors).sum(dim=-1) * (1.0 / math.sqrt(d_qk))
        scores = scores.masked_fill(~valid_neighbors, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        out = (attention.unsqueeze(-1) * v_neighbors).sum(dim=2)

        return out.transpose(1, 2).reshape(b, n, d_v, h, w).permute(0, 3, 4, 1, 2)

    natten_module = types.ModuleType("natten")
    natten_module.__loader__ = None
    natten_module.__package__ = "natten"
    natten_module.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None, is_package=False)
    natten_module.HAS_LIBNATTEN = False
    natten_module.na2d = na2d
    sys.modules["natten"] = natten_module


def _silence_flex_gemm_autotuners() -> None:
    """Keep FlexGEMM autotuning/cache enabled but prevent upstream verbose prints.

    Pixal3D upstream's inference module sets FLEX_GEMM_AUTOTUNER_VERBOSE=1 at
    import time. That is useful in a terminal, but Modly's subprocess bridge
    treats stdout as a structured protocol channel. Existing autotuner instances
    capture the verbose flag during construction, so resetting the environment
    alone is not enough after importing inference.py.
    """

    os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"
    try:
        import flex_gemm
        from flex_gemm.utils.autotuner import PersistentCacheAutoTuner
    except Exception:
        return

    def silence_module(module_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return
        for value in vars(module).values():
            if isinstance(value, PersistentCacheAutoTuner):
                value.verbose = False

    try:
        for package in ("flex_gemm.kernels.triton.spconv", "flex_gemm.kernels.triton.grid_sample"):
            root = importlib.import_module(package)
            package_path = getattr(root, "__path__", None)
            if package_path is None:
                continue
            import pkgutil

            for module in pkgutil.iter_modules(package_path, f"{package}."):
                silence_module(module.name)
        flex_gemm.utils.load_autotune_cache()
    except Exception:
        return


def _failure(code: str, message: str, **extra: Any) -> dict:
    return {"status": "failed", "code": code, "message": message, "generation_allowed": False, **extra}


_RUNTIME_ENV_KEYS = (
    "TEMP",
    "TMP",
    "TMPDIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"""['\"](?P<path>[A-Za-z]:(?:[\\/][^'\"]*)?)['\"]""")
_WINDOWS_DRIVE_PATH_FALLBACK_RE = re.compile(r"(?P<path>[A-Za-z]:[\\/]\S*)")


def _selected_runtime_env() -> dict:
    return {key: os.environ[key] for key in _RUNTIME_ENV_KEYS if key in os.environ}


def _runtime_context(job: dict, image_path: Path, output_dir: Path) -> dict:
    return {
        "cwd": os.getcwd(),
        "env": _selected_runtime_env(),
        "input_image": str(image_path),
        "output_dir": str(output_dir),
        "model_source": str(job.get("model_source") or PIXAL3D_MODEL_SOURCE),
    }


def _looks_like_windows_missing_path(exc: Exception) -> bool:
    message = str(exc).lower()
    if "winerror 3" in message:
        return True
    if not isinstance(exc, OSError):
        return False
    return any(
        marker in message
        for marker in (
            "cannot find the path",
            "cannot find path",
            "system cannot find the path specified",
            "impossibile trovare il percorso",
            "non riesce a trovare il percorso",
        )
    )


def _extract_windows_missing_path(exc: Exception) -> str | None:
    if not _looks_like_windows_missing_path(exc):
        return None
    message = str(exc)
    match = _WINDOWS_DRIVE_PATH_RE.search(message) or _WINDOWS_DRIVE_PATH_FALLBACK_RE.search(message)
    if match is None:
        return None
    return match.group("path").rstrip(".,;)")


def _runtime_failure(exc: Exception, checkpoint: str, job: dict, image_path: Path, output_dir: Path) -> dict:
    result = _failure(
        "runtime_failed",
        str(exc),
        checkpoint=checkpoint,
        exception_type=type(exc).__name__,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        runtime_context=_runtime_context(job, image_path, output_dir),
    )
    failed_path = _extract_windows_missing_path(exc)
    if failed_path is not None:
        result["path_hint"] = {
            "code": "missing_windows_path",
            "message": "A configured runtime/cache/model/input/output path appears to reference a missing Windows drive or root.",
            "path": failed_path,
        }
    return result


def _parse_low_vram(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "low_vram", "low-vram", "low vram"}:
            return True
        if normalized in {"false", "0", "no", "off", "standard"}:
            return False
        return default
    return default


def _parse_texture_size(value: Any, default: int = DEFAULT_TEXTURE_SIZE) -> int:
    safe_default = default if default in SUPPORTED_TEXTURE_SIZES else DEFAULT_TEXTURE_SIZE
    if value is None:
        return safe_default
    if isinstance(value, bool):
        return safe_default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return safe_default
    else:
        return safe_default
    return parsed if parsed in SUPPORTED_TEXTURE_SIZES else safe_default


@contextmanager
def _scoped_texture_size_env(texture_size: Any):
    parsed_texture_size = _parse_texture_size(texture_size)
    previous = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
    os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = str(parsed_texture_size)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PIXAL3D_TEXTURE_SIZE_ENV, None)
        else:
            os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = previous


def _resolve_glb(output_dir: Path, pipeline_result: Any) -> Path | None:
    if isinstance(pipeline_result, dict) and pipeline_result.get("glb_path"):
        candidate = Path(pipeline_result["glb_path"])
        return candidate if candidate.is_file() and is_contained_path(output_dir, candidate) else None
    matches = sorted(output_dir.glob("*.glb"))
    return matches[0] if matches else None


def _resolve_workspace_output_dir(workspace_root: str | Path, output_dir: str) -> Path | dict:
    try:
        candidate = require_contained_path(workspace_root, output_dir, allow_absolute=True)
    except ValueError:
        return _failure("unsafe_path", "output_dir must be contained in the workspace root")

    base = Path(workspace_root)
    allowed_roots = (base / "outputs", base / "workspace")
    if not any(is_contained_path(allowed_root, candidate) for allowed_root in allowed_roots):
        return _failure("unsafe_path", "output_dir must be contained in the workspace outputs or workspace root")
    return candidate


def _resolve_job_paths(job: dict) -> tuple[Path, Path] | dict:
    workspace_root = job.get("workspace_root")
    if not workspace_root:
        image_path = Path(job.get("input_image", ""))
        output_dir = Path(job.get("output_dir", ""))
        if not str(output_dir):
            return _failure("output_missing", "output_dir is required")
        return image_path, output_dir

    try:
        image_path = require_contained_path(workspace_root, job.get("input_image", ""), allow_absolute=True)
    except ValueError:
        return _failure("unsafe_path", "input_image must be contained in the workspace root")

    output_dir = _resolve_workspace_output_dir(workspace_root, job.get("output_dir", ""))
    if isinstance(output_dir, dict):
        return output_dir

    return image_path, output_dir


def _job_auxiliary_mode(job: dict) -> str:
    params = job.get("params") or {}
    if job.get("offline") is True or params.get("offline") is True:
        return "offline"
    return str(job.get("auxiliary_mode") or params.get("auxiliary_mode") or "default")


def _job_network_available(job: dict, auxiliary_mode: str) -> bool:
    params = job.get("params") or {}
    if "network_available" in job:
        return bool(job["network_available"])
    if "network_available" in params:
        return bool(params["network_available"])
    return auxiliary_mode not in {"offline"}


def _job_auxiliary_bootstrap_downloader(job: dict) -> Any:
    params = job.get("params") or {}
    return job.get("auxiliary_bootstrap_downloader") or params.get("auxiliary_bootstrap_downloader")


def _job_modly_home(job: dict) -> str | Path | None:
    return derive_modly_home(
        model_dir=job.get("model_source"),
        workspace_dir=job.get("workspace_root") or job.get("output_dir"),
    ) or job.get("workspace_root")


def _with_auxiliary_bootstrap_metadata(auxiliary_source: dict, bootstrap_result: dict | None) -> dict:
    if bootstrap_result is None:
        return auxiliary_source
    warnings = list(auxiliary_source.get("warnings", []))
    if bootstrap_result.get("status") != "ready":
        warnings.append(
            {
                "code": "auxiliary_bootstrap_failed_remote_fallback_preserved",
                "message": "explicit first-run auxiliary bootstrap failed; preserving remote/HF-cache fallback",
            }
        )
    enriched = {**auxiliary_source, "auxiliary_bootstrap": bootstrap_result}
    if warnings:
        enriched["warnings"] = warnings
    return enriched


def _patch_pipeline_for_runtime(
    workspace_root: str | Path,
    *,
    auxiliary_mode: str,
    network_available: bool,
    auxiliary_source: dict,
    bootstrap_result: dict | None,
) -> tuple[dict | None, dict | None]:
    patch_result = patch_pipeline(
        workspace_root,
        auxiliary_mode=auxiliary_mode,
        network_available=network_available,
    )
    patched_auxiliary_source = _with_auxiliary_bootstrap_metadata(
        patch_result.get("auxiliary_source", auxiliary_source),
        bootstrap_result,
    )
    if patch_result.get("status") != "patched":
        return (
            _failure(
                patch_result.get("code", "pipeline_substitution_required"),
                patch_result.get("message", "Pixal3D pipeline substitutions are required before generation"),
                auxiliary_mode=auxiliary_mode,
                pipeline_patch=patch_result,
                auxiliary_source=patched_auxiliary_source,
            ),
            None,
        )
    return None, patched_auxiliary_source


def _preflight_auxiliary_sources(job: dict) -> tuple[dict | None, dict | None]:
    auxiliary_mode = _job_auxiliary_mode(job)
    try:
        normalized_auxiliary_mode = normalize_auxiliary_source_mode(auxiliary_mode)
        local_required = requires_local_auxiliary(normalized_auxiliary_mode)
    except ValueError as exc:
        return _failure("invalid_auxiliary_mode", str(exc), auxiliary_mode=auxiliary_mode), None
    network_available = _job_network_available(job, normalized_auxiliary_mode)
    workspace_root = _job_modly_home(job)

    if not workspace_root:
        if local_required:
            return (
                _failure(
                    "missing_auxiliary_assets",
                    "workspace_root is required to validate local Pixal3D auxiliary assets",
                    auxiliary_mode=normalized_auxiliary_mode,
                    missing=[],
                ),
                None,
            )
        return None, None

    try:
        auxiliary_source = resolve_auxiliary_sources(
            workspace_root,
            mode=normalized_auxiliary_mode,
            network_available=network_available,
        )
    except ValueError as exc:
        return _failure("invalid_auxiliary_mode", str(exc), auxiliary_mode=normalized_auxiliary_mode), None

    if auxiliary_source["status"] == "blocked":
        return (
            _failure(
                "missing_auxiliary_assets",
                "local Pixal3D auxiliary assets are missing for the requested mode",
                auxiliary_mode=normalized_auxiliary_mode,
                missing=auxiliary_source.get("missing", []),
                auxiliary_source=auxiliary_source,
            ),
            None,
        )

    bootstrap_result = None
    if normalized_auxiliary_mode == "default" and network_available and auxiliary_source.get("status") == "fallback":
        bootstrap_result = bootstrap_auxiliary_assets(
            workspace_root,
            downloader=_job_auxiliary_bootstrap_downloader(job),
        )
        if bootstrap_result.get("status") == "ready":
            auxiliary_source = resolve_auxiliary_sources(
                workspace_root,
                mode=normalized_auxiliary_mode,
                network_available=network_available,
            )
        patch_error, patched_auxiliary_source = _patch_pipeline_for_runtime(
            workspace_root,
            auxiliary_mode=normalized_auxiliary_mode,
            network_available=network_available,
            auxiliary_source=auxiliary_source,
            bootstrap_result=bootstrap_result,
        )
        if patch_error is not None:
            return patch_error, None
        return None, patched_auxiliary_source

    patch_result = validate_pipeline_patch(
        workspace_root,
        auxiliary_mode=normalized_auxiliary_mode,
        network_available=network_available,
    )
    if patch_result["status"] != "ready":
        return (
            _failure(
                patch_result.get("code", "pipeline_substitution_required"),
                patch_result.get("message", "Pixal3D pipeline substitutions are required before generation"),
                auxiliary_mode=normalized_auxiliary_mode,
                pipeline_patch=patch_result,
                auxiliary_source=auxiliary_source,
            ),
            None,
        )

    unresolved_runtime_dependencies = auxiliary_source.get("unlocalized_runtime_dependencies", [])
    if normalized_auxiliary_mode in {"offline", "strict"} and unresolved_runtime_dependencies and not bool(job.get("allow_remote_runtime_dependencies", False)):
        return (
            _failure(
                "offline_runtime_dependencies_unresolved",
                "Some Pixal3D runtime dependencies still rely on cache or network fallback behavior.",
                auxiliary_mode=normalized_auxiliary_mode,
                auxiliary_source=auxiliary_source,
                runtime_dependencies=auxiliary_source.get("unlocalized_runtime_dependencies", []),
            ),
            None,
        )

    return None, auxiliary_source


def _preflight_runtime(job: dict) -> tuple[dict | None, dict | None]:
    readiness = job.get("readiness")
    if readiness is None:
        return _failure("readiness_required", "Pixal3D readiness must pass before generation"), None
    if readiness.get("generation_allowed") is not True:
        result = _failure(
            readiness.get("code", "readiness_required"),
            readiness.get("message", "Pixal3D readiness must pass before generation"),
        )
        if "dependency_diagnostics" in readiness:
            result["dependency_diagnostics"] = readiness["dependency_diagnostics"]
        return result, None

    auxiliary_error, auxiliary_source = _preflight_auxiliary_sources(job)
    if auxiliary_error is not None:
        return auxiliary_error, None

    runtime_lane = job.get("runtime_lane")
    if runtime_lane and runtime_lane not in SUPPORTED_RUNTIME_LANES:
        return _failure("unsupported_lane", f"runtime lane {runtime_lane!r} is not production-supported"), auxiliary_source

    asset_readiness = job.get("asset_readiness") or {}
    if asset_readiness.get("code") == "missing_assets" or job.get("assets_ready") is False:
        result = _failure("missing_assets", "required Pixal3D assets are missing")
        result["missing"] = asset_readiness.get("missing", [])
        return result, auxiliary_source

    return None, auxiliary_source


def _patch_inference_dino_source(inference_module: Any, auxiliary_source: dict) -> None:
    dino_source = auxiliary_source.get("sources", {}).get("dino", {})
    if dino_source.get("kind") != "local":
        return

    local_dino_path = str(dino_source["value"])
    configs = getattr(inference_module, "IMAGE_COND_CONFIGS", None)
    if isinstance(configs, dict):
        iterable = configs.values()
    elif isinstance(configs, (list, tuple)):
        iterable = configs
    else:
        return

    for config in iterable:
        if isinstance(config, dict) and "model_name" in config:
            config["model_name"] = local_dino_path
        elif hasattr(config, "model_name"):
            setattr(config, "model_name", local_dino_path)


def _patch_inference_moge_loader(inference_module: Any, auxiliary_source: dict) -> None:
    moge_source = auxiliary_source.get("sources", {}).get("moge", {})
    if moge_source.get("kind") != "local":
        return

    local_moge_path = str(moge_source["value"])
    load_moge_model = getattr(inference_module, "load_moge_model", None)
    if not callable(load_moge_model):
        return

    setattr(inference_module, "MOGE_MODEL_NAME", local_moge_path)

    def load_local_moge_model(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if len(args_list) >= 2:
            args_list[1] = local_moge_path
            kwargs.pop("model_name", None)
        else:
            kwargs["model_name"] = local_moge_path
        return load_moge_model(*args_list, **kwargs)

    load_local_moge_model.__name__ = getattr(load_moge_model, "__name__", "load_moge_model")
    load_local_moge_model.__doc__ = getattr(load_moge_model, "__doc__", None)
    setattr(inference_module, "load_moge_model", load_local_moge_model)


def _patch_hubconf_naf_loader(auxiliary_source: dict | None) -> None:
    if not auxiliary_source:
        return
    naf_source = auxiliary_source.get("sources", {}).get("naf", {})
    if naf_source.get("kind") != "local":
        return

    local_naf_path = str(naf_source["value"])
    fallback_allowed = auxiliary_source.get("mode") == "default"
    hubconf_module = importlib.import_module("hubconf")
    original_naf = getattr(hubconf_module, "naf", None)
    if not callable(original_naf):
        raise RuntimeError("hubconf.naf is required for local NAF checkpoint loading")
    if getattr(original_naf, "__modly_local_checkpoint__", None) == local_naf_path:
        return

    def load_local_naf(pretrained: bool = True, device: Any = "cpu") -> Any:
        try:
            naf_cls = getattr(hubconf_module, "NAF")
            model = naf_cls().to(device)
            if pretrained:
                import torch

                model.load_state_dict(torch.load(local_naf_path, map_location=device))
            return model
        except Exception:
            if fallback_allowed:
                return original_naf(pretrained=pretrained, device=device)
            raise

    load_local_naf.__name__ = getattr(original_naf, "__name__", "naf")
    load_local_naf.__doc__ = getattr(original_naf, "__doc__", None)
    load_local_naf.__wrapped__ = original_naf  # type: ignore[attr-defined]
    load_local_naf.__modly_local_checkpoint__ = local_naf_path  # type: ignore[attr-defined]
    setattr(hubconf_module, "naf", load_local_naf)


def _patch_inference_auxiliary_sources(inference_module: Any, auxiliary_source: dict | None) -> None:
    if not auxiliary_source:
        return
    _patch_hubconf_naf_loader(auxiliary_source)
    _patch_inference_dino_source(inference_module, auxiliary_source)
    _patch_inference_moge_loader(inference_module, auxiliary_source)


def run_job(job: dict, *, pipeline_factory: Callable[[str], Any] | None = None) -> dict:
    _enable_crash_diagnostics()
    _diagnostic_checkpoint("run_job:start")
    resolved_paths = _resolve_job_paths(job)
    if isinstance(resolved_paths, dict):
        return resolved_paths
    image_path, output_dir = resolved_paths

    if not image_path.is_file():
        return _failure("invalid_image", "input_image must reference an existing local image file")

    preflight_error, auxiliary_source = _preflight_runtime(job)
    if preflight_error is not None:
        return preflight_error

    checkpoint = "runtime_block"
    try:
        checkpoint = "preflight_ok"
        _diagnostic_checkpoint("run_job:preflight:ok")
        params = job.get("params") or {}
        parsed_low_vram = _parse_low_vram(params.get("low_vram"), default=True)
        parsed_texture_size = _parse_texture_size(params.get("texture_size"), default=DEFAULT_TEXTURE_SIZE)
        if pipeline_factory is not None:
            checkpoint = "pipeline_factory_create"
            _diagnostic_checkpoint("pipeline_factory:create:start")
            pipeline = pipeline_factory(job.get("model_source") or PIXAL3D_MODEL_SOURCE)
            _diagnostic_checkpoint("pipeline_factory:create:done")
            checkpoint = "pipeline_factory_call"
            _diagnostic_checkpoint("pipeline_factory:call:start")
            result = pipeline(
                image_path=str(image_path),
                output_dir=str(output_dir),
                seed=params.get("seed"),
                resolution=params.get("resolution", 1024),
                low_vram=parsed_low_vram,
                texture_size=parsed_texture_size,
                manual_fov=params.get("manual_fov"),
            )
            _diagnostic_checkpoint("pipeline_factory:call:done")
        else:
            checkpoint = "runtime_compat"
            _diagnostic_checkpoint("runtime_compat:start")
            _prepare_runtime_compat()
            _diagnostic_checkpoint("runtime_compat:done")
            checkpoint = "windows_aliases"
            _diagnostic_checkpoint("windows_aliases:start")
            _install_windows_native_module_aliases()
            _diagnostic_checkpoint("windows_aliases:done")
            checkpoint = "natten"
            _diagnostic_checkpoint("natten:start")
            _install_natten_fallback()
            _diagnostic_checkpoint("natten:done")
            checkpoint = "naf_loader_patch"
            _diagnostic_checkpoint("naf_loader_patch:start")
            _patch_hubconf_naf_loader(auxiliary_source)
            _diagnostic_checkpoint("naf_loader_patch:done")
            checkpoint = "import_inference"
            _diagnostic_checkpoint("import_inference:start")
            inference_module = importlib.import_module("inference")
            _patch_inference_auxiliary_sources(inference_module, auxiliary_source)
            run_inference = inference_module.run_inference
            _diagnostic_checkpoint("import_inference:done")

            checkpoint = "flex_gemm_silence"
            _silence_flex_gemm_autotuners()
            _diagnostic_checkpoint("flex_gemm_silence:done")

            checkpoint = "prepare_inference_args"
            seed = int(params.get("seed", -1))
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            glb_path = output_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_pixal3d.glb"
            checkpoint = "run_inference"
            _diagnostic_checkpoint("run_inference:start")
            with _scoped_texture_size_env(parsed_texture_size):
                run_inference(
                    image_path=str(image_path),
                    output_path=str(glb_path),
                    seed=seed,
                    model_path=job.get("model_source") or PIXAL3D_MODEL_SOURCE,
                    manual_fov=float(params.get("manual_fov") or -1.0),
                    low_vram=parsed_low_vram,
                    resolution=int(params.get("resolution", 1024)),
                )
            _diagnostic_checkpoint("run_inference:done")
            result = {"glb_path": str(glb_path), "pbr": {}}
    except Exception as exc:  # pragma: no cover - contract retained for real runtime failures.
        _diagnostic_checkpoint(f"runtime_exception:{type(exc).__name__}")
        return _runtime_failure(exc, checkpoint, job, image_path, output_dir)

    glb_path = _resolve_glb(output_dir, result)
    if glb_path is None:
        return _failure("output_missing", "Pixal3D runtime did not produce a GLB output")

    pbr = result.get("pbr", {}) if isinstance(result, dict) else {}
    return {
        "status": "completed",
        "model_source": PIXAL3D_MODEL_SOURCE,
        "auxiliary_source": auxiliary_source,
        "output": {"glb_path": str(glb_path), "pbr": pbr},
        "params": {
            "seed": params.get("seed"),
            "resolution": params.get("resolution", 1024),
            "low_vram": parsed_low_vram,
            "texture_size": parsed_texture_size,
            "manual_fov": params.get("manual_fov"),
        },
    }
