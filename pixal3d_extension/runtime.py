from __future__ import annotations

import importlib
import importlib.machinery
import math
import os
import random
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any, Callable

from pixal3d_extension.paths import is_contained_path, require_contained_path

PIXAL3D_MODEL_SOURCE = "TencentARC/Pixal3D"
SUPPORTED_RUNTIME_LANES = {
    "linux-aarch64-cp312-cuda124",
    "linux-x64-cp312-cuda124",
    "windows-x64-cp311-cuda124",
    "windows-x64-cp312-cuda124",
}
SUPPORTED_RUNTIME_LANE = "linux-aarch64-cp312-cuda124"


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
            continue
        try:
            sys.modules[upstream_name] = importlib.import_module(windows_name)
        except ModuleNotFoundError:
            continue


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
        importlib.import_module("natten")
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


def _failure(code: str, message: str) -> dict:
    return {"status": "failed", "code": code, "message": message, "generation_allowed": False}


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


def _resolve_glb(output_dir: Path, pipeline_result: Any) -> Path | None:
    if isinstance(pipeline_result, dict) and pipeline_result.get("glb_path"):
        candidate = Path(pipeline_result["glb_path"])
        return candidate if candidate.is_file() and is_contained_path(output_dir, candidate) else None
    matches = sorted(output_dir.glob("*.glb"))
    return matches[0] if matches else None


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

    try:
        output_dir = require_contained_path(workspace_root, job.get("output_dir", ""), allowed_root=Path(workspace_root) / "outputs")
    except ValueError:
        return _failure("unsafe_path", "output_dir must be contained in the workspace outputs root")

    return image_path, output_dir


def _preflight_runtime(job: dict) -> dict | None:
    readiness = job.get("readiness")
    if readiness is None:
        return _failure("readiness_required", "Pixal3D readiness must pass before generation")
    if readiness.get("generation_allowed") is not True:
        result = _failure(
            readiness.get("code", "readiness_required"),
            readiness.get("message", "Pixal3D readiness must pass before generation"),
        )
        if "dependency_diagnostics" in readiness:
            result["dependency_diagnostics"] = readiness["dependency_diagnostics"]
        return result

    runtime_lane = job.get("runtime_lane")
    if runtime_lane and runtime_lane not in SUPPORTED_RUNTIME_LANES:
        return _failure("unsupported_lane", f"runtime lane {runtime_lane!r} is not production-supported")

    asset_readiness = job.get("asset_readiness") or {}
    if asset_readiness.get("code") == "missing_assets" or job.get("assets_ready") is False:
        result = _failure("missing_assets", "required Pixal3D assets are missing")
        result["missing"] = asset_readiness.get("missing", [])
        return result

    return None


def run_job(job: dict, *, pipeline_factory: Callable[[str], Any] | None = None) -> dict:
    resolved_paths = _resolve_job_paths(job)
    if isinstance(resolved_paths, dict):
        return resolved_paths
    image_path, output_dir = resolved_paths

    if not image_path.is_file():
        return _failure("invalid_image", "input_image must reference an existing local image file")

    preflight_error = _preflight_runtime(job)
    if preflight_error is not None:
        return preflight_error

    try:
        params = job.get("params") or {}
        parsed_low_vram = _parse_low_vram(params.get("low_vram"), default=True)
        if pipeline_factory is not None:
            pipeline = pipeline_factory(job.get("model_source") or PIXAL3D_MODEL_SOURCE)
            result = pipeline(
                image_path=str(image_path),
                output_dir=str(output_dir),
                seed=params.get("seed"),
                resolution=params.get("resolution", 1024),
                low_vram=parsed_low_vram,
                manual_fov=params.get("manual_fov"),
            )
        else:
            _prepare_runtime_compat()
            _install_windows_native_module_aliases()
            _install_natten_fallback()
            from inference import run_inference

            _silence_flex_gemm_autotuners()

            seed = int(params.get("seed", -1))
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            glb_path = output_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_pixal3d.glb"
            run_inference(
                image_path=str(image_path),
                output_path=str(glb_path),
                seed=seed,
                model_path=job.get("model_source") or PIXAL3D_MODEL_SOURCE,
                manual_fov=float(params.get("manual_fov") or -1.0),
                low_vram=parsed_low_vram,
                resolution=int(params.get("resolution", 1024)),
            )
            result = {"glb_path": str(glb_path), "pbr": {}}
    except Exception as exc:  # pragma: no cover - contract retained for real runtime failures.
        return _failure("runtime_failed", str(exc))

    glb_path = _resolve_glb(output_dir, result)
    if glb_path is None:
        return _failure("output_missing", "Pixal3D runtime did not produce a GLB output")

    pbr = result.get("pbr", {}) if isinstance(result, dict) else {}
    return {
        "status": "completed",
        "model_source": PIXAL3D_MODEL_SOURCE,
        "output": {"glb_path": str(glb_path), "pbr": pbr},
        "params": {
            "seed": params.get("seed"),
            "resolution": params.get("resolution", 1024),
            "low_vram": parsed_low_vram,
            "manual_fov": params.get("manual_fov"),
        },
    }
