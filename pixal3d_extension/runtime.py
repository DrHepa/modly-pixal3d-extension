from __future__ import annotations

import importlib
import os
import random
import time
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
        if pipeline_factory is not None:
            pipeline = pipeline_factory(job.get("model_source") or PIXAL3D_MODEL_SOURCE)
            result = pipeline(
                image_path=str(image_path),
                output_dir=str(output_dir),
                seed=params.get("seed"),
                resolution=params.get("resolution", 1024),
                low_vram=params.get("low_vram", False),
                manual_fov=params.get("manual_fov"),
            )
        else:
            _prepare_runtime_compat()
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
                low_vram=bool(params.get("low_vram", False)),
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
            "low_vram": params.get("low_vram", False),
            "manual_fov": params.get("manual_fov"),
        },
    }
