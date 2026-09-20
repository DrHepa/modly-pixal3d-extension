"""Posed multi-view Pixal3D path; separate from WorldSculpt scene composition."""

from __future__ import annotations

import json
import importlib
import inspect
from importlib import metadata
import math
import os
import random
import sys
import tempfile
import threading
import time
import types
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from .naf_checkpoint import NAF_SIZE, NAF_SHA256, verify_naf_checkpoint as _verify_naf_checkpoint


MV_GROUP_ID = "pixal3d-mv"
BASE_DECODER_FILES = (
    "ss_dec_conv3d_16l8_fp16",
    "shape_dec_next_dc_f16c32_fp16",
    "tex_dec_next_dc_f16c32_fp16",
)
MV_MODEL_FILES = (
    "ss_flow_img_dit_1_3B_64_bf16_mv",
    "slat_flow_img2shape_dit_1_3B_512_bf16_mv",
    "slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv",
)
EXPECTED_MODEL_FILES = {
    "sparse_structure_decoder": BASE_DECODER_FILES[0],
    "sparse_structure_flow_model": MV_MODEL_FILES[0],
    "shape_slat_decoder": BASE_DECODER_FILES[1],
    "shape_slat_flow_model_512": MV_MODEL_FILES[1],
    "shape_slat_flow_model_1024": MV_MODEL_FILES[2],
    "tex_slat_decoder": BASE_DECODER_FILES[2],
    "tex_slat_flow_model_1024": MV_MODEL_FILES[3],
}
MV_WEIGHT_FILES = ("pipeline_mv.json",) + tuple(
    f"ckpts/{model}.{extension}"
    for model in MV_MODEL_FILES
    for extension in ("json", "safetensors")
)
AUXILIARY_FILES = (
    "auxiliary/dinov3/config.json",
    "auxiliary/dinov3/preprocessor_config.json",
    "auxiliary/dinov3/model.safetensors",
    "auxiliary/rmbg/config.json",
    "auxiliary/rmbg/preprocessor_config.json",
    "auxiliary/rmbg/BiRefNet_config.py",
    "auxiliary/rmbg/birefnet.py",
    "auxiliary/rmbg/model.safetensors",
)
_MV_RUN_LOCK = threading.RLock()
_MV_CACHE_LOCK = threading.RLock()
_MV_CACHE_AUTHORITY: Path | None = None


def _validate_mv_cache_layout(workspace: Path, targets: tuple[Path, ...]) -> None:
    """Reject files, symlinks, and resolved escapes before cache mutation."""

    if workspace.exists() and not workspace.is_dir():
        raise ValueError("Modly workspace path must be a directory")
    for target in targets:
        if not _within(workspace, target):
            raise ValueError("Pixal3D MV cache must remain within the Modly workspace")
        current = workspace
        for part in target.relative_to(workspace).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(
                    f"Pixal3D MV cache descendant symlink/path component is not allowed: {current}"
                )
            if current.exists() and not current.is_dir():
                raise ValueError(f"Pixal3D MV cache path component must be a directory: {current}")
            if not _within(workspace, current.resolve(strict=False)):
                raise ValueError("Pixal3D MV cache must remain within the Modly workspace")


def _validate_mv_cache_tree(cache_root: Path) -> None:
    """Fail closed on unsafe nodes anywhere in an existing cache tree."""

    if not cache_root.exists():
        return

    pending = [cache_root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ValueError(
                f"Pixal3D MV cache tree cannot be safely inspected: {directory}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    raise ValueError(
                        f"Pixal3D MV cache descendant symlink is not allowed: {path}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError(
                        f"Pixal3D MV cache descendant must be a directory or regular file: {path}"
                    )
            except OSError as exc:
                raise ValueError(
                    f"Pixal3D MV cache descendant cannot be safely inspected: {path}"
                ) from exc


def configure_mv_hf_cache(workspace_dir: str | Path) -> Path:
    """Bind Hugging Face code caches to a writable Modly workspace path.

    RMBG uses ``trust_remote_code`` even when every model file is local. The
    Transformers dynamic-module loader therefore still materializes Python
    modules in ``HF_MODULES_CACHE``. Configure that authority before importing
    Transformers/Pixal3D so a read-only or foreign-owned user cache cannot break
    otherwise fully local inference.

    The cache is intentionally stable for the extension-runner process: once
    Transformers constants are imported they remain process-global. It contains
    generated Python modules only, never model weights.
    """

    global _MV_CACHE_AUTHORITY

    workspace = Path(workspace_dir).expanduser().resolve(strict=False)
    cache_parent = workspace / ".pixal3d-runtime"
    cache = cache_parent / "huggingface"
    paths = {
        "HF_HOME": cache,
        "HF_HUB_CACHE": cache / "hub",
        "HF_MODULES_CACHE": cache / "modules",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "XDG_CACHE_HOME": cache / "xdg",
    }
    transformers_modules = paths["HF_MODULES_CACHE"] / "transformers_modules"
    targets = (cache_parent, cache, *paths.values(), transformers_modules)

    with _MV_CACHE_LOCK:
        # Validate the complete prospective layout before mkdir or environment
        # changes. In particular, an existing child symlink must not turn an
        # otherwise workspace-relative cache into an external write authority.
        _validate_mv_cache_layout(workspace, targets)
        _validate_mv_cache_tree(cache_parent)
        if _MV_CACHE_AUTHORITY is not None and _MV_CACHE_AUTHORITY != cache:
            raise RuntimeError(
                "Pixal3D Hugging Face cache is already bound to a different Modly workspace"
            )

        # Do not pretend an environment update can replace constants already
        # bound by Transformers. The generator initializes this cache as soon
        # as its workspace is known, before any Pixal3D runtime import.
        imported = sys.modules.get("transformers.dynamic_module_utils")
        imported_cache = getattr(imported, "HF_MODULES_CACHE", None) if imported is not None else None
        if imported_cache is not None and Path(imported_cache).resolve() != paths["HF_MODULES_CACHE"]:
            raise RuntimeError(
                "Transformers dynamic modules were imported before the Pixal3D workspace cache was configured"
            )

        workspace.mkdir(parents=True, exist_ok=True)
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)
        # Revalidate the complete custody boundary after creation. This catches
        # unsafe descendants rather than only the configured cache directories.
        _validate_mv_cache_layout(workspace, targets)
        _validate_mv_cache_tree(cache_parent)

        for key, path in paths.items():
            os.environ[key] = str(path)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _MV_CACHE_AUTHORITY = cache
        return cache


def mv_runtime_available() -> bool:
    """Do not report readiness with the published single-view-only core wheel."""

    try:
        distribution = metadata.distribution("pixal3d-core")
    except metadata.PackageNotFoundError:
        return False
    files = {str(path).replace("\\", "/") for path in distribution.files or ()}
    return "pixal3d/pipelines/pixal3d_mv_image_to_3d.py" in files


def _within(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _is_previous_local_ref(value: str, suffix: tuple[str, ...]) -> bool:
    """Allow a prior host root to move, but not an arbitrary relative/HF ref."""

    normalized = value.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    return (
        (Path(value).is_absolute() or PureWindowsPath(value).is_absolute())
        and ".." not in parts
        and tuple(parts[-len(suffix):]) == suffix
    )


def resolve_views_dir(scene_manifest_path: str | Path, workspace_dir: str | Path) -> Path:
    """Validate a Modly scene manifest whose sceneRoot is a posed-view directory.

    The host's scene endpoint carries a manifest path in params, not view bytes.
    Only a directory containing transforms.json plus referenced images is accepted.
    """

    workspace = Path(workspace_dir).resolve()
    scene_file = Path(scene_manifest_path).resolve()
    if not _within(workspace, scene_file) or not scene_file.is_file():
        raise ValueError("scene manifest must be a file within the Modly workspace")
    try:
        scene = json.loads(scene_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("scene manifest is not readable JSON") from exc
    if not isinstance(scene, dict) or scene.get("schema") != "modly.scene-manifest.v1":
        raise ValueError("scene manifest must use modly.scene-manifest.v1")
    scene_root = scene.get("sceneRoot")
    if not isinstance(scene_root, str) or not scene_root.strip():
        raise ValueError("sceneRoot must name a posed-view directory")
    raw_root = Path(scene_root)
    if raw_root.is_absolute() or ".." in raw_root.parts:
        raise ValueError("unsafe sceneRoot path")
    views_dir = (scene_file.parent if scene_root == "." else workspace / raw_root).resolve()
    if not _within(workspace, views_dir) or not views_dir.is_dir():
        raise ValueError("sceneRoot must resolve to a directory within the workspace")
    transforms_path = (views_dir / "transforms.json").resolve()
    if not _within(views_dir, transforms_path) or not transforms_path.is_file():
        raise ValueError("posed-view sceneRoot requires transforms.json")
    try:
        transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("transforms.json is not readable JSON") from exc
    if not isinstance(transforms, dict) or not isinstance(transforms.get("frames"), list) or not transforms["frames"]:
        raise ValueError("transforms.json requires a non-empty frames array")
    for index, frame in enumerate(transforms["frames"]):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {index} must be an object")
        raw_image = frame.get("file_path")
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise ValueError(f"frame {index} requires file_path")
        image_path = Path(raw_image)
        if image_path.is_absolute() or ".." in image_path.parts:
            raise ValueError(f"unsafe frame {index} image path")
        image_file = (views_dir / image_path).resolve()
        if not _within(views_dir, image_file) or not image_file.is_file():
            raise ValueError(f"frame {index} image must exist within sceneRoot")
        matrix = frame.get("transform_matrix")
        if not isinstance(matrix, list) or len(matrix) != 4 or any(
            not isinstance(row, list) or len(row) != 4 or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                for value in row
            ) for row in matrix
        ):
            raise ValueError(f"frame {index} requires a finite 4x4 transform_matrix")
        angle = frame.get("camera_angle_x", transforms.get("camera_angle_x"))
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not math.isfinite(angle) or not 0 < angle < math.pi:
            raise ValueError(f"frame {index} requires camera_angle_x in radians")
    return views_dir


def missing_mv_assets(mv_root: str | Path, base_root: str | Path, naf_path: str | Path) -> list[str]:
    mv = Path(mv_root)
    base = Path(base_root)
    base_files = tuple(f"ckpts/{model}.{extension}" for model in BASE_DECODER_FILES for extension in ("json", "safetensors"))
    missing = [f"{MV_GROUP_ID}/{relative}" for relative in MV_WEIGHT_FILES if not (mv / relative).is_file()]
    missing += [f"pixal3d-base/{relative}" for relative in (*base_files, *AUXILIARY_FILES) if not (base / relative).is_file()]
    if not Path(naf_path).is_file():
        missing.append("auxiliary/naf/naf_release.pth")
    return missing


def _require_custodied_file(root: Path, relative: str, group: str) -> Path:
    """Require a regular, non-symlink file owned by the supplied weight root."""

    candidate = root / relative
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{group}/{relative} must not traverse a symlink")
    if not candidate.is_file():
        raise ValueError(f"{group}/{relative} is missing or is not a regular file")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{group}/{relative} cannot be safely resolved") from exc
    if not _within(root, resolved):
        raise ValueError(f"{group}/{relative} escapes its shared weight group")
    return candidate


def validate_mv_pipeline_config(mv_root: str | Path, base_root: str | Path) -> dict[str, Any]:
    """Validate all non-NAF MV assets and return a locally rewritten config.

    This boundary is intentionally read-only: callers may safely use it before
    deciding whether an absent NAF checkpoint can be bootstrapped.
    """

    root = Path(mv_root).resolve()
    base = Path(base_root).resolve()
    if not root.is_dir() or not base.is_dir():
        raise ValueError("Pixal3D MV and base shared weight groups must be directories")

    for relative in MV_WEIGHT_FILES:
        _require_custodied_file(root, relative, MV_GROUP_ID)
    base_files = tuple(
        f"ckpts/{model}.{extension}"
        for model in BASE_DECODER_FILES
        for extension in ("json", "safetensors")
    )
    for relative in (*base_files, *AUXILIARY_FILES):
        _require_custodied_file(base, relative, "pixal3d-base")

    config = root / "pipeline_mv.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("pipeline_mv.json is not readable JSON") from exc
    if not isinstance(data, dict) or data.get("name") != "Pixal3DMVImageTo3DPipeline":
        raise ValueError("pipeline_mv.json is not the Pixal3D multi-view pipeline")
    args = data.get("args")
    if not isinstance(args, dict):
        raise ValueError("pipeline_mv.json requires an args object")
    models = args.get("models")
    if not isinstance(models, dict) or set(models) != set(EXPECTED_MODEL_FILES) or any(
        not isinstance(value, str) or PurePosixPath(value.replace("\\", "/")).name != EXPECTED_MODEL_FILES[key]
        for key, value in models.items()
    ):
        raise ValueError("pipeline_mv.json references unexpected checkpoint paths")
    for key, value in models.items():
        model_name = PurePosixPath(value.replace("\\", "/")).name
        if model_name in BASE_DECODER_FILES:
            local = str(base / "ckpts" / model_name)
            if value not in {f"ckpts/{model_name}", local} and not _is_previous_local_ref(value, ("ckpts", model_name)):
                raise ValueError(f"unexpected decoder path: {key}")
            models[key] = local
        elif value != f"ckpts/{model_name}":
            raise ValueError(f"unexpected multi-view checkpoint path: {key}")

    rembg_model = args.get("rembg_model")
    rembg_args = rembg_model.get("args") if isinstance(rembg_model, dict) else None
    if not isinstance(rembg_args, dict):
        raise ValueError("pipeline_mv.json requires rembg_model.args")
    local = str(base / "auxiliary" / "rmbg")
    rembg_ref = rembg_args.get("model_name")
    if rembg_ref not in {"briaai/RMBG-2.0", "camenduru/RMBG-2.0", local} and not (
        isinstance(rembg_ref, str) and _is_previous_local_ref(rembg_ref, ("auxiliary", "rmbg"))
    ):
        raise ValueError("pipeline_mv.json references an unexpected matting model")
    rembg_args["model_name"] = local
    return data


def prepare_mv_pipeline_config(mv_root: str | Path, base_root: str | Path, private_config: str | Path) -> Path:
    """Write a private local config without mutating Modly-downloaded weights."""

    root = Path(mv_root).resolve()
    base = Path(base_root).resolve()
    target = Path(private_config).resolve()
    if _within(root, target) or _within(base, target):
        raise ValueError("MV runtime config must not be written into a shared weight group")
    data = validate_mv_pipeline_config(root, base)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target


@contextmanager
def _private_local_loader(mv_root: Path, private_config: Path):
    """Scope upstream's pipeline loader to local files and a private config.

    Upstream joins model_path/config_file and retries failed local model loads
    through Hugging Face. This MV-only override accepts our absolute private
    config and does not retry a corrupt/missing local checkpoint remotely.
    """

    from pixal3d import models
    from pixal3d.pipelines.base import Pipeline

    original = Pipeline.__dict__["from_pretrained"]

    def local_from_pretrained(cls, path: str, config_file: str = "pipeline.json"):
        if Path(path).resolve() != mv_root or Path(config_file).resolve() != private_config:
            raise ValueError("Pixal3D MV loader requires the validated local model root and private config")
        args = json.loads(private_config.read_text(encoding="utf-8"))["args"]
        loaded = {}
        for key, value in args["models"].items():
            if hasattr(cls, "model_names_to_load") and key not in cls.model_names_to_load:
                continue
            checkpoint = Path(value) if Path(value).is_absolute() else mv_root / value
            if not checkpoint.with_suffix(".json").is_file() or not checkpoint.with_suffix(".safetensors").is_file():
                raise RuntimeError(f"local Pixal3D MV checkpoint is missing: {key}")
            loaded[key] = models.from_pretrained(str(checkpoint))
        pipeline = cls(loaded)
        pipeline._pretrained_args = args
        return pipeline

    Pipeline.from_pretrained = classmethod(local_from_pretrained)
    try:
        yield
    finally:
        Pipeline.from_pretrained = original


def _verified_naf_hubconf():
    """Reject a shadow hubconf supplied by cwd/PYTHONPATH instead of the NAF wheel."""

    distribution = metadata.distribution("naf")
    installed = distribution.locate_file("hubconf.py").resolve()
    if "hubconf.py" not in {str(path).replace("\\", "/") for path in distribution.files or ()}:
        raise RuntimeError("installed NAF distribution does not own hubconf.py")
    hubconf = importlib.import_module("hubconf")
    origin = getattr(hubconf, "__file__", None)
    if origin is None or Path(origin).resolve() != installed:
        raise RuntimeError("hubconf.py was not imported from the installed NAF distribution")
    return hubconf


def _load_local_naf(path: Path, device: Any):
    import torch

    hubconf = _verified_naf_hubconf()
    try:
        options = {"map_location": device}
        if "weights_only" in inspect.signature(torch.load).parameters:
            options["weights_only"] = True
        state = torch.load(str(path), **options)
        naf = hubconf.NAF().to(device)
        naf.load_state_dict(state)
        return naf
    except Exception as exc:
        raise RuntimeError(f"local NAF checkpoint loading failed: {path}") from exc


@contextmanager
def _local_naf_extractors(inference_module: Any, naf_path: Path, *, verify_checkpoint: bool = False):
    """Bind the local loader only to extractors created for this MV invocation."""

    original = inference_module.build_image_cond_model

    def build_local(config: dict):
        extractor = original(config)

        def load_naf(self):
            if self.naf_model is None:
                if verify_checkpoint:
                    _verify_naf_checkpoint(naf_path)
                device = next(self.model.parameters()).device
                self.naf_model = _load_local_naf(naf_path, device)
                self.naf_model.eval()
                self.naf_model.requires_grad_(False)

        extractor._load_naf = types.MethodType(load_naf, extractor)
        return extractor

    inference_module.build_image_cond_model = build_local
    try:
        yield
    finally:
        inference_module.build_image_cond_model = original


def _check_cancel(cancel_event: Any | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Pixal3D MV generation cancelled")


def run_multiview(
    *, capture_manifest_path: str | Path | None = None,
    scene_manifest_path: str | Path | None = None, workspace_dir: str | Path,
    mv_root: str | Path, base_root: str | Path, naf_path: str | Path, output_dir: str | Path,
    params: dict[str, Any], inference_runner: Callable[..., Any] | None = None,
    progress_cb: Callable[[int, str], None] | None = None, cancel_event: Any | None = None,
) -> Path:
    """Run calibrated capture media through the real upstream posed-view cascade.

    ``scene_manifest_path`` remains an explicit low-level compatibility entry for
    already-calibrated callers; the Modly node itself accepts only capture input.
    """

    _check_cancel(cancel_event)
    if (capture_manifest_path is None) == (scene_manifest_path is None):
        raise ValueError("Provide exactly one capture_manifest_path or legacy scene_manifest_path")
    if progress_cb is not None:
        progress_cb(2, "Validating calibrated capture" if capture_manifest_path is not None else "Validating posed views")
    num_views = int(params.get("num_views", 4))
    if not 1 <= num_views <= 16:
        raise ValueError("num_views must be between 1 and 16")
    if capture_manifest_path is not None:
        from .multiview_capture import prepare_capture_views, validate_mv_capture
        from .scene_prepare_contract import validate_workspace_output_parent

        validate_mv_capture(Path(capture_manifest_path), Path(workspace_dir), num_views)
        output = validate_workspace_output_parent(Path(output_dir), Path(workspace_dir), "MV output directory")
        views = prepare_capture_views(Path(capture_manifest_path), Path(workspace_dir), output, num_views, cancel_event=cancel_event)
    else:
        views = nullcontext(resolve_views_dir(scene_manifest_path, workspace_dir))
        output = Path(output_dir).resolve()
    root = Path(mv_root).resolve()
    base = Path(base_root).resolve()
    missing = missing_mv_assets(root, base, naf_path)
    if missing:
        raise RuntimeError("Pixal3D MV weights missing: " + ", ".join(missing))
    _check_cancel(cancel_event)
    resolution = int(params.get("resolution", 1024))
    if resolution not in (1024, 1536):
        raise ValueError("resolution must be 1024 or 1536")
    seed = int(params.get("seed", -1))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed is out of range")
    low_vram = params.get("low_vram", "low_vram") in ("low_vram", True)
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_pixal3d_mv.glb"

    real_inference = inference_runner is None
    with views as views_dir, tempfile.TemporaryDirectory(prefix="pixal3d-mv-config-", dir=output) as temporary:
        with _MV_RUN_LOCK:
            if real_inference:
                configure_mv_hf_cache(workspace_dir)
                _verify_naf_checkpoint(Path(naf_path).resolve())
                _verified_naf_hubconf()
                if not mv_runtime_available():
                    raise RuntimeError("Pixal3D MV Python wheel is not installed; the published single-view wheelhouse is insufficient")
                from pixal3d_extension import runtime
                runtime._prepare_runtime_compat()
                runtime._install_windows_native_module_aliases()
                runtime._install_natten_fallback()
                runtime._silence_flex_gemm_autotuners()
                from pixal3d_extension.vendor import inference_mv
                dino_path = str(base / "auxiliary" / "dinov3")
                for item in inference_mv.IMAGE_COND_CONFIGS.values():
                    item["model_name"] = dino_path
                inference_runner = inference_mv.run_inference
            _check_cancel(cancel_event)
            if progress_cb is not None:
                progress_cb(8, "Preparing local model configuration")
            private_config = prepare_mv_pipeline_config(root, base, Path(temporary) / "pipeline_mv.local.json")
            loader = _private_local_loader(root, private_config) if real_inference else nullcontext()
            naf_loader = _local_naf_extractors(inference_mv, Path(naf_path).resolve()) if real_inference else nullcontext()
            with loader, naf_loader:
                _check_cancel(cancel_event)
                inference_runner(
                    views_dir=str(views_dir), output_path=str(glb_path), num_views=num_views,
                    seed=seed, model_path=str(root), config_file=str(private_config),
                    low_vram=low_vram, resolution=resolution,
                    progress_cb=progress_cb, cancel_event=cancel_event,
                )
    _check_cancel(cancel_event)
    if not glb_path.is_file():
        raise RuntimeError("Pixal3D MV runtime did not produce a GLB")
    return glb_path
