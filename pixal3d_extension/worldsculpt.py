"""Local-only WorldSculpt scene composition using the pinned upstream scripts.

The private source/config overlay is deliberately disposable: shared Modly weights
and the vendored upstream tree are never patched in place.
"""
from __future__ import annotations

import importlib.util
import json
import os
import selectors
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .naf_checkpoint import verify_naf_checkpoint
from .process_tree import popen_process_group_kwargs, terminate_process_tree
from .worldsculpt_contract import (ADAPTER_DIRS, prepare_case_root,
                                    validate_adapters, validate_crops,
                                    validate_output, validate_scene)
from .worldsculpt_lane import ROOT as EXTENSION_ROOT, python_path

SOURCE = Path(__file__).resolve().parent.parent / "worldsculpt_upstream"
_REQUIRED_IMPORTS = ("peft", "fpsample", "iopath", "pycocotools", "ftfy", "natten", "src.model.naf")
_BASE_MODELS = ("ss_dec_conv3d_16l8_fp16", "ss_flow_img_dit_1_3B_64_bf16",
                "shape_dec_next_dc_f16c32_fp16", "slat_flow_img2shape_dit_1_3B_512_bf16",
                "slat_flow_img2shape_dit_1_3B_1024_bf16", "tex_dec_next_dc_f16c32_fp16",
                "slat_flow_imgshape2tex_dit_1_3B_1024_bf16")
_DINO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
_DINO_REFS = {_DINO, "facebook/dinov3-vitl16-pretrain-lvd1689m"}
_RMBG = "briaai/RMBG-2.0"
_RMBG_REFS = {_RMBG, "camenduru/RMBG-2.0"}
_AUXILIARY_FILES = {
    "dinov3": ("config.json", "preprocessor_config.json", "model.safetensors"),
    "rmbg": ("config.json", "preprocessor_config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"),
}
_NAF_CALL = '''self.naf_model = torch.hub.load(
                "valeoai/NAF", "naf", pretrained=True, device=device, trust_repo=True
            )'''
_NAF_LOCAL = '''from src.model.naf import NAF
            import os
            checkpoint = os.environ["WORLDSCULPT_NAF_CHECKPOINT"]
            self.naf_model = NAF().to(device)
            self.naf_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))'''


def missing_runtime() -> list[str]:
    interpreter = python_path()
    if not interpreter.is_file():
        return ["venv-worldsculpt (run setup.py --repair-worldsculpt)"]
    code = '''import json,torch,peft,fpsample,iopath,pycocotools,ftfy,natten,o_voxel
from importlib.metadata import version
assert torch.__version__ == "2.12.0+cu130" and torch.version.cuda == "13.0"
assert version("transformers") == "4.57.1"
print("ready")'''
    check = subprocess.run([str(interpreter), "-c", code], cwd=EXTENSION_ROOT, text=True, capture_output=True)
    return [] if check.returncode == 0 and check.stdout.strip() == "ready" else ["venv-worldsculpt runtime/ABI (run setup.py --repair-worldsculpt)"]


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"WorldSculpt local {label} missing or symlinked: {path}")
    return path.resolve(strict=True)


def _under(root: Path, path: Path, label: str) -> Path:
    root = root.resolve(strict=True)
    if root not in path.parents or any(part.is_symlink() for part in (path, *path.parents) if part == root or root in part.parents):
        raise ValueError(f"WorldSculpt {label} escapes or aliases its local root: {path}")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"WorldSculpt {label} escapes its local root: {path}")
    return resolved


def validate_base(base_root: Path, naf_path: Path | None) -> None:
    base = Path(base_root).resolve(strict=True)
    data = json.loads(_regular(_under(base, base / "pipeline.json", "base pipeline"), "base pipeline").read_text())
    if data.get("name") not in {"Trellis2ImageTo3DPipeline", "Pixal3DImageTo3DPipeline"}:
        raise ValueError("WorldSculpt requires a Pixal3D base pipeline")
    models = data.get("args", {}).get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("WorldSculpt base pipeline has no models")
    # The upstream loader eagerly constructs every declared base model even with --no_tex.
    for model in _BASE_MODELS:
        _regular(_under(base, base / "ckpts" / f"{model}.json", model), model)
        _regular(_under(base, base / "ckpts" / f"{model}.safetensors", model), model)
    for auxiliary, files in _AUXILIARY_FILES.items():
        for relative in files:
            _regular(_under(base, base / "auxiliary" / auxiliary / relative,
                            f"{auxiliary} {relative}"), f"{auxiliary} {relative}")
    for value in models.values():
        if not isinstance(value, str) or not value.startswith("ckpts/") or ".." in Path(value).parts:
            raise ValueError("WorldSculpt base pipeline contains a nonlocal model reference")
        _regular(_under(base, base / f"{value}.json", f"base {value}"), f"base {value}")
        _regular(_under(base, base / f"{value}.safetensors", f"base {value}"), f"base {value}")
    if naf_path is not None:
        verify_naf_checkpoint(_regular(Path(naf_path), "NAF checkpoint"))


def _private_overlay(root: Path, base: Path) -> Path:
    if not SOURCE.is_dir() or not (SOURCE / "reconstruct_batch.py").is_file():
        raise RuntimeError("Pinned WorldSculpt upstream source is absent")
    overlay = root / "source"
    shutil.copytree(SOURCE, overlay, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"), symlinks=False)
    local_base = overlay / "pretrained" / "Pixal3D"
    local_base.mkdir(parents=True)
    (local_base / "ckpts").symlink_to(base / "ckpts", target_is_directory=True)
    data = json.loads((base / "pipeline.json").read_text())
    # The upstream pipeline reads its config from this private path; do not edit shared assets.
    def localize(value):
        if isinstance(value, dict):
            return {key: localize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [localize(item) for item in value]
        if isinstance(value, str) and value in _DINO_REFS:
            return str(base / "auxiliary" / "dinov3")
        if isinstance(value, str) and value in _RMBG_REFS:
            return str(base / "auxiliary" / "rmbg")
        return value
    localized = localize(data)
    serialized = json.dumps(localized)
    if any(reference in serialized for reference in (*_DINO_REFS, *_RMBG_REFS)):
        raise ValueError("WorldSculpt base pipeline retains a remote DINO/RMBG reference")
    (local_base / "pipeline.json").write_text(serialized, encoding="utf-8")
    inference = overlay / "inference.py"
    source = inference.read_text(encoding="utf-8")
    if source.count(_DINO) != 4:
        raise RuntimeError("WorldSculpt DINO override anchor changed; refusing remote fallback")
    inference.write_text(source.replace(_DINO, str(base / "auxiliary" / "dinov3")), encoding="utf-8")
    feature = overlay / "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"
    source = feature.read_text(encoding="utf-8")
    if source.count(_NAF_CALL) != 1:
        raise RuntimeError("WorldSculpt NAF override anchor changed; refusing remote fallback")
    feature.write_text(source.replace(_NAF_CALL, _NAF_LOCAL), encoding="utf-8")
    # Fail on missing local files rather than accepting an already populated Hub cache.
    for relative in ("pixal3d/models/__init__.py", "pixal3d/pipelines/__init__.py", "pixal3d/pipelines/base.py"):
        target = overlay / relative
        source = target.read_text(encoding="utf-8")
        anchor = "from huggingface_hub import hf_hub_download"
        if source.count(anchor) != 1:
            raise RuntimeError(f"WorldSculpt local-only loader anchor changed: {relative}")
        target.write_text(source.replace(anchor, 'raise FileNotFoundError("WorldSculpt requires local model files")'), encoding="utf-8")
    for relative in ("pixal3d/modules/image_feature_extractor.py", "pixal3d/trainers/flow_matching/mixins/image_conditioned.py", "pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py"):
        target = overlay / relative
        source = target.read_text(encoding="utf-8")
        anchor = "DINOv3ViTModel.from_pretrained(model_name)"
        if source.count(anchor) != 1:
            raise RuntimeError(f"WorldSculpt DINO loader anchor changed: {relative}")
        source = source.replace(anchor, "DINOv3ViTModel.from_pretrained(model_name, local_files_only=True)")
        if relative.endswith("image_conditioned_proj.py"):
            anchor = "DINOv3ViTModel.from_pretrained(dino_model_name)"
            if source.count(anchor) != 1:
                raise RuntimeError("WorldSculpt DINO loader anchor changed")
            source = source.replace(anchor, "DINOv3ViTModel.from_pretrained(dino_model_name, local_files_only=True)")
        target.write_text(source, encoding="utf-8")
    return overlay


def _local_adapter_config(source: Path, destination: Path, base: Path) -> Path:
    """Localize DINO references in a private config; reject other remote refs."""
    data = json.loads(source.read_text(encoding="utf-8"))
    def localize(value, key=""):
        if isinstance(value, dict):
            return {name: localize(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [localize(item, key) for item in value]
        if key == "model_name" and isinstance(value, str):
            if value in {_DINO, "facebook/dinov3-vitl16-pretrain-lvd1689m"}:
                return str(base / "auxiliary/dinov3")
            if "/" in value and not Path(value).is_absolute():
                raise ValueError(f"WorldSculpt adapter has unlocalized model_name: {value}")
        return value
    destination.write_text(json.dumps(localize(data)), encoding="utf-8")
    return destination


def _run(args: list[str], cwd: Path, env: dict[str, str], cancel_event) -> str:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("WorldSculpt cancelled")
    # No shell and no inherited network cache; process termination is bounded by stage.
    proc = subprocess.Popen([str(python_path()), *args], cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            **popen_process_group_kwargs())
    tail = bytearray()
    selector = selectors.DefaultSelector()
    try:
        selector.register(proc.stdout, selectors.EVENT_READ)
        while True:
            for key, _ in selector.select(timeout=0.5):
                chunk = os.read(key.fd, 65536)
                if chunk:
                    tail.extend(chunk)
                    del tail[:-3000]
                else:
                    selector.unregister(key.fileobj)
            if cancel_event is not None and cancel_event.is_set():
                terminate_process_tree(proc)
                raise RuntimeError("WorldSculpt cancelled")
            if proc.poll() is not None and not selector.get_map():
                break
        if proc.returncode:
            raise RuntimeError(f"WorldSculpt stage {args[0]} failed ({proc.returncode}): {tail.decode(errors='replace')}")
        return tail.decode(errors="replace")
    finally:
        selector.close()
        terminate_process_tree(proc, force=proc.poll() is None)
        proc.stdout.close()


def _artifact_error(stage: str, tail: str, exc: Exception) -> RuntimeError:
    # Upstream can catch per-instance errors and exit zero without an artifact.
    return RuntimeError(f"WorldSculpt stage {stage} produced invalid artifacts: {exc}\n{tail[-3000:]}")


def resolve_scene_manifest(scene_manifest_path: str | Path, workspace_dir: str | Path) -> Path:
    """Accept only a workspace-scoped Modly sceneRoot, never an arbitrary path."""
    workspace = Path(workspace_dir).resolve(strict=True)
    manifest = Path(scene_manifest_path).resolve(strict=True)
    if workspace not in manifest.parents or not manifest.is_file():
        raise ValueError("WorldSculpt scene manifest must be inside the workspace")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "modly.scene-manifest.v1":
        raise ValueError("WorldSculpt requires modly.scene-manifest.v1")
    value = data.get("sceneRoot")
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("WorldSculpt sceneRoot must be a workspace-relative directory")
    scene = (manifest.parent if value == "." else workspace / value).resolve(strict=True)
    if workspace not in scene.parents or not scene.is_dir():
        raise ValueError("WorldSculpt sceneRoot escapes the workspace")
    return scene


def _validated_output_parent(output_dir: Path, workspace_dir: Path, scene: Path) -> Path:
    """Resolve and validate an output parent without mutating the filesystem."""
    workspace = Path(workspace_dir).resolve(strict=True)
    if not workspace.is_dir() or workspace == Path(workspace.anchor):
        raise ValueError("WorldSculpt workspace root must be a non-root directory")
    output = Path(output_dir).resolve(strict=False)
    if output == Path(output.anchor) or (output != workspace and workspace not in output.parents):
        raise ValueError("WorldSculpt output must be inside the workspace")
    if output == scene or scene in output.parents:
        raise ValueError("WorldSculpt output must be separate from input scene")
    return output


def run_worldsculpt(*, scene_dir: Path, adapter_root: Path, base_root: Path,
                     naf_path: Path, output_dir: Path, workspace_dir: Path,
                     progress_cb=None,
                     cancel_event=None, face_budget: int = 1000000) -> Path:
    """Run crop, reconstruction and composition; return validated GLB only."""
    if type(face_budget) is not int or not 1000 <= face_budget <= 3000000:
        raise ValueError("face_budget must be an integer from 1000 to 3000000")
    missing = missing_runtime()
    if missing:
        raise RuntimeError("WorldSculpt runtime dependencies missing: " + ", ".join(missing))
    scene = Path(scene_dir).resolve(strict=True)
    output = _validated_output_parent(output_dir, workspace_dir, scene)
    eligible = validate_scene(scene)
    validate_adapters(adapter_root)
    base = Path(base_root).resolve(strict=True)
    validate_base(base, naf_path)
    adapters = Path(adapter_root).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    case = prepare_case_root(output / f"worldsculpt-{uuid.uuid4().hex}")
    overlay = _private_overlay(case, base)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               WORLDSCULPT_NAF_CHECKPOINT=str(Path(naf_path).resolve(strict=True)),
               HF_HOME=str(case / "hf-cache"), TORCH_HOME=str(case / "torch-cache"),
               PYTHONNOUSERSITE="1", ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa")
    env.pop("PYTHONPATH", None)
    if progress_cb: progress_cb(5, "WorldSculpt: crop")
    crop_tail = _run(["prepare_crops_scene.py", "--scene_dir", str(scene), "--case_root", str(case),
          "--crop_resolution", "1024", "--save_alignments", "--alpha_erode_kernel", "0",
          "--alpha_erode_iters", "0", "--min_mask_ratio", "0.001", "--max_crop_ratio", "3.0",
          "--mask_fit_scale"], overlay, env, cancel_event)
    try:
        validate_crops(case, eligible)
    except (OSError, ValueError) as exc:
        raise _artifact_error("prepare_crops_scene.py", crop_tail, exc) from exc
    if progress_cb: progress_cb(30, "WorldSculpt: reconstruct")
    ss, shape = (adapters / name for name in ADAPTER_DIRS)
    ss_config = _local_adapter_config(ss / "config.json", case / "ss-config.json", base)
    shape_config = _local_adapter_config(shape / "config.json", case / "shape-config.json", base)
    recon_tail = _run(["reconstruct_batch.py", "--case_root", str(case), "--views", "all",
          "--recon_subdir", "_recon", "--instances", ",".join(eligible), "--no-ema", "--sampler", "official", "--no_tex", "--no_glb",
          "--ss_config", str(ss_config), "--ss_ckpt_dir", str(ss / "ckpts"), "--ss_step", "15000",
          "--shape_config", str(shape_config), "--shape_ckpt_dir", str(shape / "ckpts"),
          "--shape_step", "15000"], overlay, env, cancel_event)
    try:
        for name in eligible:
            mesh = case / "_recon" / name / "mesh.pt"
            _regular(_under(case, mesh, f"{name} mesh.pt"), f"{name} mesh.pt")
    except (OSError, ValueError) as exc:
        raise _artifact_error("reconstruct_batch.py", recon_tail, exc) from exc
    if progress_cb: progress_cb(75, "WorldSculpt: compose")
    compose_tail = _run(["compose_scene.py", "--case_root", str(case), "--recon_dir", str(case / "_recon"),
          "--face_budget", str(face_budget), "--glb_decimation", str(face_budget),
          "--output_dir", str(case / "_scene"),
          "--instances", ",".join(eligible), "--normal", "--no_render", "--no_video"], overlay, env, cancel_event)
    try:
        glb = validate_output(case, eligible)
    except (OSError, ValueError) as exc:
        raise _artifact_error("compose_scene.py", compose_tail, exc) from exc
    if progress_cb: progress_cb(100, "WorldSculpt: complete")
    return glb
