"""
Pixal3D multi-view inference: V posed views -> GLB.

The single-view entry point (inference.py) takes one in-the-wild photo and has to
guess its camera with MoGe-2. Here the cameras are given, so the input is a small
dataset-style directory:

    <views_dir>/
        transforms.json          mesh_scale + per-frame file_path / transform_matrix
        view00_azim000.png       any format; RGBA alpha is used as the mask if present,
        view01_azim090.png       otherwise the view is matted with the pipeline's own
        ...                      rembg model, the same one inference.py uses

transforms.json follows the Blender/NeRF convention used by the training renders:
`transform_matrix` is a 4x4 camera-to-world (Z-up world, camera looks -Z with +Y up)
and `camera_angle_x` is the horizontal fov in radians, given per frame or once at the
top level for all of them. Frame 0 is the MAIN view, and
its pose must be the canonical front view (camera at (0,-d,0)) so that
calc_mat_0 == F and the main view degenerates to the single-view path. See
assets/mv_images/example for a working input.

Usage:
    python inference_mv.py --views_dir assets/mv_images/example --output ./output_mv.glb
    python inference_mv.py --views_dir ... --num_views 4 --low_vram
    ATTN_BACKEND=sdpa python inference_mv.py --views_dir ...   # no flash_attn
"""

import os
import argparse
import json
import math
import numpy as np
import torch
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MODEL_PATH = "TencentARC/Pixal3D"
# The MV denoisers live next to the single-view ones under ckpts/*_mv; this config
# file is what points the pipeline at them.
CONFIG_FILE = "pipeline_mv.json"
DEFAULT_VIEWS_DIR = "assets/mv_images/example"

# Same four stages as inference.py, only the extractor class and the fusion differ.
# "average" keeps the fused feature shape identical to single-view, which is why the
# denoisers need no architectural change.
IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
        "multiview_fusion": "average",
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
        "multiview_fusion": "average",
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
        "multiview_fusion": "average",
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
        "multiview_fusion": "average",
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjMultiViewFeatureExtractor,
    )
    model = DinoV3ProjMultiViewFeatureExtractor(**config)
    model.eval()
    return model


def init_pipeline(model_path=MODEL_PATH, config_file=CONFIG_FILE, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path} ({config_file})...")
    pipeline = Pixal3DMVImageTo3DPipeline.from_pretrained(model_path, config_file)

    print("[ImageCond] Building DinoV3ProjMultiViewFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    cond_attrs = ['image_cond_model_ss', 'image_cond_model_shape_512',
                  'image_cond_model_shape_1024', 'image_cond_model_tex_1024']
    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        pipeline.low_vram = False
        pipeline.cuda()
        for attr in cond_attrs:
            getattr(pipeline, attr).cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in cond_attrs:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline

# ============================================================================
# View Loading
# ============================================================================

def make_rembg(pipeline):
    """
    Background removal through the pipeline's own matting model, as inference.py does.

    Only the alpha is taken from it: unlike preprocess_image we must not crop or
    rescale, because transforms.json describes the framing as given.
    """
    def rembg(image: Image.Image) -> Image.Image:
        if pipeline.low_vram:
            pipeline.rembg_model.to(pipeline.device)
        output = pipeline.rembg_model(image.convert('RGB'))
        if pipeline.low_vram:
            pipeline.rembg_model.cpu()
        return output
    return rembg


def load_rgba(path: str, rembg=None):
    """
    Read one view as RGBA, matting it first if it does not already carry a mask.

    Returns (image, was_matted). The alpha test matches preprocess_image: a fully
    opaque alpha channel counts as no mask.
    """
    image = Image.open(path)
    alpha = np.array(image.getchannel(3)) if image.mode == 'RGBA' else None
    if alpha is not None and not np.all(alpha == 255):
        return image.convert('RGBA'), False
    if rembg is None:
        raise ValueError(f"{path} has no alpha channel and no matting model was given")
    return rembg(image).convert('RGBA'), True


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """
    Turn an RGBA view into a conditioning tensor the way training read its views:
    LANCZOS resize, then premultiply by alpha so the background is black.
    """
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)


def load_views(views_dir: str, num_views: int = None, rembg=None, image_sizes=(512, 1024)) -> dict:
    """
    Load a view directory into the bundle Pixal3DMVImageTo3DPipeline.run_mv wants.

    Frame 0 is the main view. Each file is read (and matted, if needed) once and then
    resized to each distinct stage resolution -- 512 for SS / shape-512 and 1024 for
    the two 1024 stages.
    """
    with open(os.path.join(views_dir, 'transforms.json')) as f:
        meta = json.load(f)
    frames = meta['frames']
    if num_views is not None:
        if num_views > len(frames):
            raise ValueError(f"--num_views {num_views} > {len(frames)} views in {views_dir}")
        frames = frames[:num_views]

    def camera_angle_x_of(frame):
        for src in (frame, meta):
            if 'camera_angle_x' in src:
                return float(src['camera_angle_x'])
        raise KeyError(f"'camera_angle_x' missing for {frame.get('file_path')}")

    transform_matrix = torch.tensor([fr['transform_matrix'] for fr in frames],
                                    dtype=torch.float32)[None]              # [1, V, 4, 4]
    camera_angle_x = torch.tensor([camera_angle_x_of(fr) for fr in frames],
                                  dtype=torch.float32)[None]                # [1, V]
    # Derive the distance from the pose rather than trusting a 'radius' field, so it
    # can never disagree with transform_matrix.
    camera_distance = torch.norm(transform_matrix[:, :, :3, 3], dim=-1)      # [1, V]

    paths = [os.path.join(views_dir, fr['file_path']) for fr in frames]
    loaded = [load_rgba(p, rembg) for p in paths]
    rgba = [im for im, _ in loaded]
    matted = sum(was_matted for _, was_matted in loaded)
    if matted:
        print(f"[Rembg] matted {matted}/{len(paths)} view(s) that had no alpha channel")

    images = {
        size: torch.stack([to_cond_tensor(im, size) for im in rgba], dim=0)[None]
        for size in image_sizes                                             # [1, V, 3, H, W]
    }

    view_names = [fr.get('name', os.path.splitext(fr['file_path'])[0]) for fr in frames]
    mesh_scale = float(meta.get('mesh_scale', 1.0))
    print(f"[Views] V={len(frames)} ({', '.join(view_names)}) from {views_dir}")
    print(f"[Camera] fov={math.degrees(float(camera_angle_x[0, 0])):.2f}deg, "
          f"distance={float(camera_distance[0, 0]):.4f}, mesh_scale={mesh_scale:.4f}")

    return {
        'images': images,
        'camera_angle_x': camera_angle_x,
        'camera_distance': camera_distance,
        'transform_matrix': transform_matrix,
        'mesh_scale': mesh_scale,
        'view_names': view_names,
    }


def check_main_view(views: dict, atol: float = 1e-4):
    """
    Warn if frame 0 is not the canonical front view.

    The extractor maps every view through calc_mat_i = F @ inv(C_0) @ C_i, so the main
    view is always snapped onto F -- but if C_0 is not itself a front view, the whole
    rig gets rotated relative to the object and the generated mesh comes out in a
    different frame than the models were trained for.
    """
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGridMV
    F = ProjGridMV(grid_resolution=2, image_resolution=64).front_view_transform_matrix.clone()
    F[1, 3] = -views['camera_distance'][0, 0]
    err = float((views['transform_matrix'][0, 0] - F).abs().max())
    if err > atol:
        print(f"[Warning] main view (frame 0) is not the canonical front view "
              f"(max deviation {err:.3e}). The result will be posed in that view's frame.")
    else:
        print(f"[Camera] main view == canonical front view (max err {err:.1e})")

# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    views_dir: str,
    output_path: str,
    num_views: int = None,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    config_file: str = CONFIG_FILE,
    low_vram: bool = False,
    resolution: int = -1,
    progress_cb=None,
    cancel_event=None,
):
    def stage(percent, label):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Pixal3D MV generation cancelled")
        if progress_cb is not None:
            progress_cb(percent, label)

    stage(12, "Loading multi-view models")
    pipeline = init_pipeline(model_path, config_file, low_vram=low_vram)

    stage(30, "Loading posed views")
    print(f"[Inference] Loading views from: {views_dir}")
    views = load_views(views_dir, num_views, rembg=make_rembg(pipeline))
    check_main_view(views)

    stage(42, "Generating multi-view mesh")
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run_mv(
        views,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    stage(82, "Extracting textured GLB")
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    stage(96, "Writing GLB")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    stage(100, "Multi-view generation complete")
    print(f"[Done] GLB saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixal3D Multi-View Inference: posed views to GLB")
    parser.add_argument("--views_dir", type=str, default=DEFAULT_VIEWS_DIR,
                        help=f"Directory with transforms.json + RGBA views (default: {DEFAULT_VIEWS_DIR})")
    parser.add_argument("--output", type=str, default="./output_mv.glb", help="Output GLB file path")
    parser.add_argument("--num_views", type=int, default=None,
                        help="Use only the first N views (default: all in transforms.json)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--config_file", type=str, default=CONFIG_FILE,
                        help=f"Pipeline config inside model_path (default: {CONFIG_FILE})")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")

    args = parser.parse_args()

    run_inference(
        views_dir=args.views_dir,
        output_path=args.output,
        num_views=args.num_views,
        seed=args.seed,
        model_path=args.model_path,
        config_file=args.config_file,
        low_vram=args.low_vram,
        resolution=args.resolution,
    )
