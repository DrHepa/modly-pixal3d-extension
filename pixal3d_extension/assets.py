from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pixal3d_extension.paths import resolve_modly_layout, resolve_storage_path


@dataclass(frozen=True)
class AssetManifest:
    key: str
    repo_id: str
    local_root: str
    sentinels: tuple[str, ...]

    @property
    def sentinel_paths(self) -> tuple[str, ...]:
        return tuple(str(PurePosixPath(self.local_root) / sentinel) for sentinel in self.sentinels)


PRIMARY_ASSET = AssetManifest(
    key="primary",
    repo_id="TencentARC/Pixal3D",
    local_root="models/pixal3d/generate",
    sentinels=(
        "pipeline.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
)

AUXILIARY_ASSETS = {
    "dino": AssetManifest(
        key="dino",
        repo_id="camenduru/dinov3-vitl16-pretrain-lvd1689m",
        local_root="models/pixal3d/auxiliary/dinov3",
        sentinels=("config.json", "preprocessor_config.json", "model.safetensors"),
    ),
    "rmbg": AssetManifest(
        key="rmbg",
        repo_id="camenduru/RMBG-2.0",
        local_root="models/pixal3d/auxiliary/rmbg",
        sentinels=("config.json", "preprocessor_config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"),
    ),
}

REQUIRED_SENTINEL_PATHS = [
    *PRIMARY_ASSET.sentinel_paths,
    *AUXILIARY_ASSETS["dino"].sentinel_paths,
    *AUXILIARY_ASSETS["rmbg"].sentinel_paths,
]


def _missing_for_manifest(workspace_root: Path, manifest: AssetManifest) -> list[str]:
    layout = resolve_modly_layout(workspace_root)
    return [relative for relative in manifest.sentinel_paths if not resolve_storage_path(layout, relative).is_file()]


def check_asset_sentinels(workspace_root: str | Path, *, require_aux: bool = True) -> dict:
    root = Path(workspace_root)
    missing_primary = _missing_for_manifest(root, PRIMARY_ASSET)
    missing_aux = []
    if require_aux:
        for manifest in AUXILIARY_ASSETS.values():
            missing_aux.extend(_missing_for_manifest(root, manifest))

    if missing_primary:
        return {
            "status": "blocked",
            "code": "missing_primary_assets",
            "missing": missing_primary,
            "generation_allowed": False,
        }
    if missing_aux:
        return {
            "status": "blocked",
            "code": "missing_replacement_auxiliary_assets",
            "missing": missing_aux,
            "generation_allowed": False,
        }
    return {"status": "ready", "code": "assets_ready", "missing": [], "generation_allowed": True}
