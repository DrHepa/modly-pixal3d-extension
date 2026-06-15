# Pixal3D Modly Extension

Pixal3D image-to-3D model extension for Modly. It converts a single input image into a textured GLB mesh using the upstream `TencentARC/Pixal3D` model family and Modly-managed model storage.

This repository contains only the extension runtime, setup entrypoint, and a release-backed wheelhouse contract needed to prepare Pixal3D dependencies from a GitHub install. Model weights are not included; Modly downloads model assets through its UI into the normal Modly model storage.

## What setup does

```bash
python3 setup.py --prepare --json
```

- creates `venv/` inside the extension
- installs `requirements.txt`
- prepares the selected release-backed wheelhouse from `wheelhouse.manifest.json`
- installs native packages with `pip install --no-index --find-links <verified-wheelhouse>`
- runs `pip check`
- creates logical Modly model-storage folders under the configured Modly models root

It does **not** download model weights and does **not** run generation.

## Release-backed wheelhouse

`wheelhouse.manifest.json` pins the release tag, selected platform lane, archive filename, checksum, and fallback policy. Setup verifies the selected archive before extraction and installs native packages only from a verified local path using `--no-index --find-links`.

The vendored `wheels/` fallback is intentionally retained for migration/rollback. It is used only after retryable release access failures such as network/auth errors and only when every wheel is lane-compatible and hash-verified. Setup must not silently fall back to PyPI for native packages.

Current published wheelhouse targets:

- Linux `aarch64` / Python `cp312` / `cuda124`
- Linux `x64` / Python `cp312` / `cuda124`
- Windows `x64` / Python `cp311` / `cuda124` — Modly packaged app install contract
- Windows `x64` / Python `cp312` / `cuda124`

Included packaged dependencies:

- `pixal3d-core==0.1.0+modly`
- `moge==2.0.0+modly`
- `naf==0.1.0+modly`
- `utils3d==1.3+modly.headless`
- `pipeline==1.0.0+modly`
- `o-voxel==0.0.1`
- `cumesh==0.0.1`
- `flex-gemm==1.0.0`
- `nvdiffrast==0.4.0`
- `nvdiffrec-render==0.0.0`

The Windows `x64` / Python `cp311` / `cuda124` lane used by the packaged Modly app also includes `natten==0.21.0` with native `libnatten` available. Setup verifies the wheelhouse checksum, installs from the verified archive, and probes native CUDA/NATTEN availability before reporting success.

On Windows, the equivalent exact-stack native package distributions are installed from the Windows lane where names differ, such as `o-voxel-vb-ap`, `cumesh-vb`, `flex-gemm-ap`, `drtk`, and `flash-attn`.

`natten`/`libnatten` availability is lane-specific. Linux `aarch64` and Windows `x64`/`cp311`/`cuda124` include verified native NATTEN. Other lanes may treat NATTEN as optional; setup probes `natten.HAS_LIBNATTEN` and strict NAF is available only when that value is `True`.

## Modly contract

- `manifest.json` declares the model extension.
- `setup.py` prepares the extension environment.
- `generator.py` exposes `Pixal3DGenerator`.
- Model assets must live under Modly's model storage, not inside this repository.

## View-aligned generation behavior

Pixal3D generates meshes aligned to the input image projection rather than always canonicalizing the object to a universal upright/front pose. Upstream's projected render path is designed so the first rendered frame matches the projected input view.

Practical implications:

- front/straight input images should usually produce upright meshes;
- angled or isometric input images can produce angled meshes;
- that input-dependent tilt is expected Pixal3D behavior, not a Modly orientation bug;
- do not apply a fixed post-export pitch correction to all Pixal3D outputs, because it can break already-upright generations.

The extension preserves Pixal3D's input-dependent tilt/pitch behavior, but it now rewrites the final exported GLB with a fixed 180 degree yaw around glTF's Y-up axis. That correction is applied to the asset itself so Modly UI and downstream consumers receive the same front-facing orientation.

## Remaining runtime requirement

After setup succeeds, use Modly UI to download Pixal3D model assets. Real generation should be validated only after the primary Pixal3D weights and the required auxiliary assets are present.

Auxiliary model assets are stored below `models/pixal3d/auxiliary/`. Do not shorten this folder to `aux`: `AUX` is a reserved Windows device name and can break setup on normal Windows filesystems.

Required DINO/RMBG sentinels:

- `models/pixal3d/auxiliary/dinov3/config.json`
- `models/pixal3d/auxiliary/dinov3/preprocessor_config.json`
- `models/pixal3d/auxiliary/dinov3/model.safetensors`
- `models/pixal3d/auxiliary/rmbg/config.json`
- `models/pixal3d/auxiliary/rmbg/preprocessor_config.json`
- `models/pixal3d/auxiliary/rmbg/BiRefNet_config.py`
- `models/pixal3d/auxiliary/rmbg/birefnet.py`
- `models/pixal3d/auxiliary/rmbg/model.safetensors`

Normal setup does not download these weights implicitly. To explicitly seed the local DINO/RMBG auxiliary assets, run `python3 setup.py --bootstrap-auxiliary-assets --workspace-root <extension-dir> --json`; that bootstrap is allowlist-only for the files above. Default first run may attempt the same controlled bootstrap before preserving the existing `camenduru` remote fallback. `local`, `offline`, and `strict` modes never start that network bootstrap.

The pipeline patcher is local-first for DINO/RMBG: when those sentinels are complete, it writes local resolved paths into the user-local `pipeline.json` and records non-absolute logical metadata. If the auxiliary sentinels are missing in default mode, the existing `camenduru` Hugging Face IDs remain the fallback. In `local`, `offline`, or `strict` auxiliary mode, missing DINO/RMBG files fail early with `missing_auxiliary_assets` before importing upstream inference code.

This is **not** a full offline-generation guarantee yet. Upstream inference still uses MoGe `Ruicheng/moge-2-vitl`; `MoGeModel.from_pretrained()` can consume a local checkpoint file, but this extension does not yet wire a local MoGe asset path. NAF code is packaged, but its checkpoint path still comes from `torch.hub.load_state_dict_from_url("https://github.com/valeoai/NAF/releases/download/model/naf_release.pth")`, so it depends on Torch cache or future local checkpoint plumbing. Treat MoGe/NAF as remote/cache fallback until those assets are localized and tested.

## Publication status

- Repository visibility: public.
- Primary Modly packaged-app lane: Windows `x64` / Python `cp311` / CUDA `12.4`.
- Setup contract: release-backed wheelhouse with checksum verification and native import probes.
- Runtime status: Windows `x64` / Python `cp311` / CUDA `12.4` has been validated through a complete Modly Low VRAM 1024 generation, including native NATTEN sampling, GLB extraction, final GLB save, and Modly workspace fetch.
- Runtime note: use Low VRAM mode on 8GB-class GPUs; generation quality and orientation depend on the input view, with a final 180 degree GLB yaw correction applied for Modly/downstream front-facing consumption.
