# Pixal3D for Modly

Turn an image into a textured 3D asset, combine multiple views, or reconstruct
objects in a scene with WorldSculpt—all inside Modly workflows.

**Requires [Modly 0.4.3 or newer](https://github.com/lightningpixel/modly/releases/tag/v0.4.3).**
Model weights are downloaded through Modly Models; they are not bundled here.

## Install and get started

1. Open **Models/Extensions → Install from GitHub** and enter:
   `https://github.com/DrHepa/modly-pixal3d-extension`
2. Let setup finish. If dependencies need recovery, use the extension's **Repair** action.
3. In **Models**, download the shared weight groups for your chosen node (below).
4. Connect your inputs in a workflow and run the node.

Normal setup automatically attempts the special scene-preparation and WorldSculpt
environments on supported hosts. No separate manual setup is needed for a fresh install.

**Windows RTX 5090 (experimental base-only):** normal Install/Reinstall/Repair
selects the checksum-pinned r3 CPython 3.11 / SM120 / CUDA 12.8 archive automatically.
No PowerShell script or experimental opt-in manifest is required. Setup checks the
exact torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / NATTEN 0.21.6 stack, native
imports, and synchronized tiny Torch/NATTEN CUDA operations (180-second bound).
`runtime_prepared` is **not** `inference_validated`: real Low VRAM generation,
GLB/viewer quality, restart and lifecycle qualification remain unverified on RTX 5090.
MV, scene preparation and WorldSculpt Windows paths are not enabled by this lane.
Weights still come only from Modly Models.

**Upgrading from the older single-node extension?** Existing checkpoints can be
moved—not copied—into the host's shared group roots. Preserve the auxiliary
subdirectories and check Models readiness before deleting anything; do not download
a duplicate set blindly.

## Five workflow nodes

| Node | Input | Output | Required groups |
| --- | --- | --- | --- |
| **Image to 3D** | One image | Textured GLB | Base |
| **Multi-Image to 3D** | 2–4 images | Textured GLB | Base, MV, DA3 |
| **Prepare Scene from Images** | 2–8 images | Scene bundle | SAM3, DA3 |
| **Normalize Annotated Scene** | Existing annotated scene | Scene bundle | None |
| **WorldSculpt Scene to 3D** | Prepared/normalized scene | Geometry-only GLB | Base, WorldSculpt adapters |

Video scene preparation is implemented but not public until the host's
[typed model-input transport (PR #358)](https://github.com/lightningpixel/modly/pull/358) is released.

### Image → textured asset

Connect an image to **Image to 3D**. Defaults favor safer memory use: resolution
1024, Low VRAM, texture size 1024, automatic MoGe FOV, and a random seed (`-1`).
Use a fixed seed for repeatable comparisons. Higher resolution and texture size
cost more memory and time.

### Multiple views → one asset

Connect 2–4 images of the **same object**, with clear overlap between views.
Use consecutive connected image slots, starting at Primary view. In stock Modly
0.4.3, secondary images must be workspace-owned files; empty slots are compacted
by the host, so do not rely on gaps preserving camera roles.

Auto cameras use DA3 estimates, not guaranteed camera calibration. Declared camera
roles must match your actual camera rig; their FOV defaults to **20°**.
**Every MV camera mode currently requires DA3 weights and runtime**, including
Declared roles. MV defaults are resolution/texture size 1024, Low VRAM, Auto
cameras, and random seed (`-1`).

### Images → scene → WorldSculpt

1. Connect 2–8 unique images with **matching dimensions** to consecutive slots on
   **Prepare Scene from Images**.
2. Enter the object labels you want SAM3 to track.
3. Connect the resulting scene to **WorldSculpt Scene to 3D**.

Use a static scene and an overlapping camera orbit; moving objects or weak view
coverage can produce poor masks, boxes, or reconstruction. SAM3 supplies instance
masks; DA3 supplies depth and camera estimates. DA3's scale is **relative**, not
metric. WorldSculpt preserves that scene-scale limitation.

Already have frames, masks, cameras, and object boxes in an annotated Modly scene?
Use **Normalize Annotated Scene** before WorldSculpt. Normalization validates and
copies referenced scene assets; it does not estimate missing data or invent scale.

## Shared weights and first use

Modly stores each shared group once and reuses it across dependent nodes.
Hugging Face weights are UI-managed and local-only during inference.

| Group | Model sources | Used by |
| --- | --- | --- |
| `pixal3d-base` | TencentARC/Pixal3D, DINOv3, RMBG-2.0, Ruicheng/moge-2-vitl | Image, MV, WorldSculpt |
| `pixal3d-mv` | TencentARC/Pixal3D MV checkpoints | MV |
| `da3-base` | depth-anything/DA3-BASE | MV, scene preparation |
| `sam3` | facebook/sam3 (**gated**) | Scene preparation |
| `worldsculpt-adapters` | AlayaLab/WorldSculpt | WorldSculpt |

For SAM3, accept Meta's model terms and authenticate in Modly's Hugging Face
flow before downloading. See [Third-Party Notices](THIRD_PARTY_NOTICES.md).

The pinned **NAF checkpoint** is a separate GitHub asset. It automatically downloads
on first base/MV/WorldSculpt use, after the other required assets are ready, and
is verified before use. Full offline operation is not promised until all assets
and runtime dependencies are complete; native-kernel availability is a separate
requirement. A corrupt NAF checkpoint is never silently replaced.

## Requirements and validation boundaries

Setup uses a checksum-verified, release-backed wheelhouse for the primary runtime.
Provisioning lanes are Linux ARM64/x64 Python 3.12 and Windows x64 Python 3.11/3.12
with CUDA 12.4 wheels; available lanes are not blanket hardware qualification.

- **Prior core evidence:** Windows x64, CPython 3.11, CUDA 12.4.
- **Current feature evidence:** Linux ARM64 / NVIDIA GB10 runs exercised base,
  MV, scene preparation, and WorldSculpt. Viewport/perceptual validation remains partial.
- Other feature/platform paths remain **UNTESTED**; no AMD/ROCm inference claim is made.
- **Scene-preparation lane:** Linux ARM64, CPython 3.12, CUDA ≥12.6.
- **WorldSculpt lane:** exact Linux ARM64, Python 3.12, PyTorch 2.12.0+cu130,
  CUDA 13.0 environment, isolated from the primary runtime.

Memory needs depend on input, resolution, and reconstruction complexity. Low VRAM
mode reduces pressure; it is not an OOM guarantee. Maintainer packaging details
are in [the wheelhouse guide](tools/wheelhouse/README.md).

## Outputs and troubleshooting

Base and MV publish only the final GLB. Scene preparation/normalization publish a
`scene-manifest.json` bundle with essential frames, masks, camera data, object
boxes, scale, and provenance. WorldSculpt currently publishes **geometry only,
without textures**: `scene.glb` plus `scene_mesh.glb`, returning `scene.glb` to Modly.
Its default face budget is 1,000,000 per instance.

For missing weights, use **Models**; for dependency errors, use **Repair** first.
On Windows x64 with Python 3.11/3.12, Modly's `cuda_version` is driver capability,
not the installed PyTorch CUDA ABI. Pre-Blackwell GPUs with capability ≥12.4 select
the published cu124 wheelhouse, including newer drivers reporting 12.8. Setup
verifies torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, and CUDA runtime `12.4`.
This selects an installation ABI; it does not qualify GPU kernels or inference.
Blackwell remains unsupported by the published wheelhouse. Historical invocations
without GPU metadata retain the default cu124 lane.
For targeted recovery only, run `python3 setup.py --repair-scene-prep --json` or
`python3 setup.py --repair-worldsculpt --json` from the extension directory.
If NAF is corrupt or its automatic download fails, use the exact bootstrap repair
command shown in the error. Do not replace checkpoints or native wheels blindly.

## Credits and licenses

- Modly integration: **DrHepa**. Host: **[Modly by Lightning Pixel](https://github.com/lightningpixel/modly)**.
- Models/projects: **[TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D)**,
  **[AlayaLab/WorldSculpt](https://github.com/AlayaLab/WorldSculpt)**,
  **[Meta SAM3](https://github.com/facebookresearch/sam3)**, and
  **[ByteDance Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)**.
- Warm thanks to **[jtydhr88/ComfyUI-WorldSculpt](https://github.com/jtydhr88/ComfyUI-WorldSculpt)**
  for the **SAM3 + DA3 scene-preparation idea and workflow** that inspired this Modly integration.

The wrapper uses [MIT](LICENSE). Upstream code and model weights retain their own
licenses and restrictions; see [Third-Party Notices](THIRD_PARTY_NOTICES.md).
