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

**Version 0.7.2 — Windows RTX 5090 / SM120 (experimental):** normal Install/Reinstall/Repair
keeps the checksum-pinned r3 CPython 3.11 / CUDA 12.8 native archive unchanged.
It adds a separate MV Python overlay, a private WorldSculpt dependency overlay,
and extension-owned CPython 3.12 + SAM3/DA3 for scene preparation and automatic
MV camera calibration. No PowerShell, administrator rights, Developer Mode,
manual Python installation, model copies, or implicit model downloads are needed.
The primary stack stays torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / NATTEN 0.21.6.
The four stable wheelhouse lanes are unchanged.

Scene Prep provisioning checks the real SAM3 builder/tracker and DA3 inference
imports before its GPU qualification, without constructing models; missing
dependencies or DLLs fail clearly.
Portable Python Repair preserves its previous owned runtime until atomic promotion
and interpreter validation succeed, and rolls back failed promotion.
Setup retains bounded, synchronized native probes. The separate Scene Prep lane
also probes Torch SDPA, TorchVision NMS and SAM3 Triton EDT, connected components,
and NMS before reporting runtime readiness. Package installation is **not** GPU
qualification. Community feedback confirmed actual RTX 5090 single-image generation on
12 October (quality tuning pending); it was not independently reproduced in this change. MV, WorldSculpt, scene
preparation, GLB/viewer quality and lifecycle remain **experimental and not
GPU-inference-qualified** until real Windows hardware tests pass. A complete
model inference is not required merely to try the experimental UI nodes.
Weights still come from Modly Models (the existing verified NAF bootstrap is
unchanged). SAM3 retains its gated custom licence.

**Upgrading from the older single-node extension?** Existing checkpoints can be
moved—not copied—into the host's shared group roots. Preserve the auxiliary
subdirectories and check Models readiness before deleting anything; do not download
a duplicate set blindly.

## Five workflow nodes

| Node | Input | Output | Required groups |
| --- | --- | --- | --- |
| **Image to 3D** | One image | Textured GLB | Base |
| **Multi-Image to 3D** | 2–4 images | Textured GLB | Base, MV, DA3 (host readiness gate) |
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

Declared view roles skip DA3 runtime/calibration, but stock Modly 0.4.3 checks all
referenced shared groups before loading a node. Therefore DA3 **weights still
need downloading in Models even for declared roles**. Only automatic cameras
execute the DA3 runtime; this extension does not invent an optional-weight schema.

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
- **Scene-preparation lanes:** Linux ARM64, CPython 3.12, CUDA ≥12.6; experimental Windows x64, private CPython 3.12.10, torch 2.7.1+cu128 and Triton Windows 3.3.1.post21.
- **WorldSculpt lane:** exact Linux ARM64, Python 3.12, PyTorch 2.12.0+cu130,
  CUDA 13.0 environment; experimental Windows x64 inherits the exact primary cp311/cu128 stack through one controlled `.pth`. Its locked dependencies remain isolated from the primary runtime.

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
SM120 selects the experimental cu128 r3 archive and the separate advanced overlays. Historical invocations
without GPU metadata retain the default cu124 lane.
Scene Prep runtime errors identify the failed private environment, dependency,
source-provenance or native probe; use **Repair** and retain the exact error.
The Windows Python runtime is downloaded from the official checksum-pinned
CPython NuGet package, including `Python.h` and `python312.lib` for Triton.
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
