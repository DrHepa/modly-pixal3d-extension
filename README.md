# Pixal3D Modly Extension

Pixal3D image-to-3D model extension for Modly. It converts a single input image into a textured GLB mesh using the upstream `TencentARC/Pixal3D` model family and Modly-managed model storage.

This repository contains only the extension runtime, setup entrypoint, and a release-backed wheelhouse contract needed to prepare Pixal3D dependencies from a GitHub install. Model weights are not included; Modly downloads model assets through its UI into the normal Modly model storage.

## Installation

Install this repository as a Modly extension, then use Modly Models to download
the weight groups needed by the node you intend to run. Repair/setup only
creates dependency environments; it never downloads model weights.

### What setup does

```bash
python3 setup.py --prepare --json
```

- creates `venv/` inside the extension
- installs `requirements.txt`
- prepares the selected release-backed wheelhouse from `wheelhouse.manifest.json`
- installs native packages with `pip install --no-index --find-links <verified-wheelhouse>`
- verifies and reinstalls the bundled pure-Python Pixal3D MV core overlay after the release-backed wheelhouse, without changing native wheels
- runs `pip check`
- reserves the local NAF directory under the configured Modly models root; Modly creates shared weight-group folders during download
- on supported Linux ARM64 hosts, creates or repairs the separate Python 3.12
  `venv-scene-prep/` for the pinned official SAM3 and DA3 sources; unsupported
  platforms keep the primary setup successful and report only
  `scene-from-estimates` as unavailable. This lane never changes `venv/` or
  `venv-worldsculpt/`

It does **not** download model weights and does **not** run generation.

Use `python3 setup.py --repair-scene-prep --json` to repair only the scene
preparation lane. `--skip-scene-prep` is available for a primary Pixal3D-only
installation. The repair command installs dependencies and pinned source code,
but model weights remain exclusively managed by Modly Models.

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

## Scene preparation for WorldSculpt

This branch adds two typed model nodes. They are separate because estimating a
scene from pixels and validating an already annotated scene have different
trust, dependency, and licensing boundaries.

### Prepare Scene from Estimates (`capture -> scene`)

`scene-from-estimates` accepts a workspace-contained
`modly.capture-manifest.v1`. It selects frames deterministically, runs the
official SAM3 video predictor for text-guided instance tracking, and runs the
official DA3 Base multiview model for depth, confidence, intrinsics, and
world-to-camera poses. A clean-room geometry layer then:

- scales the declared DA3 processing-resolution intrinsics to the original
  frame resolution rather than estimating focal length from the principal point;
- validates and converts OpenCV `w2c` poses to Blender `c2w` poses;
- erodes masks, trims low-confidence and outlier depth, and unprojects robust points;
- rejects inconsistent camera frames and low-coverage reprojections;
- recomputes percentile AABBs after frame acceptance; and
- deduplicates only same-label instances using 3D IoU.

The output is a fresh canonical `modly.scene-manifest.v1` directory containing
PNG frames, per-instance masks, cameras, AABBs, and exact source/model
provenance. DA3 Base has **relative scale**. The output explicitly records
`scale.mode: relative` and never claims metric units. It can feed the existing
WorldSculpt scene node, but the composed result remains in that relative scene
scale unless the user supplies an external metric calibration.

User parameters cover object labels, maximum frames, frame stride, DA3 process
resolution, mask erosion, minimum retained geometry points, confidence/depth
trimming, AABB percentiles, reprojection coverage, label-aware IoU, camera
consistency, and the SAM score threshold. Defaults and bounds in
`manifest.json` are the runtime authority.

### Normalize Annotated Scene (`scene -> scene`)

`normalize-annotated-scene` has no weight groups and does not use the
scene-preparation environment. It revalidates the existing WorldSculpt scene
contract, rejects symlinks and escaped paths, copies only referenced frames,
canonical masks, camera data, and AABBs into a fresh workflow directory, and
records normalization provenance. It preserves declared scale mode
(`metric`, `relative`, or `unknown`) instead of inventing calibration.

### Capture manifest

A capture is either an ordered frame list or one video, never both. All paths
are relative to the workspace/capture root, file byte sizes are rechecked, frame
images are checked against their declared dimensions, and the isolated worker
revalidates custody before inference. Frame arrays use contiguous zero-based
indices. Videos declare their decoded `frameCount`; the worker verifies the
count and dimensions while selecting frames in decode order.

```json
{
  "schema": "modly.capture-manifest.v1",
  "captureRoot": ".",
  "kind": "frames",
  "frames": [
    {"index": 0, "path": "frames/0000.png", "width": 1920, "height": 1080, "byteSize": 123456}
  ],
  "provenance": {"source": "camera-import", "ordering": "manifest-index"}
}
```

For video captures, use `kind: "video"`, omit `frames`, add
`video: {"path", "width", "height", "byteSize", "frameCount"}`, and set
`provenance.ordering` to `decode-index`.

### Weights, authentication, and runtime

Download both groups from Modly Models before running `scene-from-estimates`:

- `sam3`: gated `facebook/sam3` revision
  `3c879f39826c281e95690f02c7821c4de09afae7`, exact files
  `config.json`, `sam3.pt`, and `LICENSE`;
- `da3-base`: `depth-anything/DA3-BASE` revision
  `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`, exact files
  `config.json` and `model.safetensors`.

SAM3 requires acceptance of Meta's custom SAM License and authenticated access
to the gated official checkpoint. It also requires Python 3.12 or newer,
PyTorch 2.7 or newer, and CUDA 12.6 or newer. This extension provisions Python
3.12 with pinned PyTorch 2.12/torchvision 0.27 in `venv-scene-prep` and fails
closed if CUDA or pinned source provenance is missing. Setup/load never fetch
weights, and worker inference runs with Hugging Face offline flags and no auth
tokens. The isolated lane currently targets Linux ARM64; installation and
inference on every platform, including that target, remain **UNTESTED** here.

The manifest, contract, geometry, setup isolation, and subprocess adapter have
focused automated coverage. A real SAM3 + DA3 inference run, installed Modly UI
end-to-end flow, output quality, and other hardware/platform combinations are
currently **UNTESTED**; no checkpoint download or large inference was started
as part of this implementation.

Licensing and clean-room attribution are documented in
`THIRD_PARTY_NOTICES.md`. The unlicensed community
`jtydhr88/ComfyUI-WorldSculpt` codebase, including `nodes_realinput.py`, is not
copied or vendored here.

## Usage

Connect **Load Capture** to **Prepare Scene from Estimates**, or connect an
existing typed scene to **Normalize Annotated Scene**. Connect either scene
output to the existing **WorldSculpt** node. Use the existing image and scene
routes unchanged for older workflows; typed capture/scene model generation uses
the generic host artifact route.

## Outputs

Both scene-preparation nodes return a path to a canonical
`modly.scene-manifest.v1` stored inside the workflow workspace. WorldSculpt
consumes that scene and returns its existing GLB mesh output. Estimated scenes
remain relative-scale unless calibrated externally.

## Parameters

The authoritative names, defaults, ranges, and descriptions are the
`params_schema` entries for each node in `manifest.json`. The normalization
node intentionally exposes no inference parameters.

## Requirements and compatibility

The primary Pixal3D, WorldSculpt, and scene-preparation environments are
isolated from each other. Scene estimation requires Linux ARM64, Python 3.12,
CUDA 12.6 or newer, and authenticated acceptance of the SAM License before
Modly can download the gated SAM3 files. See the per-feature status statements
below; support on an environment is not implied by successful static tests.

## Limitations

Real SAM3/DA3 scene-preparation inference, perceptual output quality, installed
Modly UI E2E, and non-target platforms remain **UNTESTED** in this change.
Relative DA3 depth cannot establish metric scale by itself.

## Troubleshooting

- Missing model files: download or repair the named group in Modly Models;
  setup and runtime will not fetch it implicitly.
- SAM3 access denied: accept the repository license and authenticate Modly's
  Hugging Face download flow.
- Scene-preparation environment failure: run
  `python3 setup.py --repair-scene-prep --json` and use its fail-closed report.
- CUDA/runtime rejection: verify Python, PyTorch, CUDA, and pinned source
  provenance rather than installing into the primary environment.

## Credits

Extension integration and clean-room scene preparation: DrHepa. Pixal3D is by
TencentARC; WorldSculpt is by AlayaLab; SAM3 is by Meta; Depth Anything 3 is by
ByteDance. See `THIRD_PARTY_NOTICES.md` for exact revisions and licenses.

## License

This extension's original integration code is MIT licensed; see `LICENSE`.
Third-party code, model sources, weights, and datasets retain their own terms,
including the custom SAM License. See `THIRD_PARTY_NOTICES.md` and `licenses/`.

## View-aligned generation behavior

Pixal3D generates meshes aligned to the input image projection rather than always canonicalizing the object to a universal upright/front pose. Upstream's projected render path is designed so the first rendered frame matches the projected input view.

Practical implications:

- front/straight input images should usually produce upright meshes;
- angled or isometric input images can produce angled meshes;
- that input-dependent tilt is expected Pixal3D behavior, not a Modly orientation bug;
- do not apply a fixed post-export pitch correction to all Pixal3D outputs, because it can break already-upright generations.

The extension preserves Pixal3D's exported GLB orientation. Do not apply a fixed post-export yaw correction to all Pixal3D outputs; front/back orientation can depend on upstream generation behavior and should be validated with representative inputs instead of rewritten unconditionally.

## Remaining runtime requirement

### Posed multi-view node (candidate; not yet live-accepted)

`generate-mv` is a separate `scene`-input node. It is **TencentARC Pixal3D multi-view**, not the AlayaLab WorldSculpt scene-composition pipeline. It requires both host-provided `pixal3d-base` and `pixal3d-mv` shared weight roots. The MV group contains only `pipeline_mv.json` and the four `_mv` denoiser JSON/safetensors pairs from `TencentARC/Pixal3D` revision `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`; the base group owns the three shared decoders and DINO/RMBG. The extension generates a private temporary MV config pointing decoder and matting references into the host-provided base root, without mutating downloaded weights, and validates every referenced checkpoint before importing inference. NAF must already exist at `models/pixal3d/auxiliary/naf/naf_release.pth` for MV; it is never silently downloaded by the MV path.

The scene input is a Modly `modly.scene-manifest.v1` JSON file in `Workspace` whose `sceneRoot` is a workspace-relative directory containing `transforms.json` and its referenced image files. `transforms.json` needs a non-empty `frames` array, each frame's image path and finite 4×4 camera-to-world matrix, and a horizontal `camera_angle_x` in radians (per frame or top-level). Frame 0 should be the canonical front view. This node does not accept repeated independent images or substitute single-view inference. Modly's current scene endpoint passes the validated scene manifest path in `scene_manifest_path` parameters while forwarding empty image bytes; the extension consumes the manifest path, not those bytes.

**Runtime boundary:** the published native wheelhouse still contains the single-view Pixal3D core wheel. Normal setup now checksum-verifies and force-reinstalls a bundled, additive pure-Python MV core candidate after that wheelhouse; it does not change native CUDA/NATTEN packages or claim GPU compatibility. The vendored upstream `inference_mv.py` invokes the MV cascade through a private generated config and a scoped local-only loader, so downloaded shared weight files are not modified and corrupt/missing local checkpoints cannot trigger an upstream Hugging Face fallback. If the core overlay is absent or invalid, setup fails rather than leaving an inert node. Current tests cover contracts and a mocked inference call, **not** a real MV GLB or Modly UI end-to-end. Modly also needs the multi-source and shared-weight-group host changes before either node's new layout can run live.

MV NAF loading is intercepted at the exact upstream Torch Hub call and reads only the provisioned local checkpoint; unexpected Hub requests fail closed. MV progress is reported at stage boundaries. Cancellation is checked before and after expensive stages, but the upstream model-loading, cascade, and GLB extraction calls cannot be interrupted mid-call; a cancellation request takes effect at the next boundary.

After setup succeeds, use Modly UI to download Pixal3D model assets. Real generation should be validated only after the primary Pixal3D weights and the required auxiliary assets are present.

Before validating an existing installation, move its downloaded Pixal3D root files and `auxiliary/{dinov3,rmbg,moge}` files into `models/pixal3d/_shared/pixal3d-base/` (preserving the `auxiliary/` subdirectories), or redownload the shared group in Modly. Keep `auxiliary/naf/naf_release.pth` in place. Do not copy caches blindly: verify each sentinel against the published repository file and use the host-provided shared root at runtime.

The Pixal3D, DINO, RMBG, and MoGe weights are downloaded by Modly as the extension-scoped `pixal3d-base` weight group under `models/pixal3d/_shared/pixal3d-base/`. This requires Modly support for both multi-source downloads (PR #275) and shared weight groups (PR #348). NAF remains at `models/pixal3d/auxiliary/naf/`; do not shorten `auxiliary` to `aux`, a reserved Windows device name.

Required localizable auxiliary sentinels:

- `models/pixal3d/_shared/pixal3d-base/auxiliary/dinov3/config.json`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/dinov3/preprocessor_config.json`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/dinov3/model.safetensors`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/rmbg/config.json`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/rmbg/preprocessor_config.json`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/rmbg/BiRefNet_config.py`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/rmbg/birefnet.py`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/rmbg/model.safetensors`
- `models/pixal3d/_shared/pixal3d-base/auxiliary/moge/model.pt`
- `models/pixal3d/auxiliary/naf/naf_release.pth`

Normal setup does not download weights. Use Modly Models to download the shared Hugging Face group. On first generation, the extension downloads only the NAF checkpoint from `https://github.com/valeoai/NAF/releases/download/model/naf_release.pth` if needed. The manual safety command is `python3 setup.py --bootstrap-auxiliary-assets --workspace-root <extension-dir> --json`; it downloads only NAF. `local`, `offline`, and `strict` modes require NAF to be present and never start that download.

The pipeline patcher is local-first for DINO/RMBG: when those sentinels are complete, it writes local resolved paths into the user-local `pipeline.json` and records non-absolute logical metadata. MoGe is local-first at runtime: when `models/pixal3d/_shared/pixal3d-base/auxiliary/moge/model.pt` exists, the extension wraps `inference.load_moge_model` so `MoGeModel.from_pretrained()` receives that local checkpoint file path instead of `Ruicheng/moge-2-vitl`. NAF is local-first at runtime too: when `models/pixal3d/auxiliary/naf/naf_release.pth` exists, a scoped per-extractor override verifies its exact size and SHA-256, then loads it locally without calling upstream `torch.hub.load`; the extractor factory is restored after inference. If a host-managed DINO/RMBG/MoGe sentinel is missing, generation fails instead of fetching it from a hidden cache or remote source. If NAF bootstrap fails, generation reports `naf_bootstrap_failed`. In `local`, `offline`, or `strict` auxiliary mode, missing DINO/RMBG/MoGe/NAF files fail early with `missing_auxiliary_assets` before importing upstream inference or `hubconf` code.

This is **not** a cross-platform offline-generation guarantee. A direct single-view GPU run with offline model flags completed on Linux aarch64 / GB10 with staged local assets and produced a structurally valid GLB, but a no-network test, installed Modly UI, and other platform lanes remain untested. NAF checkpoint localization is also separate from strict NAF native kernels: `natten.HAS_LIBNATTEN` must still be validated independently before claiming strict NATTEN/libnatten acceleration.

## Publication status

- Repository visibility: public.
- Primary Modly packaged-app lane: Windows `x64` / Python `cp311` / CUDA `12.4`.
- Setup contract: release-backed wheelhouse with checksum verification and native import probes.
- Runtime status: the previous single-node layout was validated on Windows `x64` / Python `cp311` / CUDA `12.4` through a complete Modly Low VRAM 1024 generation. The new shared-weight layout has static and mocked-generation coverage but still requires a live Modly run after the dependent host PRs are available.
- Runtime note: use Low VRAM mode on 8GB-class GPUs; generation quality and orientation depend on the input view and upstream Pixal3D export behavior. The extension does not rewrite the final GLB orientation with a fixed yaw transform.

### WorldSculpt scene composition (advertised branch candidate; GB10 GPU inference validated)

The `worldsculpt` branch candidate is a separate `scene → mesh` node in `manifest.json`, **not** an alias for Pixal3D posed-view generation. The manifest advertises it for Modly weight download and scene dispatch. A direct GB10 GPU run through the extension generator completed crop, reconstruction, and composition and produced a structurally valid GLB; installed Modly UI E2E and other platform lanes remain untested, so this is not a release or broad compatibility claim. It follows the pinned AlayaLab WorldSculpt geometry-only path: crop each eligible masked instance, reconstruct with the SS/shape LoRA adapters (`--no_tex --no_glb`, step 15000), then compose a normal-bearing scene GLB. It does not produce textures, renders, or a point-cloud visualization. The bundled source snapshot is `373f09f0ecddc94607b569bea0316ff6a2286501`; the UI-managed `worldsculpt-adapters` shared weight group contains exactly two adapter stages from `AlayaLab/WorldSculpt` revision `8cb81056d803c61371dd84ef18a14142a738610e`. The node also depends on the shared `pixal3d-base` group. The NAF checkpoint must already exist at `models/pixal3d/auxiliary/naf/naf_release.pth` through the explicit auxiliary bootstrap; runtime does not download it.

The input is a Modly `modly.scene-manifest.v1` whose `sceneRoot` contains `transforms.json`, camera-posed full-size PNG frames, `instances` with metric `aabb_world`, and `masks/objNN/####.png`. Literal `sceneRoot: "."` selects the manifest directory; other relative roots are workspace-relative. This implementation deliberately rejects OBB metadata, unprojectable/empty masked instances, symlinked scene files, and incomplete adapter trees. Output is a fresh private run directory under the configured workflow output directory; only a GLB validated against every eligible object is returned. The only exposed parameter is `face_budget` (1,000–3,000,000; default 1,000,000). Cancellation is checked at each subprocess boundary and during a stage; a cancelled or failed run never returns an output path.

**Readiness is intentionally fail-closed.** On the validated Linux aarch64 / Python 3.12 / CUDA 13.0 / torch 2.12.0+cu130 lane, first prepare the primary `venv`, then run `python3 setup.py --repair-worldsculpt --json` for setup or Repair. This verifies nine hash-locked offline wheels in `wheels/worldsculpt/`, creates only `venv-worldsculpt`, links the dynamically resolved primary site-packages with a `.pth` file, installs WorldSculpt-specific packages using `--no-index --no-deps`, then checks dependency closure and native imports. Repeating the command repairs the isolated lane without changing the primary venv or downloading weights. The manifest records the wheel hashes and the PyPI `iopath` 0.1.10 source archive hash; its wheel was privately built from that source. WorldSculpt subprocesses explicitly use the isolated interpreter. Other platforms and CUDA/Python combinations fail closed. Direct GB10 GPU scene inference passed with the staged synthetic scene; installed Modly UI E2E and perceptual reconstruction quality remain **UNTESTED**. The overall extension remains **PARTIAL** while MV inference and installed UI validation are outstanding. The runtime uses a disposable source/config overlay for local DINOv3 and NAF overrides, blocks HF/network cache fallback, and does not mutate shared downloaded weights.
