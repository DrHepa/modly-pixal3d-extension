# Experimental Windows SM120 advanced nodes — version 0.7.2

This integration preserves the existing r3 native archive byte-for-byte:
SHA256 `162f43973de2dcde2987c0322dbf9f143bed87d267e8f0374618bcbe1f760a97`,
295071843 bytes. The four stable wheelhouse lanes are unchanged. No CUDA native
rebuild or stack upgrade is established as necessary for these advanced nodes.
The user-reported 12 October RTX 5090 single-image success is not a new inference
qualification by this change.

## Normal user path

Install from GitHub or use Modly extension **Repair**. Setup selects the existing
experimental r3 primary stack, force-restores its verified payload and adds the
separate `0.1.0+modly.blackwell.mv` Python wheel. Its reproducible metadata policy
requires native NATTEN 0.21.6 and the real Windows alias distribution names;
the existing stable MV artifact still requires NATTEN 0.21.0.

WorldSculpt uses `venv-worldsculpt`, copies the primary cp311 interpreter through
`venv --copies`, discovers its actual `purelib`, and inherits the primary exact
Torch/CUDA/native stack through one controlled `.pth`. A separate immutable lock
pins Windows fpsample/pycocotools and the isolated pure-Python dependencies,
including pywin32 for portalocker. Overlay provenance, installed bytes and
`pip check` are required. Nothing is installed into the primary environment.

Scene preparation and automatic MV calibration use `venv-scene-prep` with private
CPython 3.12.10 from the official Python NuGet package. Setup verifies the archive
and extracted file hashes, including Python headers and import library. Repair
validates directory ownership, retains the prior runtime under an owned backup,
atomically promotes the staged runtime and rolls back a failed promotion or
interpreter validation. Locked executables never trigger delete-first recovery. The
private stack is torch 2.7.1+cu128, torchvision 0.22.1+cu128, and Triton Windows
3.3.1.post21. Runtime dependencies and wheels are checksum-pinned. SAM3 and DA3
inference-only wheels retain unmodified upstream runtime sources and real
release-wheel provenance, not fabricated VCS `direct_url.json` metadata.
SAM3 remains under its custom licence; gated weights stay in Modly Models.

SAM3 uses one GPU (`gpus_to_use=[0]`), avoiding the upstream multi-GPU NCCL branch;
model compilation and FA3 are disabled on this path. Triton remains required for
SAM3's EDT and connected-components fallbacks. DA3's existing fail-closed adapter
does not enable unused exporters or external pose alignment; this does not claim
support for upstream export, training or application extras.

Declared MV roles skip DA3 runtime/calibration in the adapter. Stock Modly 0.4.3
still requires the referenced DA3 shared weight group before node load, even for
declared roles; its readiness gate has no per-run parameters. The manifest
reference is preserved so automatic cameras receive the authoritative managed
path. Automatic cameras validate both weights and runtime before generation. Load-time readiness without run parameters checks only the
shared MV core and reports that camera mode is not selected; it is not evidence
that automatic calibration works.

## Disposable WorldSculpt source overlay

No symlinks, administrator rights, Developer Mode or model copies are required.
Every run copies the pinned source into its private temporary directory. On
Windows, the private crop helper also materializes input image frames/masks as
regular files/directories, never model weights; no crop-stage symlinks are used.
The Linux crop helper retains its existing semantics.
The pipeline model references and guarded AST checkpoint constants route to the
actual validated Modly-managed files. DINO literals are rewritten as AST string
constants, preserving Windows backslashes safely. Original source, configs and
weights are never modified. Changed Python is compiled before launch.

The controlled subprocess bootstrap installs existing native aliases, O-Voxel
Python compatibility, NATTEN and FlexGEMM compatibility **before** vendored
imports. Vendored Pixal3D precedes the installed MV core. Output draining uses a
bounded queue and reader thread, not selectors on Windows pipes. Process-tree
cancellation and zero-exit/missing-artifact validation remain mandatory.

## Evidence boundaries

Portable tests, deterministic wheel custody, dependency metadata and source
integrity are not Windows execution, GPU kernels or model inference. The Windows
CI helper exercises actual production installation/Repair code even when the
normal primary setup stops at its required GPU gate. The CI-only provisioning
entry returns `runtime_prepared: false`; it never bypasses the production native
probe or changes production setup failure into success.

A separate hardware-independent subprocess imports the real SAM3 video builder,
selected tracker/video inference classes, and existing DA3 inference adapter
without constructing models. Both production provisioning and CI collect these
per-boundary receipts before hardware qualification. Missing pure dependencies,
syntax errors or DLL failures fail; only the exact no-NVIDIA-driver RuntimeError
with CUDA unavailable may be recorded BLOCKED, with reached/unreached checks.
An import-layer/provisioning receipt never sets `runtime_prepared` true.

Hosted no-driver evidence must retain **full setup FAIL**, **native BLOCKED**,
**weights NOT_DOWNLOADED**, **inference NOT_RUN**. Unexpected DLL, ABI, import,
version, provenance, dependency or timeout errors are failures, not GPU blocks.
Before claiming runtime readiness on SM120, production setup synchronously tests
Torch CUDA/SDPA, TorchVision NMS and the pinned SAM3 Triton EDT, connected
components and NMS APIs within bounded subprocesses. Full advanced-node GPU model
inference, viewer quality, unload/restart and real UI use remain unqualified
until separately exercised. Passing native smoke permits experimental UI testing;
it does not imply complete model inference qualification.

## Reproduce the MV artifact (maintainers only)

Run `tools/wheelhouse/build-pixal3d-mv-python-wheel.py --policy blackwell` against
the pinned Pixal3D commit and original base wheel, with a new output path named
`pixal3d_core-0.1.0+modly.blackwell.mv-py3-none-any.whl`. The builder validates the
base SHA256, source commit and four source hashes, uses deterministic ZIP metadata
and regenerates RECORD. See `blackwell-mv-overlay.provenance.json` for custody.
No publishing or native rebuild is performed by this script.
