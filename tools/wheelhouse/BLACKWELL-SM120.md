# Blackwell / RTX 50-series investigation

This document tracks an experimental investigation for NVIDIA Blackwell GPUs such as the GeForce RTX 5090. It is **not** a supported Pixal3D wheelhouse lane.

## Current finding

NVIDIA lists GeForce RTX 5090 as compute capability `12.0`. The current published Pixal3D Windows lane is:

```text
windows-x64-cp311-cuda124
torch==2.6.0+cu124
CUDA toolkit 12.4.1 for native builds
```

CUDA 12.4.1 `nvcc` documentation lists supported generated-code targets through `compute_90` / `sm_90`. It does not list `compute_120` / `sm_120`. Current CUDA documentation does list `compute_120` / `sm_120`.

Therefore Blackwell support should be treated as a new exact-stack lane, not as a small edit to the existing CUDA 12.4 lane.

## Why adding one arch flag is insufficient

The Windows cp311/cu124 wheelhouse does not build every native package in this repository. It packages exact-stack external wheels for:

- `flex_gemm_ap`
- `cumesh_vb`
- `o_voxel_vb_ap`
- `drtk`
- `flash_attn`
- `nvdiffrast`
- `nvdiffrec_render`

The repository-built NATTEN candidate workflow controls `TORCH_CUDA_ARCH_LIST` / `NATTEN_CUDA_ARCH`, but rebuilding only NATTEN for `sm_120` would not prove that the full Pixal3D runtime supports RTX 50-series GPUs. Any one native CUDA wheel without Blackwell code can still raise:

```text
CUDA error: no kernel image is available for execution on the device
```

## Probe workflow

`.github/workflows/blackwell-windows-x64-cp311-cuda128-probe.yml` is a manual probe. It only checks prerequisites:

- Python `3.11` compatibility for Modly packaged-app ABI.
- CUDA toolkit candidate support for `compute_120` / `sm_120`.
- PyTorch cu128 wheel availability for the selected torch/torchvision versions.

The probe uploads `BLACKWELL-SM120-PROBE.json` as a GitHub Actions artifact. It must not update `wheelhouse.manifest.json` and must not publish a release asset.

## Candidate wheelhouse workflow

`.github/workflows/wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate.yml` is the first full Windows Blackwell candidate attempt. It targets:

```text
windows-x64-cp311-cuda128-blackwell
Python cp311
torch==2.7.1+cu128
torchvision==0.22.1+cu128
CUDA toolkit 12.8.1
TORCH_CUDA_ARCH_LIST=12.0
NATTEN v0.21.6
```

The workflow builds a NATTEN `v0.21.6` Windows wheel with Blackwell `sm_120` coverage, then assembles a candidate wheelhouse using exact-stack `cu128torch2.7-cp311-cp311-win_amd64` wheels for the other Windows native packages. NATTEN `v0.21.6` is used because upstream includes multiple post-`v0.21.0` Blackwell fixes, including Blackwell FMHA backward fixes, CUTLASS 4.4, and broader Blackwell FMHA/FNA improvements.

The candidate archive includes `WINDOWS-BLACKWELL-CANDIDATE.json` with downloaded wheel checksums and validation requirements. This workflow uploads a GitHub Actions artifact only. It must not update `wheelhouse.manifest.json`, upload release assets, or mark RTX 5090 as supported.

## Prepared setup contract (inactive)

The setup boundary now accepts `cuda_version` and `gpu_sm` in the Modly JSON payload. It normalizes `12.8`/`12.8.1` plus `120`/`12.0`/`sm_120` to the candidate selector `cuda128-blackwell`. This is a hard gate before any wheelhouse download or dependency install: because the candidate asset is intentionally absent from `wheelhouse.manifest.json`, an RTX 50-series payload currently fails with `unsupported_lane` and performs no download or install.

For a future checksum-pinned asset, the prepared lane policy is exact: torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, NATTEN `0.21.6`, torch CUDA `12.8`, compute capability `120`, every Windows native/upstream compatibility import, and `natten.HAS_LIBNATTEN == True` must all validate. A mismatched field fails readiness rather than falling back to the CUDA 12.4 policy. This code path is covered with synthetic selection and install-plan tests only; it does not activate the lane or prove hardware support.

This candidate can still fail in CI because NATTEN/CUTLASS/MSVC/CUDA 12.8 compatibility is unproven for this exact stack. A successful CI build is also not enough for publication: it must be installed and generation-tested on real RTX 50-series hardware.

First-run evidence from GitHub Actions run `27070460013` confirmed CUDA 12.8 generated `compute_120` / `sm_120` for NATTEN (`120-real`), but failed before producing a wheel because upstream NATTEN/CMake passed GCC-only flags such as `-Wconversion`, `-fno-strict-aliasing`, and `-Wall` through `nvcc -Xcompiler` to MSVC. The workflow therefore enables Git long paths for CUTLASS checkout and removes those GCC-only flags recursively before building.

Second-run evidence from GitHub Actions run `27070868300` progressed past long paths and GCC-only flag removal, then failed in `cutlass/exmy_base.h` because `CUTLASS_CXX17_OR_LATER` is enabled through `_MSVC_LANG` while `cutlass/platform/platform.h` exposed `is_unsigned_v` only under `#if (201703L <=__cplusplus)`. The workflow now patches that exact guard to also accept `_MSVC_LANG >= 201703L`.

Third-run evidence from GitHub Actions run `27071874204` progressed further into NATTEN `v0.21.0` Blackwell backward kernels before failing around `PipelineReduceTmaStore::PipelineState` parsing in `sm100_fmha_bwd_kernel_tma_warpspecialized.hpp`. Upstream NATTEN has several later Blackwell fixes, so the candidate now tests `v0.21.6` before adding local kernel patches.

Fourth-run evidence from GitHub Actions run `27075519968` failed before compiling NATTEN because the recursive patch step called `.Replace()` on `$null` from an empty patchable file in `v0.21.6`. Run `27075833950` showed an inline `[string](...)` cast was not sufficient under the workflow script execution path, so the workflow now explicitly checks `$null`, logs skipped empty files, assigns an empty string, and calls `.ToString()` before replacement.

Fifth-run evidence from GitHub Actions run `27076133932` successfully built `natten-0.21.6-cp311-cp311-win_amd64.whl`, then failed in the candidate assembly script because a PowerShell error message interpolated `$NattenVersion:` with a trailing colon, which PowerShell parsed as an invalid drive-qualified variable reference. The script now uses `${NattenVersion}:` in that message.

Sixth-run evidence from GitHub Actions run `27086897007` reached candidate assembly and failed because `$exactStackPattern = "$CudaTag`torch..."` interpreted PowerShell backtick-`t` as a tab, producing `cu128\torch2.7` instead of `cu128torch2.7`. The script now builds the pattern with braced interpolation: `${CudaTag}torch${TorchMinor}-...`.

Seventh-run evidence from GitHub Actions run `27089946691` completed successfully and uploaded the artifact `pixal3d-wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate` with size `295167752` bytes. That artifact was an Actions candidate artifact only, not a GitHub release asset, and is now expired.

Eighth-run evidence from GitHub Actions run `27274530136` also completed successfully and uploaded the artifact `pixal3d-wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate` with size `295167806` bytes. That artifact was also an Actions candidate artifact only, not a GitHub release asset, and is now expired. These successful CI builds prove the candidate workflow can assemble an archive; they do **not** prove RTX 50-series runtime support, because no real RTX 50-series hardware setup, native import, NATTEN runtime, or generation validation was performed.

Ninth-run evidence from GitHub Actions run `35533729886` was dispatched from `feat/pixal3d-upstream-multinode` at commit `cc56796d479e58f4e46f28256eacc9bf35748eec` and failed during the NATTEN build before candidate archive assembly. CUDA 12.8 rejected the hosted runner's current MSVC toolchain with `unsupported Microsoft Visual Studio version`; the candidate archive step and artifact upload step were skipped, so there is no fresh artifact or metadata from that run. The prior successful candidate artifacts remain expired.

Tenth-run evidence from GitHub Actions run `35535794357` was dispatched from `feat/pixal3d-upstream-multinode` at commit `0175c870d0643157a5851e137b1fa6342b63fa86` and failed before any MSVC toolset verification because the Visual Studio Installer returned generic exit code `1` while attempting to add `Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64`. The workflow now treats that installer code as evidence, not as the final compatibility verdict: it first checks for an existing `VC\Tools\MSVC\14.38.*` toolset and skips modification if present; otherwise it invokes the installer, re-checks the actual toolset directory regardless of the exit code, retries once only if `14.38.*` is still absent, captures VS Installer log candidates into a failure artifact, and fails closed if the CUDA-supported MSVC 14.38 toolset remains absent. This still does not accept an unsupported compiler and does not prove RTX 50-series runtime support.

Eleventh-run evidence from GitHub Actions run `35537470167` was dispatched from `feat/pixal3d-upstream-multinode` at commit `bc4c9f4173161b470fac69c99ab462a7251bced1` and completed successfully. The toolchain step installed and re-detected MSVC `14.38.33130` on the first Visual Studio Installer attempt; the compiler probe saw `cl` `19.38.33145` and `nvcc` CUDA `12.8` / `V12.8.93`. The run uploaded Actions artifact `pixal3d-wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate` with size `295161526` bytes and SHA-256 `00d274b022b200608174276d03a0e07b09cc4e6dff93bee7e557bc9fa33e3713`. Inside that artifact, `pixal3d-wheelhouse-v0.1.0-windows-x64-cp311-cuda128-blackwell.zip` has size `295071843` bytes and SHA-256 `162f43973de2dcde2987c0322dbf9f143bed87d267e8f0374618bcbe1f760a97`; it includes `natten-0.21.6-cp311-cp311-win_amd64.whl` with size `27403606` bytes and SHA-256 `eeca0a940eb5de74f92dbec218acb8f60057c674565a8d868eec6eff5ce34a32`. Static artifact inspection found a parseable `WINDOWS-BLACKWELL-CANDIDATE.json` with status `candidate_complete_unvalidated`, 13 readable wheel archives, and matching recorded checksums for the external wheels and NATTEN. This remains candidate evidence only: it is not an official lane and must not be published or added to `wheelhouse.manifest.json` until a real Windows RTX 50-series machine proves imports, `natten.HAS_LIBNATTEN == True`, and Pixal3D Low VRAM generation to a valid GLB.

## Manual Windows RTX 50 hardware-validation harness

The reproducible hardware gate is `tools/validation/validate-blackwell-rtx50.ps1` plus `tools/validation/blackwell_real_generation.py`. It is manual by design: run it only on a real self-hosted Windows x64 RTX 50-series machine that already has the candidate artifact and the authoritative Modly weights on local disk. The optional `.github/workflows/blackwell-rtx50-hardware-validation.yml` is `workflow_dispatch` only and runs on `[self-hosted, windows, x64, rtx50]`; it uploads evidence/summary only, never weights or model assets.

Required inputs are explicit and fail closed: candidate artifact path, expected SHA256, expected size, pre-provisioned Modly weights path (the authoritative Modly home or pixal3d-base directory), fixture image path, and an evidence directory. The harness verifies the artifact SHA/size before extraction, copies the extension to a temporary directory, injects the candidate archive only into that temporary copy/cache, and leaves the repository `wheelhouse.manifest.json` unsupported lanes unchanged. No secrets are used and the weights path is passed explicitly; model weights must already exist because the validation run must not prove support by downloading hidden assets. The PowerShell preflight marks auxiliary-asset proof as `deferred` and network-denial setup as `configured`; neither is a pass. The Python helper is the authority for executed proof: it must pass strict local checks for primary Pixal3D sentinels, DINO, RMBG, MoGe, and a local valid NAF sentinel before load, then install a Windows-safe network-denial guard that blocks Python socket/urllib/requests network paths during generation.

Example PowerShell invocation:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tools/validation/validate-blackwell-rtx50.ps1 `
  -CandidateArtifactPath C:\evidence\pixal3d-wheelhouse-windows-x64-cp311-cuda128-blackwell-candidate.zip `
  -ExpectedArtifactSha256 00d274b022b200608174276d03a0e07b09cc4e6dff93bee7e557bc9fa33e3713 `
  -ExpectedArtifactSizeBytes 295161526 `
  -ModlyWeightsPath C:\Modly\models\pixal3d\pixal3d-base `
  -FixtureImagePath C:\fixtures\single-view.png `
  -EvidenceDir C:\evidence\blackwell-rtx50-run
```

The orchestrator invokes `setup.py` with the real Modly JSON setup payload containing `cuda_version` `128` and `gpu_sm` `120`, then runs Repair twice to prove idempotence. It runs `pip check`, verifies exact torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, torch CUDA `12.8`, GPU SM `120`, every Windows native import including `drtk` and `flash_attn`, upstream compatibility aliases, and `natten.HAS_LIBNATTEN == True`. It then performs real Pixal3D single-view `low_vram` generation from the explicit fixture image, requires the resolved generated GLB to be an existing regular file inside the requested output directory/workspace before parsing or hashing, validates every mesh primitive with `POSITION` data for referenced accessor/bufferView/range/stride/component/count safety, FLOAT alignment, glTF-compatible stride, finite coordinates, and at least one validated TRIANGLES face, validates indexed primitives as non-empty unsigned SCALAR TRIANGLES with in-range decoded indices, expands indexed and sequential triangles to count nondegenerate faces using a scale-aware cross-product threshold (`twice_area > max_edge_length_squared * 2^-40`), records aggregated primitive/vertex/total-face/nondegenerate-face counts plus output SHA256/size/stats, performs a logical generator unload, and then performs a separate child-process runtime probe as process-level restart evidence.

Evidence is written as machine-readable `blackwell-validation.json` plus `blackwell-orchestrator.json` and setup/helper logs, distinguishing configured/deferred preflight state from executed proof and distinguishing logical unload from process-level cleanup/restart proof. Final PASS requires the Python helper's executed strict local-asset gate and executed network-denial gate to pass during generation; intermediate preflight JSON must not be interpreted as completed hardware proof. The honest evidence boundary remains `candidate_complete_unvalidated` until such a run passes on real hardware. This is not official support until real Windows RTX 50 imports, native `natten.HAS_LIBNATTEN == True`, and Low VRAM valid-GLB generation are recorded from that machine. Cancellation is reported as `not_supported_by_harness`; the harness does not fake Modly cancellation support.

## Publish criteria for a future Blackwell lane

A future Blackwell lane requires all of the following before it can be declared supported:

1. Exact Python ABI, PyTorch version, CUDA minor, and platform tag selected.
2. `nvcc --list-gpu-arch` and `nvcc --list-gpu-code` include `compute_120` / `sm_120`.
3. All required native CUDA wheels are rebuilt or sourced for the same exact stack.
4. NATTEN imports with `HAS_LIBNATTEN == True` on a real NVIDIA Windows machine.
5. Full Pixal3D Low VRAM generation succeeds on RTX 50-series hardware and exports a valid GLB.
6. Release asset is checksum-pinned in `wheelhouse.manifest.json` only after validation.

## Official promotion checklist

Do **not** move `windows-x64-cp311-cuda128-blackwell` from candidate/inactive to an official manifest lane until every item below has concrete evidence:

1. Trigger a fresh candidate workflow run from the current promotion branch.
2. Download the resulting Actions artifact to a temporary audit location, not into an installed extension, runtime venv, model directory, or Modly workspace.
3. Retain the candidate metadata file `WINDOWS-BLACKWELL-CANDIDATE.json` together with the archive name, archive byte size, archive SHA-256, workflow run URL, workflow run id, source commit, and artifact id.
4. On a real Windows RTX 50-series machine, run Modly setup/Repair with the real JSON payload (`cuda_version`/`gpu_sm`) and prove the Blackwell lane is selected.
5. In the repaired extension venv on that same machine, verify `pip check`, torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, torch CUDA `12.8`, GPU SM `120`, every required Windows native import, every upstream compatibility import, and `natten.HAS_LIBNATTEN == True`.
6. Run Pixal3D Low VRAM generation on RTX 50-series hardware under strict offline auxiliary mode with network-denial enabled, then validate that the returned GLB exists as a regular file within the output directory (and not a symlink or reparse point), is non-empty, opens as a valid mesh, and is viewer-compatible.
7. Publish the validated archive as a GitHub release asset only after the runtime evidence above is recorded.
8. Add the official lane to `wheelhouse.manifest.json` with exact `filename`, `size_bytes`, `sha256`, compression, packages, and selectors.
9. Re-run focused wheelhouse/setup contract tests and keep the docs clear that RTX 50-series support begins only at the validated release asset and manifest revision.

Until those conditions are met, RTX 5090 / Blackwell remains experimental and unsupported by the published wheelhouse.
