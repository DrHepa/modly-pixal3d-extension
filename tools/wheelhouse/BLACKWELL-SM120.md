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

This candidate can still fail in CI because NATTEN/CUTLASS/MSVC/CUDA 12.8 compatibility is unproven for this exact stack. A successful CI build is also not enough for publication: it must be installed and generation-tested on real RTX 50-series hardware.

First-run evidence from GitHub Actions run `27070460013` confirmed CUDA 12.8 generated `compute_120` / `sm_120` for NATTEN (`120-real`), but failed before producing a wheel because upstream NATTEN/CMake passed GCC-only flags such as `-Wconversion`, `-fno-strict-aliasing`, and `-Wall` through `nvcc -Xcompiler` to MSVC. The workflow therefore enables Git long paths for CUTLASS checkout and removes those GCC-only flags recursively before building.

Second-run evidence from GitHub Actions run `27070868300` progressed past long paths and GCC-only flag removal, then failed in `cutlass/exmy_base.h` because `CUTLASS_CXX17_OR_LATER` is enabled through `_MSVC_LANG` while `cutlass/platform/platform.h` exposed `is_unsigned_v` only under `#if (201703L <=__cplusplus)`. The workflow now patches that exact guard to also accept `_MSVC_LANG >= 201703L`.

Third-run evidence from GitHub Actions run `27071874204` progressed further into NATTEN `v0.21.0` Blackwell backward kernels before failing around `PipelineReduceTmaStore::PipelineState` parsing in `sm100_fmha_bwd_kernel_tma_warpspecialized.hpp`. Upstream NATTEN has several later Blackwell fixes, so the candidate now tests `v0.21.6` before adding local kernel patches.

## Publish criteria for a future Blackwell lane

A future Blackwell lane requires all of the following before it can be declared supported:

1. Exact Python ABI, PyTorch version, CUDA minor, and platform tag selected.
2. `nvcc --list-gpu-arch` and `nvcc --list-gpu-code` include `compute_120` / `sm_120`.
3. All required native CUDA wheels are rebuilt or sourced for the same exact stack.
4. NATTEN imports with `HAS_LIBNATTEN == True` on a real NVIDIA Windows machine.
5. Full Pixal3D Low VRAM generation succeeds on RTX 50-series hardware and exports a valid GLB.
6. Release asset is checksum-pinned in `wheelhouse.manifest.json` only after validation.

Until those conditions are met, RTX 5090 / Blackwell remains experimental and unsupported by the published wheelhouse.
